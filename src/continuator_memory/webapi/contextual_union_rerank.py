"""``contextual_union_rerank`` — fixed-depth candidate union + pretrained CE reorder.

Pipeline (general-purpose; no dataset-specific logic)::

    contextual_lexical top-N
      ∪  contextual_bm25 top-N
      → dedupe by memory_id (stable: lexical order, then novel BM25)
      → pretrained cross-encoder over retrieval_content
      → same candidate set, reordered

Each channel defaults to ``DEFAULT_CHANNEL_TOP_N`` (50). The union is **not**
truncated back to 50 before reranking.

Identity semantics
------------------
* A retrieval **candidate** is a consolidated ``Memory`` (``memory_id``).
* Contextual ``[TARGET]`` text is built from ``Memory.message_id`` (creating claim).
* ``source_message_ids`` is provenance for evaluation credit and may contain **multiple**
  message IDs when consolidator KEEP/restatement attaches additional sources.
* Union deduplicates by ``memory_id`` so the same memory from both channels appears once.
* Full ``source_message_ids`` are preserved on the kept candidate (no ID discarded).
* ``ranked_message_ids`` continues to expand all provenance IDs for gold scoring.

Fails closed if the cross-encoder cannot be loaded.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from continuator_memory.embedding import Embedder
from continuator_memory.models import Conversation
from continuator_memory.v2.bm25 import BM25Index
from continuator_memory.v2.memory_retrieval import RetrievalOutcome, RetrievalResult
from continuator_memory.v2.rerank import (
    DEFAULT_RERANK_MODEL,
    PairScorer,
    RerankerLoadError,
    load_cross_encoder_reranker,
)
from continuator_memory.webapi.contextual_bm25 import retrieve_contextual_bm25_evidence
from continuator_memory.webapi.contextual_rerank import (
    apply_retrieval_content_rerank,
    contextual_lexical_config,
    evidence_from_outcome,
)
from continuator_memory.v2.source_retrieval import SourceEvidence, retrieve_source_evidence

MODE_CONTEXTUAL_UNION_RERANK = "contextual_union_rerank"
# Fixed experimental protocol depth per channel (not tuned on LoCoMo).
DEFAULT_CHANNEL_TOP_N = 50


def candidate_dedupe_key(result: RetrievalResult) -> str:
    """Stable union membership key: one consolidated memory = one candidate.

    Do not use a single ``source_message_ids`` entry alone: restatement KEEP can attach
    additional provenance message IDs while ``Memory.message_id`` (contextual TARGET)
    remains the creating claim. Evaluation still credits the full provenance tuple.
    """
    if not result.memory_id:
        raise ValueError("retrieval result is missing memory_id")
    if not result.source_message_ids:
        raise ValueError(
            f"retrieval result {result.memory_id!r} has empty source_message_ids"
        )
    return result.memory_id


def target_message_id(result: RetrievalResult) -> str:
    """Backward-compatible alias used by tests; prefer ``candidate_dedupe_key`` for union.

    Returns the first provenance message id (position-ordered sources). This is often the
    creating ``Memory.message_id``, but multi-source memories remain fully credited via
    ``source_message_ids``.
    """
    ids = result.source_message_ids
    if not ids:
        raise ValueError(f"retrieval result {result.memory_id!r} has empty source_message_ids")
    return ids[0]


def stable_union_by_target_message_id(
    lexical_results: Sequence[RetrievalResult],
    bm25_results: Sequence[RetrievalResult],
) -> tuple[RetrievalResult, ...]:
    """Lexical order first; append BM25 results whose ``memory_id`` is not already present.

    Name kept for call-site stability. Dedup key is ``memory_id`` (retrieval unit).
    Kept candidates retain their full ``source_message_ids`` provenance.
    """
    seen: set[str] = set()
    merged: list[RetrievalResult] = []
    for result in list(lexical_results) + list(bm25_results):
        key = candidate_dedupe_key(result)
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
    return tuple(replace(result, rank=rank) for rank, result in enumerate(merged, 1))


# Explicit alias matching the implemented semantics.
stable_union_by_memory_id = stable_union_by_target_message_id


def retrieve_contextual_union_rerank_evidence(
    question: str,
    conversation: Conversation,
    claims,
    relations,
    embedder: Embedder,
    budget_chars: int,
    *,
    consolidator=None,
    scorer: PairScorer | None = None,
    channel_top_n: int = DEFAULT_CHANNEL_TOP_N,
    bm25_index: BM25Index | None = None,
) -> tuple[SourceEvidence, dict[str, Any]]:
    """Union contextual_lexical and contextual_bm25 pools, then CE-rerank the full union."""
    if channel_top_n < 1:
        raise ValueError("channel_top_n must be >= 1")

    lexical_evidence = retrieve_source_evidence(
        question,
        conversation,
        claims,
        relations,
        embedder,
        channel_top_n,
        budget_chars,
        config=contextual_lexical_config(),
        consolidator=consolidator,
        contextual_enrichment=True,
    )
    bm25_evidence, bm25_meta = retrieve_contextual_bm25_evidence(
        question,
        conversation,
        claims,
        relations,
        channel_top_n,
        budget_chars,
        consolidator=consolidator,
        index=bm25_index,
    )

    union_results = stable_union_by_memory_id(
        lexical_evidence.outcome.results,
        bm25_evidence.outcome.results,
    )
    # Provenance may be multi-source (restatement KEEP). Never strip IDs; never require len==1.
    for result in union_results:
        if not result.source_message_ids:
            raise RuntimeError(
                f"union candidate {result.memory_id!r} has empty source_message_ids"
            )

    union_outcome = RetrievalOutcome(
        question,
        lexical_evidence.outcome.intent,
        union_results,
        lexical_evidence.outcome.excluded,
    )
    union_evidence = evidence_from_outcome(
        question, conversation, union_outcome, budget_chars=budget_chars,
    )

    try:
        resolved = scorer if scorer is not None else load_cross_encoder_reranker()
    except RerankerLoadError:
        raise

    evidence, rows = apply_retrieval_content_rerank(
        question, conversation, union_evidence, resolved, budget_chars=budget_chars,
    )
    # Rerank must preserve full provenance tuples.
    before_sources = {r.memory_id: r.source_message_ids for r in union_results}
    for result in evidence.outcome.results:
        if result.source_message_ids != before_sources[result.memory_id]:
            raise RuntimeError(
                "rerank mutated source_message_ids for "
                f"{result.memory_id}: {before_sources[result.memory_id]!r} -> "
                f"{result.source_message_ids!r}"
            )

    meta: dict[str, Any] = {
        "union": {
            "channelTopN": channel_top_n,
            "lexicalCandidates": len(lexical_evidence.outcome.results),
            "bm25Candidates": len(bm25_evidence.outcome.results),
            "unionSize": len(union_results),
            "dedupeKey": "memory_id",
            "preRerankOrder": "contextual_lexical_then_novel_bm25",
            "truncatedBeforeRerank": False,
            "multiSourceProvenanceAllowed": True,
        },
        "rerank": {
            **(resolved.metadata() if hasattr(resolved, "metadata") else {"model": "custom"}),
            "candidatePool": len(rows),
            "passageSource": "retrieval_content",
            "defaultModel": DEFAULT_RERANK_MODEL,
            "scores": rows,
        },
        "bm25": bm25_meta.get("bm25"),
    }
    return evidence, meta
