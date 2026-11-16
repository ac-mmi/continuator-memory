"""Wide contextual union + optional reduction + pretrained CE (experimental).

Controlled multi-phase candidate-generation experiment. Does **not** change
``contextual_union_rerank`` (depth-50) or other product modes.

Phases
------
1. High recall: ``contextual_lexical`` top-N and ``contextual_bm25`` top-N
   (default ``DEFAULT_WIDE_CHANNEL_TOP_N`` = 200).
2. Stable union by ``memory_id`` (lexical first, then novel BM25); full
   ``source_message_ids`` provenance preserved.
3. Optional reduction (ablation):
   - ``none`` — full union → CE
   - ``rrf`` — standard RRF → top ``DEFAULT_REDUCED_TOP_N`` → CE
   - ``channel_quota`` — symmetric per-channel reserve + fill → CE
4. Pretrained cross-encoder reorders only (CONTINUATOR_RERANK_*).

No LoCoMo-specific parameters, training, or gold-label use in retrieval.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, Sequence

from continuator_memory.embedding import Embedder
from continuator_memory.models import Conversation
from continuator_memory.v2.bm25 import BM25Index
from continuator_memory.v2.hybrid_retrieval import DEFAULT_RRF_C
from continuator_memory.v2.memory_retrieval import RetrievalOutcome, RetrievalResult
from continuator_memory.v2.rerank import (
    DEFAULT_RERANK_MODEL,
    PairScorer,
    RerankerLoadError,
    load_cross_encoder_reranker,
)
from continuator_memory.v2.source_retrieval import SourceEvidence, retrieve_source_evidence
from continuator_memory.webapi.contextual_bm25 import retrieve_contextual_bm25_evidence
from continuator_memory.webapi.contextual_rerank import (
    apply_retrieval_content_rerank,
    contextual_lexical_config,
    evidence_from_outcome,
)
from continuator_memory.webapi.contextual_union_rerank import (
    candidate_dedupe_key,
    stable_union_by_memory_id,
)
from continuator_memory.webapi.multiquery import rrf_fusion

MODE_CONTEXTUAL_WIDE_UNION_RERANK = "contextual_wide_union_rerank"
MODE_CONTEXTUAL_WIDE_UNION_RRF_RERANK = "contextual_wide_union_rrf_rerank"
MODE_CONTEXTUAL_WIDE_UNION_QUOTA_RERANK = "contextual_wide_union_quota_rerank"

WIDE_UNION_RERANK_MODES = frozenset({
    MODE_CONTEXTUAL_WIDE_UNION_RERANK,
    MODE_CONTEXTUAL_WIDE_UNION_RRF_RERANK,
    MODE_CONTEXTUAL_WIDE_UNION_QUOTA_RERANK,
})

# General experimental configuration (not tuned on any benchmark).
DEFAULT_WIDE_CHANNEL_TOP_N = 200
DEFAULT_REDUCED_TOP_N = 100
DEFAULT_WIDE_RRF_C = DEFAULT_RRF_C  # standard RRF constant (60)
# Symmetric reserve from each channel before fill; default half of reduced pool.
DEFAULT_QUOTA_PER_CHANNEL = DEFAULT_REDUCED_TOP_N // 2

ReducerName = Literal["none", "rrf", "channel_quota"]

MODE_TO_REDUCER: dict[str, ReducerName] = {
    MODE_CONTEXTUAL_WIDE_UNION_RERANK: "none",
    MODE_CONTEXTUAL_WIDE_UNION_RRF_RERANK: "rrf",
    MODE_CONTEXTUAL_WIDE_UNION_QUOTA_RERANK: "channel_quota",
}


def union_channel_membership(
    lexical_results: Sequence[RetrievalResult],
    bm25_results: Sequence[RetrievalResult],
) -> dict[str, Any]:
    lex_ids = {candidate_dedupe_key(r) for r in lexical_results}
    bm_ids = {candidate_dedupe_key(r) for r in bm25_results}
    overlap = lex_ids & bm_ids
    return {
        "lexicalIds": frozenset(lex_ids),
        "bm25Ids": frozenset(bm_ids),
        "overlapCount": len(overlap),
        "lexicalOnlyCount": len(lex_ids - bm_ids),
        "bm25OnlyCount": len(bm_ids - lex_ids),
    }


def reduce_none(
    union_results: Sequence[RetrievalResult],
) -> tuple[tuple[RetrievalResult, ...], tuple[str, ...]]:
    """Pass the full union through; nothing dropped."""
    kept = tuple(
        replace(result, rank=rank) for rank, result in enumerate(union_results, 1)
    )
    return kept, ()


def reduce_by_rrf(
    union_results: Sequence[RetrievalResult],
    lexical_results: Sequence[RetrievalResult],
    bm25_results: Sequence[RetrievalResult],
    *,
    reduced_top_n: int,
    rrf_c: int = DEFAULT_WIDE_RRF_C,
) -> tuple[tuple[RetrievalResult, ...], tuple[str, ...]]:
    """Deterministic RRF over the two channel rankings; keep top ``reduced_top_n``.

    Kept ``RetrievalResult`` objects come from the stable union (lexical-first
    provenance/content). Channel ranks for fusion come from each channel list.
    """
    if reduced_top_n < 1:
        raise ValueError("reduced_top_n must be >= 1")
    if rrf_c < 0:
        raise ValueError("rrf_c must be >= 0")
    by_id = {candidate_dedupe_key(r): r for r in union_results}
    fused = rrf_fusion(
        [tuple(lexical_results), tuple(bm25_results)],
        rrf_c=rrf_c,
        original_weight=1.0,
    )
    kept_ids: list[str] = []
    seen: set[str] = set()
    for item in fused:
        mid = item.memory_id
        if mid not in by_id or mid in seen:
            continue
        seen.add(mid)
        kept_ids.append(mid)
        if len(kept_ids) >= reduced_top_n:
            break
    kept = tuple(
        replace(by_id[mid], rank=rank) for rank, mid in enumerate(kept_ids, 1)
    )
    dropped = tuple(
        candidate_dedupe_key(r) for r in union_results if candidate_dedupe_key(r) not in seen
    )
    return kept, dropped


def reduce_by_channel_quota(
    union_results: Sequence[RetrievalResult],
    lexical_results: Sequence[RetrievalResult],
    bm25_results: Sequence[RetrievalResult],
    *,
    reduced_top_n: int,
    quota_per_channel: int = DEFAULT_QUOTA_PER_CHANNEL,
) -> tuple[tuple[RetrievalResult, ...], tuple[str, ...]]:
    """Recall-preserving reducer: reserve each channel, dedupe, then fill.

    Policy (symmetric, deterministic, not tuned on benchmarks):
      1. Reserve top ``quota_per_channel`` from contextual_lexical (by channel rank)
      2. Reserve top ``quota_per_channel`` from contextual_bm25
      3. Dedupe by ``memory_id`` (lexical reservation order first, then BM25 novel)
      4. Fill remaining capacity from stable-union order
    """
    if reduced_top_n < 1:
        raise ValueError("reduced_top_n must be >= 1")
    if quota_per_channel < 0:
        raise ValueError("quota_per_channel must be >= 0")
    by_id = {candidate_dedupe_key(r): r for r in union_results}
    lex_sorted = sorted(lexical_results, key=lambda r: (r.rank, candidate_dedupe_key(r)))
    bm_sorted = sorted(bm25_results, key=lambda r: (r.rank, candidate_dedupe_key(r)))

    ordered_ids: list[str] = []
    seen: set[str] = set()

    def _reserve(results: Sequence[RetrievalResult], quota: int) -> None:
        """Accept up to ``quota`` novel memory_ids from this channel's rank order."""
        accepted = 0
        for result in results:
            if accepted >= quota or len(ordered_ids) >= reduced_top_n:
                break
            mid = candidate_dedupe_key(result)
            if mid not in by_id or mid in seen:
                continue
            seen.add(mid)
            ordered_ids.append(mid)
            accepted += 1

    _reserve(lex_sorted, quota_per_channel)
    _reserve(bm_sorted, quota_per_channel)

    for result in union_results:
        if len(ordered_ids) >= reduced_top_n:
            break
        mid = candidate_dedupe_key(result)
        if mid in seen:
            continue
        seen.add(mid)
        ordered_ids.append(mid)

    kept = tuple(
        replace(by_id[mid], rank=rank) for rank, mid in enumerate(ordered_ids, 1)
    )
    dropped = tuple(
        candidate_dedupe_key(r) for r in union_results if candidate_dedupe_key(r) not in seen
    )
    return kept, dropped


def apply_reduction(
    strategy: ReducerName,
    union_results: Sequence[RetrievalResult],
    lexical_results: Sequence[RetrievalResult],
    bm25_results: Sequence[RetrievalResult],
    *,
    reduced_top_n: int,
    rrf_c: int,
    quota_per_channel: int,
) -> tuple[tuple[RetrievalResult, ...], tuple[str, ...], dict[str, Any]]:
    before = len(union_results)
    if strategy == "none":
        kept, dropped = reduce_none(union_results)
    elif strategy == "rrf":
        kept, dropped = reduce_by_rrf(
            union_results,
            lexical_results,
            bm25_results,
            reduced_top_n=reduced_top_n,
            rrf_c=rrf_c,
        )
    elif strategy == "channel_quota":
        kept, dropped = reduce_by_channel_quota(
            union_results,
            lexical_results,
            bm25_results,
            reduced_top_n=reduced_top_n,
            quota_per_channel=quota_per_channel,
        )
    else:
        raise ValueError(f"unknown reduction strategy: {strategy!r}")

    meta = {
        "strategy": strategy,
        "beforeCount": before,
        "afterCount": len(kept),
        "droppedCount": len(dropped),
        "droppedMemoryIds": list(dropped),
        "reducedTopN": None if strategy == "none" else reduced_top_n,
        "rrfC": rrf_c if strategy == "rrf" else None,
        "quotaPerChannel": quota_per_channel if strategy == "channel_quota" else None,
    }
    return kept, dropped, meta


def retrieve_contextual_wide_union_rerank_evidence(
    question: str,
    conversation: Conversation,
    claims,
    relations,
    embedder: Embedder,
    budget_chars: int,
    *,
    consolidator=None,
    scorer: PairScorer | None = None,
    channel_top_n: int = DEFAULT_WIDE_CHANNEL_TOP_N,
    reducer: ReducerName = "none",
    reduced_top_n: int = DEFAULT_REDUCED_TOP_N,
    rrf_c: int = DEFAULT_WIDE_RRF_C,
    quota_per_channel: int = DEFAULT_QUOTA_PER_CHANNEL,
    bm25_index: BM25Index | None = None,
) -> tuple[SourceEvidence, dict[str, Any]]:
    """Wide dual-channel union, optional reduction, then CE reorder."""
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

    lexical_results = lexical_evidence.outcome.results
    bm25_results = bm25_evidence.outcome.results
    membership = union_channel_membership(lexical_results, bm25_results)

    union_results = stable_union_by_memory_id(lexical_results, bm25_results)
    for result in union_results:
        if not result.source_message_ids:
            raise RuntimeError(
                f"union candidate {result.memory_id!r} has empty source_message_ids"
            )

    reduced_results, _dropped, reduction_meta = apply_reduction(
        reducer,
        union_results,
        lexical_results,
        bm25_results,
        reduced_top_n=reduced_top_n,
        rrf_c=rrf_c,
        quota_per_channel=quota_per_channel,
    )
    if len(reduced_results) != len({candidate_dedupe_key(r) for r in reduced_results}):
        raise RuntimeError("reduction produced duplicate memory_id candidates")

    # Candidate accounting invariant.
    if reduction_meta["beforeCount"] != len(union_results):
        raise RuntimeError("reduction beforeCount mismatch")
    if reduction_meta["afterCount"] + reduction_meta["droppedCount"] != reduction_meta["beforeCount"]:
        raise RuntimeError("reduction accounting mismatch")

    pre_rerank_sources = {r.memory_id: r.source_message_ids for r in reduced_results}
    reduced_outcome = RetrievalOutcome(
        question,
        lexical_evidence.outcome.intent,
        reduced_results,
        lexical_evidence.outcome.excluded,
    )
    reduced_evidence = evidence_from_outcome(
        question, conversation, reduced_outcome, budget_chars=budget_chars,
    )

    try:
        resolved = scorer if scorer is not None else load_cross_encoder_reranker()
    except RerankerLoadError:
        raise

    evidence, rows = apply_retrieval_content_rerank(
        question, conversation, reduced_evidence, resolved, budget_chars=budget_chars,
    )
    if len(evidence.outcome.results) != len(reduced_results):
        raise RuntimeError(
            "rerank changed candidate count: "
            f"{len(reduced_results)} -> {len(evidence.outcome.results)}"
        )
    after_ids = {r.memory_id for r in evidence.outcome.results}
    if after_ids != set(pre_rerank_sources):
        raise RuntimeError("rerank mutated candidate memory_id set")
    for result in evidence.outcome.results:
        if result.source_message_ids != pre_rerank_sources[result.memory_id]:
            raise RuntimeError(
                "rerank mutated source_message_ids for "
                f"{result.memory_id}: {pre_rerank_sources[result.memory_id]!r} -> "
                f"{result.source_message_ids!r}"
            )

    meta: dict[str, Any] = {
        "phase1": {
            "channelTopN": channel_top_n,
            "lexicalCandidates": len(lexical_results),
            "bm25Candidates": len(bm25_results),
        },
        "union": {
            "channelTopN": channel_top_n,
            "lexicalCandidates": len(lexical_results),
            "bm25Candidates": len(bm25_results),
            "unionSize": len(union_results),
            "overlapCount": membership["overlapCount"],
            "lexicalOnlyCount": membership["lexicalOnlyCount"],
            "bm25OnlyCount": membership["bm25OnlyCount"],
            "dedupeKey": "memory_id",
            "preRerankOrder": "contextual_lexical_then_novel_bm25",
            "multiSourceProvenanceAllowed": True,
        },
        "reduction": reduction_meta,
        "rerank": {
            **(resolved.metadata() if hasattr(resolved, "metadata") else {"model": "custom"}),
            "candidatePool": len(rows),
            "candidatesIn": len(reduced_results),
            "candidatesOut": len(evidence.outcome.results),
            "passageSource": "retrieval_content",
            "defaultModel": DEFAULT_RERANK_MODEL,
            "scores": rows,
        },
        "bm25": bm25_meta.get("bm25"),
        # Compatibility aliases used by some diagnostics surfaces.
        "truncatedBeforeRerank": reducer != "none",
        "reducer": reducer,
    }
    return evidence, meta


def retrieve_for_mode(
    mode: str,
    question: str,
    conversation: Conversation,
    claims,
    relations,
    embedder: Embedder,
    budget_chars: int,
    **kwargs: Any,
) -> tuple[SourceEvidence, dict[str, Any]]:
    """Dispatch one of the three wide-union experimental modes."""
    if mode not in MODE_TO_REDUCER:
        raise ValueError(f"not a wide-union mode: {mode!r}")
    return retrieve_contextual_wide_union_rerank_evidence(
        question,
        conversation,
        claims,
        relations,
        embedder,
        budget_chars,
        reducer=MODE_TO_REDUCER[mode],
        **kwargs,
    )

