"""Contextual Lexical + pretrained cross-encoder rerank (experimental product mode).

Pipeline (general-purpose; no dataset-specific logic)::

    contextual_lexical candidate generation
      → top-N RetrievalOutcome (enriched retrieval text, target-only sources)
      → pretrained cross-encoder scores (query, retrieval_content)
      → same candidate set, reordered

Fails closed if the cross-encoder cannot be loaded.
"""

from __future__ import annotations

from typing import Any

from continuator_memory.embedding import Embedder
from continuator_memory.models import Conversation
from continuator_memory.v2.lexical import load_lexical_weight
from continuator_memory.v2.memory_retrieval import RetrievalConfig, RetrievalOutcome
from continuator_memory.v2.rerank import (
    DEFAULT_RERANK_MODEL,
    PairScorer,
    RerankerLoadError,
    load_cross_encoder_reranker,
    rerank,
)
from continuator_memory.v2.bprime_links import empty_trace
from continuator_memory.v2.source_retrieval import (
    SourceEvidence,
    resolve,
    retrieve_source_evidence,
    select_sources,
    source_chunks,
)

MODE_CONTEXTUAL_LEXICAL_RERANK = "contextual_lexical_rerank"


def contextual_lexical_config() -> RetrievalConfig:
    return RetrievalConfig(lexical_weight=load_lexical_weight(), context_weight=0.0)


def evidence_from_outcome(
    question: str,
    conversation: Conversation,
    outcome: RetrievalOutcome,
    *,
    budget_chars: int,
) -> SourceEvidence:
    """Rebuild source evidence after reordering ``outcome.results`` (same memories)."""
    matches = tuple(resolve(result, conversation) for result in outcome.results)
    selection = select_sources(matches, budget_chars)
    return SourceEvidence(
        question,
        outcome,
        matches,
        selection,
        source_chunks(selection, matches, conversation),
        bprime=empty_trace(enabled=False),
        n_seed_matches=len(matches),
    )


def apply_retrieval_content_rerank(
    question: str,
    conversation: Conversation,
    evidence: SourceEvidence,
    scorer: PairScorer,
    *,
    budget_chars: int,
) -> tuple[SourceEvidence, list[dict[str, Any]]]:
    """Reorder ``evidence.outcome`` using indexed/retrieval text; preserve source ids."""
    before_ids = tuple(r.memory_id for r in evidence.outcome.results)
    before_sources = tuple(r.source_message_ids for r in evidence.outcome.results)
    reranked, rows = rerank(
        question,
        evidence.outcome,
        conversation,
        scorer,
        passage_source="retrieval_content",
    )
    after_ids = tuple(r.memory_id for r in reranked.results)
    if sorted(after_ids) != sorted(before_ids):
        raise RuntimeError("rerank changed the candidate memory set")
    if len(after_ids) != len(before_ids):
        raise RuntimeError("rerank changed the candidate pool size")
    # Sources must remain attached to the same memories (target-only for contextual units).
    by_id = {r.memory_id: src for r, src in zip(evidence.outcome.results, before_sources)}
    for result in reranked.results:
        if result.source_message_ids != by_id[result.memory_id]:
            raise RuntimeError("rerank mutated source_message_ids")
    rebuilt = evidence_from_outcome(
        question, conversation, reranked, budget_chars=budget_chars,
    )
    return rebuilt, rows


def retrieve_contextual_lexical_rerank_evidence(
    question: str,
    conversation: Conversation,
    claims,
    relations,
    embedder: Embedder,
    k: int,
    budget_chars: int,
    *,
    consolidator=None,
    scorer: PairScorer | None = None,
) -> tuple[SourceEvidence, dict[str, Any]]:
    """Run contextual_lexical retrieval then pretrained CE rerank on the same pool."""
    base = retrieve_source_evidence(
        question,
        conversation,
        claims,
        relations,
        embedder,
        k,
        budget_chars,
        config=contextual_lexical_config(),
        consolidator=consolidator,
        contextual_enrichment=True,
    )
    try:
        resolved = scorer if scorer is not None else load_cross_encoder_reranker()
    except RerankerLoadError:
        raise
    evidence, rows = apply_retrieval_content_rerank(
        question, conversation, base, resolved, budget_chars=budget_chars,
    )
    meta: dict[str, Any] = {
        "rerank": {
            **(resolved.metadata() if hasattr(resolved, "metadata") else {"model": "custom"}),
            "candidatePool": len(rows),
            "passageSource": "retrieval_content",
            "defaultModel": DEFAULT_RERANK_MODEL,
            "scores": rows,
        }
    }
    return evidence, meta
