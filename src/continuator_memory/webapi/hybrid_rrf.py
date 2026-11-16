"""Controlled hybrid RRF experiments over existing product retrievers.

Two evaluation modes (names are CLI/product mode strings):

* ``lexical_semantic_rrf`` — run product lexical ranking and product baseline
  (dense semantic) ranking independently on the original question, then fuse
  with equal-weight RRF (``DEFAULT_RRF_C``).
* ``lexical_multiquery_rrf`` — run product lexical ranking **once** on the
  original question; run existing multi-query semantic retrieval (planner +
  per-query dense search + MQ RRF); then RRF-fuse the lexical list with the
  fused multi-query semantic list.

Existing ``baseline`` / ``lexical`` / ``multiquery`` algorithms are unchanged.
Lexical is never re-run per alternative query.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any, Iterable

from continuator_memory.embedding import Embedder
from continuator_memory.models import Conversation
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.hybrid_retrieval import DEFAULT_RRF_C
from continuator_memory.v2.lexical import load_lexical_weight
from continuator_memory.v2.memory_retrieval import MemoryRetriever, RetrievalConfig, RetrievalOutcome, RetrievalResult
from continuator_memory.v2.models import Claim, Relation
from continuator_memory.v2.source_retrieval import SourceEvidence
from continuator_memory.webapi.multiquery import (
    DEFAULT_ORIGINAL_WEIGHT,
    DEFAULT_PER_QUERY_K,
    FusedCandidate,
    _fused_rows,
    _ranking_rows,
    build_memory_retriever,
    evidence_from_rankings,
    fused_ranked_message_ids,
    resolve_search_depth,
    rrf_fusion,
    search_queries,
)
from continuator_memory.webapi.query_plan import resolve_plan

MODE_LEXICAL_SEMANTIC_RRF = "lexical_semantic_rrf"
MODE_LEXICAL_MULTIQUERY_RRF = "lexical_multiquery_rrf"
HYBRID_RRF_MODES = frozenset({MODE_LEXICAL_SEMANTIC_RRF, MODE_LEXICAL_MULTIQUERY_RRF})


def semantic_config() -> RetrievalConfig:
    """Product baseline dense config (lexical_weight=0)."""
    return RetrievalConfig()


def lexical_config() -> RetrievalConfig:
    """Product lexical config (dense + lexical overlap weight)."""
    return RetrievalConfig(lexical_weight=load_lexical_weight())


def ranking_from_fused(fused: list[FusedCandidate]) -> tuple[RetrievalResult, ...]:
    """Re-rank fused candidates as a single RetrievalResult list for a second RRF stage."""
    return tuple(
        replace(item.result, rank=index, score=item.fusion)
        for index, item in enumerate(fused, 1)
    )


def _empty_outcome(question: str) -> RetrievalOutcome:
    from continuator_memory.v2.memory_retrieval import question_intent

    return RetrievalOutcome(question, question_intent(question), ())


def _index_pair(
    conversation: Conversation,
    claims: Iterable[Claim],
    relations: Iterable[Relation],
    embedder: Embedder,
    *,
    consolidator=None,
) -> tuple[Any, MemoryRetriever, MemoryRetriever]:
    if consolidator is None:
        consolidator = consolidate(conversation.conversation_id, claims, relations)
    semantic = build_memory_retriever(
        conversation, claims, relations, embedder, semantic_config(), consolidator=consolidator,
    )
    lexical = build_memory_retriever(
        conversation, claims, relations, embedder, lexical_config(), consolidator=consolidator,
    )
    return consolidator, semantic, lexical


def retrieve_lexical_semantic_rrf_evidence(
    question: str,
    conversation: Conversation,
    claims: Iterable[Claim],
    relations: Iterable[Relation],
    embedder: Embedder,
    k: int,
    budget_chars: int,
    *,
    depth: int | None = None,
    rrf_c: int = DEFAULT_RRF_C,
    per_query_k: int = DEFAULT_PER_QUERY_K,
    consolidator=None,
) -> tuple[SourceEvidence, dict[str, Any]]:
    """A/B hybrid: independent lexical + semantic searches, equal-weight RRF."""
    if k < 1:
        raise ValueError("k must be >= 1")
    text = question.strip()
    search_depth = resolve_search_depth(k, per_query_k, depth)
    started = time.perf_counter()
    _, semantic_retriever, lexical_retriever = _index_pair(
        conversation, claims, relations, embedder, consolidator=consolidator,
    )
    retrieve_started = time.perf_counter()
    semantic_outcome = semantic_retriever.search(text, search_depth, conversation.conversation_id)
    lexical_outcome = lexical_retriever.search(text, search_depth, conversation.conversation_id)
    retrieve_ms = round((time.perf_counter() - retrieve_started) * 1000, 3)

    # Equal-weight RRF: original_weight=1 for list 0 (lexical) matches list 1 (semantic).
    rankings = [lexical_outcome.results, semantic_outcome.results]
    original = lexical_outcome if lexical_outcome.results else semantic_outcome
    if not original.results:
        original = _empty_outcome(text)
    evidence, fused, selected_fused = evidence_from_rankings(
        text,
        conversation,
        rankings,
        k=k,
        budget_chars=budget_chars,
        rrf_c=rrf_c,
        original_outcome=original,
        original_weight=DEFAULT_ORIGINAL_WEIGHT,
    )
    metric_fused = fused[:search_depth]
    diagnostics = {
        "mode": MODE_LEXICAL_SEMANTIC_RRF,
        "lexicalWeight": load_lexical_weight(),
        "rrfC": rrf_c,
        "perQueryK": search_depth,
        "channelRankings": {
            "lexical": _ranking_rows(lexical_outcome.results),
            "semantic": _ranking_rows(semantic_outcome.results),
        },
        "fusedRanking": _fused_rows(metric_fused),
        "rankedMessageIds": fused_ranked_message_ids(metric_fused),
        "retrievalComparison": {
            "latencyMs": {
                "plan": 0.0,
                "retrieve": retrieve_ms,
                "total": round((time.perf_counter() - started) * 1000, 3),
            },
        },
    }
    return evidence, diagnostics


def retrieve_lexical_multiquery_rrf_evidence(
    question: str,
    conversation: Conversation,
    claims: Iterable[Claim],
    relations: Iterable[Relation],
    embedder: Embedder,
    k: int,
    budget_chars: int,
    *,
    planner=None,
    depth: int | None = None,
    rrf_c: int = DEFAULT_RRF_C,
    per_query_k: int = DEFAULT_PER_QUERY_K,
    original_weight: float = DEFAULT_ORIGINAL_WEIGHT,
    consolidator=None,
) -> tuple[SourceEvidence, dict[str, Any]]:
    """Lexical once on the original question + multi-query semantic RRF, then channel RRF."""
    if k < 1:
        raise ValueError("k must be >= 1")
    text = question.strip()
    search_depth = resolve_search_depth(k, per_query_k, depth)
    started = time.perf_counter()

    consolidator, semantic_retriever, lexical_retriever = _index_pair(
        conversation, claims, relations, embedder, consolidator=consolidator,
    )

    # Lexical channel: original question only (never per alternative).
    retrieve_started = time.perf_counter()
    lexical_outcome = lexical_retriever.search(text, search_depth, conversation.conversation_id)
    lexical_ms = round((time.perf_counter() - retrieve_started) * 1000, 3)

    # Semantic multi-query channel: reuse planner + per-query dense search + MQ RRF.
    plan_started = time.perf_counter()
    plan = resolve_plan(text, planner, embedder=embedder)
    plan_ms = round((time.perf_counter() - plan_started) * 1000, 3)

    mq_retrieve_started = time.perf_counter()
    mq_outcomes = search_queries(
        semantic_retriever, plan.queries, conversation.conversation_id, search_depth,
    )
    mq_retrieve_ms = round((time.perf_counter() - mq_retrieve_started) * 1000, 3)
    mq_rankings = [outcome.results for outcome in mq_outcomes]
    if not mq_rankings:
        mq_rankings = [()]
    mq_fused = rrf_fusion(
        mq_rankings, rrf_c=rrf_c, original_weight=original_weight,
    )
    semantic_ranking = ranking_from_fused(mq_fused)

    # Second-stage RRF: lexical list × fused multi-query semantic list (equal weight).
    channel_rankings = [lexical_outcome.results, semantic_ranking]
    original = lexical_outcome if lexical_outcome.results else (
        mq_outcomes[0] if mq_outcomes else _empty_outcome(text)
    )
    evidence, fused, selected_fused = evidence_from_rankings(
        text,
        conversation,
        channel_rankings,
        k=k,
        budget_chars=budget_chars,
        rrf_c=rrf_c,
        original_outcome=original,
        original_weight=DEFAULT_ORIGINAL_WEIGHT,
    )
    metric_fused = fused[:search_depth]
    diagnostics = {
        "mode": MODE_LEXICAL_MULTIQUERY_RRF,
        "lexicalWeight": load_lexical_weight(),
        "rrfC": rrf_c,
        "perQueryK": search_depth,
        "originalWeight": original_weight,
        "queryPlan": plan.to_dict(),
        "queryRankings": [
            {"query": query, "rankings": _ranking_rows(outcome.results)}
            for query, outcome in zip(plan.queries, mq_outcomes)
        ],
        "channelRankings": {
            "lexical": _ranking_rows(lexical_outcome.results),
            "multiquerySemantic": _fused_rows(mq_fused[:search_depth]),
        },
        "fusedRanking": _fused_rows(metric_fused),
        "rankedMessageIds": fused_ranked_message_ids(metric_fused),
        "retrievalComparison": {
            "latencyMs": {
                "plan": plan_ms,
                "retrieveLexical": lexical_ms,
                "retrieveMultiquery": mq_retrieve_ms,
                "retrieve": round(lexical_ms + mq_retrieve_ms, 3),
                "total": round((time.perf_counter() - started) * 1000, 3),
            },
        },
        "lexicalQueryCount": 1,
        "semanticQueryCount": len(plan.queries),
    }
    return evidence, diagnostics


def retrieve_hybrid_rrf_evidence(
    mode: str,
    question: str,
    conversation: Conversation,
    claims: Iterable[Claim],
    relations: Iterable[Relation],
    embedder: Embedder,
    k: int,
    budget_chars: int,
    **kwargs,
) -> tuple[SourceEvidence, dict[str, Any]]:
    if mode == MODE_LEXICAL_SEMANTIC_RRF:
        return retrieve_lexical_semantic_rrf_evidence(
            question, conversation, claims, relations, embedder, k, budget_chars, **kwargs,
        )
    if mode == MODE_LEXICAL_MULTIQUERY_RRF:
        return retrieve_lexical_multiquery_rrf_evidence(
            question, conversation, claims, relations, embedder, k, budget_chars, **kwargs,
        )
    raise ValueError(f"unsupported hybrid RRF mode: {mode!r}")
