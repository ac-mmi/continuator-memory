"""Multi-query dense retrieval fused with reciprocal rank fusion.

Each planned query is run through the existing ``MemoryRetriever``. Candidates are
deduplicated by memory ID. The original question is always query 0 and is kept for
answer generation. Consolidation and the answer prompt are unchanged.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable

from continuator_memory.embedding import Embedder
from continuator_memory.evaluation import metrics
from continuator_memory.models import Conversation
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.hybrid_retrieval import DEFAULT_RRF_C
from continuator_memory.v2.memory_retrieval import MemoryRetriever, RetrievalConfig, RetrievalResult
from continuator_memory.v2.models import Claim, Relation
from continuator_memory.v2.turn_context import previous_turn_contexts
from continuator_memory.webapi.query_plan import QueryPlan, resolve_plan

from continuator_memory.v2.source_retrieval import (
    SourceEvidence,
    resolve,
    select_sources,
    source_chunks,
)

DEFAULT_PER_QUERY_K = 32
DEFAULT_ORIGINAL_WEIGHT = 1.0
_SCORE_DIGITS = 12
RETRIEVAL_KS = (5, 10, 15)
RRF_ORIGINAL_WEIGHT_ENV = "CONTINUATOR_RRF_ORIGINAL_WEIGHT"


@dataclass(frozen=True)
class _MetricRow:
    conversation_id: str
    source_message_ids: tuple[str, ...]


@dataclass(frozen=True)
class FusedCandidate:
    memory_id: str
    fusion: float
    ranks: tuple[int | None, ...]
    result: RetrievalResult
    source_message_ids: tuple[str, ...]

    @property
    def min_rank(self) -> int:
        present = [r for r in self.ranks if r is not None]
        return min(present) if present else 1 << 30


def load_original_weight(env: dict[str, str] | None = None) -> float:
    """Experimental. Unset env keeps equal-weight RRF (weight 1)."""
    raw = ((env if env is not None else os.environ).get(RRF_ORIGINAL_WEIGHT_ENV) or "").strip()
    if not raw:
        return DEFAULT_ORIGINAL_WEIGHT
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{RRF_ORIGINAL_WEIGHT_ENV} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{RRF_ORIGINAL_WEIGHT_ENV} must be > 0")
    return value


def rrf_fusion(
    rankings: list[tuple[RetrievalResult, ...]],
    *,
    rrf_c: int = DEFAULT_RRF_C,
    original_weight: float = DEFAULT_ORIGINAL_WEIGHT,
) -> list[FusedCandidate]:
    if rrf_c < 0:
        raise ValueError("rrf_c must be >= 0")
    if original_weight <= 0:
        raise ValueError("original_weight must be > 0")
    n_lists = len(rankings)
    scores: dict[str, float] = {}
    ranks: dict[str, list[int | None]] = {}
    chosen: dict[str, RetrievalResult] = {}
    sources: dict[str, tuple[str, ...]] = {}
    for list_index, results in enumerate(rankings):
        weight = original_weight if list_index == 0 else 1.0
        for result in results:
            mid = result.memory_id
            scores[mid] = scores.get(mid, 0.0) + weight / (rrf_c + result.rank)
            slot = ranks.setdefault(mid, [None] * n_lists)
            if slot[list_index] is None or result.rank < slot[list_index]:
                slot[list_index] = result.rank
            if mid not in chosen:
                chosen[mid] = result
            sources.setdefault(mid, tuple(dict.fromkeys(result.source_message_ids)))
    original_ids = {r.memory_id: r for r in rankings[0]} if rankings else {}
    fused = []
    for mid, fusion in scores.items():
        result = original_ids.get(mid, chosen[mid])
        fused.append(
            FusedCandidate(
                memory_id=mid,
                fusion=fusion,
                ranks=tuple(ranks[mid]),
                result=result,
                source_message_ids=sources[mid] or result.source_message_ids,
            )
        )
    fused.sort(
        key=lambda c: (
            -round(c.fusion, _SCORE_DIGITS),
            c.min_rank,
            c.memory_id,
        )
    )
    return fused


def _ranking_rows(results: Iterable[RetrievalResult]) -> list[dict[str, Any]]:
    return [
        {
            "rank": r.rank,
            "memoryId": r.memory_id,
            "sourceMessageIds": list(r.source_message_ids),
            "similarity": r.similarity,
            "score": r.score,
        }
        for r in results
    ]


def _fused_rows(candidates: list[FusedCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "rank": index,
            "memoryId": item.memory_id,
            "sourceMessageIds": list(item.source_message_ids),
            "fusion": item.fusion,
            "perQueryRanks": list(item.ranks),
            "similarity": item.result.similarity,
        }
        for index, item in enumerate(candidates, 1)
    ]


def fused_ranked_message_ids(candidates: list[FusedCandidate]) -> list[str]:
    """Message IDs in fused-rank order (first occurrence wins). Used for eval metrics."""
    ids: list[str] = []
    for item in candidates:
        for message_id in item.source_message_ids or item.result.source_message_ids:
            if message_id not in ids:
                ids.append(message_id)
    return ids


def resolve_search_depth(k: int, per_query_k: int, depth: int | None) -> int:
    """CLI/eval ``depth`` wins when set; otherwise product default ``max(k, per_query_k)``."""
    if depth is None:
        return max(k, per_query_k)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    return depth


def _message_rows(candidates: list[FusedCandidate], conversation_id: str) -> list[_MetricRow]:
    return [
        _MetricRow(conversation_id, item.source_message_ids or item.result.source_message_ids)
        for item in candidates
    ]


def retrieval_metrics(
    ranked: list[_MetricRow],
    conversation_id: str,
    expected_message_ids: Iterable[str],
    ks: tuple[int, ...] = RETRIEVAL_KS,
) -> dict[str, Any]:
    expected = list(dict.fromkeys(expected_message_ids))
    if not expected:
        return {}
    rank = metrics.first_expected_rank(ranked, conversation_id, expected)
    return {
        "expectedMessageIds": expected,
        "firstExpectedRank": rank,
        "reciprocalRank": metrics.reciprocal_rank(rank),
        "recallAt": {str(k): metrics.recall_at_k(ranked, conversation_id, expected, k) for k in ks},
        "hitAt": {str(k): metrics.hit_at_k(rank, k) for k in ks},
    }


def build_memory_retriever(
    conversation: Conversation,
    claims: Iterable[Claim],
    relations: Iterable[Relation],
    embedder: Embedder,
    config: RetrievalConfig | None = None,
    consolidator=None,
    *,
    contextual_enrichment: bool = False,
) -> MemoryRetriever:
    """Build an in-memory retriever over consolidated memories.

    When ``contextual_enrichment`` is True (``contextual_lexical`` mode), index ephemeral
    Memory copies whose ``content`` includes same-session ±1 neighbour text. Stored memories
    and ``MemorySource`` rows are unchanged; dual-embedding ``context_weight`` contexts are
    not applied (clean Option B experiment).
    """
    if consolidator is None:
        consolidator = consolidate(conversation.conversation_id, claims, relations)
    resolved = config or RetrievalConfig()
    if contextual_enrichment:
        # Do not combine enriched indexed text with the older dual-vector context path.
        resolved = replace(resolved, context_weight=0.0)
    retriever = MemoryRetriever(embedder, resolved)
    if contextual_enrichment:
        from continuator_memory.v2.contextual_units import memories_with_contextual_retrieval_text

        enriched = memories_with_contextual_retrieval_text(
            conversation, consolidator.memories.values(),
        )
        retriever.index(
            enriched,
            consolidator.sources,
            consolidator.links,
            consolidator.conversation_id,
            contexts=None,
        )
        return retriever
    contexts = None
    if retriever.config.context_weight > 0:
        contexts = previous_turn_contexts(conversation, consolidator.memories.values())
    retriever.index_consolidator(consolidator, contexts)
    return retriever


def search_queries(
    retriever: MemoryRetriever,
    queries: Iterable[str],
    conversation_id: str,
    depth: int,
) -> list:
    return [retriever.search(query, depth, conversation_id) for query in queries]


def evidence_from_rankings(
    question: str,
    conversation: Conversation,
    rankings: list[tuple[RetrievalResult, ...]],
    *,
    k: int,
    budget_chars: int,
    rrf_c: int,
    original_outcome,
    original_weight: float = DEFAULT_ORIGINAL_WEIGHT,
) -> tuple[SourceEvidence, list[FusedCandidate], list[FusedCandidate]]:
    fused = rrf_fusion(rankings, rrf_c=rrf_c, original_weight=original_weight)
    selected_fused = fused[:k]
    fused_results = tuple(
        replace(item.result, rank=rank, score=item.fusion)
        for rank, item in enumerate(selected_fused, 1)
    )
    matches = tuple(resolve(result, conversation) for result in fused_results)
    selection = select_sources(matches, budget_chars)
    evidence = SourceEvidence(
        question,
        original_outcome,
        matches,
        selection,
        source_chunks(selection, matches, conversation),
        n_seed_matches=len(matches),
    )
    return evidence, fused, selected_fused


def retrieve_multiquery_evidence(
    question: str,
    conversation: Conversation,
    claims: Iterable[Claim],
    relations: Iterable[Relation],
    embedder: Embedder,
    k: int,
    budget_chars: int,
    *,
    planner=None,
    config: RetrievalConfig | None = None,
    per_query_k: int = DEFAULT_PER_QUERY_K,
    depth: int | None = None,
    rrf_c: int = DEFAULT_RRF_C,
    expected_message_ids: Iterable[str] | None = None,
    retriever: MemoryRetriever | None = None,
    original_weight: float = DEFAULT_ORIGINAL_WEIGHT,
    consolidator=None,
) -> tuple[SourceEvidence, dict[str, Any]]:
    """Plan queries, retrieve each with MemoryRetriever, fuse with RRF, then budget sources.

    ``depth`` is the per-query ranking depth (and fused metric depth). When omitted, product
    behavior keeps ``max(k, per_query_k)``. Eval harnesses pass CLI ``--depth``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if per_query_k < 1:
        raise ValueError("per_query_k must be >= 1")

    started = time.perf_counter()
    plan = resolve_plan(question, planner, embedder=embedder)
    plan_ms = round((time.perf_counter() - started) * 1000, 3)

    retrieve_started = time.perf_counter()
    if retriever is None:
        retriever = build_memory_retriever(
            conversation, claims, relations, embedder, config, consolidator=consolidator,
        )
    search_depth = resolve_search_depth(k, per_query_k, depth)
    outcomes = search_queries(retriever, plan.queries, conversation.conversation_id, search_depth)
    retrieve_ms = round((time.perf_counter() - retrieve_started) * 1000, 3)

    rankings = [outcome.results for outcome in outcomes]
    original = outcomes[0]
    evidence, fused, selected_fused = evidence_from_rankings(
        question,
        conversation,
        rankings,
        k=k,
        budget_chars=budget_chars,
        rrf_c=rrf_c,
        original_outcome=original,
        original_weight=original_weight,
    )
    metric_fused = fused[:search_depth]

    baseline_rows = _message_rows(
        [
            FusedCandidate(r.memory_id, 0.0, (r.rank,), r, r.source_message_ids)
            for r in original.results[:k]
        ],
        conversation.conversation_id,
    )
    fused_metric_rows = _message_rows(selected_fused, conversation.conversation_id)
    expected = list(expected_message_ids or [])
    comparison = {
        "baseline": {
            "k": k,
            "rankings": _ranking_rows(original.results[:k]),
        },
        "fused": {
            "k": k,
            "rankings": _fused_rows(selected_fused),
        },
        "latencyMs": {
            "plan": plan_ms,
            "retrieve": retrieve_ms,
            "total": round((time.perf_counter() - started) * 1000, 3),
        },
    }
    if expected:
        comparison["metrics"] = {
            "baseline": retrieval_metrics(baseline_rows, conversation.conversation_id, expected),
            "fused": retrieval_metrics(fused_metric_rows, conversation.conversation_id, expected),
        }

    diagnostics = {
        "queryPlan": plan.to_dict(),
        "queryRankings": [
            {"query": query, "rankings": _ranking_rows(outcome.results)}
            for query, outcome in zip(plan.queries, outcomes)
        ],
        "fusedRanking": _fused_rows(metric_fused),
        "rankedMessageIds": fused_ranked_message_ids(metric_fused),
        "retrievalComparison": comparison,
        "rrfC": rrf_c,
        "perQueryK": search_depth,
        "originalWeight": original_weight,
    }
    return evidence, diagnostics
