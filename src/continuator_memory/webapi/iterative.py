"""Bounded iterative retrieval on top of multi-query RRF.

Round 1 is the existing query plan. Later rounds search only evaluator follow-ups.
All ranked lists are re-fused and cut to the same k. The generated answer is never
used as retrieval evidence.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from continuator_memory.embedding import Embedder
from continuator_memory.models import Conversation
from continuator_memory.v2.hybrid_retrieval import DEFAULT_RRF_C
from continuator_memory.v2.memory_retrieval import MemoryRetriever, RetrievalConfig
from continuator_memory.v2.models import Claim, Relation
from continuator_memory.webapi.evidence_eval import (
    STATUS_EVALUATOR_FAILED,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    STATUS_SUFFICIENT,
    EvidenceExcerpt,
    resolve_evaluation,
)
from continuator_memory.webapi.multiquery import (
    DEFAULT_ORIGINAL_WEIGHT,
    DEFAULT_PER_QUERY_K,
    _fused_rows,
    _message_rows,
    _ranking_rows,
    build_memory_retriever,
    evidence_from_rankings,
    retrieval_metrics,
    search_queries,
)
from continuator_memory.webapi.query_plan import resolve_plan

from continuator_memory.v2.source_retrieval import SourceEvidence

MAX_ROUNDS = 3
STOP_SUFFICIENT = "sufficient"
STOP_NO_GAP = "no_gap"
STOP_NO_NEXT_QUERIES = "no_next_queries"
STOP_DUPLICATE_QUERIES = "duplicate_queries"
STOP_NO_NEW_CANDIDATES = "no_new_candidates"
STOP_MAX_ROUNDS = "max_rounds"
STOP_EVALUATOR_FAILED = "evaluator_failed"


def _excerpts_from_evidence(evidence: SourceEvidence) -> tuple[EvidenceExcerpt, ...]:
    by_id: dict[str, str] = {}
    for match in evidence.matches:
        for src in match.sources:
            by_id.setdefault(src.message_id, src.content)
    return tuple(
        EvidenceExcerpt(mid, by_id[mid])
        for mid in evidence.selection.selected
        if mid in by_id
    )


def retrieve_iterative_evidence(
    question: str,
    conversation: Conversation,
    claims: Iterable[Claim],
    relations: Iterable[Relation],
    embedder: Embedder,
    k: int,
    budget_chars: int,
    *,
    planner=None,
    evaluator=None,
    config: RetrievalConfig | None = None,
    per_query_k: int = DEFAULT_PER_QUERY_K,
    rrf_c: int = DEFAULT_RRF_C,
    expected_message_ids: Iterable[str] | None = None,
    retriever: MemoryRetriever | None = None,
    max_rounds: int = MAX_ROUNDS,
    original_weight: float = DEFAULT_ORIGINAL_WEIGHT,
) -> tuple[SourceEvidence, dict[str, Any]]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if per_query_k < 1:
        raise ValueError("per_query_k must be >= 1")
    if max_rounds < 1 or max_rounds > MAX_ROUNDS:
        raise ValueError(f"max_rounds must be 1..{MAX_ROUNDS}")

    started = time.perf_counter()
    plan = resolve_plan(question, planner)
    plan_ms = round((time.perf_counter() - started) * 1000, 3)

    if retriever is None:
        retriever = build_memory_retriever(conversation, claims, relations, embedder, config)
    depth = max(k, per_query_k)
    cid = conversation.conversation_id

    all_rankings: list = []
    query_blocks: list[dict[str, Any]] = []
    seen_queries: list[str] = []
    selected_memory_ids: set[str] = set()
    rounds: list[dict[str, Any]] = []
    stop_reason = STOP_MAX_ROUNDS
    retrieve_ms = 0.0
    eval_ms = 0.0
    pending_queries = list(plan.queries)
    original_outcome = None
    evidence = None
    fused: list = []
    selected_fused: list = []

    for round_no in range(1, max_rounds + 1):
        if not pending_queries:
            stop_reason = STOP_NO_NEXT_QUERIES
            break
        retrieve_started = time.perf_counter()
        outcomes = search_queries(retriever, pending_queries, cid, depth)
        retrieve_ms += time.perf_counter() - retrieve_started
        if original_outcome is None:
            original_outcome = outcomes[0]
        for query, outcome in zip(pending_queries, outcomes):
            all_rankings.append(outcome.results)
            query_blocks.append({"query": query, "round": round_no, "rankings": _ranking_rows(outcome.results)})
            seen_queries.append(query)
        evidence, fused, selected_fused = evidence_from_rankings(
            question,
            conversation,
            all_rankings,
            k=k,
            budget_chars=budget_chars,
            rrf_c=rrf_c,
            original_outcome=original_outcome,
            original_weight=original_weight,
        )
        new_ids = [item.memory_id for item in selected_fused if item.memory_id not in selected_memory_ids]
        selected_memory_ids.update(item.memory_id for item in selected_fused)
        if round_no > 1 and not new_ids:
            stop_reason = STOP_NO_NEW_CANDIDATES
            rounds.append(_round_row(round_no, pending_queries, new_ids, None, evidence, 0.0))
            break

        excerpts = _excerpts_from_evidence(evidence)
        eval_started = time.perf_counter()
        evaluation = resolve_evaluation(
            question, excerpts, evaluator, seen_queries=seen_queries,
        )
        eval_ms += time.perf_counter() - eval_started
        rounds.append(_round_row(round_no, pending_queries, new_ids, evaluation, evidence, 0.0))

        if evaluation.eval_status != STATUS_OK:
            stop_reason = STOP_EVALUATOR_FAILED
            break
        if evaluation.status == STATUS_SUFFICIENT:
            stop_reason = STOP_SUFFICIENT
            break
        if not evaluation.missing_information.strip() and evaluation.status != STATUS_INSUFFICIENT:
            stop_reason = STOP_NO_GAP
            break
        if not evaluation.missing_information.strip() and evaluation.status == STATUS_INSUFFICIENT:
            stop_reason = STOP_NO_GAP
            break
        if round_no >= max_rounds:
            stop_reason = STOP_MAX_ROUNDS
            break
        if not evaluation.next_queries:
            stop_reason = STOP_NO_NEXT_QUERIES if evaluation.missing_information.strip() else STOP_NO_GAP
            break
        already = {item.casefold() for item in seen_queries}
        if all(item.casefold() in already for item in evaluation.next_queries):
            stop_reason = STOP_DUPLICATE_QUERIES
            break
        pending_queries = list(evaluation.next_queries)

    if evidence is None or original_outcome is None:
        raise RuntimeError("iterative retrieval produced no evidence")

    comparison = {
        "baseline": {
            "k": k,
            "rankings": _ranking_rows(original_outcome.results[:k]),
        },
        "fused": {
            "k": k,
            "rankings": _fused_rows(selected_fused),
        },
        "latencyMs": {
            "plan": plan_ms,
            "retrieve": round(retrieve_ms * 1000, 3),
            "evaluate": round(eval_ms * 1000, 3),
            "total": round((time.perf_counter() - started) * 1000, 3),
        },
    }
    expected = list(expected_message_ids or [])
    if expected:
        from continuator_memory.webapi.multiquery import FusedCandidate

        baseline_rows = _message_rows(
            [
                FusedCandidate(r.memory_id, 0.0, (r.rank,), r, r.source_message_ids)
                for r in original_outcome.results[:k]
            ],
            conversation.conversation_id,
        )
        comparison["metrics"] = {
            "baseline": retrieval_metrics(baseline_rows, conversation.conversation_id, expected),
            "fused": retrieval_metrics(
                _message_rows(selected_fused, conversation.conversation_id),
                conversation.conversation_id,
                expected,
            ),
        }

    diagnostics = {
        "queryPlan": plan.to_dict(),
        "queryRankings": query_blocks,
        "fusedRanking": _fused_rows(fused[:depth]),
        "retrievalComparison": comparison,
        "iterativeRounds": rounds,
        "stopReason": stop_reason,
        "roundCount": len(rounds),
        "rrfC": rrf_c,
        "perQueryK": depth,
        "originalWeight": original_weight,
    }
    return evidence, diagnostics


def _round_row(round_no, queries, new_ids, evaluation, evidence, _latency) -> dict[str, Any]:
    return {
        "round": round_no,
        "queries": list(queries),
        "newMemoryIds": list(new_ids),
        "selectedMessageIds": list(evidence.selection.selected),
        "evaluation": evaluation.to_dict() if evaluation is not None else None,
    }
