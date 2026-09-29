"""Runs the frozen Phase 1 pipeline over a benchmark dataset and scores it."""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from continuator_memory import __version__
from continuator_memory.evaluation import metrics
from continuator_memory.evaluation.dataset import BenchmarkDataset, BenchmarkQuestion
from continuator_memory.generation import GenerationError
from continuator_memory.pipeline import Pipeline

SCHEMA_VERSION = 1
RETRIEVAL_KS = (1, 3, 5, 10)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def evaluate_retrieval(question: BenchmarkQuestion, retrieved: list) -> dict[str, Any]:
    result: dict[str, Any] = {
        "expected_source_message_ids": list(question.expected_source_message_ids),
        "retrieved": [c.to_dict() for c in retrieved],
    }
    if not question.answerable:
        result.update(first_expected_rank=None, reciprocal_rank=None, recall_at={}, hit_at={})
        return result
    expected = question.expected_source_message_ids
    rank = metrics.first_expected_rank(retrieved, question.conversation_id, expected)
    result.update(
        first_expected_rank=rank,
        reciprocal_rank=metrics.reciprocal_rank(rank),
        recall_at={
            str(k): metrics.recall_at_k(retrieved, question.conversation_id, expected, k)
            for k in RETRIEVAL_KS
        },
        hit_at={str(k): metrics.hit_at_k(rank, k) for k in RETRIEVAL_KS},
    )
    return result


def evaluate_answer(question: BenchmarkQuestion, answer: str) -> dict[str, Any]:
    abstained = metrics.is_abstention(answer)
    if question.answerable:
        correct = metrics.match_answer(answer, question.match_any, question.match_all)
    else:
        correct = abstained
    return {
        "correct": correct,
        "abstained": abstained,
        "stale_mentioned": (
            metrics.mentions_any(answer, question.stale_answers) if question.stale_answers else None
        ),
    }


def _retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r["retrieval"] for r in rows if r["answerable"]]
    return {
        "n": len(scored),
        "mrr": metrics.mean([r["reciprocal_rank"] for r in scored]),
        "recall_at": {
            str(k): metrics.mean([r["recall_at"][str(k)] for r in scored]) for k in RETRIEVAL_KS
        },
        "hit_at": {
            str(k): metrics.mean([float(r["hit_at"][str(k)]) for r in scored])
            for k in RETRIEVAL_KS
        },
    }


def _answer_summary(rows: list[dict[str, Any]], generation_top_k: int) -> dict[str, Any]:
    answered = [r for r in rows if r["answer"]["correct"] is not None]
    if not answered:
        return {"n": 0}
    answerable = [r for r in answered if r["answerable"]]
    unanswerable = [r for r in answered if not r["answerable"]]
    stale = [r for r in answered if r["answer"]["stale_mentioned"] is not None]

    def acc(subset: list[dict[str, Any]]) -> float | None:
        return metrics.mean([float(r["answer"]["correct"]) for r in subset])

    summary: dict[str, Any] = {
        "n": len(answered),
        "n_correct": sum(r["answer"]["correct"] for r in answered),
        "accuracy": acc(answered),
        "n_errors": sum(1 for r in answered if r["answer"]["error"]),
        "answerable": {"n": len(answerable), "accuracy": acc(answerable)},
        "abstention": {"n": len(unanswerable), "accuracy": acc(unanswerable)},
        "false_abstention_rate": metrics.mean(
            [float(r["answer"]["abstained"]) for r in answerable]
        ),
        "stale_mention_rate": metrics.mean(
            [float(r["answer"]["stale_mentioned"]) for r in stale]
        ),
    }
    if generation_top_k <= max(RETRIEVAL_KS):
        in_context = [
            metrics.hit_at_k(r["retrieval"]["first_expected_rank"], generation_top_k)
            for r in answerable
        ]
        hit = [r for r, ok in zip(answerable, in_context) if ok]
        miss = [r for r, ok in zip(answerable, in_context) if not ok]
        summary["answerable_by_retrieval"] = {
            "k": generation_top_k,
            "retrieved": {"n": len(hit), "accuracy": acc(hit)},
            "not_retrieved": {"n": len(miss), "accuracy": acc(miss)},
        }
    return summary


def _coverage(row: dict[str, Any], sources: list[list[str]]) -> float:
    expected = set(row["retrieval"]["expected_source_message_ids"])
    covered = {mid for conv, mid in sources if conv == row["conversation_id"]} & expected
    return len(covered) / len(expected)


def _diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset-driven diagnostics; none of them change correctness."""
    answered = [r for r in rows if r["answer"]["correct"] is not None]
    answerable = [r for r in answered if r["answerable"]]
    multi = [r for r in answerable if len(r["retrieval"]["expected_source_message_ids"]) >= 2]
    stale = [r for r in answered if r["answer"]["stale_answers"]]

    def acc(subset: list[dict[str, Any]]) -> float | None:
        return metrics.mean([float(r["answer"]["correct"]) for r in subset])

    out: dict[str, Any] = {
        "current_state": {
            "n": len(stale),
            "accuracy": acc(stale),
            "clean_accuracy": metrics.mean(
                [float(r["answer"]["correct"] and not r["answer"]["stale_mentioned"])
                 for r in stale]
            ),
        },
        "multi_source": {
            "n": len(multi),
            "accuracy": acc(multi),
            "context_coverage": metrics.mean(
                [_coverage(r, r["answer"]["context_source_messages"]) for r in multi]
            ),
        },
        "context_coverage": metrics.mean(
            [_coverage(r, r["answer"]["context_source_messages"]) for r in answerable]
        ),
    }
    traced = [r for r in answered if r["answer"].get("memory") is not None]
    if traced:
        traced_answerable = [r for r in traced if r["answerable"]]
        multi_ids = {r["question_id"] for r in multi}
        traced_multi = [r for r in traced_answerable if r["question_id"] in multi_ids]
        out["memory"] = {
            "n": len(traced),
            "source_coverage": metrics.mean(
                [_coverage(r, r["answer"]["memory"]["source_messages"]) for r in traced_answerable]
            ),
            "multi_source_coverage": metrics.mean(
                [_coverage(r, r["answer"]["memory"]["source_messages"]) for r in traced_multi]
            ),
            "questions_with_relationships": metrics.mean(
                [float(r["answer"]["memory"]["n_relationships"] > 0) for r in traced]
            ),
            "mean_memories_in_context": metrics.mean(
                [float(r["answer"]["memory"]["n_memories"]) for r in traced]
            ),
            "mean_superseded_in_context": metrics.mean(
                [float(r["answer"]["memory"]["n_superseded"]) for r in traced]
            ),
        }
    return out


def aggregate(rows: list[dict[str, Any]], generation_top_k: int) -> dict[str, Any]:
    categories = sorted({r["category"] for r in rows})
    return {
        "diagnostics": _diagnostics(rows),
        "n_questions": len(rows),
        "retrieval": {
            "overall": _retrieval_summary(rows),
            "by_category": {
                c: _retrieval_summary([r for r in rows if r["category"] == c]) for c in categories
            },
        },
        "answer": {
            "overall": _answer_summary(rows, generation_top_k),
            "by_category": {
                c: _answer_summary([r for r in rows if r["category"] == c], generation_top_k)
                for c in categories
            },
        },
    }


def run_evaluation(
    dataset: BenchmarkDataset,
    pipeline: Pipeline,
    generation_top_k: int,
    retrieval_only: bool = False,
    limit: int | None = None,
    config: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ingest every dataset conversation into ``pipeline`` and score each question.

    ``pipeline`` should be backed by a fresh store. Retrieval is scored on the
    top ``max(RETRIEVAL_KS)`` chunks; answers come from ``pipeline.ask`` with
    ``generation_top_k``, i.e. unchanged Phase 1 behaviour.
    """
    started_at = datetime.now(timezone.utc)
    run_start = time.perf_counter()

    ingest_start = time.perf_counter()
    for conversation in dataset.conversations:
        pipeline.ingest(conversation)
    ingest_ms = _ms(ingest_start)

    questions = dataset.questions[:limit] if limit is not None else dataset.questions
    depth = max(RETRIEVAL_KS)
    rows: list[dict[str, Any]] = []
    for i, q in enumerate(questions, start=1):
        if progress:
            progress(f"[{i}/{len(questions)}] {q.question_id}")
        scope = q.retrieval_conversation_id

        t0 = time.perf_counter()
        retrieved = pipeline.retrieve(q.question, depth, scope)
        retrieval = evaluate_retrieval(q, retrieved)
        retrieval["latency_ms"] = _ms(t0)

        answer: dict[str, Any] = {
            "expected_answer": q.expected_answer,
            "match_any": list(q.match_any),
            "match_all": list(q.match_all),
            "stale_answers": list(q.stale_answers),
            "generated_answer": None,
            "context_chunk_ids": [],
            "context_source_messages": [],
            "correct": None,
            "abstained": None,
            "stale_mentioned": None,
            "error": None,
            "latency_ms": None,
        }
        if not retrieval_only:
            t1 = time.perf_counter()
            try:
                result = pipeline.ask(q.question, generation_top_k, scope)
                answer["generated_answer"] = result.answer
                answer["context_chunk_ids"] = [c.chunk_id for c in result.retrieved]
                answer["context_source_messages"] = sorted(
                    [c.conversation_id, mid]
                    for c in result.retrieved
                    for mid in set(c.source_message_ids)
                )
                answer.update(evaluate_answer(q, result.answer))
                trace = getattr(result, "memory_trace", None)
                if trace is not None:
                    answer["memory"] = trace
            except GenerationError as exc:
                answer.update(error=str(exc), correct=False, abstained=False)
            answer["latency_ms"] = _ms(t1)

        rows.append(
            {
                "question_id": q.question_id,
                "category": q.category,
                "conversation_id": q.conversation_id,
                "scope": q.scope,
                "question": q.question,
                "answerable": q.answerable,
                "notes": q.notes,
                "retrieval": retrieval,
                "answer": answer,
                "latency_ms": round(
                    retrieval["latency_ms"] + (answer["latency_ms"] or 0.0), 3
                ),
            }
        )

    finished_at = datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": f"{dataset.dataset_id}__{started_at.strftime('%Y%m%dT%H%M%SZ')}",
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "path": dataset.path,
                "sha256": dataset.sha256,
                "n_conversations": len(dataset.conversations),
                "n_questions_total": len(dataset.questions),
                "n_questions_evaluated": len(questions),
            },
            "config": {
                **(config or {}),
                "generation_top_k": generation_top_k,
                "retrieval_ks": list(RETRIEVAL_KS),
                "retrieval_only": retrieval_only,
                "limit": limit,
            },
            "environment": {
                "package_version": __version__,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "git_commit": _git_commit(),
                **(environment or {}),
            },
        },
        "memory": (
            pipeline.memory_diagnostics() if hasattr(pipeline, "memory_diagnostics") else None
        ),
        "timing": {"ingest_ms": ingest_ms, "total_ms": _ms(run_start)},
        "aggregate": aggregate(rows, generation_top_k),
        "questions": rows,
    }
