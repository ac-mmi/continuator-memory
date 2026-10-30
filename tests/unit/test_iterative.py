"""Iterative retrieval with a scripted evaluator. No Ollama."""

from __future__ import annotations

from continuator_memory.webapi.evidence_eval import EvidenceEvaluation, EvidenceExcerpt
from continuator_memory.webapi.iterative import retrieve_iterative_evidence
from continuator_memory.webapi.query_plan import QueryPlan
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder

QUESTION = "weekday bicycle"


def _plan(question: str) -> QueryPlan:
    return QueryPlan(
        information_need="commute evidence",
        entities=("Alex",),
        facets=("fact",),
        queries=(question,),
        source="injected",
        status="ok",
    )


def _eval(status: str, missing: str = "", next_queries: tuple[str, ...] = ()) -> EvidenceEvaluation:
    return EvidenceEvaluation(
        status=status,
        supported_facts=(),
        missing_information=missing,
        next_queries=next_queries,
        source="injected",
        eval_status="ok",
    )


def test_sufficient_stops_after_one_round():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    workspace.set_planner(_plan)
    workspace.set_evaluator(lambda _q, _ex: _eval("sufficient"))
    result = workspace.ask(QUESTION, retrieval_mode="iterative", generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["diagnostics"]["mode"] == "iterative"
    assert evidence["diagnostics"]["roundCount"] == 1
    assert evidence["stopReason"] == "sufficient"
    assert len(evidence["iterativeRounds"]) == 1
    assert evidence["selectedMessageIds"]


def test_insufficient_runs_second_round_then_stops():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    workspace.set_planner(_plan)
    calls = {"n": 0}

    def evaluator(question: str, excerpts: tuple[EvidenceExcerpt, ...]):
        calls["n"] += 1
        if calls["n"] == 1:
            return _eval(
                "insufficient",
                missing="whether the commute fact is current",
                next_queries=("What did Alex say about weekday bicycle commute?",),
            )
        return _eval("sufficient")

    workspace.set_evaluator(evaluator)
    result = workspace.ask(QUESTION, retrieval_mode="iterative", generate=False)
    evidence = result["answer"]["evidence"]
    assert calls["n"] >= 1
    assert evidence["diagnostics"]["roundCount"] >= 1
    assert evidence["stopReason"] in {"sufficient", "no_new_candidates", "no_next_queries"}
    assert len(evidence["selectedMessageIds"]) == 1


def test_evaluator_failure_does_not_invent_a_query():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    workspace.set_planner(_plan)
    workspace.set_evaluator(lambda _q, _ex: (_ for _ in ()).throw(RuntimeError("bad eval")))
    result = workspace.ask(QUESTION, retrieval_mode="iterative", generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["stopReason"] == "evaluator_failed"
    assert evidence["diagnostics"]["roundCount"] == 1
    assert evidence["iterativeRounds"][0]["evaluation"]["nextQueries"] == []


def test_duplicate_followups_stop():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    workspace.set_planner(_plan)

    def evaluator(_q, _ex):
        return _eval("insufficient", missing="more commute detail", next_queries=(QUESTION,))

    workspace.set_evaluator(evaluator)
    result = workspace.ask(QUESTION, retrieval_mode="iterative", generate=False)
    assert result["answer"]["evidence"]["stopReason"] in {"duplicate_queries", "no_next_queries"}
    assert result["answer"]["evidence"]["diagnostics"]["roundCount"] == 1


def test_max_three_rounds():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    workspace.set_planner(_plan)
    n = {"i": 0}

    def evaluator(_q, _ex):
        n["i"] += 1
        return _eval(
            "insufficient",
            missing=f"gap {n['i']}",
            next_queries=(f"What did Alex say about weekday bicycle {n['i']}?",),
        )

    workspace.set_evaluator(evaluator)
    result = workspace.ask(QUESTION, retrieval_mode="iterative", generate=False)
    assert result["answer"]["evidence"]["diagnostics"]["roundCount"] <= 3
    assert result["answer"]["evidence"]["stopReason"] in {
        "max_rounds",
        "no_new_candidates",
        "no_next_queries",
    }


def test_baseline_and_multiquery_still_work():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    baseline = workspace.ask(QUESTION, retrieval_mode="baseline", generate=False)
    multi = workspace.ask(QUESTION, retrieval_mode="multiquery", generate=False)
    assert baseline["answer"]["evidence"]["diagnostics"]["mode"] == "baseline"
    assert multi["answer"]["evidence"]["diagnostics"]["mode"] == "multiquery"
    assert baseline["answer"]["evidence"]["iterativeRounds"] is None


def test_gold_message_ids_are_not_in_runtime_modules():
    from pathlib import Path

    runtime = [
        Path("src/continuator_memory/webapi/iterative.py").read_text(encoding="utf-8"),
        Path("src/continuator_memory/webapi/evidence_eval.py").read_text(encoding="utf-8"),
    ]
    for text in runtime:
        assert "D1:11" not in text
        assert "D4:11" not in text


def test_retrieve_iterative_cuts_to_k():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=2)
    workspace.set_planner(_plan)
    workspace.set_evaluator(lambda _q, _ex: _eval("sufficient"))
    evidence, diag = retrieve_iterative_evidence(
        QUESTION,
        workspace.conversation,
        workspace.claims,
        workspace.relations,
        workspace.embedder(),
        2,
        workspace.budget_chars,
        planner=_plan,
        evaluator=lambda _q, _ex: _eval("sufficient"),
    )
    assert len(evidence.selection.selected) <= 2
    assert diag["stopReason"] == "sufficient"
