"""Multi-query RRF retrieval. Hashing embedder only; no Ollama."""

from __future__ import annotations

import pytest

from continuator_memory.v2.hybrid_retrieval import DEFAULT_RRF_C
from continuator_memory.v2.memory import MemorySource
from continuator_memory.v2.memory_retrieval import RetrievalConfig, RetrievalResult
from continuator_memory.webapi import multiquery
from continuator_memory.webapi.multiquery import (
    RRF_ORIGINAL_WEIGHT_ENV,
    build_memory_retriever,
    load_original_weight,
    retrieve_multiquery_evidence,
    rrf_fusion,
)
from continuator_memory.webapi.query_plan import QueryPlan
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder

QUESTION = "What fields would Alex be likely to pursue in her education?"


def _result(rank: int, memory_id: str, message_id: str, similarity: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        memory_id=memory_id,
        conversation_id="c",
        content=memory_id,
        similarity=similarity,
        adjustment=0.0,
        score=similarity,
        status="current",
        time_class="present",
        event_time=None,
        valid_from=None,
        valid_until=None,
        superseded_by=None,
        sources=(MemorySource(memory_id, f"{memory_id}:c0", message_id, rank - 1),),
        links=(),
    )


def test_rrf_sums_reciprocal_ranks_and_deduplicates():
    left = (_result(1, "a", "m-a"), _result(2, "b", "m-b"))
    right = (_result(1, "b", "m-b"), _result(2, "c", "m-c"))
    fused = rrf_fusion([left, right])
    by_id = {item.memory_id: item for item in fused}
    assert set(by_id) == {"a", "b", "c"}
    assert by_id["b"].fusion == pytest.approx(1 / (DEFAULT_RRF_C + 2) + 1 / (DEFAULT_RRF_C + 1))
    assert by_id["a"].fusion == pytest.approx(1 / (DEFAULT_RRF_C + 1))
    assert fused[0].memory_id == "b"
    assert by_id["b"].ranks == (2, 1)


def _consensus_paraphrases():
    return [
        (
            _result(1, "d1", "m-d1"),
            _result(2, "d2", "m-d2"),
            _result(3, "d3", "m-d3"),
            _result(4, "d4", "m-d4"),
            _result(5, "d5", "m-d5"),
        ),
        (
            _result(1, "d2", "m-d2"),
            _result(2, "d1", "m-d1"),
            _result(3, "d5", "m-d5"),
            _result(4, "d4", "m-d4"),
            _result(5, "d3", "m-d3"),
        ),
        (
            _result(1, "d3", "m-d3"),
            _result(2, "d5", "m-d5"),
            _result(3, "d1", "m-d1"),
            _result(4, "d2", "m-d2"),
            _result(5, "d4", "m-d4"),
        ),
    ]


def test_equal_weight_rrf_can_drop_original_only_top_from_k():
    """Consensus paraphrases outvote an original-only rank-1 under equal RRF."""
    gold = _result(1, "gold", "m-gold")
    original = (
        gold,
        _result(2, "keep", "m-keep"),
        *(_result(index, f"o{index}", f"m-o{index}") for index in range(3, 8)),
    )
    fused = rrf_fusion([original, *_consensus_paraphrases()])
    top5 = [item.memory_id for item in fused[:5]]
    assert fused[0].memory_id != "gold"
    assert "gold" not in top5


def test_original_weight_helps_when_distractors_are_absent_from_original():
    gold = _result(1, "gold", "m-gold")
    original = (
        gold,
        _result(2, "keep", "m-keep"),
        *(_result(index, f"o{index}", f"m-o{index}") for index in range(3, 8)),
    )
    fused = rrf_fusion([original, *_consensus_paraphrases()], original_weight=4.0)
    assert fused[0].memory_id == "gold"
    assert "gold" in [item.memory_id for item in fused[:5]]


def test_original_weight_does_not_protect_distractors_also_on_original():
    """Weighting list 0 also boosts mid-ranked original hits that paraphrases agree on."""
    gold = _result(1, "gold", "m-gold")
    original = (
        gold,
        _result(2, "keep", "m-keep"),
        *(_result(index, f"o{index}", f"m-o{index}") for index in range(3, 8)),
        _result(8, "d1", "m-d1"),
        _result(9, "d2", "m-d2"),
        _result(10, "d3", "m-d3"),
        _result(11, "d4", "m-d4"),
        _result(12, "d5", "m-d5"),
    )
    fused = rrf_fusion([original, *_consensus_paraphrases()], original_weight=4.0)
    assert fused[0].memory_id != "gold"


def test_load_original_weight_defaults_to_equal_rrf():
    assert load_original_weight({}) == 1.0
    assert load_original_weight({RRF_ORIGINAL_WEIGHT_ENV: "4"}) == 4.0


def test_multiquery_keeps_original_and_compares_baseline():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    workspace.set_planner(
        lambda question: QueryPlan(
            information_need="education or work evidence",
            entities=("Alex",),
            facets=("education", "career", "work"),
            queries=(
                question,
                "What career is Alex interested in?",
                "What kind of work does Alex want to do?",
            ),
            source="injected",
            status="ok",
        )
    )
    result = workspace.ask(QUESTION, retrieval_mode="multiquery", generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["diagnostics"]["mode"] == "multiquery"
    assert evidence["queryPlan"]["queries"][0] == QUESTION
    assert "Query plan" in result["answer"]["content"]
    assert evidence["retrievalComparison"]["baseline"]["k"] == 3
    assert evidence["retrievalComparison"]["fused"]["k"] == 3
    assert evidence["retrievalComparison"]["latencyMs"]["total"] >= 0
    assert len(evidence["queryRankings"]) == 3
    assert evidence["fusedRanking"]
    assert evidence["question"] == QUESTION


def test_planner_failure_still_retrieves_with_fallback():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=2)

    def boom(_question: str):
        raise RuntimeError("planner exploded")

    workspace.set_planner(boom)
    result = workspace.ask("weekday bicycle", retrieval_mode="multiquery", generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["queryPlan"]["status"] == "planner_failed"
    assert evidence["queryPlan"]["queries"][0] == "weekday bicycle"
    assert evidence["selectedMessageIds"]
    assert evidence["diagnostics"]["mode"] == "multiquery"


def test_multiquery_reports_retrieval_metrics_when_expected_ids_given():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    evidence, diagnostics = retrieve_multiquery_evidence(
        "weekday bicycle",
        workspace.conversation,
        workspace.claims,
        workspace.relations,
        workspace.embedder(),
        3,
        workspace.budget_chars,
        planner=None,
        expected_message_ids=["m-commute"],
    )
    metrics = diagnostics["retrievalComparison"]["metrics"]
    assert "baseline" in metrics and "fused" in metrics
    assert metrics["fused"]["expectedMessageIds"] == ["m-commute"]
    assert evidence.question == "weekday bicycle"


def test_baseline_mode_unchanged():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    result = workspace.ask("weekday bicycle", retrieval_mode="baseline", generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["diagnostics"]["mode"] == "baseline"
    assert evidence["queryPlan"] is None
    assert evidence["expandedMemoryIds"] == []


def test_build_memory_retriever_builds_contexts_only_when_enabled(monkeypatch):
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    calls = []

    def fake_contexts(conversation, memories):
        calls.append(conversation.conversation_id)
        return {}

    monkeypatch.setattr(multiquery, "previous_turn_contexts", fake_contexts)
    args = (workspace.conversation, workspace.claims, workspace.relations, workspace.embedder())
    assert build_memory_retriever(*args).config.context_weight == 0.0
    build_memory_retriever(*args, RetrievalConfig(lexical_weight=0.25))
    assert calls == []
    enabled = build_memory_retriever(*args, RetrievalConfig(context_weight=0.5))
    assert calls == [workspace.conversation.conversation_id]
    assert enabled.config.context_weight == 0.5
