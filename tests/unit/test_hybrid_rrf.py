"""Hybrid lexical/semantic RRF experiment modes. Hashing embedder only; no Ollama."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from continuator_memory.v2.hybrid_retrieval import DEFAULT_RRF_C
from continuator_memory.v2.memory import MemorySource
from continuator_memory.v2.memory_retrieval import RetrievalResult
from continuator_memory.webapi.hybrid_rrf import (
    MODE_LEXICAL_MULTIQUERY_RRF,
    MODE_LEXICAL_SEMANTIC_RRF,
    ranking_from_fused,
    retrieve_lexical_semantic_rrf_evidence,
)
from continuator_memory.webapi.multiquery import FusedCandidate, rrf_fusion
from continuator_memory.webapi.query_plan import QueryPlan
from continuator_memory.webapi.workspace import RETRIEVAL_MODES, Workspace, normalize_retrieval_mode
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


def test_modes_are_registered():
    assert MODE_LEXICAL_SEMANTIC_RRF in RETRIEVAL_MODES
    assert MODE_LEXICAL_MULTIQUERY_RRF in RETRIEVAL_MODES
    assert normalize_retrieval_mode(MODE_LEXICAL_SEMANTIC_RRF) == MODE_LEXICAL_SEMANTIC_RRF
    assert normalize_retrieval_mode("hybrid") == "lexical"


def test_rrf_fuses_lexical_and_semantic_lists_and_deduplicates():
    lexical = (_result(1, "lex-only", "m-lex"), _result(2, "both", "m-both"))
    semantic = (_result(1, "both", "m-both"), _result(2, "sem-only", "m-sem"))
    fused = rrf_fusion([lexical, semantic], rrf_c=DEFAULT_RRF_C)
    by_id = {item.memory_id: item for item in fused}
    assert set(by_id) == {"lex-only", "both", "sem-only"}
    assert by_id["both"].fusion == pytest.approx(
        1 / (DEFAULT_RRF_C + 2) + 1 / (DEFAULT_RRF_C + 1)
    )
    assert by_id["lex-only"].fusion == pytest.approx(1 / (DEFAULT_RRF_C + 1))
    assert by_id["sem-only"].fusion == pytest.approx(1 / (DEFAULT_RRF_C + 2))
    assert fused[0].memory_id == "both"
    assert by_id["both"].ranks == (2, 1)


def test_rrf_keeps_items_present_in_only_one_ranking():
    lexical = (_result(1, "only-lex", "m-l"),)
    semantic = (_result(1, "only-sem", "m-s"),)
    fused = rrf_fusion([lexical, semantic])
    assert [item.memory_id for item in fused] == ["only-lex", "only-sem"]
    assert fused[0].ranks == (1, None)
    assert fused[1].ranks == (None, 1)


def test_rrf_handles_empty_rankings():
    assert rrf_fusion([(), ()]) == []
    only_right = rrf_fusion([(), (_result(1, "a", "m-a"),)])
    assert [item.memory_id for item in only_right] == ["a"]
    only_left = rrf_fusion([(_result(1, "b", "m-b"),), ()])
    assert [item.memory_id for item in only_left] == ["b"]


def test_ranking_from_fused_preserves_order_and_ids():
    fused = rrf_fusion([
        (_result(1, "a", "m-a"), _result(2, "b", "m-b")),
        (_result(1, "b", "m-b"),),
    ])
    ranking = ranking_from_fused(fused)
    assert [row.memory_id for row in ranking] == [item.memory_id for item in fused]
    assert [row.rank for row in ranking] == list(range(1, len(ranking) + 1))


def test_second_stage_rrf_with_multiquery_semantic_list():
    lexical = (_result(1, "lex", "m-lex"), _result(2, "shared", "m-shared"))
    mq_fused = [
        FusedCandidate("shared", 0.03, (1, 1), _result(1, "shared", "m-shared"), ("m-shared",)),
        FusedCandidate("mq", 0.02, (2, None), _result(2, "mq", "m-mq"), ("m-mq",)),
    ]
    semantic_ranking = ranking_from_fused(mq_fused)
    final = rrf_fusion([lexical, semantic_ranking])
    assert {item.memory_id for item in final} == {"lex", "shared", "mq"}
    assert final[0].memory_id == "shared"


def test_lexical_semantic_rrf_workspace_returns_evidence():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    result = workspace.ask(QUESTION, retrieval_mode=MODE_LEXICAL_SEMANTIC_RRF, generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["diagnostics"]["mode"] == MODE_LEXICAL_SEMANTIC_RRF
    assert evidence["diagnostics"]["rrfC"] == DEFAULT_RRF_C
    assert "lexicalWeight" in evidence["diagnostics"]
    assert evidence["question"] == QUESTION


def test_baseline_and_lexical_modes_unchanged():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    baseline = workspace.ask(QUESTION, retrieval_mode="baseline", generate=False)
    lexical = workspace.ask(QUESTION, retrieval_mode="lexical", generate=False)
    assert baseline["answer"]["evidence"]["diagnostics"]["mode"] == "baseline"
    assert lexical["answer"]["evidence"]["diagnostics"]["mode"] == "lexical"
    assert "lexicalWeight" in lexical["answer"]["evidence"]["diagnostics"]
    assert "lexicalWeight" not in baseline["answer"]["evidence"]["diagnostics"]


def test_lexical_multiquery_rrf_runs_lexical_once_and_uses_planner(monkeypatch):
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    plan = QueryPlan(
        information_need="education evidence",
        entities=("Alex",),
        facets=("education",),
        queries=(
            QUESTION,
            "What career is Alex interested in?",
            "What kind of work does Alex want to do?",
        ),
        source="injected",
        status="ok",
    )
    workspace.set_planner(lambda _q: plan)

    calls: list[tuple[str, object]] = []
    real_search = None

    def tracking_search(self, question, k=None, conversation_id=None):
        calls.append((question, self.config.lexical_weight))
        return real_search(self, question, k, conversation_id)

    from continuator_memory.v2.memory_retrieval import MemoryRetriever

    real_search = MemoryRetriever.search
    monkeypatch.setattr(MemoryRetriever, "search", tracking_search)

    result = workspace.ask(QUESTION, retrieval_mode=MODE_LEXICAL_MULTIQUERY_RRF, generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["diagnostics"]["mode"] == MODE_LEXICAL_MULTIQUERY_RRF
    assert evidence["diagnostics"]["lexicalQueryCount"] == 1
    assert evidence["diagnostics"]["semanticQueryCount"] == 3
    # One lexical search on the original question; semantic searches for each planned query.
    lexical_calls = [q for q, weight in calls if weight and weight > 0]
    semantic_calls = [q for q, weight in calls if not weight]
    assert lexical_calls == [QUESTION]
    assert semantic_calls == list(plan.queries)


def test_retrieve_lexical_semantic_handles_empty_channels(monkeypatch):
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=2)

    empty = MagicMock()
    empty.results = ()
    empty.question = QUESTION
    empty.intent = MagicMock()

    class EmptyRetriever:
        def __init__(self, *args, **kwargs):
            self.config = MagicMock(lexical_weight=0.0)

        def search(self, *args, **kwargs):
            return empty

        def __len__(self):
            return 0

    monkeypatch.setattr(
        "continuator_memory.webapi.hybrid_rrf.build_memory_retriever",
        lambda *args, **kwargs: EmptyRetriever(),
    )
    evidence, diagnostics = retrieve_lexical_semantic_rrf_evidence(
        QUESTION,
        workspace.conversation,
        workspace.claims,
        workspace.relations,
        workspace.embedder(),
        k=2,
        budget_chars=workspace.budget_chars,
        depth=5,
    )
    assert diagnostics["mode"] == MODE_LEXICAL_SEMANTIC_RRF
    assert diagnostics["rankedMessageIds"] == []
    assert evidence.matches == ()
