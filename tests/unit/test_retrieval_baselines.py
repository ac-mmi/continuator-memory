"""Scoring and the raw-chunk control for the message-level retrieval baselines."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from tests.helpers import HashingEmbedder

_spec = importlib.util.spec_from_file_location("run_retrieval_baselines", Path(__file__).resolve().parents[2] / "scripts/run_retrieval_baselines.py")
baselines = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baselines)


def test_any_message_satisfies_a_group_and_coverage_needs_every_group():
    row = baselines.score_question([[1], [2], [3]], [[9, 2], [8]])
    assert row["recall@5"] == 0.5
    assert row["coverage@5"] is False
    row = baselines.score_question([[1], [2], [8]], [[9, 2], [8]])
    assert (row["recall@5"], row["coverage@5"]) == (1.0, True)


def test_results_beyond_k_do_not_count():
    results = [[i] for i in range(100, 106)] + [[1]]
    row = baselines.score_question(results, [[1]])
    assert (row["recall@5"], row["recall@10"], row["first_rank"], row["rr"]) == (0.0, 1.0, 7, 1 / 7)


def test_no_relevant_result_scores_zero():
    row = baselines.score_question([[1], [2]], [[9]])
    assert (row["first_rank"], row["rr"], row["recall@10"]) == (None, 0.0, 0.0)


def test_message_budget_keeps_chunks_whole_and_never_exceeds_it():
    chunks = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    row = baselines.score_question(chunks, [[5]])
    assert row["budget4_recall"] == 0.0
    assert row["budget8_recall"] == 1.0
    single = baselines.score_question([[0], [1], [2], [3], [5]], [[5]])
    assert single["budget4_recall"] == 0.0
    assert single["budget8_recall"] == 1.0


def test_repeated_messages_use_budget_once():
    row = baselines.score_question([[1], [1], [1], [2], [3], [4], [5]], [[5]])
    assert row["budget4_recall"] == 0.0
    assert row["budget8_recall"] == 1.0


def test_too_few_results_for_a_budget_is_reported():
    assert baselines.score_question([[1]], [[1]])["budget8_short"] is True
    assert baselines.score_question([[i] for i in range(10)], [[1]])["budget8_short"] is False


def test_aggregate_means_and_counts():
    rows = {"1": baselines.score_question([[1]], [[1]]), "2": baselines.score_question([[1]], [[9]])}
    out = baselines.aggregate(rows)
    assert (out["recall@5"], out["coverage@5"], out["mrr@10"]) == (0.5, 1, 0.5)


def conversation():
    texts = ["Ana bakes rye bread.", "Bram keeps bees.", "Ana sails on Friday.", "Bram plays cello.", "Ana studies Welsh.",
             "Bram repairs clocks.", "Ana grows tomatoes.", "Bram paints fences.", "Ana rides horses."]
    return Conversation("c", tuple(Message(f"c:m{i:04d}", "c", i, "user", t) for i, t in enumerate(texts)))


def test_control_uses_non_overlapping_chunks_matching_windows(tmp_path):
    with Store(str(tmp_path / "x.db")) as store:
        retriever, n = baselines.build_control(conversation(), store, HashingEmbedder())
        assert n == 3
        chunks = store.get_chunks([f"c:c{i:04d}" for i in range(3)])
        seen = [m for c in chunks for m in c.source_message_ids]
        assert len(seen) == len(set(seen)) == 9
        assert retriever.retrieve("bees", 3, "c")


def test_control_rankings_cover_every_chunk_and_differ_only_by_scoring(tmp_path):
    with Store(str(tmp_path / "x.db")) as store:
        retriever, n = baselines.build_control(conversation(), store, HashingEmbedder())
        out = baselines.control_results("Who rides horses?", retriever, "c", n, 0.25)
    assert {r["chunk_id"] for r in out["control_cosine"]} == {r["chunk_id"] for r in out["control_hybrid"]}
    assert len(out["control_cosine"]) == n
    assert out["control_cosine"][0]["msgs"] == [8]
    assert out["control_hybrid"][0]["score"] >= out["control_hybrid"][-1]["score"]


def test_zero_lexical_weight_matches_cosine_order(tmp_path):
    with Store(str(tmp_path / "x.db")) as store:
        retriever, n = baselines.build_control(conversation(), store, HashingEmbedder())
        out = baselines.control_results("bees and clocks", retriever, "c", n, 0.0)
    assert [r["chunk_id"] for r in out["control_cosine"]] == [r["chunk_id"] for r in out["control_hybrid"]]


def test_script_has_no_model_generation_path():
    text = Path(baselines.__file__).read_text(encoding="utf-8")
    for word in ("ollama", "OllamaGenerator", "generate_answer", "complete_json"):
        assert word.lower() not in text.lower().replace("no ollama call", "")
