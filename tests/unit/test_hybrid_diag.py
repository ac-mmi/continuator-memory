"""Evaluation-only helpers for Baseline vs Hybrid. No MiniLM, LoCoMo DB, or Ollama."""

from __future__ import annotations

import inspect
import os

import pytest

from continuator_memory.v2.lexical import DEFAULT_LEXICAL_WEIGHT, LEXICAL_WEIGHT_ENV
from continuator_memory.webapi import hybrid_diag
from continuator_memory.webapi.hybrid_diag import (
    MODES,
    format_gold_ranks,
    gold_ranks,
    qa_coverage,
    resolve_lexical_weight,
    select_eval_rows,
)
from continuator_memory.webapi.iterative_diag import _sample_rows, _score


def test_modes_are_baseline_and_hybrid_only():
    assert MODES == ("baseline", "hybrid")


def test_reuses_iterative_diag_scoring_and_optional_subset():
    assert hybrid_diag._score is _score
    assert hybrid_diag._sample_rows is _sample_rows


def test_sample_rows_keeps_caroline_question_first():
    qa = [
        {"question": "When did Alex arrive?", "evidence": ["m-1"]},
        {"question": "What fields would Caroline be likely to pursue in her education?",
         "evidence": ["D1:9", "D1:11"]},
        *[{"question": f"Q{i}", "evidence": [f"m-{i}"]} for i in range(12)],
    ]
    sample = _sample_rows(qa)
    assert sample[0]["question"].startswith("What fields would Caroline")
    assert sample[0]["evidence"] == ["D1:9", "D1:11"]
    assert 9 <= len(sample) <= 10


def test_qa_coverage_excludes_empty_and_unmatched_gold_only():
    qa = [
        {"question": "later scorable", "evidence": ["m-b"], "category": 2},
        {"question": "empty", "evidence": [], "category": 3},
        {"question": "unmatched", "evidence": ["D8:6; D9:17"], "category": 1},
        {"question": "first scorable", "evidence": ["m-a"], "category": 1},
        {"question": "partial still scorable", "evidence": ["m-a", "missing"], "category": 4},
    ]
    coverage = qa_coverage(qa, {"m-a", "m-b"})
    assert coverage["total"] == 5
    assert coverage["empty_evidence"] == ["empty"]
    assert coverage["unmatched_gold"] == ["unmatched"]
    assert [row["question"] for row in coverage["scorable"]] == [
        "later scorable",
        "first scorable",
        "partial still scorable",
    ]
    assert coverage["categories"] == {1: 1, 2: 1, 4: 1}


def test_all_sample_is_file_order_not_hybrid_success():
    qa = [
        {"question": "Q2", "evidence": ["m-2"], "category": 2},
        {"question": "empty", "evidence": []},
        {"question": "Q1", "evidence": ["m-1"], "category": 1},
    ]
    coverage = qa_coverage(qa, {"m-1", "m-2"})
    selected = select_eval_rows(qa, coverage, "all")
    assert [row["question"] for row in selected] == ["Q2", "Q1"]
    iterative = select_eval_rows(qa, coverage, "iterative")
    assert iterative is not selected
    with pytest.raises(ValueError, match="sample"):
        select_eval_rows(qa, coverage, "winners")


def test_gold_ranks_are_positions_in_selected_only():
    selected = ["m-a", "m-gold", "m-b"]
    assert gold_ranks(selected, ["m-gold", "m-missing"]) == {"m-gold": 2, "m-missing": None}
    assert format_gold_ranks({"D1:9": 6, "D1:11": None}) == "D1:9=6 D1:11=absent"


def test_score_does_not_need_gold_in_retrieval_input():
    scored = _score(["m-noise", "m-gold"], ["m-gold"], "conv")
    assert scored["firstExpectedRank"] == 2
    assert scored["recallAt"]["5"] == 1.0
    assert scored["hitAt"]["5"] == 1.0
    assert scored["reciprocalRank"] == pytest.approx(0.5)


def test_resolve_lexical_weight_uses_env_and_cli(monkeypatch):
    monkeypatch.delenv(LEXICAL_WEIGHT_ENV, raising=False)
    assert resolve_lexical_weight(None) == DEFAULT_LEXICAL_WEIGHT
    monkeypatch.setenv(LEXICAL_WEIGHT_ENV, "0.4")
    assert resolve_lexical_weight(None) == 0.4
    assert resolve_lexical_weight(0.3) == 0.3
    assert os.environ[LEXICAL_WEIGHT_ENV] == "0.3"
    with pytest.raises(ValueError, match="lexical_weight"):
        resolve_lexical_weight(-0.1)


def test_search_is_generate_false_and_gold_stays_out_of_retrieval():
    source = inspect.getsource(hybrid_diag.search_mode)
    assert "retriever.search(question, k, conversation_id)" in source
    assert "evidence" not in source
    run_src = inspect.getsource(hybrid_diag.run_modes)
    assert "weight=0.0" in run_src
    assert "hybrid_weight" in run_src
    main_src = inspect.getsource(hybrid_diag.main)
    ask_at = main_src.find("run_modes(retriever, row[\"question\"], cid")
    assert ask_at != -1
    assert 'row["evidence"]' in main_src[ask_at:]
