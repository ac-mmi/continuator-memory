"""Neutral-shuffle control: deterministic shuffling, eligibility, prompt equivalence, resume logic."""

from __future__ import annotations

import json
from types import SimpleNamespace

from benchmarks.experiments.graph_rag.neutral_shuffle import (
    completed,
    eligible,
    prompt_blocks,
    prompt_frame,
    same_content_different_order,
    shuffle_order,
)
from continuator_memory.models import Conversation, Message
from continuator_memory.webapi.answering import build_answer_context

ORDER = [f"D1:{i}" for i in range(1, 26)]


def test_shuffle_is_deterministic_a_permutation_and_never_the_identity():
    a = shuffle_order(ORDER, "conv-1", "Q-001")
    assert a == shuffle_order(ORDER, "conv-1", "Q-001")
    assert sorted(a) == sorted(ORDER) and a != ORDER
    assert a != shuffle_order(ORDER, "conv-1", "Q-002")
    assert shuffle_order(["x", "y"], "s", "q") == ["y", "x"]
    assert shuffle_order(["only"], "s", "q") == ["only"]


def _rec(gold, prompt_ids):
    return {"evaluationOnly": {"goldEvidence": gold}, "promptMessageIds": prompt_ids}


def test_eligibility_by_count_and_by_set():
    records = {
        ("c", "Q-1", "hybrid"): _rec(["a"], ["a", "x"]), ("c", "Q-1", "graph_hybrid"): _rec(["a"], ["a"]),
        ("c", "Q-2", "hybrid"): _rec(["a", "b"], ["a"]), ("c", "Q-2", "graph_hybrid"): _rec(["a", "b"], ["b"]),
        ("c", "Q-3", "hybrid"): _rec(["a"], []), ("c", "Q-3", "graph_hybrid"): _rec(["a"], ["a"]),
    }
    assert eligible(records, "count") == [("c", "Q-1"), ("c", "Q-2")]
    assert eligible(records, "set") == [("c", "Q-1")]


def test_shuffled_prompt_keeps_content_header_and_question():
    msgs = tuple(Message(f"m{i}", "c", i, "user", f"Speaker: turn number {i} about topic {i}", "2023-05-0%dT10:00:00" % (i % 9 + 1))
                 for i in range(6))
    conv = Conversation("c", msgs)
    ids = [m.message_id for m in msgs]
    matches = [SimpleNamespace(memory_id=f"mem:{m}", source_message_ids=(m,), statement="", rank=i, score=1.0,
                               similarity=1.0) for i, m in enumerate(ids, 1)]

    def render(order):
        ev = SimpleNamespace(selection=SimpleNamespace(selected=tuple(order), dropped=()), matches=matches)
        return build_answer_context("What topic?", ev, conv, seed_ids=[], expanded_ids=[]).prompt

    original, shuffled = render(ids), render(shuffle_order(ids, "c", "Q-1"))
    assert original != shuffled
    assert same_content_different_order(original, shuffled)
    assert [b.split("]")[0][1:] for b in prompt_blocks(shuffled)] == shuffle_order(ids, "c", "Q-1")
    assert prompt_frame(shuffled)[1].endswith("Question: What topic?\nAnswer:")
    altered = shuffled.replace("turn number 3", "turn number 33")
    assert not same_content_different_order(original, altered)


def test_resume_counts_attempts_and_never_repeats_completed_questions(tmp_path):
    path = tmp_path / "answers.jsonl"
    rows = [{"sampleId": "c", "questionId": "Q-1", "status": "error"},
            {"sampleId": "c", "questionId": "Q-1", "status": "ok"},
            {"sampleId": "c", "questionId": "Q-2", "status": "exception"}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    done, attempts = completed(path)
    assert done == {("c", "Q-1")}
    assert attempts == {("c", "Q-1"): 2, ("c", "Q-2"): 1}
    assert completed(tmp_path / "missing.jsonl") == (set(), {})
