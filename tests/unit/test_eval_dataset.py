import copy
import json

import pytest

from continuator_memory.chunking import chunk_conversation
from continuator_memory.config import DEFAULT_OVERLAP, DEFAULT_WINDOW_SIZE
from continuator_memory.evaluation.dataset import DatasetError, load_dataset, parse_dataset
from continuator_memory.evaluation.metrics import match_answer
from tests.helpers import PHASE2_BENCHMARK

REQUIRED_CATEGORIES = {
    "basic_fact",
    "topic_drift",
    "topic_return",
    "contradiction",
    "temporal_update",
    "multi_hop",
    "multi_entity",
    "distractor",
    "abstention",
}

BASE = {
    "dataset_id": "tiny",
    "conversations": [
        {
            "conversation_id": "c1",
            "messages": [
                {"id": "m1", "role": "user", "content": "My dog is Rex."},
                {"id": "m2", "role": "assistant", "content": "Nice."},
            ],
        }
    ],
    "questions": [
        {
            "id": "q1",
            "category": "basic_fact",
            "conversation_id": "c1",
            "question": "Dog name?",
            "expected_answer": "Rex",
            "expected_source_message_ids": ["m1"],
        },
        {
            "id": "q2",
            "category": "abstention",
            "conversation_id": "c1",
            "answerable": False,
            "scope": "global",
            "question": "Cat name?",
            "expected_answer": None,
            "expected_source_message_ids": [],
        },
    ],
}


def _with(mutator):
    data = copy.deepcopy(BASE)
    mutator(data)
    return data


def test_parse_defaults():
    ds = parse_dataset(BASE)
    q1, q2 = ds.questions
    assert q1.match_any == ("Rex",) and q1.match_all == ()
    assert q1.scope == "conversation" and q1.retrieval_conversation_id == "c1"
    assert q2.answerable is False and q2.match_any == ()
    assert q2.retrieval_conversation_id is None


def test_match_all_without_match_any_does_not_default_to_expected_answer():
    def m(d):
        d["questions"][0].update(expected_answer="Rex the dog", match_all=["Rex"])

    q = parse_dataset(_with(m)).questions[0]
    assert q.match_any == () and q.match_all == ("Rex",)


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda d: d["questions"][0].update(conversation_id="nope"), "unknown conversation"),
        (lambda d: d["questions"][0].update(expected_source_message_ids=["m9"]), "unknown source"),
        (lambda d: d["questions"][0].update(expected_source_message_ids=[]), "source"),
        (lambda d: d["questions"][0].update(expected_answer=""), "expected_answer"),
        (lambda d: d["questions"][0].update(scope="everywhere"), "scope"),
        (lambda d: d["questions"][1].update(expected_answer="x"), "unanswerable"),
        (lambda d: d["questions"][1].update(expected_source_message_ids=["m1"]), "unanswerable"),
        (lambda d: d["questions"][1].update(id="q1"), "duplicate question"),
        (lambda d: d["conversations"].append(copy.deepcopy(d["conversations"][0])), "duplicate"),
        (lambda d: d["questions"][0].update(match_any="Rex"), "match_any"),
        (lambda d: d.pop("dataset_id"), "dataset_id"),
        (lambda d: d["questions"][0].pop("question"), "question"),
    ],
)
def test_parse_rejects_invalid(mutator, match):
    with pytest.raises(DatasetError, match=match):
        parse_dataset(_with(mutator))


def test_load_records_path_and_hash(tmp_path):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(BASE), encoding="utf-8")
    a, b = load_dataset(path), load_dataset(path)
    assert a.path == str(path)
    assert a.sha256 == b.sha256 and len(a.sha256) == 64


# ---- structural checks on the shipped Phase 2 benchmark (no model, no LLM) ----


@pytest.fixture(scope="module")
def benchmark():
    return load_dataset(PHASE2_BENCHMARK)


def test_benchmark_size_and_categories(benchmark):
    assert len(benchmark.questions) >= 50
    assert {q.category for q in benchmark.questions} == REQUIRED_CATEGORIES


def test_benchmark_expected_answers_satisfy_their_own_matchers(benchmark):
    for q in benchmark.questions:
        if q.answerable:
            assert match_answer(q.expected_answer, q.match_any, q.match_all), q.question_id


def test_benchmark_abstention_and_distractor_setup(benchmark):
    for q in benchmark.questions:
        if q.category == "abstention":
            assert not q.answerable and not q.expected_source_message_ids
        else:
            assert q.answerable
        if q.category == "distractor":
            assert q.scope == "global"


def test_benchmark_multi_hop_sources_are_in_different_chunks(benchmark):
    conversations = {c.conversation_id: c for c in benchmark.conversations}
    for q in benchmark.questions:
        if q.category != "multi_hop":
            continue
        assert len(q.expected_source_message_ids) >= 2, q.question_id
        chunks = chunk_conversation(
            conversations[q.conversation_id], DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP
        )
        for chunk in chunks:
            assert not set(q.expected_source_message_ids) <= set(chunk.source_message_ids), (
                q.question_id
            )
