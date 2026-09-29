import json

import pytest

from continuator_memory.cli import main
from continuator_memory.evaluation.dataset import parse_dataset
from continuator_memory.evaluation.harness import run_evaluation
from continuator_memory.evaluation.report import render_markdown, write_results
from continuator_memory.generation import GenerationError
from continuator_memory.models import RetrievedChunk
from continuator_memory.pipeline import Answer, Pipeline


def _chunk(conv, mids, rank=0):
    return RetrievedChunk(rank, f"{conv}:{'-'.join(mids)}", conv, tuple(mids), 0.0, " ".join(mids))


DATASET = parse_dataset(
    {
        "dataset_id": "harness-test",
        "conversations": [
            {
                "conversation_id": "c1",
                "messages": [
                    {"id": f"m{i}", "role": "user" if i % 2 else "assistant", "content": f"t{i}"}
                    for i in range(1, 7)
                ],
            },
            {"conversation_id": "c2", "messages": [{"id": "m1", "role": "user", "content": "x"}]},
        ],
        "questions": [
            {"id": "q1", "category": "basic_fact", "conversation_id": "c1", "question": "Q1",
             "expected_answer": "Rex", "expected_source_message_ids": ["m1"]},
            {"id": "q2", "category": "basic_fact", "conversation_id": "c1", "question": "Q2",
             "expected_answer": "Bo", "stale_answers": ["Max"],
             "expected_source_message_ids": ["m3"]},
            {"id": "q3", "category": "multi_hop", "conversation_id": "c1", "question": "Q3",
             "scope": "global", "expected_answer": "Pixel and Byte",
             "match_all": ["Pixel", "Byte"], "expected_source_message_ids": ["m1", "m5"]},
            {"id": "q4", "category": "abstention", "conversation_id": "c1", "question": "Q4",
             "answerable": False, "expected_answer": None, "expected_source_message_ids": []},
            {"id": "q5", "category": "basic_fact", "conversation_id": "c1", "question": "Q5",
             "expected_answer": "Oslo", "expected_source_message_ids": ["m6"]},
        ],
    }
)

RANKINGS = {
    "Q1": [_chunk("c1", ["m1", "m2"])],
    "Q2": [_chunk("c1", ["m1", "m2"]), _chunk("c1", ["m5", "m6"]), _chunk("c1", ["m3", "m4"])],
    "Q3": [_chunk("c1", ["m5", "m6"]), _chunk("c2", ["m1"]), _chunk("c1", ["m1", "m2"])],
    "Q4": [_chunk("c1", ["m1", "m2"])],
    "Q5": [_chunk("c1", ["m1", "m2"]), _chunk("c1", ["m3", "m4"])],
}
ANSWERS = {
    "Q1": "The dog is Rex.",
    "Q2": "It is Max.",
    "Q3": "Pixel and Byte.",
    "Q4": "I don't know.",
    "Q5": "I don't know.",
}


class FakePipeline:
    def __init__(self, fail_on=()):
        self.ingested, self.retrieve_calls, self.ask_calls = [], [], []
        self.fail_on = set(fail_on)

    def ingest(self, conversation):
        self.ingested.append(conversation.conversation_id)

    def retrieve(self, question, k, conversation_id=None):
        self.retrieve_calls.append((question, k, conversation_id))
        return RANKINGS[question][:k]

    def ask(self, question, k, conversation_id=None):
        self.ask_calls.append((question, k, conversation_id))
        if question in self.fail_on:
            raise GenerationError("boom")
        return Answer(question, ANSWERS[question], "fake", "prompt", RANKINGS[question][:k])


@pytest.fixture
def results():
    return run_evaluation(DATASET, FakePipeline(), generation_top_k=5)


def test_pipeline_is_called_with_phase1_arguments():
    pipeline = FakePipeline()
    run_evaluation(DATASET, pipeline, generation_top_k=5)
    assert pipeline.ingested == ["c1", "c2"]
    assert pipeline.retrieve_calls[0] == ("Q1", 10, "c1")
    assert pipeline.retrieve_calls[2] == ("Q3", 10, None)
    assert pipeline.ask_calls == [("Q1", 5, "c1"), ("Q2", 5, "c1"), ("Q3", 5, None),
                                  ("Q4", 5, "c1"), ("Q5", 5, "c1")]


def test_per_question_retrieval(results):
    by_id = {q["question_id"]: q for q in results["questions"]}
    r = {k: v["retrieval"] for k, v in by_id.items()}
    assert [r[q]["first_expected_rank"] for q in ("q1", "q2", "q3", "q5")] == [1, 3, 1, None]
    assert r["q2"]["reciprocal_rank"] == pytest.approx(1 / 3)
    assert r["q3"]["recall_at"] == {"1": 0.5, "3": 1.0, "5": 1.0, "10": 1.0}
    assert r["q3"]["hit_at"]["1"] is True
    assert r["q5"]["reciprocal_rank"] == 0.0
    assert r["q4"]["first_expected_rank"] is None and r["q4"]["recall_at"] == {}
    assert r["q2"]["expected_source_message_ids"] == ["m3"]
    assert [c["chunk_id"] for c in r["q2"]["retrieved"]] == ["c1:m1-m2", "c1:m5-m6", "c1:m3-m4"]


def test_per_question_answers(results):
    a = {q["question_id"]: q["answer"] for q in results["questions"]}
    assert [a[q]["correct"] for q in ("q1", "q2", "q3", "q4", "q5")] == [
        True, False, True, True, False
    ]
    assert a["q2"]["stale_mentioned"] is True
    assert a["q1"]["stale_mentioned"] is None
    assert a["q5"]["abstained"] is True
    assert a["q1"]["generated_answer"] == "The dog is Rex."
    assert a["q1"]["expected_answer"] == "Rex"
    assert all(q["latency_ms"] >= 0 for q in results["questions"])


def test_aggregate_retrieval(results):
    o = results["aggregate"]["retrieval"]["overall"]
    assert o["n"] == 4
    assert o["mrr"] == pytest.approx((1 + 1 / 3 + 1 + 0) / 4)
    assert o["recall_at"] == pytest.approx({"1": 0.375, "3": 0.75, "5": 0.75, "10": 0.75})
    assert o["hit_at"] == pytest.approx({"1": 0.5, "3": 0.75, "5": 0.75, "10": 0.75})
    assert results["aggregate"]["retrieval"]["by_category"]["abstention"]["n"] == 0


def test_aggregate_answers(results):
    o = results["aggregate"]["answer"]["overall"]
    assert (o["n"], o["n_correct"], o["n_errors"]) == (5, 3, 0)
    assert o["accuracy"] == pytest.approx(0.6)
    assert o["answerable"] == {"n": 4, "accuracy": 0.5}
    assert o["abstention"] == {"n": 1, "accuracy": 1.0}
    assert o["false_abstention_rate"] == pytest.approx(0.25)
    assert o["stale_mention_rate"] == 1.0
    assert o["answerable_by_retrieval"] == {
        "k": 5,
        "retrieved": {"n": 3, "accuracy": pytest.approx(2 / 3)},
        "not_retrieved": {"n": 1, "accuracy": 0.0},
    }
    assert set(results["aggregate"]["answer"]["by_category"]) == {
        "abstention", "basic_fact", "multi_hop"
    }


def test_generation_errors_are_recorded_not_raised():
    res = run_evaluation(DATASET, FakePipeline(fail_on={"Q1"}), generation_top_k=5)
    q1 = res["questions"][0]["answer"]
    assert q1["correct"] is False and q1["error"] == "boom"
    assert res["aggregate"]["answer"]["overall"]["n_errors"] == 1


def test_retrieval_only_and_limit():
    pipeline = FakePipeline()
    res = run_evaluation(DATASET, pipeline, generation_top_k=5, retrieval_only=True, limit=2)
    assert pipeline.ask_calls == []
    assert [q["question_id"] for q in res["questions"]] == ["q1", "q2"]
    assert res["questions"][0]["answer"]["correct"] is None
    assert res["aggregate"]["answer"]["overall"] == {"n": 0}
    assert res["run"]["dataset"]["n_questions_evaluated"] == 2
    assert "Generation skipped" in render_markdown(res)


def test_write_results(tmp_path, results):
    json_path, md_path = write_results(results, tmp_path / "out")
    assert json_path.name == f"{results['run']['run_id']}.json"
    assert json.loads(json_path.read_text(encoding="utf-8")) == results
    md = md_path.read_text(encoding="utf-8")
    for heading in ("## Retrieval", "## Answers", "## Per question", "## Failures"):
        assert heading in md
    assert "### q2 (basic_fact)" in md and "### q5 (basic_fact)" in md
    assert "### q1 " not in md


def test_smoke_with_real_pipeline(store, embedder):
    class Scripted:
        model_name = "scripted"

        def generate(self, system, prompt):
            return "Rex" if "Dog" in prompt else "I don't know."

    dataset = parse_dataset(
        {
            "dataset_id": "smoke",
            "conversations": [
                {
                    "conversation_id": "c1",
                    "messages": [
                        {"id": "m1", "role": "user", "content": "My dog is called Rex."},
                        {"id": "m2", "role": "assistant", "content": "Rex is a nice name."},
                        {"id": "m3", "role": "user", "content": "I live in Oslo."},
                    ],
                }
            ],
            "questions": [
                {"id": "q1", "category": "basic_fact", "conversation_id": "c1",
                 "question": "Dog name?", "expected_answer": "Rex",
                 "expected_source_message_ids": ["m1"]},
                {"id": "q2", "category": "abstention", "conversation_id": "c1",
                 "question": "Cat name?", "answerable": False, "expected_answer": None,
                 "expected_source_message_ids": []},
            ],
        }
    )
    pipeline = Pipeline(store, embedder, Scripted(), window_size=2, overlap=1)
    res = run_evaluation(dataset, pipeline, generation_top_k=1)
    assert res["aggregate"]["answer"]["overall"]["accuracy"] == 1.0
    assert res["aggregate"]["retrieval"]["overall"]["n"] == 1
    assert len(res["questions"][0]["answer"]["context_chunk_ids"]) == 1


def test_cli_evaluate_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("dataset", "--output-dir", "--top-k", "--retrieval-only", "--limit"):
        assert flag in out


def test_cli_evaluate_bad_dataset(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["evaluate", str(bad), "-o", str(tmp_path)]) == 1
    assert "dataset_id" in capsys.readouterr().err
