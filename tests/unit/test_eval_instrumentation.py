"""Evaluation instrumentation: conditions, structured records, abstention scoring, gold isolation."""

from __future__ import annotations

import json

import pytest

from continuator_memory.evaluation.answer_scoring import score_answer, token_f1
from continuator_memory.generation import OllamaGenerator
from continuator_memory.webapi.locomo_report import (
    LocomoQuestion,
    aggregate,
    build_memory_retriever,
    classify_evidence,
    context_window_risk,
    evaluate_question,
    load_known_issues,
    question_record,
    render_report,
    retrieval_config,
    write_records,
)
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder

GOLD_ANSWER = "ZEBRA-GOLD-ANSWER-7"
TRAP = "the red lighthouse"


class RecordingGenerator:
    """Stands in for Ollama: records what the generator is shown and returns a fixed raw answer."""

    model_name = "fake"

    def __init__(self, answer: str, prompt_tokens: int = 1000):
        self.answer = answer
        self.prompt_tokens = prompt_tokens
        self.calls: list[tuple[str, str]] = []
        self.last_usage = None

    def generate(self, system, prompt, temperature=None, num_predict=None):
        self.calls.append((system, prompt))
        self.last_usage = {"num_ctx": 4096, "num_predict": num_predict, "prompt_eval_count": self.prompt_tokens}
        return self.answer


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTINUATOR_LEXICAL_WEIGHT", "0.25")
    monkeypatch.setenv("CONTINUATOR_CONTEXT_WEIGHT", "0.5")
    monkeypatch.setenv("CONTINUATOR_OLLAMA_MODEL", "fake")
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=5)
    ws.import_text("\n\n".join(f"You: I visited museum number {i} with my sister" for i in range(30)),
                   filename="c.txt", extract="messages", db_path=tmp_path / "w.db")
    ws._eval_retriever = build_memory_retriever(
        ws.conversation, ws.claims, ws.relations, ws.embedder(), retrieval_config("hybrid"))
    return ws


def _item(ws, category=1, adversarial=None, expects_abstention=None):
    gold = [ws.conversation.messages[7].message_id, ws.conversation.messages[2].message_id]
    ids = {m.message_id for m in ws.conversation.messages}
    return LocomoQuestion(0, "Which museum did I visit with my sister?", GOLD_ANSWER, category, gold,
                          classify_evidence(gold, ids), "fixture", adversarial, expects_abstention)


def _run(ws, item, condition, answer, oracle=None):
    gen = RecordingGenerator(answer)
    ws._generator = gen
    row = evaluate_question(item, ws, mode="hybrid", top_k=5, depth=50, generate=True,
                            condition=condition, oracle_message_ids=oracle)
    return row, gen


def test_score_answer_scores_abstention_instead_of_skipping():
    s = score_answer("That is not mentioned in the conversation.", "", expects_abstention=True,
                     adversarial_answer=TRAP)
    assert s["abstained"] and s["abstentionCorrect"] is True
    assert s["adversarialAnswerMatched"] is False
    assert s["tokenF1"] is None and s["semanticCorrect"] is None
    trapped = score_answer("He visited the red lighthouse.", "", expects_abstention=True, adversarial_answer=TRAP)
    assert trapped["abstentionCorrect"] is False and trapped["adversarialAnswerMatched"] is True
    normal = score_answer("Museum 7", "museum 7", expects_abstention=False)
    assert normal["abstentionCorrect"] is None and normal["exactMatch"] is True
    assert token_f1("a b", "a c") == pytest.approx(0.5)


def test_abstention_wording_from_longmemeval_pilot_is_detected():
    # No-memory answer to gpt4_372c3eed_abs, missed by metrics.ABSTENTION_MARKERS alone.
    text = ("There is not enough information provided in the given excerpts to determine the total "
            "number of years spent in formal education from high school to the completion of a Master's degree.")
    assert score_answer(text, "", expects_abstention=True)["abstentionCorrect"] is True
    assert score_answer("You have 20 playlists.", "20", expects_abstention=False)["abstained"] is False


def test_category5_expects_abstention_by_default(workspace):
    assert _item(workspace, category=5).abstention_expected is True
    assert _item(workspace, category=1).abstention_expected is False
    assert _item(workspace, category=1, expects_abstention=True).abstention_expected is True


def test_normal_condition_preserves_raw_and_final_answer_and_hides_gold(workspace):
    item = _item(workspace)
    row, gen = _run(workspace, item, "normal", "Museum 7 [not-a-real-id].")
    assert row["error"] is None
    assert row["rawAnswer"] == "Museum 7 [not-a-real-id]."
    assert row["generatedAnswer"] == "Museum 7."
    assert row["promptMessageIds"] == row["rankedMessageIds"][:5]
    system, prompt = gen.calls[0]
    for secret in (GOLD_ANSWER, "Category", "condition", "oracle", "gold"):
        assert secret not in prompt and secret not in system
    assert row["answerScore"]["raw"]["tokenF1"] is not None


def test_oracle_condition_uses_exactly_the_given_messages_in_dialogue_order(workspace):
    item = _item(workspace)
    oracle = list(item.annotation.gold_ids)
    row, gen = _run(workspace, item, "oracle", "Museum 2 and 7.", oracle=oracle)
    assert row["error"] is None
    assert row["promptMessageIds"] == [workspace.conversation.messages[2].message_id,
                                       workspace.conversation.messages[7].message_id]
    assert row["oracleMissingMessageIds"] == []
    assert GOLD_ANSWER not in gen.calls[0][1]
    # The evaluation retriever's configuration is restored after the oracle search.
    assert workspace._eval_retriever.config.min_similarity == retrieval_config("hybrid").min_similarity


def test_oracle_without_ids_is_an_error_not_a_silent_normal_run(workspace):
    row, gen = _run(workspace, _item(workspace), "oracle", "x", oracle=[])
    assert row["error"] and "oracle" in row["error"]
    assert gen.calls == []


def test_no_memory_condition_sends_no_excerpts(workspace):
    row, gen = _run(workspace, _item(workspace, category=5), "no_memory", "I don't know.")
    assert row["promptMessageIds"] == [] and row["retrievedMemories"] == []
    assert "(no source excerpts were selected for the model)" in gen.calls[0][1]
    assert row["answerScore"]["final"]["abstentionCorrect"] is True


def test_question_record_is_structured_and_flags_prompt_inclusion(workspace, tmp_path):
    row, _ = _run(workspace, _item(workspace), "normal", "Museum 7.")
    run = {"dataset": "fixture.json", "sampleId": "fixture", "conversationId": "imported"}
    issue = {"sampleId": "fixture", "questionId": "Q-001", "type": "off_by_one_gold"}
    rec = question_record(row, run, provenance=lambda mid: {"turn": mid}, known_issue=issue)
    assert rec["condition"] == "normal" and rec["questionId"] == "Q-001"
    assert rec["rawAnswer"] == "Museum 7." and rec["finalAnswer"] == "Museum 7."
    in_prompt = [r for r in rec["retrieved"] if r["inPrompt"]]
    assert len(in_prompt) == 5 and in_prompt[0]["provenance"] == [{"turn": in_prompt[0]["sourceMessageIds"][0]}]
    assert rec["generation"]["usage"]["prompt_eval_count"] == 1000
    assert rec["promptBudget"]["contextTokens"] == 4096 and rec["promptBudget"]["outputTokens"] == 256
    assert rec["promptBudget"]["estimatedPromptTokens"] + 256 <= 4096
    assert rec["generation"]["contextWindowRisk"] is False
    assert rec["evaluationOnly"]["referenceAnswer"] == GOLD_ANSWER
    assert rec["evaluationOnly"]["knownAnnotationIssue"] == issue
    path = tmp_path / "r.jsonl"
    write_records(path, [rec, rec])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["questionId"] == "Q-001"


def test_report_renders_raw_answer_and_answer_summary(workspace):
    row, _ = _run(workspace, _item(workspace, category=5, adversarial=TRAP), "normal", "Not mentioned [x].")
    summary = aggregate([row])
    assert summary["answers"]["generated"] == 1
    assert summary["answers"]["rawDiffersFromFinal"] == 1
    assert summary["answers"]["abstentionExpected"] == 1 and summary["answers"]["abstentionCorrect"] == 1
    text = render_report({"questions": [row], "summary": summary, "condition": "normal"})
    assert "### Raw answer" in text and "Not mentioned [x]." in text
    assert "abstentionCorrect=True" in text


def test_context_window_risk():
    assert context_window_risk(None) is None
    assert context_window_risk({"num_ctx": 4096, "num_predict": 256, "prompt_eval_count": 1000}) is False
    assert context_window_risk({"num_ctx": 4096, "num_predict": 256, "prompt_eval_count": 3900}) is True


def test_known_issues_manifest_loads_and_is_keyed_by_sample_and_question():
    issues = load_known_issues("benchmarks/annotations/locomo_known_issues.json")
    assert issues[("conv-30", "Q-045")]["type"] == "off_by_one_gold"
    assert all(issue["status"] in {"confirmed", "suspected"} for issue in issues.values())


def test_ollama_generator_records_usage(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"message": {"content": " hi "}, "prompt_eval_count": 12,
                               "eval_count": 3, "done_reason": "stop"}).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: Response())
    gen = OllamaGenerator("m", "http://x", num_ctx=2048)
    assert gen.last_usage is None
    assert gen.generate("s", "p", num_predict=8) == "hi"
    assert gen.last_usage == {"num_ctx": 2048, "num_predict": 8, "prompt_eval_count": 12,
                              "eval_count": 3, "done_reason": "stop"}


def test_reported_context_overflow_is_never_reported_as_ok(workspace):
    gen = RecordingGenerator("Museum 7.", prompt_tokens=4000)  # 4000 + 256 output > num_ctx 4096
    workspace._generator = gen
    row = evaluate_question(_item(workspace), workspace, mode="hybrid", top_k=5, depth=50, generate=True)
    assert row["generation"]["status"] == "context_overflow"
    assert "num_ctx" in row["generation"]["error"]
    assert row["rawAnswer"] == "Museum 7."
    assert row["answerScore"] is None


def test_question_date_reaches_the_generator_prompt_and_locomo_prompts_are_unchanged(workspace):
    plain = _item(workspace)
    row_plain, gen_plain = _run(workspace, plain, "normal", "Museum 7 [x].")
    assert "Current date" not in gen_plain.calls[0][1]

    dated = _item(workspace)
    dated.question_date = "2023/05/30 (Tue) 23:40"
    row, gen = _run(workspace, dated, "normal", "Museum 7 [x].")
    prompt = gen.calls[0][1]
    assert "Current date: 30 May 2023\nQuestion: Which museum did I visit with my sister?" in prompt
    assert row["prompt"] == prompt
    assert prompt.replace("Current date: 30 May 2023\n", "") == gen_plain.calls[0][1]
