"""Rolling summary experiment, with a scripted model. These test the plumbing and the safeguards around the
summary. Whether a real summary keeps a possibility tentative can only be measured by reading real output."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from continuator_memory.v2 import structured_targets
from continuator_memory.v2.atomizer import AtomizationRequestError
from continuator_memory.v2.store import SemanticStore
from continuator_memory.v2.structured_rolling import (
    ROLLING_EXAMPLES,
    ROLLING_PROTOCOL_VERSION,
    ROLLING_SYSTEM_PROMPT,
    SUMMARY_FIELD,
    SUMMARY_INVALID,
    SUMMARY_KEPT,
    SUMMARY_MISSING,
    SUMMARY_TRIMMED,
    SUMMARY_WINDOW_FAILED,
    RollingExtractor,
    RollingSummaryStore,
    estimate_tokens,
    ingest_rolling,
    rolling_extractor,
    rolling_output_schema,
    trim_summary,
)
from continuator_memory.v2.structured_store import StructuredStore
from continuator_memory.v2.structured_targets import (
    TARGETS_PROTOCOL_VERSION,
    TARGETS_SYSTEM_PROMPT,
    TargetExtractor,
    ingest_targets,
    targets_output_schema,
    validate_targets,
)
from continuator_memory.v2.windowing import ContextWindow, build_windows

CID = "c1"
TS = "2024-05-10"


def conversation(n=6):
    return Conversation(CID, tuple(Message(f"m{i}", CID, i, "user", f"Ana: Fact number {i}.", TS) for i in range(n)))


class Scripted:
    """Returns a numbered summary, or whatever is scripted for that call."""

    model_name = "scripted"

    def __init__(self, summaries=None, raw=None, fail=(), usage=None):
        self.summaries, self.raw, self.fail, self.usage = summaries or {}, raw or {}, set(fail), usage or {}
        self.prompts, self.schemas, self.systems = [], [], []
        self.last_usage = None

    def complete_json(self, system, prompt, schema, examples=()):
        index = len(self.prompts)
        self.prompts.append(prompt)
        self.schemas.append(schema)
        self.systems.append(system)
        if index in self.fail:
            raise AtomizationRequestError(f"scripted failure {index}")
        self.last_usage = self.usage.get(index, {"prompt_eval_count": 10, "eval_count": 5, "done_reason": "stop"})
        if index in self.raw:
            return self.raw[index]
        body = {"entities": [], "targets": {k: {"outcome": "nothing_durable"} for k in schema["properties"]["targets"]["required"]}}
        summary = self.summaries.get(index, f"Summary number {index}.")
        if summary is not ...:
            body[SUMMARY_FIELD] = summary
        return json.dumps(body)


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "scratch.db"))
    yield s
    s.close()


def run(store, client, n=6, **kwargs):
    extractor = RollingExtractor(client, **kwargs)
    report = ingest_rolling(conversation(n), store, extractor, targets_per_window=2, context_before=2)
    return report, RollingSummaryStore(store.connection).get(CID)


def mem(statement, quote, **extra):
    return {"statement": statement, "subject": "Ana", "subject_status": "stated", "predicate": "does", "object": "it",
            "object_status": "stated", "category": "fact", "statement_type": "assertion", "time_kind": "current", "quote": quote, **extra}


def response(slot, summary="Next summary."):
    return json.dumps({"entities": [], "targets": {"T1": slot}, SUMMARY_FIELD: summary})


def test_window_0_works_without_a_summary(store):
    client = Scripted()
    report, rows = run(store, client)
    assert "(none yet" in client.prompts[0]
    assert rows[0]["summary_in"] == ""
    assert rows[0]["summary_out"] == "Summary number 0."
    assert report.failed_windows == 0


def test_window_1_receives_the_summary_from_window_0(store):
    client = Scripted()
    run(store, client)
    assert "Summary number 0." in client.prompts[1]


def test_window_2_receives_the_updated_summary_not_the_original(store):
    client = Scripted()
    _, rows = run(store, client)
    assert "Summary number 1." in client.prompts[2]
    assert "Summary number 0." not in client.prompts[2]
    assert rows[2]["summary_in"] == "Summary number 1."


def test_prompt_gives_the_model_previous_summary_and_raw_messages(store):
    client = Scripted()
    run(store, client)
    prompt = client.prompts[1]
    assert "Summary number 0." in prompt
    assert "Fact number 2." in prompt and "Fact number 3." in prompt
    assert "Fact number 0." in prompt  # the existing local context mechanism is still there
    assert "not only from the memories you extracted" in ROLLING_SYSTEM_PROMPT


def test_summary_is_not_built_from_accepted_records(store):
    raw = response({"outcome": "extracted", "memories": [mem("Ana stated the first fact.", "Fact number 0")]}, "Words the model wrote.")
    client = Scripted(raw={0: raw})
    _, rows = run(store, client, n=1)
    assert rows[0]["summary_out"] == "Words the model wrote."
    assert "Ana stated the first fact." not in rows[0]["summary_out"]


def test_memories_are_still_passed_forward_separately(store):
    raw = json.dumps({"entities": [], "targets": {"T1": {"outcome": "extracted", "memories": [mem("Ana stated the first fact.", "Fact number 0")]}, "T2": {"outcome": "nothing_durable"}}, SUMMARY_FIELD: "S."})
    client = Scripted(raw={0: raw})
    run(store, client)
    assert "- Ana stated the first fact." in client.prompts[1]


def test_summary_stays_within_its_limit(store):
    long = " ".join(f"Sentence number {i} is here." for i in range(400))
    report, rows = run(store, Scripted(summaries={0: long}), max_summary_tokens=50)
    assert estimate_tokens(rows[0]["summary_out"]) <= 50
    assert rows[0]["status"] == SUMMARY_TRIMMED
    assert rows[0]["model_summary"] == long
    assert report.issues["SUMMARY_TRIMMED:flagged"] == 1


def test_trim_keeps_whole_sentences_and_the_oversized_model_text():
    text = "First stays. Second stays. Third goes because it is well past the limit set here."
    assert trim_summary(text, 7) == "First stays. Second stays."
    assert trim_summary("Short.", 400) == "Short."
    assert estimate_tokens(trim_summary("x" * 1000, 10)) <= 10


@pytest.mark.parametrize("text", ["Ana might move to Porto; this is only a possibility.", "Ana might move to Porto if the job works out. Bram asked whether she will."])
def test_tentative_wording_is_carried_unchanged(store, text):
    client = Scripted(summaries={0: text})
    _, rows = run(store, client)
    assert rows[0]["summary_out"] == text
    assert text in client.prompts[1]


def test_instructions_and_example_keep_possibilities_tentative():
    assert "never \"moved\"" in ROLLING_SYSTEM_PROMPT
    assert "might leave in June (only a possibility)" in json.loads(ROLLING_EXAMPLES[0][1])[SUMMARY_FIELD]


def test_corrections_stay_distinguishable():
    text = "Ana now lives in Porto; she earlier said Lisbon."
    assert "now X; earlier said Y" in ROLLING_SYSTEM_PROMPT
    assert "she earlier said" in json.loads(ROLLING_EXAMPLES[0][1])[SUMMARY_FIELD]
    assert trim_summary(text, 400) == text


def test_correction_wording_reaches_the_next_window(store):
    text = "Ana now lives in Porto; she earlier said Lisbon."
    client = Scripted(summaries={0: text})
    run(store, client)
    assert text in client.prompts[1]


def test_summary_text_is_never_accepted_as_a_source_quote(store):
    summary = "Ana owns a lighthouse in Vigo."
    raw = json.dumps({"entities": [], "targets": {"T1": {"outcome": "extracted", "memories": [mem("Ana owns a lighthouse in Vigo.", "Ana owns a lighthouse in Vigo")]}, "T2": {"outcome": "nothing_durable"}}, SUMMARY_FIELD: summary})
    client = Scripted(summaries={0: summary}, raw={1: raw})
    run(store, client)
    sx = StructuredStore(store.connection)
    assert sx.get_records(CID, "memory") == []
    assert any(i["code"] == "QUOTE_NOT_IN_SOURCE" for i in sx.get_windows(CID)[1]["issues"])


def test_summary_cannot_be_cited_as_a_label_or_satisfy_a_name_check(store):
    summary = "Ana works with Zephyr Foundry."
    cited = mem("Ana works with Zephyr Foundry.", "Fact number 2", context=[{"ref": "SUMMARY", "quote": summary}])
    named = mem("Zephyr Foundry hired Ana.", "Fact number 2", subject="Zephyr Foundry")
    raw = json.dumps({"entities": [{"name": "Zephyr Foundry", "type": "company"}], "targets": {"T1": {"outcome": "extracted", "memories": [cited, named]}, "T2": {"outcome": "nothing_durable"}}, SUMMARY_FIELD: "S."})
    client = Scripted(summaries={0: summary}, raw={1: raw})
    run(store, client)
    sx = StructuredStore(store.connection)
    codes = {i["code"] for i in sx.get_windows(CID)[1]["issues"]}
    assert {"INVALID_EVIDENCE_REF", "REFERENCE_NOT_IN_WINDOW", "ENTITY_NOT_IN_WINDOW"} <= codes
    assert "SUMMARY" not in json.dumps(rolling_output_schema(window_of_two())["properties"]["targets"]["properties"]["T1"]["properties"]["memories"]["items"]["properties"]["context"]["items"]["properties"]["ref"]["enum"])


def window_of_two():
    return build_windows(conversation(2), 2, 0)[0]


def test_records_come_only_from_current_targets(store):
    raw = json.dumps({"entities": [], "targets": {"T1": {"outcome": "extracted", "memories": [mem("Ana stated fact two.", "Fact number 2")]}, "T2": {"outcome": "nothing_durable"}, "C1": {"outcome": "extracted"}}, SUMMARY_FIELD: "S."})
    client = Scripted(raw={1: raw})
    run(store, client)
    sx = StructuredStore(store.connection)
    assert [m["message_id"] for m in sx.get_records(CID, "memory")] == ["m2"]
    assert "UNKNOWN_TARGET" in {i["code"] for i in sx.get_windows(CID)[1]["issues"]}
    assert client.schemas[1]["properties"]["targets"]["required"] == ["T1", "T2"]


def test_no_extra_llm_call(store):
    client = Scripted()
    report, _ = run(store, client)
    assert len(client.prompts) == report.model_calls == report.windows == 3


@pytest.mark.parametrize(("summary", "status"), [(..., SUMMARY_MISSING), ("", SUMMARY_INVALID), ("   ", SUMMARY_INVALID), (7, SUMMARY_INVALID), (None, SUMMARY_INVALID), (["x"], SUMMARY_INVALID)])
def test_bad_summary_keeps_the_previous_one_and_the_records(store, summary, status):
    client = Scripted(summaries={0: "Good summary.", 1: summary})
    report, rows = run(store, client)
    assert rows[1]["status"] == status
    assert rows[1]["summary_out"] == "Good summary."
    assert "Good summary." in client.prompts[2]
    assert report.failed_windows == 0
    assert report.windows == 3


@pytest.mark.parametrize("kwargs", [{"fail": [1]}, {"raw": {1: "{broken"}}, {"raw": {1: "[]"}}, {"usage": {1: {"prompt_eval_count": 1, "eval_count": 9, "done_reason": "length"}}}])
def test_failed_window_never_replaces_the_summary(store, kwargs):
    client = Scripted(summaries={0: "Good summary."}, **kwargs)
    report, rows = run(store, client)
    assert rows[1]["status"] == SUMMARY_WINDOW_FAILED
    assert rows[1]["summary_out"] == "Good summary."
    assert "Good summary." in client.prompts[2]
    assert report.failed_windows == 1
    assert client.prompts[2]


def test_failed_window_keeps_its_raw_text(store):
    run(store, Scripted(raw={1: "{broken"}))
    assert StructuredStore(store.connection).get_windows(CID)[1]["raw_output"] == "{broken"


def test_raw_output_keeps_the_summary_but_normalized_output_does_not(store):
    client = Scripted(summaries={0: "Keep me."})
    extractor = RollingExtractor(client)
    window = build_windows(conversation(2), 2, 0)[0]
    result = extractor.extract(window)
    assert "Keep me." in result.validated.raw
    assert SUMMARY_FIELD not in result.validated.normalized
    assert result.summary_status == SUMMARY_KEPT


def test_summary_is_required_and_other_keys_are_still_forbidden():
    window = window_of_two()
    schema = rolling_output_schema(window)
    assert SUMMARY_FIELD in schema["required"]
    assert schema["additionalProperties"] is False
    bad = {"entities": [], "targets": {"T1": {"outcome": "question"}, "T2": {"outcome": "question"}}, SUMMARY_FIELD: "S.", "extra": 1}
    result = RollingExtractor(Scripted(raw={0: json.dumps(bad)})).extract(window)
    assert result.validated.status == "error"
    assert result.summary_status == SUMMARY_WINDOW_FAILED


def test_protocol_and_tables_are_separate(store):
    run(store, Scripted())
    rows = StructuredStore(store.connection).get_windows(CID)
    assert {r["protocol"] for r in rows} == {ROLLING_PROTOCOL_VERSION}
    assert ROLLING_PROTOCOL_VERSION != TARGETS_PROTOCOL_VERSION
    assert SemanticStore(store.connection).get_claims(CID) == []


def test_factory_uses_the_baseline_configuration(monkeypatch):
    from continuator_memory.v2.atomizer import V2Settings
    monkeypatch.delenv("CONTINUATOR_V2_NUM_PREDICT", raising=False)
    extractor = rolling_extractor(V2Settings(atomizer_model="m", ollama_host="http://localhost:1"))
    assert (extractor.model_name, extractor.client.num_predict, extractor.client.num_ctx) == ("m", 4096, 8192)


# ---- the baseline stays as it was ---------------------------------------------

def test_baseline_never_references_the_experiment():
    source = inspect.getsource(structured_targets)
    assert "rolling" not in source.lower()
    assert structured_targets.TARGETS_PROTOCOL_VERSION == "structured-v1.1.1"
    assert SUMMARY_FIELD not in targets_output_schema(window_of_two())["properties"]


def test_baseline_mode_still_rejects_a_summary_and_still_runs(store):
    window = window_of_two()
    with_summary = {"entities": [], "targets": {"T1": {"outcome": "question"}, "T2": {"outcome": "question"}}, SUMMARY_FIELD: "S."}
    assert validate_targets(json.dumps(with_summary), window).issues[0].code == "UNEXPECTED_KEY"
    client = Scripted(summaries={i: ... for i in range(3)})
    report = ingest_targets(conversation(), store, TargetExtractor(client), targets_per_window=2, context_before=2)
    assert report.failed_windows == 0
    assert client.systems[0] == TARGETS_SYSTEM_PROMPT
    assert "ROLLING SUMMARY" not in client.prompts[0]


def test_experiment_adds_only_its_own_prompt_text():
    assert ROLLING_SYSTEM_PROMPT.startswith(TARGETS_SYSTEM_PROMPT)


def test_prompt_has_no_evaluation_entities():
    text = ROLLING_SYSTEM_PROMPT + json.dumps(ROLLING_EXAMPLES)
    for name in ("Maya", "Leo", "Trailmark", "FieldNotes", "Northstar", "GreenCurrent", "Hawthorne", "Bangalore", "Pixel"):
        assert name not in text


# ---- runner, with no Ollama call ------------------------------------------------

def test_runner_dry_run_writes_every_artifact_and_never_overwrites(tmp_path):
    source = tmp_path / "conv.json"
    source.write_text(json.dumps({"conversation_id": "x", "messages": [
        {"id": f"m{i}", "role": "user", "content": f"Ana: Line {i}.", "timestamp": TS} for i in range(8)]}), encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts/run_rolling_eval.py"
    root = tmp_path / "out"
    result = subprocess.run([sys.executable, str(script), "--dry-run", "--smoke", "--input", str(source), "--output-root", str(root)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STATUS: OK" in result.stdout
    assert result.stdout.count("CHECK PASS") == 5
    run_dir = next(root.iterdir())
    assert f"ARTIFACTS: {run_dir}" in result.stdout
    for name in ("config.json", "run.json", "metrics.jsonl", "eval.db", "system_prompt.txt", "examples.json", "prompt_w01.txt", "schema_w01.json",
                 "raw_w01.txt", "normalized_w01.json", "summary_w01.txt", "code/structured_rolling.py", "code/run_rolling_eval.py"):
        assert (run_dir / name).exists(), name
    row = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[1])
    for key in ("input_tokens", "output_tokens", "summary_in_tokens_est", "summary_out_tokens_est", "prompt_eval_s", "generation_s", "total_ollama_s",
                "wall_s", "memories", "relationships", "outcomes", "rejected", "done_reason", "error"):
        assert key in row
