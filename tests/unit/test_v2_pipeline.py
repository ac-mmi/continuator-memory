import json

from continuator_memory.models import parse_conversation
from continuator_memory.v2.atomizer import AtomizationRequestError, OllamaChatClient, SemanticAtomizer
from continuator_memory.v2.models import STATUS_FAILED, SUPPORT_IDENTITY
from continuator_memory.v2.prompt import EXAMPLE_CASES, EXAMPLES, SYSTEM_PROMPT, build_prompt, output_schema
from continuator_memory.v2.validation import validate_atomization
from continuator_memory.v2.pipeline import V2IngestionPipeline
from continuator_memory.v2.windowing import build_windows

CONVERSATION = {"conversation_id": "trip", "messages": [
    {"id": "m1", "role": "user", "content": "My sister Leah lives in Madrid.", "timestamp": "2026-02-01T09:00:00"},
    {"id": "m2", "role": "assistant", "content": "Nice! Will you visit her?"},
    {"id": "m3", "role": "user", "content": "Yes. She moved there because of work, so I'll stay with her."},
    {"id": "m4", "role": "user", "content": "Correction: Leah actually lives in Valencia now."},
]}


def c(cid, content, entities=(), polarity="positive"):
    return {"claim_id": cid, "content": content, "entities": list(entities), "qualifiers": [],
            "temporal": {"expression": "", "resolved_date": ""}, "polarity": polarity, "claim_type": "",
            "source_message_id": cid.rsplit(":c", 1)[0]}


def r(src, dst, rtype, evidence=""):
    return {"from_claim_id": src, "to_claim_id": dst, "type": rtype, "source_message_id": src.rsplit(":c", 1)[0],
            "evidence": evidence}


class ScriptedAtomizerClient:
    """Returns one scripted document per window (keyed by the first target message ID)."""

    model_name = "scripted-atomizer"

    def __init__(self, by_first_target):
        self.by_first_target = by_first_target
        self.calls = []

    def complete_json(self, system, prompt, schema, examples=()):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema, "examples": examples})
        first = schema["properties"]["messages"]["items"]["properties"]["source_message_id"]["enum"][0]
        doc = self.by_first_target[first]
        if isinstance(doc, Exception):
            raise doc
        return doc if isinstance(doc, str) else json.dumps(doc)


SCRIPT = {
    "m1": {"messages": [
        {"source_message_id": "m1", "status": "claims",
         "claims": [c("m1:c1", "Leah is the user's sister.", ["Leah"]), c("m1:c2", "Leah lives in Madrid.", ["Leah", "Madrid"])],
         "explicit_relations": []},
        {"source_message_id": "m2", "status": "no_durable_assertion", "claims": [], "explicit_relations": []}]},
    "m3": {"messages": [
        {"source_message_id": "m3", "status": "claims",
         "claims": [c("m3:c1", "Leah moved to Madrid because of work.", ["Leah", "Madrid"]),
                    c("m3:c2", "The user will stay with Leah.", ["Leah"])],
         "explicit_relations": [r("m3:c2", "m3:c1", "CAUSED_BY", "so")]},
        {"source_message_id": "m4", "status": "claims",
         "claims": [c("m4:c1", "Leah actually lives in Valencia now.", ["Leah", "Valencia"])],
         "explicit_relations": [r("m4:c1", "m1:c2", "UPDATES", "Correction")]}]},
}


def pipeline(store, script=SCRIPT):
    client = ScriptedAtomizerClient(script)
    return V2IngestionPipeline(store, SemanticAtomizer(client), targets_per_window=2, context_before=2), client


def test_ingest_stores_validated_semantic_records(store):
    p, client = pipeline(store)
    report = p.ingest(parse_conversation(CONVERSATION))
    assert report.to_dict() | {"latency_ms": 0} == {
        "conversation_id": "trip", "windows": 2, "messages": 4, "claims": 5, "relations": 2, "failed_windows": 0,
        "statuses": {"claims": 3, "no_durable_assertion": 1}, "issues": {}, "latency_ms": 0,
        "model_calls": 2, "prompt_tokens": 0, "output_tokens": 0, "kept_existing": False}
    records = p.semantic_records("trip")
    assert [(m.source_message_id, m.status, len(m.claims)) for m in records] == [
        ("m1", "claims", 2), ("m2", "no_durable_assertion", 0), ("m3", "claims", 2), ("m4", "claims", 1)]
    assert [x.claim_id for m in records for x in m.claims] == ["m1:c1", "m1:c2", "m3:c1", "m3:c2", "m4:c1"]
    relations = [x for m in records for x in m.explicit_relations]
    assert [(x.relation_id, x.from_claim_id, x.to_claim_id, x.type) for x in relations] == [
        ("m3:r1", "m3:c2", "m3:c1", "CAUSED_BY"), ("m4:r1", "m4:c1", "m1:c2", "UPDATES")]
    assert len(client.calls) == 2


def test_records_trace_back_to_raw_messages_without_copying_content(store):
    p, _ = pipeline(store)
    p.ingest(parse_conversation(CONVERSATION))
    m1 = p.semantic_records("trip")[0]
    assert (m1.content, m1.role, m1.timestamp) == ("My sister Leah lives in Madrid.", "user", "2026-02-01T09:00:00")
    assert m1.claims[0].provenance.timestamp == "2026-02-01T09:00:00"
    columns = {row[1] for row in store.connection.execute("PRAGMA table_info(v2_semantic_messages)")}
    assert "content" not in columns
    raw = store.get_messages("trip", [m1.claims[0].source_message_id])[0]
    assert raw.content == m1.content


def test_prior_state_is_passed_and_resolves_identity_only(store):
    p, client = pipeline(store)
    p.ingest(parse_conversation(CONVERSATION))
    second_prompt = client.calls[1]["prompt"]
    assert "PRIOR STATE (identity/reference context only, NOT evidence):\n- [m1:c1] Leah is the user's sister." in second_prompt
    assert "CONTEXT MESSAGES (do not atomize):\n[m1] user @ 2026-02-01T09:00:00:" in second_prompt
    stay = next(x for x in p.semantic.get_claims("trip") if x.claim_id == "m3:c2")
    assert stay.support == SUPPORT_IDENTITY and "m1" in stay.resolved_from
    log = p.semantic.get_log("trip")
    assert log[1]["prior_claim_ids"] == ["m1:c1", "m1:c2"] and log[1]["context_ids"] == ["m1", "m2"]


def test_request_failure_is_logged_and_other_windows_continue(store):
    script = dict(SCRIPT, m1=AtomizationRequestError("connection refused"))
    p, _ = pipeline(store, script)
    report = p.ingest(parse_conversation(CONVERSATION))
    assert report.failed_windows == 1
    statuses = [m.status for m in p.semantic_records("trip")]
    assert statuses[:2] == [STATUS_FAILED, STATUS_FAILED]
    log = p.semantic.get_log("trip")
    assert log[0]["status"] == "error" and "connection refused" in log[0]["error"]
    # without m1's claims the UPDATES target is unavailable, so only that relation is rejected
    assert any(i["code"] == "UNAVAILABLE_SOURCE" for i in log[1]["issues"])
    assert [x.claim_id for x in p.semantic.get_claims("trip")] == ["m3:c1", "m3:c2", "m4:c1"]


def test_reingest_replaces_semantic_records(store):
    p, _ = pipeline(store)
    p.ingest(parse_conversation(CONVERSATION))
    p.ingest(parse_conversation(CONVERSATION))
    assert len(p.semantic.get_claims("trip")) == 5
    assert len(p.semantic.get_log("trip")) == 2


def test_raw_chunks_and_embeddings_written_with_embedder(store, embedder):
    client = ScriptedAtomizerClient(SCRIPT)
    p = V2IngestionPipeline(store, SemanticAtomizer(client), embedder, targets_per_window=2, context_before=2)
    p.ingest(parse_conversation(CONVERSATION))
    ids, matrix = store.load_embeddings(embedder.model_name, "trip")
    assert ids and matrix.shape[0] == len(ids)


def test_v1_tables_unchanged_by_v2_schema(store):
    before = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    pipeline(store)
    after = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert after - before == {"v2_semantic_messages", "v2_claims", "v2_relations", "v2_atomization_log"}


# ---- prompt -----------------------------------------------------------------


def test_prompt_marks_targets_context_and_prior():
    conv = parse_conversation(CONVERSATION)
    w = build_windows(conv, 2, 2)[1]
    prompt = build_prompt(w)
    assert "PRIOR STATE (identity/reference context only, NOT evidence):\n(none)" in prompt
    assert "TARGET MESSAGES (atomize every one):\n[m3] user:" in prompt
    schema = output_schema(w)
    entry = schema["properties"]["messages"]["items"]
    assert entry["properties"]["source_message_id"]["enum"] == ["m3", "m4"]
    assert entry["properties"]["explicit_relations"]["items"]["properties"]["type"]["enum"] == [
        "UPDATES", "CONTRADICTS", "CAUSED_BY", "SUPPORTS", "REFERS_TO"]
    assert "confidence" not in json.dumps(schema)


def test_system_prompt_states_the_contract():
    for phrase in ("not summarizing", "SOURCE EVIDENCE", "IDENTITY CONTEXT", "never invent dates",
                   "Never infer causality from two statements occurring together", "no_durable_assertion"):
        assert phrase in SYSTEM_PROMPT


def test_examples_pass_the_validator_and_cover_the_required_behaviours():
    results = [validate_atomization(json.dumps(out), w, prior) for w, prior, out in EXAMPLE_CASES]
    assert all(v.issues == [] for v in results)
    claims = [x for v in results for x in v.claims]
    relations = [x for v in results for x in v.relations]
    and_example = results[0]
    assert len(and_example.claims) == 3 and and_example.relations == []          # "and" is not causal
    assert {x.type for x in relations} == {"CAUSED_BY", "UPDATES"}
    assert any(x.polarity == "negative" for x in claims)
    assert any(x.temporal for x in claims)
    assert any(x.support == SUPPORT_IDENTITY for x in claims)                   # pronoun via prior state
    assert any(m.status == "no_durable_assertion" for v in results for m in v.messages)
    assert len(EXAMPLES) == len(EXAMPLE_CASES)


def test_ollama_client_payload_is_deterministic():
    client = OllamaChatClient("qwen2.5:14b", "http://localhost:11434/")
    payload = client.payload("sys", "prompt", {"type": "object"}, (("p", "o"),))
    assert payload["options"] == {"temperature": 0, "seed": 0, "num_predict": 2048, "num_ctx": 8192}
    assert [m["role"] for m in payload["messages"]] == ["system", "user", "assistant", "user"]
    assert client.host == "http://localhost:11434"
