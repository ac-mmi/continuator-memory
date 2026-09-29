import io
import json
import urllib.error

import numpy as np
import pytest

from continuator_memory.memory.extraction import (
    DIRECT,
    EXTRACTION_SCHEMA,
    ExtractionError,
    ExtractionRequestError,
    KnownFact,
    MemoryExtractor,
    OllamaJSONClient,
    ground,
    parse_extraction,
)
from continuator_memory.memory.formation import MemoryFormation
from continuator_memory.memory.store import MemoryStore
from continuator_memory.models import Message, parse_conversation
from tests.helpers import ScriptedJSONClient, doc, mem, rel


def _msg(message_id, content, role="user", position=0):
    return Message(message_id, "c1", position, role, content)


def test_valid_json_is_accepted():
    raw = json.dumps(doc([mem("Omar", "role", "backend developer")],
                         [rel("Mei", "handles", "offsite budget", valid_from="May 1")]))
    result = parse_extraction(raw)
    assert result.rejected == []
    (m,) = result.memories
    assert (m.subject, m.predicate, m.value, m.kind) == ("Omar", "role", "backend developer", "state")
    assert m.previous_value is None and m.valid_from is None
    (r,) = result.relationships
    assert (r.subject, r.object, r.valid_from) == ("Mei", "offsite budget", "May 1")


@pytest.mark.parametrize(
    "raw",
    ["", "not json", "[1, 2]", '"text"', '{"foo": []}', '{"memories": {}}',
     '{"memories": [], "relationships": "x"}'],
)
def test_invalid_documents_raise(raw):
    with pytest.raises(ExtractionError):
        parse_extraction(raw)


def test_missing_required_fields_are_rejected():
    raw = json.dumps({
        "memories": [
            {"subject": "Omar", "predicate": "role"},
            {"subject": "", "predicate": "role", "value": "x"},
            {"subject": "Omar", "predicate": "role", "value": "dev", "kind": "maybe"},
            {"subject": "Omar", "predicate": "!!!", "value": "dev"},
            {"subject": "Omar", "predicate": "is", "value": "Omar"},
            "just a string",
            mem("Omar", "role", "dev"),
        ],
        "relationships": [{"subject": "Mei", "predicate": "handles"}],
    })
    result = parse_extraction(raw)
    assert [m.value for m in result.memories] == ["dev"]
    assert result.relationships == []
    assert len(result.rejected) == 7
    assert any("missing or empty 'value'" in r for r in result.rejected)
    assert any("invalid kind" in r for r in result.rejected)


def test_multiple_memories_from_one_message():
    raw = json.dumps(doc(
        [mem("Omar", "role", "backend developer"), mem("Omar", "cannot_do", "boat trips")],
        [rel("Omar", "member_of", "team")],
    ))
    result = parse_extraction(raw)
    assert [(m.predicate, m.value) for m in result.memories] == [
        ("role", "backend developer"), ("cannot_do", "boat trips")
    ]
    assert len(result.relationships) == 1


def test_grounding_drops_values_not_in_message():
    result = parse_extraction(json.dumps(doc(
        [mem("Omar", "role", "backend developer"), mem("Omar", "salary", "90k"),
         mem("Zed", "role", "developer"), mem("Omar", "role", "developer", "update", "chef")],
        [rel("Omar", "member_of", "Project Atlas"), rel("Omar", "knows", "Quentin")],
    )))
    grounded = ground(result, _msg("m2", "Omar is the backend developer."),
                      [_msg("m1", "Known: Project Atlas")])
    assert [m.value for m in grounded.memories] == ["backend developer", "developer"]
    assert grounded.memories[1].previous_value is None  # 'chef' unsupported -> cleared
    assert [r.object for r in grounded.relationships] == ["Project Atlas"]
    assert len(grounded.rejected) == 3


def test_invalid_extraction_fails_safely(store):
    texts = ["Omar is the backend developer.", "Omar can't go on boat trips."]
    conv = parse_conversation({"conversation_id": "c1", "messages": [
        {"id": f"m{i}", "role": "user", "content": t} for i, t in enumerate(texts)]})
    store.save_conversation(conv, [], np.zeros((0, 1)), "none")
    ms = MemoryStore(store.connection)
    client = ScriptedJSONClient({texts[0]: "{not valid json",
                                 texts[1]: doc([mem("Omar", "cannot_do", "boat trips")])})
    report = MemoryFormation(ms, MemoryExtractor(client), clock=lambda: "T").form(conv)
    assert report.extraction_status == {"invalid": 1, "ok": 1}
    assert [m.value for m in ms.memories("c1")] == ["boat trips"]
    status = dict(store.connection.execute("SELECT message_id, status FROM extraction_log"))
    assert status == {"m0": "invalid", "m1": "ok"}


def test_extractor_prompt_contains_known_facts_and_recent_messages(store):
    texts = ["Omar is the backend developer.", "He is also vegetarian."]
    conv = parse_conversation({"conversation_id": "c1", "messages": [
        {"id": f"m{i}", "role": "user", "content": t} for i, t in enumerate(texts)]})
    store.save_conversation(conv, [], np.zeros((0, 1)), "none")
    client = ScriptedJSONClient({texts[0]: doc([mem("Omar", "role", "backend developer")])})
    MemoryFormation(MemoryStore(store.connection), MemoryExtractor(client)).form(conv)
    second = client.prompts[1]
    assert "- Omar | role | backend developer" in second
    assert "user: Omar is the backend developer." in second
    assert second.rstrip().endswith("MESSAGE (user):\nHe is also vegetarian.\n\nJSON:")


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_json_client_sends_schema(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return _Resp(json.dumps({"message": {"content": '{"memories": []}'}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = OllamaJSONClient("m", "http://h:1/").complete_json("sys", "p", EXTRACTION_SCHEMA)
    assert out == '{"memories": []}'
    assert captured["format"] == EXTRACTION_SCHEMA
    assert captured["options"] == {"temperature": 0, "seed": 0, "num_predict": 512}
    assert captured["stream"] is False


def test_ollama_unreachable_is_a_request_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ExtractionRequestError):
        OllamaJSONClient("m", "http://h:1").complete_json("s", "p", {})


@pytest.mark.parametrize("error", [TimeoutError("timed out"), OSError("connection reset")])
def test_ollama_timeout_and_os_errors_are_request_errors(monkeypatch, error):
    def fake_urlopen(request, timeout):
        raise error

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ExtractionRequestError):
        OllamaJSONClient("m", "http://h:1").complete_json("s", "p", {})


def test_facts_list_is_accepted():
    result = parse_extraction(json.dumps({"facts": [mem("Omar", "role", "backend developer")]}))
    assert [m.value for m in result.memories] == ["backend developer"]


def test_few_shot_examples_pass_validation_and_grounding():
    from continuator_memory.memory.extraction import EXTRACTION_EXAMPLES

    for prompt, output in EXTRACTION_EXAMPLES:
        known_part, rest = prompt.split("\n\nCONTEXT - RECENT MESSAGES (do not extract):\n", 1)
        recent_part, message_part = rest.split("\n\nMESSAGE (", 1)
        role, content = message_part.split("):\n", 1)
        content = content.rsplit("\n\nJSON:", 1)[0]
        known = [KnownFact(*line[2:].split(" | ")) for line in known_part.splitlines()[1:]
                 if line.startswith("- ")]
        recent = [Message(f"r{i}", "ex", i, *line.split(": ", 1))
                  for i, line in enumerate(recent_part.splitlines()) if line != "(none)"]
        parsed = parse_extraction(output)
        assert parsed.rejected == []
        grounded = ground(parsed, Message("x", "ex", len(recent), role, content), recent, known)
        assert grounded.rejected == [], grounded.rejected
        assert len(grounded.memories) == len(json.loads(output)["facts"])
        assert all(m.grounding == DIRECT and m.evidence_message_ids == ("x",)
                   for m in grounded.memories)

