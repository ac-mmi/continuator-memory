"""Lightweight checks for the interactive HTTP API. No extraction or Ollama."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread

import pytest

import numpy as np

from continuator_memory.cli import main
from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from continuator_memory.v2.bprime_links import BPrimeConfig
from continuator_memory.v2.memory import memory_id
from continuator_memory.webapi.server import make_server
from continuator_memory.webapi.transcript import IMPORTED_CONVERSATION_ID
from continuator_memory.webapi.workspace import Workspace, open_workspace
from tests.helpers import HashingEmbedder

QUESTION = "weekday bicycle"
SEED = memory_id("seed")
GOLD = memory_id("gold")


@pytest.fixture
def workspace() -> Workspace:
    return Workspace.from_sample(embedder=HashingEmbedder(), k=1)


def test_health_does_not_load_embedder():
    ws = Workspace.from_sample(embedder=None, k=1)
    assert ws._embedder is None
    health = ws.health()
    assert health["nMemories"] == 6
    assert health["answerGeneration"] in {"not_configured", "configured", "disabled"}
    assert "generation" in health
    assert "host" not in health["generation"]
    assert "apiKey" not in health["generation"]
    assert ws._embedder is None


def test_sample_workspace_lists_memories_and_provenance(workspace: Workspace):
    assert {m["id"] for m in workspace.list_memories()} == {
        memory_id(cid) for cid in ("seed", "remote", "coffee", "gold", "uke", "cat")
    }
    detail = workspace.get_memory(SEED)
    assert detail is not None
    assert detail["evidence"][0]["message"]["id"] == "m-commute"


def test_baseline_does_not_expand_to_guitar(workspace: Workspace):
    result = workspace.ask(QUESTION, retrieval_mode="baseline", generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["diagnostics"]["mode"] == "baseline"
    assert evidence["diagnostics"]["answerGeneration"] == "not_run"
    assert SEED in evidence["memoryIds"]
    assert GOLD not in evidence["memoryIds"]
    assert evidence["expandedMemoryIds"] == []
    assert evidence["hops"] == []
    assert "m-commute" in evidence["selectedMessageIds"]
    assert "Answer generation is not run" in result["answer"]["content"]


def test_bprime_adds_guitar_via_related_link(workspace: Workspace):
    result = workspace.ask(QUESTION, retrieval_mode="bprime", generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["seedMemoryIds"] == [SEED]
    assert GOLD in evidence["expandedMemoryIds"]
    assert GOLD in evidence["memoryIds"]
    assert evidence["diagnostics"]["expandedCount"] == 1
    assert evidence["hops"][0]["pairId"] == "P01"
    assert evidence["hops"][0]["relation"] == "RELATED"
    assert "owned instrument" in (evidence["hops"][0]["rationale"] or "")
    assert "m-guitar" in evidence["selectedMessageIds"]


def test_hybrid_mode_is_available(workspace: Workspace):
    result = workspace.ask(QUESTION, retrieval_mode="hybrid", generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["diagnostics"]["mode"] == "hybrid"
    assert evidence["diagnostics"]["lexicalWeight"] == 0.25
    assert evidence["diagnostics"]["k"] == 1


def test_ask_validates_input(workspace: Workspace):
    with pytest.raises(ValueError, match="empty"):
        workspace.ask("   ")
    with pytest.raises(ValueError, match="retrieval_mode"):
        workspace.ask(QUESTION, retrieval_mode="unknown-mode")
    result = workspace.ask(QUESTION, retrieval_mode="multiquery", generate=False)
    assert result["answer"]["evidence"]["diagnostics"]["mode"] == "multiquery"
    assert result["answer"]["evidence"]["queryPlan"]["queries"][0] == QUESTION


def test_open_workspace_rejects_unknown_sample():
    with pytest.raises(ValueError, match="fixture"):
        open_workspace(sample="full-export")


def test_serve_command_does_not_build_pipeline(monkeypatch):
    called: list[str] = []
    monkeypatch.setattr("continuator_memory.cli._build_pipeline", lambda *a, **k: called.append("pipeline"))
    monkeypatch.setattr(
        "continuator_memory.webapi.server.serve_from_env",
        lambda **k: called.append(f"serve:{k['sample']}:{k['port']}"),
    )
    assert main(["serve", "--port", "9", "--sample", "fixture"]) == 0
    assert called == ["serve:fixture:9"]


def _http(httpd, method: str, path: str, body: dict | None = None, origin: str | None = None):
    host, port = httpd.server_address
    conn = HTTPConnection(host, port, timeout=5)
    headers = {}
    payload = None
    if origin:
        headers["Origin"] = origin
    if body is not None:
        payload = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(payload))
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    raw = response.read()
    conn.close()
    data = json.loads(raw) if raw else None
    return response.status, dict(response.getheaders()), data


def test_http_health_ask_and_cors(workspace: Workspace):
    httpd = make_server(workspace, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, health = _http(httpd, "GET", "/health")
        assert status == 200
        assert health["ok"] is True
        assert health["nMemories"] == 6
        assert health["answerGeneration"] in {"not_configured", "configured", "disabled"}

        status, _, memories = _http(httpd, "GET", "/memories")
        assert status == 200
        assert len(memories) == 6

        status, _, err = _http(httpd, "POST", "/ask", {"question": ""})
        assert status == 400
        assert "empty" in err["error"].lower()

        status, headers, _ = _http(
            httpd, "OPTIONS", "/ask", origin="http://127.0.0.1:5173"
        )
        assert status == 204
        allow = {k.lower(): v for k, v in headers.items()}
        assert allow["access-control-allow-origin"] == "http://127.0.0.1:5173"

        status, _, baseline = _http(
            httpd, "POST", "/ask",
            {"question": QUESTION, "retrieval_mode": "baseline", "generate": False},
        )
        assert status == 200
        base_ids = baseline["answer"]["evidence"]["memoryIds"]
        assert SEED in base_ids
        assert GOLD not in base_ids

        status, _, linked = _http(
            httpd, "POST", "/ask",
            {"question": QUESTION, "retrieval_mode": "bprime", "generate": False},
        )
        assert status == 200
        ev = linked["answer"]["evidence"]
        assert GOLD in ev["expandedMemoryIds"]
        assert ev["diagnostics"]["k"] == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


LABELED = "You: I keep a sourdough starter in the fridge.\n\nChatGPT: That starter is a durable fact.\n"


def test_failed_import_leaves_fixture(workspace: Workspace, tmp_path):
    before = workspace.health()["conversationId"]
    with pytest.raises(Exception, match="empty"):
        workspace.import_text("  \n", filename="empty.txt", extract="messages", db_path=tmp_path / "imp.db")
    assert workspace.health()["conversationId"] == before
    assert workspace.health()["sampleId"] == "fixture"


def test_import_replaces_fixture_and_answers(workspace: Workspace, tmp_path):
    result = workspace.import_text(
        LABELED, filename="conv.txt", extract="messages", db_path=tmp_path / "imp.db"
    )
    assert result["ok"] is True
    assert result["report"]["nMessages"] == 2
    assert result["report"]["nMemories"] >= 1
    health = workspace.health()
    assert health["sampleId"] == "imported"
    assert health["importedFilename"] == "conv.txt"
    assert health["extraction"] == "message_claims"
    sources = workspace.list_sources()
    assert sources[0]["origin"] == "imported"
    assert sources[0]["title"].startswith("Imported")
    asked = workspace.ask("sourdough starter", retrieval_mode="baseline", generate=False)
    evidence = asked["answer"]["evidence"]
    assert evidence["memoryIds"]
    bprime = workspace.ask("sourdough starter", retrieval_mode="bprime", generate=False)
    assert bprime["answer"]["evidence"]["diagnostics"]["mode"] == "bprime"


def test_unknown_post_route_names_method_and_path(workspace: Workspace):
    httpd = make_server(workspace, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, err = _http(httpd, "POST", "/import-conversation", {"filename": "x.txt", "text": "You: hi"})
        assert status == 404
        assert err["error"] == "POST /import-conversation is not a registered route"
        assert workspace.health()["sampleId"] == "fixture"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_import(workspace: Workspace, tmp_path, monkeypatch):
    monkeypatch.setenv("CONTINUATOR_IMPORT_DB", str(tmp_path / "http-imp.db"))
    httpd = make_server(workspace, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, err = _http(httpd, "POST", "/import", {"filename": "x.txt", "text": "   "})
        assert status == 400
        assert "empty" in err["error"].lower()
        assert workspace.health()["sampleId"] == "fixture"

        status, _, payload = _http(
            httpd,
            "POST",
            "/import",
            {"filename": "mine.txt", "text": LABELED, "extract": "messages"},
        )
        assert status == 200
        assert payload["report"]["nMessages"] == 2
        assert workspace.health()["sampleId"] == "imported"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_ask_mocked_generation_and_rejects_secrets(workspace: Workspace):
    class Fake:
        model_name = "fake-reader"

        def generate(self, system, prompt):
            return "Alex commutes on a weekday bicycle [m-commute]."

    workspace.set_generator(Fake())
    httpd = make_server(workspace, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, err = _http(
            httpd,
            "POST",
            "/ask",
            {"question": QUESTION, "generation": {"apiKey": "sk-test"}},
        )
        assert status == 400
        assert "credentials" in err["error"]

        status, _, payload = _http(
            httpd,
            "POST",
            "/ask",
            {
                "question": QUESTION,
                "retrieval_mode": "baseline",
                "generate": True,
                "generation": {"model": "fake-reader", "temperature": 0.0},
            },
        )
        assert status == 200
        answer = payload["answer"]
        assert answer["content"] == "Alex commutes on a weekday bicycle [m-commute]."
        evidence = answer["evidence"]
        assert SEED in evidence["memoryIds"]
        assert SEED in evidence["promptMemoryIds"]
        assert GOLD not in evidence["memoryIds"]
        assert evidence["generation"]["status"] == "ok"
        assert evidence["generation"]["model"] == "fake-reader"
        assert "host" not in evidence["generation"]
    finally:
        httpd.shutdown()
        httpd.server_close()


FERRY_TRANSCRIPT = (
    "You: no my usual ferry is 7 40 to 6 10\n\n"
    "ChatGPT: Got it, your usual ferry is 7:40 to 6:10.\n"
)


def _sqlite_messages_only(path, conversation_id: str, title: str) -> None:
    conv = Conversation(
        conversation_id,
        (
            Message(f"{conversation_id}:m1", conversation_id, 0, "user", "lisbon hotel dates"),
            Message(f"{conversation_id}:m2", conversation_id, 1, "assistant", "early September works"),
        ),
        title=title,
    )
    Store(path).save_conversation(conv, [], np.zeros((0, 1), dtype=np.float32), "none")


def test_from_sqlite_prefers_conversation_with_claims(tmp_path):
    db = tmp_path / "both.db"
    _sqlite_messages_only(db, "trip-planning-001", "Planning a trip to Lisbon")
    importer = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    importer.import_text(FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db)
    getattr(importer, "_store").close()
    ws = Workspace.from_sqlite(db, embedder=HashingEmbedder(), k=5)
    assert ws.conversation.conversation_id == IMPORTED_CONVERSATION_ID
    assert ws.claims
    asked = ws.ask("what is my usual ferry", generate=False)
    texts = " ".join(m["content"] for m in asked["answer"]["evidence"]["selectedMessages"])
    assert "usual ferry is 7 40 to 6 10" in texts
    assert asked["answer"]["evidence"]["memoryIds"]


def test_db_sample_uses_import_db_when_served_sqlite_has_no_claims(tmp_path, monkeypatch):
    served = tmp_path / "continuator.db"
    imported = tmp_path / "continuator-web-import.db"
    _sqlite_messages_only(served, "trip-planning-001", "Planning a trip to Lisbon")
    Workspace.from_sample(embedder=HashingEmbedder(), k=1).import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=imported
    )
    monkeypatch.setenv("CONTINUATOR_IMPORT_DB", str(imported))

    empty = Workspace.from_sqlite(served, embedder=HashingEmbedder(), k=5)
    assert empty.conversation.conversation_id == "trip-planning-001"
    assert empty.claims == ()
    stuck = empty.ask("what is my usual ferry", generate=False)
    assert stuck["answer"]["evidence"]["memoryIds"] == []
    assert stuck["answer"]["evidence"]["selectedMessageIds"] == []
    _close = getattr(empty, "_store", None)
    if _close is not None:
        _close.close()

    ws = open_workspace(sample="db", db_path=served, embedder=HashingEmbedder(), k=5)
    assert ws.health()["conversationId"] == IMPORTED_CONVERSATION_ID
    assert ws.health()["nClaims"] > 0
    asked = ws.ask("what is my usual ferry", retrieval_mode="baseline", generate=False)
    evidence = asked["answer"]["evidence"]
    assert evidence["memoryIds"]
    assert evidence["selectedMessageIds"]
    texts = " ".join(m["content"] for m in evidence["selectedMessages"])
    assert "no my usual ferry is 7 40 to 6 10" in texts
    bprime = ws.ask("what is my usual ferry", retrieval_mode="bprime", generate=False)
    assert bprime["answer"]["evidence"]["diagnostics"]["mode"] == "bprime"
    assert bprime["answer"]["evidence"]["memoryIds"]

    pinned = open_workspace(
        sample="db", db_path=served, conversation_id="trip-planning-001", embedder=HashingEmbedder(), k=5,
    )
    assert pinned.conversation.conversation_id == "trip-planning-001"
    assert pinned.ask("what is my usual ferry", generate=False)["answer"]["evidence"]["memoryIds"] == []


# ---- release boundary: API imports, the heuristic evaluator, B′ configuration ----------------

def test_api_imports_no_benchmark_or_diagnostic_module():
    code = (
        "import sys, continuator_memory.webapi.server;"
        "bad = [m for m in sys.modules if m.split('.')[0] == 'benchmarks' or m.endswith(('iterative_diag', 'ranking_diag'))];"
        "print(bad)"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env, cwd="/")
    assert out.stdout.strip() == "[]"


def test_iterative_api_mode_does_not_use_the_heuristic_evaluator(workspace: Workspace, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("heuristic_evaluation must not run on the API path")

    monkeypatch.setattr("continuator_memory.webapi.evidence_eval.heuristic_evaluation", boom)
    result = workspace.ask(QUESTION, retrieval_mode="iterative", generate=False)
    assert result["answer"]["evidence"]["diagnostics"]["mode"] == "iterative"
    assert workspace._evaluator is None


def test_bprime_without_classifications_fails_clearly(workspace: Workspace):
    workspace.classifications_path = None
    with pytest.raises(ValueError, match="CONTINUATOR_BPRIME_CLASSIFICATIONS"):
        workspace.ask(QUESTION, retrieval_mode="bprime", generate=False)
    with pytest.raises(ValueError, match="classifications_path"):
        BPrimeConfig(enabled=True).resolved_path()
    assert BPrimeConfig(enabled=False).resolved_path() is None


def test_bprime_with_an_explicit_path_is_used_as_given(workspace: Workspace, tmp_path):
    explicit = tmp_path / "links.jsonl"
    explicit.write_text(workspace.classifications_path.read_text())
    assert BPrimeConfig(enabled=True, classifications_path=explicit).resolved_path() == explicit
    workspace.classifications_path = explicit
    evidence = workspace.ask(QUESTION, retrieval_mode="bprime", generate=False)["answer"]["evidence"]
    assert GOLD in evidence["expandedMemoryIds"]
