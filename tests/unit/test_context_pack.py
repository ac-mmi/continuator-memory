"""Context Pack: retrieval without answering LLM; ask consumes the same pack."""

from __future__ import annotations

from threading import Thread

import pytest

from continuator_memory.evidence.ingest import ingest_markdown
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.v2.memory_store import MemoryStore
from continuator_memory.webapi.context_pack import ContextPack, format_context_pack
from continuator_memory.webapi.server import make_server
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder
from tests.unit.test_webapi import QUESTION, SEED, GOLD, FERRY_TRANSCRIPT, _http

TOKEN_DOC = """# Project Overview

Some introduction.

## Authentication

Users authenticate using OAuth.

### Token Refresh

Access tokens expire after one hour.
"""


@pytest.fixture
def workspace() -> Workspace:
    return Workspace.from_sample(embedder=HashingEmbedder(), k=1)


def test_context_does_not_call_answering_llm(workspace: Workspace, monkeypatch):
    calls: list[str] = []

    def boom(*_a, **_k):
        calls.append("generate")
        raise AssertionError("answering LLM must not run for context()")

    monkeypatch.setattr(
        "continuator_memory.webapi.workspace.generate_answer", boom,
    )
    pack = workspace.context(QUESTION, retrieval_mode="baseline")
    assert isinstance(pack, ContextPack)
    assert calls == []
    assert pack.diagnostics["answerGeneration"] == "not_run"
    assert SEED in {m.memory_id for m in pack.evidence.matches}


def test_context_includes_memories_and_source_evidence(workspace: Workspace):
    pack = workspace.context(QUESTION, retrieval_mode="baseline")
    assert pack.query == QUESTION
    assert pack.mode == "baseline"
    assert pack.evidence.matches
    seed = next(m for m in pack.evidence.matches if m.memory_id == SEED)
    assert "bicycle" in seed.statement.lower() or "commute" in seed.statement.lower() or seed.sources
    assert seed.source_message_ids
    assert "m-commute" in seed.source_message_ids or any(
        s.message_id == "m-commute" for s in seed.sources
    )
    assert pack.answer_context.excerpts or pack.evidence.selection.selected

    payload = pack.to_dict()
    assert payload["query"] == QUESTION
    assert any(m["id"] == SEED for m in payload["memories"])
    cite = next(c for c in payload["citations"] if c["memoryId"] == SEED)
    assert cite["messageIds"]
    evidence_ids = {row["messageId"] for row in payload["evidence"]}
    assert evidence_ids & set(cite["messageIds"])
    assert payload["documentEvidence"] == []
    assert payload["documentCitations"] == []
    assert pack.diagnostics["documentSearch"] == "skipped"
    assert pack.diagnostics["documentEvidenceCount"] == 0
    assert "RELEVANT DOCUMENT EVIDENCE" not in pack.formatted_context


def test_context_bprime_preserves_provenance_links(workspace: Workspace):
    pack = workspace.context(QUESTION, retrieval_mode="bprime")
    assert GOLD in pack.expanded_memory_ids
    assert GOLD in {m.memory_id for m in pack.evidence.matches}
    gold = next(m for m in pack.evidence.matches if m.memory_id == GOLD)
    assert gold.source_message_ids or gold.sources
    assert pack.hops
    assert pack.hops[0]["pairId"] == "P01"


def test_formatted_context_is_deterministic(workspace: Workspace):
    a = workspace.context(QUESTION, retrieval_mode="baseline")
    b = workspace.context(QUESTION, retrieval_mode="baseline")
    assert a.formatted_context == b.formatted_context
    assert format_context_pack(a) == a.formatted_context
    assert "RELEVANT LONG-TERM MEMORY" in a.formatted_context
    assert "Supporting evidence:" in a.formatted_context
    assert a.formatted_context.endswith("\n")


def test_ask_consumes_context_pack(workspace: Workspace, monkeypatch):
    seen: list[ContextPack] = []
    original = workspace.context

    def wrap(*args, **kwargs):
        pack = original(*args, **kwargs)
        seen.append(pack)
        return pack

    monkeypatch.setattr(workspace, "context", wrap)
    result = workspace.ask(QUESTION, retrieval_mode="baseline", generate=False)
    assert len(seen) == 1
    pack = seen[0]
    evidence = result["answer"]["evidence"]
    assert evidence["memoryIds"] == [m.memory_id for m in pack.evidence.matches]
    assert evidence["diagnostics"]["answerGeneration"] == "not_run"
    assert evidence["contextPack"]["formattedContext"] == pack.formatted_context
    assert SEED in evidence["memoryIds"]
    assert GOLD not in evidence["memoryIds"]


def test_http_context_endpoint(workspace: Workspace):
    httpd = make_server(workspace, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, err = _http(httpd, "POST", "/context", {"query": ""})
        assert status == 400
        assert "empty" in err["error"].lower()

        status, _, pack = _http(
            httpd,
            "POST",
            "/context",
            {"query": QUESTION, "mode": "baseline", "limit": 1},
        )
        assert status == 200
        assert pack["query"] == QUESTION
        assert pack["mode"] == "baseline"
        assert any(m["id"] == SEED for m in pack["memories"])
        assert GOLD not in {m["id"] for m in pack["memories"]}
        assert "formattedContext" in pack
        assert pack["diagnostics"]["answerGeneration"] == "not_run"
        assert "RELEVANT LONG-TERM MEMORY" in pack["formattedContext"]
        assert pack["documentEvidence"] == []
        assert pack["documentCitations"] == []
        assert pack["diagnostics"]["documentSearch"] == "skipped"

        # Alias form matching /ask field names.
        status, _, aliased = _http(
            httpd,
            "POST",
            "/context",
            {"question": QUESTION, "retrieval_mode": "bprime", "k": 1},
        )
        assert status == 200
        assert GOLD in {m["id"] for m in aliased["memories"]}
        assert aliased["expandedMemoryIds"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_context_uses_persisted_memory_snapshot(tmp_path):
    db = tmp_path / "ctx.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db,
    )
    store = getattr(ws, "_store")
    assert store is not None
    assert MemoryStore(store.connection).has_snapshot(ws.conversation.conversation_id)
    n_before = len(ws.consolidator.memories)
    pack = ws.context("usual ferry", retrieval_mode="baseline")
    assert pack.evidence.matches
    assert len(ws.consolidator.memories) == n_before
    ferry_hit = any(
        "ferry" in m.statement.lower() or any("ferry" in s.content.lower() for s in m.sources)
        for m in pack.evidence.matches
    )
    assert ferry_hit
    assert pack.diagnostics["documentSearch"] == "bm25"
    assert pack.to_dict()["documentEvidence"] == []
    assert pack.to_dict()["documentCitations"] == []
    store.close()


def test_empty_document_store_emits_empty_document_arrays(tmp_path):
    db = tmp_path / "empty-docs.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db,
    )
    # Ensure schema exists but no documents were ingested.
    EvidenceStore(getattr(ws, "_store").connection)
    pack = ws.context("usual ferry", retrieval_mode="baseline")
    payload = pack.to_dict()
    assert payload["documentEvidence"] == []
    assert payload["documentCitations"] == []
    assert pack.diagnostics["documentSearch"] == "bm25"
    assert pack.diagnostics["documentEvidenceCount"] == 0
    assert "RELEVANT DOCUMENT EVIDENCE" not in pack.formatted_context
    getattr(ws, "_store").close()


def test_document_evidence_with_provenance(tmp_path):
    db = tmp_path / "docs.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db,
    )
    path = tmp_path / "tokens.md"
    path.write_text(TOKEN_DOC, encoding="utf-8")
    ingested = ingest_markdown(
        path, EvidenceStore(getattr(ws, "_store").connection),
        clock=lambda: "2030-01-01T00:00:00+00:00",
    )
    pack = ws.context("How do access tokens expire?", retrieval_mode="baseline")
    payload = pack.to_dict()
    assert pack.diagnostics["documentSearch"] == "bm25"
    assert pack.diagnostics["documentEvidenceCount"] >= 1
    assert payload["documentEvidence"]
    top = payload["documentEvidence"][0]
    assert top["evidenceId"]
    assert top["documentId"] == ingested.document.document_id
    assert top["sourceUri"] == ingested.document.source_uri
    assert top["sourceType"] == ingested.document.source_type
    assert top["sectionId"]
    assert isinstance(top["headingPath"], list)
    assert "Access tokens expire after one hour." in top["text"]
    cite = payload["documentCitations"][0]
    assert cite["evidenceId"] == top["evidenceId"]
    assert cite["documentId"] == top["documentId"]
    assert cite["sourceUri"] == top["sourceUri"]
    assert "RELEVANT DOCUMENT EVIDENCE" in pack.formatted_context
    getattr(ws, "_store").close()


def test_mixed_memory_and_document_evidence_kept_separate(tmp_path):
    db = tmp_path / "mixed.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    ws.import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db,
    )
    path = tmp_path / "tokens.md"
    path.write_text(TOKEN_DOC, encoding="utf-8")
    ingest_markdown(
        path, EvidenceStore(getattr(ws, "_store").connection),
        clock=lambda: "2030-01-01T00:00:00+00:00",
    )
    pack = ws.context("usual ferry access tokens expire", retrieval_mode="baseline")
    payload = pack.to_dict()
    assert pack.evidence.matches
    assert payload["documentEvidence"]
    for row in payload["evidence"]:
        assert "messageId" in row
        assert "evidenceId" not in row
    for row in payload["documentEvidence"]:
        assert "evidenceId" in row
        assert "messageId" not in row
    getattr(ws, "_store").close()


def test_document_search_failure_does_not_break_context_pack(tmp_path, monkeypatch):
    db = tmp_path / "fail-docs.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db,
    )

    def boom(*_a, **_k):
        raise RuntimeError("simulated document search failure with secret details")

    monkeypatch.setattr("continuator_memory.webapi.workspace.search_evidence", boom)
    pack = ws.context("usual ferry", retrieval_mode="baseline")
    payload = pack.to_dict()
    assert pack.evidence.matches
    assert payload["documentEvidence"] == []
    assert payload["documentCitations"] == []
    assert pack.diagnostics["documentSearch"] == "error"
    assert pack.diagnostics["documentEvidenceCount"] == 0
    blob = json_dumps_safe(payload)
    assert "simulated document search failure" not in blob
    assert "secret details" not in blob
    getattr(ws, "_store").close()


def test_ask_answer_context_ignores_document_evidence(tmp_path):
    db = tmp_path / "ask-docs.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=2)
    ws.import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db,
    )
    path = tmp_path / "tokens.md"
    path.write_text(TOKEN_DOC, encoding="utf-8")
    ingest_markdown(
        path, EvidenceStore(getattr(ws, "_store").connection),
        clock=lambda: "2030-01-01T00:00:00+00:00",
    )
    pack = ws.context("usual ferry access tokens", retrieval_mode="baseline")
    assert pack.document_hits
    # AnswerContext remains message-only: excerpts and prompt memory ids are memory path.
    assert all(hasattr(ex, "message_id") for ex in pack.answer_context.excerpts)
    doc_ids = {h.unit.evidence_id for h in pack.document_hits}
    prompt = pack.answer_context.prompt
    for evidence_id in doc_ids:
        assert evidence_id not in prompt
        assert evidence_id not in pack.answer_context.system
    result = ws.ask("usual ferry access tokens", retrieval_mode="baseline", generate=False)
    evidence = result["answer"]["evidence"]
    assert evidence["memoryIds"] == [m.memory_id for m in pack.evidence.matches]
    assert evidence["selectedMessageIds"] == list(pack.evidence.selection.selected)
    assert "documentEvidence" not in evidence or evidence.get("documentEvidence") is None
    getattr(ws, "_store").close()


def json_dumps_safe(payload: dict) -> str:
    import json

    return json.dumps(payload)
