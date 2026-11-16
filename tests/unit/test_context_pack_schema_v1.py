"""Context Pack schema v1 contract — assert against actual serialization."""

from __future__ import annotations

from threading import Thread

import pytest

from continuator_memory.evidence.ingest import ingest_markdown
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.webapi.context_pack import (
    CONTEXT_PACK_REQUIRED_KEYS,
    CONTEXT_PACK_SCHEMA_VERSION,
)
from continuator_memory.webapi.server import make_server
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder
from tests.unit.test_context_pack import QUESTION, SEED, TOKEN_DOC
from tests.unit.test_webapi import FERRY_TRANSCRIPT, _http


def _assert_v1_contract(payload: dict) -> None:
    assert payload["schemaVersion"] == CONTEXT_PACK_SCHEMA_VERSION == "1"
    missing = CONTEXT_PACK_REQUIRED_KEYS - payload.keys()
    assert not missing, f"missing required keys: {sorted(missing)}"

    assert isinstance(payload["memories"], list)
    assert isinstance(payload["evidence"], list)
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["documentEvidence"], list)
    assert isinstance(payload["documentCitations"], list)
    assert isinstance(payload["formattedContext"], str)
    assert isinstance(payload["selection"], dict)
    assert isinstance(payload["diagnostics"], dict)

    # Naming quirks frozen in v1.
    for mem in payload["memories"]:
        assert "id" in mem
        assert "memoryId" not in mem

    selection = payload["selection"]
    for key in (
        "budget_chars",
        "unique_sources",
        "selected",
        "dropped_by_budget",
        "used_chars",
        "first_memory",
    ):
        assert key in selection

    for cite in payload["citations"]:
        assert "memoryId" in cite
        assert "messageIds" in cite
        assert "unresolved" in cite
        for u in cite["unresolved"]:
            assert set(u.keys()) <= {"claim_id", "message_id", "reason"}
            assert "claimId" not in u
            assert "messageId" not in u

    # Channels stay separate.
    for row in payload["evidence"]:
        assert "messageId" in row
        assert "evidenceId" not in row
    for row in payload["documentEvidence"]:
        assert "evidenceId" in row
        assert "messageId" not in row

    doc_search = payload["diagnostics"].get("documentSearch")
    assert doc_search in {"skipped", "bm25", "error"}


def test_schema_v1_fixture_pack(workspace: Workspace):
    payload = workspace.context(QUESTION, retrieval_mode="baseline").to_dict()
    _assert_v1_contract(payload)
    assert payload["documentEvidence"] == []
    assert payload["documentCitations"] == []
    assert payload["diagnostics"]["documentSearch"] == "skipped"
    assert any(m["id"] == SEED for m in payload["memories"])
    # Advanced / answer-path keys remain present for compatibility.
    assert "answerPrompt" in payload
    assert "seedMemoryIds" in payload
    assert "expandedMemoryIds" in payload
    assert "hops" in payload
    assert "queryPlan" in payload


def test_schema_v1_bprime_keeps_advanced_fields(workspace: Workspace):
    payload = workspace.context(QUESTION, retrieval_mode="bprime").to_dict()
    _assert_v1_contract(payload)
    assert payload["expandedMemoryIds"]
    assert payload["hops"]
    assert payload["schemaVersion"] == "1"


def test_schema_v1_with_documents_and_empty_doc_query(tmp_path):
    db = tmp_path / "schema-docs.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    ws.import_text(FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db)
    path = tmp_path / "tokens.md"
    path.write_text(TOKEN_DOC, encoding="utf-8")
    ingest_markdown(
        path,
        EvidenceStore(getattr(ws, "_store").connection),
        clock=lambda: "2030-01-01T00:00:00+00:00",
    )

    with_docs = ws.context("access tokens expire", retrieval_mode="baseline").to_dict()
    _assert_v1_contract(with_docs)
    assert with_docs["diagnostics"]["documentSearch"] == "bm25"
    assert with_docs["documentEvidence"]
    assert with_docs["documentCitations"]

    # Document channel can be empty arrays while search still reports bm25.
    no_doc_hits = ws.context("usual ferry", retrieval_mode="baseline").to_dict()
    _assert_v1_contract(no_doc_hits)
    assert no_doc_hits["diagnostics"]["documentSearch"] == "bm25"
    assert no_doc_hits["documentEvidence"] == []
    assert no_doc_hits["documentCitations"] == []
    getattr(ws, "_store").close()


def test_schema_v1_document_search_error_status(tmp_path, monkeypatch):
    db = tmp_path / "schema-err.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db)

    def boom(*_a, **_k):
        raise RuntimeError("secret document failure")

    monkeypatch.setattr("continuator_memory.webapi.workspace.search_evidence", boom)
    payload = ws.context("usual ferry", retrieval_mode="baseline").to_dict()
    _assert_v1_contract(payload)
    assert payload["diagnostics"]["documentSearch"] == "error"
    assert payload["documentEvidence"] == []
    assert "secret document failure" not in str(payload)
    getattr(ws, "_store").close()


def test_http_context_returns_schema_v1(workspace: Workspace):
    httpd = make_server(workspace, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, pack = _http(
            httpd,
            "POST",
            "/context",
            {"query": QUESTION, "mode": "baseline", "limit": 1},
        )
        assert status == 200
        _assert_v1_contract(pack)
        assert pack["diagnostics"]["answerGeneration"] == "not_run"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_get_unexpected_error_is_generic(workspace: Workspace, monkeypatch):
    httpd = make_server(workspace, host="127.0.0.1", port=0)

    def boom():
        raise RuntimeError("internal secret detail")

    monkeypatch.setattr(workspace, "health", boom)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = _http(httpd, "GET", "/health")
        assert status == 500
        assert body == {"error": "internal server error"}
        assert "secret" not in body["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def workspace() -> Workspace:
    return Workspace.from_sample(embedder=HashingEmbedder(), k=1)
