"""Milestone 2: document lifecycle, source inspection, corpus conversation deletion."""

from __future__ import annotations

from threading import Thread

import pytest

from continuator_memory.evidence.ingest import ingest_markdown
from continuator_memory.evidence.search import search_evidence
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.store import Store
from continuator_memory.v2.memory_store import MemoryStore
from continuator_memory.v2.store import SemanticStore
from continuator_memory.webapi.server import make_server
from continuator_memory.webapi.transcript import IMPORTED_CONVERSATION_ID
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder
from tests.unit.test_context_pack import TOKEN_DOC
from tests.unit.test_webapi import FERRY_TRANSCRIPT, _http


def _workspace_with_docs(tmp_path):
    db = tmp_path / "lifecycle.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    ws.import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db,
    )
    path = tmp_path / "tokens.md"
    path.write_text(TOKEN_DOC, encoding="utf-8")
    ingested = ingest_markdown(
        path,
        EvidenceStore(getattr(ws, "_store").connection),
        clock=lambda: "2030-01-01T00:00:00+00:00",
    )
    return ws, ingested


def test_delete_by_document_id_removes_units_and_search(tmp_path):
    ws, ingested = _workspace_with_docs(tmp_path)
    store = EvidenceStore(getattr(ws, "_store").connection)
    assert store.get_units(ingested.document.document_id)
    assert search_evidence(store, "access tokens expire", top_k=5)

    deleted, units_removed = store.delete_by_document_id(ingested.document.document_id)
    assert deleted is True
    assert units_removed >= 1
    assert store.get_document(ingested.document.document_id) is None
    assert store.get_units(ingested.document.document_id) == []
    assert search_evidence(store, "access tokens expire", top_k=5) == []

    deleted2, units2 = store.delete_by_document_id(ingested.document.document_id)
    assert deleted2 is False
    assert units2 == 0
    getattr(ws, "_store").close()


def test_document_delete_clears_context_pack_document_evidence(tmp_path):
    ws, ingested = _workspace_with_docs(tmp_path)
    before = ws.context("How do access tokens expire?", retrieval_mode="baseline")
    assert before.document_hits
    assert any(
        h.document.document_id == ingested.document.document_id for h in before.document_hits
    )

    result = ws.delete_document(ingested.document.document_id)
    assert result["deleted"] is True
    assert result["unitsRemoved"] >= 1

    after = ws.context("How do access tokens expire?", retrieval_mode="baseline")
    assert after.to_dict()["documentEvidence"] == []
    assert after.diagnostics["documentSearch"] == "bm25"
    assert after.diagnostics["documentEvidenceCount"] == 0
    getattr(ws, "_store").close()


def test_document_delete_idempotent_and_http(tmp_path):
    ws, ingested = _workspace_with_docs(tmp_path)
    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        doc_id = ingested.document.document_id
        status, _, first = _http(httpd, "DELETE", f"/documents/{doc_id}")
        assert status == 200
        assert first["deleted"] is True
        status, _, second = _http(httpd, "DELETE", f"/documents/{doc_id}")
        assert status == 200
        assert second["deleted"] is False
        assert second["unitsRemoved"] == 0
    finally:
        httpd.shutdown()
        httpd.server_close()
        getattr(ws, "_store").close()


def test_document_and_evidence_inspection_provenance(tmp_path):
    ws, ingested = _workspace_with_docs(tmp_path)
    listed = ws.list_documents()
    assert len(listed) == 1
    assert listed[0]["documentId"] == ingested.document.document_id
    assert listed[0]["sourceUri"] == ingested.document.source_uri
    assert listed[0]["unitCount"] == len(ingested.units)

    detail = ws.get_document(ingested.document.document_id)
    assert detail is not None
    assert detail["document"]["documentId"] == ingested.document.document_id
    assert len(detail["units"]) == len(ingested.units)

    unit = ingested.units[0]
    inspected = ws.get_evidence_unit(unit.evidence_id)
    assert inspected is not None
    assert inspected["evidence"]["evidenceId"] == unit.evidence_id
    assert inspected["evidence"]["documentId"] == ingested.document.document_id
    assert inspected["evidence"]["sectionId"] == unit.section_id
    assert inspected["evidence"]["headingPath"] == list(unit.heading_path)
    assert inspected["document"]["sourceUri"] == ingested.document.source_uri

    assert ws.get_document("missing-doc") is None
    assert ws.get_evidence_unit("missing-ev") is None

    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, docs = _http(httpd, "GET", "/documents")
        assert status == 200
        assert docs[0]["documentId"] == ingested.document.document_id

        status, _, one = _http(httpd, "GET", f"/documents/{ingested.document.document_id}")
        assert status == 200
        assert one["document"]["unitCount"] == len(ingested.units)

        status, _, ev = _http(httpd, "GET", f"/evidence/{unit.evidence_id}")
        assert status == 200
        assert ev["evidence"]["evidenceId"] == unit.evidence_id

        status, _, err = _http(httpd, "GET", "/documents/nope")
        assert status == 404
        assert err["error"] == "document not found"

        status, _, err = _http(httpd, "GET", "/evidence/nope")
        assert status == 404
        assert err["error"] == "evidence not found"
    finally:
        httpd.shutdown()
        httpd.server_close()
        getattr(ws, "_store").close()


def test_conversation_deletion_preserves_documents_and_blocks_stale_retrieval(tmp_path):
    ws, ingested = _workspace_with_docs(tmp_path)
    cid = ws.conversation.conversation_id
    assert cid == IMPORTED_CONVERSATION_ID
    store = getattr(ws, "_store")
    assert MemoryStore(store.connection).has_snapshot(cid)
    assert SemanticStore(store.connection).has_claims(cid)
    assert store.get_conversation(cid) is not None

    result = ws.delete_corpus_conversation(cid)
    assert result["deleted"] is True
    assert store.get_conversation(cid) is None
    assert not MemoryStore(store.connection).has_snapshot(cid)
    assert not SemanticStore(store.connection).has_claims(cid)
    assert EvidenceStore(store.connection).get_document(ingested.document.document_id) is not None
    assert EvidenceStore(store.connection).unit_count(ingested.document.document_id) >= 1

    with pytest.raises(ValueError, match="corpus conversation has been deleted"):
        ws.context("usual ferry", retrieval_mode="baseline")

    # Documents remain inspectable / searchable via product store.
    assert ws.list_documents()
    hits = search_evidence(
        EvidenceStore(store.connection), "access tokens expire", top_k=5,
    )
    assert hits

    again = ws.delete_corpus_conversation(cid)
    assert again["deleted"] is False

    # Re-import succeeds and restores retrieval.
    restored = ws.import_text(
        FERRY_TRANSCRIPT, filename="conv2.txt", extract="messages", db_path=tmp_path / "lifecycle.db",
    )
    assert restored["ok"] is True
    pack = ws.context("usual ferry", retrieval_mode="baseline")
    assert pack.evidence.matches
    store.close()


def test_conversation_delete_http_and_wrong_id(tmp_path):
    ws, _ingested = _workspace_with_docs(tmp_path)
    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, err = _http(httpd, "DELETE", "/corpus/conversations/other-id")
        assert status == 400
        assert "active Workspace" in err["error"]

        status, _, body = _http(
            httpd, "DELETE", f"/corpus/conversations/{ws.conversation.conversation_id}",
        )
        assert status == 200
        assert body["deleted"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        getattr(ws, "_store").close()


def test_fixture_workspace_rejects_product_deletes():
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    with pytest.raises(ValueError, match="product SQLite store"):
        ws.delete_document("doc:x")
    with pytest.raises(ValueError, match="product SQLite store"):
        ws.delete_corpus_conversation(ws.conversation.conversation_id)
    with pytest.raises(ValueError, match="product SQLite store"):
        ws.list_documents()

    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, err = _http(httpd, "DELETE", "/documents/doc:x")
        assert status == 400
        assert "product SQLite store" in err["error"]
        status, _, err = _http(
            httpd, "DELETE", f"/corpus/conversations/{ws.conversation.conversation_id}",
        )
        assert status == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_delete_by_document_id_on_store_alone(tmp_path):
    product = Store(tmp_path / "ev-only.db")
    store = EvidenceStore(product.connection)
    path = tmp_path / "only.md"
    path.write_text(TOKEN_DOC, encoding="utf-8")
    ingested = ingest_markdown(path, store, clock=lambda: "2030-01-01T00:00:00+00:00")
    deleted, n = store.delete_by_document_id(ingested.document.document_id)
    assert deleted and n >= 1
    product.close()
