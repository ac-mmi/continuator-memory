"""M4.3e: document BM25 stopword / minimum content-term gating."""

from __future__ import annotations

from pathlib import Path

import pytest

from continuator_memory.evidence.ingest import ingest_markdown
from continuator_memory.evidence.search import (
    document_query_content_tokens,
    search_evidence,
)
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.store import Store
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder
from tests.unit.test_webapi import FERRY_TRANSCRIPT

M43C_CORPUS = Path("tests/fixtures/m4_3c_document_evidence")
M43C_Q13 = "What is the quantum entanglement coffee brewing recipe?"


@pytest.fixture
def ev_store(tmp_path):
    product = Store(str(tmp_path / "gating.db"))
    store = EvidenceStore(product.connection)
    yield store
    product.close()


def _ingest_m43c_corpus(store: EvidenceStore) -> None:
    assert M43C_CORPUS.is_dir(), "M4.3c corpus missing"
    for path in sorted(M43C_CORPUS.glob("*.md")):
        ingest_markdown(
            path,
            store,
            source_uri=f"fixture://m4_3c/{path.stem}",
            clock=lambda: "2030-01-01T00:00:00+00:00",
        )


def test_content_tokens_drop_stopwords_preserve_content():
    assert document_query_content_tokens("What is the?") == []
    assert document_query_content_tokens("   ") == []
    tokens = document_query_content_tokens("How do access tokens expire?")
    assert "how" not in tokens and "do" not in tokens
    assert tokens == ["access", "tokens", "expire"]


def test_stopword_only_query_returns_no_document_evidence(ev_store):
    _ingest_m43c_corpus(ev_store)
    assert search_evidence(ev_store, "What is the?", top_k=10) == []
    assert search_evidence(ev_store, "How do they?", top_k=10) == []


def test_m43c_q13_stopword_leakage_fixed(ev_store):
    """M4.3c q13: stopwords must not retrieve unrelated units; unmatched content → empty."""
    _ingest_m43c_corpus(ev_store)
    content = document_query_content_tokens(M43C_Q13)
    assert content  # still has quantum/entanglement/... after gating
    assert "what" not in content and "is" not in content and "the" not in content
    hits = search_evidence(ev_store, M43C_Q13, top_k=10)
    assert hits == []


def test_mixed_stopword_content_retrieves_expected_evidence(ev_store):
    _ingest_m43c_corpus(ev_store)
    hits = search_evidence(
        ev_store,
        "How long do Aurora access tokens last before refresh?",
        top_k=10,
    )
    assert hits
    assert any(
        "Access tokens expire after one hour" in h.unit.text for h in hits
    )


def test_content_only_query_retains_bm25_behavior(ev_store):
    _ingest_m43c_corpus(ev_store)
    hits = search_evidence(ev_store, "access tokens expire", top_k=5)
    assert hits
    assert "Access tokens expire after one hour." in hits[0].unit.text


def test_rare_unmatched_content_terms_return_empty(ev_store):
    _ingest_m43c_corpus(ev_store)
    assert search_evidence(ev_store, "xyzzyplugh quuxfoobar", top_k=10) == []


def test_document_search_error_keeps_memory_results(tmp_path, monkeypatch):
    db = tmp_path / "err.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=2)
    ws.import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db,
    )
    aurora = (M43C_CORPUS / "aurora_auth_guide.md").read_text(encoding="utf-8")
    ws.ingest_markdown_text(
        aurora, filename="aurora_auth_guide.md", source_uri="fixture://m4_3c/aurora",
    )

    def boom(*_a, **_k):
        raise RuntimeError("simulated document search failure with secret details")

    monkeypatch.setattr("continuator_memory.webapi.workspace.search_evidence", boom)
    pack = ws.context("usual ferry access tokens", retrieval_mode="baseline")
    payload = pack.to_dict()
    assert pack.evidence.matches
    assert payload["documentEvidence"] == []
    assert pack.diagnostics["documentSearch"] == "error"
    assert pack.diagnostics["documentEvidenceCount"] == 0
    blob = str(payload)
    assert "simulated document search failure" not in blob
    getattr(ws, "_store").close()


def test_stopword_only_context_pack_is_successful_empty_bm25(tmp_path):
    db = tmp_path / "empty.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=2)
    ws.import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db,
    )
    aurora = (M43C_CORPUS / "aurora_auth_guide.md").read_text(encoding="utf-8")
    ws.ingest_markdown_text(
        aurora, filename="aurora.md", source_uri="fixture://m4_3c/aurora",
    )
    pack = ws.context("What is the?", retrieval_mode="baseline")
    payload = pack.to_dict()
    assert payload["documentEvidence"] == []
    assert pack.diagnostics["documentSearch"] == "bm25"
    assert pack.diagnostics["documentEvidenceCount"] == 0
    # Memory channel still runs independently.
    assert pack.diagnostics["documentSearch"] != "error"
    getattr(ws, "_store").close()


def test_deletion_and_provenance_still_work(tmp_path):
    db = tmp_path / "del.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=2)
    ws.import_text(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages", db_path=db,
    )
    aurora = (M43C_CORPUS / "aurora_auth_guide.md").read_text(encoding="utf-8")
    ingested = ws.ingest_markdown_text(
        aurora, filename="aurora.md", source_uri="fixture://m4_3c/aurora",
    )
    doc_id = ingested["document"]["documentId"]
    before = ws.context("access tokens expire one hour", retrieval_mode="baseline")
    assert before.document_hits
    assert any(h.document.document_id == doc_id for h in before.document_hits)

    deleted = ws.delete_document(doc_id)
    assert deleted["deleted"] is True
    assert deleted["unitsRemoved"] >= 1

    after = ws.context("access tokens expire one hour", retrieval_mode="baseline")
    assert after.to_dict()["documentEvidence"] == []
    assert after.diagnostics["documentSearch"] == "bm25"
    getattr(ws, "_store").close()
