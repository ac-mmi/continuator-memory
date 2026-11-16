"""Phase 1 universal evidence layer: Markdown structure, provenance, identity, BM25."""

from __future__ import annotations

from pathlib import Path

import pytest

from continuator_memory.evidence.ingest import (
    build_units_from_markdown,
    canonicalize_source_uri,
    document_id_for_uri,
    evidence_id_for,
    ingest_markdown,
)
from continuator_memory.evidence.markdown import (
    build_retrieval_text,
    parse_markdown_sections,
    section_id_for,
)
from continuator_memory.evidence.models import SOURCE_TYPE_MARKDOWN
from continuator_memory.evidence.search import resolve_provenance, search_evidence
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.store import Store
from continuator_memory.v2.bm25 import RetrievalUnit

NESTED = """# A

Intro under A.

## B

Body under B.

### C

Body under C.
"""

EXAMPLE = """# Project Overview

Some introduction.

## Authentication

Users authenticate using OAuth.

### Token Refresh

Access tokens expire after one hour.

## Database

The application uses PostgreSQL.
"""


@pytest.fixture
def ev_store(tmp_path):
    product = Store(str(tmp_path / "evidence.db"))
    store = EvidenceStore(product.connection)
    yield store
    product.close()


def test_heading_ancestry_nested():
    sections = parse_markdown_sections(NESTED)
    paths = [s.heading_path for s in sections]
    assert ("A",) in paths
    assert ("A", "B") in paths
    assert ("A", "B", "C") in paths
    by_path = {s.heading_path: s for s in sections}
    assert "Body under C." in by_path[("A", "B", "C")].text
    assert by_path[("A", "B", "C")].char_start is not None
    assert by_path[("A", "B", "C")].char_end > by_path[("A", "B", "C")].char_start


def test_example_token_refresh_ancestry():
    sections = parse_markdown_sections(EXAMPLE)
    token = next(s for s in sections if s.heading_path == ("Project Overview", "Authentication", "Token Refresh"))
    assert "Access tokens expire after one hour." in token.text
    assert section_id_for(token.heading_path) == "sec:project-overview/authentication/token-refresh"


def test_retrieval_text_deterministic():
    text = build_retrieval_text(("Authentication", "Token Refresh"), "Access tokens expire after one hour.")
    assert text == "Authentication > Token Refresh\n\nAccess tokens expire after one hour."
    again = build_retrieval_text(("Authentication", "Token Refresh"), "Access tokens expire after one hour.")
    assert text == again


def test_provenance_resolves_to_document_and_structure(ev_store, tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(EXAMPLE, encoding="utf-8")
    result = ingest_markdown(path, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00")
    assert result.document.source_type == SOURCE_TYPE_MARKDOWN
    assert result.document.source_uri == canonicalize_source_uri(path)
    assert result.units

    for unit in result.units:
        resolved = resolve_provenance(ev_store, unit.evidence_id)
        assert resolved is not None
        got_unit, got_doc = resolved
        assert got_unit.evidence_id == unit.evidence_id
        assert got_doc.document_id == result.document.document_id
        assert got_doc.source_uri == result.document.source_uri
        assert isinstance(got_unit.heading_path, tuple)
        assert got_unit.section_id.startswith("sec:")
        # Span offsets present when section was not size-split.
        assert got_unit.char_start is not None
        assert got_unit.char_end is not None
        assert EXAMPLE[got_unit.char_start:got_unit.char_end].strip() == got_unit.text


def test_identity_stable_on_reingest(ev_store, tmp_path):
    path = tmp_path / "stable.md"
    path.write_text(NESTED, encoding="utf-8")
    first = ingest_markdown(path, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00")
    second = ingest_markdown(path, ev_store, clock=lambda: "2030-01-02T00:00:00+00:00")
    assert second.replaced is True
    assert first.document.document_id == second.document.document_id
    assert first.document.document_id == document_id_for_uri(first.document.source_uri)
    assert [u.evidence_id for u in first.units] == [u.evidence_id for u in second.units]
    assert [u.evidence_id for u in first.units] == [
        evidence_id_for(first.document.document_id, i) for i in range(len(first.units))
    ]
    # Snapshot replace: exactly one document row for the URI.
    assert len(ev_store.list_documents()) == 1
    assert len(ev_store.get_units(first.document.document_id)) == len(first.units)


def test_search_finds_token_refresh(ev_store, tmp_path):
    path = tmp_path / "example.md"
    path.write_text(EXAMPLE, encoding="utf-8")
    ingest_markdown(path, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00")
    hits = search_evidence(ev_store, "How do access tokens expire?", top_k=5)
    assert hits
    top = hits[0]
    assert "Access tokens expire after one hour." in top.unit.text
    assert top.unit.heading_path == ("Project Overview", "Authentication", "Token Refresh")
    assert top.document.source_uri.endswith("example.md") or "example.md" in top.document.source_uri


def test_retrieval_units_use_evidence_ids(ev_store, tmp_path):
    from continuator_memory.evidence.search import units_to_retrieval_units

    path = tmp_path / "r.md"
    path.write_text("# T\n\nHello world.\n", encoding="utf-8")
    result = ingest_markdown(path, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00")
    rus = units_to_retrieval_units(list(result.units))
    assert all(isinstance(u, RetrievalUnit) for u in rus)
    assert rus[0].id == result.units[0].evidence_id
    assert "Hello world." in rus[0].retrieval_content


def test_build_units_preserves_ancestry_without_store():
    units = build_units_from_markdown(document_id="doc:test", source=NESTED)
    assert [u.heading_path for u in units] == [("A",), ("A", "B"), ("A", "B", "C")]


def test_conversation_store_untouched_by_evidence_tables(tmp_path):
    """Evidence tables are additive; conversation schema still works."""
    db = str(tmp_path / "shared.db")
    with Store(db) as product:
        EvidenceStore(product.connection)
        # conversations table from product schema still present
        row = product.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='conversations'"
        ).fetchone()
        assert row is not None
        ev = product.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ev_documents'"
        ).fetchone()
        assert ev is not None


def test_docs_example_file_ingests(ev_store):
    example = Path("docs/example.md")
    if not example.exists():
        pytest.skip("docs/example.md missing")
    result = ingest_markdown(example, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00")
    assert result.document.title == "Project Overview"
    assert any("Token Refresh" in u.heading_path for u in result.units)
