"""Multi-format ingest_file dispatcher: extension routing only."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from continuator_memory.evidence.ingest import (
    UnsupportedEvidenceFormatError,
    ingest_file,
    resolve_evidence_format,
)
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.store import Store


@pytest.fixture
def ev_store(tmp_path):
    product = Store(str(tmp_path / "evidence.db"))
    store = EvidenceStore(product.connection)
    yield store
    product.close()


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("notes.md", "markdown"),
        ("notes.markdown", "markdown"),
        ("page.html", "html"),
        ("page.htm", "html"),
        ("report.docx", "docx"),
        ("report.pdf", "pdf"),
        ("README.MD", "markdown"),
        ("document.HTML", "html"),
        ("REPORT.PDF", "pdf"),
        ("Brief.Docx", "docx"),
    ],
)
def test_resolve_evidence_format_by_extension(filename, expected):
    assert resolve_evidence_format(filename) == expected


@pytest.mark.parametrize(
    ("filename", "target"),
    [
        ("a.md", "continuator_memory.evidence.ingest.ingest_markdown"),
        ("a.markdown", "continuator_memory.evidence.ingest.ingest_markdown"),
        ("a.html", "continuator_memory.evidence.ingest.ingest_html"),
        ("a.htm", "continuator_memory.evidence.ingest.ingest_html"),
        ("a.docx", "continuator_memory.evidence.ingest.ingest_docx"),
        ("a.pdf", "continuator_memory.evidence.ingest.ingest_pdf"),
        ("A.MD", "continuator_memory.evidence.ingest.ingest_markdown"),
        ("A.HTML", "continuator_memory.evidence.ingest.ingest_html"),
        ("A.PDF", "continuator_memory.evidence.ingest.ingest_pdf"),
    ],
)
def test_ingest_file_routes_to_underlying_facade(ev_store, filename, target):
    sentinel = MagicMock(name="ingested")
    with patch(target, return_value=sentinel) as mocked:
        result = ingest_file(filename, ev_store, source_uri="file://test")
    assert result is sentinel
    mocked.assert_called_once()
    args, kwargs = mocked.call_args
    assert args[0] == filename
    assert args[1] is ev_store
    assert kwargs["source_uri"] == "file://test"


def test_unsupported_extension_fails_clearly(ev_store):
    with pytest.raises(UnsupportedEvidenceFormatError) as exc_info:
        ingest_file("notes.txt", ev_store)
    err = exc_info.value
    assert err.path == "notes.txt"
    assert err.extension == ".txt"
    assert ".md" in str(err)
    assert ".pdf" in str(err)


def test_no_extension_fails_clearly(ev_store):
    with pytest.raises(UnsupportedEvidenceFormatError) as exc_info:
        ingest_file("README", ev_store)
    err = exc_info.value
    assert err.extension is None
    assert "no extension" in str(err)


def test_directory_input_fails_clearly(ev_store, tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    with pytest.raises(UnsupportedEvidenceFormatError) as exc_info:
        ingest_file(directory, ev_store)
    assert "directories are not supported" in str(exc_info.value)


def test_ingest_file_equivalent_to_ingest_markdown(ev_store, tmp_path):
    path = tmp_path / "equiv.md"
    path.write_text("# Title\n\nBody text.\n", encoding="utf-8")
    via_file = ingest_file(
        path, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00"
    )
    # Re-ingest via explicit facade into a fresh store for identity comparison.
    product = Store(str(tmp_path / "other.db"))
    other = EvidenceStore(product.connection)
    try:
        from continuator_memory.evidence.ingest import ingest_markdown

        via_md = ingest_markdown(
            path, other, clock=lambda: "2030-01-01T00:00:00+00:00"
        )
    finally:
        product.close()

    assert via_file.document.document_id == via_md.document.document_id
    assert via_file.document.source_type == "markdown"
    assert via_file.document.source_type == via_md.document.source_type
    assert [u.evidence_id for u in via_file.units] == [u.evidence_id for u in via_md.units]
    assert [u.text for u in via_file.units] == [u.text for u in via_md.units]
    assert [u.heading_path for u in via_file.units] == [
        u.heading_path for u in via_md.units
    ]


def test_resolve_does_not_require_file_to_exist():
    # Routing is extension-only; existence is checked only for directories.
    assert resolve_evidence_format(Path("/tmp/does-not-exist-yet.pdf")) == "pdf"
