"""PDF adapter: digital-text page extraction → CanonicalDocument → materializer."""

from __future__ import annotations

import pytest

from continuator_memory.evidence.adapter import FormatAdapter
from continuator_memory.evidence.adapters.docx import DOCXAdapter
from continuator_memory.evidence.adapters.html import HTMLAdapter
from continuator_memory.evidence.adapters.markdown import MarkdownAdapter
from continuator_memory.evidence.adapters.pdf import PDFAdapter, PDFNoExtractableTextError
from continuator_memory.evidence.ingest import (
    build_units_from_pdf,
    canonicalize_source_uri,
    document_id_for_uri,
    evidence_id_for,
    ingest_pdf,
)
from continuator_memory.evidence.materialize import materialize_units
from continuator_memory.evidence.models import SOURCE_TYPE_PDF
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.store import Store


def _escape_pdf(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_text_pdf(pages: list[str], *, title: str | None = None) -> bytes:
    """Build a minimal digital-text PDF fixture (Helvetica) without network/deps."""
    objects: list[bytes] = []

    def add_obj(payload: bytes) -> int:
        objects.append(payload)
        return len(objects)

    font_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    for page_text in pages:
        lines = (page_text or "").split("\n")
        content_lines = ["BT", "/F1 12 Tf", "50 750 Td", "14 TL"]
        for i, line in enumerate(lines):
            if i == 0:
                content_lines.append(f"({_escape_pdf(line)}) Tj")
            else:
                content_lines.append("T*")
                content_lines.append(f"({_escape_pdf(line)}) Tj")
        content_lines.append("ET")
        stream = "\n".join(content_lines).encode("latin-1", errors="replace")
        content_id = add_obj(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
        page_id = add_obj(
            (
                f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            ).encode("ascii")
        )
        page_ids.append(page_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add_obj(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    )
    # Patch Parent references now that pages_id is known.
    for pid in page_ids:
        objects[pid - 1] = objects[pid - 1].replace(b"/Parent 0 0 R", f"/Parent {pages_id} 0 R".encode("ascii"))

    catalog_dict = f"<< /Type /Catalog /Pages {pages_id} 0 R >>"
    if title:
        info_id = add_obj(f"<< /Title ({_escape_pdf(title)}) >>".encode("latin-1", errors="replace"))
        catalog_id = add_obj(f"{catalog_dict}".encode("ascii"))
    else:
        info_id = None
        catalog_id = add_obj(catalog_dict.encode("ascii"))

    # Rebuild with correct object numbering by serializing.
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(b"trailer\n")
    trailer = f"<< /Size {len(objects) + 1} /Root {catalog_id} 0 R"
    if info_id is not None:
        trailer += f" /Info {info_id} 0 R"
    trailer += " >>\n"
    out.extend(trailer.encode("ascii"))
    out.extend(f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii"))
    return bytes(out)


def make_blank_pdf(page_count: int = 1) -> bytes:
    """PDF with empty content streams (no extractable text)."""
    return make_text_pdf([""] * page_count)


@pytest.fixture
def ev_store(tmp_path):
    product = Store(str(tmp_path / "evidence.db"))
    store = EvidenceStore(product.connection)
    yield store
    product.close()


def test_pdf_adapter_is_format_adapter():
    adapter = PDFAdapter()
    assert isinstance(adapter, FormatAdapter)
    assert adapter.name == "pdf"
    assert SOURCE_TYPE_PDF in adapter.source_types
    assert adapter.supports("notes.pdf")
    assert not adapter.supports("notes.md")
    assert adapter.supports("unknown", sniff=b"%PDF-1.4 rest")


def test_single_page_digital_text_becomes_evidence():
    data = make_text_pdf(["Hello digital PDF page."], title="Single")
    units = build_units_from_pdf(document_id="doc:test", source=data)
    assert len(units) == 1
    assert "Hello digital PDF page." in units[0].text
    assert units[0].heading_path == ()
    assert units[0].section_id == "sec:page-1"
    assert units[0].metadata["page_number"] == 1
    assert units[0].char_start is None


def test_multi_page_order_preserved():
    data = make_text_pdf(["Page one content.", "Page two content.", "Page three content."])
    units = build_units_from_pdf(document_id="doc:test", source=data)
    assert len(units) == 3
    assert [u.metadata["page_number"] for u in units] == [1, 2, 3]
    assert [u.ordinal for u in units] == [0, 1, 2]
    assert "Page one content." in units[0].text
    assert "Page two content." in units[1].text
    assert "Page three content." in units[2].text
    assert all(u.heading_path == () for u in units)


def test_page_provenance_metadata():
    data = make_text_pdf(["Alpha", "Beta"])
    canonical = PDFAdapter().parse(data, source_uri="inline://p", title_fallback="doc")
    assert canonical.metadata["page_count"] == 2
    assert canonical.metadata["pages_with_text"] == 2
    assert canonical.metadata["parser"] == "pypdf"
    assert [s.metadata["page_number"] for s in canonical.sections] == [1, 2]
    units = materialize_units(canonical, document_id="doc:test")
    assert units[0].metadata["page_number"] == 1
    assert units[1].metadata["page_number"] == 2
    assert units[0].section_id == "sec:page-1"
    assert units[1].section_id == "sec:page-2"


def test_empty_whitespace_pages_skipped_but_recorded():
    data = make_text_pdf(["First page text.", "   \n  ", "Third page text."])
    canonical = PDFAdapter().parse(data, source_uri="inline://e", title_fallback="doc")
    assert canonical.metadata["page_count"] == 3
    assert canonical.metadata["empty_pages"] == [2]
    assert [s.metadata["page_number"] for s in canonical.sections] == [1, 3]
    units = materialize_units(canonical, document_id="doc:test")
    assert len(units) == 2
    assert units[0].metadata["page_number"] == 1
    assert units[1].metadata["page_number"] == 3


def test_no_extractable_text_fails_clearly():
    data = make_blank_pdf(2)
    with pytest.raises(PDFNoExtractableTextError, match="no extractable digital text"):
        PDFAdapter().parse(data, source_uri="inline://blank", title_fallback="doc")
    with pytest.raises(PDFNoExtractableTextError):
        build_units_from_pdf(document_id="doc:test", source=data)


def test_shared_materializer_compatibility():
    data = make_text_pdf(["Materializer body."])
    canonical = PDFAdapter().parse(data, source_uri="inline://m", title_fallback="doc")
    units = materialize_units(canonical, document_id="doc:test")
    assert units[0].evidence_id == "ev:doc:test:u0000"
    assert units[0].retrieval_text == units[0].text  # empty heading_path
    assert units[0].metadata["heading_level"] == 0
    assert units[0].metadata["page_number"] == 1


def test_reingest_deterministic_identity(ev_store, tmp_path):
    data = make_text_pdf(["Stable page one.", "Stable page two."], title="Stable PDF")
    path = tmp_path / "stable.pdf"
    path.write_bytes(data)
    first = ingest_pdf(path, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00")
    second = ingest_pdf(path, ev_store, clock=lambda: "2030-01-02T00:00:00+00:00")

    assert first.document.source_type == SOURCE_TYPE_PDF
    assert first.document.source_uri == canonicalize_source_uri(path)
    assert first.document.document_id == document_id_for_uri(first.document.source_uri)
    assert first.document.title == "Stable PDF"
    assert first.document.metadata["adapter"] == "pdf"
    assert second.replaced is True
    assert first.document.document_id == second.document.document_id
    assert [u.evidence_id for u in first.units] == [u.evidence_id for u in second.units]
    assert [u.evidence_id for u in first.units] == [
        evidence_id_for(first.document.document_id, i) for i in range(len(first.units))
    ]
    assert [u.metadata["page_number"] for u in first.units] == [1, 2]
    assert len(ev_store.list_documents()) == 1


def test_binary_input_and_path_handling(tmp_path):
    data = make_text_pdf(["Binary path input."])
    path = tmp_path / "bin.pdf"
    path.write_bytes(data)
    via_bytes = build_units_from_pdf(document_id="doc:b", source=data)
    via_path = build_units_from_pdf(document_id="doc:b", source=path)
    assert via_bytes[0].text == via_path[0].text
    assert via_bytes[0].metadata["page_number"] == 1


def test_no_fake_headings_from_pdf_typography():
    data = make_text_pdf(["LARGE LOOKING TITLE\nbody under title"])
    units = build_units_from_pdf(document_id="doc:test", source=data)
    assert len(units) == 1
    assert units[0].heading_path == ()
    assert units[0].metadata.get("heading_level") == 0


def test_other_adapters_still_present():
    assert isinstance(MarkdownAdapter(), FormatAdapter)
    assert isinstance(HTMLAdapter(), FormatAdapter)
    assert isinstance(DOCXAdapter(), FormatAdapter)
    assert MarkdownAdapter().supports("x.md")
    assert HTMLAdapter().supports("x.html")
    assert DOCXAdapter().supports("x.docx")
