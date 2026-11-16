"""DOCX adapter: Heading styles → CanonicalDocument → shared materializer."""

from __future__ import annotations

from io import BytesIO

import pytest
from docx import Document as DocxDocument
from docx.enum.style import WD_STYLE_TYPE
from docx.shared import Pt

from continuator_memory.evidence.adapter import FormatAdapter
from continuator_memory.evidence.adapters.docx import DOCXAdapter
from continuator_memory.evidence.adapters.html import HTMLAdapter
from continuator_memory.evidence.adapters.markdown import MarkdownAdapter
from continuator_memory.evidence.ingest import (
    build_units_from_docx,
    build_units_from_html,
    build_units_from_markdown,
    canonicalize_source_uri,
    document_id_for_uri,
    evidence_id_for,
    ingest_docx,
)
from continuator_memory.evidence.materialize import materialize_units
from continuator_memory.evidence.models import SOURCE_TYPE_DOCX
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.store import Store


def _ensure_heading_styles(document: DocxDocument) -> None:
    """Ensure Heading 1–6 exist (python-docx default template usually has them)."""
    for level in range(1, 7):
        name = f"Heading {level}"
        try:
            document.styles[name]
        except KeyError:
            style = document.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            style.font.bold = True
            style.font.size = Pt(14 + (6 - level))


def _docx_bytes(build) -> bytes:
    document = DocxDocument()
    _ensure_heading_styles(document)
    # Remove the default empty paragraph if present.
    if document.paragraphs and not document.paragraphs[0].text:
        p = document.paragraphs[0]._element
        p.getparent().remove(p)
    build(document)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _add_heading(document: DocxDocument, text: str, level: int) -> None:
    document.add_paragraph(text, style=f"Heading {level}")


def _add_para(document: DocxDocument, text: str, *, style: str | None = None) -> None:
    if style:
        document.add_paragraph(text, style=style)
    else:
        document.add_paragraph(text)


@pytest.fixture
def ev_store(tmp_path):
    product = Store(str(tmp_path / "evidence.db"))
    store = EvidenceStore(product.connection)
    yield store
    product.close()


def test_docx_adapter_is_format_adapter():
    adapter = DOCXAdapter()
    assert isinstance(adapter, FormatAdapter)
    assert adapter.name == "docx"
    assert SOURCE_TYPE_DOCX in adapter.source_types
    assert adapter.supports("notes.docx")
    assert not adapter.supports("notes.md")
    assert adapter.supports("unknown", sniff=b"PK\x03\x04rest")


def test_heading_hierarchy_h1_h2_h3():
    data = _docx_bytes(
        lambda d: (
            _add_heading(d, "A", 1),
            _add_para(d, "Intro under A."),
            _add_heading(d, "B", 2),
            _add_para(d, "Body under B."),
            _add_heading(d, "C", 3),
            _add_para(d, "Body under C."),
        )
    )
    adapter = DOCXAdapter()
    canonical = adapter.parse(data, source_uri="inline://n", title_fallback="doc")
    assert len(canonical.sections) == 1
    root = canonical.sections[0]
    assert root.heading_path == ("A",)
    assert root.text == "Intro under A."
    assert len(root.children) == 1
    assert root.children[0].heading_path == ("A", "B")
    assert root.children[0].children[0].heading_path == ("A", "B", "C")
    assert root.children[0].children[0].text == "Body under C."

    units = materialize_units(canonical, document_id="doc:test")
    assert [u.heading_path for u in units] == [("A",), ("A", "B"), ("A", "B", "C")]
    assert [u.text for u in units] == ["Intro under A.", "Body under B.", "Body under C."]
    assert [u.ordinal for u in units] == [0, 1, 2]
    assert units[2].section_id == "sec:a/b/c"
    assert units[2].retrieval_text == "A > B > C\n\nBody under C."


def test_sibling_heading2_sections_do_not_nest():
    data = _docx_bytes(
        lambda d: (
            _add_heading(d, "Authentication", 1),
            _add_para(d, "Auth overview."),
            _add_heading(d, "Access Tokens", 2),
            _add_para(d, "Access body."),
            _add_heading(d, "Refresh Tokens", 2),
            _add_para(d, "Refresh body."),
        )
    )
    canonical = DOCXAdapter().parse(data, source_uri="inline://s", title_fallback="doc")
    root = canonical.sections[0]
    assert root.heading_path == ("Authentication",)
    assert [c.heading_path for c in root.children] == [
        ("Authentication", "Access Tokens"),
        ("Authentication", "Refresh Tokens"),
    ]
    assert all(c.children == () for c in root.children)
    units = materialize_units(canonical, document_id="doc:test")
    assert [u.heading_path for u in units] == [
        ("Authentication",),
        ("Authentication", "Access Tokens"),
        ("Authentication", "Refresh Tokens"),
    ]


def test_paragraph_body_collection():
    data = _docx_bytes(
        lambda d: (
            _add_heading(d, "Simple Doc", 1),
            _add_para(d, "Hello paragraph."),
            _add_para(d, "Second paragraph."),
        )
    )
    units = build_units_from_docx(document_id="doc:test", source=data)
    assert len(units) == 1
    assert units[0].heading_path == ("Simple Doc",)
    assert units[0].text == "Hello paragraph.\n\nSecond paragraph."
    assert units[0].char_start is None and units[0].char_end is None


def test_preamble_before_first_heading():
    data = _docx_bytes(
        lambda d: (
            _add_para(d, "Leading text before headings."),
            _add_heading(d, "First", 1),
            _add_para(d, "Body one."),
        )
    )
    units = build_units_from_docx(document_id="doc:test", source=data)
    assert [u.heading_path for u in units] == [(), ("First",)]
    assert units[0].section_id == "sec:preamble"
    assert units[0].text == "Leading text before headings."
    assert units[1].text == "Body one."


def test_no_heading_document_is_flat():
    data = _docx_bytes(
        lambda d: (
            _add_para(d, "Just a plain paragraph."),
            _add_para(d, "And another."),
        )
    )
    units = build_units_from_docx(document_id="doc:test", source=data)
    assert len(units) == 1
    assert units[0].heading_path == ()
    assert units[0].section_id == "sec:preamble"
    assert units[0].text == "Just a plain paragraph.\n\nAnd another."
    assert units[0].metadata["heading_level"] == 0


def test_multiple_heading_levels():
    data = _docx_bytes(
        lambda d: (
            _add_heading(d, "L1", 1),
            _add_para(d, "one"),
            _add_heading(d, "L2", 2),
            _add_para(d, "two"),
            _add_heading(d, "L3", 3),
            _add_para(d, "three"),
            _add_heading(d, "L4", 4),
            _add_para(d, "four"),
            _add_heading(d, "L5", 5),
            _add_para(d, "five"),
            _add_heading(d, "L6", 6),
            _add_para(d, "six"),
        )
    )
    units = build_units_from_docx(document_id="doc:test", source=data)
    assert [u.heading_path for u in units] == [
        ("L1",),
        ("L1", "L2"),
        ("L1", "L2", "L3"),
        ("L1", "L2", "L3", "L4"),
        ("L1", "L2", "L3", "L4", "L5"),
        ("L1", "L2", "L3", "L4", "L5", "L6"),
    ]
    assert [u.text for u in units] == ["one", "two", "three", "four", "five", "six"]


def test_list_item_text_preservation():
    def build(document: DocxDocument) -> None:
        _add_heading(document, "Steps", 1)
        _add_para(document, "Overview.")
        # Prefer List Bullet when available; fall back to Normal.
        style = "List Bullet"
        try:
            document.styles[style]
        except KeyError:
            style = "Normal"
        _add_para(document, "First step", style=style)
        _add_para(document, "Second step", style=style)

    data = _docx_bytes(build)
    units = build_units_from_docx(document_id="doc:test", source=data)
    assert len(units) == 1
    assert units[0].heading_path == ("Steps",)
    assert units[0].text == "Overview.\n\nFirst step\n\nSecond step"


def test_basic_table_text_serialization():
    def build(document: DocxDocument) -> None:
        _add_heading(document, "Metrics", 1)
        _add_para(document, "Summary.")
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Name"
        table.cell(0, 1).text = "Value"
        table.cell(1, 0).text = "Latency"
        table.cell(1, 1).text = "12ms"

    data = _docx_bytes(build)
    units = build_units_from_docx(document_id="doc:test", source=data)
    assert len(units) == 1
    assert "Summary." in units[0].text
    assert "Name | Value" in units[0].text
    assert "Latency | 12ms" in units[0].text


def test_shared_materializer_compatibility_and_ordering():
    data = _docx_bytes(
        lambda d: (
            _add_heading(d, "A", 1),
            _add_para(d, "Intro under A."),
            _add_heading(d, "B", 2),
            _add_para(d, "Body under B."),
            _add_heading(d, "C", 3),
            _add_para(d, "Body under C."),
        )
    )
    units = build_units_from_docx(document_id="doc:test", source=data)
    assert [u.ordinal for u in units] == list(range(len(units)))
    assert units[0].evidence_id == "ev:doc:test:u0000"
    assert units[0].retrieval_text.startswith("A\n\n")


def test_reingest_deterministic_identity(ev_store, tmp_path):
    data = _docx_bytes(
        lambda d: (
            _add_heading(d, "A", 1),
            _add_para(d, "Intro under A."),
            _add_heading(d, "B", 2),
            _add_para(d, "Body under B."),
        )
    )
    path = tmp_path / "stable.docx"
    path.write_bytes(data)
    first = ingest_docx(path, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00")
    second = ingest_docx(path, ev_store, clock=lambda: "2030-01-02T00:00:00+00:00")

    assert first.document.source_type == SOURCE_TYPE_DOCX
    assert first.document.source_uri == canonicalize_source_uri(path)
    assert first.document.document_id == document_id_for_uri(first.document.source_uri)
    assert first.document.title == "A"
    assert first.document.metadata["adapter"] == "docx"
    assert first.document.metadata["parser"] == "python-docx"
    assert second.replaced is True
    assert first.document.document_id == second.document.document_id
    assert [u.evidence_id for u in first.units] == [u.evidence_id for u in second.units]
    assert [u.evidence_id for u in first.units] == [
        evidence_id_for(first.document.document_id, i) for i in range(len(first.units))
    ]
    assert [u.heading_path for u in first.units] == [u.heading_path for u in second.units]
    assert [u.text for u in first.units] == [u.text for u in second.units]
    assert len(ev_store.list_documents()) == 1


def test_equivalent_structure_with_markdown_and_html():
    """Same conceptual structure → same heading_path / body across adapters."""
    md = """# Project Overview

Some introduction.

## Authentication

Users authenticate using OAuth.

### Token Refresh

Access tokens expire after one hour.
"""
    html = """<!doctype html><html><body>
<h1>Project Overview</h1><p>Some introduction.</p>
<h2>Authentication</h2><p>Users authenticate using OAuth.</p>
<h3>Token Refresh</h3><p>Access tokens expire after one hour.</p>
</body></html>"""
    docx = _docx_bytes(
        lambda d: (
            _add_heading(d, "Project Overview", 1),
            _add_para(d, "Some introduction."),
            _add_heading(d, "Authentication", 2),
            _add_para(d, "Users authenticate using OAuth."),
            _add_heading(d, "Token Refresh", 3),
            _add_para(d, "Access tokens expire after one hour."),
        )
    )
    md_units = build_units_from_markdown(document_id="doc:md", source=md)
    html_units = build_units_from_html(document_id="doc:html", source=html)
    docx_units = build_units_from_docx(document_id="doc:docx", source=docx)
    assert [u.heading_path for u in md_units] == [u.heading_path for u in html_units]
    assert [u.heading_path for u in md_units] == [u.heading_path for u in docx_units]
    assert [u.text for u in md_units] == [u.text for u in html_units]
    assert [u.text for u in md_units] == [u.text for u in docx_units]


def test_markdown_and_html_adapters_still_present():
    assert isinstance(MarkdownAdapter(), FormatAdapter)
    assert isinstance(HTMLAdapter(), FormatAdapter)
    assert MarkdownAdapter().supports("x.md")
    assert HTMLAdapter().supports("x.html")
