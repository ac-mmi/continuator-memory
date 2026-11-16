"""HTML adapter: heading hierarchy → CanonicalDocument → shared materializer."""

from __future__ import annotations

import pytest

from continuator_memory.evidence.adapter import FormatAdapter
from continuator_memory.evidence.adapters.html import HTMLAdapter
from continuator_memory.evidence.adapters.markdown import MarkdownAdapter
from continuator_memory.evidence.ingest import (
    build_units_from_html,
    build_units_from_markdown,
    canonicalize_source_uri,
    document_id_for_uri,
    evidence_id_for,
    ingest_html,
)
from continuator_memory.evidence.materialize import materialize_units
from continuator_memory.evidence.models import SOURCE_TYPE_HTML
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.store import Store

SIMPLE = """<!doctype html>
<html>
<head><title>Simple Doc</title></head>
<body>
<h1>Simple Doc</h1>
<p>Hello paragraph.</p>
<p>Second paragraph.</p>
</body>
</html>
"""

NESTED = """<!doctype html>
<html><body>
<h1>A</h1>
<p>Intro under A.</p>
<h2>B</h2>
<p>Body under B.</p>
<h3>C</h3>
<p>Body under C.</p>
</body></html>
"""

PREAMBLE = """<!doctype html>
<html><body>
<p>Leading text before headings.</p>
<h1>First</h1>
<p>Body one.</p>
</body></html>
"""

NO_HEADING = """<!doctype html>
<html><body>
<p>Just a plain paragraph.</p>
<p>And another.</p>
</body></html>
"""

LISTS = """<!doctype html>
<html><body>
<h1>Steps</h1>
<p>Overview.</p>
<ul>
  <li>First step</li>
  <li>Second step</li>
</ul>
</body></html>
"""

NOISE = """<!doctype html>
<html><body>
<script>window.track = true;</script>
<style>.hidden { display: none; }</style>
<noscript>Enable JS</noscript>
<template><p>template body</p></template>
<h1>Clean</h1>
<p>Visible content.</p>
</body></html>
"""

EQUIV_MD = """# Project Overview

Some introduction.

## Authentication

Users authenticate using OAuth.

### Token Refresh

Access tokens expire after one hour.

## Database

The application uses PostgreSQL.
"""

EQUIV_HTML = """<!doctype html>
<html>
<head><title>Project Overview</title></head>
<body>
<h1>Project Overview</h1>
<p>Some introduction.</p>
<h2>Authentication</h2>
<p>Users authenticate using OAuth.</p>
<h3>Token Refresh</h3>
<p>Access tokens expire after one hour.</p>
<h2>Database</h2>
<p>The application uses PostgreSQL.</p>
</body>
</html>
"""


@pytest.fixture
def ev_store(tmp_path):
    product = Store(str(tmp_path / "evidence.db"))
    store = EvidenceStore(product.connection)
    yield store
    product.close()


def test_html_adapter_is_format_adapter():
    adapter = HTMLAdapter()
    assert isinstance(adapter, FormatAdapter)
    assert adapter.name == "html"
    assert SOURCE_TYPE_HTML in adapter.source_types
    assert adapter.supports("page.html")
    assert adapter.supports("page.htm")
    assert not adapter.supports("page.md")
    assert adapter.supports("unknown", sniff="<!doctype html><html>")


def test_single_heading_and_paragraphs():
    units = build_units_from_html(document_id="doc:test", source=SIMPLE)
    assert len(units) == 1
    unit = units[0]
    assert unit.heading_path == ("Simple Doc",)
    assert unit.text == "Hello paragraph.\n\nSecond paragraph."
    assert unit.retrieval_text == "Simple Doc\n\nHello paragraph.\n\nSecond paragraph."
    assert unit.section_id == "sec:simple-doc"
    assert unit.char_start is None and unit.char_end is None


def test_nested_h1_h2_h3_hierarchy_and_ancestry():
    adapter = HTMLAdapter()
    canonical = adapter.parse(NESTED, source_uri="inline://n", title_fallback="doc")
    assert len(canonical.sections) == 1
    root = canonical.sections[0]
    assert root.heading_path == ("A",)
    assert root.text == "Intro under A."
    assert len(root.children) == 1
    child = root.children[0]
    assert child.heading_path == ("A", "B")
    assert child.text == "Body under B."
    assert len(child.children) == 1
    assert child.children[0].heading_path == ("A", "B", "C")
    assert child.children[0].text == "Body under C."

    units = materialize_units(canonical, document_id="doc:test")
    assert [u.heading_path for u in units] == [("A",), ("A", "B"), ("A", "B", "C")]
    assert [u.text for u in units] == ["Intro under A.", "Body under B.", "Body under C."]
    assert [u.ordinal for u in units] == [0, 1, 2]
    assert units[2].section_id == "sec:a/b/c"
    assert units[2].retrieval_text == "A > B > C\n\nBody under C."


def test_content_before_first_heading():
    units = build_units_from_html(document_id="doc:test", source=PREAMBLE)
    assert [u.heading_path for u in units] == [(), ("First",)]
    assert units[0].section_id == "sec:preamble"
    assert units[0].text == "Leading text before headings."
    assert units[0].retrieval_text == "Leading text before headings."
    assert units[1].text == "Body one."


def test_no_heading_html_is_flat():
    units = build_units_from_html(document_id="doc:test", source=NO_HEADING)
    assert len(units) == 1
    assert units[0].heading_path == ()
    assert units[0].section_id == "sec:preamble"
    assert units[0].text == "Just a plain paragraph.\n\nAnd another."
    assert units[0].metadata["heading_level"] == 0


def test_list_content():
    units = build_units_from_html(document_id="doc:test", source=LISTS)
    assert len(units) == 1
    assert units[0].heading_path == ("Steps",)
    assert units[0].text == "Overview.\n\nFirst step\n\nSecond step"


def test_script_style_noscript_template_ignored():
    units = build_units_from_html(document_id="doc:test", source=NOISE)
    joined = "\n".join(u.text for u in units)
    assert "Visible content." in joined
    assert "window.track" not in joined
    assert "hidden" not in joined
    assert "Enable JS" not in joined
    assert "template body" not in joined


def test_reingest_deterministic_identity(ev_store, tmp_path):
    path = tmp_path / "stable.html"
    path.write_text(NESTED, encoding="utf-8")
    first = ingest_html(path, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00")
    second = ingest_html(path, ev_store, clock=lambda: "2030-01-02T00:00:00+00:00")

    assert first.document.source_type == SOURCE_TYPE_HTML
    assert first.document.source_uri == canonicalize_source_uri(path)
    assert first.document.document_id == document_id_for_uri(first.document.source_uri)
    assert first.document.title == "A"
    assert first.document.metadata["adapter"] == "html"
    assert second.replaced is True
    assert first.document.document_id == second.document.document_id
    assert [u.evidence_id for u in first.units] == [u.evidence_id for u in second.units]
    assert [u.evidence_id for u in first.units] == [
        evidence_id_for(first.document.document_id, i) for i in range(len(first.units))
    ]
    assert [u.heading_path for u in first.units] == [u.heading_path for u in second.units]
    assert [u.text for u in first.units] == [u.text for u in second.units]
    assert len(ev_store.list_documents()) == 1


def test_materialized_units_preserve_ordering():
    units = build_units_from_html(document_id="doc:test", source=NESTED)
    assert [u.ordinal for u in units] == list(range(len(units)))
    assert units[0].evidence_id == "ev:doc:test:u0000"
    assert units[-1].evidence_id == f"ev:doc:test:u{len(units)-1:04d}"


def test_equivalent_markdown_and_html_structure():
    """Same conceptual structure → same heading_path / body text (IDs may differ by format)."""
    md_units = build_units_from_markdown(document_id="doc:md", source=EQUIV_MD)
    html_units = build_units_from_html(document_id="doc:html", source=EQUIV_HTML)
    assert [u.heading_path for u in md_units] == [u.heading_path for u in html_units]
    assert [u.text for u in md_units] == [u.text for u in html_units]
    assert [u.section_id for u in md_units] == [u.section_id for u in html_units]
    assert [u.retrieval_text for u in md_units] == [u.retrieval_text for u in html_units]
    # Cross-format IDs intentionally need not match (different document_id / source).
    assert md_units[0].evidence_id != html_units[0].evidence_id


def test_markdown_adapter_unchanged_by_html_presence():
    assert isinstance(MarkdownAdapter(), FormatAdapter)
    units = build_units_from_markdown(document_id="doc:test", source=EQUIV_MD)
    assert units[0].heading_path == ("Project Overview",)
    assert units[0].char_start is not None
