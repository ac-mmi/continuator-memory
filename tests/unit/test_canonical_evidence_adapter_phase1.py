"""Golden compatibility: Markdown → CanonicalDocument → materializer ≡ Phase-1 semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from continuator_memory.evidence.adapter import FormatAdapter
from continuator_memory.evidence.adapters.markdown import MarkdownAdapter
from continuator_memory.evidence.canonical import CanonicalDocument, CanonicalSection
from continuator_memory.evidence.ingest import (
    build_units_from_markdown,
    canonicalize_source_uri,
    document_id_for_uri,
    evidence_id_for,
    ingest_markdown,
)
from continuator_memory.evidence.materialize import materialize, materialize_units
from continuator_memory.evidence.models import SOURCE_TYPE_MARKDOWN
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.store import Store

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

PREAMBLE = """Leading text before headings.

# First

Body one.
"""

NO_HEADING = """Just a plain paragraph.

And another.
"""

LONG_SPLIT = "# H\n\n" + ("para one.\n\n" * 5) + ("x" * 50)

# Frozen Phase-1 golden outputs (captured before adapter migration).
GOLDEN_NESTED = [
    {
        "evidence_id": "ev:doc:test:u0000",
        "ordinal": 0,
        "text": "Intro under A.",
        "retrieval_text": "A\n\nIntro under A.",
        "heading_path": ["A"],
        "section_id": "sec:a",
        "char_start": 5,
        "char_end": 19,
        "metadata": {"heading_level": 1, "piece_count": 1, "piece_index": 0},
    },
    {
        "evidence_id": "ev:doc:test:u0001",
        "ordinal": 1,
        "text": "Body under B.",
        "retrieval_text": "A > B\n\nBody under B.",
        "heading_path": ["A", "B"],
        "section_id": "sec:a/b",
        "char_start": 27,
        "char_end": 40,
        "metadata": {"heading_level": 2, "piece_count": 1, "piece_index": 0},
    },
    {
        "evidence_id": "ev:doc:test:u0002",
        "ordinal": 2,
        "text": "Body under C.",
        "retrieval_text": "A > B > C\n\nBody under C.",
        "heading_path": ["A", "B", "C"],
        "section_id": "sec:a/b/c",
        "char_start": 49,
        "char_end": 62,
        "metadata": {"heading_level": 3, "piece_count": 1, "piece_index": 0},
    },
]

GOLDEN_EXAMPLE = [
    {
        "evidence_id": "ev:doc:test:u0000",
        "ordinal": 0,
        "text": "Some introduction.",
        "retrieval_text": "Project Overview\n\nSome introduction.",
        "heading_path": ["Project Overview"],
        "section_id": "sec:project-overview",
        "char_start": 20,
        "char_end": 38,
        "metadata": {"heading_level": 1, "piece_count": 1, "piece_index": 0},
    },
    {
        "evidence_id": "ev:doc:test:u0001",
        "ordinal": 1,
        "text": "Users authenticate using OAuth.",
        "retrieval_text": "Project Overview > Authentication\n\nUsers authenticate using OAuth.",
        "heading_path": ["Project Overview", "Authentication"],
        "section_id": "sec:project-overview/authentication",
        "char_start": 59,
        "char_end": 90,
        "metadata": {"heading_level": 2, "piece_count": 1, "piece_index": 0},
    },
    {
        "evidence_id": "ev:doc:test:u0002",
        "ordinal": 2,
        "text": "Access tokens expire after one hour.",
        "retrieval_text": (
            "Project Overview > Authentication > Token Refresh\n\n"
            "Access tokens expire after one hour."
        ),
        "heading_path": ["Project Overview", "Authentication", "Token Refresh"],
        "section_id": "sec:project-overview/authentication/token-refresh",
        "char_start": 111,
        "char_end": 147,
        "metadata": {"heading_level": 3, "piece_count": 1, "piece_index": 0},
    },
    {
        "evidence_id": "ev:doc:test:u0003",
        "ordinal": 3,
        "text": "The application uses PostgreSQL.",
        "retrieval_text": "Project Overview > Database\n\nThe application uses PostgreSQL.",
        "heading_path": ["Project Overview", "Database"],
        "section_id": "sec:project-overview/database",
        "char_start": 162,
        "char_end": 194,
        "metadata": {"heading_level": 2, "piece_count": 1, "piece_index": 0},
    },
]

GOLDEN_PREAMBLE = [
    {
        "evidence_id": "ev:doc:test:u0000",
        "ordinal": 0,
        "text": "Leading text before headings.",
        "retrieval_text": "Leading text before headings.",
        "heading_path": [],
        "section_id": "sec:preamble",
        "char_start": 0,
        "char_end": 29,
        "metadata": {"heading_level": 0, "piece_count": 1, "piece_index": 0},
    },
    {
        "evidence_id": "ev:doc:test:u0001",
        "ordinal": 1,
        "text": "Body one.",
        "retrieval_text": "First\n\nBody one.",
        "heading_path": ["First"],
        "section_id": "sec:first",
        "char_start": 40,
        "char_end": 49,
        "metadata": {"heading_level": 1, "piece_count": 1, "piece_index": 0},
    },
]

GOLDEN_NO_HEADING = [
    {
        "evidence_id": "ev:doc:test:u0000",
        "ordinal": 0,
        "text": "Just a plain paragraph.\n\nAnd another.",
        "retrieval_text": "Just a plain paragraph.\n\nAnd another.",
        "heading_path": [],
        "section_id": "sec:preamble",
        "char_start": 0,
        "char_end": 37,
        "metadata": {"heading_level": 0, "piece_count": 1, "piece_index": 0},
    },
]

GOLDEN_LONG_SPLIT = [
    {
        "evidence_id": "ev:doc:test:u0000",
        "ordinal": 0,
        "text": "para one.\n\npara one.",
        "retrieval_text": "H\n\npara one.\n\npara one.",
        "heading_path": ["H"],
        "section_id": "sec:h#p00",
        "char_start": None,
        "char_end": None,
        "metadata": {"heading_level": 1, "piece_count": 5, "piece_index": 0},
    },
    {
        "evidence_id": "ev:doc:test:u0001",
        "ordinal": 1,
        "text": "para one.\n\npara one.",
        "retrieval_text": "H\n\npara one.\n\npara one.",
        "heading_path": ["H"],
        "section_id": "sec:h#p01",
        "char_start": None,
        "char_end": None,
        "metadata": {"heading_level": 1, "piece_count": 5, "piece_index": 1},
    },
    {
        "evidence_id": "ev:doc:test:u0002",
        "ordinal": 2,
        "text": "para one.",
        "retrieval_text": "H\n\npara one.",
        "heading_path": ["H"],
        "section_id": "sec:h#p02",
        "char_start": None,
        "char_end": None,
        "metadata": {"heading_level": 1, "piece_count": 5, "piece_index": 2},
    },
    {
        "evidence_id": "ev:doc:test:u0003",
        "ordinal": 3,
        "text": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "retrieval_text": "H\n\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "heading_path": ["H"],
        "section_id": "sec:h#p03",
        "char_start": None,
        "char_end": None,
        "metadata": {"heading_level": 1, "piece_count": 5, "piece_index": 3},
    },
    {
        "evidence_id": "ev:doc:test:u0004",
        "ordinal": 4,
        "text": "xxxxxxxxxxxxxxxxxxxx",
        "retrieval_text": "H\n\nxxxxxxxxxxxxxxxxxxxx",
        "heading_path": ["H"],
        "section_id": "sec:h#p04",
        "char_start": None,
        "char_end": None,
        "metadata": {"heading_level": 1, "piece_count": 5, "piece_index": 4},
    },
]


def _unit_dict(unit) -> dict:
    return {
        "evidence_id": unit.evidence_id,
        "ordinal": unit.ordinal,
        "text": unit.text,
        "retrieval_text": unit.retrieval_text,
        "heading_path": list(unit.heading_path),
        "section_id": unit.section_id,
        "char_start": unit.char_start,
        "char_end": unit.char_end,
        "metadata": dict(unit.metadata),
    }


def _assert_units_match_golden(source: str, golden: list[dict], *, max_chars=None) -> None:
    units = build_units_from_markdown(
        document_id="doc:test", source=source, max_section_chars=max_chars
    )
    got = [_unit_dict(u) for u in units]
    if got != golden:
        raise AssertionError(
            "Markdown persisted unit semantics drifted from Phase-1 golden.\n"
            f"expected={json.dumps(golden, indent=2)}\n"
            f"got={json.dumps(got, indent=2)}"
        )


@pytest.fixture
def ev_store(tmp_path):
    product = Store(str(tmp_path / "evidence.db"))
    store = EvidenceStore(product.connection)
    yield store
    product.close()


def test_markdown_adapter_is_format_adapter():
    adapter = MarkdownAdapter()
    assert isinstance(adapter, FormatAdapter)
    assert adapter.name == "markdown"
    assert SOURCE_TYPE_MARKDOWN in adapter.source_types
    assert adapter.supports("notes.md")
    assert not adapter.supports("notes.pdf")


def test_golden_nested_headings():
    _assert_units_match_golden(NESTED, GOLDEN_NESTED)


def test_golden_example_document():
    _assert_units_match_golden(EXAMPLE, GOLDEN_EXAMPLE)


def test_golden_preamble_and_heading():
    _assert_units_match_golden(PREAMBLE, GOLDEN_PREAMBLE)


def test_golden_no_heading_flat():
    _assert_units_match_golden(NO_HEADING, GOLDEN_NO_HEADING)


def test_golden_max_chars_split():
    _assert_units_match_golden(LONG_SPLIT, GOLDEN_LONG_SPLIT, max_chars=30)


def test_adapter_parse_then_materialize_matches_build_units():
    adapter = MarkdownAdapter()
    canonical = adapter.parse(
        EXAMPLE,
        source_uri="inline://doc:test",
        metadata={"adapter": "markdown"},
        title_fallback="document",
    )
    assert isinstance(canonical, CanonicalDocument)
    assert canonical.title == "Project Overview"
    assert canonical.source_type == SOURCE_TYPE_MARKDOWN
    assert all(isinstance(s, CanonicalSection) for s in canonical.sections)
    assert all(s.children == () for s in canonical.sections)

    via_adapter = materialize_units(canonical, document_id="doc:test")
    via_helper = build_units_from_markdown(document_id="doc:test", source=EXAMPLE)
    assert [_unit_dict(u) for u in via_adapter] == [_unit_dict(u) for u in via_helper]
    assert [_unit_dict(u) for u in via_adapter] == GOLDEN_EXAMPLE


def test_nested_children_materialize_same_as_flat_order():
    """DFS over children must preserve reading order when adapters nest."""
    leaf_c = CanonicalSection(
        section_id="sec:a/b/c",
        heading_path=("A", "B", "C"),
        text="Body under C.",
        heading_level=3,
        title="C",
        char_start=49,
        char_end=62,
        parent_section_id="sec:a/b",
    )
    leaf_b = CanonicalSection(
        section_id="sec:a/b",
        heading_path=("A", "B"),
        text="Body under B.",
        heading_level=2,
        title="B",
        char_start=27,
        char_end=40,
        parent_section_id="sec:a",
        children=(leaf_c,),
    )
    root_a = CanonicalSection(
        section_id="sec:a",
        heading_path=("A",),
        text="Intro under A.",
        heading_level=1,
        title="A",
        char_start=5,
        char_end=19,
        children=(leaf_b,),
    )
    canonical = CanonicalDocument(
        source_uri="inline://nested",
        source_type=SOURCE_TYPE_MARKDOWN,
        content_hash="x",
        title="A",
        sections=(root_a,),
    )
    units = materialize_units(canonical, document_id="doc:test")
    assert [_unit_dict(u) for u in units] == GOLDEN_NESTED


def test_ingest_markdown_document_identity_and_reingest(ev_store, tmp_path):
    path = tmp_path / "stable.md"
    path.write_text(NESTED, encoding="utf-8")
    first = ingest_markdown(path, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00")
    second = ingest_markdown(path, ev_store, clock=lambda: "2030-01-02T00:00:00+00:00")

    assert first.document.source_uri == canonicalize_source_uri(path)
    assert first.document.document_id == document_id_for_uri(first.document.source_uri)
    assert first.document.source_type == SOURCE_TYPE_MARKDOWN
    assert first.document.title == "A"
    assert first.document.metadata["adapter"] == "markdown"
    assert second.replaced is True
    assert first.document.document_id == second.document.document_id
    assert [u.evidence_id for u in first.units] == [u.evidence_id for u in second.units]
    assert [u.evidence_id for u in first.units] == [
        evidence_id_for(first.document.document_id, i) for i in range(len(first.units))
    ]
    assert [_unit_dict(u) for u in first.units] == [
        {
            **item,
            "evidence_id": evidence_id_for(first.document.document_id, item["ordinal"]),
        }
        for item in GOLDEN_NESTED
    ]
    assert len(ev_store.list_documents()) == 1
    assert len(ev_store.get_units(first.document.document_id)) == len(first.units)


def test_ingest_via_materialize_matches_store_rows(ev_store, tmp_path):
    path = tmp_path / "example.md"
    path.write_text(EXAMPLE, encoding="utf-8")
    result = ingest_markdown(path, ev_store, clock=lambda: "2030-01-01T00:00:00+00:00")
    adapter = MarkdownAdapter()
    canonical = adapter.parse(
        EXAMPLE,
        source_uri=result.document.source_uri,
        metadata=dict(result.document.metadata),
        title_fallback=path.stem,
    )
    document, units = materialize(
        canonical,
        document_id=result.document.document_id,
        created_at="2030-01-01T00:00:00+00:00",
    )
    assert document.to_dict() == result.document.to_dict()
    assert [_unit_dict(u) for u in units] == [_unit_dict(u) for u in result.units]
    # Spans still resolve into source text.
    for unit in result.units:
        assert EXAMPLE[unit.char_start : unit.char_end].strip() == unit.text


def test_empty_hierarchy_allowed_for_future_pdf():
    """Flat section with empty heading_path must materialize without fake headings."""
    canonical = CanonicalDocument(
        source_uri="pdf://future",
        source_type="pdf",
        content_hash="abc",
        title=None,
        sections=(
            CanonicalSection(
                section_id="sec:preamble",
                heading_path=(),
                text="page body without headings",
                heading_level=0,
            ),
        ),
    )
    units = materialize_units(canonical, document_id="doc:pdf")
    assert len(units) == 1
    assert units[0].heading_path == ()
    assert units[0].section_id == "sec:preamble"
    assert units[0].retrieval_text == "page body without headings"
    assert units[0].metadata["heading_level"] == 0
