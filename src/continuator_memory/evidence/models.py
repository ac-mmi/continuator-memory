"""Canonical document and evidence-unit models for the universal evidence layer.

Parallel to V2 ``Memory`` / ``MemorySource``. Conversation memories are not migrated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SOURCE_TYPE_MARKDOWN = "markdown"
SOURCE_TYPE_TEXT = "text"
SOURCE_TYPE_PDF = "pdf"
SOURCE_TYPE_HTML = "html"
SOURCE_TYPE_DOCX = "docx"
SOURCE_TYPE_CODE = "code"
SOURCE_TYPE_CONVERSATION = "conversation"

# Markdown + HTML + DOCX + PDF implemented; other constants reserve the type space.
IMPLEMENTED_SOURCE_TYPES = frozenset(
    {SOURCE_TYPE_MARKDOWN, SOURCE_TYPE_HTML, SOURCE_TYPE_DOCX, SOURCE_TYPE_PDF}
)


@dataclass(frozen=True)
class Document:
    """One ingested source document with stable identity and content versioning."""

    document_id: str
    source_uri: str
    source_type: str
    content_hash: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_uri": self.source_uri,
            "source_type": self.source_type,
            "content_hash": self.content_hash,
            "title": self.title,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class EvidenceUnit:
    """Universal retrieval/evidence atom with structure-preserving provenance.

    Provenance chain::

        evidence_id → document_id → source_uri
        heading_path / section_id → structural location
        char_start / char_end → span in the source text (when available)
    """

    evidence_id: str
    document_id: str
    ordinal: int
    text: str
    retrieval_text: str
    heading_path: tuple[str, ...]
    section_id: str
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "ordinal": self.ordinal,
            "text": self.text,
            "retrieval_text": self.retrieval_text,
            "heading_path": list(self.heading_path),
            "section_id": self.section_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IngestedDocument:
    """Result of ingesting one source into Document + EvidenceUnit rows."""

    document: Document
    units: tuple[EvidenceUnit, ...]
    replaced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "units": [u.to_dict() for u in self.units],
            "replaced": self.replaced,
        }


@dataclass(frozen=True)
class EvidenceHit:
    """One search hit over EvidenceUnits (BM25 Phase 1)."""

    unit: EvidenceUnit
    document: Document
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "evidence_id": self.unit.evidence_id,
            "document_id": self.document.document_id,
            "source_uri": self.document.source_uri,
            "heading_path": list(self.unit.heading_path),
            "section_id": self.unit.section_id,
            "char_start": self.unit.char_start,
            "char_end": self.unit.char_end,
            "text": self.unit.text,
            "retrieval_text": self.unit.retrieval_text,
        }
