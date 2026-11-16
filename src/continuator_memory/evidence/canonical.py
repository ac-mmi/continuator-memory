"""In-memory canonical parse model for format adapters.

Persisted storage remains ``Document`` + ``EvidenceUnit``. Sections are not
persisted; adapters emit ``CanonicalDocument`` / ``CanonicalSection`` trees
(or flat lists) that the shared materializer converts into EvidenceUnits.

Hierarchy is optional: empty ``heading_path`` and leaf-only flat section lists
are first-class so formats without recoverable structure (e.g. digital PDFs)
need not invent fake headings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass(frozen=True)
class CanonicalSection:
    """One structural section in an adapter's parse output.

    May be a flat leaf (``children`` empty) or a node with nested children.
    ``heading_path`` may be empty when hierarchy is absent or for preamble.
    """

    section_id: str
    heading_path: tuple[str, ...]
    text: str
    heading_level: int = 0
    title: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    parent_section_id: str | None = None
    children: tuple["CanonicalSection", ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalDocument:
    """Format-agnostic parse result before EvidenceUnit materialization."""

    source_uri: str
    source_type: str
    content_hash: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: tuple[CanonicalSection, ...] = ()


def iter_sections_depth_first(
    sections: tuple[CanonicalSection, ...] | list[CanonicalSection],
) -> Iterator[CanonicalSection]:
    """Yield sections in document reading order (pre-order DFS)."""
    for section in sections:
        yield section
        if section.children:
            yield from iter_sections_depth_first(section.children)
