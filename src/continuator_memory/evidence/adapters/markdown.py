"""Markdown FormatAdapter: structure-first ATX headings → CanonicalDocument.

Reuses ``parse_markdown_sections``; does not change heading-stack semantics.
Emits a flat ordered section list (empty ``children``) with ``heading_path``
encoding ancestry — matching Phase-1 Markdown section order exactly.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from continuator_memory.evidence.canonical import CanonicalDocument, CanonicalSection
from continuator_memory.evidence.markdown import parse_markdown_sections, section_id_for
from continuator_memory.evidence.models import SOURCE_TYPE_MARKDOWN

_MD_SUFFIXES = (".md", ".markdown", ".mdown", ".mkd")


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _title_from_markdown(source: str, fallback: str) -> str:
    for line in source.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return fallback


def _parent_section_id(heading_path: tuple[str, ...]) -> str | None:
    if len(heading_path) <= 1:
        return None
    return section_id_for(heading_path[:-1])


def markdown_sections_to_canonical(
    source: str,
) -> tuple[CanonicalSection, ...]:
    """Map MarkdownSection list → flat CanonicalSection tuple (no nested children)."""
    sections: list[CanonicalSection] = []
    for md_section in parse_markdown_sections(source):
        path = md_section.heading_path
        title = path[-1] if path else None
        sections.append(
            CanonicalSection(
                section_id=section_id_for(path),
                heading_path=path,
                text=md_section.text,
                heading_level=md_section.heading_level,
                title=title,
                char_start=md_section.char_start,
                char_end=md_section.char_end,
                parent_section_id=_parent_section_id(path),
                children=(),
                metadata={},
            )
        )
    return tuple(sections)


class MarkdownAdapter:
    """Parse Markdown text into CanonicalDocument."""

    name = "markdown"
    source_types = frozenset({SOURCE_TYPE_MARKDOWN})

    def supports(
        self,
        path_or_uri: str | Path,
        *,
        sniff: bytes | str | None = None,
    ) -> bool:
        text = str(path_or_uri).lower().split("?", 1)[0]
        if any(text.endswith(suffix) for suffix in _MD_SUFFIXES):
            return True
        if sniff is None:
            return False
        sample = sniff.decode("utf-8", errors="replace") if isinstance(sniff, bytes) else sniff
        # Cheap ATX heading sniff — not authoritative.
        return bool(re.search(r"(?m)^#{1,6}\s+\S", sample))

    def parse(
        self,
        source: str | bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
        title_fallback: str | None = None,
    ) -> CanonicalDocument:
        if isinstance(source, bytes):
            text = source.decode("utf-8")
        else:
            text = source
        meta = dict(metadata or {})
        fallback = title_fallback
        if fallback is None:
            original = meta.get("original_path")
            if original:
                fallback = Path(str(original)).stem
            else:
                fallback = Path(source_uri).stem or "document"
        title = _title_from_markdown(text, fallback=str(fallback))
        return CanonicalDocument(
            source_uri=source_uri,
            source_type=SOURCE_TYPE_MARKDOWN,
            content_hash=_content_hash(text),
            title=title,
            metadata=meta,
            sections=markdown_sections_to_canonical(text),
        )
