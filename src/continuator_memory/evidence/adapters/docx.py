"""DOCX FormatAdapter: Word heading styles → CanonicalDocument.

Uses ``python-docx``. Primary structure signal: Heading 1–6 styles (same stack
rules as Markdown/HTML). Body: normal paragraphs, list item text, and simple
table cell serialization. No tracked changes, comments, headers/footers, or
layout reconstruction.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from continuator_memory.evidence.canonical import CanonicalDocument, CanonicalSection
from continuator_memory.evidence.markdown import section_id_for
from continuator_memory.evidence.models import SOURCE_TYPE_DOCX

_DOCX_SUFFIXES = (".docx",)
_HEADING_STYLE = re.compile(r"^Heading\s+([1-6])$", re.IGNORECASE)


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _parent_section_id(heading_path: tuple[str, ...]) -> str | None:
    if len(heading_path) <= 1:
        return None
    return section_id_for(heading_path[:-1])


@dataclass
class _SectionBuilder:
    heading_level: int
    heading_path: tuple[str, ...]
    title: str | None
    body_parts: list[str] = field(default_factory=list)
    children: list["_SectionBuilder"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_body(self, text: str) -> None:
        cleaned = (text or "").strip()
        if cleaned:
            self.body_parts.append(cleaned)

    def to_canonical(self) -> CanonicalSection | None:
        text = "\n\n".join(self.body_parts).strip()
        children = tuple(
            child
            for child in (c.to_canonical() for c in self.children)
            if child is not None
        )
        if not text and not children:
            return None
        path = self.heading_path
        return CanonicalSection(
            section_id=section_id_for(path),
            heading_path=path,
            text=text,
            heading_level=self.heading_level,
            title=self.title,
            char_start=None,
            char_end=None,
            parent_section_id=_parent_section_id(path),
            children=children,
            metadata=dict(self.metadata),
        )


def _heading_level(paragraph: Paragraph) -> int | None:
    style = paragraph.style
    if style is None:
        return None
    name = (style.name or "").strip()
    match = _HEADING_STYLE.match(name)
    if match:
        return int(match.group(1))
    return None


def _paragraph_text(paragraph: Paragraph) -> str:
    return _normalize_text(paragraph.text or "")


def _table_text(table: Table) -> str:
    """Serialize a table into readable ordered text (rows as ``cell | cell``)."""
    rows: list[str] = []
    for row in table.rows:
        cells = [_normalize_text(cell.text or "") for cell in row.cells]
        # python-docx repeats merged-cell text; collapse adjacent duplicates lightly.
        collapsed: list[str] = []
        for cell in cells:
            if not cell:
                continue
            if collapsed and collapsed[-1] == cell:
                continue
            collapsed.append(cell)
        if collapsed:
            rows.append(" | ".join(collapsed))
    return "\n".join(rows)


def _iter_block_items(document: DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield paragraphs and tables in document-body order."""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _open_docx(source: str | bytes | Path | BinaryIO) -> tuple[DocxDocument, bytes]:
    """Open a DOCX and return (document, raw_bytes for hashing)."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        return DocxDocument(io.BytesIO(data)), data
    if hasattr(source, "read"):
        data = source.read()  # type: ignore[union-attr]
        if isinstance(data, str):
            data = data.encode("utf-8")
        return DocxDocument(io.BytesIO(data)), bytes(data)
    path = Path(source)  # type: ignore[arg-type]
    data = path.read_bytes()
    return DocxDocument(io.BytesIO(data)), data


def _title_from_docx(document: DocxDocument, fallback: str) -> str:
    core_title = _normalize_text(document.core_properties.title or "")
    if core_title:
        return core_title
    for block in _iter_block_items(document):
        if isinstance(block, Paragraph) and _heading_level(block) == 1:
            title = _paragraph_text(block)
            if title:
                return title
    for block in _iter_block_items(document):
        if isinstance(block, Paragraph):
            text = _paragraph_text(block)
            if text:
                return text[:120]
    return fallback


def docx_to_canonical_sections(
    document: DocxDocument,
) -> tuple[tuple[CanonicalSection, ...], dict[str, Any]]:
    """Parse a python-docx Document into nested CanonicalSections."""
    roots: list[_SectionBuilder] = []
    stack: list[_SectionBuilder] = []
    preamble = _SectionBuilder(
        heading_level=0,
        heading_path=(),
        title=None,
        metadata={"docx_role": "preamble"},
    )

    def current_builder() -> _SectionBuilder:
        return stack[-1] if stack else preamble

    for block in _iter_block_items(document):
        if isinstance(block, Table):
            text = _table_text(block)
            if text:
                current_builder().add_body(text)
            continue

        level = _heading_level(block)
        text = _paragraph_text(block)
        if level is not None:
            title = text
            if not title:
                continue
            while stack and stack[-1].heading_level >= level:
                stack.pop()
            path = (stack[-1].heading_path + (title,)) if stack else (title,)
            node = _SectionBuilder(
                heading_level=level,
                heading_path=path,
                title=title,
                metadata={
                    "docx_style": f"Heading {level}",
                    "docx_heading_level": level,
                },
            )
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
            continue

        if text:
            current_builder().add_body(text)

    sections: list[CanonicalSection] = []
    preamble_section = preamble.to_canonical()
    if preamble_section is not None:
        sections.append(preamble_section)
    for root_builder in roots:
        canonical = root_builder.to_canonical()
        if canonical is not None:
            sections.append(canonical)

    parse_meta = {"parser": "python-docx"}
    return tuple(sections), parse_meta


class DOCXAdapter:
    """Parse DOCX into CanonicalDocument.

    ``parse`` accepts raw ``.docx`` bytes, or a filesystem path as ``str``/``Path``
    via the ``source`` argument (binary ZIP package — not UTF-8 text).
    """

    name = "docx"
    source_types = frozenset({SOURCE_TYPE_DOCX})

    def supports(
        self,
        path_or_uri: str | Path,
        *,
        sniff: bytes | str | None = None,
    ) -> bool:
        text = str(path_or_uri).lower().split("?", 1)[0]
        if any(text.endswith(suffix) for suffix in _DOCX_SUFFIXES):
            return True
        if sniff is None:
            return False
        sample = sniff if isinstance(sniff, (bytes, bytearray)) else sniff.encode("utf-8", errors="replace")
        # DOCX is a ZIP package; PK\x03\x04 magic.
        return len(sample) >= 4 and sample[:2] == b"PK"

    def parse(
        self,
        source: str | bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
        title_fallback: str | None = None,
    ) -> CanonicalDocument:
        meta = dict(metadata or {})
        fallback = title_fallback
        if fallback is None:
            original = meta.get("original_path")
            if original:
                fallback = Path(str(original)).stem
            else:
                fallback = Path(source_uri).stem or "document"

        # str that looks like a path to an existing .docx → read bytes from disk.
        if isinstance(source, str) and source.lower().endswith(".docx") and Path(source).is_file():
            document, raw = _open_docx(source)
        elif isinstance(source, (bytes, bytearray)):
            document, raw = _open_docx(bytes(source))
        elif isinstance(source, str):
            # Not valid as UTF-8 DOCX content; treat as path if present, else error.
            path = Path(source)
            if path.is_file():
                document, raw = _open_docx(path)
            else:
                raise ValueError(
                    "DOCXAdapter.parse expects .docx bytes or an existing .docx path; "
                    "plain text is not a DOCX package"
                )
        else:
            document, raw = _open_docx(source)

        title = _title_from_docx(document, fallback=str(fallback))
        sections, parse_meta = docx_to_canonical_sections(document)
        meta = {**meta, **parse_meta}
        core_title = _normalize_text(document.core_properties.title or "")
        if core_title:
            meta.setdefault("docx_core_title", core_title)

        return CanonicalDocument(
            source_uri=source_uri,
            source_type=SOURCE_TYPE_DOCX,
            content_hash=_content_hash(raw),
            title=title,
            metadata=meta,
            sections=sections,
        )
