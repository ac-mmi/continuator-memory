"""PDF FormatAdapter: digital-text extraction → CanonicalDocument.

Uses ``pypdf`` for page-level text extraction. Hierarchy is intentionally weak:
each page with extractable text becomes a flat ``CanonicalSection`` with
``heading_path=()`` and ``metadata["page_number"]`` (1-based). No typography-
based heading inference, no OCR, no layout reconstruction.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from typing import Any, BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from continuator_memory.evidence.canonical import CanonicalDocument, CanonicalSection
from continuator_memory.evidence.models import SOURCE_TYPE_PDF

_PDF_SUFFIXES = (".pdf",)
_PDF_MAGIC = b"%PDF-"


class PDFNoExtractableTextError(ValueError):
    """Raised when a PDF has no digital extractable text (OCR is not supported)."""


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_page_text(text: str) -> str:
    """Light cleanup only — preserve line breaks from the extractor when useful."""
    if not text:
        return ""
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    body = "\n".join(lines)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _open_pdf(source: str | bytes | Path | BinaryIO) -> tuple[PdfReader, bytes]:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        return PdfReader(io.BytesIO(data)), data
    if hasattr(source, "read"):
        data = source.read()  # type: ignore[union-attr]
        if isinstance(data, str):
            data = data.encode("latin-1", errors="replace")
        return PdfReader(io.BytesIO(bytes(data))), bytes(data)
    path = Path(source)  # type: ignore[arg-type]
    data = path.read_bytes()
    return PdfReader(io.BytesIO(data)), data


def _title_from_pdf(reader: PdfReader, fallback: str) -> str:
    meta = reader.metadata
    if meta is not None:
        raw = meta.title
        if raw:
            title = re.sub(r"\s+", " ", str(raw)).strip()
            if title:
                return title
    return fallback


def pdf_to_canonical_sections(
    reader: PdfReader,
) -> tuple[tuple[CanonicalSection, ...], dict[str, Any]]:
    """One flat section per page with extractable text; empty pages recorded only."""
    sections: list[CanonicalSection] = []
    empty_pages: list[int] = []
    page_count = len(reader.pages)

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            # Treat extractor failures on a page as no digital text for that page.
            raw = ""
        text = _normalize_page_text(raw)
        if not text:
            empty_pages.append(page_number)
            continue
        sections.append(
            CanonicalSection(
                section_id=f"sec:page-{page_number}",
                heading_path=(),
                text=text,
                heading_level=0,
                title=None,
                char_start=None,
                char_end=None,
                parent_section_id=None,
                children=(),
                metadata={
                    "page_number": page_number,
                    "pdf_role": "page",
                },
            )
        )

    parse_meta: dict[str, Any] = {
        "parser": "pypdf",
        "page_count": page_count,
        "pages_with_text": len(sections),
    }
    if empty_pages:
        parse_meta["empty_pages"] = empty_pages
    return tuple(sections), parse_meta


class PDFAdapter:
    """Parse digital-text PDFs into CanonicalDocument (page-level flat sections)."""

    name = "pdf"
    source_types = frozenset({SOURCE_TYPE_PDF})

    def supports(
        self,
        path_or_uri: str | Path,
        *,
        sniff: bytes | str | None = None,
    ) -> bool:
        text = str(path_or_uri).lower().split("?", 1)[0]
        if any(text.endswith(suffix) for suffix in _PDF_SUFFIXES):
            return True
        if sniff is None:
            return False
        sample = sniff if isinstance(sniff, (bytes, bytearray)) else sniff.encode(
            "latin-1", errors="replace"
        )
        return sample[:5] == _PDF_MAGIC

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

        if isinstance(source, str) and source.lower().endswith(".pdf") and Path(source).is_file():
            reader, raw = _open_pdf(source)
        elif isinstance(source, (bytes, bytearray)):
            reader, raw = _open_pdf(bytes(source))
        elif isinstance(source, str):
            path = Path(source)
            if path.is_file():
                reader, raw = _open_pdf(path)
            else:
                raise ValueError(
                    "PDFAdapter.parse expects PDF bytes or an existing .pdf path; "
                    "plain text is not a PDF"
                )
        else:
            reader, raw = _open_pdf(source)

        try:
            # Touch pages early to surface corrupt PDFs.
            _ = len(reader.pages)
        except PdfReadError as exc:
            raise ValueError(f"unreadable PDF: {exc}") from exc

        title = _title_from_pdf(reader, fallback=str(fallback))
        sections, parse_meta = pdf_to_canonical_sections(reader)
        if not sections:
            raise PDFNoExtractableTextError(
                "PDF has no extractable digital text "
                "(scanned/image-only PDFs require OCR, which is not supported)"
            )

        meta = {**meta, **parse_meta}
        return CanonicalDocument(
            source_uri=source_uri,
            source_type=SOURCE_TYPE_PDF,
            content_hash=_content_hash(raw),
            title=title,
            metadata=meta,
            sections=sections,
        )
