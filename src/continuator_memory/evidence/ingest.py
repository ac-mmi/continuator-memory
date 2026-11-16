"""Ingest Markdown / HTML / DOCX / PDF into Document + EvidenceUnit rows.

Public facades::

    ingest_markdown / ingest_html / ingest_docx / ingest_pdf
        → FormatAdapter.parse → CanonicalDocument
        → materialize → Document + EvidenceUnit → EvidenceStore

    ingest_file
        → extension route → one of the facades above
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from continuator_memory.evidence.adapters.docx import DOCXAdapter
from continuator_memory.evidence.adapters.html import HTMLAdapter
from continuator_memory.evidence.adapters.markdown import MarkdownAdapter
from continuator_memory.evidence.adapters.pdf import PDFAdapter
from continuator_memory.evidence.materialize import evidence_id_for, materialize
from continuator_memory.evidence.models import EvidenceUnit, IngestedDocument
from continuator_memory.evidence.store import EvidenceStore

# Extension point: None means keep whole structural sections (Phase 1 default).
DEFAULT_MAX_SECTION_CHARS: int | None = None

# Case-insensitive extension → format name used by ingest_file.
EVIDENCE_EXTENSION_ROUTES: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".docx": "docx",
    ".pdf": "pdf",
}

SUPPORTED_EVIDENCE_EXTENSIONS: tuple[str, ...] = tuple(
    sorted(EVIDENCE_EXTENSION_ROUTES.keys())
)

# Re-export for existing call sites / tests.
__all__ = [
    "DEFAULT_MAX_SECTION_CHARS",
    "EVIDENCE_EXTENSION_ROUTES",
    "SUPPORTED_EVIDENCE_EXTENSIONS",
    "UnsupportedEvidenceFormatError",
    "build_units_from_docx",
    "build_units_from_html",
    "build_units_from_markdown",
    "build_units_from_pdf",
    "canonicalize_source_uri",
    "content_hash_for",
    "document_id_for_uri",
    "evidence_id_for",
    "ingest_docx",
    "ingest_file",
    "ingest_html",
    "ingest_markdown",
    "ingest_pdf",
    "resolve_evidence_format",
]


class UnsupportedEvidenceFormatError(ValueError):
    """Raised when ingest_file cannot route a path by extension."""

    def __init__(
        self,
        path: str | Path,
        *,
        extension: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.path = str(path)
        self.extension = extension
        supported = ", ".join(SUPPORTED_EVIDENCE_EXTENSIONS)
        if reason:
            message = (
                f"{reason}: {self.path!r} "
                f"(supported extensions: {supported})"
            )
        elif not extension:
            message = (
                f"unsupported evidence file (no extension): {self.path!r} "
                f"(supported extensions: {supported})"
            )
        else:
            message = (
                f"unsupported evidence format {extension!r} for {self.path!r} "
                f"(supported extensions: {supported})"
            )
        super().__init__(message)


def resolve_evidence_format(source_path: str | Path) -> str:
    """Return format name (markdown/html/docx/pdf) from a file path extension.

    Raises ``UnsupportedEvidenceFormatError`` for directories, missing
    extensions, or unsupported extensions. Does not sniff file contents.
    """
    path = Path(source_path)
    if path.exists() and path.is_dir():
        raise UnsupportedEvidenceFormatError(
            path,
            extension=None,
            reason="directories are not supported; pass a single file",
        )
    extension = path.suffix.lower()
    if not extension:
        raise UnsupportedEvidenceFormatError(path, extension=None)
    format_name = EVIDENCE_EXTENSION_ROUTES.get(extension)
    if format_name is None:
        raise UnsupportedEvidenceFormatError(path, extension=extension)
    return format_name



def canonicalize_source_uri(source_path: str | Path) -> str:
    """Stable source URI used for document identity.

    Local files use ``file://`` absolute paths so the same file yields the same
    ``document_id`` across re-ingestions. Non-existent paths keep the provided string.
    """
    path = Path(source_path)
    if path.exists():
        return path.resolve().as_uri()
    text = str(source_path).strip()
    if text.startswith(("file:", "http://", "https://")):
        return text
    # Relative / fictional paths: normalize separators without inventing a host.
    return Path(text).as_posix()


def document_id_for_uri(source_uri: str) -> str:
    digest = hashlib.sha256(source_uri.encode("utf-8")).hexdigest()[:24]
    return f"doc:{digest}"


def content_hash_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_units_from_markdown(
    *,
    document_id: str,
    source: str,
    max_section_chars: int | None = DEFAULT_MAX_SECTION_CHARS,
) -> list[EvidenceUnit]:
    """Parse Markdown into EvidenceUnits (structure-first; optional size fallback).

    Compatibility wrapper: MarkdownAdapter → materialize_units.
    """
    from continuator_memory.evidence.materialize import materialize_units

    adapter = MarkdownAdapter()
    canonical = adapter.parse(
        source,
        source_uri=f"inline://{document_id}",
        metadata={"adapter": "markdown"},
        title_fallback="document",
    )
    return materialize_units(
        canonical,
        document_id=document_id,
        max_section_chars=max_section_chars,
    )


def build_units_from_html(
    *,
    document_id: str,
    source: str,
    max_section_chars: int | None = DEFAULT_MAX_SECTION_CHARS,
) -> list[EvidenceUnit]:
    """Parse HTML into EvidenceUnits via HTMLAdapter → materialize_units."""
    from continuator_memory.evidence.materialize import materialize_units

    adapter = HTMLAdapter()
    canonical = adapter.parse(
        source,
        source_uri=f"inline://{document_id}",
        metadata={"adapter": "html"},
        title_fallback="document",
    )
    return materialize_units(
        canonical,
        document_id=document_id,
        max_section_chars=max_section_chars,
    )


def build_units_from_docx(
    *,
    document_id: str,
    source: str | bytes | Path,
    max_section_chars: int | None = DEFAULT_MAX_SECTION_CHARS,
) -> list[EvidenceUnit]:
    """Parse DOCX into EvidenceUnits via DOCXAdapter → materialize_units.

    ``source`` is raw ``.docx`` bytes or a filesystem path to a ``.docx`` file.
    """
    from continuator_memory.evidence.materialize import materialize_units

    if isinstance(source, Path):
        payload: str | bytes = source.read_bytes()
    else:
        payload = source
    adapter = DOCXAdapter()
    canonical = adapter.parse(
        payload,
        source_uri=f"inline://{document_id}",
        metadata={"adapter": "docx"},
        title_fallback="document",
    )
    return materialize_units(
        canonical,
        document_id=document_id,
        max_section_chars=max_section_chars,
    )


def _ingest_text_adapter(
    *,
    adapter_name: str,
    adapter: MarkdownAdapter | HTMLAdapter,
    source_path: str | Path,
    store: EvidenceStore,
    text: str | None,
    source_uri: str | None,
    max_section_chars: int | None,
    metadata: dict[str, Any] | None,
    clock: Callable[[], str] | None,
) -> IngestedDocument:
    path = Path(source_path)
    if text is None:
        text = path.read_text(encoding="utf-8")
    uri = source_uri or canonicalize_source_uri(path)
    doc_id = document_id_for_uri(uri)
    now = (clock or (lambda: datetime.now(timezone.utc).isoformat()))()
    meta = {
        "adapter": adapter_name,
        "original_path": str(source_path),
        **(metadata or {}),
    }
    canonical = adapter.parse(
        text,
        source_uri=uri,
        metadata=meta,
        title_fallback=path.stem,
    )
    document, units = materialize(
        canonical,
        document_id=doc_id,
        max_section_chars=max_section_chars,
        created_at=now,
    )
    replaced = store.save_document(document, units, replace_uri=True)
    return IngestedDocument(document=document, units=tuple(units), replaced=replaced)


def ingest_markdown(
    source_path: str | Path,
    store: EvidenceStore,
    *,
    text: str | None = None,
    source_uri: str | None = None,
    max_section_chars: int | None = DEFAULT_MAX_SECTION_CHARS,
    metadata: dict[str, Any] | None = None,
    clock: Callable[[], str] | None = None,
) -> IngestedDocument:
    """Ingest a Markdown file (or provided text) into the evidence store.

    Re-ingestion of the same ``source_uri`` replaces the prior document snapshot.
    ``document_id`` is stable for a given URI; unit ids are recomputed from the
    structural ordinal after parsing (identical content → identical unit ids).

    Internally uses MarkdownAdapter + shared materializer; persisted semantics
    match Phase-1 Markdown ingestion.
    """
    return _ingest_text_adapter(
        adapter_name="markdown",
        adapter=MarkdownAdapter(),
        source_path=source_path,
        store=store,
        text=text,
        source_uri=source_uri,
        max_section_chars=max_section_chars,
        metadata=metadata,
        clock=clock,
    )


def ingest_html(
    source_path: str | Path,
    store: EvidenceStore,
    *,
    text: str | None = None,
    source_uri: str | None = None,
    max_section_chars: int | None = DEFAULT_MAX_SECTION_CHARS,
    metadata: dict[str, Any] | None = None,
    clock: Callable[[], str] | None = None,
) -> IngestedDocument:
    """Ingest an HTML file (or provided text) into the evidence store.

    Same snapshot-replace / stable ``document_id`` rules as ``ingest_markdown``.
    Internally: HTMLAdapter → CanonicalDocument → shared materializer.
    """
    return _ingest_text_adapter(
        adapter_name="html",
        adapter=HTMLAdapter(),
        source_path=source_path,
        store=store,
        text=text,
        source_uri=source_uri,
        max_section_chars=max_section_chars,
        metadata=metadata,
        clock=clock,
    )


def build_units_from_pdf(
    *,
    document_id: str,
    source: str | bytes | Path,
    max_section_chars: int | None = DEFAULT_MAX_SECTION_CHARS,
) -> list[EvidenceUnit]:
    """Parse PDF into EvidenceUnits via PDFAdapter → materialize_units.

    ``source`` is raw PDF bytes or a filesystem path to a ``.pdf`` file.
    """
    from continuator_memory.evidence.materialize import materialize_units

    if isinstance(source, Path):
        payload: str | bytes = source.read_bytes()
    else:
        payload = source
    adapter = PDFAdapter()
    canonical = adapter.parse(
        payload,
        source_uri=f"inline://{document_id}",
        metadata={"adapter": "pdf"},
        title_fallback="document",
    )
    return materialize_units(
        canonical,
        document_id=document_id,
        max_section_chars=max_section_chars,
    )


def _ingest_binary_adapter(
    *,
    adapter_name: str,
    adapter: DOCXAdapter | PDFAdapter,
    source_path: str | Path,
    store: EvidenceStore,
    data: bytes | None,
    source_uri: str | None,
    max_section_chars: int | None,
    metadata: dict[str, Any] | None,
    clock: Callable[[], str] | None,
) -> IngestedDocument:
    path = Path(source_path)
    payload = data if data is not None else path.read_bytes()
    uri = source_uri or canonicalize_source_uri(path)
    doc_id = document_id_for_uri(uri)
    now = (clock or (lambda: datetime.now(timezone.utc).isoformat()))()
    meta = {
        "adapter": adapter_name,
        "original_path": str(source_path),
        **(metadata or {}),
    }
    canonical = adapter.parse(
        payload,
        source_uri=uri,
        metadata=meta,
        title_fallback=path.stem,
    )
    document, units = materialize(
        canonical,
        document_id=doc_id,
        max_section_chars=max_section_chars,
        created_at=now,
    )
    replaced = store.save_document(document, units, replace_uri=True)
    return IngestedDocument(document=document, units=tuple(units), replaced=replaced)


def ingest_docx(
    source_path: str | Path,
    store: EvidenceStore,
    *,
    data: bytes | None = None,
    source_uri: str | None = None,
    max_section_chars: int | None = DEFAULT_MAX_SECTION_CHARS,
    metadata: dict[str, Any] | None = None,
    clock: Callable[[], str] | None = None,
) -> IngestedDocument:
    """Ingest a DOCX file (or provided bytes) into the evidence store.

    Same snapshot-replace / stable ``document_id`` rules as ``ingest_markdown``.
    Internally: DOCXAdapter → CanonicalDocument → shared materializer.
    Reads the file as binary (DOCX is a ZIP package).
    """
    return _ingest_binary_adapter(
        adapter_name="docx",
        adapter=DOCXAdapter(),
        source_path=source_path,
        store=store,
        data=data,
        source_uri=source_uri,
        max_section_chars=max_section_chars,
        metadata=metadata,
        clock=clock,
    )


def ingest_pdf(
    source_path: str | Path,
    store: EvidenceStore,
    *,
    data: bytes | None = None,
    source_uri: str | None = None,
    max_section_chars: int | None = DEFAULT_MAX_SECTION_CHARS,
    metadata: dict[str, Any] | None = None,
    clock: Callable[[], str] | None = None,
) -> IngestedDocument:
    """Ingest a digital-text PDF file (or provided bytes) into the evidence store.

    Same snapshot-replace / stable ``document_id`` rules as ``ingest_markdown``.
    Internally: PDFAdapter → CanonicalDocument → shared materializer.
    Reads the file as binary. Scanned/image-only PDFs raise
    ``PDFNoExtractableTextError`` (OCR is not supported).
    """
    return _ingest_binary_adapter(
        adapter_name="pdf",
        adapter=PDFAdapter(),
        source_path=source_path,
        store=store,
        data=data,
        source_uri=source_uri,
        max_section_chars=max_section_chars,
        metadata=metadata,
        clock=clock,
    )


def ingest_file(
    source_path: str | Path,
    store: EvidenceStore,
    *,
    text: str | None = None,
    data: bytes | None = None,
    source_uri: str | None = None,
    max_section_chars: int | None = DEFAULT_MAX_SECTION_CHARS,
    metadata: dict[str, Any] | None = None,
    clock: Callable[[], str] | None = None,
) -> IngestedDocument:
    """Ingest a supported evidence file by routing on path extension.

    Delegates to ``ingest_markdown`` / ``ingest_html`` / ``ingest_docx`` /
    ``ingest_pdf``. Does not sniff MIME/content and does not ingest directories.
    """
    format_name = resolve_evidence_format(source_path)
    common = {
        "source_uri": source_uri,
        "max_section_chars": max_section_chars,
        "metadata": metadata,
        "clock": clock,
    }
    if format_name == "markdown":
        return ingest_markdown(source_path, store, text=text, **common)
    if format_name == "html":
        return ingest_html(source_path, store, text=text, **common)
    if format_name == "docx":
        return ingest_docx(source_path, store, data=data, **common)
    if format_name == "pdf":
        return ingest_pdf(source_path, store, data=data, **common)
    # Defensive: resolve_evidence_format should have already rejected this.
    raise UnsupportedEvidenceFormatError(source_path)
