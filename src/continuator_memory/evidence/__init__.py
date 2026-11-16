"""Parallel universal evidence layer (Markdown + HTML + DOCX + PDF adapters).

Isolated from the conversation ``Memory`` pipeline. Public entry points:

* ``ingest_file`` — extension-routed convenience over format-specific ingest
* ``ingest_markdown`` / ``ingest_html`` / ``ingest_docx`` / ``ingest_pdf``
  — format adapters → Document + EvidenceUnits
* ``search_evidence`` — BM25 over EvidenceUnits via existing ``RetrievalUnit``
* ``EvidenceStore`` — additive ``ev_`` SQLite tables
"""

from continuator_memory.evidence.ingest import (
    UnsupportedEvidenceFormatError,
    ingest_docx,
    ingest_file,
    ingest_html,
    ingest_markdown,
    ingest_pdf,
)
from continuator_memory.evidence.models import (
    Document,
    EvidenceHit,
    EvidenceUnit,
    IngestedDocument,
)
from continuator_memory.evidence.search import resolve_provenance, search_evidence
from continuator_memory.evidence.store import EvidenceStore

__all__ = [
    "Document",
    "EvidenceHit",
    "EvidenceStore",
    "EvidenceUnit",
    "IngestedDocument",
    "UnsupportedEvidenceFormatError",
    "ingest_docx",
    "ingest_file",
    "ingest_html",
    "ingest_markdown",
    "ingest_pdf",
    "resolve_provenance",
    "search_evidence",
]
