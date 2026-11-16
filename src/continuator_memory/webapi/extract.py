"""Product import helpers for the V2 ingestion pipeline.

Successful ingest writes validated claims and a consolidated ``MemoryStore`` snapshot.
LLM-backed extractors run only when ``extract`` is ``v2`` or ``memories``; the backend is
selected via ``CONTINUATOR_V2_EXTRACTOR`` / ``V2Settings.extractor`` (default: Ollama).
"""

from __future__ import annotations

from continuator_memory.models import Conversation
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import V2Settings
from continuator_memory.v2.extractors import (
    EXTRACT_MODES,
    MemoryExtractor,
    PassThroughAtomizer,
    PassThroughExtractor,
    extractor_for_import_mode,
)
from continuator_memory.v2.pipeline import V2IngestReport, V2IngestionPipeline
from continuator_memory.v2.store import SemanticStore

__all__ = [
    "EXTRACT_MODES",
    "PassThroughAtomizer",
    "PassThroughExtractor",
    "ingest_imported",
]


def ingest_imported(
    conversation: Conversation,
    store: Store,
    *,
    extract: str = "messages",
    atomizer: MemoryExtractor | None = None,
    settings: V2Settings | None = None,
) -> V2IngestReport:
    """Ingest through ``V2IngestionPipeline`` using a ``MemoryExtractor``.

    Pass ``atomizer`` to inject any compatible extractor (test double, future MemReader, …).
    Otherwise ``extractor_for_import_mode`` chooses passthrough or the configured backend.
    """
    resolved = extractor_for_import_mode(extract, settings=settings, atomizer=atomizer)
    SemanticStore(store.connection)  # ensure V2 tables exist
    return V2IngestionPipeline(store, resolved, embedder=None).ingest(conversation)
