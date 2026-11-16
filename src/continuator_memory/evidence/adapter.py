"""Minimal FormatAdapter contract for evidence ingestion.

Adapters parse bytes/text into ``CanonicalDocument``. They must not write to
the store, call search, download models, or invent hierarchy when none exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from continuator_memory.evidence.canonical import CanonicalDocument


@runtime_checkable
class FormatAdapter(Protocol):
    """Parse a source into the in-memory canonical evidence representation."""

    name: str
    source_types: frozenset[str]

    def supports(
        self,
        path_or_uri: str | Path,
        *,
        sniff: bytes | str | None = None,
    ) -> bool:
        """Return whether this adapter can handle the given path/URI (and optional sniff)."""
        ...

    def parse(
        self,
        source: str | bytes,
        *,
        source_uri: str,
        metadata: dict[str, Any] | None = None,
    ) -> CanonicalDocument:
        """Format-specific parse. No store writes. No retrieval."""
        ...
