"""Shared materializer: CanonicalDocument → Document + EvidenceUnit rows.

Owns ordinal assignment, deterministic evidence ids, default retrieval_text
policy, optional max_chars splitting, and span/metadata preservation.
Format adapters must not invent these identities themselves.
"""

from __future__ import annotations

from typing import Any

from continuator_memory.evidence.canonical import (
    CanonicalDocument,
    iter_sections_depth_first,
)
from continuator_memory.evidence.markdown import (
    build_retrieval_text,
    section_id_for,
    split_section_text,
)
from continuator_memory.evidence.models import Document, EvidenceUnit


def evidence_id_for(document_id: str, ordinal: int) -> str:
    """Deterministic unit id from document + structural ordinal."""
    return f"ev:{document_id}:u{ordinal:04d}"


def materialize_document(
    canonical: CanonicalDocument,
    *,
    document_id: str,
    created_at: str | None = None,
) -> Document:
    """Build the persisted Document row from a canonical parse."""
    return Document(
        document_id=document_id,
        source_uri=canonical.source_uri,
        source_type=canonical.source_type,
        content_hash=canonical.content_hash,
        title=canonical.title,
        metadata=dict(canonical.metadata),
        created_at=created_at,
    )


def materialize_units(
    canonical: CanonicalDocument,
    *,
    document_id: str,
    max_section_chars: int | None = None,
) -> list[EvidenceUnit]:
    """Convert canonical sections (flat or nested) into EvidenceUnits.

    Semantics match the Phase-1 Markdown materialization rules:
    - document order via depth-first traversal
    - optional ``max_section_chars`` split via ``split_section_text``
    - ``section_id`` gains ``#pNN`` suffixes when split
    - char spans kept only for unsplit sections
    - ``retrieval_text`` = ancestry join + body (empty path → body only)
    """
    units: list[EvidenceUnit] = []
    ordinal = 0
    for section in iter_sections_depth_first(canonical.sections):
        pieces = split_section_text(section.text, max_chars=max_section_chars)
        if not pieces:
            continue
        n_pieces = len(pieces)
        base_section_id = section.section_id or section_id_for(section.heading_path)
        for piece_index, piece in enumerate(pieces):
            section_key = base_section_id
            if n_pieces > 1:
                section_key = f"{base_section_id}#p{piece_index:02d}"
            if n_pieces == 1:
                char_start, char_end = section.char_start, section.char_end
            else:
                char_start, char_end = None, None
            retrieval = build_retrieval_text(section.heading_path, piece)
            meta: dict[str, Any] = {
                "heading_level": section.heading_level,
                "piece_index": piece_index,
                "piece_count": n_pieces,
            }
            # Preserve adapter-specific section metadata without clobbering piece keys.
            for key, value in section.metadata.items():
                if key not in meta:
                    meta[key] = value
            units.append(
                EvidenceUnit(
                    evidence_id=evidence_id_for(document_id, ordinal),
                    document_id=document_id,
                    ordinal=ordinal,
                    text=piece,
                    retrieval_text=retrieval,
                    heading_path=section.heading_path,
                    section_id=section_key,
                    char_start=char_start,
                    char_end=char_end,
                    metadata=meta,
                )
            )
            ordinal += 1
    return units


def materialize(
    canonical: CanonicalDocument,
    *,
    document_id: str,
    max_section_chars: int | None = None,
    created_at: str | None = None,
) -> tuple[Document, list[EvidenceUnit]]:
    """Materialize Document + EvidenceUnits from a canonical parse."""
    document = materialize_document(
        canonical, document_id=document_id, created_at=created_at
    )
    units = materialize_units(
        canonical, document_id=document_id, max_section_chars=max_section_chars
    )
    return document, units
