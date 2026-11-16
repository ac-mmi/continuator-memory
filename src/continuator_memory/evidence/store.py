"""SQLite persistence for universal documents and evidence units.

Own ``ev_`` tables — no foreign keys into conversations/messages/memories.
May share a SQLite file with the product ``Store`` via the same connection.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from continuator_memory.evidence.models import Document, EvidenceUnit

EV_SCHEMA = """
CREATE TABLE IF NOT EXISTS ev_documents (
    document_id TEXT PRIMARY KEY,
    source_uri TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    title TEXT,
    content_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ev_units (
    evidence_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL
        REFERENCES ev_documents(document_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    retrieval_text TEXT NOT NULL,
    heading_path_json TEXT NOT NULL,
    section_id TEXT NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    metadata_json TEXT NOT NULL,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_ev_units_document
    ON ev_units(document_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_ev_documents_source_uri
    ON ev_documents(source_uri);
"""


class EvidenceStore:
    """Additive document/evidence persistence."""

    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(EV_SCHEMA)

    @classmethod
    def open(cls, path: str | Path) -> EvidenceStore:
        path_s = str(path)
        if path_s != ":memory:":
            Path(path_s).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path_s, check_same_thread=False)
        return cls(conn)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EvidenceStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_document(self, document_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT document_id, source_uri, source_type, title, content_hash, "
            "metadata_json, created_at FROM ev_documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return _row_to_document(row) if row else None

    def get_document_by_uri(self, source_uri: str) -> Document | None:
        row = self._conn.execute(
            "SELECT document_id, source_uri, source_type, title, content_hash, "
            "metadata_json, created_at FROM ev_documents WHERE source_uri = ?",
            (source_uri,),
        ).fetchone()
        return _row_to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        rows = self._conn.execute(
            "SELECT document_id, source_uri, source_type, title, content_hash, "
            "metadata_json, created_at FROM ev_documents ORDER BY created_at, document_id"
        ).fetchall()
        return [_row_to_document(row) for row in rows]

    def get_unit(self, evidence_id: str) -> EvidenceUnit | None:
        row = self._conn.execute(
            "SELECT evidence_id, document_id, ordinal, text, retrieval_text, "
            "heading_path_json, section_id, char_start, char_end, metadata_json "
            "FROM ev_units WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        return _row_to_unit(row) if row else None

    def get_units(self, document_id: str) -> list[EvidenceUnit]:
        rows = self._conn.execute(
            "SELECT evidence_id, document_id, ordinal, text, retrieval_text, "
            "heading_path_json, section_id, char_start, char_end, metadata_json "
            "FROM ev_units WHERE document_id = ? ORDER BY ordinal",
            (document_id,),
        ).fetchall()
        return [_row_to_unit(row) for row in rows]

    def list_units(self) -> list[EvidenceUnit]:
        rows = self._conn.execute(
            "SELECT evidence_id, document_id, ordinal, text, retrieval_text, "
            "heading_path_json, section_id, char_start, char_end, metadata_json "
            "FROM ev_units ORDER BY document_id, ordinal"
        ).fetchall()
        return [_row_to_unit(row) for row in rows]

    def unit_count(self, document_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM ev_units WHERE document_id = ?", (document_id,)
        ).fetchone()
        return int(row[0]) if row else 0

    def delete_by_uri(self, source_uri: str) -> bool:
        """Delete the document (and cascaded units) for ``source_uri``. Returns whether one existed."""
        existing = self.get_document_by_uri(source_uri)
        if existing is None:
            return False
        with self._conn:
            self._conn.execute(
                "DELETE FROM ev_documents WHERE source_uri = ?", (source_uri,)
            )
        return True

    def delete_by_document_id(self, document_id: str) -> tuple[bool, int]:
        """Delete by ``document_id`` with unit CASCADE.

        Returns ``(deleted, units_removed)``. Idempotent: missing id → ``(False, 0)``.
        """
        existing = self.get_document(document_id)
        if existing is None:
            return False, 0
        units_removed = self.unit_count(document_id)
        with self._conn:
            self._conn.execute(
                "DELETE FROM ev_documents WHERE document_id = ?", (document_id,)
            )
        return True, units_removed

    def save_document(
        self,
        document: Document,
        units: list[EvidenceUnit],
        *,
        replace_uri: bool = True,
    ) -> bool:
        """Persist document + units atomically.

        When ``replace_uri`` is True (default), any prior row with the same
        ``source_uri`` is removed first (snapshot replace). Returns whether a
        previous document was replaced.
        """
        replaced = False
        with self._conn:
            if replace_uri:
                prior = self._conn.execute(
                    "SELECT document_id FROM ev_documents WHERE source_uri = ?",
                    (document.source_uri,),
                ).fetchone()
                if prior is not None:
                    replaced = True
                    self._conn.execute(
                        "DELETE FROM ev_documents WHERE source_uri = ?",
                        (document.source_uri,),
                    )
            self._conn.execute(
                "INSERT INTO ev_documents (document_id, source_uri, source_type, title, "
                "content_hash, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    document.document_id,
                    document.source_uri,
                    document.source_type,
                    document.title,
                    document.content_hash,
                    json.dumps(document.metadata, sort_keys=True),
                    document.created_at or "",
                ),
            )
            for unit in units:
                if unit.document_id != document.document_id:
                    raise ValueError(
                        f"unit {unit.evidence_id} document_id mismatch: "
                        f"{unit.document_id} != {document.document_id}"
                    )
                self._conn.execute(
                    "INSERT INTO ev_units (evidence_id, document_id, ordinal, text, "
                    "retrieval_text, heading_path_json, section_id, char_start, char_end, "
                    "metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        unit.evidence_id,
                        unit.document_id,
                        unit.ordinal,
                        unit.text,
                        unit.retrieval_text,
                        json.dumps(list(unit.heading_path)),
                        unit.section_id,
                        unit.char_start,
                        unit.char_end,
                        json.dumps(unit.metadata, sort_keys=True),
                    ),
                )
        return replaced


def _row_to_document(row: tuple[Any, ...]) -> Document:
    document_id, source_uri, source_type, title, content_hash, metadata_json, created_at = row
    return Document(
        document_id=document_id,
        source_uri=source_uri,
        source_type=source_type,
        title=title,
        content_hash=content_hash,
        metadata=json.loads(metadata_json) if metadata_json else {},
        created_at=created_at or None,
    )


def _row_to_unit(row: tuple[Any, ...]) -> EvidenceUnit:
    (
        evidence_id,
        document_id,
        ordinal,
        text,
        retrieval_text,
        heading_path_json,
        section_id,
        char_start,
        char_end,
        metadata_json,
    ) = row
    return EvidenceUnit(
        evidence_id=evidence_id,
        document_id=document_id,
        ordinal=ordinal,
        text=text,
        retrieval_text=retrieval_text,
        heading_path=tuple(json.loads(heading_path_json)),
        section_id=section_id,
        char_start=char_start,
        char_end=char_end,
        metadata=json.loads(metadata_json) if metadata_json else {},
    )
