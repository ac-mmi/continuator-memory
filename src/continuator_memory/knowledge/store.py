"""Storage for knowledge documents. Only its own ``kd_`` tables are written; raw tables are only referenced.

The tables are created when a ``KnowledgeStore`` is opened, so point it at a database meant for this experiment.
Rows cascade from ``chunks``: re-saving a conversation would remove them, so ``ingest`` never does that.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

KD_SCHEMA = """
CREATE TABLE IF NOT EXISTS kd_documents (
    document_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    format_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'nothing_durable', 'truncated', 'failed')),
    content_markdown TEXT,
    content_hash TEXT,
    source_text_hash TEXT NOT NULL,
    context_message_ids_json TEXT NOT NULL,
    claims_json TEXT NOT NULL,
    issues_json TEXT NOT NULL,
    uncited_target_ids_json TEXT NOT NULL,
    error TEXT,
    usage_json TEXT,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (chunk_id, prompt_version, model_name)
);
CREATE TABLE IF NOT EXISTS kd_passages (
    passage_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES kd_documents(document_id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    passage_index INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    content TEXT NOT NULL,
    embedded_text TEXT NOT NULL,
    claim_indexes_json TEXT NOT NULL,
    message_ids_json TEXT NOT NULL,
    citation_source TEXT NOT NULL CHECK (citation_source IN ('cited', 'chunk_fallback')),
    review_status TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    UNIQUE (document_id, passage_index)
);
CREATE TABLE IF NOT EXISTS kd_passage_embeddings (
    passage_id TEXT NOT NULL REFERENCES kd_passages(passage_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (passage_id, model_name)
);
"""


@dataclass(frozen=True)
class PassageRow:
    passage_id: str
    passage_index: int
    heading_path: str
    content: str
    embedded_text: str
    claim_indexes: tuple[int, ...]
    message_ids: tuple[str, ...]
    citation_source: str
    review_status: str
    token_estimate: int


@dataclass(frozen=True)
class DocumentRow:
    document_id: str
    conversation_id: str
    chunk_id: str
    chunk_index: int
    format_version: str
    prompt_version: str
    model_name: str
    status: str
    content_markdown: str | None
    content_hash: str | None
    source_text_hash: str
    context_message_ids: tuple[str, ...]
    claims: tuple[dict[str, Any], ...]
    issues: tuple[str, ...]
    uncited_target_ids: tuple[str, ...]
    error: str | None
    usage: dict[str, Any] | None
    latency_ms: float
    created_at: str


class KnowledgeStore:
    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(KD_SCHEMA)

    def has_documents(self, conversation_id: str, prompt_version: str, model_name: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM kd_documents WHERE conversation_id = ? AND prompt_version = ? AND model_name = ? LIMIT 1",
            (conversation_id, prompt_version, model_name)).fetchone() is not None

    def delete_documents(self, conversation_id: str, prompt_version: str, model_name: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM kd_documents WHERE conversation_id = ? AND prompt_version = ? AND model_name = ?",
                               (conversation_id, prompt_version, model_name))

    def save_document(self, document: DocumentRow, passages: list[PassageRow], vectors: np.ndarray | None, embedding_model: str | None) -> None:
        """One transaction, so a document is never stored without its passages and embeddings."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO kd_documents (document_id, conversation_id, chunk_id, chunk_index, format_version, prompt_version, "
                "model_name, status, content_markdown, content_hash, source_text_hash, context_message_ids_json, claims_json, "
                "issues_json, uncited_target_ids_json, error, usage_json, latency_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (document.document_id, document.conversation_id, document.chunk_id, document.chunk_index, document.format_version,
                 document.prompt_version, document.model_name, document.status, document.content_markdown, document.content_hash,
                 document.source_text_hash, json.dumps(list(document.context_message_ids)), json.dumps(list(document.claims)),
                 json.dumps(list(document.issues)), json.dumps(list(document.uncited_target_ids)), document.error,
                 json.dumps(document.usage) if document.usage else None, document.latency_ms, document.created_at))
            for i, p in enumerate(passages):
                self._conn.execute(
                    "INSERT INTO kd_passages (passage_id, document_id, conversation_id, passage_index, heading_path, content, "
                    "embedded_text, claim_indexes_json, message_ids_json, citation_source, review_status, token_estimate) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (p.passage_id, document.document_id, document.conversation_id, p.passage_index, p.heading_path, p.content,
                     p.embedded_text, json.dumps(list(p.claim_indexes)), json.dumps(list(p.message_ids)), p.citation_source,
                     p.review_status, p.token_estimate))
                if vectors is not None:
                    vector = np.asarray(vectors[i], dtype=np.float32)
                    self._conn.execute("INSERT INTO kd_passage_embeddings (passage_id, model_name, dim, vector) VALUES (?, ?, ?, ?)",
                                       (p.passage_id, embedding_model, vector.shape[0], vector.tobytes()))

    def get_documents(self, conversation_id: str) -> list[dict[str, Any]]:
        columns = ("document_id", "chunk_id", "chunk_index", "status", "content_markdown", "content_hash", "source_text_hash",
                   "context_message_ids_json", "claims_json", "issues_json", "uncited_target_ids_json", "error", "usage_json",
                   "prompt_version", "format_version", "model_name", "created_at")
        rows = self._conn.execute(f"SELECT {', '.join(columns)} FROM kd_documents WHERE conversation_id = ? ORDER BY chunk_index",
                                  (conversation_id,)).fetchall()
        return [_decode(dict(zip(columns, row))) for row in rows]

    def get_passages(self, document_id: str) -> list[dict[str, Any]]:
        columns = ("passage_id", "passage_index", "heading_path", "content", "embedded_text", "claim_indexes_json", "message_ids_json",
                   "citation_source", "review_status", "token_estimate")
        rows = self._conn.execute(f"SELECT {', '.join(columns)} FROM kd_passages WHERE document_id = ? ORDER BY passage_index",
                                  (document_id,)).fetchall()
        return [_decode(dict(zip(columns, row))) for row in rows]

    def load_embeddings(self, passage_ids: list[str], model_name: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for passage_id in passage_ids:
            row = self._conn.execute("SELECT vector FROM kd_passage_embeddings WHERE passage_id = ? AND model_name = ?",
                                     (passage_id, model_name)).fetchone()
            if row:
                out[passage_id] = np.frombuffer(row[0], dtype=np.float32)
        return out


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    for key in [k for k in row if k.endswith("_json")]:
        value = row.pop(key)
        row[key[: -len("_json")]] = json.loads(value) if value else None
    return row
