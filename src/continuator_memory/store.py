"""SQLite persistence for conversations, messages, chunks, provenance and embeddings."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from continuator_memory.models import Chunk, Conversation, Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    message_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT,
    PRIMARY KEY (conversation_id, message_id),
    UNIQUE (conversation_id, position)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE (conversation_id, position)
);

CREATE TABLE IF NOT EXISTS chunk_messages (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, message_id),
    FOREIGN KEY (conversation_id, message_id)
        REFERENCES messages(conversation_id, message_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (chunk_id, model_name)
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- writes -----------------------------------------------------------

    def save_conversation(
        self,
        conversation: Conversation,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        model_name: str,
    ) -> None:
        """Atomically store a conversation with its chunks and embeddings.

        An existing conversation with the same ID is replaced.
        """
        with self._conn:
            self.write_conversation(conversation, chunks, embeddings, model_name)

    def write_conversation(
        self,
        conversation: Conversation,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        model_name: str,
    ) -> None:
        """``save_conversation`` inside the caller's open transaction; does not commit."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings must have one row per chunk")
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        cid = conversation.conversation_id
        self._conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (cid,))
        self._conn.execute(
            "INSERT INTO conversations (conversation_id, title) VALUES (?, ?)",
            (cid, conversation.title),
        )
        self._conn.executemany(
            "INSERT INTO messages (message_id, conversation_id, position, role, content, "
            "timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (m.message_id, cid, m.position, m.role, m.content, m.timestamp)
                for m in conversation.messages
            ],
        )
        for chunk, vector in zip(chunks, vectors):
            self._conn.execute(
                "INSERT INTO chunks (chunk_id, conversation_id, position, text, metadata_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    chunk.chunk_id,
                    chunk.conversation_id,
                    chunk.position,
                    chunk.text,
                    json.dumps(chunk.metadata, sort_keys=True),
                ),
            )
            self._conn.executemany(
                "INSERT INTO chunk_messages (chunk_id, conversation_id, message_id, position) "
                "VALUES (?, ?, ?, ?)",
                [
                    (chunk.chunk_id, chunk.conversation_id, mid, i)
                    for i, mid in enumerate(chunk.source_message_ids)
                ],
            )
            self._conn.execute(
                "INSERT INTO chunk_embeddings (chunk_id, model_name, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                (chunk.chunk_id, model_name, vector.shape[0], vector.tobytes()),
            )

    # ---- reads ------------------------------------------------------------

    def list_conversations(self) -> list[tuple[str, str | None]]:
        rows = self._conn.execute(
            "SELECT conversation_id, title FROM conversations ORDER BY conversation_id"
        )
        return list(rows)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        row = self._conn.execute(
            "SELECT title FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            return None
        rows = self._conn.execute(
            "SELECT message_id, position, role, content, timestamp FROM messages "
            "WHERE conversation_id = ? ORDER BY position",
            (conversation_id,),
        )
        messages = tuple(
            Message(
                message_id=mid,
                conversation_id=conversation_id,
                position=pos,
                role=role,
                content=content,
                timestamp=ts,
            )
            for mid, pos, role, content, ts in rows
        )
        return Conversation(conversation_id=conversation_id, messages=messages, title=row[0])

    def get_messages(
        self, conversation_id: str, message_ids: list[str] | tuple[str, ...]
    ) -> list[Message]:
        """Return messages of one conversation in the order of ``message_ids``."""
        if not message_ids:
            return []
        placeholders = ",".join("?" * len(message_ids))
        rows = self._conn.execute(
            "SELECT message_id, conversation_id, position, role, content, timestamp "
            f"FROM messages WHERE conversation_id = ? AND message_id IN ({placeholders})",
            (conversation_id, *message_ids),
        )
        by_id = {r[0]: Message(*r) for r in rows}
        return [by_id[mid] for mid in message_ids if mid in by_id]

    def _source_message_ids(self, chunk_id: str) -> tuple[str, ...]:
        rows = self._conn.execute(
            "SELECT message_id FROM chunk_messages WHERE chunk_id = ? ORDER BY position",
            (chunk_id,),
        )
        return tuple(r[0] for r in rows)

    def get_chunks(
        self, chunk_ids: list[str] | None = None, conversation_id: str | None = None
    ) -> list[Chunk]:
        """Return chunks filtered by ID and/or conversation, ordered by (conversation, position)."""
        sql = "SELECT chunk_id, conversation_id, position, text, metadata_json FROM chunks"
        clauses, params = [], []
        if chunk_ids is not None:
            if not chunk_ids:
                return []
            clauses.append(f"chunk_id IN ({','.join('?' * len(chunk_ids))})")
            params.extend(chunk_ids)
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY conversation_id, position"
        return [
            Chunk(
                chunk_id=chunk_id,
                conversation_id=conv_id,
                position=pos,
                text=text,
                source_message_ids=self._source_message_ids(chunk_id),
                metadata=json.loads(meta),
            )
            for chunk_id, conv_id, pos, text, meta in self._conn.execute(sql, params)
        ]

    def load_embeddings(
        self, model_name: str, conversation_id: str | None = None
    ) -> tuple[list[str], np.ndarray]:
        """Return (chunk_ids, matrix) for all chunks embedded with ``model_name``.

        Rows are ordered by chunk_id so the result is deterministic.
        """
        sql = (
            "SELECT e.chunk_id, e.dim, e.vector FROM chunk_embeddings e "
            "JOIN chunks c ON c.chunk_id = e.chunk_id WHERE e.model_name = ?"
        )
        params: list[str] = [model_name]
        if conversation_id is not None:
            sql += " AND c.conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY e.chunk_id"
        rows = self._conn.execute(sql, params).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        dims = {r[1] for r in rows}
        if len(dims) != 1:
            raise ValueError(f"inconsistent embedding dims for {model_name}: {sorted(dims)}")
        matrix = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        return [r[0] for r in rows], matrix
