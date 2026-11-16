"""Storage and runner for structured extraction. Only its own ``sx_`` tables are written.

No ``v2_`` table is touched, so consolidation and retrieval are unaffected. The tables are created when
a ``StructuredStore`` is opened, so point it at a database you mean to use for this.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from continuator_memory.models import Conversation
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import DEFAULT_PRIOR_LIMIT
from continuator_memory.v2.structured_extraction import (
    STRUCTURED_PROTOCOL_VERSION,
    StructuredExtractor,
    StructuredMemory,
)
from continuator_memory.v2.windowing import DEFAULT_CONTEXT_BEFORE, DEFAULT_TARGETS_PER_WINDOW, build_windows

SX_SCHEMA = """
CREATE TABLE IF NOT EXISTS sx_windows (
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    window_index INTEGER NOT NULL,
    target_ids_json TEXT NOT NULL,
    context_ids_json TEXT NOT NULL,
    prior_memory_ids_json TEXT NOT NULL,
    protocol TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'invalid', 'error')),
    raw_output TEXT,
    error TEXT,
    issues_json TEXT NOT NULL,
    usage_json TEXT,
    latency_ms REAL NOT NULL,
    PRIMARY KEY (conversation_id, window_index)
);
CREATE TABLE IF NOT EXISTS sx_records (
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    record_id TEXT NOT NULL,
    window_index INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('entity', 'memory', 'relationship')),
    data_json TEXT NOT NULL,
    PRIMARY KEY (conversation_id, record_id)
);
CREATE INDEX IF NOT EXISTS sx_records_kind ON sx_records (conversation_id, kind);
CREATE TABLE IF NOT EXISTS sx_message_outcomes (
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    message_id TEXT NOT NULL,
    window_index INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    note TEXT,
    PRIMARY KEY (conversation_id, message_id)
);
"""


class StructuredStore:
    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        self._conn.executescript(SX_SCHEMA)

    def has_windows(self, conversation_id: str) -> bool:
        return self._conn.execute("SELECT 1 FROM sx_windows WHERE conversation_id = ? LIMIT 1",
                                  (conversation_id,)).fetchone() is not None

    def clear_conversation(self, conversation_id: str) -> None:
        with self._conn:
            for table in ("sx_message_outcomes", "sx_records", "sx_windows"):
                self._conn.execute(f"DELETE FROM {table} WHERE conversation_id = ?", (conversation_id,))

    def save_window(self, validated: Any, model: str, latency_ms: float, prior_memory_ids: tuple[str, ...] = (),
                    usage: dict[str, Any] | None = None, protocol: str = STRUCTURED_PROTOCOL_VERSION) -> None:
        window = validated.window
        cid = window.conversation_id
        rows = [*(("entity", r.entity_id, r) for r in validated.entities),
                *(("memory", r.memory_id, r) for r in validated.memories),
                *(("relationship", r.relationship_id, r) for r in validated.relationships)]
        with self._conn:
            self._conn.execute(
                "INSERT INTO sx_windows (conversation_id, window_index, target_ids_json, context_ids_json, "
                "prior_memory_ids_json, protocol, model, status, raw_output, error, issues_json, usage_json, latency_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (cid, window.index, json.dumps(list(window.target_ids)), json.dumps(list(window.context_ids)),
                 json.dumps(list(prior_memory_ids)), protocol, model, validated.status,
                 validated.raw, validated.error, json.dumps([i.to_dict() for i in validated.issues]),
                 json.dumps(usage) if usage else None, latency_ms))
            for kind, record_id, record in rows:
                self._conn.execute(
                    "INSERT INTO sx_records (conversation_id, record_id, window_index, kind, data_json) "
                    "VALUES (?, ?, ?, ?, ?)", (cid, record_id, window.index, kind, json.dumps(record.to_dict(), sort_keys=True)))
            for outcome in getattr(validated, "outcomes", ()):
                self._conn.execute(
                    "INSERT INTO sx_message_outcomes (conversation_id, message_id, window_index, outcome, note) "
                    "VALUES (?, ?, ?, ?, ?)", (cid, outcome.message_id, window.index, outcome.outcome, outcome.note))

    def get_records(self, conversation_id: str, kind: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT data_json FROM sx_records WHERE conversation_id = ? AND kind = ? ORDER BY window_index, rowid",
            (conversation_id, kind)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_outcomes(self, conversation_id: str) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT message_id, window_index, outcome, note FROM sx_message_outcomes "
                                  "WHERE conversation_id = ?", (conversation_id,)).fetchall()
        return {r[0]: {"window_index": r[1], "outcome": r[2], "note": r[3]} for r in rows}

    def get_windows(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT window_index, status, raw_output, error, issues_json, usage_json, latency_ms, model, protocol "
            "FROM sx_windows WHERE conversation_id = ? ORDER BY window_index", (conversation_id,)).fetchall()
        return [{"window_index": r[0], "status": r[1], "raw_output": r[2], "error": r[3], "issues": json.loads(r[4]),
                 "usage": json.loads(r[5]) if r[5] else None, "latency_ms": r[6], "model": r[7], "protocol": r[8]}
                for r in rows]


@dataclass
class StructuredIngestReport:
    conversation_id: str
    windows: int = 0
    failed_windows: int = 0
    model_calls: int = 0
    entities: int = 0
    memories: int = 0
    relationships: int = 0
    needs_review: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    issues: Counter = field(default_factory=Counter)
    truncated_windows: int = 0
    unaccounted_targets: list[str] = field(default_factory=list)

    def add_window(self, validated: Any, usage: dict[str, Any] | None, latency_ms: float) -> None:
        usage = usage or {}
        self.windows += 1
        self.failed_windows += validated.error is not None
        self.model_calls += 1
        self.entities += len(validated.entities)
        self.memories += len(validated.memories)
        self.relationships += len(validated.relationships)
        self.needs_review += sum(r.needs_review for r in (*validated.memories, *validated.relationships))
        self.prompt_tokens += usage.get("prompt_eval_count") or 0
        self.output_tokens += usage.get("eval_count") or 0
        self.latency_ms += latency_ms
        self.issues.update(f"{i.code}:{i.action}" for i in validated.issues)
        self.truncated_windows += any(i.code == "TRUNCATED_RESPONSE" for i in validated.issues)
        self.unaccounted_targets += list(getattr(validated, "unaccounted_targets", ()))

    def to_dict(self) -> dict[str, Any]:
        return {"conversation_id": self.conversation_id, "windows": self.windows, "failed_windows": self.failed_windows,
                "model_calls": self.model_calls, "entities": self.entities, "memories": self.memories,
                "relationships": self.relationships, "needs_review": self.needs_review,
                "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
                "latency_ms": round(self.latency_ms, 3), "issues": dict(sorted(self.issues.items())),
                "truncated_windows": self.truncated_windows, "unaccounted_targets": list(self.unaccounted_targets)}


def prepare_conversation(conversation: Conversation, store: Store, replace: bool) -> StructuredStore:
    """Store the conversation only if absent, because ``save_conversation`` replaces it and its chunks."""
    cid = conversation.conversation_id
    existing = store.get_conversation(cid)
    if existing is None:
        store.save_conversation(conversation, [], np.zeros((0, 0), dtype=np.float32), "none")
    elif existing.messages != conversation.messages:
        raise ValueError(f"conversation {cid!r} is already stored with different messages")
    structured = StructuredStore(store.connection)
    if structured.has_windows(cid):
        if not replace:
            raise ValueError(f"conversation {cid!r} already has structured extraction; pass replace=True")
        structured.clear_conversation(cid)
    return structured


def ingest_structured(
    conversation: Conversation,
    store: Store,
    extractor: StructuredExtractor,
    targets_per_window: int = DEFAULT_TARGETS_PER_WINDOW,
    context_before: int = DEFAULT_CONTEXT_BEFORE,
    prior_limit: int = DEFAULT_PRIOR_LIMIT,
    replace: bool = False,
) -> StructuredIngestReport:
    """Run one call per window, sequentially, saving each window as it finishes so partial results survive."""
    structured = prepare_conversation(conversation, store, replace)
    report = StructuredIngestReport(conversation.conversation_id)
    prior: list[StructuredMemory] = []
    for window in build_windows(conversation, targets_per_window, context_before):
        window_prior = prior[-prior_limit:] if prior_limit else []
        result = extractor.extract(window, window_prior)
        validated = result.validated
        structured.save_window(validated, extractor.model_name, result.latency_ms,
                               tuple(m.memory_id for m in window_prior), result.usage)
        prior.extend(validated.memories)
        report.add_window(validated, result.usage, result.latency_ms)
    return report
