"""SQLite persistence for V2 semantic records.

Lives in the V1 database next to the raw ``messages`` table. Semantic rows reference raw
messages by (conversation_id, message_id) instead of copying their content, and cascade from
``conversations`` like the V1 tables.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from continuator_memory.v2.models import (
    RELATION_TYPES,
    STATUSES,
    Claim,
    Provenance,
    Relation,
    SemanticMessage,
    Temporal,
)
from continuator_memory.v2.validation import ValidatedWindow

_TYPES = ", ".join(f"'{t}'" for t in RELATION_TYPES)
_STATUSES = ", ".join(f"'{s}'" for s in STATUSES)

V2_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS v2_semantic_messages (
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    message_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({_STATUSES})),
    window_index INTEGER NOT NULL,
    PRIMARY KEY (conversation_id, message_id),
    FOREIGN KEY (conversation_id, message_id)
        REFERENCES messages(conversation_id, message_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS v2_claims (
    conversation_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    entities_json TEXT NOT NULL,
    qualifiers_json TEXT NOT NULL,
    temporal_json TEXT,
    polarity TEXT,
    claim_type TEXT,
    support TEXT NOT NULL,
    resolved_from_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (conversation_id, claim_id),
    FOREIGN KEY (conversation_id, message_id)
        REFERENCES v2_semantic_messages(conversation_id, message_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS v2_relations (
    conversation_id TEXT NOT NULL,
    relation_id TEXT NOT NULL,
    from_claim_id TEXT NOT NULL,
    to_claim_id TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ({_TYPES})),
    message_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    evidence TEXT,
    evidence_start INTEGER,
    evidence_end INTEGER,
    PRIMARY KEY (conversation_id, relation_id),
    FOREIGN KEY (conversation_id, message_id)
        REFERENCES v2_semantic_messages(conversation_id, message_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, from_claim_id)
        REFERENCES v2_claims(conversation_id, claim_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, to_claim_id)
        REFERENCES v2_claims(conversation_id, claim_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS v2_atomization_log (
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    window_index INTEGER NOT NULL,
    target_ids_json TEXT NOT NULL,
    context_ids_json TEXT NOT NULL,
    prior_claim_ids_json TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'invalid', 'error')),
    raw_output TEXT,
    error TEXT,
    issues_json TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    PRIMARY KEY (conversation_id, window_index)
);
"""


class SemanticStore:
    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        self._conn.executescript(V2_SCHEMA)

    def has_claims(self, conversation_id: str) -> bool:
        return self._conn.execute("SELECT 1 FROM v2_claims WHERE conversation_id = ? LIMIT 1",
                                  (conversation_id,)).fetchone() is not None

    def delete_conversation(self, conversation_id: str) -> None:
        with self._conn:
            self.clear_conversation(conversation_id)

    def clear_conversation(self, conversation_id: str) -> None:
        """``delete_conversation`` inside the caller's open transaction; does not commit."""
        for table in ("v2_relations", "v2_claims", "v2_semantic_messages", "v2_atomization_log"):
            self._conn.execute(f"DELETE FROM {table} WHERE conversation_id = ?", (conversation_id,))

    def save_window(self, validated: ValidatedWindow, model: str, latency_ms: float,
                    prior_claim_ids: list[str] | tuple[str, ...] = ()) -> None:
        with self._conn:
            self.write_window(validated, model, latency_ms, prior_claim_ids)

    def write_window(self, validated: ValidatedWindow, model: str, latency_ms: float,
                     prior_claim_ids: list[str] | tuple[str, ...] = ()) -> None:
        """``save_window`` inside the caller's open transaction; does not commit."""
        window = validated.window
        cid = window.conversation_id
        for m in validated.messages:
            self._conn.execute(
                "INSERT INTO v2_semantic_messages (conversation_id, message_id, status, window_index) "
                "VALUES (?, ?, ?, ?)", (cid, m.source_message_id, m.status, window.index))
            for ordinal, c in enumerate(m.claims, 1):
                self._conn.execute(
                    "INSERT INTO v2_claims (conversation_id, claim_id, message_id, ordinal, content, "
                    "entities_json, qualifiers_json, temporal_json, polarity, claim_type, support, "
                    "resolved_from_json, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (cid, c.claim_id, m.source_message_id, ordinal, c.content, json.dumps(list(c.entities)),
                     json.dumps(list(c.qualifiers)), json.dumps(c.temporal.to_dict()) if c.temporal else None,
                     c.polarity, c.claim_type, c.support, json.dumps(list(c.resolved_from)),
                     json.dumps(c.metadata, sort_keys=True)))
        for m in validated.messages:
            for ordinal, r in enumerate(m.explicit_relations, 1):
                span = r.evidence_span or (None, None)
                self._conn.execute(
                    "INSERT INTO v2_relations (conversation_id, relation_id, from_claim_id, to_claim_id, "
                    "type, message_id, ordinal, evidence, evidence_start, evidence_end) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (cid, r.relation_id, r.from_claim_id, r.to_claim_id, r.type, r.source_message_id,
                     ordinal, r.evidence, span[0], span[1]))
        self._conn.execute(
            "INSERT INTO v2_atomization_log (conversation_id, window_index, target_ids_json, "
            "context_ids_json, prior_claim_ids_json, model, status, raw_output, error, issues_json, "
            "latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (cid, window.index, json.dumps(list(window.target_ids)), json.dumps(list(window.context_ids)),
             json.dumps(list(prior_claim_ids)), model, validated.status, validated.raw, validated.error,
             json.dumps([i.to_dict() for i in validated.issues]), latency_ms))

    # ---- reads ------------------------------------------------------------

    def _provenance(self, conversation_id: str) -> dict[str, tuple[Provenance, str]]:
        rows = self._conn.execute(
            "SELECT message_id, position, role, timestamp, content FROM messages WHERE conversation_id = ?",
            (conversation_id,))
        return {mid: (Provenance(conversation_id, mid, pos, role, ts), content)
                for mid, pos, role, ts, content in rows}

    def get_claims(self, conversation_id: str) -> list[Claim]:
        prov = self._provenance(conversation_id)
        rows = self._conn.execute(
            "SELECT c.claim_id, c.message_id, c.content, c.entities_json, c.qualifiers_json, c.temporal_json, "
            "c.polarity, c.claim_type, c.support, c.resolved_from_json, c.metadata_json FROM v2_claims c "
            "JOIN messages m ON m.conversation_id = c.conversation_id AND m.message_id = c.message_id "
            "WHERE c.conversation_id = ? ORDER BY m.position, c.ordinal", (conversation_id,))
        out = []
        for cid, mid, content, ents, quals, temporal, polarity, ctype, support, resolved, meta in rows:
            t = json.loads(temporal) if temporal else None
            out.append(Claim(cid, content, prov[mid][0], tuple(json.loads(ents)), tuple(json.loads(quals)),
                             Temporal(t["expression"], t.get("resolved_date")) if t else None, polarity, ctype,
                             support, tuple(json.loads(resolved)), json.loads(meta)))
        return out

    def get_relations(self, conversation_id: str) -> list[Relation]:
        rows = self._conn.execute(
            "SELECT r.relation_id, r.from_claim_id, r.to_claim_id, r.type, r.message_id, r.evidence, "
            "r.evidence_start, r.evidence_end FROM v2_relations r "
            "JOIN messages m ON m.conversation_id = r.conversation_id AND m.message_id = r.message_id "
            "WHERE r.conversation_id = ? ORDER BY m.position, r.ordinal", (conversation_id,))
        return [Relation(rid, src, dst, rtype, mid, ev, (s, e) if s is not None else None)
                for rid, src, dst, rtype, mid, ev, s, e in rows]

    def get_semantic_messages(self, conversation_id: str) -> list[SemanticMessage]:
        """Semantic records in message order; content, role and timestamp come from the raw store."""
        prov = self._provenance(conversation_id)
        claims: dict[str, list[Claim]] = {}
        for c in self.get_claims(conversation_id):
            claims.setdefault(c.source_message_id, []).append(c)
        relations: dict[str, list[Relation]] = {}
        for r in self.get_relations(conversation_id):
            relations.setdefault(r.source_message_id, []).append(r)
        rows = self._conn.execute(
            "SELECT s.message_id, s.status FROM v2_semantic_messages s JOIN messages m "
            "ON m.conversation_id = s.conversation_id AND m.message_id = s.message_id "
            "WHERE s.conversation_id = ? ORDER BY m.position", (conversation_id,))
        return [SemanticMessage(prov[mid][0], prov[mid][1], status, tuple(claims.get(mid, ())),
                                tuple(relations.get(mid, ()))) for mid, status in rows]

    def get_log(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT window_index, target_ids_json, context_ids_json, prior_claim_ids_json, model, status, "
            "raw_output, error, issues_json, latency_ms FROM v2_atomization_log WHERE conversation_id = ? "
            "ORDER BY window_index", (conversation_id,))
        keys = ("window_index", "target_ids", "context_ids", "prior_claim_ids", "model", "status", "raw_output",
                "error", "issues", "latency_ms")
        out = []
        for row in rows:
            d = dict(zip(keys, row))
            for k in ("target_ids", "context_ids", "prior_claim_ids", "issues"):
                d[k] = json.loads(d[k])
            out.append(d)
        return out
