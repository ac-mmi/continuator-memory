"""SQLite persistence for V2 consolidated memories (phase 2A).

Memories are derived deterministically from the stored claim stream, so ``save`` writes a
snapshot of a ``Consolidator`` for one conversation, replacing any earlier snapshot. Sources
reference raw messages by (conversation_id, message_id); nothing here copies message content.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable

from continuator_memory.v2.consolidation import Consolidator
from continuator_memory.v2.memory import (
    ACTIONS,
    LINK_TYPES,
    MEMORY_STATUSES,
    Decision,
    Memory,
    MemoryLink,
    MemorySource,
)
from continuator_memory.v2.models import Claim, Relation
from continuator_memory.v2.store import SemanticStore

_STATUSES = ", ".join(f"'{s}'" for s in MEMORY_STATUSES)
_LINKS = ", ".join(f"'{t}'" for t in LINK_TYPES)
_ACTIONS = ", ".join(f"'{a}'" for a in ACTIONS)

MEMORY_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS v2_memories (
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL,
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    status TEXT NOT NULL CHECK (status IN ({_STATUSES})),
    claim_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    created_at TEXT,
    valid_from TEXT,
    valid_until TEXT,
    subject TEXT,
    entities_json TEXT NOT NULL,
    claim_type TEXT,
    polarity TEXT,
    temporal TEXT,
    superseded_by TEXT,
    event_time TEXT,
    time_class TEXT NOT NULL,
    attribute TEXT,
    PRIMARY KEY (conversation_id, memory_id)
);

CREATE TABLE IF NOT EXISTS v2_memory_sources (
    conversation_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    observed_at TEXT,
    PRIMARY KEY (conversation_id, memory_id, claim_id),
    FOREIGN KEY (conversation_id, memory_id)
        REFERENCES v2_memories(conversation_id, memory_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, message_id)
        REFERENCES messages(conversation_id, message_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS v2_memory_versions (
    conversation_id TEXT NOT NULL,
    newer_memory_id TEXT NOT NULL,
    older_memory_id TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ({_LINKS})),
    claim_id TEXT NOT NULL,
    relation_id TEXT,
    ordinal INTEGER NOT NULL,
    PRIMARY KEY (conversation_id, newer_memory_id, older_memory_id, type),
    FOREIGN KEY (conversation_id, newer_memory_id)
        REFERENCES v2_memories(conversation_id, memory_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, older_memory_id)
        REFERENCES v2_memories(conversation_id, memory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS v2_memory_decisions (
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    claim_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ({_ACTIONS})),
    memory_id TEXT,
    superseded_json TEXT NOT NULL,
    contradicted_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    unresolved TEXT,
    PRIMARY KEY (conversation_id, ordinal)
);
"""

_TABLES = ("v2_memory_decisions", "v2_memory_versions", "v2_memory_sources", "v2_memories")


class MemoryStore:
    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        self._conn.executescript(MEMORY_SCHEMA)

    def clear_conversation(self, conversation_id: str) -> None:
        """Delete memory snapshot rows inside the caller's open transaction; does not commit."""
        for table in _TABLES:
            self._conn.execute(f"DELETE FROM {table} WHERE conversation_id = ?", (conversation_id,))

    def delete_conversation(self, conversation_id: str) -> None:
        with self._conn:
            self.clear_conversation(conversation_id)

    def save(self, consolidator: Consolidator) -> None:
        cid = consolidator.conversation_id
        self.delete_conversation(cid)
        with self._conn:
            for m in consolidator.memories.values():
                self._conn.execute(
                    "INSERT INTO v2_memories (conversation_id, memory_id, content, status, claim_id, message_id, "
                    "position, created_at, valid_from, valid_until, subject, entities_json, claim_type, polarity, "
                    "temporal, superseded_by, event_time, time_class, attribute) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (cid, m.memory_id, m.content, m.status, m.claim_id, m.message_id, m.position, m.created_at,
                     m.valid_from, m.valid_until, m.subject, json.dumps(list(m.entities)), m.claim_type,
                     m.polarity, m.temporal, m.superseded_by, m.event_time, m.time_class, m.attribute))
            for s in consolidator.sources:
                self._conn.execute(
                    "INSERT INTO v2_memory_sources (conversation_id, memory_id, claim_id, message_id, position, "
                    "observed_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (cid, s.memory_id, s.claim_id, s.message_id, s.position, s.observed_at))
            for ordinal, link in enumerate(consolidator.links, 1):
                self._conn.execute(
                    "INSERT INTO v2_memory_versions (conversation_id, newer_memory_id, older_memory_id, type, "
                    "claim_id, relation_id, ordinal) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (cid, link.newer_memory_id, link.older_memory_id, link.type, link.claim_id,
                     link.relation_id, ordinal))
            for ordinal, d in enumerate(consolidator.decisions, 1):
                self._conn.execute(
                    "INSERT INTO v2_memory_decisions (conversation_id, ordinal, claim_id, action, memory_id, "
                    "superseded_json, contradicted_json, reason, unresolved) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (cid, ordinal, d.claim_id, d.action, d.memory_id, json.dumps(list(d.superseded)),
                     json.dumps(list(d.contradicted)), d.reason, d.unresolved))

    # ---- reads ------------------------------------------------------------

    def get_memories(self, conversation_id: str, current_only: bool = False) -> list[Memory]:
        sql = ("SELECT memory_id, content, status, claim_id, message_id, position, created_at, valid_from, "
               "valid_until, subject, entities_json, claim_type, polarity, temporal, superseded_by, event_time, "
               "time_class, attribute FROM v2_memories WHERE conversation_id = ?")
        if current_only:
            sql += " AND status IN ('current', 'contested')"
        rows = self._conn.execute(sql + " ORDER BY position, memory_id", (conversation_id,))
        return [Memory(mid, conversation_id, content, status, claim_id, message_id, position, created_at,
                       valid_from, valid_until, subject, tuple(json.loads(entities)), claim_type, polarity,
                       temporal, superseded_by, event_time, time_class, attribute)
                for (mid, content, status, claim_id, message_id, position, created_at, valid_from, valid_until,
                     subject, entities, claim_type, polarity, temporal, superseded_by, event_time,
                     time_class, attribute) in rows]

    def get_sources(self, conversation_id: str, memory_id: str | None = None) -> list[MemorySource]:
        sql = ("SELECT memory_id, claim_id, message_id, position, observed_at FROM v2_memory_sources "
               "WHERE conversation_id = ?")
        args: tuple = (conversation_id,)
        if memory_id is not None:
            sql += " AND memory_id = ?"
            args += (memory_id,)
        rows = self._conn.execute(sql + " ORDER BY position, claim_id", args)
        return [MemorySource(*row) for row in rows]

    def get_links(self, conversation_id: str) -> list[MemoryLink]:
        rows = self._conn.execute(
            "SELECT newer_memory_id, older_memory_id, type, claim_id, relation_id FROM v2_memory_versions "
            "WHERE conversation_id = ? ORDER BY ordinal", (conversation_id,))
        return [MemoryLink(*row) for row in rows]

    def get_decisions(self, conversation_id: str) -> list[Decision]:
        rows = self._conn.execute(
            "SELECT claim_id, action, memory_id, superseded_json, contradicted_json, reason, unresolved "
            "FROM v2_memory_decisions WHERE conversation_id = ? ORDER BY ordinal", (conversation_id,))
        return [Decision(claim_id, action, mid, tuple(json.loads(sup)), tuple(json.loads(con)), reason, unresolved)
                for claim_id, action, mid, sup, con, reason, unresolved in rows]

    def has_snapshot(self, conversation_id: str) -> bool:
        """True when any consolidated memory row exists for the conversation."""
        for table in _TABLES:
            if self._conn.execute(
                f"SELECT 1 FROM {table} WHERE conversation_id = ? LIMIT 1", (conversation_id,)
            ).fetchone():
                return True
        return False

    def load_consolidator(self, conversation_id: str) -> Consolidator | None:
        """Hydrate a Consolidator from a persisted snapshot, or None if none is stored.

        Does not re-run consolidation. Identity indexes used only during ``apply`` are left
        empty; memories, sources, links, decisions and claim→memory provenance are restored.
        """
        if not self.has_snapshot(conversation_id):
            return None
        consolidator = Consolidator(conversation_id)
        for memory in self.get_memories(conversation_id):
            consolidator.memories[memory.memory_id] = memory
        consolidator.sources = self.get_sources(conversation_id)
        consolidator.links = self.get_links(conversation_id)
        consolidator.decisions = self.get_decisions(conversation_id)
        for source in consolidator.sources:
            consolidator._memory_of[source.claim_id] = source.memory_id
        return consolidator


def consolidate_stored(connection: sqlite3.Connection, conversation_id: str) -> Consolidator:
    """Consolidate the conversation's stored V2 claims and relations and save the result."""
    semantic = SemanticStore(connection)
    consolidator = Consolidator(conversation_id)
    consolidator.apply(semantic.get_claims(conversation_id), semantic.get_relations(conversation_id))
    MemoryStore(connection).save(consolidator)
    return consolidator


def resolve_consolidator(
    connection: sqlite3.Connection,
    conversation_id: str,
    claims: Iterable[Claim] | None = None,
    relations: Iterable[Relation] | None = None,
    *,
    persist_if_missing: bool = False,
) -> Consolidator:
    """Prefer a persisted memory snapshot; otherwise consolidate (optionally persist).

    Product retrieval and Workspace listing use this so older databases without a
    snapshot still work, while successful V2 ingests write snapshots eagerly.
    """
    from continuator_memory.v2.consolidation import consolidate

    store = MemoryStore(connection)
    loaded = store.load_consolidator(conversation_id)
    if loaded is not None:
        return loaded
    if claims is None or relations is None:
        semantic = SemanticStore(connection)
        claims = semantic.get_claims(conversation_id) if claims is None else claims
        relations = semantic.get_relations(conversation_id) if relations is None else relations
    consolidator = consolidate(conversation_id, claims, relations)
    if persist_if_missing:
        store.save(consolidator)
    return consolidator
