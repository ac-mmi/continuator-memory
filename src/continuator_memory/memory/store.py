"""SQLite persistence for entities, memories, relationships and formation logs.

Lives in the same database as the Phase 1 tables; every row cascades from
``conversations`` so re-ingesting a conversation rebuilds its memory.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from continuator_memory.memory.models import (
    Entity,
    Memory,
    Relationship,
    normalize_name,
    normalize_value,
)

MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    UNIQUE (conversation_id, normalized_name)
);

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    normalized_value TEXT NOT NULL,
    kind TEXT NOT NULL,
    observed_position INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    valid_from TEXT,
    valid_until TEXT,
    is_current INTEGER NOT NULL CHECK (is_current IN (0, 1)),
    supersedes_memory_id TEXT REFERENCES memories(memory_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS memories_slot ON memories (entity_id, predicate, is_current);

CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (memory_id, message_id),
    FOREIGN KEY (conversation_id, message_id)
        REFERENCES messages(conversation_id, message_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS relationships (
    relationship_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    subject_entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object_entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    observed_position INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    valid_from TEXT,
    valid_until TEXT,
    is_current INTEGER NOT NULL CHECK (is_current IN (0, 1)),
    supersedes_relationship_id TEXT
        REFERENCES relationships(relationship_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS relationships_subject ON relationships (subject_entity_id, predicate);
CREATE INDEX IF NOT EXISTS relationships_object ON relationships (object_entity_id, predicate);

CREATE TABLE IF NOT EXISTS relationship_sources (
    relationship_id TEXT NOT NULL
        REFERENCES relationships(relationship_id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (relationship_id, message_id),
    FOREIGN KEY (conversation_id, message_id)
        REFERENCES messages(conversation_id, message_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS extraction_log (
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    message_id TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ok', 'invalid', 'error')),
    n_memories INTEGER NOT NULL,
    n_relationships INTEGER NOT NULL,
    rejected_json TEXT NOT NULL,
    raw_output TEXT,
    error TEXT,
    latency_ms REAL NOT NULL,
    grounding_json TEXT,
    PRIMARY KEY (conversation_id, message_id)
);

CREATE TABLE IF NOT EXISTS consolidation_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    message_id TEXT NOT NULL,
    target TEXT NOT NULL CHECK (target IN ('memory', 'relationship')),
    action TEXT NOT NULL,
    record_id TEXT,
    superseded_ids_json TEXT NOT NULL,
    reason TEXT NOT NULL
);
"""

_MEMORY_SELECT = """
SELECT m.memory_id, m.conversation_id, m.entity_id, e.name, m.predicate, m.value, m.kind,
       m.observed_position, m.created_at, m.valid_from, m.valid_until, m.is_current,
       m.supersedes_memory_id
FROM memories m JOIN entities e ON e.entity_id = m.entity_id
"""

_REL_SELECT = """
SELECT r.relationship_id, r.conversation_id, r.subject_entity_id, s.name, r.predicate,
       r.object_entity_id, o.name, r.kind, r.observed_position, r.created_at, r.valid_from,
       r.valid_until, r.is_current, r.supersedes_relationship_id
FROM relationships r
JOIN entities s ON s.entity_id = r.subject_entity_id
JOIN entities o ON o.entity_id = r.object_entity_id
"""


def _alias_match(short: str, long: str, long_raw: str) -> bool:
    """``short`` names the same entity as ``long`` (e.g. 'Hana' / 'Hana Kobayashi')."""
    if "user" in (short, long) or "'s" in long_raw or "’s" in long_raw:
        return False
    s, l_ = short.split(), long.split()
    if not s or len(s) >= len(l_) or len(short) < 3:
        return False
    return l_[: len(s)] == s or l_[-len(s):] == s


class MemoryStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.executescript(MEMORY_SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(extraction_log)")}
        if "grounding_json" not in columns:
            conn.execute("ALTER TABLE extraction_log ADD COLUMN grounding_json TEXT")

    # ---- ids --------------------------------------------------------------

    def _next_id(self, table: str, conversation_id: str, prefix: str) -> str:
        (n,) = self.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return f"{conversation_id}:{prefix}{n + 1:04d}"

    # ---- entities ---------------------------------------------------------

    def entities(self, conversation_id: str | None = None) -> list[Entity]:
        sql = "SELECT entity_id, conversation_id, name, normalized_name FROM entities"
        params: tuple = ()
        if conversation_id is not None:
            sql += " WHERE conversation_id = ?"
            params = (conversation_id,)
        return [Entity(*r) for r in self.conn.execute(sql + " ORDER BY entity_id", params)]

    def get_entity(self, entity_id: str) -> Entity | None:
        row = self.conn.execute(
            "SELECT entity_id, conversation_id, name, normalized_name FROM entities "
            "WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        return Entity(*row) if row else None

    def find_entity(self, conversation_id: str, name: str) -> Entity | None:
        key = normalize_name(name)
        if not key:
            return None
        candidates = self.entities(conversation_id)
        for e in candidates:
            if e.normalized_name == key:
                return e
        aliases = [
            e
            for e in candidates
            if _alias_match(key, e.normalized_name, e.name)
            or _alias_match(e.normalized_name, key, name)
        ]
        return aliases[0] if len(aliases) == 1 else None

    def resolve_entity(self, conversation_id: str, name: str) -> Entity:
        """Return the matching entity, creating it if needed. A longer alias renames it."""
        name = name.strip()
        key = normalize_name(name)
        existing = self.find_entity(conversation_id, name)
        if existing is not None:
            if len(key.split()) > len(existing.normalized_name.split()):
                self.conn.execute(
                    "UPDATE entities SET name = ?, normalized_name = ? WHERE entity_id = ?",
                    (name, key, existing.entity_id),
                )
                return Entity(existing.entity_id, conversation_id, name, key)
            return existing
        entity_id = self._next_id("entities", conversation_id, "ent")
        display = "user" if key == "user" else name
        self.conn.execute(
            "INSERT INTO entities (entity_id, conversation_id, name, normalized_name) "
            "VALUES (?, ?, ?, ?)",
            (entity_id, conversation_id, display, key),
        )
        return Entity(entity_id, conversation_id, display, key)

    # ---- memories ---------------------------------------------------------

    def _sources(self, table: str, id_col: str, record_id: str) -> tuple[str, ...]:
        rows = self.conn.execute(
            f"SELECT message_id FROM {table} WHERE {id_col} = ? ORDER BY position, message_id",
            (record_id,),
        )
        return tuple(r[0] for r in rows)

    def _add_source(self, table: str, id_col: str, record_id: str, conv: str, message_id: str):
        self.conn.execute(
            f"INSERT OR IGNORE INTO {table} ({id_col}, conversation_id, message_id, position) "
            "SELECT ?, ?, ?, position FROM messages WHERE conversation_id = ? AND message_id = ?",
            (record_id, conv, message_id, conv, message_id),
        )

    def insert_memory(
        self,
        conversation_id: str,
        entity: Entity,
        predicate: str,
        value: str,
        kind: str,
        message_id: str,
        observed_position: int,
        created_at: str,
        valid_from: str | None,
        is_current: bool = True,
        supersedes_memory_id: str | None = None,
    ) -> str:
        memory_id = self._next_id("memories", conversation_id, "mem")
        self.conn.execute(
            "INSERT INTO memories (memory_id, conversation_id, entity_id, predicate, value, "
            "normalized_value, kind, observed_position, created_at, valid_from, valid_until, "
            "is_current, supersedes_memory_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                memory_id, conversation_id, entity.entity_id, predicate, value,
                normalize_value(value), kind, observed_position, created_at, valid_from,
                int(is_current), supersedes_memory_id,
            ),
        )
        self._add_source("memory_sources", "memory_id", memory_id, conversation_id, message_id)
        return memory_id

    def add_memory_source(self, memory_id: str, conversation_id: str, message_id: str) -> None:
        self._add_source("memory_sources", "memory_id", memory_id, conversation_id, message_id)

    def close_memory(self, memory_id: str, valid_until: str | None) -> None:
        self.conn.execute(
            "UPDATE memories SET is_current = 0, valid_until = ? WHERE memory_id = ?",
            (valid_until, memory_id),
        )

    def set_supersedes(self, memory_id: str, supersedes_id: str) -> None:
        self.conn.execute(
            "UPDATE memories SET supersedes_memory_id = ? WHERE memory_id = ?",
            (supersedes_id, memory_id),
        )

    def _to_memory(self, row: tuple) -> Memory:
        return Memory(
            memory_id=row[0], conversation_id=row[1], entity_id=row[2], subject=row[3],
            predicate=row[4], value=row[5], kind=row[6],
            source_message_ids=self._sources("memory_sources", "memory_id", row[0]),
            observed_position=row[7], created_at=row[8], valid_from=row[9], valid_until=row[10],
            is_current=bool(row[11]), supersedes_memory_id=row[12],
        )

    def memories(
        self,
        conversation_id: str | None = None,
        entity_id: str | None = None,
        predicate: str | None = None,
        current_only: bool = False,
    ) -> list[Memory]:
        clauses, params = [], []
        for col, val in (
            ("m.conversation_id", conversation_id),
            ("m.entity_id", entity_id),
            ("m.predicate", predicate),
        ):
            if val is not None:
                clauses.append(f"{col} = ?")
                params.append(val)
        if current_only:
            clauses.append("m.is_current = 1")
        sql = _MEMORY_SELECT + (" WHERE " + " AND ".join(clauses) if clauses else "")
        sql += " ORDER BY m.conversation_id, m.observed_position, m.memory_id"
        return [self._to_memory(r) for r in self.conn.execute(sql, params)]

    def get_memory(self, memory_id: str) -> Memory | None:
        row = self.conn.execute(_MEMORY_SELECT + " WHERE m.memory_id = ?", (memory_id,)).fetchone()
        return self._to_memory(row) if row else None

    # ---- relationships ----------------------------------------------------

    def insert_relationship(
        self,
        conversation_id: str,
        subject: Entity,
        predicate: str,
        obj: Entity,
        kind: str,
        message_id: str,
        observed_position: int,
        created_at: str,
        valid_from: str | None,
        is_current: bool = True,
        supersedes_relationship_id: str | None = None,
    ) -> str:
        rel_id = self._next_id("relationships", conversation_id, "rel")
        self.conn.execute(
            "INSERT INTO relationships (relationship_id, conversation_id, subject_entity_id, "
            "predicate, object_entity_id, kind, observed_position, created_at, valid_from, "
            "valid_until, is_current, supersedes_relationship_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                rel_id, conversation_id, subject.entity_id, predicate, obj.entity_id, kind,
                observed_position, created_at, valid_from, int(is_current),
                supersedes_relationship_id,
            ),
        )
        self._add_source(
            "relationship_sources", "relationship_id", rel_id, conversation_id, message_id
        )
        return rel_id

    def add_relationship_source(self, rel_id: str, conversation_id: str, message_id: str) -> None:
        self._add_source(
            "relationship_sources", "relationship_id", rel_id, conversation_id, message_id
        )

    def close_relationship(self, rel_id: str, valid_until: str | None) -> None:
        self.conn.execute(
            "UPDATE relationships SET is_current = 0, valid_until = ? WHERE relationship_id = ?",
            (valid_until, rel_id),
        )

    def set_relationship_supersedes(self, rel_id: str, supersedes_id: str) -> None:
        self.conn.execute(
            "UPDATE relationships SET supersedes_relationship_id = ? WHERE relationship_id = ?",
            (supersedes_id, rel_id),
        )

    def _to_relationship(self, row: tuple) -> Relationship:
        return Relationship(
            relationship_id=row[0], conversation_id=row[1], subject_entity_id=row[2],
            subject=row[3], predicate=row[4], object_entity_id=row[5], object=row[6],
            kind=row[7],
            source_message_ids=self._sources("relationship_sources", "relationship_id", row[0]),
            observed_position=row[8], created_at=row[9], valid_from=row[10],
            valid_until=row[11], is_current=bool(row[12]), supersedes_relationship_id=row[13],
        )

    def relationships(
        self,
        conversation_id: str | None = None,
        entity_id: str | None = None,
        predicate: str | None = None,
        current_only: bool = False,
    ) -> list[Relationship]:
        """Relationships filtered by conversation, predicate, and/or an entity on either end."""
        clauses, params = [], []
        if conversation_id is not None:
            clauses.append("r.conversation_id = ?")
            params.append(conversation_id)
        if entity_id is not None:
            clauses.append("(r.subject_entity_id = ? OR r.object_entity_id = ?)")
            params += [entity_id, entity_id]
        if predicate is not None:
            clauses.append("r.predicate = ?")
            params.append(predicate)
        if current_only:
            clauses.append("r.is_current = 1")
        sql = _REL_SELECT + (" WHERE " + " AND ".join(clauses) if clauses else "")
        sql += " ORDER BY r.conversation_id, r.observed_position, r.relationship_id"
        return [self._to_relationship(r) for r in self.conn.execute(sql, params)]

    def neighbors(self, entity_id: str, current_only: bool = False) -> list[tuple[Relationship, str]]:
        """One-hop traversal: (relationship, id of the entity on the other end)."""
        out = []
        for rel in self.relationships(entity_id=entity_id, current_only=current_only):
            other = rel.object_entity_id if rel.subject_entity_id == entity_id else rel.subject_entity_id
            out.append((rel, other))
        return out

    # ---- logs -------------------------------------------------------------

    def log_extraction(
        self,
        conversation_id: str,
        message_id: str,
        model: str,
        status: str,
        n_memories: int,
        n_relationships: int,
        rejected: list[str],
        raw_output: str | None,
        error: str | None,
        latency_ms: float,
        grounding: list[dict[str, Any]] | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO extraction_log (conversation_id, message_id, model, status, "
            "n_memories, n_relationships, rejected_json, raw_output, error, latency_ms, "
            "grounding_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id, message_id, model, status, n_memories, n_relationships,
                json.dumps(rejected), raw_output, error, latency_ms,
                None if grounding is None else json.dumps(grounding),
            ),
        )

    def log_consolidation(
        self,
        conversation_id: str,
        message_id: str,
        target: str,
        action: str,
        record_id: str | None,
        superseded_ids: list[str],
        reason: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO consolidation_log (conversation_id, message_id, target, action, "
            "record_id, superseded_ids_json, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, message_id, target, action, record_id, json.dumps(superseded_ids),
             reason),
        )

    def stats(self, conversation_id: str | None = None) -> dict[str, Any]:
        where, params = ("WHERE conversation_id = ?", (conversation_id,)) if conversation_id else ("", ())

        def scalar(sql: str) -> int:
            return self.conn.execute(sql, params).fetchone()[0] or 0

        actions: dict[str, dict[str, int]] = {"memory": {}, "relationship": {}}
        for target, action, n in self.conn.execute(
            f"SELECT target, action, COUNT(*) FROM consolidation_log {where} "
            "GROUP BY target, action ORDER BY target, action",
            params,
        ):
            actions[target][action] = n
        statuses = dict(
            self.conn.execute(
                f"SELECT status, COUNT(*) FROM extraction_log {where} GROUP BY status", params
            ).fetchall()
        )
        return {
            "entities": scalar(f"SELECT COUNT(*) FROM entities {where}"),
            "memories": scalar(f"SELECT COUNT(*) FROM memories {where}"),
            "memories_current": scalar(
                f"SELECT COUNT(*) FROM memories {where} {'AND' if where else 'WHERE'} is_current = 1"
            ),
            "relationships": scalar(f"SELECT COUNT(*) FROM relationships {where}"),
            "relationships_current": scalar(
                f"SELECT COUNT(*) FROM relationships {where} "
                f"{'AND' if where else 'WHERE'} is_current = 1"
            ),
            "extraction_status": statuses,
            "extracted_memories": scalar(f"SELECT SUM(n_memories) FROM extraction_log {where}"),
            "extracted_relationships": scalar(
                f"SELECT SUM(n_relationships) FROM extraction_log {where}"
            ),
            "consolidation_actions": actions,
            "extraction_latency_ms": round(
                self.conn.execute(
                    f"SELECT COALESCE(SUM(latency_ms), 0) FROM extraction_log {where}", params
                ).fetchone()[0],
                3,
            ),
        }
