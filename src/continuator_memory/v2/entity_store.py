"""SQLite persistence for phase 5B entities and memory <-> entity references.

Every key includes ``conversation_id``. An entity cascades from the claim that first referenced it,
so re-ingesting a conversation (which replaces its claims) removes its entities, their names and
references; ``save`` replaces a conversation's snapshot, and deterministic IDs make a re-run write
the same rows. Names and references point at raw messages and claims; no message text is copied.
"""

from __future__ import annotations

import sqlite3

from continuator_memory.v2.consolidation import _ordinal
from continuator_memory.v2.context import ContextualMemory
from continuator_memory.v2.context_store import ContextStore, raw_messages, stored_contextual_memories
from continuator_memory.v2.entities import (
    ENTITY_STATUSES,
    ENTITY_TYPES,
    NAME_KINDS,
    REF_ROLES,
    REF_STATUSES,
    Entity,
    EntityName,
    EntityRef,
    EntityResolution,
    EntityTyper,
    resolve_entities,
)
from continuator_memory.v2.memory_store import MemoryStore
from continuator_memory.v2.store import SemanticStore


def _values(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


ENTITY_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS v2_entities (
    conversation_id TEXT NOT NULL
        REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    local_key TEXT NOT NULL,
    canonical_name TEXT,
    entity_type TEXT NOT NULL CHECK (entity_type IN ({_values(ENTITY_TYPES)})),
    status TEXT NOT NULL CHECK (status IN ({_values(ENTITY_STATUSES)})),
    speaker_role TEXT,
    established_by_claim_id TEXT NOT NULL,
    established_by_message_id TEXT NOT NULL,
    PRIMARY KEY (conversation_id, entity_id),
    FOREIGN KEY (conversation_id, established_by_claim_id)
        REFERENCES v2_claims(conversation_id, claim_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS v2_entity_aliases (
    conversation_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
    kind TEXT NOT NULL CHECK (kind IN ({_values(NAME_KINDS)})),
    message_id TEXT NOT NULL,
    claim_id TEXT,
    PRIMARY KEY (conversation_id, entity_id, ordinal),
    FOREIGN KEY (conversation_id, entity_id)
        REFERENCES v2_entities(conversation_id, entity_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, message_id)
        REFERENCES messages(conversation_id, message_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS v2_memory_entities (
    conversation_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ({_values(REF_ROLES)})),
    entity_id TEXT,
    name TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ({_values(REF_STATUSES)})),
    detail TEXT,
    PRIMARY KEY (conversation_id, memory_id, ordinal),
    CHECK ((entity_id IS NULL) = (status = 'ambiguous')),
    FOREIGN KEY (conversation_id, memory_id)
        REFERENCES v2_memories(conversation_id, memory_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, entity_id)
        REFERENCES v2_entities(conversation_id, entity_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, claim_id)
        REFERENCES v2_claims(conversation_id, claim_id) ON DELETE CASCADE
);
"""

_TABLES = ("v2_memory_entities", "v2_entity_aliases", "v2_entities")


class EntityStore:
    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        SemanticStore(connection)  # v2_claims
        MemoryStore(connection)  # v2_memories
        self._conn.executescript(ENTITY_SCHEMA)

    def clear_conversation(self, conversation_id: str) -> None:
        """Delete entity rows inside the caller's open transaction; does not commit."""
        for table in _TABLES:
            self._conn.execute(f"DELETE FROM {table} WHERE conversation_id = ?", (conversation_id,))

    def delete_conversation(self, conversation_id: str) -> None:
        with self._conn:
            self.clear_conversation(conversation_id)

    def save(self, resolution: EntityResolution) -> None:
        cid = resolution.conversation_id
        self.delete_conversation(cid)
        with self._conn:
            for ordinal, e in enumerate(resolution.entities.values(), 1):
                self._conn.execute(
                    "INSERT INTO v2_entities (conversation_id, entity_id, ordinal, local_key, canonical_name, "
                    "entity_type, status, speaker_role, established_by_claim_id, established_by_message_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (cid, e.entity_id, ordinal, e.local_key, e.canonical_name, e.entity_type, e.status,
                     e.speaker_role, e.established_by_claim_id, e.established_by_message_id))
                for n_ordinal, n in enumerate(e.names, 1):
                    self._conn.execute(
                        "INSERT INTO v2_entity_aliases (conversation_id, entity_id, ordinal, name, kind, message_id, "
                        "claim_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (cid, e.entity_id, n_ordinal, n.name, n.kind, n.message_id, n.claim_id))
            for memory_id, refs in resolution.refs.items():
                for ordinal, r in enumerate(refs, 1):
                    self._conn.execute(
                        "INSERT INTO v2_memory_entities (conversation_id, memory_id, ordinal, role, entity_id, name, "
                        "claim_id, message_id, status, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (cid, memory_id, ordinal, r.role, r.entity_id, r.name, r.claim_id, r.message_id, r.status,
                         r.detail))

    def get(self, conversation_id: str) -> EntityResolution:
        names: dict[str, list[EntityName]] = {}
        for eid, name, kind, mid, claim_id in self._conn.execute(
                "SELECT entity_id, name, kind, message_id, claim_id FROM v2_entity_aliases "
                "WHERE conversation_id = ? ORDER BY entity_id, ordinal", (conversation_id,)):
            names.setdefault(eid, []).append(EntityName(name, kind, mid, claim_id))
        out = EntityResolution(conversation_id)
        for eid, key, canonical, etype, status, role, claim_id, mid in self._conn.execute(
                "SELECT entity_id, local_key, canonical_name, entity_type, status, speaker_role, "
                "established_by_claim_id, established_by_message_id FROM v2_entities "
                "WHERE conversation_id = ? ORDER BY ordinal", (conversation_id,)):
            out.entities[eid] = Entity(eid, conversation_id, key, canonical, etype, status, role, claim_id, mid,
                                       tuple(names.get(eid, ())))
        refs: dict[str, list[EntityRef]] = {}
        for memory_id, role, eid, name, claim_id, mid, status, detail in self._conn.execute(
                "SELECT memory_id, role, entity_id, name, claim_id, message_id, status, detail "
                "FROM v2_memory_entities WHERE conversation_id = ? ORDER BY memory_id, ordinal", (conversation_id,)):
            refs.setdefault(memory_id, []).append(EntityRef(role, claim_id, mid, name, eid, status, detail))
        memories = sorted(MemoryStore(self._conn).get_memories(conversation_id),
                          key=lambda m: (m.position, _ordinal(m.claim_id), m.memory_id))
        out.refs = {m.memory_id: tuple(refs[m.memory_id]) for m in memories if m.memory_id in refs}
        return out


def resolve_entities_stored(connection: sqlite3.Connection, conversation_id: str,
                            typer: EntityTyper | None = None) -> EntityResolution:
    """Resolve entities from the stored claims, claim contexts and memories, and save them.

    Run after ``consolidate_stored`` and ``contextualize_stored``: saving memories again replaces
    them and cascades their entity references away."""
    memories = MemoryStore(connection)
    resolution = resolve_entities(conversation_id, raw_messages(connection, conversation_id),
                                  SemanticStore(connection).get_claims(conversation_id),
                                  ContextStore(connection).get_contexts(conversation_id),
                                  memories.get_memories(conversation_id), memories.get_sources(conversation_id),
                                  typer)
    EntityStore(connection).save(resolution)
    return resolution


def stored_entity_memories(connection: sqlite3.Connection, conversation_id: str) -> list[ContextualMemory]:
    """Stored contextual memories with their stored entity references attached."""
    return EntityStore(connection).get(conversation_id).attach(stored_contextual_memories(connection, conversation_id))
