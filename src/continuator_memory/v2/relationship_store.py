"""SQLite persistence for phase 5C EXTENDS / DERIVES relationships.

The phase 2A-2C links stay in ``v2_memory_versions``, written by ``MemoryStore`` exactly as
before; retrieval reads that table to decide which memory is newer, so additive edges are kept
out of it. ``v2_memory_relationships`` holds only EXTENDS / DERIVES, keyed like the link table
by (conversation_id, source, target, type). Every edge cascades from both memories, the claim
that created it and its validated relation, so re-consolidating or re-ingesting a conversation
removes its edges; ``save`` replaces a conversation's snapshot and a re-run writes the same rows.
"""

from __future__ import annotations

import sqlite3

from continuator_memory.v2.memory_store import MemoryStore
from continuator_memory.v2.relationships import (
    ADDITIVE_ORIGINS,
    ADDITIVE_TYPES,
    MemoryRelationship,
    RelationshipGraph,
    build_relationships,
)
from continuator_memory.v2.store import SemanticStore


def _values(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


RELATIONSHIP_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS v2_memory_relationships (
    conversation_id TEXT NOT NULL,
    source_memory_id TEXT NOT NULL,
    target_memory_id TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ({_values(ADDITIVE_TYPES)})),
    ordinal INTEGER NOT NULL,
    claim_id TEXT NOT NULL,
    relation_id TEXT,
    origin TEXT NOT NULL CHECK (origin IN ({_values(ADDITIVE_ORIGINS)})),
    reason TEXT NOT NULL,
    PRIMARY KEY (conversation_id, source_memory_id, target_memory_id, type),
    UNIQUE (conversation_id, ordinal),
    CHECK (source_memory_id <> target_memory_id),
    FOREIGN KEY (conversation_id, source_memory_id)
        REFERENCES v2_memories(conversation_id, memory_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, target_memory_id)
        REFERENCES v2_memories(conversation_id, memory_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, claim_id)
        REFERENCES v2_claims(conversation_id, claim_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id, relation_id)
        REFERENCES v2_relations(conversation_id, relation_id) ON DELETE CASCADE
);
"""


class RelationshipStore:
    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        SemanticStore(connection)  # v2_claims, v2_relations
        MemoryStore(connection)  # v2_memories
        self._conn.executescript(RELATIONSHIP_SCHEMA)

    def clear_conversation(self, conversation_id: str) -> None:
        """Delete relationship rows inside the caller's open transaction; does not commit."""
        self._conn.execute(
            "DELETE FROM v2_memory_relationships WHERE conversation_id = ?",
            (conversation_id,),
        )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._conn:
            self.clear_conversation(conversation_id)

    def save(self, graph: RelationshipGraph) -> None:
        cid = graph.conversation_id
        self.delete_conversation(cid)
        with self._conn:
            for ordinal, r in enumerate(graph.additive, 1):
                self._conn.execute(
                    "INSERT INTO v2_memory_relationships (conversation_id, source_memory_id, target_memory_id, type, "
                    "ordinal, claim_id, relation_id, origin, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (cid, r.source_memory_id, r.target_memory_id, r.type, ordinal, r.claim_id, r.relation_id,
                     r.origin, r.reason))

    def get(self, conversation_id: str) -> RelationshipGraph:
        """The stored graph: stored memories and links plus the stored EXTENDS / DERIVES edges."""
        memories = MemoryStore(self._conn)
        graph = RelationshipGraph(conversation_id, memories.get_memories(conversation_id),
                                  memories.get_links(conversation_id), memories.get_decisions(conversation_id))
        for row in self._conn.execute(
                "SELECT source_memory_id, target_memory_id, type, claim_id, relation_id, origin, reason "
                "FROM v2_memory_relationships WHERE conversation_id = ? ORDER BY ordinal", (conversation_id,)):
            rejection = graph.add(MemoryRelationship(conversation_id, *row))
            if rejection is not None:
                raise ValueError(f"stored relationship is invalid: {rejection.code}: {rejection.detail}")
        return graph


def build_relationships_stored(connection: sqlite3.Connection, conversation_id: str) -> RelationshipGraph:
    """Build the relationship graph from the stored consolidation state, claims and relations,
    and save its EXTENDS / DERIVES edges. Run after ``consolidate_stored``: saving memories again
    replaces them and cascades their edges away."""
    semantic, memories = SemanticStore(connection), MemoryStore(connection)
    graph = build_relationships(conversation_id, memories.get_memories(conversation_id),
                                memories.get_links(conversation_id), memories.get_sources(conversation_id),
                                memories.get_decisions(conversation_id), semantic.get_claims(conversation_id),
                                semantic.get_relations(conversation_id))
    RelationshipStore(connection).save(graph)
    return graph
