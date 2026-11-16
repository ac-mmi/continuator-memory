"""Load the stored phase 5A-5C knowledge of a conversation for query-time expansion (phase 5D)."""

from __future__ import annotations

import sqlite3

from continuator_memory.v2.context_store import contextualize_stored, raw_messages
from continuator_memory.v2.entity_store import EntityStore, resolve_entities_stored, stored_entity_memories
from continuator_memory.v2.expansion import ConversationKnowledge
from continuator_memory.v2.memory_store import consolidate_stored
from continuator_memory.v2.relationship_store import RelationshipStore, build_relationships_stored


def stored_knowledge(connection: sqlite3.Connection, conversation_id: str) -> ConversationKnowledge:
    """Read-only: the stored relationship graph, contextual memories with entity references,
    entities and raw messages."""
    return ConversationKnowledge(
        conversation_id, RelationshipStore(connection).get(conversation_id),
        {m.memory.memory_id: m for m in stored_entity_memories(connection, conversation_id)},
        tuple(raw_messages(connection, conversation_id)),
        dict(EntityStore(connection).get(conversation_id).entities))


def build_knowledge_stored(connection: sqlite3.Connection, conversation_id: str) -> ConversationKnowledge:
    """Consolidate, contextualize, resolve entities and build relationships from the stored
    claims (each step saves and replaces its own rows), then load the result."""
    consolidate_stored(connection, conversation_id)
    contextualize_stored(connection, conversation_id)
    resolve_entities_stored(connection, conversation_id)
    build_relationships_stored(connection, conversation_id)
    return stored_knowledge(connection, conversation_id)
