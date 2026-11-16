"""SQLite persistence for phase 5A claim contexts.

One row per validated claim, cascading from ``v2_claims``. A row stores the source window as a
reference (positions and ordered message IDs) and the derived contextual content; message text,
roles and timestamps stay in the raw ``messages`` table and are recovered with ``resolve_window``.
Source windows are rebuilt from the stored atomization log, so they are exactly the windows the
atomizer was given.
"""

from __future__ import annotations

import json
import sqlite3

from continuator_memory.models import Message
from continuator_memory.v2.context import (
    ClaimContext,
    ContextualMemory,
    SourceContextError,
    SourceWindow,
    Speaker,
    contextual_memories,
    contextualize_claims,
)
from continuator_memory.v2.memory_store import MemoryStore
from continuator_memory.v2.store import SemanticStore

CONTEXT_SCHEMA = """
CREATE TABLE IF NOT EXISTS v2_claim_contexts (
    conversation_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    speaker_json TEXT NOT NULL,
    window_start INTEGER NOT NULL,
    window_end INTEGER NOT NULL,
    window_message_ids_json TEXT NOT NULL,
    contextual_content TEXT NOT NULL CHECK (length(trim(contextual_content)) > 0),
    resolved_speakers_json TEXT NOT NULL,
    unresolved_reference INTEGER NOT NULL,
    resolved_from_json TEXT NOT NULL,
    PRIMARY KEY (conversation_id, claim_id),
    FOREIGN KEY (conversation_id, claim_id)
        REFERENCES v2_claims(conversation_id, claim_id) ON DELETE CASCADE
);
"""


def raw_messages(connection: sqlite3.Connection, conversation_id: str) -> list[Message]:
    rows = connection.execute(
        "SELECT message_id, conversation_id, position, role, content, timestamp FROM messages "
        "WHERE conversation_id = ? ORDER BY position", (conversation_id,))
    return [Message(*row) for row in rows]


def stored_windows(connection: sqlite3.Connection, conversation_id: str) -> dict[str, SourceWindow]:
    """Source window of every target message, from the stored atomization log."""
    position = {m.message_id: m.position for m in raw_messages(connection, conversation_id)}
    out: dict[str, SourceWindow] = {}
    for entry in SemanticStore(connection).get_log(conversation_id):
        ids = sorted(entry["context_ids"] + entry["target_ids"], key=lambda mid: position[mid])
        for target in entry["target_ids"]:
            own = [mid for mid in ids if position[mid] < position[target]] + [target]
            out[target] = SourceWindow(conversation_id, position[own[0]], position[target], tuple(own))
    return out


class ContextStore:
    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        SemanticStore(connection)  # v2_claims must exist for the foreign key
        self._conn.executescript(CONTEXT_SCHEMA)

    def clear_conversation(self, conversation_id: str) -> None:
        """Delete claim-context rows inside the caller's open transaction; does not commit."""
        self._conn.execute(
            "DELETE FROM v2_claim_contexts WHERE conversation_id = ?",
            (conversation_id,),
        )

    def delete_conversation(self, conversation_id: str) -> None:
        with self._conn:
            self.clear_conversation(conversation_id)

    def save(self, conversation_id: str, contexts: dict[str, ClaimContext]) -> None:
        self.delete_conversation(conversation_id)
        with self._conn:
            for c in contexts.values():
                if c.conversation_id != conversation_id:
                    raise SourceContextError(f"{c.claim_id} belongs to {c.conversation_id}, not {conversation_id}")
                self._conn.execute(
                    "INSERT INTO v2_claim_contexts (conversation_id, claim_id, message_id, position, speaker_json, "
                    "window_start, window_end, window_message_ids_json, contextual_content, resolved_speakers_json, "
                    "unresolved_reference, resolved_from_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (conversation_id, c.claim_id, c.message_id, c.position, json.dumps(c.speaker.to_dict()),
                     c.window.start, c.window.end, json.dumps(list(c.window.message_ids)), c.contextual_content,
                     json.dumps([s.to_dict() for s in c.resolved_speakers]), int(c.unresolved_reference),
                     json.dumps(list(c.resolved_from))))

    def get_contexts(self, conversation_id: str) -> dict[str, ClaimContext]:
        rows = self._conn.execute(
            "SELECT claim_id, message_id, position, speaker_json, window_start, window_end, window_message_ids_json, "
            "contextual_content, resolved_speakers_json, unresolved_reference, resolved_from_json "
            "FROM v2_claim_contexts WHERE conversation_id = ? ORDER BY position, claim_id", (conversation_id,))
        out = {}
        for cid, mid, pos, speaker, start, end, ids, content, resolved, unresolved, resolved_from in rows:
            out[cid] = ClaimContext(
                cid, conversation_id, mid, pos, Speaker.from_dict(json.loads(speaker)),
                SourceWindow(conversation_id, start, end, tuple(json.loads(ids))), content,
                tuple(Speaker.from_dict(s) for s in json.loads(resolved)), bool(unresolved),
                tuple(json.loads(resolved_from)))
        return out


def contextualize_stored(connection: sqlite3.Connection, conversation_id: str) -> dict[str, ClaimContext]:
    """Contextualize the conversation's stored V2 claims against their stored windows and save them."""
    contexts = contextualize_claims(raw_messages(connection, conversation_id),
                                    SemanticStore(connection).get_claims(conversation_id),
                                    stored_windows(connection, conversation_id))
    ContextStore(connection).save(conversation_id, contexts)
    return contexts


def stored_contextual_memories(connection: sqlite3.Connection, conversation_id: str) -> list[ContextualMemory]:
    """Stored memories (``MemoryStore``) joined with stored claim contexts (``ContextStore``)."""
    memories = MemoryStore(connection)
    return contextual_memories(memories.get_memories(conversation_id), memories.get_sources(conversation_id),
                               ContextStore(connection).get_contexts(conversation_id))
