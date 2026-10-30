"""V1 memory contract: a read-only record view over consolidated memories.

    conversation messages (source evidence, unchanged)
      -> extraction (``extract="memories"``): validated claims, one or more per message, or none
      -> consolidation (unchanged): versioned memories + sources + links
      -> ``memory_records``: one ``MemoryRecord`` per memory

The record keeps the normalized memory separate from its evidence: ``content`` is the memory, while
``sources`` lists every message that stated it (message ID, speaker, role, timestamp, claim ID).
Nothing here calls a model, writes storage or changes retrieval; memory IDs are the consolidation's
(``mem:<message_id>:c<n>``), so they are stable for the same extraction output.

Fields:
  memory_type   from the extracting claim's metadata (memory protocol only; None otherwise)
  assertion     "explicit" when the validator found the claim's content stems in the source message;
                "identity_resolved" when a pronoun or reference was resolved from context. This is the
                lexical check of ``validation.classify_support``, not proof the statement is faithful.
  status        current / superseded / historical / contested / future / conditional (consolidation).
                A memory is superseded only by an explicit UPDATES relation or an identity-safe slot
                match; ambiguous changes leave both versions in place, with their sources.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from continuator_memory.models import Conversation
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.graph_index import split_speaker
from continuator_memory.v2.models import SUPPORT_SOURCE, Claim, Relation

ASSERTION_EXPLICIT = "explicit"
ASSERTION_IDENTITY_RESOLVED = "identity_resolved"


@dataclass(frozen=True)
class SourceRef:
    message_id: str
    claim_id: str
    speaker: str
    role: str
    timestamp: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"message_id": self.message_id, "claim_id": self.claim_id, "speaker": self.speaker,
                "role": self.role, "timestamp": self.timestamp}


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    conversation_id: str
    content: str
    memory_type: str | None
    subject: str | None
    entities: tuple[str, ...]
    assertion: str
    status: str
    superseded_by: str | None
    temporal: str | None
    event_time: str | None
    valid_from: str | None
    valid_until: str | None
    polarity: str | None
    sources: tuple[SourceRef, ...]
    links: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "conversation_id": self.conversation_id, "content": self.content,
                "memory_type": self.memory_type, "subject": self.subject, "entities": list(self.entities),
                "assertion": self.assertion, "status": self.status, "superseded_by": self.superseded_by,
                "temporal": self.temporal, "event_time": self.event_time, "valid_from": self.valid_from,
                "valid_until": self.valid_until, "polarity": self.polarity,
                "sources": [s.to_dict() for s in self.sources], "links": list(self.links)}


def memory_records(conversation: Conversation, claims: Iterable[Claim],
                   relations: Iterable[Relation] = ()) -> list[MemoryRecord]:
    claims = list(claims)
    consolidator = consolidate(conversation.conversation_id, claims, relations)
    by_claim = {c.claim_id: c for c in claims}
    messages = {m.message_id: m for m in conversation.messages}
    sources: dict[str, list] = {}
    for s in consolidator.sources:
        sources.setdefault(s.memory_id, []).append(s)
    links: dict[str, list] = {}
    for link in consolidator.links:
        for mid in (link.newer_memory_id, link.older_memory_id):
            links.setdefault(mid, []).append(link.to_dict())
    records = []
    for mem in sorted(consolidator.memories.values(), key=lambda m: (m.position, m.memory_id)):
        refs = []
        for s in sorted(sources.get(mem.memory_id, []), key=lambda s: (s.position, s.claim_id)):
            msg = messages.get(s.message_id)
            speaker, role = (split_speaker(msg.content, msg.role)[0], msg.role) if msg else ("unknown", "unknown")
            refs.append(SourceRef(s.message_id, s.claim_id, speaker, role, s.observed_at))
        creating = by_claim.get(mem.claim_id)
        memory_type = (creating.metadata.get("memory_type") if creating else None) or next(
            (by_claim[r.claim_id].metadata.get("memory_type") for r in refs
             if r.claim_id in by_claim and by_claim[r.claim_id].metadata.get("memory_type")), None)
        assertion = ASSERTION_EXPLICIT if creating is None or creating.support == SUPPORT_SOURCE else ASSERTION_IDENTITY_RESOLVED
        records.append(MemoryRecord(
            mem.memory_id, mem.conversation_id, mem.content, memory_type, mem.subject, tuple(mem.entities),
            assertion, mem.status, mem.superseded_by, mem.temporal, mem.event_time, mem.valid_from,
            mem.valid_until, mem.polarity, tuple(refs), tuple(links.get(mem.memory_id, ()))))
    return records
