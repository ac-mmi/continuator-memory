"""V2 consolidated memory model (phase 2A): memories, their sources and version links.

A memory is a validated claim promoted to persistent state; its content is the claim content,
never rewritten. Memories are never deleted: a change of state creates a new memory, links it
to the older one and marks the older one superseded, which keeps its sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STATUS_CURRENT = "current"
STATUS_SUPERSEDED = "superseded"
# Stated as a past state ("used to ...") when first observed; never treated as current.
STATUS_HISTORICAL = "historical"
# Explicitly contradicted; both sides stay, no winner is chosen.
STATUS_CONTESTED = "contested"
# A stated future state or plan; not current, and it never supersedes a current memory.
STATUS_FUTURE = "future"
# A hypothetical or conditional state; not current, and it never supersedes.
STATUS_CONDITIONAL = "conditional"
MEMORY_STATUSES = (STATUS_CURRENT, STATUS_SUPERSEDED, STATUS_HISTORICAL, STATUS_CONTESTED, STATUS_FUTURE,
                   STATUS_CONDITIONAL)
CURRENT_STATUSES = (STATUS_CURRENT, STATUS_CONTESTED)

LINK_UPDATES = "UPDATES"  # from a validated UPDATES relation
LINK_SUPERSEDES = "SUPERSEDES"  # same subject and claim_type, newer claim states a change
LINK_CONTRADICTS = "CONTRADICTS"  # from a validated CONTRADICTS relation
LINK_REFERS_TO = "REFERS_TO"  # from a validated REFERS_TO relation; no state change
LINK_TYPES = (LINK_UPDATES, LINK_SUPERSEDES, LINK_CONTRADICTS, LINK_REFERS_TO)

CREATE, KEEP, UPDATE, SUPERSEDE = "CREATE", "KEEP", "UPDATE", "SUPERSEDE"
CONTRADICTION, HISTORICAL, IGNORE = "CONTRADICTION", "HISTORICAL", "IGNORE"
ACTIONS = (CREATE, KEEP, UPDATE, SUPERSEDE, CONTRADICTION, HISTORICAL, IGNORE)
# Actions that create a new memory for the claim.
CREATING_ACTIONS = (CREATE, UPDATE, SUPERSEDE, CONTRADICTION, HISTORICAL)

# Machine-readable prefixes of ``Decision.unresolved`` entries ("<code>: <detail>").
IDENTITY_AMBIGUOUS = "identity_ambiguous"  # unresolved pronoun, or a name bound to several entities
SUBJECT_MISSING = "subject_missing"  # change stated for an attribute, but no safe subject identity
SLOT_AMBIGUOUS = "slot_ambiguous"  # several current memories occupy the slot
RELATION_TARGET_MISSING = "relation_target_missing"
RELATION_TARGET_AMBIGUOUS = "relation_target_ambiguous"
RELATION_TARGET_SUPERSEDED = "relation_target_superseded"
RELATION_IDENTITY_CONFLICT = "relation_identity_conflict"
TEMPORAL_AMBIGUOUS = "temporal_ambiguous"  # temporal meaning or event order insufficient (phase 2B)
RESTATEMENT_AMBIGUOUS = "restatement_ambiguous"  # restates a non-current memory, or several
REASON_CODES = (IDENTITY_AMBIGUOUS, SUBJECT_MISSING, SLOT_AMBIGUOUS, RELATION_TARGET_MISSING,
                RELATION_TARGET_AMBIGUOUS, RELATION_TARGET_SUPERSEDED, RELATION_IDENTITY_CONFLICT,
                TEMPORAL_AMBIGUOUS, RESTATEMENT_AMBIGUOUS)


def memory_id(claim_id: str) -> str:
    return f"mem:{claim_id}"


@dataclass
class Memory:
    memory_id: str
    conversation_id: str
    content: str
    status: str
    claim_id: str  # the claim that created the memory
    message_id: str
    position: int
    # Observation time (timestamp of the creating message), not the real-world event date.
    created_at: str | None = None
    # Validity bounds, only as stated by a claim (year, month or day precision); never from
    # message or ingestion time. valid_until may also come from the start date of the version
    # that explicitly replaced this one.
    valid_from: str | None = None
    valid_until: str | None = None
    subject: str | None = None
    entities: tuple[str, ...] = ()
    claim_type: str | None = None
    polarity: str | None = None
    temporal: str | None = None
    superseded_by: str | None = None
    # A stated point in time that is not a validity boundary ("visited Rome in 2019").
    event_time: str | None = None
    time_class: str = "unspecified"  # see continuator_memory.v2.temporal
    # Structurally normalized slot attribute (metadata "attribute", else claim_type).
    attribute: str | None = None

    @property
    def is_current(self) -> bool:
        return self.status in CURRENT_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "conversation_id": self.conversation_id, "content": self.content,
                "status": self.status, "claim_id": self.claim_id, "message_id": self.message_id,
                "position": self.position, "created_at": self.created_at, "valid_from": self.valid_from,
                "valid_until": self.valid_until, "subject": self.subject, "entities": list(self.entities),
                "claim_type": self.claim_type, "polarity": self.polarity, "temporal": self.temporal,
                "superseded_by": self.superseded_by, "event_time": self.event_time, "time_class": self.time_class,
                "attribute": self.attribute}


@dataclass(frozen=True)
class MemorySource:
    memory_id: str
    claim_id: str
    message_id: str
    position: int
    observed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "claim_id": self.claim_id, "message_id": self.message_id,
                "position": self.position, "observed_at": self.observed_at}


@dataclass(frozen=True)
class MemoryLink:
    newer_memory_id: str
    older_memory_id: str
    type: str
    claim_id: str  # the claim whose consolidation created the link
    relation_id: str | None = None  # the validated V2 relation, if the link came from one

    def to_dict(self) -> dict[str, Any]:
        return {"newer_memory_id": self.newer_memory_id, "older_memory_id": self.older_memory_id,
                "type": self.type, "claim_id": self.claim_id, "relation_id": self.relation_id}


@dataclass(frozen=True)
class Decision:
    """What consolidation did with one claim, and why."""

    claim_id: str
    action: str
    memory_id: str | None
    superseded: tuple[str, ...] = ()
    contradicted: tuple[str, ...] = ()
    reason: str = ""
    # Set when an identity or relation could not be resolved safely, so memories were kept apart.
    unresolved: str | None = None

    @property
    def unresolved_codes(self) -> tuple[str, ...]:
        codes = (part.split(":", 1)[0] for part in (self.unresolved or "").split("; "))
        return tuple(dict.fromkeys(c for c in codes if c in REASON_CODES))

    def to_dict(self) -> dict[str, Any]:
        return {"claim_id": self.claim_id, "action": self.action, "memory_id": self.memory_id,
                "superseded": list(self.superseded), "contradicted": list(self.contradicted),
                "reason": self.reason, "unresolved": self.unresolved}
