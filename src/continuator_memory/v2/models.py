"""V2 semantic record model: claims, explicit relations and per-message records.

``Claim.content`` is the truth-bearing field. Entities, qualifiers, temporal information and
polarity are optional annotations over it; there is no predicate/value slot and no ontology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from continuator_memory.models import Message

RELATION_TYPES = ("UPDATES", "CONTRADICTS", "CAUSED_BY", "SUPPORTS", "REFERS_TO")
POLARITIES = ("positive", "negative")

# Statuses the model may return for a target message.
STATUS_CLAIMS = "claims"
STATUS_NO_DURABLE = "no_durable_assertion"
MODEL_STATUSES = (STATUS_CLAIMS, STATUS_NO_DURABLE)
# Statuses set by validation.
STATUS_REJECTED = "rejected"  # the model returned claims but none survived validation
STATUS_MISSING = "missing"  # the model returned no entry for the message
STATUS_FAILED = "failed"  # the whole window failed (request error, invalid JSON or schema)
STATUSES = (*MODEL_STATUSES, STATUS_REJECTED, STATUS_MISSING, STATUS_FAILED)

# Memory types of the V1 memory-extraction protocol (``continuator_memory.v2.memory_prompt``). Stored in
# ``Claim.metadata["memory_type"]``, never in ``claim_type``: consolidation uses ``claim_type`` as a slot
# attribute, and two different preferences of one person must not occupy the same slot.
MEMORY_TYPES = ("preference", "personal_fact", "goal", "project", "event", "decision", "relationship",
                "procedural")

# How a claim is supported. Context may resolve identity/reference, never supply new facts.
SUPPORT_SOURCE = "SOURCE"
SUPPORT_IDENTITY = "IDENTITY_CONTEXT"


def claim_id(message_id: str, ordinal: int) -> str:
    return f"{message_id}:c{ordinal}"


def relation_id(message_id: str, ordinal: int) -> str:
    return f"{message_id}:r{ordinal}"


@dataclass(frozen=True)
class Provenance:
    """Reference to the raw source message; the content itself stays in the raw store."""

    conversation_id: str
    message_id: str
    position: int
    role: str
    timestamp: str | None = None

    @classmethod
    def of(cls, message: Message) -> Provenance:
        return cls(message.conversation_id, message.message_id, message.position, message.role,
                   message.timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {"conversation_id": self.conversation_id, "message_id": self.message_id,
                "position": self.position, "role": self.role, "timestamp": self.timestamp}


@dataclass(frozen=True)
class Temporal:
    """A temporal expression as stated; ``resolved_date`` only when a timestamp allows it."""

    expression: str
    resolved_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"expression": self.expression, "resolved_date": self.resolved_date}


@dataclass(frozen=True)
class Claim:
    claim_id: str
    content: str
    provenance: Provenance
    entities: tuple[str, ...] = ()
    qualifiers: tuple[str, ...] = ()
    temporal: Temporal | None = None
    polarity: str | None = None
    claim_type: str | None = None
    support: str = SUPPORT_SOURCE
    # Context message / prior claim IDs that resolved an identity or reference in the claim.
    resolved_from: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_message_id(self) -> str:
        return self.provenance.message_id

    def to_dict(self) -> dict[str, Any]:
        return {"claim_id": self.claim_id, "content": self.content,
                "source_message_id": self.source_message_id, "entities": list(self.entities),
                "qualifiers": list(self.qualifiers),
                "temporal": self.temporal.to_dict() if self.temporal else None,
                "polarity": self.polarity, "claim_type": self.claim_type, "support": self.support,
                "resolved_from": list(self.resolved_from), "metadata": dict(self.metadata),
                "provenance": self.provenance.to_dict()}


@dataclass(frozen=True)
class Relation:
    relation_id: str
    from_claim_id: str
    to_claim_id: str
    type: str
    source_message_id: str
    evidence: str | None = None
    evidence_span: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"relation_id": self.relation_id, "from_claim_id": self.from_claim_id,
                "to_claim_id": self.to_claim_id, "type": self.type,
                "source_message_id": self.source_message_id, "evidence": self.evidence,
                "evidence_span": list(self.evidence_span) if self.evidence_span else None}


@dataclass(frozen=True)
class SemanticMessage:
    provenance: Provenance
    content: str
    status: str
    claims: tuple[Claim, ...] = ()
    explicit_relations: tuple[Relation, ...] = ()

    @property
    def source_message_id(self) -> str:
        return self.provenance.message_id

    @property
    def role(self) -> str:
        return self.provenance.role

    @property
    def timestamp(self) -> str | None:
        return self.provenance.timestamp

    def to_dict(self) -> dict[str, Any]:
        return {"source_message_id": self.source_message_id, "role": self.role,
                "content": self.content, "timestamp": self.timestamp, "status": self.status,
                "claims": [c.to_dict() for c in self.claims],
                "explicit_relations": [r.to_dict() for r in self.explicit_relations]}
