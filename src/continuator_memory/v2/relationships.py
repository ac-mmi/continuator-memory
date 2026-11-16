"""Memory relationship graph (phase 5C): consolidation links plus EXTENDS / DERIVES edges.

Every edge points from a source memory to a target memory of the same conversation:

    newer memory    --UPDATES / SUPERSEDES / CONTRADICTS / REFERS_TO-->  older memory
    newer memory    --EXTENDS-->  existing memory it adds information to
    derived memory  --DERIVES-->  each memory supporting it (one edge per supporting memory)

The first four are the phase 2A-2C ``MemoryLink``s, taken unchanged from consolidation (their
``newer_memory_id`` is the source). EXTENDS and DERIVES are additive: they never change a
memory's status, ``superseded_by`` or validity, and consolidation never reads them.

  type         meaning                          replaces target?   target remains current?
  UPDATES      newer statement changes target   conceptually yes   per temporal logic
  SUPERSEDES   newer state replaces target      yes                usually no
  CONTRADICTS  conflicting statement            no                 yes, contested
  REFERS_TO    references another memory        no                 yes
  EXTENDS      adds information                 no                 yes
  DERIVES      conclusion supported by target   no                 yes

No LLM proposes edges. ``build_relationships`` creates only:

  DERIVES  from a validated SUPPORTS relation (evidence claim -> supported claim) whose
           evidence is an explicit inference cue ("so", "therefore", "in total", "which means")
           stated in the supported claim's own message: the supported memory DERIVES the
           evidence memory. SUPPORTS without such a cue (a confirmation, an echo) creates nothing.
  EXTENDS  from a later memory B to an earlier memory A of the same conversation when B
           names A's resolved subject and every other entity of A, says something A does
           not, both are positive, non-ambiguous, of the same status (current or historical),
           not the same slot, B states no change and neither states an end of state ("left",
           "no longer"), and no other edge already relates them.

Anything else is left absent: a missing relationship is preferred to a false one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from continuator_memory.memory.models import normalize_value
from continuator_memory.memory.text import STOPWORDS
from continuator_memory.v2.consolidation import _ordinal, claim_order, has_unresolved_reference
from continuator_memory.v2.memory import (
    IDENTITY_AMBIGUOUS,
    LINK_CONTRADICTS,
    LINK_REFERS_TO,
    LINK_SUPERSEDES,
    LINK_TYPES,
    LINK_UPDATES,
    STATUS_CURRENT,
    STATUS_HISTORICAL,
    Decision,
    Memory,
    MemoryLink,
    MemorySource,
)
from continuator_memory.v2.models import Claim, Relation
from continuator_memory.v2.temporal import TIME_AMBIGUOUS_PAST, TIME_MIXED, TIME_TRANSITION

LINK_EXTENDS = "EXTENDS"
LINK_DERIVES = "DERIVES"
ADDITIVE_TYPES = (LINK_EXTENDS, LINK_DERIVES)
RELATIONSHIP_TYPES = (*LINK_TYPES, *ADDITIVE_TYPES)


@dataclass(frozen=True)
class RelationSemantics:
    meaning: str
    replaces_target: str
    target_remains_current: str


SEMANTICS = {
    LINK_UPDATES: RelationSemantics("newer statement changes target", "conceptually yes",
                                    "depends on existing temporal logic"),
    LINK_SUPERSEDES: RelationSemantics("newer state replaces target", "yes", "usually no"),
    LINK_CONTRADICTS: RelationSemantics("conflicting statement", "no", "yes/contested"),
    LINK_REFERS_TO: RelationSemantics("references another memory", "no", "yes"),
    LINK_EXTENDS: RelationSemantics("adds information", "no", "yes"),
    LINK_DERIVES: RelationSemantics("conclusion supported by target", "no", "yes"),
}

ORIGIN_CONSOLIDATION = "consolidation"  # a phase 2A-2C MemoryLink
ORIGIN_SUPPORTS = "supports_relation"  # DERIVES from a validated SUPPORTS relation with an inference cue
ORIGIN_EXTENDS_RULE = "extends_rule"  # EXTENDS from the deterministic rule above
ORIGIN_EXPLICIT = "explicit"  # added through ``RelationshipGraph.add`` by a deterministic caller
ADDITIVE_ORIGINS = (ORIGIN_SUPPORTS, ORIGIN_EXTENDS_RULE, ORIGIN_EXPLICIT)

INVALID_TYPE = "invalid_type"
INVALID_ORIGIN = "invalid_origin"
CROSS_CONVERSATION = "cross_conversation"
MISSING_SOURCE = "missing_source"
MISSING_TARGET = "missing_target"
SELF_LINK = "self_link"
DUPLICATE = "duplicate"
ALREADY_RELATED = "already_related"  # another edge (either direction) already relates the pair
CYCLE = "cycle"
REJECTION_CODES = (INVALID_TYPE, INVALID_ORIGIN, CROSS_CONVERSATION, MISSING_SOURCE, MISSING_TARGET, SELF_LINK,
                   DUPLICATE, ALREADY_RELATED, CYCLE)

# The whole (normalized) evidence of a SUPPORTS relation must be one of these to create DERIVES.
INFERENCE_CUE = re.compile(
    r"(?:and )?(?:so|therefore|thus|hence|so in total|in total|altogether|overall|"
    r"(?:so )?(?:which|that|this) means|(?:which|that) makes|(?:which|that) brings (?:it|the total) to)")
END_OF_STATE = re.compile(
    r"\b(left|leaves|leaving|quit|quits|stopped|stops|ended|ends|resigned|retired|fired|laid off|sold|lost|"
    r"cancel(?:l)?ed|closed|broke up|divorced|gave up|moved out|no longer|anymore|former|ex)\b")
_COMPARABLE_STATUSES = (STATUS_CURRENT, STATUS_HISTORICAL)


def relationship_id(conversation_id: str, source_memory_id: str, type_: str, target_memory_id: str) -> str:
    return (f"rel:{quote(conversation_id, safe='')}/{quote(source_memory_id, safe=':')}/{type_}/"
            f"{quote(target_memory_id, safe=':')}")


def memory_order(memory: Memory) -> tuple[int, int, str]:
    return memory.position, _ordinal(memory.claim_id), memory.memory_id


@dataclass(frozen=True)
class MemoryRelationship:
    conversation_id: str
    source_memory_id: str
    target_memory_id: str
    type: str
    claim_id: str  # the claim whose consolidation or relation created the edge
    relation_id: str | None = None  # the validated V2 relation, if the edge came from one
    origin: str = ORIGIN_EXPLICIT
    reason: str = ""

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.conversation_id, self.source_memory_id, self.target_memory_id, self.type

    @property
    def relationship_id(self) -> str:
        return relationship_id(self.conversation_id, self.source_memory_id, self.type, self.target_memory_id)

    @property
    def additive(self) -> bool:
        return self.type in ADDITIVE_TYPES

    def to_dict(self) -> dict[str, Any]:
        return {"relationship_id": self.relationship_id, "conversation_id": self.conversation_id,
                "source_memory_id": self.source_memory_id, "target_memory_id": self.target_memory_id,
                "type": self.type, "claim_id": self.claim_id, "relation_id": self.relation_id,
                "origin": self.origin, "reason": self.reason}


@dataclass(frozen=True)
class RelationshipRejection:
    code: str
    relationship: MemoryRelationship
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail, "relationship": self.relationship.to_dict()}


class RelationshipGraph:
    """All relationships of one conversation's memories. Consolidation links are taken as they
    are; EXTENDS / DERIVES edges enter only through ``add``, which validates them. The graph
    never modifies a memory."""

    def __init__(self, conversation_id: str, memories: Iterable[Memory], links: Iterable[MemoryLink] = (),
                 decisions: Iterable[Decision] = ()):
        self.conversation_id = conversation_id
        self.memories = {m.memory_id: m for m in memories}
        reasons: dict[str, str] = {}
        for d in decisions:
            reasons.setdefault(d.claim_id, f"{d.action}: {d.reason}")
        self.links = [MemoryRelationship(conversation_id, link.newer_memory_id, link.older_memory_id, link.type,
                                         link.claim_id, link.relation_id, ORIGIN_CONSOLIDATION,
                                         reasons.get(link.claim_id, ""))
                      for link in links]
        self._additive: dict[tuple[str, str, str, str], MemoryRelationship] = {}
        self.rejections: list[RelationshipRejection] = []

    # ---- views --------------------------------------------------------------

    @property
    def additive(self) -> list[MemoryRelationship]:
        """EXTENDS / DERIVES edges: by source memory, EXTENDS before DERIVES, then target memory."""
        return sorted(self._additive.values(), key=lambda r: (memory_order(self.memories[r.source_memory_id]),
                                                              ADDITIVE_TYPES.index(r.type),
                                                              memory_order(self.memories[r.target_memory_id])))

    @property
    def edges(self) -> list[MemoryRelationship]:
        return [*self.links, *self.additive]

    def outgoing(self, memory_id: str, type_: str | None = None) -> list[MemoryRelationship]:
        return [r for r in self.edges if r.source_memory_id == memory_id and type_ in (None, r.type)]

    def incoming(self, memory_id: str, type_: str | None = None) -> list[MemoryRelationship]:
        return [r for r in self.edges if r.target_memory_id == memory_id and type_ in (None, r.type)]

    def related(self, a: str, b: str) -> list[MemoryRelationship]:
        """Edges between ``a`` and ``b`` in either direction."""
        return [r for r in self.edges if {r.source_memory_id, r.target_memory_id} == {a, b}]

    def to_dict(self) -> dict[str, Any]:
        return {"conversation_id": self.conversation_id, "edges": [r.to_dict() for r in self.edges],
                "rejections": [r.to_dict() for r in self.rejections]}

    # ---- add ----------------------------------------------------------------

    def add(self, rel: MemoryRelationship) -> RelationshipRejection | None:
        """Add an EXTENDS / DERIVES edge, or record and return why it was refused."""
        problem = self._problem(rel)
        if problem is not None:
            rejection = RelationshipRejection(*problem, rel)
            self.rejections.append(rejection)
            return rejection
        self._additive[rel.key] = rel
        return None

    def derive(self, derived_memory_id: str, supporting_memory_ids: Sequence[str], claim_id: str,
               reason: str, origin: str = ORIGIN_EXPLICIT,
               relation_id: str | None = None) -> list[RelationshipRejection]:
        """One DERIVES edge from the derived memory to each supporting memory."""
        out = [self.add(MemoryRelationship(self.conversation_id, derived_memory_id, target, LINK_DERIVES,
                                           claim_id, relation_id, origin, reason))
               for target in supporting_memory_ids]
        return [r for r in out if r is not None]

    def _problem(self, rel: MemoryRelationship) -> tuple[str, str] | None:
        if rel.type not in ADDITIVE_TYPES:
            detail = ("comes only from consolidation" if rel.type in LINK_TYPES
                      else f"not one of {', '.join(RELATIONSHIP_TYPES)}")
            return INVALID_TYPE, f"{rel.type}: {detail}"
        if rel.origin not in ADDITIVE_ORIGINS:
            return INVALID_ORIGIN, f"{rel.origin!r} is not an additive origin"
        if rel.conversation_id != self.conversation_id:
            return CROSS_CONVERSATION, f"edge of {rel.conversation_id!r} offered to {self.conversation_id!r}"
        for role, mid, code in (("source", rel.source_memory_id, MISSING_SOURCE),
                                ("target", rel.target_memory_id, MISSING_TARGET)):
            mem = self.memories.get(mid)
            if mem is None:
                return code, f"{role} memory {mid!r} not in {self.conversation_id!r}"
            if mem.conversation_id != self.conversation_id:
                return CROSS_CONVERSATION, f"{role} memory {mid!r} belongs to {mem.conversation_id!r}"
        if rel.source_memory_id == rel.target_memory_id:
            return SELF_LINK, f"{rel.source_memory_id} {rel.type} itself"
        if rel.key in self._additive:
            return DUPLICATE, f"{rel.relationship_id} already exists"
        existing = self.related(rel.source_memory_id, rel.target_memory_id)
        if existing:
            return ALREADY_RELATED, ", ".join(f"{r.source_memory_id} {r.type} {r.target_memory_id}" for r in existing)
        if self._reaches(rel.target_memory_id, rel.source_memory_id, rel.type):
            return CYCLE, f"{rel.target_memory_id} already reaches {rel.source_memory_id} through {rel.type}"
        return None

    def _reaches(self, start: str, goal: str, type_: str) -> bool:
        seen, frontier = {start}, [start]
        while frontier:
            node = frontier.pop()
            for r in self._additive.values():
                if r.type == type_ and r.source_memory_id == node and r.target_memory_id not in seen:
                    if r.target_memory_id == goal:
                        return True
                    seen.add(r.target_memory_id)
                    frontier.append(r.target_memory_id)
        return False


def versioning_cycles(links: Iterable[MemoryLink]) -> list[tuple[str, str]]:
    """Pairs related by UPDATES / SUPERSEDES in both directions (should never exist)."""
    pairs = {(link.newer_memory_id, link.older_memory_id) for link in links
             if link.type in (LINK_UPDATES, LINK_SUPERSEDES)}
    return sorted((a, b) for a, b in pairs if (b, a) in pairs and a < b)


# ---- deterministic builder --------------------------------------------------


def derives_cue(relation: Relation, supported: Claim | None) -> bool:
    """A SUPPORTS relation stated as an inference in the supported claim's own message."""
    return (relation.type == "SUPPORTS" and supported is not None
            and relation.source_message_id == supported.source_message_id
            and INFERENCE_CUE.fullmatch(normalize_value(relation.evidence or "")) is not None)


def _words(text: str) -> set[str]:
    return {w for w in normalize_value(text).split() if w not in STOPWORDS}


def extends_reason(older: Memory, newer: Memory, claims: dict[str, Claim], blocked: set[str]) -> str | None:
    """Why ``newer`` EXTENDS ``older`` under the deterministic rule, or None."""
    subject = older.subject
    others = [e for e in older.entities if e != subject]
    a, b = claims.get(older.claim_id), claims.get(newer.claim_id)
    if (a is None or b is None or not subject or newer.subject != subject or not others
            or not set(others) <= set(newer.entities)):
        return None
    if older.status != newer.status or older.status not in _COMPARABLE_STATUSES:
        return None
    if "negative" in (older.polarity, newer.polarity) or (older.attribute and older.attribute == newer.attribute):
        return None
    if newer.time_class in (TIME_TRANSITION, TIME_MIXED, TIME_AMBIGUOUS_PAST) or older.time_class in (
            TIME_MIXED, TIME_AMBIGUOUS_PAST):
        return None
    if {older.claim_id, newer.claim_id} & blocked or any(has_unresolved_reference(c) for c in (a, b)):
        return None
    if any(END_OF_STATE.search(m.content.lower()) for m in (older, newer)):
        return None
    if not _words(newer.content) - _words(older.content):
        return None
    return f"same subject {subject!r}; names {', '.join(repr(e) for e in others)} and adds information"


def build_relationships(conversation_id: str, memories: Iterable[Memory], links: Iterable[MemoryLink],
                        sources: Iterable[MemorySource], decisions: Iterable[Decision], claims: Iterable[Claim],
                        relations: Iterable[Relation]) -> RelationshipGraph:
    """The conversation's relationship graph from its consolidation state, claims and relations."""
    decisions = list(decisions)
    graph = RelationshipGraph(conversation_id, memories, links, decisions)
    memory_of = {s.claim_id: s.memory_id for s in sources}
    by_id = {c.claim_id: c for c in claims if c.provenance.conversation_id == conversation_id}
    blocked = {d.claim_id for d in decisions if IDENTITY_AMBIGUOUS in d.unresolved_codes}

    supports = [r for r in relations if r.type == "SUPPORTS" and r.to_claim_id in by_id and r.from_claim_id in by_id]
    for r in sorted(supports, key=lambda r: (claim_order(by_id[r.to_claim_id]), claim_order(by_id[r.from_claim_id]),
                                             r.relation_id)):
        if derives_cue(r, by_id[r.to_claim_id]):
            graph.add(MemoryRelationship(
                conversation_id, memory_of.get(r.to_claim_id, ""), memory_of.get(r.from_claim_id, ""), LINK_DERIVES,
                r.to_claim_id, r.relation_id, ORIGIN_SUPPORTS,
                f"{r.from_claim_id} SUPPORTS {r.to_claim_id}, stated as an inference ({r.evidence!r})"))

    ordered = sorted(graph.memories.values(), key=memory_order)
    for i, newer in enumerate(ordered):
        for older in ordered[:i]:
            if graph.related(older.memory_id, newer.memory_id):
                continue
            reason = extends_reason(older, newer, by_id, blocked)
            if reason:
                graph.add(MemoryRelationship(conversation_id, newer.memory_id, older.memory_id, LINK_EXTENDS,
                                             newer.claim_id, None, ORIGIN_EXTENDS_RULE, reason))
    return graph
