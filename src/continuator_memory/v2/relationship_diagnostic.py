"""Relationship-graph coverage diagnostic (phase 5E, analysis only).

Generates conservative relationship *candidates* from one conversation's existing phase 2-5C
representation and explains, for each, what phase 5C (and the consolidation rules it inherits)
currently does with it. Nothing here persists, changes or adds a production relationship; the
oracle graph (``oracle_knowledge``) is an in-memory copy for counterfactual diagnostics only.

Candidate generation reads only the conversation's memories, claims, relations, decisions,
entity references, contextual content and messages: never a question, an answer or a gold
label. Candidate sources (``candidate_reason``):

  EXISTING_VERSION_CHAIN / EXISTING_LINK   every edge already in the phase 5C graph
  EXPLICIT_RELATION                        a validated UPDATES / CONTRADICTS / REFERS_TO relation
  SUPPORTING_MEMORY_PLUS_EXPLICIT_INFERENCE  a validated SUPPORTS relation (-> DERIVES)
  INFERENCE_CUE_WITHOUT_RELATION           an inference cue ("so", "which means", ...) in a message,
                                           between claims of that message (or opening a message,
                                           towards anchored memories of its source window), with
                                           no SUPPORTS relation (-> DERIVES)
  SAME_ENTITY_SHARED_TERM_CHANGE_STATED    two memories anchored to the same entity, sharing a
                                           non-entity content word, the newer stating a change
                                           (-> SUPERSEDES, the consolidation slot-rule type)
  SAME_SUBJECT_ADDITIONAL_INFORMATION      two anchored memories otherwise (-> EXTENDS)

A pair is anchored by the memories' consolidation subject, a shared phase 5B entity (any role
but "speaker") or a shared memory entity (structured anchors), or else by an entity name known
to the conversation occurring in both contextual contents (a text-only anchor). Unanchored
pairs never become candidates (no all-pairs relationships).

Each candidate is checked in the order the deciding rule checks it (``extends_reason``,
``derives_cue``, the consolidation slot and explicit-relation rules). Every failing check is a
reason code of one class:

  NOT_APPLICABLE       the representation contradicts the relationship (different resolved
                       subjects, different slots, a restatement, the same memory)
  INSUFFICIENT_EVIDENCE  information the rule needs is absent from every structured layer
                       (no subject, no shared entity, no slot, no SUPPORTS relation, an
                       unresolved or ambiguous identity)
  REJECTED_RULE        the structured layers hold the needed information but a phase 5C /
                       consolidation rule refuses the relationship

The decision is NOT_APPLICABLE if any not-applicable check fails, else INSUFFICIENT_EVIDENCE if
any representation check fails, else REJECTED_RULE if any rule check fails (else the edge
exists: ACCEPTED_EXISTING). ``rejection_reason`` is the first failing check in rule order.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from continuator_memory.memory.models import normalize_value
from continuator_memory.memory.text import STOPWORDS
from continuator_memory.v2.consolidation import claim_order, has_unresolved_reference
from continuator_memory.v2.context import ContextualMemory
from continuator_memory.v2.entities import REF_RESOLVED, ROLE_POSSESSOR, ROLE_SPEAKER, ROLE_SUBJECT
from continuator_memory.v2.expansion import ALWAYS, HISTORY, INCOMING, OUTGOING, POLICY, ConversationKnowledge
from continuator_memory.v2.memory import (
    IDENTITY_AMBIGUOUS,
    LINK_CONTRADICTS,
    LINK_REFERS_TO,
    LINK_SUPERSEDES,
    LINK_UPDATES,
    RELATION_IDENTITY_CONFLICT,
    RELATION_TARGET_AMBIGUOUS,
    RELATION_TARGET_MISSING,
    RELATION_TARGET_SUPERSEDED,
    STATUS_CONDITIONAL,
    STATUS_CURRENT,
    STATUS_FUTURE,
    STATUS_HISTORICAL,
    STATUS_SUPERSEDED,
    Decision,
    Memory,
    MemoryLink,
)
from continuator_memory.v2.models import Claim, Relation
from continuator_memory.v2.relationships import (
    END_OF_STATE,
    INFERENCE_CUE,
    LINK_DERIVES,
    LINK_EXTENDS,
    ORIGIN_EXPLICIT,
    RELATIONSHIP_TYPES,
    MemoryRelationship,
    RelationshipGraph,
    derives_cue,
    extends_reason,
    memory_order,
    relationship_id,
)
from continuator_memory.v2.temporal import (
    CHANGING_CLASSES,
    REPLACEABLE_CLASSES,
    TIME_AMBIGUOUS_PAST,
    TIME_MIXED,
    TIME_TRANSITION,
    TRANSITION,
)

# ---- vocabulary -----------------------------------------------------------------------------

ACCEPTED_EXISTING = "ACCEPTED_EXISTING"
REJECTED_RULE = "REJECTED_RULE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
NOT_APPLICABLE = "NOT_APPLICABLE"
DUPLICATE = "DUPLICATE"
DECISIONS = (ACCEPTED_EXISTING, REJECTED_RULE, INSUFFICIENT_EVIDENCE, NOT_APPLICABLE, DUPLICATE)

EXISTING_VERSION_CHAIN = "EXISTING_VERSION_CHAIN"
EXISTING_LINK = "EXISTING_LINK"
EXPLICIT_RELATION = "EXPLICIT_RELATION"
SUPPORTING_MEMORY_PLUS_EXPLICIT_INFERENCE = "SUPPORTING_MEMORY_PLUS_EXPLICIT_INFERENCE"
INFERENCE_CUE_WITHOUT_RELATION = "INFERENCE_CUE_WITHOUT_RELATION"
SAME_ENTITY_SHARED_TERM_CHANGE_STATED = "SAME_ENTITY_SHARED_TERM_CHANGE_STATED"
SAME_SUBJECT_ADDITIONAL_INFORMATION = "SAME_SUBJECT_ADDITIONAL_INFORMATION"
CANDIDATE_REASONS = (EXISTING_VERSION_CHAIN, EXISTING_LINK, EXPLICIT_RELATION, SUPPORTING_MEMORY_PLUS_EXPLICIT_INFERENCE,
                     INFERENCE_CUE_WITHOUT_RELATION, SAME_ENTITY_SHARED_TERM_CHANGE_STATED,
                     SAME_SUBJECT_ADDITIONAL_INFORMATION)

ANCHOR_SUBJECT, ANCHOR_ENTITY, ANCHOR_TEXT = "subject_field", "entity_layer", "text_only"
ANCHOR_RELATION, ANCHOR_MESSAGE, ANCHOR_EDGE = "explicit_relation", "same_message", "existing_edge"

NA, GAP, RULE = "not_applicable", "representation", "rule"
# reason -> (class, the existing rule or representation fact it reflects)
REASONS: dict[str, tuple[str, str]] = {
    "SAME_MEMORY": (NA, "both claims consolidated into one memory"),
    "DIFFERENT_SUBJECT": (NA, "resolved subjects differ and no phase 5B entity joins them"),
    "DIFFERENT_SLOT": (NA, "slot attributes differ (consolidation slot rule)"),
    "RESTATEMENT": (NA, "same stated value: a restatement, not a change"),
    "MISSING_CLAIM": (GAP, "a source claim is not stored"),
    "MISSING_SUBJECT": (GAP, "no consolidation subject and no shared phase 5B subject/possessor entity"),
    "MISSING_ENTITY": (GAP, "a shared name occurs in the text but no structured entity joins the memories"),
    "MISSING_SLOT": (GAP, "no slot attribute (consolidation slot rule needs subject and attribute)"),
    "AMBIGUOUS_ENTITY": (GAP, "identity ambiguous in consolidation (extends_reason blocked claims)"),
    "UNRESOLVED_PRONOUN": (GAP, "unresolved third-person reference (has_unresolved_reference)"),
    "NO_SUPPORTING_RELATION": (GAP, "inference cue in the text but no validated SUPPORTS relation"),
    "SUBJECT_ONLY_IN_ENTITY_LAYER": (RULE, "phase 5B joins the subjects but extends_reason compares memory.subject"),
    "NO_OTHER_ENTITY": (RULE, "extends_reason needs the older memory to have an entity besides its subject"),
    "TARGET_ENTITIES_NOT_NAMED": (RULE, "extends_reason needs the newer memory to name every other entity"),
    "STATUS_MISMATCH": (RULE, "extends_reason needs equal status, current or historical"),
    "NEGATIVE_POLARITY": (RULE, "extends_reason refuses negative claims"),
    "SAME_SLOT": (RULE, "extends_reason refuses the same slot attribute"),
    "TEMPORAL_CLASS": (RULE, "transition/mixed/ambiguous-past time class (extends_reason, slot rule)"),
    "END_OF_STATE": (RULE, "end-of-state wording (extends_reason END_OF_STATE)"),
    "NO_NEW_INFORMATION": (RULE, "newer memory adds no content word (extends_reason)"),
    "NO_EXPLICIT_INFERENCE_CUE": (RULE, "SUPPORTS evidence is not an inference cue (derives_cue)"),
    "CUE_NOT_IN_SUPPORTED_MESSAGE": (RULE, "SUPPORTS stated outside the supported claim's message (derives_cue)"),
    "ALREADY_RELATED": (RULE, "another edge already relates the pair (RelationshipGraph.add)"),
    "TARGET_ALREADY_SUPERSEDED": (RULE, "target already superseded (consolidation explicit/slot rule)"),
    "STATUS_NOT_REPLACEABLE": (RULE, "older memory not current/replaceable (consolidation slot rule)"),
    "NOT_A_CHANGE": (RULE, "newer claim states no present state or change (consolidation slot rule)"),
    "DIFFERENT_EXPLICIT_SUBJECT": (RULE, "explicit subjects differ (consolidation explicit-relation rule)"),
    "OTHER_EXISTING_RULE": (RULE, "an existing consolidation rule outcome (see detail)"),
}
_CONSOLIDATION_CODES = {IDENTITY_AMBIGUOUS: "UNRESOLVED_PRONOUN", RELATION_IDENTITY_CONFLICT: "DIFFERENT_EXPLICIT_SUBJECT",
                        RELATION_TARGET_AMBIGUOUS: "AMBIGUOUS_ENTITY", RELATION_TARGET_MISSING: "MISSING_CLAIM",
                        RELATION_TARGET_SUPERSEDED: "TARGET_ALREADY_SUPERSEDED"}
_EXPLICIT_TYPES = {LINK_UPDATES: LINK_UPDATES, LINK_CONTRADICTS: LINK_CONTRADICTS, LINK_REFERS_TO: LINK_REFERS_TO,
                   "SUPPORTS": LINK_DERIVES}
DIRECTION_OF = {LINK_UPDATES: "newer_to_older", LINK_SUPERSEDES: "newer_to_older", LINK_CONTRADICTS: "newer_to_older",
                LINK_REFERS_TO: "referrer_to_referenced", LINK_EXTENDS: "newer_to_extended",
                LINK_DERIVES: "derived_to_support"}

# "so" as an inference, not an intensifier ("so excited") or a quantity ("so far", "or so").
TEXT_CUE = re.compile(r"\b(therefore|thus|hence|which means|that means|in total|altogether|"
                      r"so(?! (?:much|many|far|good|excited|happy|sorry|glad|nice|cool|great|tired|busy|long|that)\b))\b")
VALUE_TOKEN = re.compile(r"\d[\d,.:]*|\b(?:january|february|march|april|may|june|july|august|september|october|"
                         r"november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b")
CHANGE_WORDS = {"now", "moves", "moved", "move", "changed", "changes", "switched", "takes", "over", "new", "longer",
                "starting", "anymore", "actually", "instead"}


def reason_class(reason: str) -> str:
    return REASONS[reason][0]


def decide(failing: Sequence[str]) -> str:
    classes = {reason_class(r) for r in failing}
    if NA in classes:
        return NOT_APPLICABLE
    if GAP in classes:
        return INSUFFICIENT_EVIDENCE
    return REJECTED_RULE


# ---- evidence -------------------------------------------------------------------------------


@dataclass
class ConversationEvidence:
    """One conversation's stored representation, as the diagnostic reads it."""

    knowledge: ConversationKnowledge
    claims: dict[str, Claim]
    relations: list[Relation]
    decisions: list[Decision]
    memory_of: dict[str, str]  # claim_id -> memory_id

    @property
    def conversation_id(self) -> str:
        return self.knowledge.conversation_id

    @property
    def graph(self) -> RelationshipGraph:
        return self.knowledge.graph

    def memory(self, memory_id: str) -> Memory:
        return self.graph.memories[memory_id]

    def contextual(self, memory_id: str) -> ContextualMemory:
        return self.knowledge.memories[memory_id]


def load_evidence(connection, conversation_id: str, knowledge: ConversationKnowledge) -> ConversationEvidence:
    from continuator_memory.v2.memory_store import MemoryStore
    from continuator_memory.v2.store import SemanticStore

    semantic, memories = SemanticStore(connection), MemoryStore(connection)
    return ConversationEvidence(knowledge, {c.claim_id: c for c in semantic.get_claims(conversation_id)},
                                semantic.get_relations(conversation_id), memories.get_decisions(conversation_id),
                                {s.claim_id: s.memory_id for s in memories.get_sources(conversation_id)})


@dataclass(frozen=True)
class RelationshipCandidate:
    conversation_id: str
    source_memory_id: str
    target_memory_id: str
    relationship_type: str
    candidate_reason: str
    anchor: str
    evidence_claim_ids: tuple[str, ...]
    evidence_message_ids: tuple[str, ...]
    evidence: str
    source_entities: tuple[str, ...]
    target_entities: tuple[str, ...]
    source_status: str
    target_status: str
    source_slot: str | None
    target_slot: str | None
    source_subject: str | None
    target_subject: str | None
    existing_relationship: str | None
    decision: str
    rejection_reason: str | None
    failing_checks: tuple[str, ...] = ()
    detail: str = ""

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.conversation_id, self.source_memory_id, self.target_memory_id, self.relationship_type

    @property
    def candidate_id(self) -> str:
        return relationship_id(self.conversation_id, self.source_memory_id, self.relationship_type,
                               self.target_memory_id)

    @property
    def direction(self) -> str:
        return DIRECTION_OF[self.relationship_type]

    @property
    def missing(self) -> bool:
        """Not in the graph and not refuted by the representation (rejected or insufficient)."""
        return self.decision in (REJECTED_RULE, INSUFFICIENT_EVIDENCE)

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "conversation_id": self.conversation_id,
                "source_memory_id": self.source_memory_id, "target_memory_id": self.target_memory_id,
                "relationship_type": self.relationship_type, "direction": self.direction,
                "candidate_reason": self.candidate_reason, "anchor": self.anchor,
                "evidence_claim_ids": list(self.evidence_claim_ids),
                "evidence_message_ids": list(self.evidence_message_ids), "evidence": self.evidence,
                "source_entities": list(self.source_entities), "target_entities": list(self.target_entities),
                "source_status": self.source_status, "target_status": self.target_status,
                "source_slot": self.source_slot, "target_slot": self.target_slot,
                "source_subject": self.source_subject, "target_subject": self.target_subject,
                "existing_relationship": self.existing_relationship, "decision": self.decision,
                "rejection_reason": self.rejection_reason, "failing_checks": list(self.failing_checks),
                "detail": self.detail}


# ---- helpers --------------------------------------------------------------------------------


def _entity_ids(cm: ContextualMemory, roles: Iterable[str] | None = None) -> tuple[str, ...]:
    wanted = set(roles) if roles is not None else None
    return tuple(dict.fromkeys(r.entity_id for r in cm.entity_refs if r.entity_id and r.status == REF_RESOLVED
                               and r.role != ROLE_SPEAKER and (wanted is None or r.role in wanted)))


def _content_words(text: str) -> set[str]:
    return {w for w in normalize_value(text).split() if w not in STOPWORDS}


def _names(ev: ConversationEvidence) -> list[str]:
    names = set()
    for e in ev.knowledge.entities.values():
        names.update(n for n in (e.canonical_name, *e.aliases, *(x.name for x in e.names)) if n)
    for m in ev.graph.memories.values():
        names.update(m.entities)
    return sorted({n.lower() for n in names if len(n) >= 3})


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w'-]){re.escape(name)}(?![\w-])", text.lower()) is not None


def anchor(ev: ConversationEvidence, a: str, b: str) -> tuple[str | None, str]:
    """How two memories are tied to the same entity, if at all."""
    ma, mb = ev.memory(a), ev.memory(b)
    if ma.subject and ma.subject == mb.subject:
        return ANCHOR_SUBJECT, f"same consolidation subject {ma.subject!r}"
    shared = [e for e in _entity_ids(ev.contextual(a)) if e in set(_entity_ids(ev.contextual(b)))]
    if shared:
        return ANCHOR_ENTITY, f"shared phase 5B entity {', '.join(shared)}"
    common = [e for e in ma.entities if e in set(mb.entities)]
    if common:
        return ANCHOR_ENTITY, f"shared memory entity {', '.join(repr(e) for e in common)}"
    ta, tb = ev.contextual(a).contextual_content, ev.contextual(b).contextual_content
    names = [n for n in _names(ev) if _mentions(ta, n) and _mentions(tb, n)]
    if names:
        return ANCHOR_TEXT, f"name {names[0]!r} in both texts, no shared structured entity"
    return None, ""


def _subject_check(ev: ConversationEvidence, older: str, newer: str) -> str | None:
    mo, mn = ev.memory(older), ev.memory(newer)
    roles = (ROLE_SUBJECT, ROLE_POSSESSOR)
    joined = set(_entity_ids(ev.contextual(older), roles)) & set(_entity_ids(ev.contextual(newer), roles))
    if mo.subject and mo.subject == mn.subject:
        return None
    if joined:
        return "SUBJECT_ONLY_IN_ENTITY_LAYER"
    if mo.subject and mn.subject:
        return "DIFFERENT_SUBJECT"
    return "MISSING_SUBJECT"


def _blocked(ev: ConversationEvidence) -> set[str]:
    return {d.claim_id for d in ev.decisions if IDENTITY_AMBIGUOUS in d.unresolved_codes}


def extends_checks(ev: ConversationEvidence, older: str, newer: str, anchored: str) -> list[str]:
    """Failing checks of ``extends_reason`` for newer -> older, in its order."""
    mo, mn = ev.memory(older), ev.memory(newer)
    a, b = ev.claims.get(mo.claim_id), ev.claims.get(mn.claim_id)
    failing: list[str] = []
    if a is None or b is None:
        return ["MISSING_CLAIM"]
    subject = _subject_check(ev, older, newer)
    if subject:
        failing.append("MISSING_ENTITY" if anchored == ANCHOR_TEXT and subject != "SUBJECT_ONLY_IN_ENTITY_LAYER"
                       else subject)
    others = [e for e in mo.entities if e != mo.subject]
    if not others:
        failing.append("NO_OTHER_ENTITY")
    elif not set(others) <= set(mn.entities):
        failing.append("TARGET_ENTITIES_NOT_NAMED")
    if mo.status != mn.status or mo.status not in (STATUS_CURRENT, STATUS_HISTORICAL):
        failing.append("STATUS_MISMATCH")
    if "negative" in (mo.polarity, mn.polarity):
        failing.append("NEGATIVE_POLARITY")
    if mo.attribute and mo.attribute == mn.attribute:
        failing.append("SAME_SLOT")
    if mn.time_class in (TIME_TRANSITION, TIME_MIXED, TIME_AMBIGUOUS_PAST) or mo.time_class in (
            TIME_MIXED, TIME_AMBIGUOUS_PAST):
        failing.append("TEMPORAL_CLASS")
    if {mo.claim_id, mn.claim_id} & _blocked(ev):
        failing.append("AMBIGUOUS_ENTITY")
    if any(has_unresolved_reference(c) for c in (a, b)):
        failing.append("UNRESOLVED_PRONOUN")
    if any(END_OF_STATE.search(m.content.lower()) for m in (mo, mn)):
        failing.append("END_OF_STATE")
    if not _content_words(mn.content) - _content_words(mo.content):
        failing.append("NO_NEW_INFORMATION")
    passes = extends_reason(mo, mn, ev.claims, _blocked(ev)) is not None
    if passes != (not failing):
        raise AssertionError(f"{newer} EXTENDS {older}: diagnostic checks {failing} disagree with extends_reason")
    return failing


def _changes(m: Memory) -> bool:
    return m.time_class in CHANGING_CLASSES or TRANSITION.search(m.content.lower()) is not None


def _terms(ev: ConversationEvidence, m: Memory) -> set[str]:
    entity_words = {w for e in m.entities for w in normalize_value(e).split()}
    return {w for w in _content_words(m.content) - entity_words - CHANGE_WORDS
            if not VALUE_TOKEN.fullmatch(w) and len(w) > 2}


def _values(m: Memory) -> tuple[str, ...]:
    return tuple(sorted(VALUE_TOKEN.findall(m.content.lower())))


def version_checks(ev: ConversationEvidence, older: str, newer: str, anchored: str) -> list[str]:
    """Failing checks of the consolidation slot rule (SUPERSEDES) for newer -> older, in its order."""
    mo, mn = ev.memory(older), ev.memory(newer)
    claim = ev.claims.get(mn.claim_id)
    if claim is None or ev.claims.get(mo.claim_id) is None:
        return ["MISSING_CLAIM"]
    failing: list[str] = []
    if mn.time_class in (TIME_MIXED, TIME_AMBIGUOUS_PAST):
        failing.append("TEMPORAL_CLASS")
    elif mn.time_class not in CHANGING_CLASSES:
        failing.append("NOT_A_CHANGE")
    if has_unresolved_reference(claim):
        failing.append("UNRESOLVED_PRONOUN")
    if not mn.attribute or not mo.attribute:
        failing.append("MISSING_SLOT")
    elif mn.attribute != mo.attribute:
        failing.append("DIFFERENT_SLOT")
    subject = _subject_check(ev, older, newer)
    if subject:
        failing.append("MISSING_ENTITY" if anchored == ANCHOR_TEXT and subject != "SUBJECT_ONLY_IN_ENTITY_LAYER"
                       else subject)
    if mo.status == STATUS_SUPERSEDED:
        failing.append("TARGET_ALREADY_SUPERSEDED")
    elif mo.status != STATUS_CURRENT or mo.time_class not in REPLACEABLE_CLASSES:
        failing.append("STATUS_NOT_REPLACEABLE")
    if _values(mo) and _values(mo) == _values(mn):
        failing.append("RESTATEMENT")
    if not failing:
        failing.append("OTHER_EXISTING_RULE")  # every slot-rule condition holds, yet consolidation linked nothing
    return failing


# ---- candidate generation -------------------------------------------------------------------


def _candidate(ev: ConversationEvidence, source: str, target: str, type_: str, reason: str, anchored: str,
               claims: Sequence[str], evidence: str, failing: Sequence[str], existing: str | None = None,
               decision: str | None = None, detail: str = "") -> RelationshipCandidate:
    ms, mt = ev.memory(source), ev.memory(target)
    messages = tuple(dict.fromkeys(ev.claims[c].source_message_id for c in claims if c in ev.claims))
    failing = tuple(failing)
    return RelationshipCandidate(
        ev.conversation_id, source, target, type_, reason, anchored, tuple(claims), messages, evidence,
        _entity_ids(ev.contextual(source)), _entity_ids(ev.contextual(target)), ms.status, mt.status, ms.attribute,
        mt.attribute, ms.subject, mt.subject, existing,
        decision or (ACCEPTED_EXISTING if existing and not failing else decide(failing)),
        failing[0] if failing else None, failing, detail)


def _cue_position(text: str) -> int | None:
    match = TEXT_CUE.search(text.lower())
    return match.start() if match else None


def _claim_side(claim: Claim, message: str, cue: int) -> int:
    """-1 if most of the claim's content words occur before the cue in the message, 1 if after, else 0."""
    lowered = message.lower()
    hits = [lowered.find(w) for w in normalize_value(claim.content).split()
            if w not in STOPWORDS and len(w) > 3 and w in lowered]
    after = sum(h > cue for h in hits)
    return (after > len(hits) - after) - (after < len(hits) - after)


def generate_candidates(ev: ConversationEvidence) -> list[RelationshipCandidate]:
    """Every candidate of one conversation, deterministically ordered by generator then memory order."""
    graph, out = ev.graph, []
    keys: set[tuple[str, str, str, str]] = set()
    pairs: set[frozenset[str]] = set()

    def emit(c: RelationshipCandidate) -> None:
        if c.key in keys and c.decision != ACCEPTED_EXISTING:
            c = RelationshipCandidate(**{**c.__dict__, "decision": DUPLICATE})
        keys.add(c.key)
        pairs.add(frozenset((c.source_memory_id, c.target_memory_id)))
        out.append(c)

    for e in graph.edges:
        reason = EXISTING_VERSION_CHAIN if e.type in (LINK_UPDATES, LINK_SUPERSEDES) else EXISTING_LINK
        emit(_candidate(ev, e.source_memory_id, e.target_memory_id, e.type, reason, ANCHOR_EDGE, (e.claim_id,),
                        f"{e.origin}: {e.reason}", (), e.relationship_id))

    for r in sorted(ev.relations, key=lambda r: (claim_order(ev.claims[r.from_claim_id]) if r.from_claim_id in ev.claims
                                                  else (10**9, 0), r.relation_id)):
        type_ = _EXPLICIT_TYPES.get(r.type)
        if type_ is None:
            continue
        a, b = ev.memory_of.get(r.from_claim_id), ev.memory_of.get(r.to_claim_id)
        if a is None or b is None or a == b:
            continue  # the claim pair has no two memories to relate (reported in the relation audit)
        source, target = (b, a) if type_ == LINK_DERIVES else (a, b)
        reason = SUPPORTING_MEMORY_PLUS_EXPLICIT_INFERENCE if type_ == LINK_DERIVES else EXPLICIT_RELATION
        existing = [x for x in graph.related(source, target) if x.type == type_]
        if existing:
            emit(_candidate(ev, source, target, type_, reason, ANCHOR_RELATION, (r.from_claim_id, r.to_claim_id),
                            f"{r.relation_id} {r.type} ({r.evidence!r})", (), existing[0].relationship_id,
                            decision=DUPLICATE))
            continue
        failing, detail = [], ""
        if type_ == LINK_DERIVES:
            supported = ev.claims.get(r.to_claim_id)
            if INFERENCE_CUE.fullmatch(normalize_value(r.evidence or "")) is None:
                failing.append("NO_EXPLICIT_INFERENCE_CUE")
            if supported is None or r.source_message_id != supported.source_message_id:
                failing.append("CUE_NOT_IN_SUPPORTED_MESSAGE")
            if graph.related(source, target):
                failing.append("ALREADY_RELATED")
            if derives_cue(r, supported) != (not failing or failing == ["ALREADY_RELATED"]):
                raise AssertionError(f"{r.relation_id}: diagnostic checks {failing} disagree with derives_cue")
        else:
            decision = next((d for d in ev.decisions if d.claim_id == r.from_claim_id), None)
            parts = [p for p in ((decision.unresolved or "").split("; ") if decision else []) if r.relation_id in p]
            for p in parts:
                failing.append(_CONSOLIDATION_CODES.get(p.split(":", 1)[0], "OTHER_EXISTING_RULE"))
                detail = p
            if graph.related(source, target):
                failing.append("ALREADY_RELATED")
            if not failing:
                failing.append("OTHER_EXISTING_RULE")
                detail = f"consolidation: {decision.action}: {decision.reason}" if decision else "no decision"
        emit(_candidate(ev, source, target, type_, reason, ANCHOR_RELATION, (r.from_claim_id, r.to_claim_id),
                        f"{r.relation_id} {r.type} ({r.evidence!r})", failing, detail=detail))

    ordered = sorted(graph.memories.values(), key=memory_order)
    for i, newer in enumerate(ordered):
        for older in ordered[:i]:
            pair = frozenset((older.memory_id, newer.memory_id))
            if pair in pairs:
                continue
            anchored, why = anchor(ev, older.memory_id, newer.memory_id)
            if anchored is None:
                continue
            claims = (newer.claim_id, older.claim_id)
            shared = sorted(_terms(ev, older) & _terms(ev, newer))
            if (_changes(newer) and shared and newer.status not in (STATUS_FUTURE, STATUS_CONDITIONAL)
                    and older.status not in (STATUS_FUTURE, STATUS_CONDITIONAL)):
                emit(_candidate(ev, newer.memory_id, older.memory_id, LINK_SUPERSEDES,
                                SAME_ENTITY_SHARED_TERM_CHANGE_STATED, anchored, claims,
                                f"{why}; shared term {', '.join(repr(t) for t in shared)}; newer states a change",
                                version_checks(ev, older.memory_id, newer.memory_id, anchored)))
            else:
                emit(_candidate(ev, newer.memory_id, older.memory_id, LINK_EXTENDS,
                                SAME_SUBJECT_ADDITIONAL_INFORMATION, anchored, claims, why,
                                extends_checks(ev, older.memory_id, newer.memory_id, anchored)))

    supported_pairs = {(r.to_claim_id, r.from_claim_id) for r in ev.relations if r.type == "SUPPORTS"}
    messages = {m.message_id: m for m in ev.knowledge.messages}
    claims_of: dict[str, list[Claim]] = {}
    for c in sorted(ev.claims.values(), key=claim_order):
        if c.claim_id in ev.memory_of:
            claims_of.setdefault(c.source_message_id, []).append(c)
    for mid, claims in claims_of.items():
        text = messages[mid].content if mid in messages else ""
        cue = _cue_position(text)
        if cue is None:
            continue
        placed = [(c, _claim_side(c, text, cue)) for c in claims]
        conclusions = [c for c, side in placed if side > 0]
        supports = [(c, ANCHOR_MESSAGE, "same message, before the cue") for c, side in placed if side < 0]
        if cue == 0 or not supports:
            window = next(s.window for s in ev.contextual(ev.memory_of[claims[0].claim_id]).sources
                          if s.claim_id == claims[0].claim_id)
            earlier = [c for c in sorted(ev.claims.values(), key=claim_order)
                       if c.source_message_id in window.message_ids and c.source_message_id != mid
                       and c.claim_id in ev.memory_of]
            for concl in conclusions or claims:
                for c in earlier:
                    anchored, why = anchor(ev, ev.memory_of[c.claim_id], ev.memory_of[concl.claim_id])
                    if anchored is not None:
                        supports.append((c, anchored, f"source window, {why}"))
            conclusions = conclusions or claims
        for concl in conclusions:
            for sup, anchored, why in supports:
                d, s = ev.memory_of[concl.claim_id], ev.memory_of[sup.claim_id]
                if d == s or (concl.claim_id, sup.claim_id) in supported_pairs:
                    continue
                failing = ["NO_SUPPORTING_RELATION", *(["ALREADY_RELATED"] if graph.related(d, s) else [])]
                emit(_candidate(ev, d, s, LINK_DERIVES, INFERENCE_CUE_WITHOUT_RELATION, anchored,
                                (concl.claim_id, sup.claim_id),
                                f"cue {TEXT_CUE.search(text.lower()).group(0)!r} in {mid}: {why}", failing))
    return out


# ---- phase 5D compatibility and connections -------------------------------------------------


@dataclass(frozen=True)
class Traversal:
    traversable: bool
    direction: str  # from the seed
    role: str
    include: str


def traversal(source: str, target: str, type_: str, seed: str, neighbour_status: str, past: bool) -> Traversal:
    """Whether the phase 5D policy would follow this edge from ``seed`` to the other end."""
    direction = OUTGOING if seed == source else INCOMING
    rule = POLICY[(type_, direction)]
    ok = rule.include == ALWAYS or (rule.include == HISTORY and (past or neighbour_status != STATUS_SUPERSEDED))
    return Traversal(ok, direction, rule.role, rule.include)


@dataclass(frozen=True)
class Connection:
    candidate: RelationshipCandidate
    seed: str
    target: str
    traversal: Traversal

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate.candidate_id, "type": self.candidate.relationship_type,
                "decision": self.candidate.decision, "rejection_reason": self.candidate.rejection_reason,
                "seed": self.seed, "target": self.target, "direction_from_seed": self.traversal.direction,
                "phase5d_role": self.traversal.role, "phase5d_include": self.traversal.include,
                "phase5d_traversable": self.traversal.traversable}


def connections(candidates: Iterable[RelationshipCandidate], seeds: Iterable[str], targets: Iterable[str],
                statuses: Mapping[str, str], past: bool) -> list[Connection]:
    """Candidates joining a seed to a target outside the seeds, in either direction (duplicates skipped)."""
    seed_set, target_set, out = set(seeds), set(targets) - set(seeds), []
    for c in candidates:
        if c.decision == DUPLICATE:
            continue
        for seed, other in ((c.source_memory_id, c.target_memory_id), (c.target_memory_id, c.source_memory_id)):
            if seed in seed_set and other in target_set:
                out.append(Connection(c, seed, other, traversal(c.source_memory_id, c.target_memory_id,
                                                                c.relationship_type, seed, statuses[other], past)))
    return out


# ---- oracle graph (diagnostic only, never persisted) ---------------------------------------


@dataclass
class OracleResult:
    knowledge: ConversationKnowledge
    added: list[str] = field(default_factory=list)
    refused: list[dict[str, str]] = field(default_factory=list)


def oracle_knowledge(knowledge: ConversationKnowledge, candidates: Iterable[RelationshipCandidate]) -> OracleResult:
    """An in-memory copy of ``knowledge`` whose graph also holds ``candidates`` (oracle relationship
    diagnostic). The original graph and knowledge are not modified."""
    g = knowledge.graph
    chosen = [c for c in candidates if c.conversation_id == g.conversation_id and c.missing]
    links = [MemoryLink(r.source_memory_id, r.target_memory_id, r.type, r.claim_id, r.relation_id) for r in g.links]
    links += [MemoryLink(c.source_memory_id, c.target_memory_id, c.relationship_type, c.evidence_claim_ids[0])
              for c in chosen if c.relationship_type not in (LINK_EXTENDS, LINK_DERIVES)]
    oracle = RelationshipGraph(g.conversation_id, g.memories.values(), links)
    oracle.links = [*g.links, *(MemoryRelationship(r.conversation_id, r.source_memory_id, r.target_memory_id, r.type,
                                                   r.claim_id, None, ORIGIN_EXPLICIT, "oracle relationship diagnostic")
                                for r in oracle.links[len(g.links):])]
    result = OracleResult(ConversationKnowledge(knowledge.conversation_id, oracle, knowledge.memories,
                                                knowledge.messages, knowledge.entities))
    result.added = [c.candidate_id for c in chosen if c.relationship_type not in (LINK_EXTENDS, LINK_DERIVES)]
    for r in g.additive:
        oracle._additive[r.key] = r
    for c in chosen:
        if c.relationship_type in (LINK_EXTENDS, LINK_DERIVES):
            rejection = oracle.add(MemoryRelationship(c.conversation_id, c.source_memory_id, c.target_memory_id,
                                                      c.relationship_type, c.evidence_claim_ids[0], None,
                                                      ORIGIN_EXPLICIT, "oracle relationship diagnostic"))
            if rejection is None:
                result.added.append(c.candidate_id)
            else:
                result.refused.append({"candidate_id": c.candidate_id, "code": rejection.code})
    return result


def type_counts(candidates: Iterable[RelationshipCandidate]) -> dict[str, dict[str, int]]:
    out = {t: {d: 0 for d in DECISIONS} for t in RELATIONSHIP_TYPES}
    for c in candidates:
        out[c.relationship_type][c.decision] += 1
    return out
