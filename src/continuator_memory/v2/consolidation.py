"""Deterministic consolidation of validated V2 claims into versioned memories (phases 2A-2C).

Claims are consumed in conversation order (message position, then claim ordinal). There is no
predicate vocabulary and no LLM: two claims are treated as the same underlying fact only on
one of these signals, and otherwise each claim keeps its own memory (a duplicate memory is
always preferred to a false merge).

  explicit relation  a validated UPDATES / CONTRADICTS / REFERS_TO relation. Its target is
                     resolved by exact claim ID, then exact memory ID, then (only if neither
                     exists) the single memory compatible with the claim's identity: the same
                     slot for UPDATES / CONTRADICTS, a superset of its entities for REFERS_TO.
                     Several candidates or none leave it unresolved. UPDATES / CONTRADICTS are
                     refused when either claim contains an unresolved third-person reference,
                     the explicit subjects (metadata subject / subject_id) resolve to different
                     identities, or the target is already superseded.
  restatement        same normalized content, polarity, entities, qualifiers and an absolute
                     (or identically resolved) time. Never for claims with an unresolved
                     reference or relative time ("tomorrow" may name a different occasion).
  past restatement   a past-state claim whose content, minus tense words and its temporal
                     expression, equals exactly one non-current memory's ("used to live in
                     Delhi" / "lives in Delhi"): adds provenance, never reactivates it.
  slot               same resolved subject identity and same attribute (see
                     ``continuator_memory.v2.identity``), and the newer claim states a present
                     state or a change. Only one current memory may occupy the slot.

Identity (``identity``) decides whether two claims are about the same entity and slot;
temporal semantics (``temporal``) decide, for a matched slot, which version is current.
Observation order is not truth: only present-state and change claims can supersede, and when
both sides carry stated dates, event order decides which version is newer.

Rules, in order:
  IGNORE         claim already consolidated, or from another conversation.
  UPDATE         explicit UPDATES from a present/change claim to a current memory that is not
                 dated after it: new memory, old one superseded, linked.
  CONTRADICTION  explicit CONTRADICTS, or the same change stated again with a disjoint date
                 ("moved to X in 2024" / "in 2023"): new memory, both contested, linked.
  KEEP           restatement of a non-superseded memory, or past restatement of a superseded
                 or historical one: one more source, status unchanged.
  HISTORICAL     past-state claim (historical, never current, never supersedes), or a change
                 dated before the current memory of its slot (recorded as its older version).
  SUPERSEDE      slot match whose event order is later or not contradicted by dates.
  CREATE         everything else: future and conditional claims (status future/conditional,
                 never supersede), ambiguous past tense, mixed past/present wording, and any
                 uncertain case (reported unresolved with a ``memory.REASON_CODES`` prefix).

Validity: ``valid_from``/``valid_until`` come only from the claim's temporal annotation; a
superseded memory's ``valid_until`` is the replacing version's stated start, if it has one.
Message timestamps are observation time only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from continuator_memory.memory.models import normalize_value
from continuator_memory.memory.text import STOPWORDS
from continuator_memory.v2.identity import (
    ClaimIdentity,
    IdentityRegistry,
    claim_subject,
    entity_key,
    entity_keys,
)
from continuator_memory.v2.memory import (
    CONTRADICTION,
    CREATE,
    CREATING_ACTIONS,
    HISTORICAL,
    IDENTITY_AMBIGUOUS,
    IGNORE,
    KEEP,
    LINK_CONTRADICTS,
    LINK_REFERS_TO,
    LINK_SUPERSEDES,
    LINK_UPDATES,
    RELATION_IDENTITY_CONFLICT,
    RELATION_TARGET_AMBIGUOUS,
    RELATION_TARGET_MISSING,
    RELATION_TARGET_SUPERSEDED,
    RESTATEMENT_AMBIGUOUS,
    SLOT_AMBIGUOUS,
    STATUS_CONDITIONAL,
    STATUS_CONTESTED,
    STATUS_CURRENT,
    STATUS_FUTURE,
    STATUS_HISTORICAL,
    STATUS_SUPERSEDED,
    SUBJECT_MISSING,
    SUPERSEDE,
    TEMPORAL_AMBIGUOUS,
    UPDATE,
    Decision,
    Memory,
    MemoryLink,
    MemorySource,
    memory_id,
)
from continuator_memory.v2.models import Claim, Relation
from continuator_memory.v2.temporal import (
    CHANGING_CLASSES,
    REPLACEABLE_CLASSES,
    TIME_AMBIGUOUS_PAST,
    TIME_CONDITIONAL,
    TIME_FUTURE,
    TIME_MIXED,
    TIME_PAST,
    TIME_TRANSITION,
    TemporalBounds,
    before,
    blocks_change,
    classify,
    observation_date,
    temporal_bounds,
    undated_content,
)

__all__ = ["Consolidator", "ConsolidationResult", "claim_subject", "consolidate", "entity_key", "event_order",
           "fact_key", "restatement_key"]

AMBIGUOUS_PRONOUNS = frozenset("he she him her his hers they them their theirs it its".split())
RELATIVE_TIME = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|next|last|this (?:week|month|year|weekend|morning|evening)|ago|"
    r"recently|soon)\b", re.IGNORECASE)
YEAR = re.compile(r"\b\d{4}\b")
# Words that carry tense or currentness, not the fact itself.
TENSE_WORDS = frozenset(
    "used formerly previously former originally past back then earlier before now currently current still "
    "nowadays anymore".split())
STATUS_OF_CLASS = {TIME_PAST: STATUS_HISTORICAL, TIME_FUTURE: STATUS_FUTURE, TIME_CONDITIONAL: STATUS_CONDITIONAL}
LATER, EARLIER, UNKNOWN = "later", "earlier", "unknown"
STATE_RELATIONS = (LINK_UPDATES, LINK_CONTRADICTS, LINK_REFERS_TO)


def has_unresolved_reference(claim: Claim) -> bool:
    return bool(AMBIGUOUS_PRONOUNS & set(normalize_value(claim.content).split()))


def restatement_key(claim: Claim) -> tuple | None:
    """Equality key for "the same information"; None when equal wording may not mean equal facts."""
    if has_unresolved_reference(claim):
        return None
    resolved = claim.temporal.resolved_date if claim.temporal else None
    if resolved:
        when: str | None = resolved
    elif claim.temporal:
        if not YEAR.search(claim.temporal.expression):
            return None
        when = normalize_value(claim.temporal.expression)
    else:
        when = None
    if not resolved and RELATIVE_TIME.search(claim.content):
        return None
    return (normalize_value(claim.content), claim.polarity or "positive", frozenset(entity_keys(claim)),
            frozenset(normalize_value(q) for q in claim.qualifiers), when)


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) - len(suffix) >= 3 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    return token[:-1] if len(token) > 3 and token.endswith("e") else token


def fact_key(claim: Claim) -> tuple | None:
    """The fact without tense or time: ordered stemmed content words, polarity and entities.
    "The user used to live in Delhi." and "The user lives in Delhi." share a key."""
    if has_unresolved_reference(claim):
        return None
    words = tuple(_stem(w) for w in undated_content(claim).split() if w not in STOPWORDS and w not in TENSE_WORDS)
    return words, claim.polarity or "positive", frozenset(entity_keys(claim))


def _ordinal(claim_id: str) -> int:
    tail = claim_id.rsplit(":c", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def claim_order(claim: Claim) -> tuple[int, int, str]:
    return claim.provenance.position, _ordinal(claim.claim_id), claim.claim_id


def event_order(new: Memory, old: Memory) -> str:
    """Whether ``new``'s state starts after ``old``'s, using stated dates only.

    Both starts stated: compare them. Only ``new``'s start stated: it must fall after the
    moment ``old`` was observed to hold. Only ``old``'s start stated: ``new`` must not have
    been observed before it. With no stated dates there is no contrary evidence (LATER)."""
    start = new.valid_from
    if start:
        if old.valid_from:
            if before(old.valid_from, start):
                return LATER
            return EARLIER if before(start, old.valid_from) else UNKNOWN
        old_observed = observation_date(old.created_at)
        return UNKNOWN if old_observed and not before(old_observed, start) else LATER
    new_observed = observation_date(new.created_at)
    if old.valid_from and new_observed and before(new_observed, old.valid_from):
        return UNKNOWN
    return LATER


# ---- consolidation ----------------------------------------------------------


@dataclass
class ConsolidationResult:
    decisions: list[Decision]
    links: list[MemoryLink]
    unresolved_relations: list[tuple[str, str]] = field(default_factory=list)  # (relation_id, reason)

    def _of(self, *actions: str) -> list[Decision]:
        return [d for d in self.decisions if d.action in actions]

    @property
    def created(self) -> list[str]:
        return [d.memory_id for d in self._of(*CREATING_ACTIONS)]

    @property
    def kept(self) -> list[tuple[str, str]]:
        return [(d.claim_id, d.memory_id) for d in self._of(KEEP)]

    @property
    def superseded(self) -> list[str]:
        return [m for d in self.decisions for m in d.superseded]

    @property
    def contradictions(self) -> list[tuple[str, str]]:
        return [(d.memory_id, m) for d in self.decisions for m in d.contradicted]

    @property
    def unresolved(self) -> list[Decision]:
        return [d for d in self.decisions if d.unresolved]

    @property
    def ignored(self) -> list[str]:
        return [d.claim_id for d in self._of(IGNORE)]


def _joined(problems: Sequence[str]) -> str | None:
    return "; ".join(dict.fromkeys(problems)) or None


class Consolidator:
    """Consolidation state for one conversation.

    ``apply`` consumes claims as they arrive. ``add`` does the same but guarantees the result
    equals a full replay of every claim and relation it has been given: when a batch contains
    a claim that sorts before one already consolidated, or a relation whose source claim was
    already consolidated, it replays the whole stream instead of applying incrementally."""

    def __init__(self, conversation_id: str, claim_type_slots: bool = True):
        self.conversation_id = conversation_id
        self.claim_type_slots = claim_type_slots
        self.memories: dict[str, Memory] = {}
        self.sources: list[MemorySource] = []
        self.links: list[MemoryLink] = []
        self.decisions: list[Decision] = []
        self.identities = IdentityRegistry()
        self._claims: dict[str, Claim] = {}
        self._identity: dict[str, ClaimIdentity] = {}  # claim_id -> identity at consolidation time
        self._memory_of: dict[str, str] = {}  # claim_id -> memory_id
        self._by_key: dict[tuple, str] = {}  # restatement key -> latest memory_id with it
        self._fact: dict[str, tuple | None] = {}  # memory_id -> fact_key of its claim
        self._undated: dict[str, tuple] = {}  # memory_id -> (undated content, polarity, entities)
        self._pending: dict[str, list[Relation]] = {}  # from_claim_id -> relations not yet applied
        self._seen_relations: set[str] = set()
        self._last_order: tuple[int, int, str] | None = None
        self._stream_claims: dict[str, Claim] = {}  # everything given to ``add``, for replay
        self._stream_relations: dict[str, Relation] = {}

    # ---- views --------------------------------------------------------------

    def current(self) -> list[Memory]:
        return [m for m in self._ordered() if m.is_current]

    def _ordered(self) -> list[Memory]:
        return sorted(self.memories.values(), key=lambda m: (m.position, _ordinal(m.claim_id), m.memory_id))

    def sources_of(self, mid: str) -> list[MemorySource]:
        return [s for s in self.sources if s.memory_id == mid]

    def history(self, mid: str) -> list[Memory]:
        """Older versions of ``mid``, newest first, following UPDATES/SUPERSEDES links."""
        out: list[Memory] = []
        frontier = [mid]
        while frontier:
            newer = frontier.pop(0)
            for link in self.links:
                if (link.newer_memory_id == newer and link.type in (LINK_UPDATES, LINK_SUPERSEDES)
                        and link.older_memory_id not in {m.memory_id for m in out}):
                    out.append(self.memories[link.older_memory_id])
                    frontier.append(link.older_memory_id)
        return out

    def memory_for_claim(self, claim_id: str) -> Memory | None:
        mid = self._memory_of.get(claim_id)
        return self.memories[mid] if mid else None

    # ---- apply --------------------------------------------------------------

    def apply(self, claims: Iterable[Claim], relations: Iterable[Relation] = ()) -> ConsolidationResult:
        first_decision, first_link = len(self.decisions), len(self.links)
        unresolved_relations: list[tuple[str, str]] = []
        for r in sorted(relations, key=lambda r: (r.from_claim_id, r.relation_id)):
            if r.relation_id in self._seen_relations:
                continue
            self._seen_relations.add(r.relation_id)
            if r.type not in STATE_RELATIONS:
                continue  # CAUSED_BY / SUPPORTS do not change memory state
            if r.from_claim_id in self._claims:
                unresolved_relations.append((r.relation_id, f"{r.from_claim_id} was consolidated before the relation arrived"))
                continue
            self._pending.setdefault(r.from_claim_id, []).append(r)
        for claim in sorted(claims, key=claim_order):
            self.decisions.append(self._consolidate(claim))
        return ConsolidationResult(self.decisions[first_decision:], self.links[first_link:], unresolved_relations)

    def add(self, claims: Iterable[Claim], relations: Iterable[Relation] = ()) -> ConsolidationResult:
        """Incremental consolidation whose state always equals a full replay of the stream."""
        new_claims = [c for c in claims if c.claim_id not in self._stream_claims]
        new_relations = [r for r in relations if r.relation_id not in self._stream_relations]
        for c in new_claims:
            self._stream_claims[c.claim_id] = c
        for r in new_relations:
            self._stream_relations[r.relation_id] = r
        out_of_order = self._last_order is not None and any(claim_order(c) < self._last_order for c in new_claims)
        late_relation = any(r.type in STATE_RELATIONS and r.from_claim_id in self._claims for r in new_relations)
        if not (out_of_order or late_relation):
            return self.apply(new_claims, new_relations)
        replay = Consolidator(self.conversation_id, self.claim_type_slots)
        replay.apply(self._stream_claims.values(), self._stream_relations.values())
        replay._stream_claims, replay._stream_relations = self._stream_claims, self._stream_relations
        self.__dict__.update(replay.__dict__)
        new_ids = {c.claim_id for c in new_claims}
        return ConsolidationResult([d for d in self.decisions if d.claim_id in new_ids],
                                   [link for link in self.links if link.claim_id in new_ids])

    def _consolidate(self, claim: Claim) -> Decision:
        cid = claim.claim_id
        if cid in self._claims:
            return Decision(cid, IGNORE, self._memory_of.get(cid), reason="claim already consolidated")
        if claim.provenance.conversation_id != self.conversation_id:
            return Decision(cid, IGNORE, None, reason=f"claim belongs to {claim.provenance.conversation_id}")
        self._claims[cid] = claim
        order = claim_order(claim)
        self._last_order = order if self._last_order is None or order > self._last_order else self._last_order
        ident = self.identities.identify(claim, self.claim_type_slots)
        self._identity[cid] = ident
        relations = self._pending.pop(cid, [])
        time_class = classify(claim)
        bounds = temporal_bounds(claim, time_class == TIME_TRANSITION)
        if any(r.type in (LINK_UPDATES, LINK_CONTRADICTS) for r in relations):
            return self._apply_explicit(claim, ident, relations, time_class, bounds)

        key = restatement_key(claim)
        existing = self.memories.get(self._by_key[key]) if key is not None and key in self._by_key else None
        if existing is not None and existing.status != STATUS_SUPERSEDED:
            self._add_source(existing, claim)
            return Decision(cid, KEEP, existing.memory_id, reason=f"restates {existing.memory_id}",
                            unresolved=_joined([*ident.problems, *self._refers_to(claim, ident, existing, relations)]))
        if existing is not None:
            mem = self._create(claim, ident, STATUS_CURRENT, time_class, bounds)
            problems = [*ident.problems,
                        f"{RESTATEMENT_AMBIGUOUS}: restates superseded {existing.memory_id}; not treated as current again",
                        *self._refers_to(claim, ident, mem, relations)]
            return Decision(cid, CREATE, mem.memory_id, reason="restatement of a superseded memory",
                            unresolved=_joined(problems))

        if time_class == TIME_PAST:
            return self._past(claim, ident, relations, bounds)
        if time_class in (TIME_FUTURE, TIME_CONDITIONAL):
            mem = self._create(claim, ident, STATUS_OF_CLASS[time_class], time_class, bounds)
            return Decision(cid, CREATE, mem.memory_id, reason=f"{time_class} statement; not current",
                            unresolved=_joined([*ident.problems, *self._refers_to(claim, ident, mem, relations)]))

        conflict = self._date_conflict(claim, time_class, bounds)
        if conflict is not None:
            mem = self._create(claim, ident, STATUS_CONTESTED, time_class, bounds)
            self.links.append(MemoryLink(mem.memory_id, conflict.memory_id, LINK_CONTRADICTS, cid))
            conflict.status = STATUS_CONTESTED
            return Decision(cid, CONTRADICTION, mem.memory_id, contradicted=(conflict.memory_id,),
                            reason=f"same change as {conflict.memory_id} with a conflicting date",
                            unresolved=_joined([*ident.problems, *self._refers_to(claim, ident, mem, relations)]))

        target, problem = self._slot_target(claim, ident, time_class)
        mem = self._create(claim, ident, STATUS_CURRENT, time_class, bounds)
        problems = [*ident.problems, *([problem] if problem else []), *self._refers_to(claim, ident, mem, relations)]
        if target is None:
            return Decision(cid, CREATE, mem.memory_id, reason="no safe existing match", unresolved=_joined(problems))
        order = event_order(mem, target)
        if order == LATER:
            self._supersede(target, mem, LINK_SUPERSEDES, cid)
            return Decision(cid, SUPERSEDE, mem.memory_id, superseded=(target.memory_id,),
                            reason=f"same subject and attribute as {target.memory_id}, change stated",
                            unresolved=_joined(problems))
        if order == EARLIER:
            self._supersede(mem, target, LINK_SUPERSEDES, cid)
            return Decision(cid, HISTORICAL, mem.memory_id,
                            reason=f"dated before {target.memory_id}; recorded as its earlier version",
                            unresolved=_joined(problems))
        problems.insert(0, f"{TEMPORAL_AMBIGUOUS}: event order relative to {target.memory_id} cannot be "
                           f"established from stated dates")
        return Decision(cid, CREATE, mem.memory_id, reason="no safe existing match", unresolved=_joined(problems))

    def _past(self, claim: Claim, ident: ClaimIdentity, relations: Sequence[Relation],
              bounds: TemporalBounds) -> Decision:
        key = fact_key(claim)
        matches = [m for m in self._ordered() if key is not None and self._fact.get(m.memory_id) == key]
        current = [m.memory_id for m in matches if m.is_current]
        earlier = [m for m in matches if m.status in (STATUS_SUPERSEDED, STATUS_HISTORICAL)]
        if len(earlier) == 1 and not current:
            target = earlier[0]
            self._add_source(target, claim)
            return Decision(claim.claim_id, KEEP, target.memory_id,
                            reason=f"past-state restatement of {target.memory_id}; status unchanged",
                            unresolved=_joined([*ident.problems, *self._refers_to(claim, ident, target, relations)]))
        mem = self._create(claim, ident, STATUS_HISTORICAL, TIME_PAST, bounds)
        problems = list(ident.problems)
        if current:
            problems.append(f"{RESTATEMENT_AMBIGUOUS}: past-state restatement of current {', '.join(current)}; "
                            f"currentness unchanged")
        elif len(earlier) > 1:
            problems.append(f"{RESTATEMENT_AMBIGUOUS}: past-state restatement matches {len(earlier)} non-current "
                            f"memories; kept apart")
        problems += self._refers_to(claim, ident, mem, relations)
        return Decision(claim.claim_id, HISTORICAL, mem.memory_id, reason="states a past state",
                        unresolved=_joined(problems))

    def _apply_explicit(self, claim: Claim, ident: ClaimIdentity, relations: Sequence[Relation], time_class: str,
                        bounds: TemporalBounds) -> Decision:
        mem = self._create(claim, ident, STATUS_OF_CLASS.get(time_class, STATUS_CURRENT), time_class, bounds)
        changes_state = time_class not in (TIME_PAST, TIME_FUTURE, TIME_CONDITIONAL, TIME_AMBIGUOUS_PAST, TIME_MIXED)
        superseded: list[str] = []
        contradicted: list[str] = []
        problems: list[str] = list(ident.problems)
        for r in relations:
            if r.type == LINK_REFERS_TO:
                continue
            target, problem = self._relation_target(claim, ident, r, mem)
            if problem:
                problems.append(problem)
                continue
            if r.type == LINK_UPDATES:
                order = event_order(mem, target) if changes_state and target.is_current else None
                if order == LATER:
                    self._supersede(target, mem, LINK_UPDATES, claim.claim_id, r.relation_id)
                    superseded.append(target.memory_id)
                    continue
                self.links.append(MemoryLink(mem.memory_id, target.memory_id, LINK_UPDATES, claim.claim_id,
                                             r.relation_id))
                if target.is_current and not changes_state:
                    problems.append(f"{TEMPORAL_AMBIGUOUS}: UPDATES {r.relation_id}: {time_class} claim does not "
                                    f"supersede {target.memory_id}")
                elif order is not None:
                    problems.append(f"{TEMPORAL_AMBIGUOUS}: UPDATES {r.relation_id}: stated dates do not place it "
                                    f"after {target.memory_id}")
            else:
                self.links.append(MemoryLink(mem.memory_id, target.memory_id, LINK_CONTRADICTS, claim.claim_id,
                                             r.relation_id))
                if target.status == STATUS_CURRENT:
                    target.status = STATUS_CONTESTED
                if mem.status == STATUS_CURRENT:
                    mem.status = STATUS_CONTESTED
                contradicted.append(target.memory_id)
        problems += self._refers_to(claim, ident, mem, relations)
        if superseded:
            action, reason = UPDATE, "explicit UPDATES relation"
        elif contradicted:
            action, reason = CONTRADICTION, "explicit CONTRADICTS relation"
        elif mem.status == STATUS_HISTORICAL:
            action, reason = HISTORICAL, "states a past state"
        elif any(link.newer_memory_id == mem.memory_id for link in self.links):
            action, reason = CREATE, "explicit UPDATES without supersession; linked only"
        else:
            action, reason = CREATE, "explicit relation could not be applied safely"
        return Decision(claim.claim_id, action, mem.memory_id, tuple(superseded), tuple(contradicted), reason,
                        _joined(problems))

    # ---- targets ------------------------------------------------------------

    def _relation_target(self, claim: Claim, ident: ClaimIdentity, r: Relation,
                         own: Memory) -> tuple[Memory | None, str | None]:
        """Resolve a relation's target memory: exact claim ID, exact memory ID, then the single
        identity-compatible memory. UPDATES / CONTRADICTS also need compatible identities."""
        label = f"{r.type} {r.relation_id}"
        if r.to_claim_id in self._claims:
            target = self.memory_for_claim(r.to_claim_id)
        elif r.to_claim_id in self.memories:
            target = self.memories[r.to_claim_id]
        else:
            target, problem = self._compatible_target(claim, ident, r, own)
            if problem:
                return None, problem
        if r.type == LINK_REFERS_TO:
            return (None, None) if target.memory_id == own.memory_id else (target, None)
        if target.memory_id == own.memory_id:
            return None, f"{RELATION_TARGET_MISSING}: {label}: target is the claim's own memory"
        older = self._claims[target.claim_id]
        for c in (claim, older):
            if has_unresolved_reference(c):
                return None, f"{IDENTITY_AMBIGUOUS}: {label}: unresolved reference in {c.claim_id}"
        a, b = ident.explicit_subject, self._identity[older.claim_id].explicit_subject
        if a and b and a != b:
            return None, f"{RELATION_IDENTITY_CONFLICT}: {label}: explicit subjects differ ({a!r} vs {b!r})"
        if target.status == STATUS_SUPERSEDED:
            return None, (f"{RELATION_TARGET_SUPERSEDED}: {label}: target {target.memory_id} already superseded "
                          f"by {target.superseded_by}")
        return target, None

    def _compatible_target(self, claim: Claim, ident: ClaimIdentity, r: Relation,
                           own: Memory) -> tuple[Memory | None, str | None]:
        label = f"{r.type} {r.relation_id}"
        missing = f"{RELATION_TARGET_MISSING}: {label}: target {r.to_claim_id} not consolidated"
        if has_unresolved_reference(claim):
            return None, f"{missing}, and the claim has an unresolved reference"
        others = [m for m in self._ordered() if m.memory_id != own.memory_id]
        if r.type == LINK_REFERS_TO:
            wanted = set(ident.entities)
            candidates = [m for m in others if wanted and wanted <= set(m.entities)]
        elif ident.subject and ident.attribute:
            candidates = [m for m in others if m.is_current and m.subject == ident.subject
                          and m.attribute == ident.attribute]
        else:
            candidates = []
        if not candidates:
            return None, missing
        if len(candidates) > 1:
            return None, (f"{RELATION_TARGET_AMBIGUOUS}: {label}: target {r.to_claim_id} not consolidated and "
                          f"{len(candidates)} memories are compatible")
        return candidates[0], None

    def _slot_target(self, claim: Claim, ident: ClaimIdentity, time_class: str) -> tuple[Memory | None, str | None]:
        """The single current memory this claim may replace by the slot rule, or a reason it can't."""
        if time_class == TIME_MIXED:
            return None, f"{TEMPORAL_AMBIGUOUS}: claim mixes past-state and present wording; not used for state transitions"
        if time_class == TIME_AMBIGUOUS_PAST:
            if ident.subject and ident.attribute:
                return None, (f"{TEMPORAL_AMBIGUOUS}: past tense without explicit past-state wording; "
                              f"not used for state transitions")
            return None, None
        if time_class not in CHANGING_CLASSES or blocks_change(claim):
            return None, None
        if has_unresolved_reference(claim):
            return None, f"{IDENTITY_AMBIGUOUS}: change stated but the claim has an unresolved reference"
        if not ident.attribute:
            return None, None
        in_attribute = [m for m in self._ordered() if m.status == STATUS_CURRENT and m.time_class in REPLACEABLE_CLASSES
                        and m.attribute == ident.attribute]
        if not ident.subject:
            if in_attribute:
                return None, (f"{SUBJECT_MISSING}: change stated for {ident.attribute!r} without a safe subject "
                              f"identity; {len(in_attribute)} memories left unchanged")
            return None, None
        candidates = [m for m in in_attribute if m.subject == ident.subject]
        if not candidates:
            return None, None
        if len(candidates) > 1:
            return None, (f"{SLOT_AMBIGUOUS}: change stated but {len(candidates)} current memories share subject and "
                          f"claim_type/attribute")
        return candidates[0], None

    def _date_conflict(self, claim: Claim, time_class: str, bounds: TemporalBounds) -> Memory | None:
        """A current memory stating the same change with a disjoint date."""
        if time_class != TIME_TRANSITION or not bounds.valid_from or has_unresolved_reference(claim):
            return None
        key = (undated_content(claim), claim.polarity or "positive", frozenset(entity_keys(claim)))
        for m in self._ordered():
            if (m.is_current and m.time_class == TIME_TRANSITION and m.valid_from and self._undated[m.memory_id] == key
                    and (before(m.valid_from, bounds.valid_from) or before(bounds.valid_from, m.valid_from))):
                return m
        return None

    def _refers_to(self, claim: Claim, ident: ClaimIdentity, mem: Memory, relations: Sequence[Relation]) -> list[str]:
        problems = []
        for r in relations:
            if r.type != LINK_REFERS_TO:
                continue
            target, problem = self._relation_target(claim, ident, r, mem)
            if problem:
                problems.append(problem)
            elif target is not None:
                self.links.append(MemoryLink(mem.memory_id, target.memory_id, LINK_REFERS_TO, claim.claim_id,
                                             r.relation_id))
        return problems

    # ---- state changes ------------------------------------------------------

    def _create(self, claim: Claim, ident: ClaimIdentity, status: str, time_class: str,
                bounds: TemporalBounds) -> Memory:
        p = claim.provenance
        mem = Memory(
            memory_id=memory_id(claim.claim_id), conversation_id=p.conversation_id, content=claim.content,
            status=status, claim_id=claim.claim_id, message_id=p.message_id, position=p.position,
            created_at=p.timestamp, valid_from=bounds.valid_from, valid_until=bounds.valid_until,
            subject=ident.subject, entities=ident.entities, claim_type=claim.claim_type,
            polarity=claim.polarity, temporal=claim.temporal.expression if claim.temporal else None,
            event_time=bounds.event_time, time_class=time_class, attribute=ident.attribute)
        self.memories[mem.memory_id] = mem
        key = restatement_key(claim)
        if key is not None:
            self._by_key[key] = mem.memory_id
        self._fact[mem.memory_id] = fact_key(claim)
        self._undated[mem.memory_id] = (undated_content(claim), claim.polarity or "positive",
                                        frozenset(entity_keys(claim)))
        self._add_source(mem, claim)
        return mem

    def _add_source(self, mem: Memory, claim: Claim) -> None:
        p = claim.provenance
        self.sources.append(MemorySource(mem.memory_id, claim.claim_id, p.message_id, p.position, p.timestamp))
        self._memory_of[claim.claim_id] = mem.memory_id

    def _supersede(self, older: Memory, newer: Memory, link_type: str, claim_id: str,
                   relation_id: str | None = None) -> None:
        older.status = STATUS_SUPERSEDED
        older.superseded_by = newer.memory_id
        if older.valid_until is None:
            older.valid_until = newer.valid_from
        self.links.append(MemoryLink(newer.memory_id, older.memory_id, link_type, claim_id, relation_id))


def consolidate(conversation_id: str, claims: Iterable[Claim], relations: Iterable[Relation] = (),
                claim_type_slots: bool = True) -> Consolidator:
    consolidator = Consolidator(conversation_id, claim_type_slots)
    consolidator.apply(claims, relations)
    return consolidator
