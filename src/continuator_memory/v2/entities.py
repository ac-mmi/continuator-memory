"""Entity-centric memory (phase 5B): conversation-scoped entities and memory <-> entity references.

    source window --> claim --> contextual memory (5A) --> entities + entity references (5B)

Nothing here calls a model or changes memories, consolidation or retrieval. Entities are derived
from what the earlier phases already established:

  speaker entities  one per conversational role that a memory involves ("user", "assistant"). Its
                    name, aliases and ambiguity come from the phase 5A ``SpeakerRegistry`` over the
                    whole conversation: a role named twice differently stays ambiguous and unnamed.
  named entities    the claim's own entities (phase 1, validated against the source) that are
                    written as names (a capitalized word present in the claim or source message),
                    keyed by the phase 2C identity: exact normalized name, an explicit ``entity_ids``
                    ID, or an explicit ``aliases`` declaration. Lowercase things ("the picnic") are
                    not entities.

Identity is conversation-local: entity IDs embed the conversation, and nothing is merged across
conversations or by spelling, similarity or embeddings. Inside a conversation a name joins a
speaker entity only when that exact name (or a declared alias) is the speaker's established name.
A name the phase 2C registry cannot resolve, or that names two speakers, is kept as an ambiguous
reference with no entity.

Reference roles, per source claim of a memory:

  speaker    the role that asserted the claim
  subject    the phase 2C subject: explicit metadata subject / subject_id, or the claim opening with
             the user / assistant / one of its own entities, not in the possessive
  possessor  the claim opens with "<entity>'s ..." ("The user's dog is Pepper.")
  mentioned  every other entity of the claim, and "the user" / "the assistant" mentioned in it

Entity types come from a small deterministic ``EntityTyper`` over the memories that mention the
entity (``CueEntityTyper``: explicit wording cues only, conflicting cues give "other").
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol
from urllib.parse import quote

from continuator_memory.memory.models import normalize_value
from continuator_memory.models import Message
from continuator_memory.v2.consolidation import _ordinal, claim_order
from continuator_memory.v2.context import (
    SPEAKER_AMBIGUOUS,
    SPEAKER_NAMED,
    SPEAKER_STATUSES,
    SPEAKER_UNNAMED,
    ClaimContext,
    ContextualMemory,
    MessageLookup,
    SourceContext,
    SourceContextError,
    SourceWindow,
    SpeakerRegistry,
    resolve_window,
)
from continuator_memory.v2.identity import SPEAKERS, IdentityRegistry, _mapping, claim_subject, entity_key, entity_keys
from continuator_memory.v2.memory import IDENTITY_AMBIGUOUS, Memory, MemorySource
from continuator_memory.v2.models import Claim

PERSON, ORGANIZATION, PROJECT, PLACE, PRODUCT, PET, OTHER = (
    "person", "organization", "project", "place", "product", "pet", "other")
ENTITY_TYPES = (PERSON, ORGANIZATION, PROJECT, PLACE, PRODUCT, PET, OTHER)
ENTITY_STATUSES = SPEAKER_STATUSES  # named / unnamed (a known actor without a name) / ambiguous

ROLE_SPEAKER, ROLE_SUBJECT, ROLE_POSSESSOR, ROLE_MENTIONED = "speaker", "subject", "possessor", "mentioned"
REF_ROLES = (ROLE_SPEAKER, ROLE_SUBJECT, ROLE_POSSESSOR, ROLE_MENTIONED)
REF_RESOLVED, REF_AMBIGUOUS = "resolved", "ambiguous"
REF_STATUSES = (REF_RESOLVED, REF_AMBIGUOUS)
# How an entity name is established: its canonical name, an explicit alias, or one of several
# conflicting names of an ambiguous entity.
NAME_CANONICAL, NAME_ALIAS, NAME_CANDIDATE = "canonical", "alias", "candidate"
NAME_KINDS = (NAME_CANONICAL, NAME_ALIAS, NAME_CANDIDATE)

SPEAKER_TYPES = {"user": PERSON}
FRAME = re.compile(r"\bthe (user|assistant)\b", re.IGNORECASE)


def entity_id(conversation_id: str, local_key: str) -> str:
    """Deterministic, conversation-scoped entity ID; the encoded conversation ID contains no "/"."""
    return f"ent:{quote(conversation_id, safe='')}/{local_key}"


# ---- model ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityName:
    """A name of an entity and the source that established it (no message text is copied)."""

    name: str
    kind: str
    message_id: str
    claim_id: str | None = None  # None when a message, not a claim, established it (self-naming)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "message_id": self.message_id, "claim_id": self.claim_id}


@dataclass(frozen=True)
class Entity:
    entity_id: str
    conversation_id: str
    local_key: str  # "role:user", "name:<phase 2C key>" or "id:<explicit id>"
    canonical_name: str | None  # None for an unnamed or ambiguous conversational actor
    entity_type: str
    status: str
    speaker_role: str | None  # the conversational role this entity is, if any
    established_by_claim_id: str  # the first claim (conversation order) that references the entity
    established_by_message_id: str
    names: tuple[EntityName, ...] = ()

    @property
    def label(self) -> str:
        return self.canonical_name or (f"the {self.speaker_role}" if self.speaker_role else self.local_key)

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(n.name for n in self.names if n.kind == NAME_ALIAS))

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "conversation_id": self.conversation_id, "local_key": self.local_key,
                "canonical_name": self.canonical_name, "entity_type": self.entity_type, "status": self.status,
                "speaker_role": self.speaker_role, "established_by_claim_id": self.established_by_claim_id,
                "established_by_message_id": self.established_by_message_id,
                "names": [n.to_dict() for n in self.names]}


@dataclass(frozen=True)
class EntityRef:
    """One memory -> entity reference, with the source claim and message that carry it."""

    role: str
    claim_id: str
    message_id: str
    name: str  # as written in the claim (or the role, for speaker references)
    entity_id: str | None  # None only for an ambiguous reference
    status: str = REF_RESOLVED
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "entity_id": self.entity_id, "name": self.name, "claim_id": self.claim_id,
                "message_id": self.message_id, "status": self.status, "detail": self.detail}


# ---- typing ---------------------------------------------------------------------------------


class EntityTyper(Protocol):
    def type_of(self, names: Sequence[str], texts: Sequence[str]) -> str: ...


_PERSON_NOUNS = ("sister|brother|mother|father|mom|mum|dad|wife|husband|partner|girlfriend|boyfriend|son|daughter|"
                 "friend|colleague|coworker|boss|manager|cousin|aunt|uncle|niece|nephew|grandmother|grandfather|"
                 "neighbou?r|roommate|lead|designer|engineer|developer|teacher|doctor")
_PET_NOUNS = "dog|cat|puppy|kitten|pet|parrot|hamster|rabbit|horse"
_DEVICE_NOUNS = "laptop|phone|car|bike|computer|camera|tablet|watch"


def _cues(n: str) -> dict[str, tuple[str, ...]]:
    return {
        PERSON: (rf"\b{n}\s+is\s+(?:the\s+user's|\w+'s|our|my|the|a|an)\s+(?:[\w-]+\s+)?(?i:{_PERSON_NOUNS})\b",
                 rf"\b(?i:{_PERSON_NOUNS})\s+{n}\b"),
        PET: (rf"\b(?i:{_PET_NOUNS})\s+(?:(?i:is|was)\s+)?(?:(?i:called|named)\s+)?{n}\b",
              rf"\b{n}\s+is\s+(?:a|an|the\s+user's|\w+'s|my)\s+(?:[\w-]+\s+){{0,3}}(?i:{_PET_NOUNS})\b"),
        ORGANIZATION: (rf"\b(?i:works?|working|worked|employed|job|role|position)\s+(?i:at|for|by|with)\s+{n}\b",
                       rf"\b(?i:works?|working|worked)\s+as\s+(?:a|an)\s+[\w\s-]+?\s+(?i:at|for)\s+{n}\b",
                       rf"\b(?i:joined|joining)\s+{n}\b",
                       rf"\b{n},?\s+(?:Inc|Ltd|LLC|GmbH|Corp|Corporation|Company)\b"),
        PROJECT: (rf"\b(?i:project)\s+{n}\b", rf"\b{n}\s+(?i:project)\b"),
        PLACE: (rf"\b(?i:lives?|living|lived|moved|moving|relocated|relocating|based|visit|visiting|visited|flight|"
                rf"flying|travel\w*|trip|office|stay\w*)\s+(?:(?i:in|to|at)\s+)?{n}\b",
                rf"\b(?i:visiting|flying in|flying)\s+(?i:from)\s+{n}\b"),
        PRODUCT: (rf"\b(?i:bought|buying|buy|purchased|ordered|owns|uses|using)\s+(?:(?i:a|an|the|their|his|her|my)\s+)?"
                  rf"(?:(?i:new)\s+)?{n}\b",
                  rf"\b(?i:{_DEVICE_NOUNS})\s+is\s+(?:a|an)\s+{n}\b"),
    }


class CueEntityTyper:
    """Type from explicit wording around the entity's names; no cue or conflicting cues -> "other"."""

    def type_of(self, names: Sequence[str], texts: Sequence[str]) -> str:
        found = set()
        for name in dict.fromkeys(names):
            if name.split()[0] == "Project":
                found.add(PROJECT)
            n = rf"(?<![\w-]){re.escape(name)}(?![\w-])"
            for etype, patterns in _cues(n).items():
                if any(re.search(p, t) for p in patterns for t in texts):
                    found.add(etype)
        return found.pop() if len(found) == 1 else OTHER


# ---- resolution -----------------------------------------------------------------------------


def claim_possessor(claim: Claim) -> str | None:
    """The identity key the claim opens with in the possessive ("The user's ...", "Leah's ...")."""
    text = normalize_value(claim.content)
    for key in sorted({*SPEAKERS, *entity_keys(claim)}, key=lambda k: (-len(k), k)):
        for prefix in (key, f"the {key}"):
            if text.startswith(prefix + " s "):
                return key
    return None


def _written(raw: str, texts: Sequence[str]) -> bool:
    """``raw`` is written as a name: it has a capitalized word and occurs, as written, in a text."""
    if not re.search(r"(?<![\w'])[A-Z]", raw):
        return False
    return any(re.search(rf"(?<![\w-]){re.escape(raw)}(?![\w-])", t) for t in texts)


@dataclass
class EntityResolution:
    conversation_id: str
    entities: dict[str, Entity] = field(default_factory=dict)  # entity_id -> entity, in first-reference order
    refs: dict[str, tuple[EntityRef, ...]] = field(default_factory=dict)  # memory_id -> refs, in memory order

    def memories_of(self, eid: str, role: str | None = None) -> list[str]:
        return [mid for mid, refs in self.refs.items()
                if any(r.entity_id == eid and (role is None or r.role == role) for r in refs)]

    def attach(self, memories: Iterable[ContextualMemory]) -> list[ContextualMemory]:
        return [replace(m, entity_refs=self.refs.get(m.memory.memory_id, ())) for m in memories]

    def to_dict(self) -> dict[str, Any]:
        return {"conversation_id": self.conversation_id, "entities": [e.to_dict() for e in self.entities.values()],
                "memory_entities": [{"memory_id": mid, "refs": [r.to_dict() for r in refs]}
                                    for mid, refs in self.refs.items()]}


class _Candidate:
    """Per local key: the written names, claims and texts that reference it, in order."""

    def __init__(self, claim: Claim, message: Message) -> None:
        self.first = (claim.claim_id, message.message_id)
        self.written: dict[str, tuple[str, str]] = {}  # entity key of a written form -> (form, claim_id)
        self.texts: list[str] = []


def resolve_entities(conversation_id: str, messages: Iterable[Message], claims: Iterable[Claim],
                     contexts: Mapping[str, ClaimContext], memories: Iterable[Memory], sources: Iterable[MemorySource],
                     typer: EntityTyper | None = None) -> EntityResolution:
    """Entities and memory references for one conversation; see the module docstring."""
    typer = typer or CueEntityTyper()
    cid = conversation_id
    messages = sorted(messages, key=lambda m: m.position)
    by_message = {m.message_id: m for m in messages}
    memories = sorted((m for m in memories if m.conversation_id == cid),
                      key=lambda m: (m.position, _ordinal(m.claim_id), m.memory_id))
    if not memories:
        return EntityResolution(cid)
    last = messages[-1].position
    registry = SpeakerRegistry(messages)
    speakers = {role: registry.speaker(role, last) for role in dict.fromkeys(m.role for m in messages)}

    name_roles: dict[str, set[str]] = {}  # name key -> speaker roles it is the established name or alias of
    for role, sp in speakers.items():
        if sp.status == SPEAKER_NAMED:
            for name in (sp.name, *sp.alias_names):
                name_roles.setdefault(entity_key(name), set()).add(role)

    def local_key(identity: str) -> tuple[str | None, str | None]:
        if identity in SPEAKERS:
            return f"role:{identity}", None
        roles = name_roles.get(identity, set())
        if len(roles) > 1:
            return None, f"{IDENTITY_AMBIGUOUS}: {identity!r} is the name of several speakers {sorted(roles)}"
        if roles:
            return f"role:{next(iter(roles))}", None
        return (identity if identity.startswith("id:") else f"name:{identity}"), None

    # Phase 2C identity, replayed in consolidation order.
    identities = IdentityRegistry()
    per_claim: dict[str, list[tuple[str, str, str | None, str | None]]] = {}  # (role, written, key, problem)
    candidates: dict[str, _Candidate] = {}
    declared_aliases: list[tuple[str, str, Claim]] = []  # (alias as written, canonical key, claim)
    claims = sorted({c.claim_id: c for c in claims if c.provenance.conversation_id == cid}.values(), key=claim_order)
    for claim in claims:
        ident = identities.identify(claim)
        source = by_message[claim.provenance.message_id]
        texts = (claim.content, source.content)
        resolved = [(raw, *identities.resolve(raw, claim)) for raw in claim.entities]
        refs: list[tuple[str, str, str | None, str | None]] = [
            (ROLE_SPEAKER, claim.provenance.role, f"role:{claim.provenance.role}", None)]

        def add(role: str, raw: str, identity: str | None, problem: str | None) -> None:
            key, conflict = local_key(identity) if identity else (None, problem)
            if key is None:
                if conflict and not any(k is None and entity_key(r) == entity_key(raw) for _, r, k, _ in refs):
                    refs.append((role, raw, None, conflict))
                return
            forms = [r for r, i, _ in resolved if i == identity] + [raw]
            if key.startswith("name:") and not any(_written(f, texts) for f in forms):
                return  # not written as a name: not an entity
            if any(k == key and r != ROLE_SPEAKER for r, _, k, _ in refs):
                return
            refs.append((role, raw, key, None))

        def written_form(key: str | None) -> str:
            if key is None:
                return ""
            return next((r for r in claim.entities if entity_key(r) == key), f"the {key}" if key in SPEAKERS else key)

        explicit = claim.metadata.get("subject_id") or claim.metadata.get("subject")
        positional = None if explicit else claim_subject(claim)
        if explicit:
            raw = str(next((r for r, i, _ in resolved if ident.subject and i == ident.subject), None)
                      or claim.metadata.get("subject") or explicit)
        else:
            raw = written_form(positional)
        if ident.subject:
            add(ROLE_SUBJECT, raw, ident.subject, None)
        elif explicit or positional:
            add(ROLE_SUBJECT, raw, *identities.resolve(raw, claim))
        possessor = claim_possessor(claim)
        if possessor and possessor != positional:
            add(ROLE_POSSESSOR, written_form(possessor), *identities.resolve(possessor, claim))
        for raw, identity, problem in resolved:
            add(ROLE_MENTIONED, raw, identity, problem)
        for match in FRAME.finditer(claim.content):
            role = match.group(1).lower()
            if not re.search(rf"\b{role}s?\b", source.content, re.IGNORECASE):
                add(ROLE_MENTIONED, match.group(0), role, None)
        per_claim[claim.claim_id] = refs
        for alias, canonical in _mapping(claim.metadata.get("aliases")).items():
            declared_aliases.append((next((r for r in claim.entities if entity_key(r) == alias), alias),
                                     entity_key(canonical), claim))

    # Memory references, in memory order and source-claim order (creating claim first).
    by_memory: dict[str, list[MemorySource]] = {}
    for s in sources:
        by_memory.setdefault(s.memory_id, []).append(s)
    claim_by_id = {c.claim_id: c for c in claims}
    memory_refs: dict[str, list[tuple[str, str, str | None, str | None, Claim]]] = {}
    for m in memories:
        own = sorted(by_memory.get(m.memory_id, ()), key=lambda s: (s.claim_id != m.claim_id, s.position, s.claim_id))
        rows = memory_refs.setdefault(m.memory_id, [])
        for s in own:
            claim = claim_by_id[s.claim_id]
            rows += [(*ref, claim) for ref in per_claim.get(s.claim_id, ())]

    # Candidates in first-reference order (conversation order of claims that are memory sources).
    memory_claims = {row[4].claim_id for rows in memory_refs.values() for row in rows}
    for claim in claims:
        if claim.claim_id not in memory_claims:
            continue
        for role, raw, key, _ in per_claim.get(claim.claim_id, ()):
            if key is None:
                continue
            cand = candidates.get(key) or candidates.setdefault(key, _Candidate(claim, by_message[claim.provenance.message_id]))
            if role != ROLE_SPEAKER and key.startswith(("name:", "id:")):
                cand.written.setdefault(entity_key(raw), (raw, claim.claim_id))
            ctx = contexts.get(claim.claim_id)
            for text in (claim.content, ctx.contextual_content if ctx else None):
                if text and text not in cand.texts:
                    cand.texts.append(text)

    result = EntityResolution(cid)
    for key, cand in candidates.items():
        eid = entity_id(cid, key)
        claim_id, message_id = cand.first
        if key.startswith("role:"):
            role = key.split(":", 1)[1]
            sp = speakers.get(role)
            names: list[EntityName] = []
            if sp and sp.status == SPEAKER_NAMED:
                names += [EntityName(sp.name, NAME_CANONICAL, mid) for mid in sp.evidence]
                names += [EntityName(a, NAME_ALIAS, mid) for a, mid in sp.aliases]
            elif sp and sp.status == SPEAKER_AMBIGUOUS:
                names += [EntityName(n, NAME_CANDIDATE, mid) for n, mid, alias in registry.declarations(role, last)
                          if not alias]
            status = sp.status if sp else SPEAKER_UNNAMED
            entity = Entity(eid, cid, key, sp.name if sp and sp.status == SPEAKER_NAMED else None,
                            SPEAKER_TYPES.get(role, OTHER), status, role, claim_id, message_id, tuple(names))
        else:
            forms = list(cand.written.values())
            canonical, first_claim = forms[0]
            names = [EntityName(canonical, NAME_CANONICAL, claim_by_id[first_claim].provenance.message_id, first_claim)]
            if key.startswith("id:"):  # other names explicitly given the same ID
                names += [EntityName(f, NAME_ALIAS, claim_by_id[c].provenance.message_id, c) for f, c in forms[1:]]
            entity = Entity(eid, cid, key, canonical, typer.type_of([f for f, _ in forms], cand.texts),
                            SPEAKER_NAMED, None, claim_id, message_id, tuple(names))
        result.entities[eid] = entity

    for alias, canonical_key, claim in declared_aliases:
        key, _ = local_key(canonical_key)
        eid = entity_id(cid, key) if key else None
        if eid in result.entities:
            e = result.entities[eid]
            name = EntityName(alias, NAME_ALIAS, claim.provenance.message_id, claim.claim_id)
            if all(n.name != alias for n in e.names):
                result.entities[eid] = replace(e, names=(*e.names, name))

    for mid, rows in memory_refs.items():
        refs = []
        for role, raw, key, problem, claim in rows:
            eid = entity_id(cid, key) if key else None
            if eid is not None and eid not in result.entities:
                continue
            refs.append(EntityRef(role, claim.claim_id, claim.provenance.message_id, raw, eid,
                                  REF_RESOLVED if eid else REF_AMBIGUOUS, problem))
        result.refs[mid] = tuple(refs)
    return result


# ---- provenance -----------------------------------------------------------------------------


def entity_evidence(entity: Entity, contexts: Mapping[str, ClaimContext], messages: MessageLookup) -> list[SourceContext]:
    """The source windows behind an entity: the claim that first referenced it and every name's
    source (the claim's window, or the naming message itself), deduplicated, in that order."""
    windows: list[SourceWindow] = [contexts[entity.established_by_claim_id].window]
    for n in entity.names:
        if n.claim_id:
            windows.append(contexts[n.claim_id].window)
        else:
            found = messages.get_messages(entity.conversation_id, [n.message_id])
            if not found:
                raise SourceContextError(f"{entity.entity_id}: naming message {n.message_id} is not stored")
            windows.append(SourceWindow(entity.conversation_id, found[0].position, found[0].position, (n.message_id,)))
    return [resolve_window(w, messages) for w in dict.fromkeys(windows)]
