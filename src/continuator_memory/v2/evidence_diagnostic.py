"""Evidence representation and context assembly diagnostic (phase 5F, analysis only).

Compares what the original messages say with what the stored V2 representation of them says,
and describes how retrieved evidence items relate to each other. Nothing here changes, persists
or adds a memory, entity, relationship, ranking or prompt; every function reads the stored
representation of one conversation (``ConversationView``) and never a question, an answer or a
gold label.

Comparison unit: one source message against a set of memories derived from it, either all of
them (the representation) or the retrieved ones (a context). Only the message's claim-bearing
sentences are compared: sentences sharing a content word with a claim extracted from that
message (conversational filler that produced no claim is not counted as lost). Each information
type gets one state:

  speaker      preserved_named   a memory names the asserting speaker (Phase 5A speaker name)
               generic_role      a memory says "the user" / "the assistant" (the role only)
               resolved_differently  a memory attributes a first-person claim to the other role
               missing           first-person (or assistant-asserted) source, no attribution
               ambiguous         Phase 5A marks the asserting speaker ambiguous
               not_applicable    third-person source, nothing to attribute
  entity       named Phase 5B entities (not conversational roles) written in the source
               sentences vs referenced by / written in the memories
  temporal     date expressions and past / present / future cues (the ``temporal`` module's
               patterns, unchanged) vs memory content, temporal expression, event time,
               validity bounds, status and time class
  polarity     negation, change, conditional and correction cues vs memory polarity,
               content, status, time class and consolidation / relationship links
  provenance   conversation, source messages, source windows and claims all resolve

States other than speaker: preserved (every source marker is represented), lost (a marker is
represented nowhere), ambiguous (lost, but the representation holds an ambiguous reference or
time class), not_applicable (no marker in the source).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from continuator_memory.memory.text import STOPWORDS
from continuator_memory.models import Message
from continuator_memory.v2.context import (
    SPEAKER_AMBIGUOUS,
    ContextualMemory,
    SourceContextError,
    SpeakerRegistry,
    global_memory_id,
    resolve_window,
)
from continuator_memory.v2.entities import REF_AMBIGUOUS, ROLE_SPEAKER
from continuator_memory.v2.expansion import ConversationKnowledge
from continuator_memory.v2.memory import (
    LINK_CONTRADICTS,
    LINK_SUPERSEDES,
    LINK_UPDATES,
    STATUS_CONDITIONAL,
    STATUS_CONTESTED,
    STATUS_CURRENT,
    STATUS_FUTURE,
    STATUS_HISTORICAL,
    STATUS_SUPERSEDED,
    MemorySource,
)
from continuator_memory.v2.models import Claim
from continuator_memory.v2.temporal import (
    CONDITIONAL,
    FUTURE,
    PAST_ANCHOR,
    PAST_STATE,
    PAST_STATIVE,
    PRESENT,
    TIME_AMBIGUOUS_PAST,
    TIME_CONDITIONAL,
    TIME_FUTURE,
    TIME_MIXED,
    TIME_PAST,
    TIME_TRANSITION,
    TRANSITION,
)

ORIGIN_RAW, ORIGIN_MEMORY, ORIGIN_HYBRID = "raw", "memory", "hybrid"

PRESERVED, LOST, AMBIGUOUS, NOT_APPLICABLE = "preserved", "lost", "ambiguous", "not_applicable"
STATES = (PRESERVED, LOST, AMBIGUOUS, NOT_APPLICABLE)

SPEAKER_NAMED = "preserved_named"
SPEAKER_ROLE = "generic_role"
SPEAKER_DIFFERENT = "resolved_differently"
SPEAKER_MISSING = "missing"
SPEAKER_AMBIGUOUS_STATE = "ambiguous"
SPEAKER_NA = "not_applicable"
SPEAKER_STATES = (SPEAKER_NAMED, SPEAKER_ROLE, SPEAKER_DIFFERENT, SPEAKER_MISSING, SPEAKER_AMBIGUOUS_STATE, SPEAKER_NA)
# the best attribution among a message's memories decides the message's speaker state
_SPEAKER_RANK = {s: i for i, s in enumerate((SPEAKER_NAMED, SPEAKER_ROLE, SPEAKER_DIFFERENT, SPEAKER_MISSING,
                                             SPEAKER_NA))}

SPEAKER_LOSS = "SPEAKER_LOSS"
ENTITY_LOSS = "ENTITY_LOSS"
TEMPORAL_LOSS = "TEMPORAL_LOSS"
POLARITY_LOSS = "POLARITY_LOSS"
PROVENANCE_LOSS = "PROVENANCE_LOSS"
CROSS_MEMORY_LINK_LOSS = "CROSS_MEMORY_LINK_LOSS"
NONE = "NONE"
LOSS_FLAGS = (SPEAKER_LOSS, ENTITY_LOSS, TEMPORAL_LOSS, POLARITY_LOSS, PROVENANCE_LOSS, CROSS_MEMORY_LINK_LOSS, NONE)

LINK_SAME_SOURCE = "same_source"
LINK_EDGE = "relationship_edge"
LINK_SAME_ENTITY = "same_entity"
LINK_WINDOW = "window_overlap"
LINK_KINDS = (LINK_SAME_SOURCE, LINK_EDGE, LINK_SAME_ENTITY, LINK_WINDOW)

SINGLE_MEMORY_EVIDENCE = "SINGLE_MEMORY_EVIDENCE"
MULTI_MEMORY_SAME_SOURCE = "MULTI_MEMORY_SAME_SOURCE"
MULTI_MEMORY_SAME_ENTITY = "MULTI_MEMORY_SAME_ENTITY"
MULTI_MEMORY_RELATED = "MULTI_MEMORY_RELATED"
MULTI_MEMORY_DISCONNECTED = "MULTI_MEMORY_DISCONNECTED"
GROUP_CATEGORIES = (SINGLE_MEMORY_EVIDENCE, MULTI_MEMORY_SAME_SOURCE, MULTI_MEMORY_SAME_ENTITY, MULTI_MEMORY_RELATED,
                    MULTI_MEMORY_DISCONNECTED)

FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b")
NEGATION = re.compile(r"\b(not|no|never|none|nothing|nobody|neither|nor|without|cannot)\b|n't\b")
CORRECTION = re.compile(r"\b(actually|correction|i meant|scratch that|changed my mind|my mistake|wrong)\b")
_MONTH = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t|tember)?|"
          r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
_WEEKDAY = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
DATE_EXPRESSION = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+of\s+(?:{_MONTH})|(?:{_MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH})|(?:{_MONTH})\s+(?:19|20)\d{{2}}|(?:19|20)\d{{2}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}|{_WEEKDAY}|\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm)|\d{{1,2}}:\d{{2}}"
    rf"|today|tomorrow|tonight|yesterday|(?:next|last|this)\s+(?:week|month|year|weekend|{_WEEKDAY}))\b")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def normalize(text: str) -> str:
    """Lowercased cue text, without capitalized "May"/"Will" inside a sentence (as ``temporal.cue_text``)."""
    text = re.sub(r"(?<=[^\s.!?] )(May|Will)\b", " ", text.replace("\u2019", "'"))
    return " ".join(text.lower().split())


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(normalize(text)) if len(w) >= 3 and w not in STOPWORDS}


def mentions(text: str, name: str) -> bool:
    """``name`` written as a whole word (a possessive "'s" counts)."""
    return re.search(rf"(?<![\w'-]){re.escape(name.lower())}(?:'s)?(?![\w'-])", normalize(text)) is not None


def _frame(role: str) -> re.Pattern[str]:
    return re.compile(rf"\bthe {role}('s)?\b")


def temporal_markers(text: str) -> set[str]:
    t = normalize(text)
    out = {f"date:{m.group(0)}" for m in DATE_EXPRESSION.finditer(t)}
    if PAST_STATE.search(t) or PAST_ANCHOR.search(t) or PAST_STATIVE.search(t):
        out.add("past")
    if PRESENT.search(t):
        out.add("present")
    if FUTURE.search(t):
        out.add("future")
    return out


def state_markers(text: str) -> set[str]:
    t = normalize(text)
    found = {"negation": NEGATION, "change": TRANSITION, "conditional": CONDITIONAL, "correction": CORRECTION}
    return {k for k, p in found.items() if p.search(t)}


# ---- the stored representation of one conversation ---------------------------------------


@dataclass
class ConversationView:
    knowledge: ConversationKnowledge
    claims: dict[str, Claim]
    sources: dict[str, tuple[MemorySource, ...]]  # memory_id -> memory sources (MemoryStore)
    registry: SpeakerRegistry = field(init=False)
    messages: dict[str, Message] = field(init=False)
    names: dict[str, tuple[str, ...]] = field(init=False)  # named (non-role) entity -> names
    claims_of: dict[str, tuple[str, ...]] = field(init=False)  # message -> claim ids
    memories_of: dict[str, tuple[str, ...]] = field(init=False)  # message -> memory ids, memory order

    def __post_init__(self) -> None:
        self.registry = SpeakerRegistry(self.knowledge.messages)
        self.messages = {m.message_id: m for m in self.knowledge.messages}
        self.names = {}
        for eid, e in self.knowledge.entities.items():
            if e.local_key.startswith("role:"):
                continue
            names = dict.fromkeys(n for n in (e.canonical_name, *(x.name for x in e.names)) if n and len(n) >= 2)
            if names:
                self.names[eid] = tuple(names)
        claims_of: dict[str, list[str]] = {}
        for c in sorted(self.claims.values(), key=lambda c: (c.provenance.position, c.claim_id)):
            claims_of.setdefault(c.provenance.message_id, []).append(c.claim_id)
        self.claims_of = {m: tuple(cs) for m, cs in claims_of.items()}
        memories_of: dict[str, list[str]] = {}
        for mid in self.knowledge.memories:
            for message_id in self.source_messages(mid):
                memories_of.setdefault(message_id, []).append(mid)
        self.memories_of = {m: tuple(ms) for m, ms in memories_of.items()}

    @property
    def conversation_id(self) -> str:
        return self.knowledge.conversation_id

    def memory(self, memory_id: str) -> ContextualMemory:
        return self.knowledge.memories[memory_id]

    def source_messages(self, memory_id: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.message_id for s in self.sources.get(memory_id, ())))

    def windows(self, memory_id: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.window.window_id for s in self.memory(memory_id).sources))

    def window_messages(self, memory_id: str) -> set[str]:
        return {m for s in self.memory(memory_id).sources for m in s.window.message_ids}

    def message_window(self, message_id: str) -> str | None:
        """The Phase 5A source window ending at the message (None when it produced no claim)."""
        for cm in self.knowledge.memories.values():
            for s in cm.sources:
                if s.message_id == message_id:
                    return s.window.window_id
        return None

    def named_entities(self, memory_id: str) -> tuple[str, ...]:
        cm = self.memory(memory_id)
        refs = [r.entity_id for r in cm.entity_refs if r.entity_id in self.names and r.role != ROLE_SPEAKER]
        written = [eid for eid, names in self.names.items() if any(mentions(cm.memory.content, n) for n in names)]
        return tuple(dict.fromkeys(refs + sorted(written)))

    def written_entities(self, text: str) -> set[str]:
        return {eid for eid, names in self.names.items() if any(mentions(text, n) for n in names)}

    def claim_text(self, message_id: str) -> str:
        """The claim-bearing sentences of a message."""
        message = self.messages[message_id]
        claim_words = set().union(*(content_words(self.claims[c].content) for c in self.claims_of.get(message_id, ())))
        sentences = [s for s in _SENTENCE.split(message.content.strip()) if content_words(s) & claim_words]
        return " ".join(sentences)

    def edges(self) -> list[tuple[str, str, str]]:
        g = self.knowledge.graph
        return [(r.source_memory_id, r.target_memory_id, r.type) for r in [*g.links, *g.additive]]


def load_view(connection, conversation_id: str, knowledge: ConversationKnowledge) -> ConversationView:
    from continuator_memory.v2.memory_store import MemoryStore
    from continuator_memory.v2.store import SemanticStore

    sources: dict[str, list[MemorySource]] = {}
    for s in MemoryStore(connection).get_sources(conversation_id):
        sources.setdefault(s.memory_id, []).append(s)
    claims = {c.claim_id: c for c in SemanticStore(connection).get_claims(conversation_id)}
    return ConversationView(knowledge, claims, {m: tuple(sorted(ss, key=lambda s: (s.position, s.claim_id)))
                                                for m, ss in sources.items()})


# ---- comparisons --------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    state: str
    source: tuple[str, ...] = ()  # markers / entities in the source sentences
    represented: tuple[str, ...] = ()  # of those, represented by the memories
    lost: tuple[str, ...] = ()
    introduced: tuple[str, ...] = ()  # in the memories, not written in the source sentences
    ambiguous: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "source": list(self.source), "represented": list(self.represented),
                "lost": list(self.lost), "introduced": list(self.introduced), "ambiguous": list(self.ambiguous)}


def _compare(source: set[str], represented: set[str], ambiguous: Iterable[str] = (),
             introduced: Iterable[str] = ()) -> Comparison:
    lost = source - represented
    ambiguous = tuple(sorted(set(ambiguous)))
    state = NOT_APPLICABLE if not source else PRESERVED if not lost else AMBIGUOUS if ambiguous else LOST
    return Comparison(state, tuple(sorted(source)), tuple(sorted(source & represented)), tuple(sorted(lost)),
                      tuple(sorted(set(introduced))), ambiguous)


def speaker_attribution(view: ConversationView, memory_id: str, text: str) -> str:
    """How ``text`` (a representation of the memory) attributes the memory's asserting speaker."""
    cm = view.memory(memory_id)
    speaker = cm.speaker
    if speaker.status == SPEAKER_AMBIGUOUS:
        return SPEAKER_AMBIGUOUS_STATE
    if speaker.name and any(mentions(text, n) for n in (speaker.name, *speaker.alias_names)):
        return SPEAKER_NAMED
    t = normalize(text)
    if _frame(speaker.role).search(t):
        return SPEAKER_ROLE
    first_person = bool(FIRST_PERSON.search(normalize(view.claim_text(cm.context.message_id))))
    other_role = "assistant" if speaker.role == "user" else "user"
    other = view.registry.speaker(other_role, cm.context.position)
    if first_person and (_frame(other_role).search(t) or (other.name and mentions(text, other.name))):
        return SPEAKER_DIFFERENT
    return SPEAKER_MISSING if first_person or speaker.role != "user" else SPEAKER_NA


def compare_speaker(view: ConversationView, memory_ids: Sequence[str], contextual: bool = False) -> str:
    states = [speaker_attribution(view, m, view.memory(m).contextual_content if contextual
                                  else view.memory(m).memory.content) for m in memory_ids]
    if not states:
        return SPEAKER_NA
    if SPEAKER_AMBIGUOUS_STATE in states:
        return SPEAKER_AMBIGUOUS_STATE
    return min(states, key=_SPEAKER_RANK.__getitem__)


def compare_entities(view: ConversationView, message_id: str, memory_ids: Sequence[str]) -> Comparison:
    source = view.written_entities(view.claim_text(message_id))
    represented = set().union(*(view.named_entities(m) for m in memory_ids)) if memory_ids else set()
    ambiguous = [r.name for m in memory_ids for r in view.memory(m).entity_refs if r.status == REF_AMBIGUOUS]
    return _compare(source, represented, ambiguous, represented - source)


def _memory_time(view: ConversationView, memory_id: str) -> tuple[set[str], set[str]]:
    m = view.memory(memory_id).memory
    text = " ".join(x for x in (m.content, m.temporal, m.event_time, m.valid_from, m.valid_until) if x)
    found = temporal_markers(text)
    if m.status in (STATUS_HISTORICAL, STATUS_SUPERSEDED) or m.time_class == TIME_PAST:
        found.add("past")
    if m.status == STATUS_CURRENT:
        found.add("present")
    if m.status == STATUS_FUTURE or m.time_class == TIME_FUTURE:
        found.add("future")
    ambiguous = {m.time_class} if m.time_class in (TIME_AMBIGUOUS_PAST, TIME_MIXED) else set()
    return found, ambiguous


def _dates_represented(source: set[str], represented_text: str, found: set[str]) -> set[str]:
    """A source date is represented when every word of it occurs in the memory text."""
    words = set(_WORD.findall(normalize(represented_text)))
    return {d for d in source if d.startswith("date:") and set(_WORD.findall(d[5:])) <= words} | (found & source)


def compare_temporal(view: ConversationView, message_id: str, memory_ids: Sequence[str]) -> Comparison:
    source = temporal_markers(view.claim_text(message_id))
    found: set[str] = set()
    ambiguous: set[str] = set()
    texts = []
    for mid in memory_ids:
        f, a = _memory_time(view, mid)
        found |= f
        ambiguous |= a
        m = view.memory(mid).memory
        texts += [x for x in (m.content, m.temporal, m.event_time, m.valid_from, m.valid_until) if x]
    represented = _dates_represented(source, " ".join(texts), found)
    return _compare(source, represented, ambiguous)


def _memory_state(view: ConversationView, memory_id: str) -> set[str]:
    m = view.memory(memory_id).memory
    found = state_markers(m.content)
    if m.polarity == "negative":
        found.add("negation")
    linked = {(a, t) for a, b, t in view.edges() if memory_id in (a, b)}
    newer = {t for a, _, t in view.edges() if a == memory_id}
    if m.time_class == TIME_TRANSITION or m.status == STATUS_SUPERSEDED or any(
            t in (LINK_UPDATES, LINK_SUPERSEDES) for _, t in linked):
        found.add("change")
    if m.status == STATUS_CONDITIONAL or m.time_class == TIME_CONDITIONAL:
        found.add("conditional")
    if m.status == STATUS_CONTESTED or newer & {LINK_UPDATES, LINK_CONTRADICTS, LINK_SUPERSEDES}:
        found.add("correction")
    return found


def compare_polarity(view: ConversationView, message_id: str, memory_ids: Sequence[str]) -> Comparison:
    source = state_markers(view.claim_text(message_id))
    represented = set().union(*(_memory_state(view, m) for m in memory_ids)) if memory_ids else set()
    return _compare(source, represented)


def provenance(view: ConversationView, memory_id: str) -> dict[str, bool]:
    cm = view.memory(memory_id)
    sources = view.sources.get(memory_id, ())
    messages = [s.message_id for s in sources]
    lookup = _Lookup(view)
    try:
        windows = all(resolve_window(c.window, lookup).window.source_message_id == c.message_id for c in cm.sources)
    except SourceContextError:
        windows = False
    out = {"conversation": cm.memory.conversation_id == view.conversation_id
           and all(s.conversation_id == view.conversation_id for s in cm.sources),
           "source_message": bool(messages) and cm.memory.message_id in messages
           and all(m in view.messages for m in messages),
           "source_window": bool(cm.sources) and windows,
           "claim": bool(sources) and all(s.claim_id in view.claims for s in sources)
           and {s.claim_id for s in sources} == {c.claim_id for c in cm.sources}}
    out["complete"] = all(out.values())
    return out


class _Lookup:
    def __init__(self, view: ConversationView):
        self._view = view

    def get_messages(self, conversation_id: str, message_ids) -> list[Message]:
        if conversation_id != self._view.conversation_id:
            return []
        return [self._view.messages[m] for m in message_ids if m in self._view.messages]


@dataclass(frozen=True)
class MessageComparison:
    """One source message against a set of memories derived from it."""

    conversation_id: str
    message_id: str
    role: str
    memory_ids: tuple[str, ...]
    speaker: str
    contextual_speaker: str
    entity: Comparison
    temporal: Comparison
    polarity: Comparison
    provenance_complete: bool

    def loss_flags(self) -> list[str]:
        flags = []
        if self.speaker in (SPEAKER_MISSING, SPEAKER_DIFFERENT):
            flags.append(SPEAKER_LOSS)
        if self.entity.state == LOST:
            flags.append(ENTITY_LOSS)
        if self.temporal.state == LOST:
            flags.append(TEMPORAL_LOSS)
        if self.polarity.state == LOST:
            flags.append(POLARITY_LOSS)
        if not self.provenance_complete:
            flags.append(PROVENANCE_LOSS)
        return flags

    def ambiguities(self) -> list[str]:
        out = ["speaker"] if self.speaker == SPEAKER_AMBIGUOUS_STATE else []
        return out + [k for k, c in (("entity", self.entity), ("temporal", self.temporal)) if c.state == AMBIGUOUS]

    def to_dict(self) -> dict[str, Any]:
        return {"conversation_id": self.conversation_id, "message_id": self.message_id, "role": self.role,
                "memory_ids": list(self.memory_ids), "speaker": self.speaker,
                "contextual_speaker": self.contextual_speaker, "entity": self.entity.to_dict(),
                "temporal": self.temporal.to_dict(), "polarity": self.polarity.to_dict(),
                "provenance_complete": self.provenance_complete, "loss_flags": self.loss_flags()}


def compare_message(view: ConversationView, message_id: str, memory_ids: Sequence[str] | None = None) -> MessageComparison:
    """``memory_ids`` defaults to every memory derived from the message (the representation)."""
    own = view.memories_of.get(message_id, ())
    chosen = tuple(m for m in own if memory_ids is None or m in set(memory_ids))
    return MessageComparison(
        view.conversation_id, message_id, view.messages[message_id].role, chosen,
        compare_speaker(view, chosen), compare_speaker(view, chosen, contextual=True),
        compare_entities(view, message_id, chosen), compare_temporal(view, message_id, chosen),
        compare_polarity(view, message_id, chosen), all(provenance(view, m)["complete"] for m in chosen))


# ---- links between memories and fragmentation ----------------------------------------------


Ref = tuple[str, str]  # (conversation_id, memory_id)


def memory_links(views: Mapping[str, ConversationView], a: Ref, b: Ref) -> tuple[str, ...]:
    """Stored links between two memories (none across conversations)."""
    if a[0] != b[0] or a == b:
        return ()
    view = views[a[0]]
    out = []
    if set(view.source_messages(a[1])) & set(view.source_messages(b[1])):
        out.append(LINK_SAME_SOURCE)
    if view.knowledge.graph.related(a[1], b[1]):
        out.append(LINK_EDGE)
    if set(view.named_entities(a[1])) & set(view.named_entities(b[1])):
        out.append(LINK_SAME_ENTITY)
    if view.window_messages(a[1]) & view.window_messages(b[1]):
        out.append(LINK_WINDOW)
    return tuple(out)


def components(views: Mapping[str, ConversationView], refs: Sequence[Ref]) -> list[list[Ref]]:
    """Connected groups of ``refs`` under any stored link, in first-occurrence order."""
    refs = list(dict.fromkeys(refs))
    parent = {r: r for r in refs}

    def find(r: Ref) -> Ref:
        while parent[r] != r:
            r = parent[r]
        return r

    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            if memory_links(views, a, b):
                parent[find(b)] = find(a)
    groups: dict[Ref, list[Ref]] = {}
    for r in refs:
        groups.setdefault(find(r), []).append(r)
    return list(groups.values())


def group_category(views: Mapping[str, ConversationView], refs: Sequence[Ref]) -> str:
    refs = list(dict.fromkeys(refs))
    if len(refs) <= 1:
        return SINGLE_MEMORY_EVIDENCE
    if len({r[0] for r in refs}) == 1:
        view = views[refs[0][0]]
        if set.intersection(*(set(view.source_messages(m)) for _, m in refs)):
            return MULTI_MEMORY_SAME_SOURCE
        if set.intersection(*(set(view.named_entities(m)) for _, m in refs)):
            return MULTI_MEMORY_SAME_ENTITY
    return MULTI_MEMORY_RELATED if len(components(views, refs)) == 1 else MULTI_MEMORY_DISCONNECTED


# ---- evidence items -----------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceItem:
    origin: str
    rank: int
    score: float | None
    conversation_id: str
    message_ids: tuple[str, ...]
    memory_id: str | None
    global_memory_id: str | None
    claim_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    speaker: dict[str, Any]
    temporal: dict[str, Any]
    status: str | None
    content: str
    contextual_content: str | None
    source_text: tuple[str, ...]
    source_window_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]  # memories derived from the item's messages (message items), or itself
    provenance: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {"origin": self.origin, "rank": self.rank, "score": self.score, "conversation_id": self.conversation_id,
                "message_ids": list(self.message_ids), "memory_id": self.memory_id,
                "global_memory_id": self.global_memory_id, "claim_ids": list(self.claim_ids),
                "entity_ids": list(self.entity_ids), "speaker": self.speaker, "temporal": self.temporal,
                "status": self.status, "content": self.content, "contextual_content": self.contextual_content,
                "source_text": list(self.source_text), "source_window_ids": list(self.source_window_ids),
                "memory_ids": list(self.memory_ids), "provenance": self.provenance}


def memory_item(view: ConversationView, memory_id: str, rank: int, score: float | None) -> EvidenceItem:
    cm = view.memory(memory_id)
    m = cm.memory
    messages = view.source_messages(memory_id)
    return EvidenceItem(
        ORIGIN_MEMORY, rank, score, view.conversation_id, messages, memory_id,
        global_memory_id(view.conversation_id, memory_id), tuple(s.claim_id for s in view.sources.get(memory_id, ())),
        view.named_entities(memory_id),
        {**cm.speaker.to_dict(), "attribution": speaker_attribution(view, memory_id, m.content),
         "contextual_attribution": speaker_attribution(view, memory_id, cm.contextual_content)},
        {"message_timestamps": [view.messages[x].timestamp for x in messages], "event_time": m.event_time,
         "time_class": m.time_class, "temporal_expression": m.temporal, "valid_from": m.valid_from,
         "valid_until": m.valid_until},
        m.status, m.content, cm.contextual_content, tuple(view.messages[x].content for x in messages),
        view.windows(memory_id), (memory_id,), provenance(view, memory_id))


def message_item(view: ConversationView, message_id: str, rank: int, score: float | None, origin: str) -> EvidenceItem:
    message = view.messages[message_id]
    speaker = view.registry.speaker(message.role, message.position)
    window = view.message_window(message_id)
    return EvidenceItem(
        origin, rank, score, view.conversation_id, (message_id,), None, None, view.claims_of.get(message_id, ()),
        tuple(sorted(view.written_entities(message.content))), speaker.to_dict(),
        {"message_timestamps": [message.timestamp]}, None, message.content, None, (message.content,),
        (window,) if window else (), view.memories_of.get(message_id, ()),
        {"conversation": True, "source_message": True, "source_window": window is not None,
         "claim": message_id in view.claims_of})


# ---- context assembly metrics -------------------------------------------------------------


def assembly_metrics(views: Mapping[str, ConversationView], items: Sequence[EvidenceItem]) -> dict[str, Any]:
    """Size and redundancy of one assembled context."""
    messages = [(i.conversation_id, m) for i in items for m in i.message_ids]
    windows = [(i.conversation_id, w) for i in items for w in i.source_window_ids]
    memories = {(i.conversation_id, m) for i in items for m in i.memory_ids}
    speakers = {(i.conversation_id, i.speaker["role"], i.speaker["name"]) for i in items}
    relationships = {(c, a, b, t) for c in {r[0] for r in memories} for a, b, t in views[c].edges()
                     if (c, a) in memories and (c, b) in memories}
    unique_messages = len(set(messages))
    return {"items": len(items), "unique_source_messages": unique_messages, "source_windows": len(set(windows)),
            "entities": len({(i.conversation_id, e) for i in items for e in i.entity_ids}),
            "speakers": len(speakers), "claims": len({(i.conversation_id, c) for i in items for c in i.claim_ids}),
            "relationships": len(relationships),
            "duplicate_source_window_rate": 1 - len(set(windows)) / len(windows) if windows else 0.0,
            "duplicate_message_rate": 1 - unique_messages / len(messages) if messages else 0.0,
            "memory_to_source_message_ratio": len(items) / unique_messages if unique_messages else None}


def distribution(values: Iterable[Any]) -> dict[str, int]:
    """Value -> count, numeric values in ascending order (floats rounded to 3 places)."""
    counts = Counter(None if v is None else round(v, 3) if isinstance(v, float) else v for v in values)
    numeric = sorted(k for k in counts if isinstance(k, (int, float)))
    other = sorted((k for k in counts if not isinstance(k, (int, float))), key=str)
    return {str(k).lower() if k is None else str(k): counts[k] for k in [*numeric, *other]}
