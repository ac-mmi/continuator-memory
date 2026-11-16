"""Contextual atomic memories (phase 5A): speaker identity and source windows for V2 memories.

    context window --> atomization --> validated claims --> contextualization --> consolidation
                                                                  |
                       ClaimContext: claim + its source window + speakers established by the source

A claim's ``content`` stays exactly as validated and consolidation still runs on it, so memory
state and versioning are unchanged. Contextualization adds, per claim:

  source window      the ordered messages the atomizer could use to resolve the claim: the
                     preceding messages of its atomization window and the claim's own message.
                     Stored as a reference (conversation, positions, message IDs), never as text;
                     ``resolve_window`` recovers the raw messages and verifies them.
  speaker            the role that asserted the claim and, only when the conversation itself
                     establishes it at or before the claim, that role's name (see ``SpeakerRegistry``).
  contextual content the claim with the atomizer's speaker frame ("the user", "the assistant")
                     replaced by an established name, prefixed with an attribution when a
                     non-user role asserted it. Nothing is inferred: an unnamed or ambiguously
                     named role keeps its frame wording, and other pronouns are left as stated
                     and flagged (``unresolved_reference``) so a reader consults the window.

``contextual_memories`` joins consolidated memories with the contexts of their source claims.
``global_memory_id`` namespaces the conversation-local ``memory_id`` so it is unique across
conversations; the local ID is kept unchanged for the existing phases.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.consolidation import _ordinal, claim_order, has_unresolved_reference
from continuator_memory.v2.memory import Memory, MemorySource
from continuator_memory.v2.models import Claim
from continuator_memory.v2.windowing import ContextWindow

ROLE_LABELS = {"user": "the user", "assistant": "the assistant", "system": "the system"}
SPEAKER_NAMED, SPEAKER_UNNAMED, SPEAKER_AMBIGUOUS = "named", "unnamed", "ambiguous"
SPEAKER_STATUSES = (SPEAKER_NAMED, SPEAKER_UNNAMED, SPEAKER_AMBIGUOUS)

_NAME = r"[A-Z][a-z][\w'-]*"
# "my name is Sam": a name on its own.
STRONG_NAMING = re.compile(rf"\b(?i:my name is|my name's)\s+({_NAME}(?:\s+{_NAME})?)")
# "call me Sam": the name if the role states no other, else an explicit alias of the stated name.
ALIAS_NAMING = re.compile(rf"\b(?i:call me)\s+({_NAME}(?:\s+{_NAME})?)")
# "I'm Sam." / "I am Sam," : also matches "I'm Spanish.", so it needs corroboration.
WEAK_NAMING = re.compile(rf"\b(?i:i'm|i am)\s+({_NAME})(?=\s*(?:[.,!;:]|$))")
GREETING = re.compile(r"^\s*(?i:hi|hello|hey|good (?:morning|afternoon|evening))\b")


def global_memory_id(conversation_id: str, memory_id: str) -> str:
    """Conversation-namespaced memory ID. The encoded conversation ID contains no "/", so the
    first "/" splits it back and distinct (conversation, memory) pairs never collide."""
    return f"{quote(conversation_id, safe='')}/{memory_id}"


# ---- source windows ---------------------------------------------------------


class SourceContextError(ValueError):
    """A source reference does not match the canonical stored messages."""


@dataclass(frozen=True)
class SourceWindow:
    """Reference to a contiguous, ordered run of raw messages ending at the claim's message."""

    conversation_id: str
    start: int  # position of the first message
    end: int  # position of the claim's own message (inclusive)
    message_ids: tuple[str, ...]

    @property
    def window_id(self) -> str:
        return f"{quote(self.conversation_id, safe='')}/{self.start}-{self.end}"

    @property
    def source_message_id(self) -> str:
        return self.message_ids[-1]

    def to_dict(self) -> dict[str, Any]:
        return {"window_id": self.window_id, "conversation_id": self.conversation_id, "start": self.start,
                "end": self.end, "message_ids": list(self.message_ids)}


def _window_of(conversation_id: str, messages: Sequence[Message]) -> SourceWindow:
    ordered = sorted(messages, key=lambda m: m.position)
    return SourceWindow(conversation_id, ordered[0].position, ordered[-1].position,
                        tuple(m.message_id for m in ordered))


def windows_by_message(windows: Iterable[ContextWindow]) -> dict[str, SourceWindow]:
    """The source window of every target message of the given atomization windows."""
    return {t.message_id: _window_of(w.conversation_id, (*w.preceding(t), t)) for w in windows for t in w.targets}


class MessageLookup(Protocol):
    def get_messages(self, conversation_id: str, message_ids: list[str] | tuple[str, ...]) -> list[Message]: ...


class ConversationMessages:
    """``MessageLookup`` over in-memory conversations (the V1 ``Store`` is the stored one)."""

    def __init__(self, conversations: Iterable[Conversation]):
        self._by_id = {(c.conversation_id, m.message_id): m for c in conversations for m in c.messages}

    def get_messages(self, conversation_id: str, message_ids: list[str] | tuple[str, ...]) -> list[Message]:
        return [self._by_id[(conversation_id, mid)] for mid in message_ids if (conversation_id, mid) in self._by_id]


@dataclass(frozen=True)
class SourceContext:
    """A resolved source window: the original messages, in order, with roles and text."""

    window: SourceWindow
    messages: tuple[Message, ...]

    def format(self) -> str:
        return "\n".join(f"[{m.message_id}] {m.role}: {m.content}" for m in self.messages)

    def to_dict(self) -> dict[str, Any]:
        return {"window": self.window.to_dict(),
                "messages": [{"message_id": m.message_id, "position": m.position, "role": m.role,
                              "content": m.content, "timestamp": m.timestamp} for m in self.messages]}


def resolve_window(window: SourceWindow, messages: MessageLookup) -> SourceContext:
    found = tuple(messages.get_messages(window.conversation_id, window.message_ids))
    if tuple(m.message_id for m in found) != window.message_ids:
        raise SourceContextError(f"{window.window_id}: stored messages {[m.message_id for m in found]} != "
                                 f"{list(window.message_ids)}")
    positions = [m.position for m in found]
    if positions != list(range(window.start, window.end + 1)):
        raise SourceContextError(f"{window.window_id}: positions {positions} are not {window.start}..{window.end}")
    return SourceContext(window, found)


# ---- speakers ---------------------------------------------------------------


@dataclass(frozen=True)
class Speaker:
    role: str
    name: str | None = None
    status: str = SPEAKER_UNNAMED
    evidence: tuple[str, ...] = ()  # message IDs where the role named itself (or was corroborated)
    aliases: tuple[tuple[str, str], ...] = ()  # (alias, message ID) declared with "call me", in order

    @property
    def label(self) -> str:
        return self.name or ROLE_LABELS.get(self.role, self.role)

    @property
    def alias_names(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(a for a, _ in self.aliases))

    def to_dict(self) -> dict[str, Any]:
        out = {"role": self.role, "name": self.name, "status": self.status, "evidence": list(self.evidence)}
        if self.aliases:
            out["aliases"] = [list(a) for a in self.aliases]
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Speaker:
        return cls(d["role"], d.get("name"), d.get("status", SPEAKER_UNNAMED), tuple(d.get("evidence", ())),
                   tuple((a, mid) for a, mid in d.get("aliases", ())))


def _text(message: Message) -> str:
    return message.content.replace("\u2019", "'")


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w'-]){re.escape(name)}(?![\w'-])", text) is not None


class SpeakerRegistry:
    """Names a conversation establishes for its roles, using only the conversation's messages.

    A role is named by its own messages: "my name is X" always, "I'm X" only when the message
    opens with a greeting or another role's message mentions X. "Call me Y" names the role when
    it states no other name, and is an explicit alias of the stated name otherwise. Only messages
    at or before the queried position count. Two different stated names for one role make it
    ambiguous; no name is ever taken from another role's statements alone or carried across roles,
    and names are never related by spelling ("Sam" is not an alias of "Samuel" unless declared)."""

    def __init__(self, messages: Iterable[Message]):
        self._messages = sorted(messages, key=lambda m: m.position)
        # (position, role, message_id, name, weak, alias)
        self._events: list[tuple[int, str, str, str, bool, bool]] = []
        for m in self._messages:
            text = _text(m)
            for match in STRONG_NAMING.finditer(text):
                self._events.append((m.position, m.role, m.message_id, match.group(1), False, False))
            for match in ALIAS_NAMING.finditer(text):
                self._events.append((m.position, m.role, m.message_id, match.group(1), False, True))
            for match in WEAK_NAMING.finditer(text):
                if not any(match.group(1) == e[3] and e[2] == m.message_id for e in self._events):
                    self._events.append((m.position, m.role, m.message_id, match.group(1),
                                         not GREETING.match(text), False))

    def declarations(self, role: str, position: int) -> list[tuple[str, str, bool]]:
        """(name, message ID, is "call me") for every accepted naming of ``role`` up to ``position``."""
        out = []
        for pos, event_role, mid, name, weak, alias in self._events:
            if pos > position or event_role != role:
                continue
            if weak and not any(m.role != role and m.position <= position and _mentions(_text(m), name)
                                for m in self._messages):
                continue
            out.append((name, mid, alias))
        return out

    def speaker(self, role: str, position: int) -> Speaker:
        names: dict[str, tuple[str, list[str]]] = {}  # first token -> (first full form, evidence)
        called: dict[str, tuple[str, list[str]]] = {}  # same, for "call me" declarations
        for name, mid, alias in self.declarations(role, position):
            form, evidence = (called if alias else names).setdefault(name.split()[0].lower(), (name, []))
            if mid not in evidence:
                evidence.append(mid)
        if not names:
            names, called = called, {}
        if not names:
            return Speaker(role)
        if len(names) > 1:
            return Speaker(role, None, SPEAKER_AMBIGUOUS, tuple(mid for _, ev in names.values() for mid in ev))
        (key, (form, evidence)), = names.items()
        aliases = tuple(sorted(((a, mid) for k, (a, ev) in called.items() if k != key for mid in ev),
                               key=lambda am: (self._position(am[1]), am[0])))
        return Speaker(role, form, SPEAKER_NAMED, tuple(evidence), aliases)

    def _position(self, message_id: str) -> int:
        return next(m.position for m in self._messages if m.message_id == message_id)


# ---- claim contextualization ------------------------------------------------


@dataclass(frozen=True)
class ClaimContext:
    claim_id: str
    conversation_id: str
    message_id: str
    position: int
    speaker: Speaker  # the role that asserted the claim
    window: SourceWindow
    contextual_content: str
    resolved_speakers: tuple[Speaker, ...] = ()  # roles whose frame wording was replaced by their name
    unresolved_reference: bool = False  # a pronoun is left as stated; the window holds its referent
    resolved_from: tuple[str, ...] = ()  # the claim's own identity-context sources (validation)

    def to_dict(self) -> dict[str, Any]:
        return {"claim_id": self.claim_id, "conversation_id": self.conversation_id, "message_id": self.message_id,
                "position": self.position, "speaker": self.speaker.to_dict(), "window": self.window.to_dict(),
                "contextual_content": self.contextual_content,
                "resolved_speakers": [s.to_dict() for s in self.resolved_speakers],
                "unresolved_reference": self.unresolved_reference, "resolved_from": list(self.resolved_from)}


def _frame_pattern(role: str) -> re.Pattern[str]:
    return re.compile(rf"\b([Tt])he {role}('s)?\b")


def contextual_content(claim: Claim, source: Message, registry: SpeakerRegistry) -> tuple[str, tuple[Speaker, ...]]:
    """The claim with established speaker names and attribution; see the module docstring."""
    text = claim.content
    resolved = []
    for role in ("user", "assistant"):
        pattern = _frame_pattern(role)
        if not pattern.search(text) or re.search(rf"\b{role}s?\b", source.content, re.IGNORECASE):
            continue  # no frame wording, or the source itself talks about a "user"/"assistant"
        speaker = registry.speaker(role, claim.provenance.position)
        if speaker.status == SPEAKER_NAMED:
            text = pattern.sub(lambda m: f"{speaker.name}{m.group(2) or ''}", text)
            resolved.append(speaker)
    role = claim.provenance.role
    if role != "user":
        speaker = registry.speaker(role, claim.provenance.position)
        own = (speaker.label, ROLE_LABELS.get(role, role))
        if not any(claim.content.lower().startswith(s.lower()) or text.lower().startswith(s.lower()) for s in own):
            label = speaker.label
            text = f"{label[0].upper()}{label[1:]} stated: {text}"
    return text, tuple(resolved)


def contextualize_claims(messages: Iterable[Message], claims: Iterable[Claim],
                         windows: Mapping[str, SourceWindow]) -> dict[str, ClaimContext]:
    """Contexts of one conversation's claims, in claim order. ``messages`` is the conversation."""
    messages = list(messages)
    by_id = {m.message_id: m for m in messages}
    registry = SpeakerRegistry(messages)
    out: dict[str, ClaimContext] = {}
    for claim in sorted(claims, key=claim_order):
        p = claim.provenance
        window = windows.get(p.message_id)
        if window is None or window.source_message_id != p.message_id:
            raise SourceContextError(f"{claim.claim_id}: no source window ending at {p.message_id}")
        content, resolved = contextual_content(claim, by_id[p.message_id], registry)
        out[claim.claim_id] = ClaimContext(
            claim.claim_id, p.conversation_id, p.message_id, p.position, registry.speaker(p.role, p.position),
            window, content, resolved, has_unresolved_reference(Claim(claim.claim_id, content, p)),
            claim.resolved_from)
    return out


# ---- contextual memories ----------------------------------------------------


@dataclass(frozen=True)
class ContextualMemory:
    """A consolidated memory with the contexts of its source claims (creating claim first)."""

    memory: Memory
    sources: tuple[ClaimContext, ...]
    # Phase 5B entity references (``continuator_memory.v2.entities.EntityRef``), source-claim order.
    entity_refs: tuple[Any, ...] = ()

    def entity_ids(self, role: str | None = None) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r.entity_id for r in self.entity_refs
                                   if r.entity_id and (role is None or r.role == role)))

    @property
    def global_id(self) -> str:
        return global_memory_id(self.memory.conversation_id, self.memory.memory_id)

    @property
    def context(self) -> ClaimContext:
        return next(s for s in self.sources if s.claim_id == self.memory.claim_id)

    @property
    def contextual_content(self) -> str:
        return self.context.contextual_content

    @property
    def speaker(self) -> Speaker:
        return self.context.speaker

    def evidence(self, messages: MessageLookup) -> list[SourceContext]:
        """The original source windows of every source claim, deduplicated, in source order."""
        windows = dict.fromkeys(s.window for s in self.sources)
        return [resolve_window(w, messages) for w in windows]

    def to_dict(self) -> dict[str, Any]:
        m = self.memory
        out = {"global_id": self.global_id, "memory_id": m.memory_id, "conversation_id": m.conversation_id,
               "content": m.content, "contextual_content": self.contextual_content, "status": m.status,
               "time_class": m.time_class, "valid_from": m.valid_from, "valid_until": m.valid_until,
               "superseded_by": m.superseded_by, "speaker": self.speaker.to_dict(),
               "sources": [s.to_dict() for s in self.sources]}
        if self.entity_refs:
            out["entity_refs"] = [r.to_dict() for r in self.entity_refs]
        return out


def contextual_memories(memories: Iterable[Memory], sources: Iterable[MemorySource],
                        contexts: Mapping[str, ClaimContext]) -> list[ContextualMemory]:
    """Join one conversation's memories with their source claim contexts, in memory order."""
    by_memory: dict[str, list[MemorySource]] = {}
    for s in sources:
        by_memory.setdefault(s.memory_id, []).append(s)
    out = []
    for m in sorted(memories, key=lambda m: (m.position, _ordinal(m.claim_id), m.memory_id)):
        own = sorted(by_memory.get(m.memory_id, ()), key=lambda s: (s.claim_id != m.claim_id, s.position, s.claim_id))
        missing = [s.claim_id for s in own if s.claim_id not in contexts]
        if m.claim_id not in contexts or missing:
            raise SourceContextError(f"{m.memory_id}: no context for claims {missing or [m.claim_id]}")
        out.append(ContextualMemory(m, tuple(contexts[s.claim_id] for s in own)))
    return out


def format_contextual_memory(memory: ContextualMemory, messages: MessageLookup) -> str:
    """Memory plus its original source context, as one evidence block for an answer layer."""
    m = memory.memory
    blocks = [f"[{memory.global_id}] {memory.contextual_content} (status: {m.status}; asserted by "
              f"{memory.speaker.label})"]
    for ctx in memory.evidence(messages):
        blocks.append(f"source {ctx.window.window_id}:\n{ctx.format()}")
    return "\n".join(blocks)
