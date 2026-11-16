"""Structured per-window extraction: entities, atomic memories and relationships from one LLM call.

Opt-in and isolated. Nothing here is reached by ``V2IngestionPipeline``, ``memory_atomizer`` or the web
API, and its output goes to its own tables (``structured_store``), so consolidation and retrieval never
see it. Original messages stay the source of truth.

One window is one ``complete_json`` call. Record IDs are generated here, never by the model.

Support. Every memory and relationship cites verbatim quotes from window messages, and a quote that is
not in its message rejects the record. A memory's statement is then compared with the cited messages:
  SOURCE            every content word occurs in the cited messages.
  IDENTITY_CONTEXT  the remaining words are names found elsewhere in the window or in prior memories.
  REVIEW            kept and marked ``needs_review``: a few other words differ, or the wording may have
                    dropped a negation or hedge, or stated a question as fact.
  rejected          a name or number with no source, no overlap with the quotes, or mostly unsupported.
Like ``validation``, this compares words. It does not establish entailment, so ``confirmed`` means
"stated first-hand, with no review reason", not "true".

Attribution. Speaker and role come from the cited message, never from the model. A statement about one
participant made by another is ``reported``, and is not confirmed.

Time. ``temporal.kind`` describes how the window states something. There is deliberately no status such
as superseded, and "current" does not mean globally latest: that needs the whole conversation.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from continuator_memory.memory.models import normalize_value
from continuator_memory.models import Message
from continuator_memory.v2.atomizer import (
    AtomizationRequestError,
    JSONCompletionClient,
    OllamaChatClient,
    V2Settings,
)
from continuator_memory.v2.graph_index import split_speaker
from continuator_memory.v2.models import SUPPORT_IDENTITY, SUPPORT_SOURCE
from continuator_memory.v2.prompt import _message_line
from continuator_memory.v2.validation import (
    FLAGGED,
    FRAME_TOKENS,
    NAME_SPAN,
    PRONOUNS,
    REJECTED,
    ValidationIssue,
    _comparable,
    tokens,
)
from continuator_memory.v2.windowing import ContextWindow

STRUCTURED_PROTOCOL_VERSION = "structured-v1"

SUPPORT_REVIEW = "REVIEW"
CATEGORIES = ("fact", "decision", "preference", "goal", "event", "plan", "condition")
CATEGORY_UNCLASSIFIED = "unclassified"
ASSERTION_TYPES = ("assertion", "confirmation", "question", "hypothetical", "suggestion", "assumption")
ASSERTION_UNCLASSIFIED = "unclassified"
FIRST_HAND_ASSERTIONS = ("assertion", "confirmation")
TEMPORAL_KINDS = ("historical", "current", "future", "conditional", "correction", "unspecified")
KIND_UNSPECIFIED = "unspecified"
# Suggested, not closed: another lowercase word is kept and flagged.
ENTITY_TYPES = ("person", "project", "company", "organization", "product", "place", "event", "concept")

MAX_UNRESOLVED_WORDS = 2
MAX_MISSING_FRACTION = 0.5
MAX_STATEMENT_WORDS = 40
MAX_LABEL_CHARS = 40

_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")
_NEGATION = re.compile(r"\b(?:not|no|never|without|nothing|none|neither|nor|nowhere|against)\b|n['’]t\b", re.IGNORECASE)
_HEDGE = re.compile(r"\b(?:might|may|maybe|perhaps|possibly|could|if|unless|probably)\b", re.IGNORECASE)
_DATE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")
_YEAR = re.compile(r"\b(\d{4})\b")
_DAY = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\b")
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december")
_MONTH = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.IGNORECASE)
_NUMBER = re.compile(r"\b\d[\d.,:/-]*\b")
_FIRST_PERSON = frozenset("i me my mine myself we us our ours".split())
_SECOND_PERSON = frozenset("you your yours".split())

STRUCTURED_SYSTEM_PROMPT = """\
You extract structured long-term memory from a window of a conversation, for TARGET messages only.

Return three lists.
entities: people, projects, companies, organizations, products, places, events and concepts named in a \
TARGET message. name is exactly as written. aliases only when the message states another name for it. \
Do not decide that names in different messages are the same thing unless a message says so.
memories: atomic facts, each one sentence understandable on its own. Split independent facts. Keep a \
reason in "reason", not in a second fact. Replace I/me/my with the speaker's name shown at the start of \
the message, otherwise "the user" or "the assistant". Do not assume anyone's gender. Skip greetings, \
thanks, small talk and reactions. subject, predicate and object restate the statement; object may be empty.
relationships: only links a message explicitly states between two named things. Never link things only \
because they appear in the same window. type is a short snake_case verb such as develops, works_at, \
formerly_named, prefers, depends_on, postponed_to, replaced_by, decided_against or related_to.

Rules:
1. EVIDENCE: every memory and relationship needs evidence: message_id and a quote copied exactly from \
that message. Cite a CONTEXT message only alongside a TARGET message, when it is needed to understand it, \
such as the question that "Yes" answers. Do not use a CONTEXT message on its own. Never invent quotes or IDs.
2. assertion_type: assertion (the speaker states it), confirmation (the speaker confirms what another \
person said or asked; cite both messages), question, hypothetical, suggestion or assumption. Keep what \
one person says about another attributed to the message that said it.
3. category: fact, decision, preference, goal, event, plan or condition.
4. temporal.expression: the time words exactly as written, or empty. temporal.normalized: YYYY, YYYY-MM \
or YYYY-MM-DD only when the expression and message timestamp make it certain, or empty. temporal.kind: \
historical, current, future, conditional, correction or unspecified, as this message states it. Never say \
a fact is outdated or superseded; you only see part of the conversation.
5. Keep negation, hedges and conditions in the statement ("might move", not "moves").
6. If a TARGET message has nothing durable, extract nothing from it. Empty lists are valid.
"""

_EXAMPLE_PROMPT = """\
PRIOR MEMORIES (identity/reference only, NOT evidence):
(none)

CONTEXT MESSAGES (do not extract):
[x1] user @ 2024-03-05: Ines: Rafi, are you still at Harbour Print?

TARGET MESSAGES (extract from each):
[x2] assistant @ 2024-03-05: Rafi: Yes, but I might leave in June. I chose Tallow over Ember because it is cheaper.
[x3] user @ 2024-03-05: Ines: Nice! I don't drink coffee anymore.

JSON:"""

_EXAMPLE_OUTPUT = {
    "entities": [
        {"name": "Rafi", "type": "person", "aliases": [], "message_ids": ["x2"]},
        {"name": "Tallow", "type": "product", "aliases": [], "message_ids": ["x2"]},
        {"name": "Ember", "type": "product", "aliases": [], "message_ids": ["x2"]},
        {"name": "Ines", "type": "person", "aliases": [], "message_ids": ["x3"]}],
    "memories": [
        {"statement": "Rafi is at Harbour Print.", "subject": "Rafi", "predicate": "is at", "object": "Harbour Print",
         "category": "fact", "assertion_type": "confirmation", "reason": "",
         "temporal": {"expression": "", "normalized": "", "kind": "current"},
         "evidence": [{"message_id": "x1", "quote": "are you still at Harbour Print?"},
                      {"message_id": "x2", "quote": "Yes"}]},
        {"statement": "Rafi might leave in June.", "subject": "Rafi", "predicate": "might leave", "object": "",
         "category": "condition", "assertion_type": "hypothetical", "reason": "",
         "temporal": {"expression": "in June", "normalized": "2024-06", "kind": "conditional"},
         "evidence": [{"message_id": "x2", "quote": "I might leave in June"}]},
        {"statement": "Rafi chose Tallow over Ember.", "subject": "Rafi", "predicate": "chose", "object": "Tallow",
         "category": "decision", "assertion_type": "assertion", "reason": "Tallow is cheaper",
         "temporal": {"expression": "", "normalized": "", "kind": "historical"},
         "evidence": [{"message_id": "x2", "quote": "I chose Tallow over Ember because it is cheaper"}]},
        {"statement": "Ines does not drink coffee anymore.", "subject": "Ines", "predicate": "does not drink",
         "object": "coffee", "category": "preference", "assertion_type": "assertion", "reason": "",
         "temporal": {"expression": "", "normalized": "", "kind": "correction"},
         "evidence": [{"message_id": "x3", "quote": "I don't drink coffee anymore"}]}],
    "relationships": [
        {"subject": "Rafi", "type": "works_at", "object": "Harbour Print", "assertion_type": "confirmation",
         "temporal": {"expression": "", "normalized": "", "kind": "current"},
         "evidence": [{"message_id": "x1", "quote": "are you still at Harbour Print?"},
                      {"message_id": "x2", "quote": "Yes"}]}],
}
STRUCTURED_EXAMPLES: tuple[tuple[str, str], ...] = ((_EXAMPLE_PROMPT, json.dumps(_EXAMPLE_OUTPUT)),)


@dataclass(frozen=True)
class Evidence:
    message_id: str
    quote: str

    def to_dict(self) -> dict[str, Any]:
        return {"message_id": self.message_id, "quote": self.quote}


@dataclass(frozen=True)
class StructuredTemporal:
    expression: str | None = None
    normalized: str | None = None
    kind: str = KIND_UNSPECIFIED

    def to_dict(self) -> dict[str, Any]:
        return {"expression": self.expression, "normalized": self.normalized, "kind": self.kind}


@dataclass(frozen=True)
class EntityMention:
    entity_id: str
    name: str
    type: str
    aliases: tuple[str, ...]
    message_ids: tuple[str, ...]
    speakers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "name": self.name, "type": self.type, "aliases": list(self.aliases),
                "message_ids": list(self.message_ids), "speakers": list(self.speakers)}


@dataclass(frozen=True)
class StructuredMemory:
    memory_id: str
    statement: str
    subject: str
    predicate: str
    object: str
    category: str
    assertion_type: str
    reason: str | None
    temporal: StructuredTemporal
    evidence: tuple[Evidence, ...]
    speaker: str
    speaker_role: str
    reported: bool
    support: str
    review_reasons: tuple[str, ...] = ()

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.message_id for e in self.evidence))

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)

    @property
    def confirmed(self) -> bool:
        return self.assertion_type in FIRST_HAND_ASSERTIONS and not self.reported and not self.needs_review

    def to_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "statement": self.statement, "subject": self.subject,
                "predicate": self.predicate, "object": self.object, "category": self.category,
                "assertion_type": self.assertion_type, "reason": self.reason, "temporal": self.temporal.to_dict(),
                "evidence": [e.to_dict() for e in self.evidence], "message_ids": list(self.message_ids),
                "speaker": self.speaker, "speaker_role": self.speaker_role, "reported": self.reported,
                "support": self.support, "review_reasons": list(self.review_reasons),
                "needs_review": self.needs_review, "confirmed": self.confirmed}


@dataclass(frozen=True)
class StructuredRelationship:
    relationship_id: str
    subject: str
    type: str
    object: str
    assertion_type: str
    temporal: StructuredTemporal
    evidence: tuple[Evidence, ...]
    speaker: str
    speaker_role: str
    reported: bool
    review_reasons: tuple[str, ...] = ()

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.message_id for e in self.evidence))

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)

    def to_dict(self) -> dict[str, Any]:
        return {"relationship_id": self.relationship_id, "subject": self.subject, "type": self.type,
                "object": self.object, "assertion_type": self.assertion_type, "temporal": self.temporal.to_dict(),
                "evidence": [e.to_dict() for e in self.evidence], "message_ids": list(self.message_ids),
                "speaker": self.speaker, "speaker_role": self.speaker_role, "reported": self.reported,
                "review_reasons": list(self.review_reasons), "needs_review": self.needs_review}


@dataclass
class StructuredWindow:
    window: ContextWindow
    entities: list[EntityMention] = field(default_factory=list)
    memories: list[StructuredMemory] = field(default_factory=list)
    relationships: list[StructuredRelationship] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    raw: str | None = None
    error: str | None = None

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        return "invalid" if any(i.action == REJECTED for i in self.issues) else "ok"

    @property
    def targets_without_output(self) -> tuple[str, ...]:
        """Targets no record cites: nothing durable, or something the model missed."""
        cited = {mid for r in (*self.memories, *self.relationships) for mid in r.message_ids}
        cited |= {mid for e in self.entities for mid in e.message_ids}
        return tuple(mid for mid in self.window.target_ids if mid not in cited)


@dataclass
class StructuredResult:
    validated: StructuredWindow
    prompt: str
    latency_ms: float
    usage: dict[str, Any] | None = None


def build_structured_prompt(window: ContextWindow, prior: Sequence[StructuredMemory] = ()) -> str:
    prior_text = "\n".join(f"- {m.statement}" for m in prior) or "(none)"
    context = "\n".join(_message_line(m) for m in window.context) or "(none)"
    targets = "\n".join(_message_line(m) for m in window.targets)
    return (f"PRIOR MEMORIES (identity/reference only, NOT evidence):\n{prior_text}\n\n"
            f"CONTEXT MESSAGES (do not extract):\n{context}\n\n"
            f"TARGET MESSAGES (extract from each):\n{targets}\n\nJSON:")


def structured_output_schema(window: ContextWindow) -> dict[str, Any]:
    text = {"type": "string"}
    ids = {"type": "string", "enum": [m.message_id for m in window.messages]}
    strings = {"type": "array", "items": text}
    evidence = {"type": "array", "items": {"type": "object", "required": ["message_id", "quote"],
                                           "properties": {"message_id": ids, "quote": text}}}
    temporal = {"type": "object", "required": ["expression", "normalized", "kind"],
                "properties": {"expression": text, "normalized": text,
                               "kind": {"type": "string", "enum": list(TEMPORAL_KINDS)}}}
    assertion = {"type": "string", "enum": list(ASSERTION_TYPES)}
    entity = {"type": "object", "required": ["name", "type", "aliases", "message_ids"],
              "properties": {"name": text, "type": text, "aliases": strings,
                             "message_ids": {"type": "array", "items": ids}}}
    memory = {"type": "object",
              "required": ["statement", "subject", "predicate", "object", "category", "assertion_type", "reason",
                           "temporal", "evidence"],
              "properties": {"statement": text, "subject": text, "predicate": text, "object": text,
                             "category": {"type": "string", "enum": list(CATEGORIES)},
                             "assertion_type": assertion, "reason": text, "temporal": temporal, "evidence": evidence}}
    relationship = {"type": "object", "required": ["subject", "type", "object", "assertion_type", "temporal", "evidence"],
                    "properties": {"subject": text, "type": text, "object": text, "assertion_type": assertion,
                                   "temporal": temporal, "evidence": evidence}}
    return {"type": "object", "required": ["entities", "memories", "relationships"],
            "properties": {"entities": {"type": "array", "items": entity},
                           "memories": {"type": "array", "items": memory},
                           "relationships": {"type": "array", "items": relationship}}}


def _normal(text: str) -> str:
    return " ".join(text.replace("’", "'").replace("‘", "'").lower().split())


def _text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def window_ref(window: ContextWindow) -> str:
    return f"{window.conversation_id}:w{window.index:04d}"


def _speaker_of(message: Message) -> str:
    return split_speaker(message.content, message.role)[0]


def _first_token_is(subject: str, speaker: str) -> bool:
    words, name = normalize_value(subject).split(), normalize_value(speaker).split()
    if words[:1] == ["the"]:
        words = words[1:]
    return bool(name) and words[:len(name)] == name


class _Checker:
    """Validation state for one window."""

    def __init__(self, window: ContextWindow, prior: Sequence[StructuredMemory], issues: list[ValidationIssue]):
        self.window, self.issues = window, issues
        self.by_id = {m.message_id: m for m in window.messages}
        self.targets = set(window.target_ids)
        self.window_tokens = {t for m in window.messages for t in tokens(m.content)}
        self.window_names = {t for m in window.messages for span in NAME_SPAN.findall(m.content) for t in tokens(span)}
        self.prior_names = {t for m in prior for part in (m.subject, m.object) for t in tokens(part)}
        self.speakers = {_speaker_of(m) for m in window.messages}

    def issue(self, code: str, action: str, detail: str, ref: str, message_id: str | None = None) -> None:
        self.issues.append(ValidationIssue(code, action, detail, message_id, ref))

    def evidence(self, raw: Any, ref: str) -> tuple[Evidence, ...] | None:
        if not isinstance(raw, list) or not raw:
            self.issue("MISSING_EVIDENCE", REJECTED, "no evidence given", ref)
            return None
        out: list[Evidence] = []
        for item in raw:
            mid = item.get("message_id") if isinstance(item, dict) else None
            quote = _text(item.get("quote")) if isinstance(item, dict) else ""
            if mid not in self.by_id:
                self.issue("INVALID_EVIDENCE_MESSAGE", REJECTED, f"{mid!r} is not a message of this window", ref)
                return None
            if not quote or _normal(quote) not in _normal(self.by_id[mid].content):
                self.issue("QUOTE_NOT_IN_SOURCE", REJECTED, f"quote {quote!r} is not in {mid}", ref, mid)
                return None
            out.append(Evidence(mid, quote))
        if not any(e.message_id in self.targets for e in out):
            self.issue("CONTEXT_ONLY_EVIDENCE", REJECTED, "evidence cites no target message", ref)
            return None
        return tuple(dict.fromkeys(out))

    def asserting_message(self, evidence: Sequence[Evidence], assertion_type: str) -> Message:
        """The target that makes the statement: the confirming reply, otherwise the first target cited."""
        cited = [self.by_id[e.message_id] for e in evidence if e.message_id in self.targets]
        return cited[-1] if assertion_type == "confirmation" else cited[0]

    def assertion(self, raw: Any, ref: str) -> str:
        if raw in ASSERTION_TYPES:
            return raw
        self.issue("INVALID_ASSERTION_TYPE", FLAGGED, f"{raw!r} not in {list(ASSERTION_TYPES)}", ref)
        return ASSERTION_UNCLASSIFIED

    def assertion_reasons(self, assertion_type: str, evidence: Sequence[Evidence], speaker: str, ref: str) -> list[str]:
        reasons: list[str] = []
        if assertion_type == ASSERTION_UNCLASSIFIED:
            reasons.append("INVALID_ASSERTION_TYPE")
        if assertion_type in FIRST_HAND_ASSERTIONS and all(e.quote.rstrip().endswith("?") for e in evidence):
            self.issue("QUESTION_AS_FACT", FLAGGED, "only questions are cited for a stated fact", ref)
            reasons.append("QUESTION_AS_FACT")
        if assertion_type == "confirmation" and not any(
                _speaker_of(self.by_id[e.message_id]) != speaker for e in evidence):
            self.issue("CONFIRMATION_WITHOUT_PROMPT", FLAGGED, "no message from another speaker is cited", ref)
            reasons.append("CONFIRMATION_WITHOUT_PROMPT")
        return reasons

    def reported(self, subject: str, speaker: str) -> bool:
        return any(_first_token_is(subject, other) for other in self.speakers - {speaker}) \
            and not _first_token_is(subject, speaker)

    def temporal(self, raw: Any, evidence: Sequence[Evidence], source: Message, ref: str) -> StructuredTemporal:
        if not isinstance(raw, dict):
            if raw not in (None, ""):
                self.issue("MALFORMED_TEMPORAL", FLAGGED, "temporal is not an object; dropped", ref)
            return StructuredTemporal()
        kind = raw.get("kind")
        if kind not in TEMPORAL_KINDS:
            if kind not in (None, ""):
                self.issue("INVALID_TEMPORAL_KIND", FLAGGED, f"{kind!r} not in {list(TEMPORAL_KINDS)}", ref)
            kind = KIND_UNSPECIFIED
        expression, normalized = _text(raw.get("expression")), _text(raw.get("normalized"))
        if expression and not any(_normal(expression) in _normal(self.by_id[e.message_id].content) for e in evidence):
            self.issue("TEMPORAL_NOT_IN_SOURCE", FLAGGED, f"{expression!r} is not in the cited messages; dropped", ref)
            expression = ""
        if normalized and not expression:
            self.issue("TEMPORAL_UNSUPPORTED", FLAGGED, f"{normalized!r} has no stated expression; dropped", ref)
            normalized = ""
        if normalized:
            problem = _normalized_problem(normalized, expression, kind, source.timestamp)
            if problem:
                self.issue(problem, FLAGGED, f"{normalized!r} for {expression!r}; dropped", ref)
                normalized = ""
        return StructuredTemporal(expression or None, normalized or None, kind)

    def statement_support(self, statement: str, evidence: Sequence[Evidence], ref: str) -> tuple[str, list[str]] | None:
        """(support, review reasons), or None after recording a rejection."""
        words = tokens(statement) - FRAME_TOKENS
        if not words or not _comparable(statement):
            self.issue("UNVERIFIABLE_CLAIM", REJECTED, f"{statement!r}: cannot be compared with the source", ref)
            return None
        cited = {t for e in evidence for t in tokens(self.by_id[e.message_id].content)}
        quoted = {t for e in evidence for t in tokens(e.quote)}
        if not words & quoted:
            self.issue("QUOTE_UNRELATED", REJECTED, f"{statement!r}: shares no words with its quotes", ref)
            return None
        missing = words - cited
        if not missing:
            return SUPPORT_SOURCE, []
        known_names = self.window_names | self.prior_names
        invented = {t for t in _hard_tokens(statement) if t in missing and t not in known_names}
        if invented:
            self.issue("UNSUPPORTED_CLAIM", REJECTED, f"{statement!r}: no source for {sorted(invented)}", ref)
            return None
        if missing <= known_names:
            return SUPPORT_IDENTITY, []
        unresolved = missing - known_names - self.window_tokens
        if len(unresolved) > MAX_UNRESOLVED_WORDS or len(missing) / len(words) > MAX_MISSING_FRACTION:
            self.issue("UNSUPPORTED_CLAIM", REJECTED, f"{statement!r}: not supported ({sorted(missing)})", ref)
            return None
        self.issue("LEXICAL_DIFFERENCE", FLAGGED, f"{statement!r}: words not in cited messages ({sorted(missing)})", ref)
        return SUPPORT_REVIEW, ["LEXICAL_DIFFERENCE"]

    def wording_reasons(self, statement: str, assertion_type: str, category: str, kind: str,
                        evidence: Sequence[Evidence], ref: str) -> list[str]:
        reasons: list[str] = []
        quotes = " ".join(e.quote for e in evidence)
        if bool(_NEGATION.search(statement)) != bool(_NEGATION.search(quotes)):
            self.issue("NEGATION_MISMATCH", FLAGGED, f"{statement!r} and its quotes differ in negation", ref)
            reasons.append("NEGATION_MISMATCH")
        if (_HEDGE.search(quotes) and not _HEDGE.search(statement) and assertion_type == "assertion"
                and category != "condition" and kind != "conditional"):
            self.issue("HEDGE_DROPPED", FLAGGED, f"{statement!r} drops a hedge or condition in its quotes", ref)
            reasons.append("HEDGE_DROPPED")
        if len(statement.split()) > MAX_STATEMENT_WORDS:
            self.issue("NOT_ATOMIC", FLAGGED, f"statement has more than {MAX_STATEMENT_WORDS} words", ref)
            reasons.append("NOT_ATOMIC")
        return reasons

    def endpoint_supported(self, endpoint: str, evidence: Sequence[Evidence], speaker: str, assertion_type: str) -> bool:
        wanted = tokens(endpoint)
        if not wanted:
            return False
        quoted = {t for e in evidence for t in tokens(e.quote)}
        if wanted <= quoted:
            return True
        words = {w for e in evidence for w in normalize_value(e.quote).split()}
        if _first_token_is(endpoint, speaker):
            return bool(words & _FIRST_PERSON) or (assertion_type == "confirmation" and bool(words & _SECOND_PERSON))
        referring = bool(words & (PRONOUNS - _FIRST_PERSON))
        return referring and wanted <= (self.window_names | self.prior_names)


def _hard_tokens(statement: str) -> set[str]:
    """Names (other than a sentence-initial word) and numbers: words a paraphrase cannot change."""
    found: set[str] = set()
    for match in NAME_SPAN.finditer(statement):
        span = match.group(0)
        if match.start() == 0:
            span = " ".join(span.split()[1:])
        found |= tokens(span)
    for number in _NUMBER.findall(statement):
        found |= tokens(number)
    return found


def _normalized_problem(normalized: str, expression: str, kind: str, timestamp: str | None) -> str | None:
    match = _DATE.match(normalized)
    if not match:
        return "MALFORMED_TEMPORAL"
    year, month, day = match.group(1), match.group(2), match.group(3)
    try:
        date(int(year), int(month or 1), int(day or 1))
    except ValueError:
        return "MALFORMED_TEMPORAL"
    stated_year = _YEAR.search(expression)
    if stated_year and stated_year.group(1) != year:
        return "TEMPORAL_MISMATCH"
    stated_month = _MONTH.search(expression)
    if stated_month and month and _MONTHS.index(stated_month.group(1).lower()) + 1 != int(month):
        return "TEMPORAL_MISMATCH"
    stated_day = _DAY.search(_YEAR.sub("", expression))
    if stated_day and day and int(stated_day.group(1)) != int(day):
        return "TEMPORAL_MISMATCH"
    if not stated_year and not timestamp:
        return "RESOLVED_WITHOUT_TIMESTAMP"
    if timestamp:
        observed = timestamp[: len(normalized)]
        if (kind == "historical" and normalized > observed) or (kind == "future" and normalized < observed):
            return "TEMPORAL_IMPLAUSIBLE"
    return None


def _entities(raw: list[Any], check: _Checker, ref: str) -> list[EntityMention]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for i, item in enumerate(raw):
        iref = f"entities[{i}]"
        if not isinstance(item, dict):
            check.issue("INVALID_SCHEMA", REJECTED, "entity is not an object", iref)
            continue
        name = _text(item.get("name"))
        if not name:
            check.issue("EMPTY_ENTITY", REJECTED, "entity has no name", iref)
            continue
        raw_ids = item.get("message_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            check.issue("MISSING_EVIDENCE", REJECTED, f"entity {name!r} cites no messages", iref)
            continue
        unknown = [m for m in raw_ids if m not in check.by_id]
        if unknown:
            check.issue("INVALID_EVIDENCE_MESSAGE", REJECTED, f"entity {name!r} cites unknown {unknown!r}", iref)
            continue
        targets = [m for m in dict.fromkeys(raw_ids) if m in check.targets]
        containing = [m for m in targets if _normal(name) in _normal(check.by_id[m].content)]
        if not containing:
            check.issue("ENTITY_NOT_IN_SOURCE", REJECTED, f"entity {name!r} is not in a cited target message", iref)
            continue
        etype = _text(item.get("type")).lower().replace(" ", "_")
        if not _LABEL.match(etype) or len(etype) > MAX_LABEL_CHARS:
            check.issue("INVALID_ENTITY_TYPE", FLAGGED, f"type {item.get('type')!r}; kept as concept", iref)
            etype = "concept"
        elif etype not in ENTITY_TYPES:
            check.issue("UNKNOWN_ENTITY_TYPE", FLAGGED, f"type {etype!r} kept as stated", iref)
        aliases = []
        raw_aliases = item.get("aliases")
        for alias in raw_aliases if isinstance(raw_aliases, list) else []:
            text = _text(alias)
            if text and _normal(text) != _normal(name):
                if any(_normal(text) in _normal(check.by_id[m].content) for m in containing):
                    aliases.append(text)
                else:
                    check.issue("UNSUPPORTED_ALIAS", FLAGGED, f"alias {text!r} of {name!r} is not in its messages", iref)
        row = merged.setdefault((_normal(name), etype), {"name": name, "type": etype, "aliases": [], "ids": []})
        row["aliases"] += [a for a in aliases if a not in row["aliases"]]
        row["ids"] += [m for m in containing if m not in row["ids"]]
    return [EntityMention(f"{ref}:e{n}", r["name"], r["type"], tuple(r["aliases"]), tuple(r["ids"]),
                          tuple(dict.fromkeys(_speaker_of(check.by_id[m]) for m in r["ids"])))
            for n, r in enumerate(merged.values(), 1)]


def _memories(raw: list[Any], check: _Checker, ref: str) -> list[StructuredMemory]:
    out: list[StructuredMemory] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        iref = f"memories[{i}]"
        if not isinstance(item, dict):
            check.issue("INVALID_SCHEMA", REJECTED, "memory is not an object", iref)
            continue
        statement, subject = _text(item.get("statement")), _text(item.get("subject"))
        if not statement or not subject:
            check.issue("EMPTY_CLAIM", REJECTED, "memory needs a statement and a subject", iref)
            continue
        evidence = check.evidence(item.get("evidence"), iref)
        if evidence is None:
            continue
        support = check.statement_support(statement, evidence, iref)
        if support is None:
            continue
        level, reasons = support
        if _normal(statement) in seen:
            check.issue("DUPLICATE_MEMORY", FLAGGED, f"{statement!r} repeated; first kept", iref)
            continue
        assertion_type = check.assertion(item.get("assertion_type"), iref)
        category = item.get("category")
        if category not in CATEGORIES:
            check.issue("INVALID_CATEGORY", FLAGGED, f"{category!r} not in {list(CATEGORIES)}", iref)
            reasons = [*reasons, "INVALID_CATEGORY"]
            category = CATEGORY_UNCLASSIFIED
        source = check.asserting_message(evidence, assertion_type)
        speaker = _speaker_of(source)
        temporal = check.temporal(item.get("temporal"), evidence, source, iref)
        reasons = [*reasons, *check.assertion_reasons(assertion_type, evidence, speaker, iref),
                   *check.wording_reasons(statement, assertion_type, category, temporal.kind, evidence, iref)]
        reason = _text(item.get("reason")) or None
        if reason:
            cited = {t for e in evidence for t in tokens(check.by_id[e.message_id].content)}
            if not (tokens(reason) - FRAME_TOKENS) & cited:
                check.issue("REASON_NOT_IN_SOURCE", FLAGGED, f"reason {reason!r} shares no words with its evidence; dropped", iref)
                reason = None
            elif (tokens(reason) - FRAME_TOKENS) - cited:
                check.issue("LEXICAL_DIFFERENCE", FLAGGED, f"reason {reason!r} differs from its evidence", iref)
                reasons.append("REASON_LEXICAL_DIFFERENCE")
        seen.add(_normal(statement))
        out.append(StructuredMemory(
            f"{ref}:m{len(out) + 1}", statement, subject, _text(item.get("predicate")), _text(item.get("object")),
            category, assertion_type, reason, temporal, evidence, speaker, source.role,
            check.reported(subject, speaker), SUPPORT_REVIEW if reasons else level, tuple(dict.fromkeys(reasons))))
    return out


def _relationships(raw: list[Any], check: _Checker, ref: str) -> list[StructuredRelationship]:
    out: list[StructuredRelationship] = []
    seen: set[tuple[str, str, str]] = set()
    for i, item in enumerate(raw):
        iref = f"relationships[{i}]"
        if not isinstance(item, dict):
            check.issue("INVALID_SCHEMA", REJECTED, "relationship is not an object", iref)
            continue
        subject, obj = _text(item.get("subject")), _text(item.get("object"))
        rtype = _text(item.get("type")).lower().replace(" ", "_").replace("-", "_")
        if not subject or not obj:
            check.issue("EMPTY_RELATIONSHIP", REJECTED, "relationship needs a subject and an object", iref)
            continue
        if not _LABEL.match(rtype) or len(rtype) > MAX_LABEL_CHARS:
            check.issue("INVALID_RELATIONSHIP_TYPE", REJECTED, f"type {item.get('type')!r}", iref)
            continue
        evidence = check.evidence(item.get("evidence"), iref)
        if evidence is None:
            continue
        assertion_type = check.assertion(item.get("assertion_type"), iref)
        source = check.asserting_message(evidence, assertion_type)
        speaker = _speaker_of(source)
        unsupported = [e for e in (subject, obj) if not check.endpoint_supported(e, evidence, speaker, assertion_type)]
        if unsupported:
            check.issue("ENDPOINT_NOT_IN_EVIDENCE", REJECTED,
                        f"{subject!r} {rtype} {obj!r}: {unsupported!r} not stated by the quotes", iref)
            continue
        key = (_normal(subject), rtype, _normal(obj))
        if key in seen:
            check.issue("DUPLICATE_RELATIONSHIP", FLAGGED, f"{key} repeated; first kept", iref)
            continue
        seen.add(key)
        temporal = check.temporal(item.get("temporal"), evidence, source, iref)
        reasons = check.assertion_reasons(assertion_type, evidence, speaker, iref)
        out.append(StructuredRelationship(
            f"{ref}:r{len(out) + 1}", subject, rtype, obj, assertion_type, temporal, evidence, speaker, source.role,
            check.reported(subject, speaker), tuple(reasons)))
    return out


def validate_structured(raw: str | None, window: ContextWindow, prior: Sequence[StructuredMemory] = (),
                        done_reason: str | None = None) -> StructuredWindow:
    """Turn one model response into validated records; bad records are dropped with a recorded issue."""
    def failed(code: str, error: str) -> StructuredWindow:
        return StructuredWindow(window, issues=[ValidationIssue(code, REJECTED, error)], raw=raw, error=error)

    if done_reason == "length":
        return failed("TRUNCATED_RESPONSE", "the model hit its output limit (done_reason=length)")
    try:
        data = json.loads(raw or "")
    except (json.JSONDecodeError, TypeError) as exc:
        return failed("INVALID_JSON", f"invalid JSON: {exc}")
    sections = ("entities", "memories", "relationships")
    if not isinstance(data, dict) or not all(isinstance(data.get(s), list) for s in sections):
        return failed("INVALID_SCHEMA", f"top level must be an object with lists {list(sections)}")
    issues: list[ValidationIssue] = []
    check = _Checker(window, prior, issues)
    ref = window_ref(window)
    return StructuredWindow(window, _entities(data["entities"], check, ref), _memories(data["memories"], check, ref),
                            _relationships(data["relationships"], check, ref), issues, raw)


class StructuredExtractor:
    """One ``complete_json`` call per window, using the same client protocol as ``SemanticAtomizer``."""

    def __init__(self, client: JSONCompletionClient):
        self.client = client
        self.model_name = client.model_name

    def extract(self, window: ContextWindow, prior: Sequence[StructuredMemory] = ()) -> StructuredResult:
        prompt = build_structured_prompt(window, prior)
        if hasattr(self.client, "last_usage"):
            self.client.last_usage = None
        t0 = time.perf_counter()
        usage = None
        try:
            raw = self.client.complete_json(STRUCTURED_SYSTEM_PROMPT, prompt, structured_output_schema(window),
                                            STRUCTURED_EXAMPLES)
        except AtomizationRequestError as exc:
            validated = StructuredWindow(window, issues=[ValidationIssue("REQUEST_FAILED", REJECTED, str(exc))], error=str(exc))
        else:
            usage = getattr(self.client, "last_usage", None)
            validated = validate_structured(raw, window, prior, (usage or {}).get("done_reason"))
        return StructuredResult(validated, prompt, round((time.perf_counter() - t0) * 1000, 3), usage)


def structured_extractor(settings: V2Settings | None = None) -> StructuredExtractor:
    s = settings or V2Settings.from_env()
    return StructuredExtractor(OllamaChatClient(s.atomizer_model, s.ollama_host, s.num_ctx, s.num_predict))
