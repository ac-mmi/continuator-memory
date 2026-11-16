"""Structured extraction v1.1: every target message gets an accountable outcome. Opt-in, like Phase 1.

Windows are unchanged: ``build_windows`` makes each message a target in exactly one window and shows some
earlier messages as context. What changes is the contract. The schema has one required slot per target
(``T1``, ``T2``, ...), so a target cannot be skipped and no slot can repeat. Records are filed under the
target that owns them. Context messages (``C1``, ...) can only be cited alongside a target's own quote, for
example the question that "Yes" answers. The model never sees global message IDs.

Code checks structure, and meaning stays with the model. This module deliberately has no word overlap,
stemming, or keyword or regex reading of natural language. What it can establish:
  - the JSON has the required shape and the labels are valid;
  - quotes occur verbatim in the messages they cite;
  - a stated name occurs literally in the window or in earlier memories;
  - fields do not contradict each other, such as a historical date after the message was sent.
A verified quote proves where a claim came from, not that the claim follows from it. Every record therefore
carries ``semantic_support: "not_verified"``, and ``evidence_verified`` only ever means the quotes exist.

Statuses of a kept record:
  evidence_verified   quotes exist and nothing below applies. The meaning is the model's reading.
  uncertain           the model said it was unsure, or a field is incomplete; reasons are recorded.
  needs_cross_window  a reference needs more than this window to resolve, or the record was recovered from
                      a context message whose own window produced nothing.
  conflicting_fields  the record's own fields contradict each other.
Structurally invalid records are rejected, with the claim text kept in the issue.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import (
    DEFAULT_PRIOR_LIMIT,
    AtomizationRequestError,
    JSONCompletionClient,
    OllamaChatClient,
    V2Settings,
)
from continuator_memory.v2.structured_extraction import (
    CATEGORIES,
    CATEGORY_UNCLASSIFIED,
    ENTITY_TYPES,
    KIND_UNSPECIFIED,
    MAX_LABEL_CHARS,
    TEMPORAL_KINDS,
    Evidence,
    EntityMention,
    StructuredTemporal,
    _DATE,
    _LABEL,
    _normal,
    _speaker_of,
    _text,
    window_ref,
)
from continuator_memory.v2.structured_store import StructuredIngestReport, StructuredStore, prepare_conversation
from continuator_memory.v2.validation import FLAGGED, REJECTED, ValidationIssue
from continuator_memory.v2.windowing import (
    DEFAULT_CONTEXT_BEFORE,
    DEFAULT_TARGETS_PER_WINDOW,
    ContextWindow,
    build_windows,
)

TARGETS_PROTOCOL_VERSION = "structured-v1.1.1"

STATUS_EVIDENCE_VERIFIED = "evidence_verified"
STATUS_UNCERTAIN = "uncertain"
STATUS_NEEDS_CROSS_WINDOW = "needs_cross_window"
STATUS_CONFLICTING = "conflicting_fields"
SEMANTIC_NOT_VERIFIED = "not_verified"

OUTCOME_EXTRACTED = "extracted"
OUTCOME_NOTHING_DURABLE = "nothing_durable"
OUTCOME_QUESTION = "question"
OUTCOME_UNCLEAR = "unclear"
MODEL_OUTCOMES = (OUTCOME_EXTRACTED, OUTCOME_NOTHING_DURABLE, OUTCOME_QUESTION, OUTCOME_UNCLEAR)
OUTCOME_NOT_RETURNED = "not_returned"  # set by code: parseable response with no usable slot for this target
OUTCOME_UNPROCESSED = "unprocessed"  # set by code: the window failed, so nothing is known about this target

# A question is an outcome, not a statement type: a bare question does not become a memory.
STATEMENT_TYPES = ("assertion", "confirmation", "correction", "reported", "suggestion", "hypothetical")
STATEMENT_UNCLASSIFIED = "unclassified"
REFERENCE_STATED = "stated"
REFERENCE_RESOLVED = "resolved_from_context"
REFERENCE_UNRESOLVED = "unresolved"
REFERENCE_NOT_APPLICABLE = "not_applicable"
SUBJECT_STATUSES = (REFERENCE_STATED, REFERENCE_RESOLVED, REFERENCE_UNRESOLVED)
OBJECT_STATUSES = (*SUBJECT_STATUSES, REFERENCE_NOT_APPLICABLE)

# The contract as code. The model may omit optional fields; ``normalize_response`` fills them in, so the rest of
# the application always sees the complete structure. Any other key rejects the object that carries it.
_LIST_DEFAULT = "list"
_SHARED_REQUIRED = {"statement_type": "", "time_kind": "", "quote": ""}
_SHARED_OPTIONAL = {"context": _LIST_DEFAULT, "uncertain": "", "time_expression": "", "time_value": None}
MEMORY_FIELDS: dict[str, Any] = {
    "statement": "", "subject": "", "subject_status": "", "predicate": "", "object": "", "object_status": "",
    "category": "", **_SHARED_REQUIRED, "reason": "", "reason_quote": "", **_SHARED_OPTIONAL}
RELATIONSHIP_FIELDS: dict[str, Any] = {"subject": "", "type": "", "object": "", **_SHARED_REQUIRED, **_SHARED_OPTIONAL}
SLOT_FIELDS: dict[str, Any] = {"outcome": "", "note": "", "memories": _LIST_DEFAULT, "relationships": _LIST_DEFAULT}
ENTITY_FIELDS: dict[str, Any] = {"name": "", "type": "", "aliases": _LIST_DEFAULT}
CITATION_FIELDS: dict[str, Any] = {"ref": "", "quote": ""}
RESPONSE_FIELDS = ("entities", "targets")

# Largest v1 output was 1,751 tokens of 2,048. An env value still wins. Truncation fails explicitly.
DEFAULT_TARGETS_NUM_PREDICT = 4096

TARGETS_SYSTEM_PROMPT = """\
You extract long-term memory from a conversation. Read it the way a careful person would, then record what \
each TARGET message means. T1, T2, ... are TARGET messages. C1, C2, ... are CONTEXT: earlier messages, \
already processed, shown only so you can understand the targets. KNOWN lists earlier memories, also only \
for understanding references.

Return every target under "targets", with an outcome:
extracted (it has memories or relationships), nothing_durable (greetings, thanks, reactions, small talk), \
question (it only asks something) or unclear (you cannot tell what it means; say why in "note").
A question does not establish anything. Never record what a question assumes. If a later target answers it, \
record the fact under the answer and cite the question in "context".

memories: each is one fact that makes sense alone. Split independent facts. Do not repeat a fact. Write \
"statement" as a plain sentence, naming people instead of I/me/you. Do not assume anyone's gender.
quote: words copied exactly from this target message, as short as still supports the fact.
context: only when a context or other target message is needed to understand this one: {"ref", "quote"}, \
copied exactly.
statement_type: assertion (the speaker states it), confirmation (the speaker confirms what another person \
said or asked; cite that message in context), correction (the speaker corrects something said earlier), \
reported (the speaker passes on something about someone else, which they may not know first-hand), \
suggestion, or hypothetical (a possibility, not a fact). A suggestion is not a decision. A plan is not a \
completed event. Keep negation and uncertainty in the statement.
category: fact, decision, preference, goal, event, plan or condition.
subject, predicate, object: the fact as a triple. predicate is a whole verb phrase ("prefers", "target date \
is"). The subject or object may be an event or proposition ("the beta launch") when no person or thing fits.
subject_status / object_status: stated (the message names it), resolved_from_context (you worked out what \
"it" or "they" means from the conversation), unresolved (you cannot tell; leave the field empty). \
object_status may also be not_applicable, when the fact has no object. Never invent a value to fill a field.
reason: why, only when the speaker gives a reason, with reason_quote copied exactly.
uncertain: say why here if you are not sure of your reading. Otherwise leave it out.
time_kind: decide this separately for each record, from what that record says, never from where the message \
is in the conversation: historical (it was so or happened), current (the speaker says it holds now), future \
(a plan or expectation), conditional (it depends on something), correction (replaces something earlier), or \
unspecified. A message's date is when it was said, not when the fact applies. You see only part of the \
conversation, so never say that a fact is outdated or the latest.
time_expression: the time words copied exactly. time_value: YYYY, YYYY-MM or YYYY-MM-DD only if the words and \
message date make it certain. Leave both out if there are none or if it is ambiguous.

relationships: only links a target message itself states between two named things, with the same quote, \
statement_type and time fields. type is a short snake_case verb (works_at, develops, formerly_named, \
depends_on, postponed_to, replaced_by, decided_against). Never link things because they appear together, and \
do not restate a memory as a relationship unless it links two named things.

entities: each named person, project, company, organization, product, place, event or concept once for this \
window, with name exactly as written and aliases only if a message states another name. Do not decide that \
different names are the same thing.

Use only the keys described here. Include reason, context, uncertain, time_expression, time_value and \
relationships whenever the conversation supplies them. Omit one only when there is nothing meaningful to put \
in it; never fill a field with an empty or placeholder value. Empty lists are valid.
"""

_EXAMPLE_PROMPT = """\
KNOWN (earlier memories, for references only):
(none)

CONTEXT (do not extract):
[C1] user @ 2024-03-05: Ines: Rafi, are you still at Harbour Print?

TARGETS:
[T1] assistant @ 2024-03-05: Rafi: Yes, but I might leave in June. I chose Tallow over Ember because it is cheaper.
[T2] user @ 2024-03-05: Ines: Nice! Actually it's Thursday I swim, not Tuesday. Do you still cycle?

JSON:"""

_EXAMPLE_OUTPUT = {
    "entities": [{"name": "Rafi", "type": "person"}, {"name": "Harbour Print", "type": "company"},
                 {"name": "Tallow", "type": "product"}, {"name": "Ember", "type": "product"},
                 {"name": "Ines", "type": "person"}],
    "targets": {
        "T1": {"outcome": "extracted", "memories": [
            {"statement": "Rafi is at Harbour Print.", "subject": "Rafi", "subject_status": "stated",
             "predicate": "is at", "object": "Harbour Print", "object_status": "stated", "category": "fact",
             "statement_type": "confirmation", "time_kind": "current", "quote": "Yes",
             "context": [{"ref": "C1", "quote": "are you still at Harbour Print?"}]},
            {"statement": "Rafi might leave Harbour Print in June.", "subject": "Rafi", "subject_status": "stated",
             "predicate": "might leave", "object": "Harbour Print", "object_status": "resolved_from_context",
             "category": "condition", "statement_type": "hypothetical", "time_kind": "conditional",
             "time_expression": "in June", "time_value": "2024-06", "quote": "I might leave in June",
             "context": [{"ref": "C1", "quote": "are you still at Harbour Print?"}]},
            {"statement": "Rafi chose Tallow over Ember.", "subject": "Rafi", "subject_status": "stated",
             "predicate": "chose", "object": "Tallow", "object_status": "stated", "category": "decision",
             "statement_type": "assertion", "time_kind": "historical", "reason": "It is cheaper.",
             "reason_quote": "because it is cheaper", "quote": "I chose Tallow over Ember"}],
            "relationships": [
            {"subject": "Rafi", "type": "works_at", "object": "Harbour Print", "statement_type": "confirmation",
             "time_kind": "current", "quote": "Yes", "context": [{"ref": "C1", "quote": "are you still at Harbour Print?"}]}]},
        "T2": {"outcome": "extracted", "memories": [
            {"statement": "Ines swims on Thursday, not Tuesday.", "subject": "Ines", "subject_status": "stated",
             "predicate": "swims on", "object": "Thursday", "object_status": "stated", "category": "fact",
             "statement_type": "correction", "time_kind": "correction", "quote": "it's Thursday I swim, not Tuesday"}]},
    },
}
TARGETS_EXAMPLES: tuple[tuple[str, str], ...] = ((_EXAMPLE_PROMPT, json.dumps(_EXAMPLE_OUTPUT)),)


@dataclass(frozen=True)
class TargetOutcome:
    message_id: str
    outcome: str
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"message_id": self.message_id, "outcome": self.outcome, "note": self.note}


def _status(reasons: dict[str, list[str]]) -> str:
    if reasons["conflicting"]:
        return STATUS_CONFLICTING
    if reasons["cross_window"]:
        return STATUS_NEEDS_CROSS_WINDOW
    return STATUS_UNCERTAIN if reasons["uncertain"] else STATUS_EVIDENCE_VERIFIED


def _new_reasons() -> dict[str, list[str]]:
    return {"conflicting": [], "cross_window": [], "uncertain": []}


def _all_reasons(reasons: dict[str, list[str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(r for group in reasons.values() for r in group))


@dataclass(frozen=True)
class TargetMemory:
    memory_id: str
    message_id: str
    statement: str
    subject: str | None
    subject_status: str
    predicate: str
    object: str | None
    object_status: str
    category: str
    statement_type: str
    reason: str | None
    temporal: StructuredTemporal
    evidence: tuple[Evidence, ...]
    speaker: str
    speaker_role: str
    status: str
    review_reasons: tuple[str, ...] = ()
    model_uncertainty: str | None = None
    reason_quote: str | None = None

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.message_id for e in self.evidence))

    @property
    def needs_review(self) -> bool:
        return self.status != STATUS_EVIDENCE_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {"reason_quote": self.reason_quote, "memory_id": self.memory_id, "message_id": self.message_id, "statement": self.statement,
                "subject": self.subject, "subject_status": self.subject_status, "predicate": self.predicate,
                "object": self.object, "object_status": self.object_status, "category": self.category,
                "statement_type": self.statement_type, "reason": self.reason, "temporal": self.temporal.to_dict(),
                "evidence": [e.to_dict() for e in self.evidence], "message_ids": list(self.message_ids),
                "speaker": self.speaker, "speaker_role": self.speaker_role, "status": self.status,
                "review_reasons": list(self.review_reasons), "model_uncertainty": self.model_uncertainty,
                "needs_review": self.needs_review, "semantic_support": SEMANTIC_NOT_VERIFIED}


@dataclass(frozen=True)
class TargetRelationship:
    relationship_id: str
    message_id: str
    subject: str
    type: str
    object: str
    statement_type: str
    temporal: StructuredTemporal
    evidence: tuple[Evidence, ...]
    speaker: str
    speaker_role: str
    status: str
    review_reasons: tuple[str, ...] = ()
    model_uncertainty: str | None = None

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(e.message_id for e in self.evidence))

    @property
    def needs_review(self) -> bool:
        return self.status != STATUS_EVIDENCE_VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {"relationship_id": self.relationship_id, "message_id": self.message_id, "subject": self.subject,
                "type": self.type, "object": self.object, "statement_type": self.statement_type,
                "temporal": self.temporal.to_dict(), "evidence": [e.to_dict() for e in self.evidence],
                "message_ids": list(self.message_ids), "speaker": self.speaker, "speaker_role": self.speaker_role,
                "status": self.status, "review_reasons": list(self.review_reasons),
                "model_uncertainty": self.model_uncertainty, "needs_review": self.needs_review,
                "semantic_support": SEMANTIC_NOT_VERIFIED}


@dataclass
class TargetWindow:
    window: ContextWindow
    outcomes: list[TargetOutcome] = field(default_factory=list)
    entities: list[EntityMention] = field(default_factory=list)
    memories: list[TargetMemory] = field(default_factory=list)
    relationships: list[TargetRelationship] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    raw: str | None = None
    error: str | None = None
    normalized: dict[str, Any] | None = None  # the response after ``normalize_response``

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        return "invalid" if any(i.action == REJECTED for i in self.issues) else "ok"

    @property
    def unaccounted_targets(self) -> tuple[str, ...]:
        return tuple(o.message_id for o in self.outcomes if o.outcome in (OUTCOME_NOT_RETURNED, OUTCOME_UNPROCESSED))


@dataclass
class TargetResult:
    validated: TargetWindow
    prompt: str
    latency_ms: float
    usage: dict[str, Any] | None = None


def labels(window: ContextWindow) -> dict[str, Message]:
    """Local label -> message. The model only ever sees these labels."""
    out = {f"T{i}": m for i, m in enumerate(window.targets, 1)}
    out.update({f"C{i}": m for i, m in enumerate(window.context, 1)})
    return out


def _labelled_line(label: str, message: Message) -> str:
    stamp = f" @ {message.timestamp}" if message.timestamp else ""
    return f"[{label}] {message.role}{stamp}: {message.content}"


def build_targets_prompt(window: ContextWindow, prior: Sequence[TargetMemory] = ()) -> str:
    named = labels(window)
    known = "\n".join(f"- {m.statement}" for m in prior) or "(none)"
    context = "\n".join(_labelled_line(k, m) for k, m in named.items() if k.startswith("C")) or "(none)"
    targets = "\n".join(_labelled_line(k, m) for k, m in named.items() if k.startswith("T"))
    return (f"KNOWN (earlier memories, for references only):\n{known}\n\nCONTEXT (do not extract):\n{context}\n\n"
            f"TARGETS:\n{targets}\n\nJSON:")


def targets_output_schema(window: ContextWindow) -> dict[str, Any]:
    text = {"type": "string"}
    named = labels(window)
    cite = {"type": "array", "items": {"type": "object", "required": ["ref", "quote"], "additionalProperties": False,
                                       "properties": {"ref": {"type": "string", "enum": list(named)}, "quote": text}}}
    shared = {"quote": text, "context": cite, "uncertain": text,
              "statement_type": {"type": "string", "enum": list(STATEMENT_TYPES)},
              "time_kind": {"type": "string", "enum": list(TEMPORAL_KINDS)},
              "time_expression": text, "time_value": text}
    memory = {"type": "object", "additionalProperties": False,
              "required": ["statement", "subject", "subject_status", "predicate", "object", "object_status", "category",
                           "statement_type", "time_kind", "quote"],
              "properties": {"statement": text, "subject": text, "predicate": text, "object": text, "reason": text,
                             "subject_status": {"type": "string", "enum": list(SUBJECT_STATUSES)},
                             "object_status": {"type": "string", "enum": list(OBJECT_STATUSES)},
                             "category": {"type": "string", "enum": list(CATEGORIES)},
                             "reason_quote": text, **shared}}
    relationship = {"type": "object", "additionalProperties": False,
                    "required": ["subject", "type", "object", "statement_type", "time_kind", "quote"],
                    "properties": {"subject": text, "type": text, "object": text, **shared}}
    slot = {"type": "object", "required": ["outcome"], "additionalProperties": False,
            "properties": {"outcome": {"type": "string", "enum": list(MODEL_OUTCOMES)}, "note": text,
                           "memories": {"type": "array", "items": memory},
                           "relationships": {"type": "array", "items": relationship}}}
    targets = [k for k in named if k.startswith("T")]
    entity = {"type": "object", "required": ["name", "type"], "additionalProperties": False,
              "properties": {"name": text, "type": text, "aliases": {"type": "array", "items": text}}}
    return {"type": "object", "required": ["entities", "targets"], "additionalProperties": False,
            "properties": {"entities": {"type": "array", "items": entity},
                           "targets": {"type": "object", "required": targets, "additionalProperties": False,
                                       "properties": {k: slot for k in targets}}}}


def _normalize_object(item: dict[str, Any], fields: dict[str, Any], ref: str,
                      issues: list[ValidationIssue]) -> dict[str, Any] | None:
    """A complete copy of ``item``: returned values are kept, absent or null ones get their default.

    Returns None, after recording why, if it has a key outside ``fields`` or a value of the wrong type.
    """
    described = _text(item.get("statement")) or _text(item.get("name")) or _text(item.get("subject")) or ref
    unexpected = sorted(k for k in item if k not in fields)
    if unexpected:
        issues.append(ValidationIssue("UNEXPECTED_KEY", REJECTED, f"{described!r}: unexpected keys {unexpected}", None, ref))
        return None
    out: dict[str, Any] = {}
    for key, default in fields.items():
        value = item.get(key)
        if value is None:
            out[key] = [] if default == _LIST_DEFAULT else default
            continue
        if not isinstance(value, list if default == _LIST_DEFAULT else str):
            issues.append(ValidationIssue("MALFORMED_FIELD", REJECTED,
                                          f"{described!r}: {key} has type {type(value).__name__}", None, ref))
            return None
        out[key] = value
    return out


def _normalize_cited(item: dict[str, Any], fields: dict[str, Any], ref: str,
                     issues: list[ValidationIssue]) -> dict[str, Any] | None:
    out = _normalize_object(item, fields, ref, issues)
    if out is None:
        return None
    citations = []
    for i, cite in enumerate(out["context"]):
        citation = _normalize_object(cite, CITATION_FIELDS, f"{ref}.context[{i}]", issues) if isinstance(cite, dict) else None
        if citation is None:
            if not isinstance(cite, dict):
                issues.append(ValidationIssue("MALFORMED_FIELD", REJECTED, f"context[{i}] is not an object", None, ref))
            return None
        citations.append(citation)
    out["context"] = citations
    return out


def _normalize_list(items: list[Any], normalize: Any, ref: str) -> list[Any]:
    """Non-objects pass through, so the record checks report them as before; rejected objects are removed."""
    kept = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            kept.append(item)
            continue
        record = normalize(item, f"{ref}[{i}]")
        if record is not None:
            kept.append(record)
    return kept


def normalize_response(data: dict[str, Any], issues: list[ValidationIssue]) -> dict[str, Any] | None:
    """The complete application-level structure, or None if the top level itself breaks the contract."""
    unexpected = sorted(k for k in data if k not in RESPONSE_FIELDS)
    if unexpected:
        issues.append(ValidationIssue("UNEXPECTED_KEY", REJECTED, f"unexpected top-level keys {unexpected}", None, "response"))
        return None
    slots: dict[str, Any] = {}
    for label, raw_slot in data["targets"].items():
        ref = f"targets.{label}"
        slot = _normalize_object(raw_slot, SLOT_FIELDS, ref, issues) if isinstance(raw_slot, dict) else None
        if slot is not None:
            slot["memories"] = _normalize_list(slot["memories"], lambda it, r: _normalize_cited(it, MEMORY_FIELDS, r, issues),
                                               f"{ref}.memories")
            slot["relationships"] = _normalize_list(slot["relationships"], lambda it, r: _normalize_cited(it, RELATIONSHIP_FIELDS, r, issues),
                                                    f"{ref}.relationships")
        slots[label] = slot
    entities = _normalize_list(data["entities"], lambda it, r: _normalize_object(it, ENTITY_FIELDS, r, issues), "entities")
    return {"entities": entities, "targets": slots}


class _Checker:
    """Structural checks for one window."""

    def __init__(self, window: ContextWindow, prior: Sequence[TargetMemory], covered: Collection[str],
                 issues: list[ValidationIssue]):
        self.window, self.covered, self.issues = window, set(covered), issues
        self.named = labels(window)
        self.targets = set(window.target_ids)
        self.by_id = {m.message_id: m for m in window.messages}
        self.window_text = _normal(" ".join(m.content for m in window.messages))
        self.known_text = _normal(" ".join(" ".join(filter(None, (m.statement, m.subject, m.object))) for m in prior))

    def issue(self, code: str, action: str, detail: str, ref: str, message_id: str | None = None) -> None:
        self.issues.append(ValidationIssue(code, action, detail, message_id, ref))

    def in_message(self, quote: str, message: Message) -> bool:
        return bool(quote) and _normal(quote) in _normal(message.content)

    def locate_owner(self, quote: str, filed_under: Message) -> Message | None:
        """The message holding the record's quote: where it was filed, else any other message of the window."""
        if self.in_message(quote, filed_under):
            return filed_under
        return next((m for m in (*self.window.targets, *self.window.context) if self.in_message(quote, m)), None)

    def speakers_of(self, message: Message) -> tuple[str, str]:
        return _speaker_of(message), message.role

    def literal_in_window(self, name: str, owner: Message) -> bool:
        wanted = _normal(name)
        role_words = {owner.role, f"the {owner.role}", _normal(_speaker_of(owner))}
        return wanted in role_words or wanted in self.window_text or wanted in self.known_text

    def temporal(self, item: dict[str, Any], evidence: Sequence[Evidence], owner: Message, ref: str,
                 reasons: dict[str, list[str]]) -> StructuredTemporal:
        kind = item.get("time_kind")
        if kind not in TEMPORAL_KINDS:
            self.issue("INVALID_TEMPORAL_KIND", FLAGGED, f"{kind!r} not in {list(TEMPORAL_KINDS)}", ref)
            reasons["uncertain"].append("INVALID_TEMPORAL_KIND")
            kind = KIND_UNSPECIFIED
        expression, value = _text(item.get("time_expression")), _text(item.get("time_value"))
        if expression and not any(self.in_message(expression, self.by_id[e.message_id]) for e in evidence):
            self.issue("TEMPORAL_NOT_IN_SOURCE", FLAGGED, f"{expression!r} is not in the cited messages; dropped", ref)
            reasons["uncertain"].append("TEMPORAL_NOT_IN_SOURCE")
            expression = ""
        if value and not expression:
            self.issue("TEMPORAL_UNSUPPORTED", FLAGGED, f"{value!r} has no stated expression; dropped", ref)
            reasons["uncertain"].append("TEMPORAL_UNSUPPORTED")
            value = ""
        if value:
            match = _DATE.match(value)
            valid = bool(match)
            if match:
                try:
                    date(int(match.group(1)), int(match.group(2) or 1), int(match.group(3) or 1))
                except ValueError:
                    valid = False
            if not valid:
                self.issue("MALFORMED_TEMPORAL", FLAGGED, f"{value!r} is not YYYY, YYYY-MM or YYYY-MM-DD; dropped", ref)
                reasons["uncertain"].append("MALFORMED_TEMPORAL")
                value = ""
            elif not owner.timestamp and match.group(1) not in expression:
                self.issue("RESOLVED_WITHOUT_TIMESTAMP", FLAGGED, f"{value!r} for {expression!r} with no message date; dropped", ref)
                reasons["uncertain"].append("RESOLVED_WITHOUT_TIMESTAMP")
                value = ""
            elif owner.timestamp:
                said = owner.timestamp[: len(value)]
                if (kind == "historical" and value > said) or (kind == "future" and value < said):
                    self.issue("TEMPORAL_CONFLICT", FLAGGED, f"{kind} but {value!r} against message date {owner.timestamp}", ref)
                    reasons["conflicting"].append("TEMPORAL_CONFLICT")
        return StructuredTemporal(expression or None, value or None, kind)

    def statement_type(self, raw: Any, ref: str, reasons: dict[str, list[str]]) -> str:
        if raw in STATEMENT_TYPES:
            return raw
        self.issue("INVALID_STATEMENT_TYPE", FLAGGED, f"{raw!r} not in {list(STATEMENT_TYPES)}", ref)
        reasons["uncertain"].append("INVALID_STATEMENT_TYPE")
        return STATEMENT_UNCLASSIFIED

    def evidence(self, item: dict[str, Any], filed_under: Message, description: str, ref: str,
                 reasons: dict[str, list[str]]) -> tuple[Message, tuple[Evidence, ...]] | None:
        """(owner, evidence), or None after recording why the record cannot be kept."""
        quote = _text(item.get("quote"))
        if not quote:
            self.issue("MISSING_EVIDENCE", REJECTED, f"{description!r} has no quote", ref, filed_under.message_id)
            return None
        owner = self.locate_owner(quote, filed_under)
        if owner is None:
            self.issue("QUOTE_NOT_IN_SOURCE", REJECTED, f"{description!r}: quote {quote!r} is in no message of this window",
                       ref, filed_under.message_id)
            return None
        if owner.message_id not in self.targets:
            if owner.message_id in self.covered:
                self.issue("CONTEXT_REPEAT_IGNORED", FLAGGED, f"{description!r} repeats {owner.message_id}, already extracted",
                           ref, owner.message_id)
                return None
            self.issue("RECOVERED_FROM_CONTEXT", FLAGGED, f"{description!r}: {owner.message_id} produced nothing in its own window",
                       ref, owner.message_id)
            reasons["cross_window"].append("RECOVERED_FROM_CONTEXT")
        elif owner is not filed_under:
            self.issue("FILED_UNDER_WRONG_TARGET", FLAGGED, f"{description!r} moved to {owner.message_id}", ref, owner.message_id)
        cited = [Evidence(owner.message_id, quote)]
        raw_context = item.get("context")
        for extra in raw_context if isinstance(raw_context, list) else []:
            label = extra.get("ref") if isinstance(extra, dict) else None
            text = _text(extra.get("quote")) if isinstance(extra, dict) else ""
            message = self.named.get(label) if isinstance(label, str) else None
            if message is None:
                self.issue("INVALID_EVIDENCE_REF", REJECTED, f"{description!r}: {label!r} is not a label in this window", ref)
                return None
            if not self.in_message(text, message):
                self.issue("QUOTE_NOT_IN_SOURCE", REJECTED, f"{description!r}: quote {text!r} is not in {label}", ref,
                           message.message_id)
                return None
            cited.append(Evidence(message.message_id, text))
        return owner, tuple(dict.fromkeys(cited))

    def confirmation_prompt(self, statement_type: str, evidence: Sequence[Evidence], owner: Message, ref: str,
                            reasons: dict[str, list[str]]) -> None:
        if statement_type == "confirmation" and not any(
                _speaker_of(self.by_id[e.message_id]) != _speaker_of(owner) for e in evidence):
            self.issue("CONFIRMATION_WITHOUT_PROMPT", FLAGGED, "no message from another speaker is cited", ref)
            reasons["uncertain"].append("CONFIRMATION_WITHOUT_PROMPT")

    def model_uncertainty(self, item: dict[str, Any], reasons: dict[str, list[str]]) -> str | None:
        note = _text(item.get("uncertain")) or None
        if note:
            reasons["uncertain"].append("MODEL_UNCERTAIN")
        return note

    def reference(self, value: str, status: Any, allowed: tuple[str, ...], label: str, owner: Message, ref: str,
                  reasons: dict[str, list[str]]) -> tuple[str | None, str]:
        """Keep a subject or object as given. Missing values stay missing and explicit; nothing is filled in."""
        code = label.upper()
        if status not in allowed:
            self.issue(f"INVALID_{code}_STATUS", FLAGGED, f"{status!r} not in {list(allowed)}", ref)
            reasons["uncertain"].append(f"INVALID_{code}_STATUS")
            status = REFERENCE_UNRESOLVED
        if status == REFERENCE_UNRESOLVED:
            reasons["cross_window"].append(f"{code}_UNRESOLVED")
        elif status != REFERENCE_NOT_APPLICABLE and not value:
            self.issue(f"{code}_MISSING", FLAGGED, f"{label} is {status} but empty", ref)
            reasons["uncertain"].append(f"{code}_MISSING")
        return value or None, status

    def memory(self, item: Any, filed_under: Message, ref: str, record_id: str) -> TargetMemory | None:
        if not isinstance(item, dict):
            self.issue("INVALID_SCHEMA", REJECTED, "memory is not an object", ref, filed_under.message_id)
            return None
        statement = _text(item.get("statement"))
        if not statement:
            self.issue("EMPTY_STATEMENT", REJECTED, "memory has no statement", ref, filed_under.message_id)
            return None
        reasons = _new_reasons()
        located = self.evidence(item, filed_under, statement, ref, reasons)
        if located is None:
            return None
        owner, evidence = located
        subject, subject_status = self.reference(_text(item.get("subject")), item.get("subject_status"),
                                                 SUBJECT_STATUSES, "subject", owner, ref, reasons)
        if subject and subject_status != REFERENCE_UNRESOLVED and not self.literal_in_window(subject, owner):
            self.issue("REFERENCE_NOT_IN_WINDOW", FLAGGED, f"subject {subject!r} is not named in this window", ref)
            reasons["cross_window"].append("REFERENCE_NOT_IN_WINDOW")
        obj, object_status = self.reference(_text(item.get("object")), item.get("object_status"), OBJECT_STATUSES,
                                            "object", owner, ref, reasons)
        predicate = _text(item.get("predicate"))
        if not predicate or not predicate[0].isalnum():
            self.issue("INCOMPLETE_PREDICATE", FLAGGED, f"predicate {predicate!r} is not a whole phrase", ref)
            reasons["uncertain"].append("INCOMPLETE_PREDICATE")
        category = item.get("category")
        if category not in CATEGORIES:
            self.issue("INVALID_CATEGORY", FLAGGED, f"{category!r} not in {list(CATEGORIES)}", ref)
            reasons["uncertain"].append("INVALID_CATEGORY")
            category = CATEGORY_UNCLASSIFIED
        statement_type = self.statement_type(item.get("statement_type"), ref, reasons)
        self.confirmation_prompt(statement_type, evidence, owner, ref, reasons)
        reason = _text(item.get("reason")) or None
        reason_quote = _text(item.get("reason_quote"))
        if reason:
            if not reason_quote:
                self.issue("REASON_WITHOUT_EVIDENCE", FLAGGED, f"reason {reason!r} has no quote", ref)
                reasons["uncertain"].append("REASON_WITHOUT_EVIDENCE")
            elif not any(self.in_message(reason_quote, self.by_id[e.message_id]) for e in evidence):
                self.issue("REASON_QUOTE_NOT_IN_SOURCE", FLAGGED, f"quote {reason_quote!r} for {reason!r} is not cited; reason dropped", ref)
                reasons["uncertain"].append("REASON_QUOTE_NOT_IN_SOURCE")
                reason = None
                reason_quote = ""
        temporal = self.temporal(item, evidence, owner, ref, reasons)
        note = self.model_uncertainty(item, reasons)
        speaker, role = self.speakers_of(owner)
        return TargetMemory(record_id, owner.message_id, statement, subject, subject_status, predicate, obj, object_status,
                            category, statement_type, reason, temporal, evidence, speaker, role, _status(reasons),
                            _all_reasons(reasons), note, reason_quote or None)

    def relationship(self, item: Any, filed_under: Message, ref: str, record_id: str) -> TargetRelationship | None:
        if not isinstance(item, dict):
            self.issue("INVALID_SCHEMA", REJECTED, "relationship is not an object", ref, filed_under.message_id)
            return None
        subject, obj = _text(item.get("subject")), _text(item.get("object"))
        rtype = _text(item.get("type")).lower().replace(" ", "_").replace("-", "_")
        described = f"{subject} {rtype} {obj}"
        if not subject or not obj:
            self.issue("EMPTY_RELATIONSHIP", REJECTED, f"{described!r} needs a subject and an object", ref, filed_under.message_id)
            return None
        if not _LABEL.match(rtype) or len(rtype) > MAX_LABEL_CHARS:
            self.issue("INVALID_RELATIONSHIP_TYPE", REJECTED, f"type {item.get('type')!r}", ref, filed_under.message_id)
            return None
        reasons = _new_reasons()
        located = self.evidence(item, filed_under, described, ref, reasons)
        if located is None:
            return None
        owner, evidence = located
        missing = [e for e in (subject, obj) if not self.literal_in_window(e, owner)]
        if missing:
            self.issue("REFERENCE_NOT_IN_WINDOW", FLAGGED, f"{described!r}: {missing!r} not named in this window", ref)
            reasons["cross_window"].append("REFERENCE_NOT_IN_WINDOW")
        statement_type = self.statement_type(item.get("statement_type"), ref, reasons)
        self.confirmation_prompt(statement_type, evidence, owner, ref, reasons)
        temporal = self.temporal(item, evidence, owner, ref, reasons)
        note = self.model_uncertainty(item, reasons)
        speaker, role = self.speakers_of(owner)
        return TargetRelationship(record_id, owner.message_id, subject, rtype, obj, statement_type, temporal, evidence,
                                  speaker, role, _status(reasons), _all_reasons(reasons), note)

    def entities(self, raw: Any, ref: str) -> list[EntityMention]:
        out: dict[str, EntityMention] = {}
        for i, item in enumerate(raw):
            iref = f"entities[{i}]"
            name = _text(item.get("name")) if isinstance(item, dict) else ""
            if not name:
                self.issue("EMPTY_ENTITY", REJECTED, "entity has no name", iref)
                continue
            key = _normal(name)
            if key not in self.window_text and key not in self.known_text:
                self.issue("ENTITY_NOT_IN_WINDOW", REJECTED, f"entity {name!r} is not named in this window", iref)
                continue
            etype = _text(item.get("type")).lower().replace(" ", "_")
            if not _LABEL.match(etype) or len(etype) > MAX_LABEL_CHARS:
                self.issue("INVALID_ENTITY_TYPE", FLAGGED, f"type {item.get('type')!r}; kept as concept", iref)
                etype = "concept"
            elif etype not in ENTITY_TYPES:
                self.issue("UNKNOWN_ENTITY_TYPE", FLAGGED, f"type {etype!r} kept as stated", iref)
            if key in out:
                self.issue("DUPLICATE_ENTITY", FLAGGED, f"{name!r} repeated; first kept", iref)
                continue
            raw_aliases = item.get("aliases")
            aliases = []
            for alias in raw_aliases if isinstance(raw_aliases, list) else []:
                text = _text(alias)
                if text and _normal(text) != key:
                    if _normal(text) in self.window_text:
                        aliases.append(text)
                    else:
                        self.issue("UNSUPPORTED_ALIAS", FLAGGED, f"alias {text!r} of {name!r} is not in this window", iref)
            mentions = tuple(m.message_id for m in self.window.targets if key in _normal(m.content))
            speakers = tuple(dict.fromkeys(_speaker_of(self.by_id[m]) for m in mentions))
            out[key] = EntityMention(f"{ref}:e{len(out) + 1}", name, etype, tuple(aliases), mentions, speakers)
        return list(out.values())


def validate_targets(raw: str | None, window: ContextWindow, prior: Sequence[TargetMemory] = (),
                     done_reason: str | None = None, covered: Collection[str] = ()) -> TargetWindow:
    """Structural validation of one response. ``covered``: earlier messages that already produced records."""
    def failed(code: str, error: str) -> TargetWindow:
        outcomes = [TargetOutcome(mid, OUTCOME_UNPROCESSED) for mid in window.target_ids]
        return TargetWindow(window, outcomes, issues=[ValidationIssue(code, REJECTED, error)], raw=raw, error=error)

    if done_reason == "length":
        return failed("TRUNCATED_RESPONSE", "the model hit its output limit (done_reason=length); nothing from this "
                                            "window was kept, and its targets are unprocessed")
    try:
        data = json.loads(raw or "")
    except (json.JSONDecodeError, TypeError) as exc:
        return failed("INVALID_JSON", f"invalid JSON: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("entities"), list) or not isinstance(data.get("targets"), dict):
        return failed("INVALID_SCHEMA", "top level must be an object with an 'entities' list and a 'targets' object")

    issues: list[ValidationIssue] = []
    normalized = normalize_response(data, issues)
    if normalized is None:
        return failed("UNEXPECTED_KEY", issues[0].detail)
    check = _Checker(window, prior, covered, issues)
    ref = window_ref(window)
    entities = check.entities(normalized["entities"], ref)
    slots = normalized["targets"]
    for label in slots:
        if label not in check.named or label.startswith("C"):
            issues.append(ValidationIssue("UNKNOWN_TARGET", REJECTED, f"{label!r} is not a target of this window", None, f"targets.{label}"))
    outcomes: list[TargetOutcome] = []
    memories: list[TargetMemory] = []
    relationships: list[TargetRelationship] = []
    seen_statements: set[str] = set()
    seen_links: set[tuple[str, str, str]] = set()
    for i, message in enumerate(window.targets, 1):
        label = f"T{i}"
        slot = slots.get(label)
        if not isinstance(slot, dict):
            issues.append(ValidationIssue("MISSING_TARGET_OUTCOME", REJECTED, f"no usable outcome for {label}", message.message_id, f"targets.{label}"))
            outcomes.append(TargetOutcome(message.message_id, OUTCOME_NOT_RETURNED))
            continue
        outcome = slot.get("outcome")
        if outcome not in MODEL_OUTCOMES:
            issues.append(ValidationIssue("INVALID_OUTCOME", FLAGGED, f"{outcome!r} not in {list(MODEL_OUTCOMES)}", message.message_id, f"targets.{label}"))
            outcome = OUTCOME_UNCLEAR
        outcomes.append(TargetOutcome(message.message_id, outcome, _text(slot.get("note")) or None))
        kept = 0
        raw_memories = slot.get("memories")
        for j, item in enumerate(raw_memories if isinstance(raw_memories, list) else []):
            iref = f"targets.{label}.memories[{j}]"
            record = check.memory(item, message, iref, f"{ref}:m{len(memories) + 1}")
            if record is None:
                continue
            key = _normal(record.statement)
            if key in seen_statements:
                check.issue("DUPLICATE_MEMORY", FLAGGED, f"{record.statement!r} repeated; first kept", iref, record.message_id)
                continue
            seen_statements.add(key)
            memories.append(record)
            kept += 1
        raw_links = slot.get("relationships")
        for j, item in enumerate(raw_links if isinstance(raw_links, list) else []):
            iref = f"targets.{label}.relationships[{j}]"
            link = check.relationship(item, message, iref, f"{ref}:r{len(relationships) + 1}")
            if link is None:
                continue
            key3 = (_normal(link.subject), link.type, _normal(link.object))
            if key3 in seen_links:
                check.issue("DUPLICATE_RELATIONSHIP", FLAGGED, f"{key3} repeated; first kept", iref, link.message_id)
                continue
            seen_links.add(key3)
            relationships.append(link)
            kept += 1
        if kept and outcome in (OUTCOME_NOTHING_DURABLE, OUTCOME_QUESTION):
            check.issue("OUTCOME_CONFLICT", FLAGGED, f"outcome {outcome} but records were given; records kept", f"targets.{label}", message.message_id)
    return TargetWindow(window, outcomes, entities, memories, relationships, issues, raw, normalized=normalized)


class TargetExtractor:
    """One ``complete_json`` call per window, using the same client protocol as ``SemanticAtomizer``."""

    protocol = TARGETS_PROTOCOL_VERSION

    def __init__(self, client: JSONCompletionClient):
        self.client = client
        self.model_name = client.model_name

    def extract(self, window: ContextWindow, prior: Sequence[TargetMemory] = (), covered: Collection[str] = ()) -> TargetResult:
        prompt = build_targets_prompt(window, prior)
        if hasattr(self.client, "last_usage"):
            self.client.last_usage = None
        t0 = time.perf_counter()
        usage = None
        try:
            raw = self.client.complete_json(TARGETS_SYSTEM_PROMPT, prompt, targets_output_schema(window), TARGETS_EXAMPLES)
        except AtomizationRequestError as exc:
            outcomes = [TargetOutcome(mid, OUTCOME_UNPROCESSED) for mid in window.target_ids]
            validated = TargetWindow(window, outcomes, issues=[ValidationIssue("REQUEST_FAILED", REJECTED, str(exc))], error=str(exc))
        else:
            usage = getattr(self.client, "last_usage", None)
            validated = validate_targets(raw, window, prior, (usage or {}).get("done_reason"), covered)
        return TargetResult(validated, prompt, round((time.perf_counter() - t0) * 1000, 3), usage)


def target_extractor(settings: V2Settings | None = None) -> TargetExtractor:
    s = settings or V2Settings.from_env()
    num_predict = s.num_predict if "CONTINUATOR_V2_NUM_PREDICT" in os.environ else DEFAULT_TARGETS_NUM_PREDICT
    return TargetExtractor(OllamaChatClient(s.atomizer_model, s.ollama_host, s.num_ctx, num_predict))


def ingest_targets(
    conversation: Conversation,
    store: Store,
    extractor: TargetExtractor,
    targets_per_window: int = DEFAULT_TARGETS_PER_WINDOW,
    context_before: int = DEFAULT_CONTEXT_BEFORE,
    prior_limit: int = DEFAULT_PRIOR_LIMIT,
    replace: bool = False,
) -> StructuredIngestReport:
    """One sequential call per window. Each window is saved as it finishes, so partial results survive."""
    structured = prepare_conversation(conversation, store, replace)
    report = StructuredIngestReport(conversation.conversation_id)
    prior: list[TargetMemory] = []
    covered: set[str] = set()
    for window in build_windows(conversation, targets_per_window, context_before):
        window_prior = prior[-prior_limit:] if prior_limit else []
        result = extractor.extract(window, window_prior, covered)
        validated = result.validated
        structured.save_window(validated, extractor.model_name, result.latency_ms,
                               tuple(m.memory_id for m in window_prior), result.usage, extractor.protocol)
        prior.extend(validated.memories)
        covered |= {r.message_id for r in (*validated.memories, *validated.relationships)}
        report.add_window(validated, result.usage, result.latency_ms)
    return report


def unowned_messages(conversation: Conversation, targets_per_window: int = DEFAULT_TARGETS_PER_WINDOW,
                     context_before: int = DEFAULT_CONTEXT_BEFORE) -> list[str]:
    """Messages not a target in exactly one window. Always empty for ``build_windows``; a guard for changes to it."""
    owners = [mid for w in build_windows(conversation, targets_per_window, context_before) for mid in w.target_ids]
    return [m.message_id for m in conversation.messages if owners.count(m.message_id) != 1]
