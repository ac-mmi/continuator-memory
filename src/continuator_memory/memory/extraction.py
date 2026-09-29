"""LLM-based memory extraction with strict validation.

The model proposes candidates; nothing it returns reaches the store without
passing ``parse_extraction`` (schema/type checks) and ``ground`` (the values
must actually occur in the source text).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from continuator_memory.memory.models import (
    ASSERTION_KINDS,
    normalize_name,
    normalize_predicate,
    normalize_value,
)
from continuator_memory.memory.text import content_tokens, predicate_tokens
from continuator_memory.models import Message

MAX_TEXT = 200
NULLISH = frozenset({"", "null", "none", "n/a", "unknown"})


class ExtractionError(ValueError):
    """The model output is not a valid extraction document."""


class ExtractionRequestError(ExtractionError):
    """The extraction model could not be reached or returned no content."""


DIRECT, STATED_EARLIER, CONTEXTUAL = "DIRECT", "STATED_EARLIER", "CONTEXTUAL"
COPIED, UNSUPPORTED = "COPIED", "UNSUPPORTED"


@dataclass(frozen=True)
class ExtractedMemory:
    subject: str
    predicate: str
    value: str
    kind: str = "state"
    previous_value: str | None = None
    valid_from: str | None = None
    # Set by ``ground``: how the value is supported, and the messages supporting it.
    # The first evidence ID is the message the fact is attributed to.
    grounding: str | None = None
    evidence_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnownFact:
    """A current stored fact shown to the extractor as context."""

    subject: str
    predicate: str
    value: str
    source_message_ids: tuple[str, ...] = ()

    @property
    def line(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.value}"

    @property
    def key(self) -> tuple[str, str, str]:
        return (normalize_name(self.subject), normalize_predicate(self.predicate),
                normalize_value(self.value))


@dataclass(frozen=True)
class ExtractedRelationship:
    subject: str
    predicate: str
    object: str
    kind: str = "state"
    previous: str | None = None
    valid_from: str | None = None


@dataclass
class ExtractionResult:
    memories: list[ExtractedMemory] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    grounding: list[dict[str, Any]] = field(default_factory=list)


# ---- validation -------------------------------------------------------------


def _optional(item: dict[str, Any], key: str) -> str | None:
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExtractionError(f"'{key}' must be a string or null")
    value = value.strip()
    return None if value.casefold() in NULLISH else value[:MAX_TEXT]


def _required(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or value.strip().casefold() in NULLISH:
        raise ExtractionError(f"missing or empty '{key}'")
    value = value.strip()
    if len(value) > MAX_TEXT:
        raise ExtractionError(f"'{key}' longer than {MAX_TEXT} characters")
    return value


def _kind(item: dict[str, Any]) -> str:
    kind = item.get("kind", "state")
    if kind is None:
        return "state"
    if not isinstance(kind, str) or kind.strip().lower() not in ASSERTION_KINDS:
        raise ExtractionError(f"invalid kind {kind!r}")
    return kind.strip().lower()


def _parse_memory(item: Any) -> ExtractedMemory:
    if not isinstance(item, dict):
        raise ExtractionError("item is not an object")
    subject, value = _required(item, "subject"), _required(item, "value")
    predicate = _required(item, "predicate")
    if not normalize_predicate(predicate):
        raise ExtractionError("predicate has no usable characters")
    if normalize_name(subject) == normalize_name(value):
        raise ExtractionError("subject and value are identical")
    return ExtractedMemory(
        subject=subject,
        predicate=predicate,
        value=value,
        kind=_kind(item),
        previous_value=_optional(item, "previous_value"),
        valid_from=_optional(item, "valid_from"),
    )


def _parse_relationship(item: Any) -> ExtractedRelationship:
    if not isinstance(item, dict):
        raise ExtractionError("item is not an object")
    subject, obj = _required(item, "subject"), _required(item, "object")
    predicate = _required(item, "predicate")
    if not normalize_predicate(predicate):
        raise ExtractionError("predicate has no usable characters")
    if normalize_name(subject) == normalize_name(obj):
        raise ExtractionError("subject and object are identical")
    return ExtractedRelationship(
        subject=subject,
        predicate=predicate,
        object=obj,
        kind=_kind(item),
        previous=_optional(item, "previous"),
        valid_from=_optional(item, "valid_from"),
    )


def parse_extraction(raw: str) -> ExtractionResult:
    """Parse model output. Raises ``ExtractionError`` if the document itself is invalid;
    individual invalid items are dropped and described in ``rejected``.

    Accepts ``facts`` (subject/predicate/value items, routed to memories or
    relationships later by the formation step) and explicit ``memories`` /
    ``relationships`` lists.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ExtractionError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ExtractionError("top-level JSON value must be an object")
    if not {"facts", "memories", "relationships"} & data.keys():
        raise ExtractionError("object has none of 'facts', 'memories', 'relationships'")
    result = ExtractionResult()
    for key, parser, sink in (
        ("facts", _parse_memory, result.memories),
        ("memories", _parse_memory, result.memories),
        ("relationships", _parse_relationship, result.relationships),
    ):
        items = data.get(key, [])
        if not isinstance(items, list):
            raise ExtractionError(f"'{key}' must be a list")
        for i, item in enumerate(items):
            try:
                sink.append(parser(item))
            except ExtractionError as exc:
                result.rejected.append(f"{key}[{i}]: {exc}")
    return result


def _grounded(text: str, source_tokens: set[str]) -> bool:
    tokens = content_tokens(text)
    return bool(tokens) and bool(tokens & source_tokens)


def _known_tokens(fact: KnownFact) -> set[str]:
    return content_tokens(fact.subject) | predicate_tokens(fact.predicate) | content_tokens(fact.value)


def classify_value(
    value: str, message: Message, recent: Sequence[Message], known: Sequence[KnownFact]
) -> tuple[str | None, tuple[str, ...]]:
    """Return (grounding, evidence message IDs) for a value, or (None, ()) if unsupported.

    With V = value tokens, M = message tokens, R_i = tokens of each recent message and
    K = known-fact tokens, checked in order:
      DIRECT          V ⊆ M                            evidence: the message
      STATED_EARLIER  V ⊆ R_i (earliest such message)  evidence: that message
      CONTEXTUAL      V ⊆ M ∪ R ∪ K and V contains a token of M absent from R ∪ K
                      evidence: the message, then the recent messages / known-fact
                      sources supplying V - M
    """
    v = content_tokens(value)
    if not v:
        return None, ()
    m = content_tokens(message.content)
    if v <= m:
        return DIRECT, (message.message_id,)
    recent_tokens = [(r.message_id, content_tokens(r.content)) for r in recent]
    for message_id, tokens in recent_tokens:
        if v <= tokens:
            return STATED_EARLIER, (message_id,)
    known_tokens = [(k.source_message_ids, _known_tokens(k)) for k in known]
    context: set[str] = set()
    for _, tokens in recent_tokens:
        context |= tokens
    for _, tokens in known_tokens:
        context |= tokens
    if not (v <= m | context and v & (m - context)):
        return None, ()
    inherited = v - m
    evidence = [message.message_id]
    for message_id, tokens in recent_tokens:
        if inherited & tokens and message_id not in evidence:
            evidence.append(message_id)
    for source_ids, tokens in known_tokens:
        if inherited & tokens:
            evidence += [s for s in source_ids if s not in evidence]
    return CONTEXTUAL, tuple(evidence)


def ground(
    result: ExtractionResult,
    message: Message,
    recent: Sequence[Message] = (),
    known: Sequence[KnownFact] = (),
) -> ExtractionResult:
    """Drop items whose values are not supported by the source text.

    Memory values are classified by ``classify_value``. Unsupported values are rejected
    as COPIED when they repeat a current known fact, else as UNSUPPORTED. Subjects and
    relationship endpoints need one content word from the message or its context.
    Unsupported ``previous`` values are cleared rather than rejected.
    """
    message_tokens = content_tokens(message.content)
    context_tokens = set(message_tokens)
    for r in recent:
        context_tokens |= content_tokens(r.content)
    for k in known:
        context_tokens |= content_tokens(k.line)
    known_keys = {k.key for k in known}

    def subject_ok(name: str) -> bool:
        return normalize_name(name) == "user" or _grounded(name, context_tokens)

    out = ExtractionResult(rejected=list(result.rejected))
    for m in result.memories:
        if not subject_ok(m.subject):
            out.rejected.append(f"memory {m.subject!r}: subject not grounded")
            continue
        grounding, evidence = classify_value(m.value, message, recent, known)
        if grounding is None:
            key = (normalize_name(m.subject), normalize_predicate(m.predicate),
                   normalize_value(m.value))
            grounding = COPIED if key in known_keys else UNSUPPORTED
            reason = "copied known fact" if grounding == COPIED else "value not grounded"
            out.rejected.append(f"memory {m.subject!r}.{m.predicate}: {reason}")
        else:
            prev = m.previous_value if m.previous_value and _grounded(
                m.previous_value, context_tokens
            ) else None
            out.memories.append(ExtractedMemory(
                m.subject, m.predicate, m.value, m.kind, prev, m.valid_from,
                grounding, evidence,
            ))
        out.grounding.append({
            "subject": m.subject, "predicate": m.predicate, "value": m.value,
            "grounding": grounding, "evidence": list(evidence),
        })
    for r in result.relationships:
        if not subject_ok(r.subject):
            out.rejected.append(f"relationship {r.subject!r}: subject not grounded")
        elif not (normalize_name(r.object) == "user" or _grounded(r.object, context_tokens)):
            out.rejected.append(f"relationship {r.subject!r}.{r.predicate}: object not grounded")
        elif not (_grounded(r.subject, message_tokens) or _grounded(r.object, message_tokens)):
            out.rejected.append(f"relationship {r.subject!r}.{r.predicate}: not in message")
        else:
            prev = r.previous if r.previous and _grounded(r.previous, context_tokens) else None
            out.relationships.append(
                ExtractedRelationship(r.subject, r.predicate, r.object, r.kind, prev, r.valid_from)
            )
    return out


# ---- prompting --------------------------------------------------------------

_STR = {"type": "string"}
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": _STR,
                    "predicate": _STR,
                    "value": _STR,
                    "kind": {"type": "string", "enum": list(ASSERTION_KINDS)},
                    "previous_value": _STR,
                },
                "required": ["subject", "predicate", "value", "kind", "previous_value"],
            },
        },
    },
    "required": ["facts"],
}

EXTRACTION_SYSTEM_PROMPT = """\
You extract durable facts from the MESSAGE of a chat and return JSON: {"facts": [...]}.

Extract ONLY what the MESSAGE itself states. KNOWN FACTS and RECENT MESSAGES are context for \
resolving names and noticing changes; never copy facts from them.

Each fact is subject -> predicate -> value:
- subject: who or what the fact is about (a person, project, place, organisation, team, \
event or thing). Use a person's name when it is known; use "user" for the speaker only if \
their name is unknown. Reuse the subject names from KNOWN FACTS.
- predicate: short snake_case attribute or relation, e.g. works_at, lives_in, role, \
start_date, handles. Reuse the predicate from KNOWN FACTS when the same attribute is meant.
- value: short, copied from the MESSAGE.
- kind: "state" states a fact; "update" says a value changed, moved or was replaced; \
"correction" fixes an earlier mistake; "reference" only mentions something (for example \
talking about an old value) without asserting it is true now.
- previous_value: the old value being replaced, if known, else "".
- One fact per item; a message with several facts produces several items.
- Greetings, questions and generic advice produce no facts: return {"facts": []}.
"""


def build_extraction_prompt(
    message: Message, recent: list[Message], known_facts: list[str]
) -> str:
    facts = "\n".join(f"- {line}" for line in known_facts) or "(none)"
    history = "\n".join(f"{m.role}: {m.content}" for m in recent) or "(none)"
    return (
        f"CONTEXT - KNOWN FACTS (do not extract):\n{facts}\n\n"
        f"CONTEXT - RECENT MESSAGES (do not extract):\n{history}\n\n"
        f"MESSAGE ({message.role}):\n{message.content}\n\nJSON:"
    )


def _example(
    role: str, content: str, recent: list[tuple[str, str]], known: list[str], facts: list[tuple]
) -> tuple[str, str]:
    msgs = [Message(f"r{i}", "example", i, r, c) for i, (r, c) in enumerate(recent)]
    prompt = build_extraction_prompt(
        Message("x", "example", len(msgs), role, content), msgs, known
    )
    keys = ("subject", "predicate", "value", "kind", "previous_value")
    return prompt, json.dumps({"facts": [dict(zip(keys, f)) for f in facts]})


EXTRACTION_EXAMPLES: tuple[tuple[str, str], ...] = (
    _example(
        "user", "Hi! I'm Priya. I work as a nurse at St. Mary's Hospital and I live in Leeds.",
        [], [],
        [("Priya", "occupation", "nurse", "state", ""),
         ("Priya", "works_at", "St. Mary's Hospital", "state", ""),
         ("Priya", "lives_in", "Leeds", "state", "")],
    ),
    _example(
        "assistant", "Got it: Priya handles payroll and the team meets on Fridays at 9.",
        [("user", "Priya will handle payroll. We meet every Friday at 9.")],
        ["Priya | occupation | nurse"],
        [("Priya", "handles", "payroll", "state", ""),
         ("team", "meeting_time", "Fridays at 9", "state", "")],
    ),
    _example(
        "user", "Big news: the team meeting moved to the York office because Leeds is full.",
        [("user", "The team meeting is at the Leeds office.")],
        ["team | meeting_place | Leeds office"],
        [("team", "meeting_place", "York office", "update", "Leeds office")],
    ),
    _example(
        "user", "Sorry, I got that wrong, my dentist is actually Dr. Park.",
        [("user", "My dentist is Dr. Lee.")],
        ["Priya | dentist | Dr. Lee"],
        [("Priya", "dentist", "Dr. Park", "correction", "Dr. Lee")],
    ),
    _example(
        "assistant", "Congratulations on the move! How is the new commute?",
        [("user", "Big news: the team meeting moved to the York office because Leeds is full.")],
        ["team | meeting_place | York office"],
        [],
    ),
    _example(
        "user", "We talked about the old Leeds office at lunch today.",
        [], ["team | meeting_place | York office"],
        [("team", "meeting_place", "Leeds office", "reference", "")],
    ),
)


class JSONCompletionClient(Protocol):
    model_name: str

    def complete_json(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        examples: tuple[tuple[str, str], ...] = (),
    ) -> str: ...


# Small models can loop forever under a JSON schema; a capped reply ends as truncated
# JSON, which parse_extraction rejects as invalid.
MAX_OUTPUT_TOKENS = 512

# Prompt plus MAX_OUTPUT_TOKENS must fit, or Ollama silently truncates the prompt.
# Measured Phase 3.6 extraction prompts: 1071-1260 tokens (1772 with the output cap).
EXTRACTION_NUM_CTX = 4096


class OllamaJSONClient:
    """Ollama /api/chat with a JSON schema constraint, temperature 0 and a fixed seed."""

    def __init__(
        self,
        model_name: str,
        host: str,
        timeout: float = 300.0,
        seed: int = 0,
        num_ctx: int = EXTRACTION_NUM_CTX,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.seed = seed
        self.num_ctx = num_ctx

    def complete_json(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        examples: tuple[tuple[str, str], ...] = (),
    ) -> str:
        messages = [{"role": "system", "content": system}]
        for example_prompt, example_output in examples:
            messages.append({"role": "user", "content": example_prompt})
            messages.append({"role": "assistant", "content": example_output})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0,
                "seed": self.seed,
                "num_predict": MAX_OUTPUT_TOKENS,
                "num_ctx": self.num_ctx,
            },
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["message"]["content"]
        except (
            urllib.error.URLError, TimeoutError, OSError, KeyError, TypeError,
            json.JSONDecodeError,
        ) as exc:
            raise ExtractionRequestError(f"extraction request failed: {exc}") from exc


class MemoryExtractor:
    def __init__(self, client: JSONCompletionClient):
        self.client = client
        self.model_name = client.model_name

    def extract(
        self, message: Message, recent: list[Message], known_facts: list[KnownFact]
    ) -> tuple[ExtractionResult, str]:
        """Return (validated, grounded result, raw model output).

        Raises ``ExtractionError`` if the request fails or the output is invalid.
        """
        prompt = build_extraction_prompt(message, recent, [k.line for k in known_facts])
        raw = self.client.complete_json(
            EXTRACTION_SYSTEM_PROMPT, prompt, EXTRACTION_SCHEMA, EXTRACTION_EXAMPLES
        )
        return ground(parse_extraction(raw), message, recent, known_facts), raw
