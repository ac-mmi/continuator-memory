"""Experiment: Phase 1.1.1 extraction plus a rolling conversation summary. Opt-in and isolated.

Each window's single model call also returns ``rolling_summary``, which is passed to the next window. There
is no separate summarization call. Everything else is the unchanged ``structured_targets`` path: target slots,
normalization, evidence checks, deduplication and literal name checks, including their known defects.

The summary is working context only:
  - It is never shown to the validator, so it cannot supply a quote or satisfy a name check.
  - Code carries it forward as the model wrote it. Code cannot tell whether it kept a possibility tentative;
    that is measured by reading the saved summaries.
  - It is written by the model from the previous summary and the window's raw messages, never built from
    accepted records, so extraction omissions are not fed back into it by code.
  - If a window fails or its summary is unusable, the previous summary is kept and a flag records why.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

from continuator_memory.models import Conversation
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import (
    DEFAULT_PRIOR_LIMIT,
    AtomizationRequestError,
    JSONCompletionClient,
    OllamaChatClient,
    V2Settings,
)
from continuator_memory.v2.structured_store import StructuredIngestReport, prepare_conversation
from continuator_memory.v2.structured_targets import (
    DEFAULT_TARGETS_NUM_PREDICT,
    OUTCOME_UNPROCESSED,
    TARGETS_EXAMPLES,
    TARGETS_SYSTEM_PROMPT,
    TargetMemory,
    TargetOutcome,
    TargetWindow,
    build_targets_prompt,
    targets_output_schema,
    validate_targets,
)
from continuator_memory.v2.validation import FLAGGED, REJECTED, ValidationIssue
from continuator_memory.v2.windowing import DEFAULT_CONTEXT_BEFORE, DEFAULT_TARGETS_PER_WINDOW, ContextWindow, build_windows

ROLLING_PROTOCOL_VERSION = "structured-v1.1.1-rolling"
SUMMARY_FIELD = "rolling_summary"
DEFAULT_MAX_SUMMARY_TOKENS = 400
CHARS_PER_TOKEN = 4  # No local tokenizer for the model; a size estimate, labelled as one wherever it is saved.

SUMMARY_KEPT = "updated"
SUMMARY_TRIMMED = "trimmed"
SUMMARY_MISSING = "missing_kept_previous"
SUMMARY_INVALID = "invalid_kept_previous"
SUMMARY_WINDOW_FAILED = "window_failed_kept_previous"

_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|\n+")  # Punctuation only: where a cut may fall, not what text means.

ROLLING_SYSTEM_PROMPT = TARGETS_SYSTEM_PROMPT + """
ROLLING SUMMARY is your working notes on everything before this window. Use it to understand people, projects, \
references and earlier decisions. It is background, not evidence: never quote it, and only quote real messages. \
Do not create memories or relationships from it. Records still come only from TARGET messages.

Also return rolling_summary: the notes for the next window. Rewrite them from the previous ROLLING SUMMARY and \
the actual CONTEXT and TARGET messages, not only from the memories you extracted. Update and condense; do not \
just append. Under 300 words. Cover people and who they are, projects and what they are about, facts and \
preferences, decisions with their stated reasons, relationships, dates, tentative plans, open questions, and \
corrections.
Keep these distinctions in your wording: a possibility stays a possibility ("might move", never "moved"); a \
question stays a question; a suggestion is not a decision; a plan is not a completed event; say what is current \
and what used to be true; when something is corrected, keep both ("now X; earlier said Y"). Attribute to the \
speaker where it matters. Do not add anything the conversation does not say.
"""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def build_rolling_prompt(window: ContextWindow, prior: Sequence[TargetMemory] = (), summary: str = "") -> str:
    return (f"ROLLING SUMMARY (background from earlier windows, NOT evidence):\n{summary or '(none yet: this is the start)'}\n\n"
            f"{build_targets_prompt(window, prior)}")


def rolling_output_schema(window: ContextWindow) -> dict[str, Any]:
    schema = targets_output_schema(window)
    schema["properties"][SUMMARY_FIELD] = {"type": "string"}
    schema["required"] = [*schema["required"], SUMMARY_FIELD]
    return schema


_example_prompt, _example_output = TARGETS_EXAMPLES[0]
ROLLING_EXAMPLES: tuple[tuple[str, str], ...] = ((
    f"ROLLING SUMMARY (background from earlier windows, NOT evidence):\n(none yet: this is the start)\n\n{_example_prompt}",
    json.dumps({**json.loads(_example_output), SUMMARY_FIELD: (
        "Ines and Rafi are talking. Rafi is at Harbour Print and might leave in June (only a possibility). "
        "Rafi chose Tallow over Ember because it is cheaper. Ines now says she swims on Thursday; she earlier said "
        "Tuesday. Open question from Ines: does Rafi still cycle?")})),)


def trim_summary(summary: str, max_tokens: int) -> str:
    """Whole sentences or lines up to the limit, so a cut never leaves half a statement."""
    limit = max_tokens * CHARS_PER_TOKEN
    if len(summary) <= limit:
        return summary
    kept = ""
    for match in _BOUNDARY.finditer(summary):
        if match.start() > limit:
            break
        kept = summary[: match.start()]
    return kept.strip() or summary[:limit].strip()


@dataclass
class RollingResult:
    validated: TargetWindow
    prompt: str
    latency_ms: float
    usage: dict[str, Any] | None
    summary_in: str
    summary_out: str
    summary_status: str
    model_summary: str | None = None


def _split_summary(raw: str) -> tuple[str, Any, bool]:
    """(the response without the summary, the summary value, whether the key was present)."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, None, False
    if not isinstance(data, dict) or SUMMARY_FIELD not in data:
        return raw, None, False
    value = data.pop(SUMMARY_FIELD)
    return json.dumps(data), value, True


class RollingExtractor:
    """One ``complete_json`` call per window, returning records and the next summary together."""

    protocol = ROLLING_PROTOCOL_VERSION

    def __init__(self, client: JSONCompletionClient, max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS):
        self.client = client
        self.model_name = client.model_name
        self.max_summary_tokens = max_summary_tokens

    def extract(self, window: ContextWindow, prior: Sequence[TargetMemory] = (), covered: Collection[str] = (),
                summary: str = "") -> RollingResult:
        prompt = build_rolling_prompt(window, prior, summary)
        if hasattr(self.client, "last_usage"):
            self.client.last_usage = None
        t0 = time.perf_counter()
        try:
            raw = self.client.complete_json(ROLLING_SYSTEM_PROMPT, prompt, rolling_output_schema(window), ROLLING_EXAMPLES)
        except AtomizationRequestError as exc:
            outcomes = [TargetOutcome(mid, OUTCOME_UNPROCESSED) for mid in window.target_ids]
            validated = TargetWindow(window, outcomes, issues=[ValidationIssue("REQUEST_FAILED", REJECTED, str(exc))], error=str(exc))
            return RollingResult(validated, prompt, round((time.perf_counter() - t0) * 1000, 3), None, summary, summary, SUMMARY_WINDOW_FAILED)
        usage = getattr(self.client, "last_usage", None)
        rest, value, present = _split_summary(raw)
        validated = validate_targets(rest, window, prior, (usage or {}).get("done_reason"), covered)
        validated.raw = raw
        latency = round((time.perf_counter() - t0) * 1000, 3)
        model_summary = value if isinstance(value, str) else None
        out, status = self._next_summary(value, present, summary, validated)
        return RollingResult(validated, prompt, latency, usage, summary, out, status, model_summary)

    def _next_summary(self, value: Any, present: bool, previous: str, validated: TargetWindow) -> tuple[str, str]:
        if validated.error:
            return previous, SUMMARY_WINDOW_FAILED
        if not present:
            validated.issues.append(ValidationIssue("SUMMARY_MISSING", FLAGGED, "no rolling_summary; previous summary kept"))
            return previous, SUMMARY_MISSING
        if not isinstance(value, str) or not value.strip():
            validated.issues.append(ValidationIssue("SUMMARY_INVALID", FLAGGED, f"rolling_summary is {type(value).__name__} or empty; previous summary kept"))
            return previous, SUMMARY_INVALID
        text = value.strip()
        trimmed = trim_summary(text, self.max_summary_tokens)
        if trimmed != text:
            validated.issues.append(ValidationIssue("SUMMARY_TRIMMED", FLAGGED, f"about {estimate_tokens(text)} tokens, limit {self.max_summary_tokens}; trimmed at a sentence boundary"))
            return trimmed, SUMMARY_TRIMMED
        return text, SUMMARY_KEPT


ROLLING_SCHEMA = """
CREATE TABLE IF NOT EXISTS sx_rolling_summaries (
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    window_index INTEGER NOT NULL,
    summary_in TEXT NOT NULL,
    summary_out TEXT NOT NULL,
    model_summary TEXT,
    status TEXT NOT NULL,
    PRIMARY KEY (conversation_id, window_index)
);
"""


class RollingSummaryStore:
    def __init__(self, connection: sqlite3.Connection):
        self._conn = connection
        self._conn.executescript(ROLLING_SCHEMA)

    def clear(self, conversation_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM sx_rolling_summaries WHERE conversation_id = ?", (conversation_id,))

    def save(self, conversation_id: str, window_index: int, result: RollingResult) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO sx_rolling_summaries (conversation_id, window_index, summary_in, summary_out, model_summary, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conversation_id, window_index, result.summary_in, result.summary_out, result.model_summary, result.summary_status))

    def get(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT window_index, summary_in, summary_out, model_summary, status FROM sx_rolling_summaries "
            "WHERE conversation_id = ? ORDER BY window_index", (conversation_id,)).fetchall()
        return [{"window_index": r[0], "summary_in": r[1], "summary_out": r[2], "model_summary": r[3], "status": r[4]} for r in rows]


@dataclass
class RollingIngestReport(StructuredIngestReport):
    summary_statuses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "summary_statuses": list(self.summary_statuses)}


def ingest_rolling(
    conversation: Conversation,
    store: Store,
    extractor: RollingExtractor,
    targets_per_window: int = DEFAULT_TARGETS_PER_WINDOW,
    context_before: int = DEFAULT_CONTEXT_BEFORE,
    prior_limit: int = DEFAULT_PRIOR_LIMIT,
    replace: bool = False,
    on_window: Any = None,
) -> RollingIngestReport:
    """One sequential call per window. Each window and summary is saved as it finishes, so partial runs survive."""
    structured = prepare_conversation(conversation, store, replace)
    summaries = RollingSummaryStore(store.connection)
    cid = conversation.conversation_id
    summaries.clear(cid)
    report = RollingIngestReport(cid)
    prior: list[TargetMemory] = []
    covered: set[str] = set()
    summary = ""
    for window in build_windows(conversation, targets_per_window, context_before):
        window_prior = prior[-prior_limit:] if prior_limit else []
        result = extractor.extract(window, window_prior, covered, summary)
        validated = result.validated
        structured.save_window(validated, extractor.model_name, result.latency_ms,
                               tuple(m.memory_id for m in window_prior), result.usage, extractor.protocol)
        summaries.save(cid, window.index, result)
        prior.extend(validated.memories)
        covered |= {r.message_id for r in (*validated.memories, *validated.relationships)}
        summary = result.summary_out
        report.add_window(validated, result.usage, result.latency_ms)
        report.summary_statuses.append(result.summary_status)
        if on_window:
            on_window(window, result)
    return report


def rolling_extractor(settings: V2Settings | None = None, max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS) -> RollingExtractor:
    s = settings or V2Settings.from_env()
    num_predict = s.num_predict if "CONTINUATOR_V2_NUM_PREDICT" in os.environ else DEFAULT_TARGETS_NUM_PREDICT
    return RollingExtractor(OllamaChatClient(s.atomizer_model, s.ollama_host, s.num_ctx, num_predict), max_summary_tokens)
