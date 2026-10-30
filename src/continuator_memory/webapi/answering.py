"""Grounded answer generation from already-retrieved source evidence.

The model sees only budgeted source excerpts plus memory IDs that those excerpts
belong to. It does not search memory itself. Hosts and API keys stay server-side.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from typing import Any

from continuator_memory.generation import ANSWER_NUM_CTX, GenerationError, Generator, OllamaGenerator
from continuator_memory.models import Conversation
from continuator_memory.v2.temporal import (
    DATE_MENTION,
    RELATION_CURRENT,
    RELATION_FUTURE,
    DateResolution,
    date_mention_parts,
    date_mentions,
    format_date,
    message_date_resolutions,
    message_day,
)
from continuator_memory.webapi.prompt_budget import PromptBudget, allocate, estimate_tokens, trim_to_tokens
from continuator_memory.webapi.query_plan import FUNCTION_WORDS, entities_from_question, tokens

_CITE = re.compile(r"\[([^\[\]]{1,80})\]")
_SENTENCE_BREAK = re.compile(r"[.!?\n]")
_PRESENTATION = re.compile(
    r"(?i)\b(?:relevant\s+)?(?:excerpt|source|passage|quote)\b.{0,80}?"
    r"\b(?:states?|says?|reads?|mentions?)\b"
)
_ONLY_UNSUPPORTED = re.compile(r"^\s*\(unsupported\)\s*[.:]?\s*$", re.I)
# A sentence made only of these words points at a source and states nothing ("Citation:",
# "This is supported by the following excerpt:"); once its citation is stripped it is dropped.
_ORPHAN_ATTRIBUTION_WORDS = frozenset({
    "this", "that", "it", "is", "was", "are", "be", "can", "the", "a", "an", "following", "above", "below",
    "excerpt", "excerpts", "source", "sources", "message", "messages", "id", "ids", "citation", "citations",
    "reference", "references", "supported", "mentioned", "stated", "found", "seen", "shown", "cited",
    "quoted", "by", "in", "from", "see", "as", "according", "to", "based", "on", "and", "which",
})
_UNSUPPORTED_MARK = re.compile(r"\s*\(unsupported\)", re.I)
# A trailing source reference ("... yesterday according to her statement in message D1."). It must
# end the sentence, contain no comma and name a source, so it never spans claim text.
_ATTRIBUTION_TAIL = re.compile(
    r"(?i)\s*,?\s*"
    r"(?:according to|as (?:stated|said|mentioned)(?:\s+in)?)\b"
    r"(?=[^.!?\n,]*\b(?:excerpts?|sources?|messages?|passages?|statements?|quotes?|conversations?"
    r"|[a-z]{1,3}\d+(?::\d+)?)\b)"
    r"[^.!?\n,]{0,120}?"
    r"(?=\s*[.!?]?\s*$)"
)

SYSTEM_PROMPT = (
    "You answer questions using only the provided source excerpts. "
    "Do not use outside knowledge and do not invent facts, degrees, certifications, or source ids. "
    "Do not retrieve or assume other memories. "
    "The excerpts are retrieval neighbors, not equally relevant. "
    "Use an excerpt only if it directly answers the question. "
    "Do not treat hobbies, coping tools, or other mentioned topics as the answer "
    "unless the excerpt explicitly states they are. "
    "If a detail is not supported, omit it; if nothing supports the answer, say so clearly. "
    "When you rely on an excerpt, cite only the message id in square brackets that appears "
    "on the same line as that excerpt's text, immediately after the claim. "
    "Cite an id only when that excerpt's text supports the claim. "
    "Every claim you keep must include that citation; if no excerpt states the claim, omit it. "
    "Do not copy excerpt lines, roles, or memory ids into the answer."
)

NOT_CONFIGURED_ANSWER = (
    "Answer generation is not configured. Set CONTINUATOR_OLLAMA_MODEL "
    "(or CONTINUATOR_ANSWER_MODEL) and ensure Ollama is running, "
    "or send generate=false to inspect retrieval only."
)

MAX_MODEL_NAME = 80
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_PREDICT = 256
# The prompt is budgeted in tokens against the answer model's num_ctx (see prompt_budget). A
# character ceiling is optional (CONTINUATOR_ANSWER_MAX_CHARS); None means no character ceiling.
DEFAULT_MAX_CONTEXT_CHARS: int | None = None
MIN_EXCERPT_CHARS = 200  # smallest share of an excerpt when an optional character ceiling applies
# Top-ranked memories whose source messages are offered to the answerer (before the prompt
# budget). Ranking depth is configured separately and is not affected.
DEFAULT_ANSWER_TOP_K = 25
MAX_ANSWER_TOP_K = 32
ANSWER_TOP_K_ENV = "CONTINUATOR_ANSWER_TOP_K"


def load_answer_top_k(explicit: int | None = None, env: dict[str, str] | None = None) -> int:
    """Answer evidence count: an explicit value wins, then CONTINUATOR_ANSWER_TOP_K, then the default."""
    if explicit is not None:
        value, source = explicit, "top_k"
    else:
        raw = ((env if env is not None else os.environ).get(ANSWER_TOP_K_ENV) or "").strip()
        if not raw:
            return DEFAULT_ANSWER_TOP_K
        try:
            value, source = int(raw), ANSWER_TOP_K_ENV
        except ValueError as exc:
            raise ValueError(f"{ANSWER_TOP_K_ENV} must be an integer") from exc
    if not 1 <= value <= MAX_ANSWER_TOP_K:
        raise ValueError(f"{source} must be 1..{MAX_ANSWER_TOP_K}")
    return value


@dataclass(frozen=True)
class GenerationSettings:
    enabled: bool
    model: str
    temperature: float = DEFAULT_TEMPERATURE
    num_predict: int = DEFAULT_NUM_PREDICT
    max_context_chars: int | None = DEFAULT_MAX_CONTEXT_CHARS
    provider: str = "ollama"
    num_ctx: int = ANSWER_NUM_CTX

    @property
    def configured(self) -> bool:
        return bool(self.model.strip())

    def public_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "provider": self.provider if self.configured else None,
            "model": self.model or None,
            "temperature": self.temperature,
            "numPredict": self.num_predict,
            "maxContextChars": self.max_context_chars,
            "numCtx": self.num_ctx,
        }


DATE_NOTE_LABEL = "derived from the message date, not part of the message"
_RELATION_SUFFIX = {RELATION_FUTURE: " (still ahead at the message date)",
                    RELATION_CURRENT: " (current at the message date)"}


@dataclass
class PromptExcerpt:
    message_id: str
    role: str
    content: str
    memory_ids: tuple[str, ...]
    truncated: bool = False
    timestamp: str | None = None  # the source message's own timestamp, unchanged
    date_resolutions: tuple[DateResolution, ...] = ()  # computed from this excerpt's text and timestamp only

    @property
    def message_date(self) -> str | None:
        day = message_day(self.timestamp)
        return format_date(day) if day else None

    @property
    def date_note(self) -> str | None:
        """Derived reading of the excerpt's relative times; never part of the source text."""
        if not self.date_resolutions:
            return None
        parts = [f'"{r.phrase}" = {r.value}{_RELATION_SUFFIX.get(r.relation, "")}' for r in self.date_resolutions]
        return f"({DATE_NOTE_LABEL}: {'; '.join(parts)})"


@dataclass
class AnswerContext:
    question: str
    system: str
    prompt: str
    retrieved_memory_ids: tuple[str, ...]
    prompt_memory_ids: tuple[str, ...]
    dropped_memory_ids: tuple[str, ...]
    excerpts: tuple[PromptExcerpt, ...]
    dropped_message_ids: tuple[str, ...]
    memories: tuple[dict[str, Any], ...]
    prompt_chars: int
    truncated: bool
    max_context_chars: int | None
    budget: dict[str, Any] = field(default_factory=dict)


def load_generation_settings() -> GenerationSettings:
    env = os.environ
    flag = env.get("CONTINUATOR_ANSWER_ENABLED", "1").strip().lower()
    enabled = flag not in {"0", "false", "no", "off"}
    model = (env.get("CONTINUATOR_ANSWER_MODEL") or env.get("CONTINUATOR_OLLAMA_MODEL") or "").strip()
    return GenerationSettings(
        enabled=enabled,
        model=model,
        temperature=_float_env("CONTINUATOR_ANSWER_TEMPERATURE", DEFAULT_TEMPERATURE),
        num_predict=int(env.get("CONTINUATOR_ANSWER_NUM_PREDICT", DEFAULT_NUM_PREDICT)),
        max_context_chars=int(env["CONTINUATOR_ANSWER_MAX_CHARS"])
        if env.get("CONTINUATOR_ANSWER_MAX_CHARS", "").strip() else DEFAULT_MAX_CONTEXT_CHARS,
        num_ctx=int(env.get("CONTINUATOR_ANSWER_NUM_CTX", ANSWER_NUM_CTX)),
    )


def parse_generation_overrides(raw: Any, base: GenerationSettings) -> GenerationSettings:
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise ValueError("generation must be a JSON object")
    model = base.model
    if "model" in raw:
        if not isinstance(raw["model"], str) or not raw["model"].strip():
            raise ValueError("generation.model must be a non-empty string")
        model = raw["model"].strip()
        if len(model) > MAX_MODEL_NAME or "://" in model or "\n" in model or model.startswith("/"):
            raise ValueError("generation.model is not a valid local model name")
    temperature = base.temperature
    if "temperature" in raw:
        if not isinstance(raw["temperature"], (int, float)) or isinstance(raw["temperature"], bool):
            raise ValueError("generation.temperature must be a number")
        temperature = float(raw["temperature"])
        if temperature < 0 or temperature > 2:
            raise ValueError("generation.temperature must be between 0 and 2")
    num_predict = base.num_predict
    if "num_predict" in raw or "numPredict" in raw:
        value = raw.get("num_predict", raw.get("numPredict"))
        if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > 2048:
            raise ValueError("generation.num_predict must be an integer 1..2048")
        num_predict = value
    max_chars = base.max_context_chars
    if "max_context_chars" in raw or "maxContextChars" in raw:
        value = raw.get("max_context_chars", raw.get("maxContextChars"))
        if not isinstance(value, int) or isinstance(value, bool) or value < 200 or value > 200_000:
            raise ValueError("generation.max_context_chars must be an integer 200..200000")
        max_chars = value
    forbidden = {"host", "ollama_host", "api_key", "apiKey", "token"}
    if forbidden & set(raw):
        raise ValueError("generation cannot set server host or credentials")
    return GenerationSettings(
        enabled=base.enabled,
        model=model,
        temperature=temperature,
        num_predict=num_predict,
        max_context_chars=max_chars,
        provider=base.provider,
        num_ctx=base.num_ctx,
    )


def build_answer_context(
    question: str,
    evidence,
    conversation: Conversation,
    *,
    seed_ids: list[str],
    expanded_ids: list[str],
    max_context_chars: int | None = DEFAULT_MAX_CONTEXT_CHARS,
    context_tokens: int = ANSWER_NUM_CTX,
    output_tokens: int = DEFAULT_NUM_PREDICT,
    system: str = SYSTEM_PROMPT,
) -> AnswerContext:
    """Render the answer prompt within a token budget.

    The budget is ``context_tokens`` (the model's num_ctx) minus ``output_tokens`` (num_predict),
    the chat template, the system prompt and the question frame. Excerpts stay in rank order;
    long excerpts are trimmed to a shared cap before any excerpt is left out, and only the
    lowest-ranked ones are left out when even a minimal share does not fit. ``max_context_chars``
    is an optional extra ceiling on excerpt characters, applied the same way.
    """
    selected = list(evidence.selection.selected)
    dropped_messages = list(evidence.selection.dropped)
    by_id = {m.message_id: m for m in conversation.messages}
    retrieved_ids = [m.memory_id for m in evidence.matches]
    owners: dict[str, list[str]] = {}
    for match in evidence.matches:
        for sid in match.source_message_ids:
            owners.setdefault(sid, []).append(match.memory_id)

    def excerpt(message, text: str, cut: bool) -> PromptExcerpt:
        mem_ids = tuple(dict.fromkeys(owners.get(message.message_id, [])))
        resolutions = message_date_resolutions(_resolvable_text(text), message_day(message.timestamp))
        return PromptExcerpt(message.message_id, message.role, text, mem_ids, cut, message.timestamp, resolutions)

    candidates = [by_id[mid] for mid in selected if mid in by_id]
    budget = PromptBudget(context_tokens, output_tokens)
    available = budget.evidence_tokens(f"{system}\n{_render_prompt(question, [])}")
    marker = estimate_tokens("… [truncated]")
    full = [excerpt(m, m.content, False) for m in candidates]
    fixed = [estimate_tokens(_render_block(replace(item, content=""))) + marker + 1 for item in full]
    content = [estimate_tokens(m.content) for m in candidates]
    plan = allocate(fixed, content, available)
    keep, char_cap = plan.keep, None
    if max_context_chars is not None:
        by_chars = allocate([0] * len(candidates), [len(m.content) for m in candidates], max_context_chars,
                            min_content=MIN_EXCERPT_CHARS)
        keep, char_cap = min(keep, by_chars.keep), by_chars.content_cap

    def render(count: int) -> tuple[list[PromptExcerpt], list[str]]:
        items, trimmed = [], []
        for message in candidates[:count]:
            text = message.content
            if char_cap is not None and len(text) > char_cap:
                text = text[:char_cap].rstrip()
            if plan.content_cap is not None and estimate_tokens(text) > plan.content_cap:
                text = trim_to_tokens(text, plan.content_cap, query=question)
            cut = text != message.content
            if cut:
                text += "…"
                trimmed.append(message.message_id)
            items.append(excerpt(message, text, cut))
        return items, trimmed

    excerpts, trimmed_ids = render(keep)
    prompt = _render_prompt(question, excerpts)
    estimated = budget.template_tokens + estimate_tokens(system) + estimate_tokens(prompt)
    while excerpts and estimated + output_tokens > context_tokens:  # rounding guard; never exceed
        keep -= 1
        excerpts, trimmed_ids = render(keep)
        prompt = _render_prompt(question, excerpts)
        estimated = budget.template_tokens + estimate_tokens(system) + estimate_tokens(prompt)
    budget_dropped = [m.message_id for m in candidates[keep:]]
    used_message_ids = [item.message_id for item in excerpts]

    used_set = set(used_message_ids)
    prompt_memory_ids: list[str] = []
    memory_rows: list[dict[str, Any]] = []
    expanded = set(expanded_ids)
    seeds = set(seed_ids)
    for match in evidence.matches:
        in_prompt = any(sid in used_set for sid in match.source_message_ids)
        if in_prompt:
            prompt_memory_ids.append(match.memory_id)
        memory_rows.append({
            "id": match.memory_id,
            "statement": match.statement,
            "rank": match.rank,
            "score": match.score,
            "similarity": match.similarity,
            "origin": (
                "seed" if match.memory_id in seeds
                else ("bprime" if match.memory_id in expanded else "retrieved")
            ),
            "sourceMessageIds": list(match.source_message_ids),
            "inPrompt": in_prompt,
        })
    dropped_memory_ids = [mid for mid in retrieved_ids if mid not in set(prompt_memory_ids)]
    dropped_messages = list(dict.fromkeys([*dropped_messages, *budget_dropped]))

    return AnswerContext(
        question=question,
        system=system,
        prompt=prompt,
        retrieved_memory_ids=tuple(retrieved_ids),
        prompt_memory_ids=tuple(prompt_memory_ids),
        dropped_memory_ids=tuple(dropped_memory_ids),
        excerpts=tuple(excerpts),
        dropped_message_ids=tuple(dict.fromkeys(dropped_messages)),
        memories=tuple(memory_rows),
        prompt_chars=len(prompt),
        truncated=bool(trimmed_ids or budget_dropped),
        max_context_chars=max_context_chars,
        budget={
            "contextTokens": context_tokens,
            "outputTokens": output_tokens,
            "templateTokens": budget.template_tokens,
            "evidenceTokensAvailable": available,
            "estimatedPromptTokens": estimated,
            "contentCapTokens": plan.content_cap,
            "maxContextChars": max_context_chars,
            "trimmedMessageIds": trimmed_ids,
            "droppedForBudget": budget_dropped,
            "estimator": "conservative-qwen2-bpe-v1",
        },
    )


def _render_block(item: PromptExcerpt) -> str:
    note = " [truncated]" if item.truncated else ""
    dated = f"(message date: {item.message_date}) " if item.message_date else ""
    block = f"[{item.message_id}] {dated}{item.content}{note}"
    if item.date_note:
        block += f"\n{item.date_note}"
    if item.memory_ids:
        block += f"\nlinked memory {', '.join(item.memory_ids)}"
    return block


def _render_prompt(question: str, excerpts: list[PromptExcerpt]) -> str:
    body = "\n\n".join(_render_block(item) for item in excerpts) if excerpts else (
        "(no source excerpts were selected for the model)")
    return (
        "Source excerpts. Cite using only the square-bracket message id on the same "
        "line as the supporting text.\n\n"
        f"{body}\n\nQuestion: {question}\nAnswer:"
    )


def context_to_evidence_fields(ctx: AnswerContext) -> dict[str, Any]:
    return {
        "question": ctx.question,
        "promptMemoryIds": list(ctx.prompt_memory_ids),
        "droppedMemoryIds": list(ctx.dropped_memory_ids),
        "droppedMessageIds": list(ctx.dropped_message_ids),
        "memories": list(ctx.memories),
        "promptExcerpts": [
            {
                "id": e.message_id,
                "role": e.role if e.role in {"user", "assistant", "system"} else "user",
                "content": e.content,
                "memoryIds": list(e.memory_ids),
                "truncated": e.truncated,
            }
            for e in ctx.excerpts
        ],
    }


_ATTRIBUTION_TERMS = frozenset({
    "according", "statement", "message", "excerpt", "source", "passage",
    "says", "said", "stated", "states", "mentioned", "mentions",
    "quoted", "quote", "citing", "cited",
})
_TEMPORAL_TERMS = frozenset({
    "yesterday", "today", "tomorrow", "tonight", "morning", "afternoon",
    "evening", "night", "weekday", "weekend",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
})


def _content_terms(text: str) -> list[str]:
    return [tok for tok in tokens(text) if tok not in FUNCTION_WORDS and len(tok) >= 3]


def _ignore_terms(question: str) -> set[str]:
    ignored: set[str] = set(_ATTRIBUTION_TERMS)
    for tok in _content_terms(question):
        ignored.add(tok)
        if tok.endswith("'s") and len(tok) > 2:
            ignored.add(tok[:-2])
    for entity in entities_from_question(question):
        for tok in tokens(entity):
            ignored.add(tok)
            if tok.endswith("'s") and len(tok) > 2:
                ignored.add(tok[:-2])
    return ignored


def _claim_before(text: str, cite_at: int) -> str:
    """Claim text associated with a citation at ``cite_at``.

    If the citation sits after sentence-ending punctuation (``claim. [id]``),
    the span after the last break is empty; use the preceding sentence.
    """
    starts = [0, *[match.end() for match in _SENTENCE_BREAK.finditer(text, 0, cite_at)]]
    for start in reversed(starts):
        claim = _CITE.sub(" ", text[start:cite_at])
        claim = " ".join(claim.split()).strip()
        if claim:
            return claim
    return ""


def _term_in_excerpt(tok: str, excerpt_terms: set[str]) -> bool:
    if tok in excerpt_terms:
        return True
    return any(
        min(len(tok), len(other)) >= 5
        and (tok.startswith(other) or other.startswith(tok))
        for other in excerpt_terms
    )


def _resolvable_text(content: str) -> str:
    """Message text whose relative times are resolved; shared-image caption lines are left out."""
    return "\n".join(line for line in content.split("\n") if not line.startswith("[image: "))


# Words a date claim may use around the date itself; they never count as content support.
_DATE_WORDS = frozenset({
    "week", "weekend", "night", "around", "before", "after", "about", "approximately", "roughly",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
})


def _dates_compatible(claimed: tuple, derived: tuple) -> bool:
    """Every part the claim states (day, month, year) is stated identically by the derived date."""
    return any(part is not None for part in claimed) and all(
        c is None or c == d for c, d in zip(claimed, derived))


# "around 2019", "in about 2019", "2019 or so": the claim keeps a derived approximation as one.
_APPROX_CLAIM_BEFORE = re.compile(r"\b(?:around|about|approximately|roughly|circa)\s+(?:\w+\s+){0,2}$", re.I)
_APPROX_CLAIM_AFTER = re.compile(r"^(?:\s+or so\b|-ish\b)", re.I)
_APPROX_WORDS = frozenset({"approximately", "roughly", "circa", "ish"})
# Wording that keeps a still-ahead date a plan rather than a completed event.
_PLAN_CLAIM = re.compile(
    r"\b(?:will|shall|going to|plans?|planned|planning|intends?|intended|intending|scheduled|"
    r"expects?|expected|upcoming|about to|set to|(?:is|are|am|'s|'re) going)\b|\w'll\b", re.I)


def _note_date_supported(claim: str, mention: re.Match, date_resolutions: tuple[DateResolution, ...]) -> bool:
    """This claimed date matches one of the excerpt's resolutions without dropping its approximation
    or turning a still-ahead date into a completed event."""
    claimed = date_mention_parts(mention)
    for resolution in date_resolutions:
        if not any(_dates_compatible(claimed, d) for d in date_mentions(resolution.value)):
            continue
        if resolution.value.startswith("around ") and not (
                _APPROX_CLAIM_BEFORE.search(claim[:mention.start()])
                or _APPROX_CLAIM_AFTER.match(claim[mention.end():])):
            continue
        if resolution.relation == RELATION_FUTURE and not _PLAN_CLAIM.search(claim):
            continue
        return True
    return False


def _note_supports_claim(claim: str, excerpt: str, question: str,
                         date_resolutions: tuple[DateResolution, ...]) -> bool:
    """A derived note supports only the date: each claimed date must be written in the excerpt or match
    one of this excerpt's own resolved values, and every other content word must still be found in the
    excerpt text."""
    mentions = list(DATE_MENTION.finditer(claim))
    if not mentions or not date_resolutions:
        return False
    written = date_mentions(excerpt)
    for mention in mentions:
        if any(_dates_compatible(date_mention_parts(mention), d) for d in written):
            continue
        if not _note_date_supported(claim, mention, date_resolutions):
            return False
    ignored = _ignore_terms(question)
    rest_text = _PLAN_CLAIM.sub(" ", DATE_MENTION.sub(" ", claim))
    rest = [tok for tok in _content_terms(rest_text)
            if tok not in ignored and tok not in _DATE_WORDS and tok not in _APPROX_WORDS]
    excerpt_terms = set(_content_terms(excerpt))
    return all(_term_in_excerpt(tok, excerpt_terms) for tok in rest)


def _date_only_from_note(claim: str, excerpt: str) -> bool:
    """The claim states a date that the excerpt text does not, so only a derived note could support it."""
    written = date_mentions(excerpt)
    return any(not any(_dates_compatible(c, d) for d in written) for c in date_mentions(claim))


def excerpt_supports_claim(claim: str, excerpt: str, *, question: str = "",
                           date_resolutions: tuple[DateResolution, ...] = ()) -> bool:
    if date_resolutions and _date_only_from_note(claim, excerpt):
        # Word overlap with the text must not vouch for a date the text never states.
        return _note_supports_claim(claim, excerpt, question, date_resolutions)
    if _text_supports_claim(claim, excerpt, question):
        return True
    return bool(date_resolutions) and _note_supports_claim(claim, excerpt, question, date_resolutions)


def _text_supports_claim(claim: str, excerpt: str, question: str) -> bool:
    ignored = _ignore_terms(question)
    claim_terms = [tok for tok in _content_terms(claim) if tok not in ignored]
    if not claim_terms:
        return False
    excerpt_terms = set(_content_terms(excerpt))
    claim_times = [tok for tok in claim_terms if tok in _TEMPORAL_TERMS]
    if claim_times:
        return any(_term_in_excerpt(tok, excerpt_terms) for tok in claim_times)
    distinctive = [tok for tok in claim_terms if len(tok) >= 4]
    needed = distinctive or claim_terms
    return any(_term_in_excerpt(tok, excerpt_terms) for tok in needed)


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    in_quote = False
    opener = ""
    closers = {"\"": "\"", "'": "'", "“": "”"}
    for index, char in enumerate(text):
        if not in_quote and char in closers:
            in_quote = True
            opener = char
            continue
        if in_quote and char == closers[opener]:
            in_quote = False
            opener = ""
            continue
        if in_quote:
            continue
        if _SENTENCE_BREAK.match(char):
            chunk = text[start:index + 1]
            if chunk.strip():
                parts.append(chunk)
            start = index + 1
    tail = text[start:]
    if tail.strip():
        parts.append(tail)
    return parts


def _remove_snippet(text: str, snippet: str) -> str:
    snippet = snippet.strip()
    if len(snippet) < 8:
        return text
    return re.sub(re.escape(snippet), "", text, flags=re.IGNORECASE)


def _prompt_header_re(item: PromptExcerpt) -> re.Pattern[str] | None:
    """Match copied prompt chrome for a rejected excerpt."""
    parts = [rf"\(role:\s*{re.escape(item.role)};\s*memories:\s*[^)]*\)"]
    if item.memory_ids:
        mem = ", ".join(item.memory_ids)
        parts.append(rf"linked memory {re.escape(mem)}")
    if item.message_date:
        parts.append(rf"\(message date: {re.escape(item.message_date)}\)")
    if item.date_note:
        parts.append(re.escape(item.date_note))
    return re.compile("|".join(parts))


def _strip_attribution_tail(sentence: str) -> str:
    """Drop a trailing source reference; a sentence that starts with one keeps its claim."""
    match = _ATTRIBUTION_TAIL.search(sentence)
    if match is None or not sentence[:match.start()].strip():
        return sentence
    return sentence[:match.start()] + sentence[match.end():]


def _strip_rejected_ids(text: str, rejected_ids: list[str], kept_ids: set[str]) -> str:
    for message_id in rejected_ids:
        if message_id in kept_ids:
            continue
        text = re.sub(rf"(?<!\[){re.escape(message_id)}(?!\])", "", text)
    return text


def _strip_rejected_evidence_prose(
    text: str,
    decisions: list[dict[str, Any]],
    excerpts: tuple[PromptExcerpt, ...],
    question: str,
) -> str:
    """Remove copied prompt blocks and bodies of rejected excerpts.

    Uses the excerpt text and memory IDs already supplied to the validator
    (including ``mem:<message_id>:c0``), not guessed answer phrases.
    """
    del question
    rejected_ids = [row["id"] for row in decisions if not row["kept"]]
    if not rejected_ids:
        return text
    allowed = {item.message_id: item for item in excerpts}
    kept_ids = {row["id"] for row in decisions if row["kept"]}
    kept_contents = {
        allowed[mid].content.strip() for mid in kept_ids if mid in allowed and allowed[mid].content.strip()
    }
    for message_id in rejected_ids:
        item = allowed.get(message_id)
        if item is None:
            continue
        header = _prompt_header_re(item)
        if header is not None:
            text = header.sub("", text)
        for memory_id in item.memory_ids:
            text = text.replace(memory_id, "")
        if item.content.strip() and item.content.strip() not in kept_contents:
            text = _remove_snippet(text, item.content)
    text = _strip_rejected_ids(text, rejected_ids, kept_ids)
    kept_sentences: list[str] = []
    for sentence in _split_sentences(text):
        compact = _UNSUPPORTED_MARK.sub("", sentence).strip()
        if _ONLY_UNSUPPORTED.match(compact) or not compact:
            continue
        has_kept_cite = any(f"[{mid}]" in compact for mid in kept_ids)
        has_kept_excerpt = any(content and content in compact for content in kept_contents)
        if not has_kept_cite:
            compact = _strip_attribution_tail(compact).strip()
            words = re.findall(r"\w+", compact.lower())
            if words and set(words) <= _ORPHAN_ATTRIBUTION_WORDS:
                continue
        if _PRESENTATION.search(compact) and not has_kept_cite and not has_kept_excerpt:
            leftover = _PRESENTATION.sub("", compact)
            leftover = leftover.strip(" :.\n\"'")
            if len(re.findall(r"\w+", leftover)) < 3:
                continue
        if not re.search(r"\w", compact):
            continue
        kept_sentences.append(compact)
    cleaned = " ".join(kept_sentences)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" +([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    return cleaned.strip()


def validate_answer_citations(
    text: str,
    excerpts: tuple[PromptExcerpt, ...],
    *,
    question: str = "",
) -> tuple[str, dict[str, Any]]:
    """Keep only citations whose supplied excerpt supports the nearby claim.

    Invalid IDs are never rewritten to a different source. Raw text is unchanged
    here; the caller stores it in diagnostics.
    """
    allowed = {item.message_id: item for item in excerpts}
    matches = list(_CITE.finditer(text))
    if not matches:
        return text, {"citations": [], "stripped": [], "rawAnswer": text}

    decisions: list[dict[str, Any]] = []
    for match in matches:
        cite_id = match.group(1).strip()
        claim = _claim_before(text, match.start())
        row = {"id": cite_id, "claim": claim, "kept": False, "reason": "supported"}
        excerpt = allowed.get(cite_id)
        if excerpt is None:
            row["reason"] = "not_in_prompt"
        elif excerpt_supports_claim(claim, excerpt.content, question=question,
                                    date_resolutions=excerpt.date_resolutions):
            row["kept"] = True
        else:
            row["reason"] = "unsupported_excerpt"
        decisions.append(row)

    parts: list[str] = []
    cursor = 0
    stripped: list[dict[str, Any]] = []
    for match, row in zip(matches, decisions):
        parts.append(text[cursor:match.start()])
        if row["kept"]:
            parts.append(match.group(0))
        else:
            stripped.append(row)
        cursor = match.end()
    parts.append(text[cursor:])
    cleaned = re.sub(r"[ \t]{2,}", " ", "".join(parts))
    cleaned = re.sub(r" +([,.;:])", r"\1", cleaned).strip()
    cleaned = _strip_rejected_evidence_prose(cleaned, decisions, excerpts, question)
    return cleaned, {
        "citations": decisions,
        "stripped": stripped,
        "rawAnswer": text,
    }


CONTEXT_OVERFLOW_ANSWER = (
    "The answer prompt does not fit the answer model's context window; no answer was generated. "
    "Reduce the evidence count or raise CONTINUATOR_ANSWER_NUM_CTX."
)


def estimated_prompt_tokens(ctx: AnswerContext) -> int:
    estimate = ctx.budget.get("estimatedPromptTokens") if ctx.budget else None
    if estimate is None:
        template = ctx.budget.get("templateTokens") if ctx.budget else None
        estimate = (template or 0) + estimate_tokens(ctx.system) + estimate_tokens(ctx.prompt)
    return estimate


def generate_answer(
    ctx: AnswerContext,
    settings: GenerationSettings,
    generator: Generator | None,
) -> tuple[str, dict[str, Any]]:
    estimate = estimated_prompt_tokens(ctx)
    meta = {
        **settings.public_dict(),
        "promptChars": ctx.prompt_chars,
        "truncated": ctx.truncated,
        "estimatedPromptTokens": estimate,
    }
    if not settings.configured:
        return NOT_CONFIGURED_ANSWER, {**meta, "status": "not_configured"}
    if generator is None:
        return NOT_CONFIGURED_ANSWER, {**meta, "status": "not_configured"}
    if estimate + settings.num_predict > settings.num_ctx:
        # Ollama would silently drop part of the prompt; refuse instead.
        return CONTEXT_OVERFLOW_ANSWER, {**meta, "status": "context_overflow",
                                         "error": f"estimated {estimate} prompt tokens + {settings.num_predict} "
                                                  f"output tokens > num_ctx {settings.num_ctx}"}
    try:
        try:
            text = generator.generate(
                ctx.system,
                ctx.prompt,
                temperature=settings.temperature,
                num_predict=settings.num_predict,
            )
        except TypeError:
            text = generator.generate(ctx.system, ctx.prompt)
    except GenerationError as exc:
        return (
            f"The answer model could not be reached. {exc} "
            "Retrieval still ran; open the answer inspector for the evidence.",
            {**meta, "status": "error", "error": str(exc)},
        )
    raw = text.strip()
    display, citation_meta = validate_answer_citations(raw, ctx.excerpts, question=ctx.question)
    status = "ok"
    usage = getattr(generator, "last_usage", None) or {}
    reported = usage.get("prompt_eval_count")
    if reported is not None and reported + settings.num_predict > (usage.get("num_ctx") or settings.num_ctx):
        # The estimate was exceeded; Ollama may have truncated the input. Never report this as ok.
        status = "context_overflow"
        meta["error"] = (f"Ollama reported {reported} prompt tokens + {settings.num_predict} output tokens "
                         f"> num_ctx {usage.get('num_ctx') or settings.num_ctx}")
    return display, {
        **meta,
        "status": status,
        "model": getattr(generator, "model_name", settings.model),
        "rawAnswer": raw,
        "citations": citation_meta["citations"],
        "strippedCitations": citation_meta["stripped"],
    }


def make_ollama_generator(settings: GenerationSettings) -> OllamaGenerator:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return OllamaGenerator(settings.model, host, num_ctx=settings.num_ctx)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)
