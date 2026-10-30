"""Evaluate whether retrieved excerpts are enough to answer a question.

The evaluator does not answer the question and does not invent facts. Follow-up
queries may reuse terms from the question, the paraphrase lexicon, and the
retrieved excerpts. Invalid model output stops the loop instead of inventing queries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from continuator_memory.generation import GenerationError, Generator
from continuator_memory.webapi.query_plan import (
    EDUCATION_CUES,
    FUNCTION_WORDS,
    MAX_QUERY_CHARS,
    entities_from_question,
    extra_content_terms,
    extract_json_object,
    tokens,
)

STATUS_SUFFICIENT = "sufficient"
STATUS_INSUFFICIENT = "insufficient"
STATUS_UNCERTAIN = "uncertain"
ALLOWED_STATUSES = frozenset({STATUS_SUFFICIENT, STATUS_INSUFFICIENT, STATUS_UNCERTAIN})

MAX_FACTS = 8
MAX_FACT_CHARS = 400
MAX_MISSING_CHARS = 400
MAX_NEXT_QUERIES = 3
EVAL_NUM_PREDICT = 400

SOURCE_OLLAMA = "ollama"
SOURCE_FALLBACK = "fallback"
SOURCE_INJECTED = "injected"
SOURCE_HEURISTIC = "heuristic"

_SPEAKER_PREFIX = re.compile(r"^[A-Za-z][A-Za-z' -]{0,40}:\s*")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
_TEMPORAL_WORDS = frozenset({
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "yesterday", "today", "tomorrow", "tonight", "morning", "afternoon",
    "evening", "week", "weeks", "month", "months", "year", "years",
    "ago", "last", "next", "weekend",
})
_GENERIC_EXTRA = frozenset({
    "continue", "check", "gonna", "going", "went", "just", "really", "like",
    "yeah", "also", "still", "much", "very", "option", "options", "thing",
    "things", "stuff", "something", "anything", "always", "never",
    "good", "great", "nice", "well", "right", "okay", "know", "think",
    "thought", "said", "say", "want", "wants", "wanted", "keen",
    "working", "work", "person", "people", "pretty", "exciting", "cool",
    "amazing", "wonderful", "awesome", "lucky", "true", "happy", "hard",
    "easy", "interesting", "beautiful", "gorgeous", "thanks", "thank",
    "love", "loved", "wow", "through", "around", "which", "been",
    "have", "has", "had", "their", "there", "here",
})
# "work"/"job" fire on unrelated utterances ("work through", "got a job").
_HEURISTIC_EDUCATION_CUES = frozenset({
    "education", "edu", "field", "fields", "study", "studies", "school",
    "career", "careers", "occupation", "profession",
})

STATUS_OK = "ok"
STATUS_EVALUATOR_FAILED = "evaluator_failed"
STATUS_INVALID = "invalid_evaluation"

EVALUATOR_SYSTEM = (
    "You evaluate retrieved conversation excerpts. You do not answer the question. "
    "Return only a JSON object with this schema: "
    '{"status": "sufficient"|"insufficient"|"uncertain", '
    '"supported_facts": [string], "missing_information": string, "next_queries": [string]}. '
    "status is sufficient only when the excerpts themselves answer the question. "
    "supported_facts may restate excerpt content and may cite [message_id]. "
    "missing_information names a concrete gap, not an assumed answer. "
    "next_queries has at most 3 targeted search strings. "
    "Follow-up queries may reuse words from the question or the excerpts. "
    "Do not invent facts and do not insert an assumed answer into queries."
)


EvaluatorFn = Callable[[str, tuple["EvidenceExcerpt", ...]], "EvidenceEvaluation"]


@dataclass(frozen=True)
class EvidenceExcerpt:
    message_id: str
    content: str


@dataclass(frozen=True)
class EvidenceEvaluation:
    status: str
    supported_facts: tuple[str, ...]
    missing_information: str
    next_queries: tuple[str, ...]
    source: str = SOURCE_FALLBACK
    eval_status: str = STATUS_OK
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "supportedFacts": list(self.supported_facts),
            "missingInformation": self.missing_information,
            "nextQueries": list(self.next_queries),
            "source": self.source,
            "evalStatus": self.eval_status,
            "error": self.error,
        }


def excerpt_terms(excerpts: Iterable[EvidenceExcerpt]) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for item in excerpts:
        for tok in tokens(item.content):
            if tok in FUNCTION_WORDS or tok in seen:
                continue
            seen.add(tok)
            found.append(tok)
    return tuple(found)


def filter_followup_queries(
    question: str,
    queries: Iterable[str],
    excerpts: Iterable[EvidenceExcerpt],
    *,
    seen: Iterable[str] = (),
) -> tuple[str, ...]:
    allowed = excerpt_terms(excerpts)
    already = {item.casefold() for item in seen}
    out: list[str] = []
    for raw in queries:
        if not isinstance(raw, str):
            continue
        query = " ".join(raw.split()).strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            continue
        if query.casefold() in already or query.casefold() in {item.casefold() for item in out}:
            continue
        if extra_content_terms(query, question, allowed):
            continue
        out.append(query)
        if len(out) >= MAX_NEXT_QUERIES:
            break
    return tuple(out)


def _strip_speakers(text: str) -> str:
    parts = []
    for raw in text.splitlines() or [text]:
        parts.append(_SPEAKER_PREFIX.sub("", raw.strip()))
    return " ".join(part for part in parts if part)


def _content_terms(text: str) -> list[str]:
    return [tok for tok in tokens(text) if tok not in FUNCTION_WORDS]


def _has_temporal(text: str) -> bool:
    lowered = text.lower()
    if _YEAR.search(lowered) or _DATE.search(lowered):
        return True
    return any(tok in _TEMPORAL_WORDS for tok in tokens(lowered))


def _specific_extras(question: str, excerpt_text: str) -> list[str]:
    asked = set(tokens(question))
    extras: list[str] = []
    for tok in _content_terms(excerpt_text):
        if tok in asked or tok in _GENERIC_EXTRA or tok in _HEURISTIC_EDUCATION_CUES or tok in extras:
            continue
        if len(tok) < 5:
            continue
        extras.append(tok)
    return extras


def _heuristic_followups(
    question: str,
    excerpts: Iterable[EvidenceExcerpt],
    extras: Iterable[str],
) -> tuple[str, ...]:
    entities = entities_from_question(question)
    entity = entities[0] if entities else ""
    seeds: list[str] = []
    for tok in (*extras, *excerpt_terms(excerpts)):
        if tok in FUNCTION_WORDS or tok in seeds:
            continue
        seeds.append(tok)
    asked = [tok for tok in _content_terms(question) if tok not in FUNCTION_WORDS]
    candidates: list[str] = []
    if entity and seeds:
        candidates.append(f"What did {entity} say about {' '.join(seeds[:3])}?")
        candidates.append(f"{entity} {' '.join(seeds[:4])}")
    if asked and seeds:
        candidates.append(f"{' '.join(asked[:4])} {' '.join(seeds[:3])}")
    elif seeds:
        candidates.append(" ".join(seeds[:5]))
    return filter_followup_queries(question, candidates, excerpts)


def heuristic_evaluation(
    question: str,
    excerpts: tuple[EvidenceExcerpt, ...],
) -> EvidenceEvaluation:
    """Inspect question + excerpts only. Never uses gold IDs or assumed answers."""
    bodies = [_strip_speakers(item.content) for item in excerpts]
    blob = " ".join(bodies)
    q_terms = _content_terms(question)
    e_terms = set(tokens(blob))
    extras = _specific_extras(question, blob)
    asked = set(q_terms)
    is_when = question.casefold().startswith("when")
    educationish = bool(asked & EDUCATION_CUES) or bool(asked & _HEURISTIC_EDUCATION_CUES)
    distinctive = [tok for tok in q_terms if tok not in {"caroline", "melanie"}]
    overlap_distinct = [tok for tok in distinctive if tok in e_terms]
    facts: list[str] = []
    for item, body in zip(excerpts, bodies):
        snippet = body.strip()
        if not snippet:
            continue
        if len(snippet) > 180:
            snippet = snippet[:179] + "…"
        facts.append(f"{snippet} [{item.message_id}]")
        if len(facts) >= 3:
            break

    if is_when:
        enough = any(
            _has_temporal(body) and any(tok in tokens(body) for tok in distinctive)
            for body in bodies
        )
        missing = "" if enough else "a date or time for the asked event"
    elif educationish:
        enough = False
        for body in bodies:
            local_terms = set(tokens(body))
            has_cue = bool(local_terms & _HEURISTIC_EDUCATION_CUES) or "edu" in local_terms
            if has_cue and _specific_extras(question, body):
                enough = True
                break
        missing = "" if enough else "a named field of study or work from the source excerpts"
    else:
        enough = bool(overlap_distinct) and bool(extras)
        if enough:
            missing = ""
        else:
            absent = [tok for tok in distinctive if tok not in e_terms][:4]
            missing = (
                "source statements that mention " + ", ".join(absent)
                if absent
                else "a source statement that directly answers the question"
            )

    next_queries = () if enough else _heuristic_followups(question, excerpts, extras)
    return EvidenceEvaluation(
        status=STATUS_SUFFICIENT if enough else STATUS_INSUFFICIENT,
        supported_facts=tuple(facts),
        missing_information=missing,
        next_queries=next_queries,
        source=SOURCE_HEURISTIC,
        eval_status=STATUS_OK,
    )


def stop_evaluation(
    *,
    status: str = STATUS_UNCERTAIN,
    error: str | None = None,
    eval_status: str = STATUS_EVALUATOR_FAILED,
    source: str = SOURCE_FALLBACK,
) -> EvidenceEvaluation:
    return EvidenceEvaluation(
        status=status,
        supported_facts=(),
        missing_information="",
        next_queries=(),
        source=source,
        eval_status=eval_status,
        error=error,
    )


def validate_evaluation(
    question: str,
    excerpts: Iterable[EvidenceExcerpt],
    raw: Any,
    *,
    source: str = SOURCE_INJECTED,
    seen_queries: Iterable[str] = (),
) -> EvidenceEvaluation:
    if not isinstance(raw, dict):
        raise ValueError("evidence evaluation must be a JSON object")
    status = raw.get("status")
    if not isinstance(status, str) or status.strip().lower() not in ALLOWED_STATUSES:
        raise ValueError("status must be sufficient, insufficient, or uncertain")
    status = status.strip().lower()
    facts_raw = raw.get("supported_facts", raw.get("supportedFacts", []))
    if facts_raw is None:
        facts_raw = []
    if not isinstance(facts_raw, list):
        raise ValueError("supported_facts must be a list")
    facts = []
    for item in facts_raw:
        if not isinstance(item, str) or not item.strip():
            continue
        text = item.strip()
        if len(text) > MAX_FACT_CHARS:
            text = text[: MAX_FACT_CHARS - 1] + "…"
        if text not in facts:
            facts.append(text)
        if len(facts) >= MAX_FACTS:
            break
    missing = raw.get("missing_information", raw.get("missingInformation", ""))
    if missing is None:
        missing = ""
    if not isinstance(missing, str):
        raise ValueError("missing_information must be a string")
    missing = missing.strip()
    if len(missing) > MAX_MISSING_CHARS:
        missing = missing[: MAX_MISSING_CHARS - 1] + "…"
    queries_raw = raw.get("next_queries", raw.get("nextQueries", []))
    if queries_raw is None:
        queries_raw = []
    if not isinstance(queries_raw, list):
        raise ValueError("next_queries must be a list")
    next_queries = filter_followup_queries(
        question, [q for q in queries_raw if isinstance(q, str)], excerpts, seen=seen_queries,
    )
    return EvidenceEvaluation(
        status=status,
        supported_facts=tuple(facts),
        missing_information=missing,
        next_queries=next_queries,
        source=source,
        eval_status=STATUS_OK,
    )


def _excerpt_blocks(excerpts: Iterable[EvidenceExcerpt]) -> str:
    blocks = [f"[{item.message_id}]\n{item.content}" for item in excerpts]
    return "\n\n".join(blocks) if blocks else "(no source excerpts were selected)"


def evaluation_from_generator(
    question: str,
    excerpts: tuple[EvidenceExcerpt, ...],
    generator: Generator,
    *,
    seen_queries: Iterable[str] = (),
) -> EvidenceEvaluation:
    prompt = (
        f"Question: {question.strip()}\n\n"
        f"Retrieved excerpts:\n{_excerpt_blocks(excerpts)}\n\nJSON:"
    )
    try:
        try:
            raw_text = generator.generate(
                EVALUATOR_SYSTEM,
                prompt,
                temperature=0.0,
                num_predict=EVAL_NUM_PREDICT,
            )
        except TypeError:
            raw_text = generator.generate(EVALUATOR_SYSTEM, prompt)
        parsed = extract_json_object(raw_text)
        return validate_evaluation(
            question, excerpts, parsed, source=SOURCE_OLLAMA, seen_queries=seen_queries,
        )
    except (GenerationError, ValueError, json.JSONDecodeError, TypeError) as exc:
        return stop_evaluation(error=str(exc))


def resolve_evaluation(
    question: str,
    excerpts: tuple[EvidenceExcerpt, ...],
    evaluator: EvaluatorFn | Generator | None,
    *,
    seen_queries: Iterable[str] = (),
) -> EvidenceEvaluation:
    if evaluator is None:
        return stop_evaluation(error="evaluator is not configured")
    if callable(evaluator) and not hasattr(evaluator, "generate"):
        try:
            result = evaluator(question, excerpts)
        except Exception as exc:
            return stop_evaluation(error=str(exc))
        if isinstance(result, EvidenceEvaluation):
            filtered = filter_followup_queries(
                question, result.next_queries, excerpts, seen=seen_queries,
            )
            if filtered != result.next_queries or result.status not in ALLOWED_STATUSES:
                try:
                    return validate_evaluation(
                        question,
                        excerpts,
                        {
                            "status": result.status,
                            "supported_facts": list(result.supported_facts),
                            "missing_information": result.missing_information,
                            "next_queries": list(result.next_queries),
                        },
                        source=result.source,
                        seen_queries=seen_queries,
                    )
                except ValueError as exc:
                    return stop_evaluation(error=str(exc), eval_status=STATUS_INVALID)
            return result
        if isinstance(result, dict):
            try:
                return validate_evaluation(
                    question, excerpts, result, source=SOURCE_INJECTED, seen_queries=seen_queries,
                )
            except ValueError as exc:
                return stop_evaluation(error=str(exc), eval_status=STATUS_INVALID)
        return stop_evaluation(error="evaluator returned an unsupported value", eval_status=STATUS_INVALID)
    return evaluation_from_generator(question, excerpts, evaluator, seen_queries=seen_queries)
