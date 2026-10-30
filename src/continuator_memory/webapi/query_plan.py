"""Structured retrieval query plans. The planner restates the information need; it does not answer.

Ollama is optional. Malformed model output falls back to a deterministic paraphrase plan
built only from the question and a closed paraphrase lexicon — never from guessed answers.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from continuator_memory.generation import GenerationError, Generator

MIN_QUERIES = 1
MAX_ALTERNATIVES = 2
MAX_QUERIES = 1 + MAX_ALTERNATIVES
MAX_QUERY_CHARS = 400
MAX_NEED_CHARS = 400
MAX_ENTITIES = 8
MAX_ENTITY_CHARS = 80
MAX_FACETS = 8
MAX_FACET_CHARS = 40

PLANNER_NUM_PREDICT = 400

SOURCE_OLLAMA = "ollama"
SOURCE_FALLBACK = "fallback"
SOURCE_INJECTED = "injected"

STATUS_OK = "ok"
STATUS_FALLBACK = "fallback"
STATUS_PLANNER_FAILED = "planner_failed"
STATUS_INVALID = "invalid_plan"

_TOKEN = re.compile(r"[a-z0-9']+")
_NAME = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_QUESTION_START = frozenset({
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "would", "could", "should", "does", "did", "do", "is", "are", "was", "were",
})

# Words a paraphrase may add. Specific answers (jobs, fields, hobbies) are not in this set.
PARAPHRASE_LEXICON = frozenset({
    "career", "careers", "job", "jobs", "work", "working", "occupation", "profession",
    "professional", "study", "studies", "studying", "education", "educational",
    "field", "fields", "interest", "interests", "interested", "plan", "plans",
    "planning", "goal", "goals", "pursue", "pursuing", "train", "training",
    "role", "roles", "learn", "learning", "school", "want", "wants", "wanted",
    "kind", "type", "types", "person", "people", "someone", "they", "their", "them",
    "say", "said", "about", "relevant", "facts", "fact", "stated", "statement",
    "answer", "answers", "answered", "question", "did", "does", "do", "is", "are",
    "was", "were",
    "what", "which", "who", "where", "when", "why", "how", "this", "that",
    "for", "to", "in", "of", "or", "and", "the", "a", "an", "her", "his",
    "she", "he", "likely", "would", "be", "to",
    "date", "time", "day", "place", "location",
})

FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with", "by",
    "from", "at", "as", "into", "about", "her", "his", "she", "he", "they",
    "their", "them", "it", "its", "be", "is", "are", "was", "were", "been",
    "being", "do", "does", "did", "would", "could", "should", "might", "may",
    "can", "will", "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "this", "that", "these", "those", "any", "some",
})

ALLOWED_FACETS = frozenset({
    "education", "career", "work", "job", "study", "people", "person", "time",
    "place", "preference", "plan", "goal", "relationship", "event", "identity",
    "statement", "fact", "interest",
})

EDUCATION_CUES = frozenset({
    "education", "edu", "field", "fields", "study", "studies", "school",
    "career", "careers", "job", "jobs", "work", "occupation", "profession",
})
# "school" is often a place. Career routing requires a topic cue, not a venue word.
EDUCATION_TOPIC_CUES = frozenset({
    "education", "edu", "field", "fields", "study", "studies",
    "career", "careers", "job", "jobs", "occupation", "profession",
})
EDUCATION_REWRITE_CUES = EDUCATION_TOPIC_CUES | frozenset({
    "work", "working", "train", "training", "school",
})
ANSWER_TYPES = frozenset({
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
})
ANSWER_TYPE_MARKERS = {
    "when": frozenset({"when", "date", "time", "day"}),
    "where": frozenset({"where", "place", "location"}),
    "who": frozenset({"who", "whom", "whose"}),
    "why": frozenset({"why", "reason"}),
    "how": frozenset({"how"}),
    "what": frozenset({"what", "which"}),
    "which": frozenset({"what", "which"}),
    "whom": frozenset({"who", "whom"}),
    "whose": frozenset({"who", "whose"}),
}
WEAK_EVENT_VERBS = frozenset({
    "go", "going", "went", "gone", "get", "got", "give", "gave", "given",
    "make", "made", "take", "took", "come", "came", "run", "ran",
})
_GENERIC_REWRITE = (
    "answers this question",
    "stated facts",
    "are relevant",
)
_LEAD = re.compile(
    r"^(?P<atype>what|which|who|whom|whose|where|when|why|how)\b"
    r"(?P<mid>.*?)\b(?P<aux>did|does|do|is|are|was|were|would|could|should|will|has|have|had)\b\s*",
    re.IGNORECASE,
)

PLANNER_SYSTEM = (
    "You write retrieval query plans. You do not answer the question. "
    "Return only a JSON object with this schema: "
    '{"information_need": string, "entities": [string], "facets": [string], "queries": [string]}. '
    "information_need restates what evidence to find, not the answer. "
    "entities are names or referents that already appear in the question. "
    "facets are types of evidence (education, career, work, people, time, event), not specific answers. "
    "queries must start with the user's question exactly, then at most two alternatives. "
    "Each alternative must keep the same subject, the same event or topic, and the same "
    "requested answer type (when, where, who, what, why). "
    "If the question asks when an event happened, alternatives must still ask for a date or time "
    "and must name that event. "
    "Do not write generic speaker-level queries such as "
    "'what did they say that answers this question' or 'which stated facts are relevant'. "
    "Do not treat a place word such as school as a career or education question when the "
    "event is something else, for example a speech. "
    "Do not name jobs, fields of study, hobbies, or other answers that are not already in the question. "
    "If you cannot write a safe alternative, return only the original question. "
    "Do not use outside knowledge."
)


PlannerFn = Callable[[str], "QueryPlan"]


@dataclass(frozen=True)
class QuestionIntent:
    answer_type: str
    entities: tuple[str, ...]
    event_terms: tuple[str, ...]
    education: bool
    remainder: str


@dataclass(frozen=True)
class QueryPlan:
    information_need: str
    entities: tuple[str, ...]
    facets: tuple[str, ...]
    queries: tuple[str, ...]
    source: str = SOURCE_FALLBACK
    status: str = STATUS_OK
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "informationNeed": self.information_need,
            "entities": list(self.entities),
            "facets": list(self.facets),
            "queries": list(self.queries),
            "source": self.source,
            "status": self.status,
            "error": self.error,
        }


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("\u2019", "'"))


def extra_content_terms(
    query: str,
    question: str,
    extra_allowed: Iterable[str] = (),
) -> tuple[str, ...]:
    """Content words in ``query`` that are not in the question and not allowed paraphrases."""
    allowed = set(tokens(question)) | PARAPHRASE_LEXICON | FUNCTION_WORDS | {
        tok.lower().replace("\u2019", "'") for tok in extra_allowed
    }
    for tok in tokens(question):
        if tok.endswith("'s") and len(tok) > 2:
            allowed.add(tok[:-2])
    for entity in entities_from_question(question):
        allowed.update(tokens(entity))
    extra = []
    for tok in tokens(query):
        if tok not in allowed and tok not in extra:
            extra.append(tok)
    return tuple(extra)


def entities_from_question(question: str) -> tuple[str, ...]:
    found = []
    for match in _NAME.finditer(question):
        name = match.group(1)
        if name.lower() in _QUESTION_START:
            continue
        if name not in found:
            found.append(name)
        if len(found) >= MAX_ENTITIES:
            break
    return tuple(found)


def _entity_tokens(entities: Iterable[str]) -> set[str]:
    found: set[str] = set()
    for entity in entities:
        for tok in tokens(entity):
            found.add(tok)
            found.add(tok.rstrip("s"))
    return found


def _is_education_intent(toks: set[str], answer_type: str) -> bool:
    if toks & EDUCATION_TOPIC_CUES:
        return True
    if "school" in toks and answer_type in {"what", "which"}:
        return not (toks & {"speech", "talk", "lecture", "visit", "game", "race"})
    return False


def analyze_intent(question: str) -> QuestionIntent:
    text = question.strip()
    toks = tokens(text)
    answer_type = toks[0] if toks and toks[0] in ANSWER_TYPES else "other"
    entities = entities_from_question(text)
    match = _LEAD.match(text)
    mid = match.group("mid") if match else ""
    if match:
        remainder = text[match.end():].rstrip(" ?")
    else:
        remainder = re.sub(
            r"^(what|which|who|whom|whose|where|when|why|how)\b\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).rstrip(" ?")
    entity_toks = _entity_tokens(entities)
    event: list[str] = []
    for tok in tokens(f"{mid} {remainder}"):
        stem = tok.rstrip("'s")
        if tok in FUNCTION_WORDS or tok in entity_toks or stem in entity_toks:
            continue
        if tok in WEAK_EVENT_VERBS:
            continue
        if tok not in event:
            event.append(tok)
    return QuestionIntent(
        answer_type=answer_type,
        entities=entities,
        event_terms=tuple(event),
        education=_is_education_intent(set(toks), answer_type),
        remainder=remainder,
    )


def facets_from_question(question: str) -> tuple[str, ...]:
    intent = analyze_intent(question)
    facets: list[str] = []
    if intent.education:
        facets.extend(["education", "career", "work"])
    if intent.answer_type == "when":
        facets.append("time")
    if intent.answer_type == "where":
        facets.append("place")
    if intent.event_terms and not intent.education:
        facets.append("event")
    if not facets:
        facets.append("fact")
    return tuple(dict.fromkeys(facets))[:MAX_FACETS]


def _swap_question_word(question: str, new_lead: str) -> str | None:
    text = question.strip()
    match = re.match(r"^(what|which|who|whom|whose|where|when|why|how)\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    rest = text[match.end():].lstrip()
    if not rest:
        return None
    out = f"{new_lead} {rest}"
    if not out.endswith("?"):
        out += "?"
    return out


def _rewrite_candidates(question: str, intent: QuestionIntent) -> list[str]:
    name = intent.entities[0] if intent.entities else "they"
    event_phrase = " ".join(intent.event_terms)
    if intent.education:
        return [
            f"What career is {name} interested in?",
            f"What kind of work does {name} want to do?",
        ]
    if intent.answer_type == "when":
        return [
            item for item in (
                _swap_question_word(question, "What date"),
                _swap_question_word(question, "What time"),
            )
            if item
        ]
    if intent.answer_type == "where":
        return [item for item in (_swap_question_word(question, "What place"),) if item]
    if event_phrase:
        return [f"What did {name} say about {event_phrase}?"]
    return []


def _preserves_answer_type(query: str, intent: QuestionIntent) -> bool:
    markers = ANSWER_TYPE_MARKERS.get(intent.answer_type)
    if not markers:
        return True
    return bool(set(tokens(query)) & markers)


def _preserves_event(query: str, intent: QuestionIntent) -> bool:
    qtoks = set(tokens(query))
    if intent.education:
        return bool(qtoks & EDUCATION_REWRITE_CUES)
    needed = [tok for tok in intent.event_terms if tok not in WEAK_EVENT_VERBS]
    if not needed:
        return True
    distinctive = [tok for tok in needed if len(tok) >= 4]
    return all(tok in qtoks for tok in (distinctive or needed))


def _is_generic_rewrite(query: str) -> bool:
    lowered = query.casefold()
    return any(marker in lowered for marker in _GENERIC_REWRITE)


def is_safe_rewrite(query: str, question: str, intent: QuestionIntent | None = None) -> bool:
    intent = intent or analyze_intent(question)
    if extra_content_terms(query, question):
        return False
    if _is_generic_rewrite(query):
        return False
    if not _preserves_event(query, intent):
        return False
    if not _preserves_answer_type(query, intent):
        return False
    return True


def fallback_queries(question: str, entities: tuple[str, ...] | None = None) -> tuple[str, ...]:
    intent = analyze_intent(question)
    if entities is None:
        entities = intent.entities
    elif entities != intent.entities:
        intent = QuestionIntent(
            answer_type=intent.answer_type,
            entities=tuple(entities) or intent.entities,
            event_terms=intent.event_terms,
            education=intent.education,
            remainder=intent.remainder,
        )
    return _finalize_queries(question, _rewrite_candidates(question, intent), intent=intent)


def fallback_plan(question: str, *, status: str = STATUS_FALLBACK, error: str | None = None) -> QueryPlan:
    text = question.strip()
    entities = entities_from_question(text)
    return QueryPlan(
        information_need=f"Find evidence that answers: {text}",
        entities=entities,
        facets=facets_from_question(text),
        queries=fallback_queries(text, entities),
        source=SOURCE_FALLBACK,
        status=status,
        error=error,
    )


def _finalize_queries(
    original: str,
    candidates: list[str],
    *,
    intent: QuestionIntent | None = None,
) -> tuple[str, ...]:
    intent = intent or analyze_intent(original)
    seen: set[str] = set()
    out: list[str] = []
    for raw in [original, *candidates]:
        query = " ".join(raw.split()).strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            continue
        key = query.casefold()
        if key in seen:
            continue
        if key != original.strip().casefold() and not is_safe_rewrite(query, original, intent):
            continue
        seen.add(key)
        out.append(query)
        if len(out) >= MAX_QUERIES:
            break
    if not out or out[0].casefold() != original.strip().casefold():
        out = [original.strip(), *[q for q in out if q.casefold() != original.strip().casefold()]]
    return tuple(out[:MAX_QUERIES])


def validate_plan(question: str, raw: Any, *, source: str = SOURCE_INJECTED) -> QueryPlan:
    if not isinstance(raw, dict):
        raise ValueError("query plan must be a JSON object")
    need = raw.get("information_need", raw.get("informationNeed"))
    if not isinstance(need, str) or not need.strip() or len(need) > MAX_NEED_CHARS:
        raise ValueError("information_need must be a non-empty string")
    entities = _read_entities(raw)
    facets = _read_facets(raw, question)
    queries_raw = raw.get("queries")
    if not isinstance(queries_raw, list) or not queries_raw:
        raise ValueError("queries must be a non-empty list")
    queries = _finalize_queries(question, [q for q in queries_raw if isinstance(q, str)])
    if not queries:
        raise ValueError("queries must include the original question")
    return QueryPlan(
        information_need=need.strip(),
        entities=entities,
        facets=facets,
        queries=queries,
        source=source,
        status=STATUS_OK,
    )


def _read_entities(raw: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    if isinstance(raw.get("entity"), str) and raw["entity"].strip():
        values.append(raw["entity"].strip())
    ents = raw.get("entities")
    if isinstance(ents, list):
        for item in ents:
            if isinstance(item, str) and item.strip():
                values.append(item.strip())
    cleaned = []
    for item in values:
        if len(item) > MAX_ENTITY_CHARS:
            continue
        if item not in cleaned:
            cleaned.append(item)
        if len(cleaned) >= MAX_ENTITIES:
            break
    return tuple(cleaned)


def _read_facets(raw: dict[str, Any], question: str) -> tuple[str, ...]:
    values: list[str] = []
    facets = raw.get("facets")
    if isinstance(facets, list):
        for item in facets:
            if not isinstance(item, str):
                continue
            key = item.strip().lower()
            if key in ALLOWED_FACETS and key not in values:
                values.append(key)
            if len(values) >= MAX_FACETS:
                break
    return tuple(values) or facets_from_question(question)


def extract_json_object(text: str) -> Any:
    blob = text.strip()
    if blob.startswith("```"):
        parts = blob.split("```")
        blob = parts[1] if len(parts) > 1 else blob
        if blob.lstrip().lower().startswith("json"):
            blob = blob.lstrip()[4:]
    start, end = blob.find("{"), blob.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("planner output is not a JSON object")
    return json.loads(blob[start : end + 1])


def plan_from_generator(question: str, generator: Generator) -> QueryPlan:
    prompt = f"Question: {question.strip()}\nJSON:"
    try:
        try:
            raw_text = generator.generate(
                PLANNER_SYSTEM,
                prompt,
                temperature=0.0,
                num_predict=PLANNER_NUM_PREDICT,
            )
        except TypeError:
            raw_text = generator.generate(PLANNER_SYSTEM, prompt)
        parsed = extract_json_object(raw_text)
        return validate_plan(question, parsed, source=SOURCE_OLLAMA)
    except (GenerationError, ValueError, json.JSONDecodeError, TypeError) as exc:
        return fallback_plan(question, status=STATUS_PLANNER_FAILED, error=str(exc))


def load_planner_model() -> str:
    env = os.environ
    return (
        env.get("CONTINUATOR_QUERY_PLAN_MODEL")
        or env.get("CONTINUATOR_ANSWER_MODEL")
        or env.get("CONTINUATOR_OLLAMA_MODEL")
        or ""
    ).strip()


def resolve_plan(question: str, planner: PlannerFn | Generator | None) -> QueryPlan:
    if planner is None:
        return fallback_plan(question)
    if callable(planner) and not hasattr(planner, "generate"):
        try:
            plan = planner(question)
        except Exception as exc:
            return fallback_plan(question, status=STATUS_PLANNER_FAILED, error=str(exc))
        if isinstance(plan, QueryPlan):
            try:
                cleaned = validate_plan(question, plan.to_dict(), source=plan.source)
            except ValueError as exc:
                return fallback_plan(question, status=STATUS_INVALID, error=str(exc))
            return QueryPlan(
                information_need=cleaned.information_need,
                entities=cleaned.entities or plan.entities,
                facets=cleaned.facets,
                queries=cleaned.queries,
                source=plan.source,
                status=plan.status,
                error=plan.error,
            )
        if isinstance(plan, dict):
            try:
                return validate_plan(question, plan, source=SOURCE_INJECTED)
            except ValueError as exc:
                return fallback_plan(question, status=STATUS_INVALID, error=str(exc))
        return fallback_plan(question, status=STATUS_INVALID, error="planner returned an unsupported value")
    return plan_from_generator(question, planner)
