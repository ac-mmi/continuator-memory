"""Structured retrieval query plans. The planner restates the information need; it does not answer.

Ollama is optional. Malformed model output falls back to a deterministic paraphrase plan
built only from the question and a closed paraphrase lexicon — never from guessed answers.

Planner alternatives are filtered by a layered rewrite validator (structure, answer type,
entities, event/focus, constraints, MiniLM low-similarity veto). The closed paraphrase
lexicon is only for fallback template generation, not as an accept/reject allowlist.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from continuator_memory.generation import GenerationError, Generator

MIN_QUERIES = 1
MAX_ALTERNATIVES = 2
MAX_QUERIES = 1 + MAX_ALTERNATIVES
MAX_QUERY_CHARS = 400
MIN_QUERY_CHARS = 3
MAX_NEED_CHARS = 400
MAX_ENTITIES = 8
MAX_ENTITY_CHARS = 80
MAX_FACETS = 8
MAX_FACET_CHARS = 40

# Provisional low-similarity veto from LoCoMo rewrite pilots. Not an acceptance threshold.
MIN_REWRITE_SIMILARITY = 0.65

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
_WHAT_HEAD = re.compile(
    r"^(?P<wh>what|which)\s+(?P<head>[\w']+)(?:\s+(?P<head2>[\w']+))?\s+"
    r"(?P<aux>did|does|do|is|are|was|were|will|would|has|have|had|can|could)\b",
    re.IGNORECASE,
)
_APPOSITIVE_WHO = re.compile(r",\s*who\s+(?:is|are|was|were)\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTHS = frozenset({
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
})
_TEMPORAL_RELATIVE = frozenset({
    "next", "last", "previous", "earlier", "later", "upcoming", "another",
})
_YES_NO_LEADS = frozenset({
    "is", "are", "was", "were", "do", "does", "did", "has", "have", "had",
    "would", "could", "should", "can", "will",
})
# Meta question operators — allow paraphrases like "what time/date/purpose…".
# Not a closed world for all content words (verbs such as "attend" are allowed separately).
_QUESTION_OPERATORS = frozenset({
    "time", "date", "day", "place", "location", "purpose", "meaning", "reason",
    "frequency", "duration", "activity", "activities", "emotion", "emotions",
    "experience", "experiences", "genre", "genres", "theme", "themes",
    "subject", "subjects", "method", "methods", "type", "types", "kind", "kinds",
    "career", "path", "job", "jobs", "work", "role", "roles", "favorite", "favorites",
    "year", "years", "name", "names", "aspect", "aspects", "reaction", "reactions",
    "feeling", "feelings", "exercise", "hobby", "hobbies", "condition", "conditions",
    "plan", "plans", "goal", "goals", "offer", "offers", "thing", "things",
    "event", "events", "area", "areas", "city", "cities", "step", "steps",
    "detail", "details", "process", "way", "ways", "factor", "factors",
})
_DROPPABLE_EVENT = WEAK_EVENT_VERBS | frozenset({
    "watching", "creating", "making", "taking", "going", "getting", "having",
    "feeling", "feel", "felt", "loving", "love", "liked", "likes", "enjoy",
    "enjoys", "enjoyed", "doing", "saying", "said", "start", "started",
    "starting", "begin", "began", "beginning", "stop", "stopped", "stopping",
    "learning", "studying", "describe", "describes", "described", "mention",
    "mentioned", "mentions", "stated", "states", "believe", "believes",
    "thought", "think", "thinks", "makes", "make", "made", "helps", "helped",
    "help", "benefited", "benefit", "serves", "serve", "using", "use", "used",
    "adding", "add", "added", "incorporate", "incorporating", "produce",
    "producing", "create", "created", "drives", "drive", "capture", "captured",
    "arrange", "arranged", "organizing", "organize", "organized", "finalizing",
    "finalize", "sealing", "seal", "sealed", "dealing", "deal", "adopt",
    "adopted", "adopting", "travel", "traveled", "travelling", "traveling",
    "spend", "spending", "spent", "get", "got", "getting",     "recommend",
    "recommended", "reccomended", "suggest", "suggested", "symbolize",
    "symbolizes", "explore", "explores", "explored", "performing", "perform",
    "remain", "remaining", "stay", "staying", "engage", "engages", "engaged",
    "plan", "plans", "planning", "planned", "receive", "received", "receiving",
    "have", "has", "had", "while", "something", "often", "one", "ones",
    "process", "description", "according", "own", "signed", "sign", "suffering",
    "suffer",
})
# Adjectives/adverbs that paraphrases often swap without changing the need.
_DROPPABLE_MODIFIERS = frozenset({
    "stunning", "amazing", "safe", "fun", "enjoyable", "secure", "suspected",
    "possible", "shared", "common", "physical", "electronic", "extraordinary",
    "exceptional", "long", "first", "main", "favorite", "favorites",
    # Weak "spending time with X" scaffolding — the companion entity is the anchor.
    "time", "times",
    # WH scaffolding often restated with operators (kind/type/genres…).
    "kind", "kinds", "type", "types",
    "particularly", "looking", "away",
})
# Optional venue/context nouns that paraphrases may drop while keeping the focus.
_OPTIONAL_EVENT = frozenset({
    "workshop", "session", "class", "meeting", "conference", "office",
})
# Light stem aliases so clear paraphrases share an event anchor.
_STEM_ALIASES = {
    "photo": "pictur", "photos": "pictur", "picture": "pictur", "pictures": "pictur",
    "movie": "film", "movies": "film", "film": "film", "films": "film",
    "trip": "travel", "trips": "travel", "travel": "travel", "visit": "travel",
    "visits": "travel", "hobby": "activ", "hobbies": "activ", "activity": "activ",
    "activities": "activ", "exercise": "activ", "exercises": "activ",
    "sport": "activ", "sports": "activ",
    "song": "music", "songs": "music", "music": "music",
    "problem": "condit", "problems": "condit", "condition": "condit",
    "conditions": "condit", "health": "medic", "medical": "medic",
    "reaction": "feel", "reactions": "feel",
    "agreement": "deal", "deal": "deal", "space": "environ", "environment": "environ",
    "workday": "day", "injury": "knee", "recovery": "knee",
    "checkup": "checkup", "checkups": "checkup",
    "court": "play", "playing": "play", "play": "play",
}

_REWRITE_EMBEDDER = None

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
class RewriteDecision:
    accepted: bool
    reason: str
    similarity: float | None = None
    details: tuple[str, ...] = ()


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


def _norm_token(tok: str) -> str:
    text = tok.casefold().replace("\u2019", "'")
    if text.endswith("'s") and len(text) > 2:
        text = text[:-2]
    return text


def _stem(tok: str) -> str:
    text = _norm_token(tok)
    if text in _STEM_ALIASES:
        return _STEM_ALIASES[text]
    for suffix in ("tion", "sion", "ing", "ers", "er", "ed", "es", "s", "ly"):
        if len(text) > len(suffix) + 3 and text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return _STEM_ALIASES.get(text, text)


def _token_set(text: str) -> set[str]:
    return {_norm_token(tok) for tok in tokens(text)}


def _stem_set(text: str) -> set[str]:
    return {_stem(tok) for tok in tokens(text)}


def _content_token_list(text: str) -> list[str]:
    return [
        _norm_token(tok)
        for tok in tokens(text)
        if _norm_token(tok) not in FUNCTION_WORDS
    ]


def _near_typo_match(tok: str, candidates: set[str]) -> bool:
    """Allow one-character typos for longer tokens (reccomended ≈ recommended)."""
    if len(tok) < 6:
        return False
    for other in candidates:
        if abs(len(other) - len(tok)) > 1 or len(other) < 6:
            continue
        # Hamming-ish: count mismatches on zipped prefixes + length delta.
        mismatches = abs(len(other) - len(tok))
        for a, b in zip(tok, other):
            if a != b:
                mismatches += 1
                if mismatches > 1:
                    break
        if mismatches <= 1:
            return True
    return False


def _question_intent_key(text: str) -> str:
    toks = tokens(text)
    if not toks:
        return "other"
    lead = toks[0]
    if lead in ANSWER_TYPES:
        if lead == "how" and len(toks) > 1:
            if toks[1] == "many":
                return "how_many"
            if toks[1] == "often":
                return "how_often"
            if toks[1] == "long":
                return "how_long"
        return lead
    if lead in _YES_NO_LEADS:
        return "yes_no"
    return "other"


def _preserves_answer_type(query: str, intent: QuestionIntent) -> bool:
    markers = ANSWER_TYPE_MARKERS.get(intent.answer_type)
    if not markers:
        return True
    return bool(set(tokens(query)) & markers)


def _answer_types_compatible(original_key: str, query_key: str, intent: QuestionIntent, query: str) -> bool:
    if original_key == query_key:
        return True
    # Statements / non-questions may not drop a WH/yes-no intent.
    if query_key == "other" and original_key != "other":
        if intent.answer_type in ANSWER_TYPE_MARKERS:
            return _preserves_answer_type(query, intent)
        return False
    if original_key == "other":
        # yes/no (and similar) must not become an unrelated WH question.
        if query_key in ANSWER_TYPES or query_key.startswith("how_"):
            return False
        return True
    # Common safe paraphrases: when→what date/time, where→what place, how→what …
    compatible = {
        frozenset({"when", "what"}),
        frozenset({"where", "what"}),
        frozenset({"how", "what"}),
        frozenset({"how_often", "what"}),
        frozenset({"how_long", "what"}),
        frozenset({"what", "which"}),
    }
    if frozenset({original_key, query_key}) in compatible:
        if intent.answer_type in {"when", "where"}:
            return _preserves_answer_type(query, intent)
        return True
    if intent.answer_type in ANSWER_TYPE_MARKERS:
        return _preserves_answer_type(query, intent)
    return False


def _preserves_event(query: str, intent: QuestionIntent) -> bool:
    qtoks = _token_set(query)
    qstems = _stem_set(query)
    if intent.education:
        return bool(qtoks & EDUCATION_REWRITE_CUES)
    needed = [
        tok for tok in intent.event_terms
        if tok not in _DROPPABLE_EVENT
        and tok not in _DROPPABLE_MODIFIERS
        and tok not in _OPTIONAL_EVENT
        and len(tok) >= 3
        and not tok.isdigit()
        and not _YEAR.fullmatch(tok)
    ]
    if not needed:
        return True

    def _hit(tok: str) -> bool:
        if tok in qtoks or _stem(tok) in qstems:
            return True
        return _near_typo_match(tok, qtoks) or _near_typo_match(_stem(tok), qstems)

    hits = [tok for tok in needed if _hit(tok)]
    # Short needs are brittle: require full coverage (meet≠saw, dogs≠pets).
    if len(needed) <= 3:
        return len(hits) == len(needed)
    # Longer needs: every long anchor must survive; require ~2/3 overall coverage.
    anchors = [tok for tok in needed if len(tok) >= 7]
    if anchors and not all(_hit(tok) for tok in anchors):
        return False
    return len(hits) >= max(1, (len(needed) * 2 + 2) // 3)


def _is_generic_rewrite(query: str) -> bool:
    lowered = query.casefold()
    return any(marker in lowered for marker in _GENERIC_REWRITE)


def _entities_preserved(question: str, query: str, intent: QuestionIntent) -> str | None:
    """Return a rejection reason if named entities are not preserved."""
    original_entities = intent.entities or entities_from_question(question)
    query_entities = entities_from_question(query)
    query_toks = _token_set(query)
    for entity in original_entities:
        parts = [_norm_token(tok) for tok in tokens(entity)]
        if not parts:
            continue
        if not all(part in query_toks or f"{part}'s" in query.casefold() for part in parts):
            # Allow full multi-word entity as contiguous casefold substring.
            if entity.casefold() not in query.casefold() and entity.casefold().rstrip("s") not in query.casefold():
                return "missing_entity"
    original_norm = {
        " ".join(_norm_token(tok) for tok in tokens(entity))
        for entity in original_entities
    }
    original_parts = {_norm_token(tok) for entity in original_entities for tok in tokens(entity)}
    for entity in query_entities:
        key = " ".join(_norm_token(tok) for tok in tokens(entity))
        parts = {_norm_token(tok) for tok in tokens(entity)}
        if key in original_norm:
            continue
        if parts and parts <= original_parts:
            continue
        # New capitalized name-like span not present in the original.
        if entity.casefold() not in question.casefold():
            return "new_entity"
    return None


def _temporal_markers(text: str) -> set[str]:
    found = {m.group(0) for m in _YEAR.finditer(text)}
    for tok in tokens(text):
        norm = _norm_token(tok)
        if norm in _MONTHS or norm in _TEMPORAL_RELATIVE:
            found.add(norm)
        # Bare day-of-month numbers are ambiguous; keep numeric tokens near months/years
        # via the explicit year/month/relative set only.
    # Day numbers that appear beside a month (e.g. "May 3, 2023").
    for match in re.finditer(
        r"\b(?:%s)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:%s)\b"
        % ("|".join(sorted(_MONTHS, key=len, reverse=True)),
           "|".join(sorted(_MONTHS, key=len, reverse=True))),
        text,
        flags=re.IGNORECASE,
    ):
        day = match.group(1) or match.group(2)
        if day:
            found.add(f"day:{int(day)}")
    return found


def _constraints_preserved(question: str, query: str) -> str | None:
    if query.count("?") > 1:
        return "compound_query"
    if _APPOSITIVE_WHO.search(query) and not _APPOSITIVE_WHO.search(question):
        return "added_constraint"
    q_lower = query.casefold()
    o_lower = question.casefold()
    for marker in (" other than ", " instead of ", " rather than ", " as opposed to "):
        if marker in q_lower and marker not in o_lower:
            return "added_constraint"
    # Reject clear tense/aspect flips of the information need (plan/future → already started).
    original_future = any(tok in _token_set(question) for tok in ("plan", "plans", "planning", "will", "going"))
    candidate_started = any(
        tok in _token_set(query) for tok in ("started", "start", "begun", "already", "completed")
    )
    if original_future and candidate_started and not any(
        tok in _token_set(question) for tok in ("started", "start", "begun", "already")
    ):
        return "added_constraint"
    q_temporal = _temporal_markers(question)
    c_temporal = _temporal_markers(query)
    if c_temporal != q_temporal:
        return "temporal_constraint"
    # Reject open what/which rewrites that insert a non-operator focus noun
    # ("what games does…" narrowing "what does…").
    match = _WHAT_HEAD.match(query.strip())
    if match:
        heads = [_norm_token(match.group("head"))]
        if match.group("head2"):
            heads.append(_norm_token(match.group("head2")))
        original = _token_set(question)
        original_stems = _stem_set(question)
        # "what physical exercise has…" — modifier + operator is fine.
        if (
            len(heads) >= 2
            and heads[0] in _DROPPABLE_MODIFIERS
            and (heads[1] in _QUESTION_OPERATORS or heads[1] in original or _stem(heads[1]) in original_stems)
        ):
            return None
        for head in heads:
            if head in FUNCTION_WORDS or head in _YES_NO_LEADS:
                continue
            if head in original or head in _QUESTION_OPERATORS or _stem(head) in original_stems:
                continue
            if head in _DROPPABLE_MODIFIERS:
                continue
            if head.rstrip("s") in original or head.rstrip("s") in _QUESTION_OPERATORS:
                continue
            if head in {"of"}:
                continue
            return "added_focus"
    return None


def _ordered_content_subsequence(question: str, query: str) -> bool:
    """True when original content tokens appear in order inside the candidate."""
    needle = _content_token_list(question)
    hay = _content_token_list(query)
    if not needle:
        return False
    i = 0
    for tok in hay:
        if i < len(needle) and tok == needle[i]:
            i += 1
    return i == len(needle)


def _insertion_elaboration(question: str, query: str) -> bool:
    """Reject near-copies that only insert new non-operator content (e.g. appended guesses)."""
    if not _ordered_content_subsequence(question, query):
        return False
    original = _token_set(question)
    original_stems = _stem_set(question)
    novel = []
    for tok in _content_token_list(query):
        if tok in original or _stem(tok) in original_stems:
            continue
        if tok in _QUESTION_OPERATORS or tok in ANSWER_TYPES or tok in _YES_NO_LEADS:
            continue
        if tok in _DROPPABLE_EVENT or tok in _DROPPABLE_MODIFIERS:
            continue
        novel.append(tok)
    return bool(novel)


def _default_rewrite_embedder():
    global _REWRITE_EMBEDDER
    if _REWRITE_EMBEDDER is None:
        from continuator_memory.config import DEFAULT_EMBEDDING_MODEL
        from continuator_memory.embedding import SentenceTransformerEmbedder

        _REWRITE_EMBEDDER = SentenceTransformerEmbedder(DEFAULT_EMBEDDING_MODEL)
    return _REWRITE_EMBEDDER


def _embedder_supports_rewrite_similarity(embedder: Any | None) -> bool:
    """Only sentence-transformer MiniLM (or equivalent) is calibrated for the 0.65 veto."""
    if embedder is None:
        return False
    from continuator_memory.embedding import SentenceTransformerEmbedder

    if isinstance(embedder, SentenceTransformerEmbedder):
        return True
    name = str(getattr(embedder, "model_name", "") or "").lower()
    return "minilm" in name or "sentence-transformers" in name


def rewrite_cosine_similarity(
    question: str,
    query: str,
    embedder: Any | None = None,
    *,
    precomputed: dict[str, np.ndarray] | None = None,
) -> float:
    """MiniLM cosine between original and candidate (normalized embeddings)."""
    if precomputed is not None:
        a = precomputed[question]
        b = precomputed[query]
        return float(np.dot(a, b))
    model = embedder if _embedder_supports_rewrite_similarity(embedder) else _default_rewrite_embedder()
    vectors = model.embed([question, query])
    return float(np.dot(vectors[0], vectors[1]))


def evaluate_rewrite(
    query: str,
    question: str,
    intent: QuestionIntent | None = None,
    *,
    embedder: Any | None = None,
    similarity: float | None = None,
    precomputed: dict[str, np.ndarray] | None = None,
) -> RewriteDecision:
    """Layered rewrite validator. Similarity is a veto only (< MIN_REWRITE_SIMILARITY)."""
    text = " ".join(query.split()).strip()
    original = question.strip()
    intent = intent or analyze_intent(original)

    if not text or len(text) < MIN_QUERY_CHARS:
        return RewriteDecision(False, "empty")
    if len(text) > MAX_QUERY_CHARS:
        return RewriteDecision(False, "too_long")
    if text.casefold() == original.casefold():
        return RewriteDecision(True, "accepted", similarity=1.0)
    if _is_generic_rewrite(text):
        return RewriteDecision(False, "generic_rewrite")

    original_key = _question_intent_key(original)
    query_key = _question_intent_key(text)
    if not _answer_types_compatible(original_key, query_key, intent, text):
        return RewriteDecision(False, "answer_type_not_preserved")

    entity_reason = _entities_preserved(original, text, intent)
    if entity_reason:
        return RewriteDecision(False, entity_reason)

    if not _preserves_event(text, intent):
        return RewriteDecision(False, "event_not_preserved")

    constraint_reason = _constraints_preserved(original, text)
    if constraint_reason:
        return RewriteDecision(False, constraint_reason)

    if _insertion_elaboration(original, text):
        return RewriteDecision(False, "added_constraint")

    if similarity is None and (
        precomputed is not None or _embedder_supports_rewrite_similarity(embedder)
    ):
        similarity = rewrite_cosine_similarity(
            original, text, embedder=embedder, precomputed=precomputed,
        )
    if similarity is not None and similarity < MIN_REWRITE_SIMILARITY:
        return RewriteDecision(False, "low_similarity", similarity=similarity)

    return RewriteDecision(True, "accepted", similarity=similarity)


def is_safe_rewrite(
    query: str,
    question: str,
    intent: QuestionIntent | None = None,
    *,
    embedder: Any | None = None,
    similarity: float | None = None,
) -> bool:
    return evaluate_rewrite(
        query, question, intent, embedder=embedder, similarity=similarity,
    ).accepted


def fallback_queries(
    question: str,
    entities: tuple[str, ...] | None = None,
    *,
    embedder: Any | None = None,
) -> tuple[str, ...]:
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
    return _finalize_queries(
        question, _rewrite_candidates(question, intent), intent=intent, embedder=embedder,
    )


def fallback_plan(
    question: str,
    *,
    status: str = STATUS_FALLBACK,
    error: str | None = None,
    embedder: Any | None = None,
) -> QueryPlan:
    text = question.strip()
    entities = entities_from_question(text)
    return QueryPlan(
        information_need=f"Find evidence that answers: {text}",
        entities=entities,
        facets=facets_from_question(text),
        queries=fallback_queries(text, entities, embedder=embedder),
        source=SOURCE_FALLBACK,
        status=status,
        error=error,
    )


def _finalize_queries(
    original: str,
    candidates: list[str],
    *,
    intent: QuestionIntent | None = None,
    embedder: Any | None = None,
) -> tuple[str, ...]:
    intent = intent or analyze_intent(original)
    cleaned_candidates: list[str] = []
    for raw in candidates:
        query = " ".join(raw.split()).strip()
        if query:
            cleaned_candidates.append(query)

    precomputed: dict[str, np.ndarray] | None = None
    need_sim = [
        q for q in cleaned_candidates
        if q.casefold() != original.strip().casefold()
    ]
    # Apply the calibrated MiniLM veto only with a sentence-transformer embedder.
    # Hashing / test embedders skip similarity; other layers still run.
    if need_sim and _embedder_supports_rewrite_similarity(embedder):
        texts = [original.strip(), *need_sim]
        vectors = embedder.embed(texts)
        precomputed = {text: vectors[i] for i, text in enumerate(texts)}

    seen: set[str] = set()
    out: list[str] = []
    for raw in [original, *cleaned_candidates]:
        query = " ".join(raw.split()).strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            continue
        key = query.casefold()
        if key in seen:
            continue
        if key != original.strip().casefold():
            decision = evaluate_rewrite(
                query, original, intent, embedder=embedder, precomputed=precomputed,
            )
            if not decision.accepted:
                continue
        seen.add(key)
        out.append(query)
        if len(out) >= MAX_QUERIES:
            break
    if not out or out[0].casefold() != original.strip().casefold():
        out = [original.strip(), *[q for q in out if q.casefold() != original.strip().casefold()]]
    return tuple(out[:MAX_QUERIES])


def validate_plan(
    question: str,
    raw: Any,
    *,
    source: str = SOURCE_INJECTED,
    embedder: Any | None = None,
) -> QueryPlan:
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
    queries = _finalize_queries(
        question,
        [q for q in queries_raw if isinstance(q, str)],
        embedder=embedder,
    )
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


def plan_from_generator(
    question: str,
    generator: Generator,
    *,
    embedder: Any | None = None,
) -> QueryPlan:
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
        return validate_plan(question, parsed, source=SOURCE_OLLAMA, embedder=embedder)
    except (GenerationError, ValueError, json.JSONDecodeError, TypeError) as exc:
        return fallback_plan(
            question, status=STATUS_PLANNER_FAILED, error=str(exc), embedder=embedder,
        )


def load_planner_model() -> str:
    env = os.environ
    return (
        env.get("CONTINUATOR_QUERY_PLAN_MODEL")
        or env.get("CONTINUATOR_ANSWER_MODEL")
        or env.get("CONTINUATOR_OLLAMA_MODEL")
        or ""
    ).strip()


def resolve_plan(
    question: str,
    planner: PlannerFn | Generator | None,
    *,
    embedder: Any | None = None,
) -> QueryPlan:
    if planner is None:
        return fallback_plan(question, embedder=embedder)
    if callable(planner) and not hasattr(planner, "generate"):
        try:
            plan = planner(question)
        except Exception as exc:
            return fallback_plan(
                question, status=STATUS_PLANNER_FAILED, error=str(exc), embedder=embedder,
            )
        if isinstance(plan, QueryPlan):
            try:
                cleaned = validate_plan(
                    question, plan.to_dict(), source=plan.source, embedder=embedder,
                )
            except ValueError as exc:
                return fallback_plan(
                    question, status=STATUS_INVALID, error=str(exc), embedder=embedder,
                )
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
                return validate_plan(
                    question, plan, source=SOURCE_INJECTED, embedder=embedder,
                )
            except ValueError as exc:
                return fallback_plan(
                    question, status=STATUS_INVALID, error=str(exc), embedder=embedder,
                )
        return fallback_plan(
            question,
            status=STATUS_INVALID,
            error="planner returned an unsupported value",
            embedder=embedder,
        )
    return plan_from_generator(question, planner, embedder=embedder)
