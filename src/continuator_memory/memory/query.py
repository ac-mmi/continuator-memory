"""Lightweight, replaceable question analysis.

Maps a question onto the vocabulary of the memory store (entity names and
predicates) by token overlap. The only language-specific parts are small
closed-class lexicons: generic past-reference words used to detect historical
intent (current state is the default), and the question/relative words,
auxiliaries, and prepositions used to split a question into a requested
attribute and an identifying constraint:

  "Which city is the person who handles the offsite budget from?"
   wh-word ─┘ └─ requested attribute    └─ constraint clause ─┘ └─ frame
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from continuator_memory.memory.models import Entity, normalize_value
from continuator_memory.memory.text import STOPWORDS, content_tokens, predicate_tokens

HISTORICAL_MARKERS = frozenset(
    {
        "original", "originally", "initial", "initially", "previous", "previously",
        "before", "earlier", "former", "formerly", "old", "prior",
    }
)
QUESTION_WORDS = frozenset({"what", "which", "who", "whose", "where", "when", "how"})
RELATIVE_WORDS = frozenset({"who", "whose", "that", "which"})
AUXILIARIES = frozenset(
    {"is", "are", "was", "were", "be", "been", "will", "would", "do", "does", "did",
     "can", "could", "cannot", "should", "has", "have", "had"}
)
PREPOSITIONS = frozenset(
    {"of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "onto", "about"}
)


@dataclass(frozen=True)
class QuestionConstraint:
    """A relative clause identifying an entity that is not named ("the person who ...")."""

    marker: str  # the relative word that opened the clause
    head: str  # first content token of the clause: the condition's relation
    terms: frozenset[str]  # all content tokens of the clause
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"marker": self.marker, "head": self.head, "terms": sorted(self.terms),
                "text": self.text}


@dataclass(frozen=True)
class RequestedAttribute:
    text: str
    tokens: frozenset[str]
    frame: str | None  # stranded final preposition ("... from?"), if any

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "tokens": sorted(self.tokens), "frame": self.frame}


@dataclass(frozen=True)
class QuestionAnalysis:
    tokens: frozenset[str]
    entities: tuple[Entity, ...]
    predicates: tuple[str, ...]
    relation_targets: tuple[str, ...]
    temporal_scope: str  # "current" (default) or "historical"
    question: str = ""
    targets: tuple[Entity, ...] = ()
    constraints: tuple[QuestionConstraint, ...] = ()
    requested_attribute: RequestedAttribute | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "targets": [e.name for e in self.targets],
            "constraints": [c.to_dict() for c in self.constraints],
            "requested_attribute": (
                self.requested_attribute.to_dict() if self.requested_attribute else None
            ),
            "temporal_scope": self.temporal_scope,
        }


class QuestionAnalyzer(Protocol):
    def analyze(
        self,
        question: str,
        entities: list[Entity],
        predicates: set[str],
        relation_predicates: set[str],
    ) -> QuestionAnalysis: ...


def _content_run(words: list[str], start: int, skip: set[str]) -> list[str]:
    """Words from ``start``, skipping structural words and ``skip``, up to the next one."""
    i = start
    while i < len(words) and (words[i] in STOPWORDS or words[i] in AUXILIARIES
                              or words[i] in skip):
        i += 1
    run = []
    while i < len(words) and words[i] not in STOPWORDS and words[i] not in AUXILIARIES \
            and words[i] not in skip:
        run.append(words[i])
        i += 1
    return run


def parse_structure(
    question: str, target_words: set[str]
) -> tuple[tuple[QuestionConstraint, ...], RequestedAttribute | None]:
    """Split a question into an identifying constraint and a requested attribute."""
    words = normalize_value(question).split()
    wh = next((i for i, w in enumerate(words) if w in QUESTION_WORDS), None)
    if wh is None:
        return (), None
    frame = words[-1] if len(words) > wh + 1 and words[-1] in PREPOSITIONS else None
    body = words[:-1] if frame else words

    constraints: tuple[QuestionConstraint, ...] = ()
    end = len(body)
    rel = next((i for i in range(wh + 2, len(body)) if body[i] in RELATIVE_WORDS), None)
    if rel is not None:
        clause = body[rel + 1:]
        head = _content_run(clause, 0, set())
        if head:
            constraints = (QuestionConstraint(
                marker=body[rel],
                head=next(iter(content_tokens(head[0])), head[0]),
                terms=frozenset(content_tokens(" ".join(clause))),
                text=" ".join(clause),
            ),)
            end = rel

    before = body[:end]
    if wh + 1 < len(before) and before[wh + 1] not in STOPWORDS | AUXILIARIES:
        run = _content_run(before, wh + 1, set())
    else:
        run = _content_run(before, wh + 1, target_words | {"s"})
    tokens = frozenset(content_tokens(" ".join(run)))
    if not tokens and frame is None:
        return constraints, None
    return constraints, RequestedAttribute(" ".join(run), tokens, frame)


class LexicalQuestionAnalyzer:
    def analyze(
        self,
        question: str,
        entities: list[Entity],
        predicates: set[str],
        relation_predicates: set[str],
    ) -> QuestionAnalysis:
        tokens = frozenset(content_tokens(question))
        matched = []
        targets = []
        for e in entities:
            name_tokens = content_tokens(e.name)
            if e.normalized_name == "user" or not name_tokens:
                continue
            overlap = name_tokens & tokens
            if overlap and len(overlap) * 2 >= len(name_tokens):
                matched.append(e)
            if name_tokens <= tokens:
                targets.append(e)
        raw_words = set(normalize_value(question).split())
        target_words = {w for e in targets for w in normalize_value(e.name).split()}
        constraints, requested = parse_structure(question, target_words)
        return QuestionAnalysis(
            tokens=tokens,
            entities=tuple(matched),
            predicates=tuple(sorted(p for p in predicates if predicate_tokens(p) & tokens)),
            relation_targets=tuple(
                sorted(p for p in relation_predicates if predicate_tokens(p) & tokens)
            ),
            temporal_scope="historical" if raw_words & HISTORICAL_MARKERS else "current",
            question=question,
            targets=tuple(targets),
            constraints=constraints,
            requested_attribute=requested,
        )
