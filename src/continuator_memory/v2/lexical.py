"""Lexical overlap for the experimental hybrid retrieval mode.

Baseline cosine ranking is unchanged. Hybrid adds
``score = similarity + adjustment + lexical_weight * overlap``.
Overlap is the fraction of question content stems that appear in the memory.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NAME = re.compile(r"\b([A-Z][a-z]{2,})\b")
_SUFFIXES = ("ing", "ed", "es", "s")
DEFAULT_LEXICAL_WEIGHT = 0.25
LEXICAL_WEIGHT_ENV = "CONTINUATOR_LEXICAL_WEIGHT"
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with", "by",
    "from", "at", "as", "into", "about", "her", "his", "she", "he", "they",
    "their", "them", "it", "its", "be", "is", "are", "was", "were", "been",
    "being", "do", "does", "did", "would", "could", "should", "might", "may",
    "can", "will", "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "this", "that", "these", "those", "any", "some",
})


def normalize(text: str) -> str:
    return text.lower().replace("\u2019", "'")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def stem(word: str) -> str:
    token = normalize(word)
    if len(token) <= 4:
        return token
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def question_entities(question: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_NAME.findall(question)))


def content_stems(text: str, extra_ignore: Iterable[str] = ()) -> set[str]:
    ignored = set(STOP_WORDS)
    for raw in extra_ignore:
        for tok in tokenize(raw):
            ignored.add(tok)
            ignored.add(stem(tok))
    stems: set[str] = set()
    for tok in tokenize(text):
        if tok in ignored or len(tok) < 3:
            continue
        item = stem(tok)
        if item in ignored or len(item) < 3:
            continue
        stems.add(item)
    return stems


def question_stems(question: str) -> set[str]:
    return content_stems(question, question_entities(question))


def lexical_overlap(question: str, memory: str) -> float:
    """Fraction of question content stems found in ``memory``. 0 if none remain."""
    asked = question_stems(question)
    if not asked:
        return 0.0
    return len(asked & content_stems(memory)) / len(asked)


def fuse_scores(
    similarity: float,
    lexical: float,
    *,
    weight: float,
    adjustment: float = 0.0,
) -> float:
    """``similarity + adjustment + weight * lexical``. Weight 0 is Baseline."""
    if weight < 0:
        raise ValueError("lexical weight must be >= 0")
    return similarity + adjustment + weight * lexical


def load_lexical_weight() -> float:
    raw = os.environ.get(LEXICAL_WEIGHT_ENV)
    if raw is None or raw == "":
        return DEFAULT_LEXICAL_WEIGHT
    return float(raw)
