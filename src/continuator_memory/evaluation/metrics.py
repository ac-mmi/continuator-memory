"""Deterministic retrieval and answer metrics."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import Protocol

ABSTENTION_MARKERS = (
    "don't know",
    "do not know",
    "not mentioned",
    "isn't mentioned",
    "no mention",
    "not mention",
    "doesn't mention",
    "does not mention",
    "no information",
    "not provided",
    "not specified",
    "not stated",
    "not available",
    "not included",
    "not contain",
    "doesn't contain",
    "does not contain",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "unable to determine",
    "cannot answer",
    "can't answer",
    "unknown",
    "not clear from",
)


class RetrievedLike(Protocol):
    conversation_id: str
    source_message_ids: Sequence[str]


# ---- retrieval ------------------------------------------------------------


def _relevant_ids(chunk: RetrievedLike, conversation_id: str, expected: set[str]) -> set[str]:
    if chunk.conversation_id != conversation_id:
        return set()
    return expected.intersection(chunk.source_message_ids)


def first_expected_rank(
    retrieved: Sequence[RetrievedLike], conversation_id: str, expected: Iterable[str]
) -> int | None:
    """1-based rank of the first chunk containing any expected source message, else None."""
    expected_set = set(expected)
    for rank, chunk in enumerate(retrieved, start=1):
        if _relevant_ids(chunk, conversation_id, expected_set):
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def recall_at_k(
    retrieved: Sequence[RetrievedLike], conversation_id: str, expected: Iterable[str], k: int
) -> float:
    """Fraction of expected source messages covered by the top-k chunks."""
    if k < 1:
        raise ValueError("k must be >= 1")
    expected_set = set(expected)
    if not expected_set:
        raise ValueError("recall is undefined without expected sources")
    covered: set[str] = set()
    for chunk in retrieved[:k]:
        covered |= _relevant_ids(chunk, conversation_id, expected_set)
    return len(covered) / len(expected_set)


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


# ---- answers --------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Casefold, strip accents, map punctuation to spaces, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", stripped).split())


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word match of normalized ``phrase`` inside normalized ``text``."""
    needle = normalize_text(phrase)
    if not needle:
        return False
    return f" {needle} " in f" {normalize_text(text)} "


def match_answer(answer: str, match_any: Sequence[str], match_all: Sequence[str]) -> bool:
    """True if at least one ``match_any`` phrase (when given) and every ``match_all`` phrase occur."""
    if not match_any and not match_all:
        raise ValueError("need at least one phrase to match")
    any_ok = not match_any or any(contains_phrase(answer, p) for p in match_any)
    all_ok = all(contains_phrase(answer, p) for p in match_all)
    return any_ok and all_ok


def is_abstention(answer: str) -> bool:
    return any(contains_phrase(answer, marker) for marker in ABSTENTION_MARKERS)


def mentions_any(answer: str, phrases: Sequence[str]) -> bool:
    return any(contains_phrase(answer, p) for p in phrases)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None
