"""Tokenization shared by extraction grounding and query analysis."""

from __future__ import annotations

from continuator_memory.memory.models import normalize_value

STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from into onto about as is are was were be
    been being am do does did doing have has had having it its this that these those there here
    what which who whom whose when where why how i me my mine we us our you your yours he him his
    she her hers they them their theirs s t not no so than then too very can could will would
    shall should may might must just also only any all some each every both either neither
    one ones up down out over under again further once
    """.split()
)


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> set[str]:
    """Normalized, lightly stemmed tokens with stopwords removed."""
    return {_stem(t) for t in normalize_value(text).split() if t not in STOPWORDS}


def predicate_tokens(predicate: str) -> set[str]:
    return content_tokens(predicate.replace("_", " "))
