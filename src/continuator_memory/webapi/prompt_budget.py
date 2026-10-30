"""Token budget for the answer prompt.

The answer model runs with an explicit context window (``num_ctx``). The prompt must leave room
for the system instructions, the question, the chat template and the generated output
(``num_predict``); Ollama otherwise truncates the input silently.

Ollama exposes no tokenize endpoint, so token counts are *estimated* by
``estimate_tokens``, a deliberately conservative approximation of Qwen2's byte-level BPE
pre-tokenizer: digits count one each, ASCII words of up to 6 letters count one and longer words
one per 6 letters, punctuation one per 2 bytes, non-ASCII letters one per 2 bytes, whitespace
runs one. On the 36 LongMemEval pilot prompts with Ollama-reported counts it never
under-counted (smallest margin +10.6% on a table-heavy prompt, median +19%). Characters are
never treated as tokens.

``allocate`` keeps the ranking: every excerpt that fits gets at least ``MIN_EXCERPT_TOKENS`` of
content, long excerpts are trimmed to a shared cap, and only when even that minimum does not fit
are the lowest-ranked excerpts left out.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from continuator_memory.v2.lexical import content_stems, question_stems

TEMPLATE_OVERHEAD_TOKENS = 16  # chat template markers around system and user turns
MIN_EXCERPT_TOKENS = 48  # smallest useful share of an excerpt's content
_LONG_WORD = 6
_PUNCT_BYTES = 2
_PIECE = re.compile(r"'(?:[sdmt]|ll|ve|re)\b|[^\r\n\w]?[^\W\d_]+|\d|[ ]?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+")


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for Qwen2-style byte-level BPE."""
    n = 0
    for match in _PIECE.finditer(text or ""):
        piece = match.group(0)
        core = piece.strip()
        if not core:
            n += 1
        elif core.isdigit():
            n += 1
        elif core[-1].isalpha():
            letters = core.lstrip("".join(ch for ch in core if not ch.isalpha()))
            if letters.isascii():
                n += 1 if len(letters) <= _LONG_WORD else math.ceil(len(letters) / _LONG_WORD)
            else:
                n += math.ceil(len(letters.encode("utf-8")) / 2)
            if len(core) > len(letters):
                n += 1
        else:
            n += max(1, math.ceil(len(core.encode("utf-8")) / _PUNCT_BYTES))
    return n


# Sentence or line boundaries: after sentence-ending punctuation (optionally closed by a quote or
# bracket) followed by whitespace, or at a line break. "$5.50" and "e.g." without a space stay whole.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"')\]])\s+|\n+")
_GAP = " … "
_NEIGHBOUR_WEIGHT = 0.5


def _head(text: str, max_tokens: int) -> str:
    """Longest prefix of ``text`` whose estimate is at most ``max_tokens``, cut at a word boundary."""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    space = cut.rfind(" ")
    if space > len(cut) - 40 and space > 0:
        cut = cut[:space]
    return cut.rstrip()


def _segments(text: str) -> list[tuple[str, str]]:
    """(segment, separator after it) pairs that rebuild ``text`` exactly."""
    out, start = [], 0
    for match in _BOUNDARY.finditer(text):
        out.append((text[start:match.start()], match.group(0)))
        start = match.end()
    out.append((text[start:], ""))
    return [(seg, sep) for seg, sep in out if seg.strip()]


def _render(segments: list[tuple[str, str]], chosen: set[int]) -> str:
    parts: list[str] = []
    previous = None
    for i in sorted(chosen):
        if previous is None:
            if i > 0:
                parts.append(_GAP.lstrip())
        elif i != previous + 1:
            parts.append(_GAP)
        elif previous is not None:
            parts.append(segments[previous][1])
        parts.append(segments[i][0].strip() if previous is None or i != previous + 1 else segments[i][0])
        previous = i
    return "".join(parts).rstrip()


def trim_to_tokens(text: str, max_tokens: int, query: str | None = None) -> str:
    """Shorten ``text`` to at most ``max_tokens`` (estimated).

    Without a query, or when no sentence shares a content stem with it, the longest prefix is kept.
    With a query, whole sentences are kept in their original order: sentences sharing question
    stems first, then their neighbours (``_NEIGHBOUR_WEIGHT``), then the rest from the start. Every
    omitted stretch is marked with an ellipsis, so trimmed text is never silently joined.
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    segments = _segments(text)
    terms = question_stems(query) if query else set()
    scores = [len(terms & content_stems(seg)) for seg, _ in segments] if terms else []
    if not scores or not any(scores) or len(segments) < 2:
        return _head(text, max_tokens)
    priority = [
        score + _NEIGHBOUR_WEIGHT * max([scores[j] for j in (i - 1, i + 1) if 0 <= j < len(scores)] or [0])
        for i, score in enumerate(scores)
    ]
    order = sorted((i for i in range(len(segments)) if priority[i] > 0), key=lambda i: (-priority[i], i))
    order += [i for i in range(len(segments)) if priority[i] == 0]
    chosen: set[int] = set()
    for i in order:
        if estimate_tokens(_render(segments, chosen | {i})) <= max_tokens:
            chosen.add(i)
    if not chosen:
        best = order[0]
        prefix = _GAP.lstrip() if best > 0 else ""
        return prefix + _head(segments[best][0].strip(), max_tokens - estimate_tokens(prefix))
    return _render(segments, chosen)


@dataclass(frozen=True)
class PromptBudget:
    context_tokens: int  # the model's num_ctx
    output_tokens: int  # num_predict, reserved for the answer
    template_tokens: int = TEMPLATE_OVERHEAD_TOKENS

    def __post_init__(self) -> None:
        if self.context_tokens < 1 or self.output_tokens < 0 or self.template_tokens < 0:
            raise ValueError("prompt budget values must be positive")

    def evidence_tokens(self, fixed_text: str) -> int:
        """Tokens left for excerpts once the fixed text (system prompt, frame, question) is counted."""
        return self.context_tokens - self.output_tokens - self.template_tokens - estimate_tokens(fixed_text)


@dataclass(frozen=True)
class Allocation:
    keep: int  # the first ``keep`` excerpts are included, in rank order
    content_cap: int | None  # per-excerpt content cap in tokens; None means nothing is trimmed


def allocate(fixed: list[int], content: list[int], available: int,
             min_content: int = MIN_EXCERPT_TOKENS) -> Allocation:
    """Choose how many top-ranked excerpts to keep and a shared content cap that fits ``available``."""
    if len(fixed) != len(content):
        raise ValueError("fixed and content must have the same length")
    if sum(fixed) + sum(content) <= available:
        return Allocation(len(fixed), None)
    keep = len(fixed)
    while keep and sum(f + min(c, min_content) for f, c in zip(fixed[:keep], content[:keep])) > available:
        keep -= 1
    if not keep:
        return Allocation(0, 0)
    lo, hi = min(min_content, max(content[:keep])), max(content[:keep])
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(f + min(c, mid) for f, c in zip(fixed[:keep], content[:keep])) <= available:
            lo = mid
        else:
            hi = mid - 1
    return Allocation(keep, lo)
