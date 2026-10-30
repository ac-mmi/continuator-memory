"""Evaluation-only answer scoring for benchmark harnesses.

Nothing here is ever passed to retrieval or generation. Abstention questions (LoCoMo
category 5, LongMemEval ``_abs``) are scored by whether the answer abstains, instead of being
skipped. Token F1 and exact match are auxiliary string metrics: they are reported but are not
semantic correctness, which stays ``None`` until a judge or a human label fills it in.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from continuator_memory.evaluation.metrics import contains_phrase, is_abstention, normalize_text

SEMANTIC_NOT_JUDGED = "not_judged"
# Abstention wordings missed by metrics.ABSTENTION_MARKERS; seen in the LongMemEval pilot and in the
# benchmark's own abstention references ("The information provided is not enough").
EXTRA_ABSTENTION_MARKERS = (
    "not enough information",
    "information provided is not enough",
    "insufficient information",
)


def abstained(answer: str) -> bool:
    return is_abstention(answer) or any(contains_phrase(answer, m) for m in EXTRA_ABSTENTION_MARKERS)


def token_f1(prediction: str, reference: str) -> float:
    """Multiset token F1 over normalized text; both empty is 1.0, exactly one empty is 0.0."""
    pred = normalize_text(prediction).split()
    gold = normalize_text(reference).split()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_answer(
    answer: str,
    reference: str | None,
    *,
    expects_abstention: bool,
    adversarial_answer: str | None = None,
) -> dict[str, Any]:
    """Score one final answer against evaluation-only labels."""
    text = answer or ""
    did_abstain = abstained(text)
    trap = (adversarial_answer or "").strip()
    ref = (reference or "").strip()
    return {
        "abstained": did_abstain,
        "expectsAbstention": expects_abstention,
        "abstentionCorrect": did_abstain if expects_abstention else None,
        "adversarialAnswerMatched": contains_phrase(text, trap) if trap else None,
        "exactMatch": normalize_text(text) == normalize_text(ref) if ref else None,
        "tokenF1": token_f1(text, ref) if ref else None,
        "semanticCorrect": None,
        "semanticJudge": SEMANTIC_NOT_JUDGED,
    }
