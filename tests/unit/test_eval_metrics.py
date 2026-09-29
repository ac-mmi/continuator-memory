from dataclasses import dataclass

import pytest

from continuator_memory.evaluation.metrics import (
    contains_phrase,
    first_expected_rank,
    hit_at_k,
    is_abstention,
    match_answer,
    mean,
    mentions_any,
    normalize_text,
    recall_at_k,
    reciprocal_rank,
)


@dataclass
class C:
    conversation_id: str
    source_message_ids: tuple


RANKED = [
    C("conv", ("m1", "m2")),
    C("other", ("m5",)),  # same message ID, wrong conversation: never relevant
    C("conv", ("m4", "m5")),
    C("conv", ("m7",)),
]


def test_first_expected_rank():
    assert first_expected_rank(RANKED, "conv", ["m2"]) == 1
    assert first_expected_rank(RANKED, "conv", ["m5"]) == 3
    assert first_expected_rank(RANKED, "conv", ["m7", "m5"]) == 3
    assert first_expected_rank(RANKED, "conv", ["m9"]) is None
    assert first_expected_rank([], "conv", ["m1"]) is None


def test_first_expected_rank_ignores_other_conversations():
    assert first_expected_rank(RANKED, "other", ["m1"]) is None
    assert first_expected_rank(RANKED, "other", ["m5"]) == 2


def test_reciprocal_rank():
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(None) == 0.0


def test_recall_at_k_single_source():
    assert recall_at_k(RANKED, "conv", ["m5"], 1) == 0.0
    assert recall_at_k(RANKED, "conv", ["m5"], 2) == 0.0
    assert recall_at_k(RANKED, "conv", ["m5"], 3) == 1.0
    assert recall_at_k(RANKED, "conv", ["m5"], 10) == 1.0


def test_recall_at_k_multi_source_is_fractional():
    expected = ["m1", "m7"]
    assert recall_at_k(RANKED, "conv", expected, 1) == 0.5
    assert recall_at_k(RANKED, "conv", expected, 3) == 0.5
    assert recall_at_k(RANKED, "conv", expected, 4) == 1.0


def test_recall_at_k_duplicate_expected_ids_and_errors():
    assert recall_at_k(RANKED, "conv", ["m1", "m1"], 1) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(RANKED, "conv", [], 1)
    with pytest.raises(ValueError):
        recall_at_k(RANKED, "conv", ["m1"], 0)


def test_hit_at_k():
    assert hit_at_k(3, 3) is True
    assert hit_at_k(3, 1) is False
    assert hit_at_k(None, 10) is False


def test_mrr_example():
    ranks = [1, 2, None, 4]
    assert mean([reciprocal_rank(r) for r in ranks]) == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert mean([]) is None


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Hello,   World!", "hello world"),
        ("Remédios Café", "remedios cafe"),
        ("harbor-72", "harbor 72"),
        ("55,000 €", "55 000"),
        ("I don’t know", "i don t know"),
        ("  ", ""),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_contains_phrase_uses_word_boundaries():
    assert contains_phrase("The launch is on July 1st.", "July 1st")
    assert not contains_phrase("The launch is on July 1st.", "July 1")
    assert not contains_phrase("It was 16 euros", "6")
    assert contains_phrase("Pixel and Byte", "byte")
    assert contains_phrase("The password is HARBOR-72!", "harbor-72")
    assert not contains_phrase("anything", "")


def test_match_answer_any_and_all():
    assert match_answer("It's a beagle.", ["beagle"], [])
    assert not match_answer("It's a collie.", ["beagle"], [])
    assert match_answer("Cats: Pixel and Byte", [], ["Pixel", "Byte"])
    assert not match_answer("Cats: Pixel", [], ["Pixel", "Byte"])
    assert match_answer("Omar, backend dev", ["Omar"], ["backend"])
    assert not match_answer("Omar", ["Omar"], ["backend"])
    with pytest.raises(ValueError):
        match_answer("x", [], [])


@pytest.mark.parametrize(
    "answer",
    [
        "I don't know.",
        "I do not know based on the excerpts.",
        "The excerpts don’t mention that; I don't know.",
        "This is not mentioned in the conversation.",
        "There is no information about Omar's surname.",
        "The favorite movie is not specified.",
    ],
)
def test_is_abstention_positive(answer):
    assert is_abstention(answer)


@pytest.mark.parametrize("answer", ["Biscuit.", "The launch is on July 1st [atlas:c0008]."])
def test_is_abstention_negative(answer):
    assert not is_abstention(answer)


def test_mentions_any():
    assert mentions_any("Moved from June 3rd to July 1st", ["June 3rd", "June 17th"])
    assert not mentions_any("July 1st", ["June 3rd"])
