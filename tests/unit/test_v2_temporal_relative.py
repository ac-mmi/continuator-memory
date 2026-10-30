"""Relative time expressions resolved against a message's own date (shared by answering and the judge)."""

from __future__ import annotations

from datetime import date

import pytest

from continuator_memory.v2.temporal import (
    RELATION_CURRENT,
    RELATION_FUTURE,
    RELATION_PAST,
    DateResolution,
    date_mentions,
    message_date_resolutions,
    message_day,
    resolve_relative_expressions,
)

TUESDAY = date(2024, 3, 12)


@pytest.mark.parametrize("raw, expected", [
    ("2024-03-12T13:56:00", TUESDAY),
    ("2024-03-12T13:56:00Z", TUESDAY),
    ("2024-03-12T23:30:00+05:00", TUESDAY),
    ("2024-03-12", TUESDAY),
    (None, None),
    ("", None),
    ("   ", None),
    ("1:56 pm on 12 March, 2024", None),  # unparsed source clock is kept as text upstream, never guessed
    ("not a date", None),
])
def test_message_day_accepts_iso_only(raw, expected):
    assert message_day(raw) == expected


def test_relative_dates_resolve_against_the_message_date():
    assert message_date_resolutions("I went to the workshop yesterday.", TUESDAY) == (
        DateResolution("yesterday", "11 March 2024", RELATION_PAST),)
    assert message_date_resolutions("We met last Saturday.", TUESDAY) == (
        DateResolution("last Saturday", "Saturday 9 March 2024", RELATION_PAST),)
    assert message_date_resolutions("I painted it last year.", TUESDAY) == (
        DateResolution("last year", "2023", RELATION_PAST),)


def test_the_same_words_resolve_differently_for_different_message_dates():
    text = "I signed up yesterday."
    assert message_date_resolutions(text, date(2024, 3, 12))[0].value == "11 March 2024"
    assert message_date_resolutions(text, date(2024, 8, 1))[0].value == "31 July 2024"


def test_no_date_means_no_resolution():
    assert message_date_resolutions("I went yesterday.", None) == ()
    assert message_date_resolutions("", TUESDAY) == ()
    assert message_date_resolutions("Nothing relative here.", TUESDAY) == ()


@pytest.mark.parametrize("text", [
    "Time flies, it feels like just yesterday I wore this dress!",
    "It seems like yesterday that we met.",
    "It was as if it were yesterday.",
])
def test_figurative_uses_are_not_resolved(text):
    assert message_date_resolutions(text, TUESDAY) == ()


def test_figurative_and_literal_uses_in_one_message():
    text = "Yesterday was great. It feels like just yesterday we started."
    assert [r.value for r in message_date_resolutions(text, TUESDAY)] == ["11 March 2024"]
    assert message_date_resolutions("Like I said, yesterday I went.", TUESDAY)[0].value == "11 March 2024"


def test_approximate_qualifiers_stay_visible():
    assert message_date_resolutions("I started playing about five years ago.", TUESDAY) == (
        DateResolution("about five years ago", "around 2019", RELATION_PAST),)
    assert message_date_resolutions("We moved around two weeks ago.", TUESDAY) == (
        DateResolution("around two weeks ago", "around 27 February 2024", RELATION_PAST),)


@pytest.mark.parametrize("text", ["I quit over five years ago.", "It was nearly two weeks ago.",
                                  "She left more than 3 days ago."])
def test_bounded_qualifiers_are_not_turned_into_dates(text):
    assert message_date_resolutions(text, TUESDAY) == ()


def test_future_plans_are_labelled_as_future_not_as_events():
    assert message_date_resolutions("We're going camping next month.", TUESDAY) == (
        DateResolution("next month", "April 2024", RELATION_FUTURE),)
    assert message_date_resolutions("See you tomorrow!", TUESDAY)[0].relation == RELATION_FUTURE
    assert message_date_resolutions("I'm going to a conference this month.", TUESDAY) == (
        DateResolution("this month", "March 2024", RELATION_CURRENT),)
    assert message_date_resolutions("That trip this past weekend was wild.", TUESDAY)[0].relation == RELATION_PAST


def test_same_day_expressions_are_left_to_the_message_date():
    assert message_date_resolutions("Kids today have it hard.", TUESDAY) == ()
    assert [r.phrase for r in message_date_resolutions("Today I rested; I ran last Saturday.", TUESDAY)] == [
        "last Saturday"]


def test_resolver_itself_is_unchanged_by_the_message_layer():
    """The layer filters; the underlying resolver still reports every expression it matches."""
    assert resolve_relative_expressions("it feels like just yesterday", TUESDAY) == [("yesterday", "11 March 2024")]


@pytest.mark.parametrize("text, expected", [
    ("on 7 May 2024", [(7, 5, 2024)]),
    ("on May 7, 2024", [(7, 5, 2024)]),
    ("on 7th of May", [(7, 5, None)]),
    ("in June 2024", [(None, 6, 2024)]),
    ("back in 2022", [(None, None, 2022)]),
    ("on Saturday 9 March 2024", [(9, 3, 2024)]),
    ("the weekend of 9 March 2024 to 10 March 2024", [(9, 3, 2024), (10, 3, 2024)]),
    ("I may go, maybe", []),
    ("for 4 years", []),
])
def test_date_mentions(text, expected):
    assert date_mentions(text) == expected
