"""Deterministic temporal semantics for V2 claims (phase 2B).

Two kinds of time are kept apart:
  observation time  when the message was sent (``Provenance.timestamp``); recorded on memories
                    and sources, never used as an event or validity date.
  event/valid time  when the claim says something happened or held; taken only from the
                    claim's own temporal expression or Phase 1's ``resolved_date``.

Dates keep the precision they were stated with: "2024", "2024-03" or "2024-03-03". Nothing is
padded to a day, and relative expressions ("last year") are not resolved here; that is Phase
1's ``resolved_date``. Two dates are ordered only when their intervals do not overlap.

Temporal classes, decided from the claim's wording (content, qualifiers, temporal expression):
  conditional     if / unless / might / may / could / would / maybe / perhaps / possibly
  future          will / 'll / going to / plans to / intends to / about to / scheduled to
  mixed           explicit past-state and explicit present wording in one claim
  past            explicit past-state wording (used to, formerly, previously, earlier, before
                  ...), a past anchor (last year, ago, yesterday) or a past-tense stative verb
                  (lived, worked, preferred, ...), or an explicit end date before observation
  transition      explicit change wording (moved, switched, took over, no longer, ...)
  present         explicit present wording (now, currently, current, still, has lived ...)
  ambiguous_past  past tense of "be" without any of the above ("was in Delhi"): narration or a
                  past state, so it is kept as stated and never used for state transitions
  unspecified     none of the above (Phase 2A default)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import NamedTuple

from continuator_memory.memory.models import normalize_value
from continuator_memory.v2.models import Claim

TIME_PRESENT = "present"
TIME_TRANSITION = "transition"
TIME_PAST = "past"
TIME_AMBIGUOUS_PAST = "ambiguous_past"
TIME_MIXED = "mixed"
TIME_FUTURE = "future"
TIME_CONDITIONAL = "conditional"
TIME_UNSPECIFIED = "unspecified"
TIME_CLASSES = (TIME_PRESENT, TIME_TRANSITION, TIME_PAST, TIME_AMBIGUOUS_PAST, TIME_MIXED, TIME_FUTURE,
                TIME_CONDITIONAL, TIME_UNSPECIFIED)
# Classes that may change which memory is current.
CHANGING_CLASSES = (TIME_PRESENT, TIME_TRANSITION)
# Classes whose memory may be replaced by a later change in the same slot.
REPLACEABLE_CLASSES = (TIME_PRESENT, TIME_TRANSITION, TIME_UNSPECIFIED)

# All cue patterns are matched against lowercased text (see ``cue_text``).
CONDITIONAL = re.compile(r"\b(if|unless|might|may|could|would|maybe|perhaps|possibly|in case)\b")
FUTURE = re.compile(r"\b(will|shall|going to|plans? to|planning to|intends? to|about to|scheduled to)\b|\w'll\b")
# Desires and deliberation: the current state is unchanged, but nothing is superseded either.
NOT_A_CHANGE = re.compile(r"\b(wants? to|hopes? to|considering|thinking about|thinks about)\b")
PAST_STATE = re.compile(
    r"\b(used to|formerly|previously|former|originally|in the past|back then|at the time|earlier)\b"
    r"|\bbefore (?:that|then)\b|^before\b|\bbefore\s*[.!]?\s*$")
PAST_ANCHOR = re.compile(
    r"\b(last (?:year|month|week|summer|winter|spring|fall|autumn|time)|yesterday|ago)\b")
PAST_STATIVE = re.compile(
    r"\b(lived|worked|stayed|preferred|liked|loved|owned|studied|belonged|rented|attended|managed|led|had)\b")
PAST_BE = re.compile(r"\b(was|were)\b")
TRANSITION = re.compile(
    r"\b(moved|switched|changed|instead|no longer|anymore|took over|taking over|takes over|replaced|"
    r"has become|became|from now on)\b")
PRESENT = re.compile(
    r"\b(now|currently|current|these days|nowadays|at the moment|at present|still)\b"
    r"|\b(?:has|have) (?:been|lived|worked|stayed|preferred|liked|owned)\b")

MONTHS = {name: i for i, names in enumerate((
    ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
    ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
    ("nov", "november"), ("dec", "december")), 1) for name in names}
_MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE_TOKEN = re.compile(
    rf"\b(?P<iso>\d{{4}}-\d{{2}}-\d{{2}})\b"
    rf"|\b(?P<mdm>{_MONTH})\.?\s+(?P<mdd>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<mdy>\d{{4}})\b"
    rf"|\b(?P<mym>{_MONTH})\.?\s+(?P<myy>\d{{4}})\b"
    rf"|\b(?P<y>(?:19|20)\d{{2}})\b", re.IGNORECASE)
RANGE_JOIN = re.compile(r"\b(to|and|until|till|through)\b|[-\u2013]")
START_CUE = re.compile(r"\b(since|starting|as of|from|beginning)\b")
END_CUE = re.compile(r"\b(until|till|through|up to|ending)\b")
ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def cue_text(claim: Claim) -> str:
    """Content, qualifiers and temporal expression, lowercased, without capitalized "May"/"Will"
    inside a sentence (a month or a name, not a modal)."""
    parts = [claim.content, *claim.qualifiers]
    if claim.temporal:
        parts.append(claim.temporal.expression)
    text = " ".join(parts)
    text = re.sub(r"(?<=[^\s.!?] )(May|Will)\b", " ", text)
    return " ".join(text.lower().replace("\u2019", "'").split())


# ---- dates ------------------------------------------------------------------


def _date(match: re.Match) -> str:
    if match.group("iso"):
        return match.group("iso")
    if match.group("mdy"):
        return f"{match.group('mdy')}-{MONTHS[match.group('mdm').lower()]:02d}-{int(match.group('mdd')):02d}"
    if match.group("myy"):
        return f"{match.group('myy')}-{MONTHS[match.group('mym').lower()]:02d}"
    return match.group("y")


def interval(date: str) -> tuple[str, str]:
    """First and last day covered by a date of any supported precision."""
    parts = date.split("-")
    if len(parts) == 1:
        return f"{parts[0]}-01-01", f"{parts[0]}-12-31"
    if len(parts) == 2:
        return f"{date}-01", f"{date}-31"
    return date, date


def before(a: str, b: str) -> bool:
    """``a`` ends strictly before ``b`` starts; overlapping dates are not ordered."""
    return interval(a)[1] < interval(b)[0]


def observation_date(timestamp: str | None) -> str | None:
    day = (timestamp or "")[:10]
    return day if ISO_DAY.match(day) else None


@dataclass(frozen=True)
class TemporalBounds:
    valid_from: str | None = None  # the state/event starts
    valid_until: str | None = None  # the state stops (only when stated)
    event_time: str | None = None  # a stated point in time that is not a boundary


def temporal_bounds(claim: Claim, transition: bool) -> TemporalBounds:
    """Bounds stated by the claim's temporal annotation. A point date is a start only for a
    transition ("moved to X in 2024"); otherwise it says when, not since or until when."""
    if claim.temporal is None:
        return TemporalBounds()
    expression = claim.temporal.expression
    found = list(DATE_TOKEN.finditer(expression))
    if len(found) >= 2 and RANGE_JOIN.search(expression[found[0].end():found[1].start()]):
        return TemporalBounds(_date(found[0]), _date(found[1]))
    date = _date(found[0]) if found else claim.temporal.resolved_date
    if not date:
        return TemporalBounds()
    cue = expression.lower()
    if START_CUE.search(cue):
        return TemporalBounds(valid_from=date)
    if END_CUE.search(cue):
        return TemporalBounds(valid_until=date)
    return TemporalBounds(valid_from=date if transition else None, event_time=date)


# ---- classification ---------------------------------------------------------


def classify(claim: Claim) -> str:
    text = cue_text(claim)
    if CONDITIONAL.search(text):
        return TIME_CONDITIONAL
    if FUTURE.search(text):
        return TIME_FUTURE
    past, present = PAST_STATE.search(text), PRESENT.search(text)
    if past and present:
        return TIME_MIXED
    if past:
        return TIME_PAST
    if TRANSITION.search(text):
        return TIME_TRANSITION
    if present:
        return TIME_PRESENT
    until = temporal_bounds(claim, False).valid_until
    observed = observation_date(claim.provenance.timestamp)
    if PAST_ANCHOR.search(text) or PAST_STATIVE.search(text) or (until and observed and before(until, observed)):
        return TIME_PAST
    if PAST_BE.search(text):
        return TIME_AMBIGUOUS_PAST
    return TIME_UNSPECIFIED


def blocks_change(claim: Claim) -> bool:
    return bool(NOT_A_CHANGE.search(cue_text(claim)))


def undated_content(claim: Claim) -> str:
    """Normalized content with the temporal expression removed."""
    text = normalize_value(claim.content)
    if claim.temporal:
        expression = normalize_value(claim.temporal.expression)
        if expression:
            text = f" {text} ".replace(f" {expression} ", " ", 1)
    return " ".join(text.split())


# ---- relative time expressions in message text ------------------------------------------------
# Moved unchanged from scripts/judge_locomo_answers.py so the answer path and the offline judge share
# one implementation. Expressions are resolved against the date the message was sent.

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def format_date(day: date) -> str:
    """Absolute date without a weekday, e.g. ``2 July 2023``."""
    return f"{day.day} {day:%B %Y}"


def format_day(day: date) -> str:
    """Absolute date with its weekday, used only when the expression itself names a weekday."""
    return f"{_WEEKDAYS[day.weekday()]} {day.day} {day:%B %Y}"


def _shift_month(day: date, months: int) -> date:
    index = day.year * 12 + day.month - 1 + months
    return date(index // 12, index % 12 + 1, 1)


def _previous_weekday(day: date, weekday: int) -> date:
    """Most recent date strictly before ``day`` that falls on ``weekday`` (0 = Monday)."""
    delta = (day.weekday() - weekday) % 7 or 7
    return day - timedelta(days=delta)


def _next_weekday(day: date, weekday: int) -> date:
    """First date strictly after ``day`` that falls on ``weekday``."""
    delta = (weekday - day.weekday()) % 7 or 7
    return day + timedelta(days=delta)


_WEEKDAY_WORD = r"(mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:r(?:s(?:day)?)?)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)"
_WEEKDAY_INDEX = {name[:3].lower(): i for i, name in enumerate(_WEEKDAYS)}
_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "couple of": 2}
_AGO_UNIT = r"(\d+|a couple of|couple of|an|a|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|week|month|year)s?\s+ago"


def _weekend(start: date) -> str:
    return f"the weekend of {format_date(start)} to {format_date(start + timedelta(days=1))}"


def _ago(match: re.Match, day: date) -> str:
    raw, unit = match.group(1).lower(), match.group(2).lower()
    n = int(raw) if raw.isdigit() else _NUMBER_WORDS[raw]
    if unit == "day":
        return format_date(day - timedelta(days=n))
    if unit == "week":
        return f"around {format_date(day - timedelta(days=7 * n))}"
    if unit == "month":
        return f"{_shift_month(day, -n):%B %Y}"
    return str(day.year - n)


# Longest expressions first; a span claimed by one pattern is not matched again.
_RELATIVE_RULES: tuple[tuple[re.Pattern, Callable[[re.Match, date], str]], ...] = tuple(
    (re.compile(pattern, re.I), resolve) for pattern, resolve in (
        (r"\bthe day before yesterday\b", lambda m, d: format_date(d - timedelta(days=2))),
        (r"\byesterday\b", lambda m, d: format_date(d - timedelta(days=1))),
        (r"\blast night\b", lambda m, d: f"the night of {format_date(d - timedelta(days=1))}"),
        (r"\b(?:today|tonight|this morning|this afternoon|this evening)\b", lambda m, d: format_date(d)),
        (r"\btomorrow\b", lambda m, d: format_date(d + timedelta(days=1))),
        (r"\b(?:last|this past)\s+weekend\b", lambda m, d: _weekend(_previous_weekday(d, 5))),
        (r"\b(?:next|this coming)\s+weekend\b", lambda m, d: _weekend(_next_weekday(d, 5))),
        (r"\b(?:last|this past)\s+week\b", lambda m, d: f"the week before {format_date(d)}"),
        (r"\bnext\s+week\b", lambda m, d: f"the week after {format_date(d)}"),
        (r"\blast\s+month\b", lambda m, d: f"{_shift_month(d, -1):%B %Y}"),
        (r"\bnext\s+month\b", lambda m, d: f"{_shift_month(d, 1):%B %Y}"),
        (r"\bthis\s+month\b", lambda m, d: f"{d:%B %Y}"),
        (r"\blast\s+year\b", lambda m, d: str(d.year - 1)),
        (r"\bnext\s+year\b", lambda m, d: str(d.year + 1)),
        (r"\bthis\s+year\b", lambda m, d: str(d.year)),
        (rf"\b(?:last|this past)\s+{_WEEKDAY_WORD}\b",
         lambda m, d: format_day(_previous_weekday(d, _WEEKDAY_INDEX[m.group(1)[:3].lower()]))),
        (rf"\bnext\s+{_WEEKDAY_WORD}\b",
         lambda m, d: format_day(_next_weekday(d, _WEEKDAY_INDEX[m.group(1)[:3].lower()]))),
        (rf"\b{_AGO_UNIT}\b", _ago),
    )
)


def resolve_relative_expressions(text: str, day: date) -> list[tuple[str, str]]:
    """Relative time expressions found in ``text``, each resolved against ``day`` (the turn's session date)."""
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str, str]] = []
    for pattern, resolve in _RELATIVE_RULES:
        for match in pattern.finditer(text or ""):
            start, end = match.span()
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            found.append((start, match.group(0), resolve(match, day)))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _, phrase, resolved in sorted(found):
        if phrase.lower() not in seen:
            seen.add(phrase.lower())
            out.append((phrase, resolved))
    return out


# ---- resolutions shown next to a message -----------------------------------------------------
# A layer over resolve_relative_expressions for answer context. It never edits the message; it only
# decides which resolutions are safe to show and how to label them.

RELATION_PAST = "past"
RELATION_CURRENT = "current"
RELATION_FUTURE = "future"

# "feels like just yesterday", "as if it were yesterday": figurative, not a date.
_FIGURATIVE_BEFORE = re.compile(r"\b(?:like|as if|as though)(?:\s+(?:just|only|it\s+was|it\s+were))*\s+$", re.I)
# "about five years ago": keep the approximation visible in the resolved value.
_APPROX_BEFORE = re.compile(r"\b(about|around|roughly|approximately|some)\s+$", re.I)
# "over five years ago", "nearly a week ago": a bound, not a date; nothing is shown.
_BOUND_BEFORE = re.compile(r"\b(?:over|more than|less than|almost|nearly|at least|under)\s+$", re.I)
_FUTURE_PHRASE = re.compile(r"\b(?:next|tomorrow|coming)\b", re.I)
_CURRENT_PHRASE = re.compile(r"^(?:this|today|tonight)\b(?!\s+past)", re.I)


class DateResolution(NamedTuple):
    phrase: str  # as written in the message, with an approximate qualifier if one precedes it
    value: str  # absolute date, period or range, as resolve_relative_expressions states it
    relation: str  # RELATION_PAST, RELATION_CURRENT or RELATION_FUTURE, relative to the message date


# LongMemEval session dates: "2023/03/04 (Sat) 14:32". The weekday and the time are optional.
_SLASH_TIMESTAMP = re.compile(
    r"^(\d{4})/(\d{1,2})/(\d{1,2})"
    r"(?:\s*\((?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\))?"
    r"(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?$",
    re.I,
)


def parse_timestamp(timestamp: str | None) -> datetime | None:
    """A message timestamp as a datetime: ISO-8601, or ``YYYY/MM/DD [(Ddd)] [HH:MM[:SS]]``.

    The weekday is informational and not checked against the date. None when missing or unsupported.
    """
    if not isinstance(timestamp, str) or not timestamp.strip():
        return None
    text = timestamp.strip()
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        pass
    match = _SLASH_TIMESTAMP.match(text)
    if not match:
        return None
    year, month, day, hour, minute, second = (int(g) if g else 0 for g in match.groups())
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def normalize_timestamp(timestamp: str | None) -> str | None:
    """Canonical ISO-8601 form of a supported message timestamp; None when unsupported."""
    parsed = parse_timestamp(timestamp)
    return parsed.isoformat() if parsed else None


def message_day(timestamp: str | None) -> date | None:
    """Calendar day of a supported message timestamp (see ``parse_timestamp``); None otherwise."""
    parsed = parse_timestamp(timestamp)
    return parsed.date() if parsed else None


def _occurrence_kind(text: str, phrase: str) -> tuple[str, str | None] | None:
    """("plain", None) or ("approx", qualifier) for the first usable occurrence; None if none is usable."""
    for match in re.finditer(rf"\b{re.escape(phrase)}\b", text, re.I):
        before = text[: match.start()]
        if _FIGURATIVE_BEFORE.search(before) or _BOUND_BEFORE.search(before):
            continue
        approx = _APPROX_BEFORE.search(before)
        return ("approx", approx.group(1)) if approx else ("plain", None)
    return None


def message_date_resolutions(text: str, day: date | None) -> tuple[DateResolution, ...]:
    """Relative expressions in one message resolved against that message's own date.

    Results equal to the message date itself (today, tonight, this morning) are left out because the
    date is shown anyway; figurative and bounded uses are left out because they name no date.
    """
    if day is None or not text:
        return ()
    out: list[DateResolution] = []
    for phrase, value in resolve_relative_expressions(text, day):
        if value == format_date(day):
            continue
        kind = _occurrence_kind(text, phrase)
        if kind is None:
            continue
        if kind[0] == "approx":
            phrase = f"{kind[1]} {phrase}"
            value = value if value.startswith("around ") else f"around {value}"
        if _FUTURE_PHRASE.search(phrase):
            relation = RELATION_FUTURE
        elif _CURRENT_PHRASE.match(phrase):
            relation = RELATION_CURRENT
        else:
            relation = RELATION_PAST
        out.append(DateResolution(phrase, value, relation))
    return tuple(out)


_MONTH_NAME = "|".join(sorted(MONTHS, key=len, reverse=True))
DATE_MENTION = re.compile(
    rf"\b(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m1>{_MONTH_NAME})\b\.?(?:,?\s+(?P<y1>\d{{4}})\b)?"
    rf"|\b(?P<m2>{_MONTH_NAME})\.?\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y2>\d{{4}})\b)?"
    rf"|\b(?P<m3>{_MONTH_NAME})\.?\s+(?P<y3>\d{{4}})\b"
    rf"|\b(?P<y4>(?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)


def date_mention_parts(m: re.Match) -> tuple[int | None, int | None, int | None]:
    """(day, month, year) of one ``DATE_MENTION`` match; unknown parts are None."""
    day = m.group("d1") or m.group("d2")
    month = m.group("m1") or m.group("m2") or m.group("m3")
    year = m.group("y1") or m.group("y2") or m.group("y3") or m.group("y4")
    return int(day) if day else None, MONTHS[month.lower()] if month else None, int(year) if year else None


def date_mentions(text: str) -> list[tuple[int | None, int | None, int | None]]:
    """(day, month, year) for each absolute date written in ``text``; unknown parts are None."""
    return [date_mention_parts(m) for m in DATE_MENTION.finditer(text or "")]
