"""Exploratory offline LLM-as-a-judge for two LoCoMo generation reports (blinded).

Reads two existing reports written by ``continuator_memory.webapi.locomo_report`` (a baseline run
and a context-enhanced run over the same questions), aligns every question with the official
LoCoMo record, and asks a local Ollama model to grade both generated answers. Nothing here touches
retrieval or answer generation, and no data leaves the machine: the judge host must be local
unless ``--allow-remote-host`` is passed.

Pipeline per question (judge version 6):

1. Normalize each answer in code: remove citation markers and meta wording about sources
   ("according to the conversation", "based on the excerpts").
2. Claim extraction (model): rewrite each distinct normalized answer as atomic factual claims with
   relative dates resolved; modifiers such as "recommended by X" become their own claims. The
   extractor never sees the reference answer or the run name. If a speaker name or quoted title in
   the answer is missing from the claims, or a claim's substantive words are mostly absent from the
   answer (an invented claim), extraction is repeated once with that feedback. Gaps the repeat does
   not fix are reported, never dropped. Weekday names next to full dates are removed in code.
3. Claim scoring (model): per claim, a verdict (supported, contradicted, not mentioned), the turns
   the claim is drawn from (``source_ids``, whatever the verdict) and, for contradicted claims, the
   contradicting turns; whether each reference item is covered; then correctness and completeness
   against the reference. Scoring is cached by the normalized claim set, so answers that make the same
   claims get the same scores.
4. In code: grounding is derived from the verdicts. Citation accuracy is derived from the cited ids:
   nonexistent turns are detected directly, and an existing cited turn is accurate when it is a
   source of one of the answer's claims. Citation accuracy does not depend on the verdicts, and
   citations never change factual scores. Completeness is derived from the reference items when the
   reference is a list (all covered = 2, some = 1, none = 0); a single-fact reference keeps the judge's
   completeness. An answer whose only claims say the information is not available scores 0 for
   correctness and completeness when the reference gives an answer. Overrides are recorded next to the
   judge's own scores, which are kept.
5. In code: a question is flagged as inconsistent, and left out of the headline win/loss/tie count, when
   the same claim got different verdicts in the two answers, equal answers or claim sets got different
   scores, or one answer adds only supported claims to the other's but scores lower correctness.

The judge sees the conversation around every gold-evidence turn and every cited turn (a window of
neighbouring turns in the same session), plus every turn either answer model was shown in its
prompt (no extra neighbours). Both answers are judged against the same turns. Relative time expressions are resolved in code per turn,
against that turn's own session date, and shown next to the turn.

This is an exploratory LLM-judge evaluation, not a ground-truth evaluator and not an official
LoCoMo score. See ``LIMITATIONS``.

Usage:
    python scripts/judge_locomo_answers.py --overwrite          # rerun on the existing reports
    python scripts/judge_locomo_answers.py --dry-run            # align and show prompts, no model calls
    python scripts/judge_locomo_answers.py --questions Q-017,Q-024 --output-json x.json --output-md x.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from continuator_memory.envfile import load_root_env  # noqa: E402
from continuator_memory.evaluation.metrics import normalize_text  # noqa: E402
from continuator_memory.generation import GenerationError  # noqa: E402
from continuator_memory.v2.temporal import format_date, format_day, resolve_relative_expressions  # noqa: E402
from continuator_memory.webapi.locomo import parse_locomo_datetime  # noqa: E402
from continuator_memory.webapi.query_plan import FUNCTION_WORDS  # noqa: E402
from score_locomo_answers import ParsedReport, parse_report  # noqa: E402

DEFAULT_DATASET = Path("locomo_conv26.json")
DEFAULT_BASELINE = Path("reports/pilot_conv26_q30_ctx0.md")
DEFAULT_CONTEXT = Path("reports/pilot_conv26_q30_ctx05.md")
DEFAULT_OUTPUT_JSON = Path("reports/pilot_conv26_q30_judgements.json")
DEFAULT_OUTPUT_MD = Path("reports/pilot_conv26_q30_judgements.md")
DEFAULT_MODEL = "qwen2.5:7b"
JUDGE_MODEL_ENV = "CONTINUATOR_JUDGE_MODEL"
DEFAULT_HOST = "http://localhost:11434"
LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 1024
DEFAULT_SEED = 0
DEFAULT_CONTEXT_WINDOW = 3
JUDGE_VERSION = "6"

RUN_BASELINE = "baseline"
RUN_CONTEXT = "context"
RUNS = (RUN_BASELINE, RUN_CONTEXT)
LABELS = ("A", "B")
DIMENSIONS = ("correctness", "grounding", "completeness")
SCORED_DIMENSIONS = ("correctness", "completeness")  # grounding is derived from claim verdicts

VERDICT_SUPPORTED = "supported"
VERDICT_CONTRADICTED = "contradicted"
VERDICT_NOT_MENTIONED = "not_mentioned"
VERDICTS = (VERDICT_SUPPORTED, VERDICT_CONTRADICTED, VERDICT_NOT_MENTIONED)

STATUS_OK = "ok"
STATUS_TIMEOUT = "timeout"
STATUS_REQUEST_ERROR = "request_error"
STATUS_INVALID_RESPONSE = "invalid_response"
STATUS_NOT_RUN = "not_run"

CITATION_NONE = "no_citations"
CITATION_ACCURATE = "accurate"
CITATION_MIXED = "mixed"
CITATION_INACCURATE = "inaccurate"

DISCLAIMER = (
    "Exploratory LLM-judge evaluation with a local model. It is not a ground-truth evaluation, not an "
    "official LoCoMo score, and not comparable with published LoCoMo results."
)

LIMITATIONS = (
    "The judge is the same model family and size (qwen2.5:7b by default) that generated the answers, so "
    "it may prefer its own phrasing and share its blind spots. Use --model to cross-check with another "
    "local model.",
    "A 7B judge makes mistakes: claim extraction can drop or distort a claim, claim verdicts can be wrong, "
    "and the turns it lists as claim sources decide citation accuracy. Spot-check results against the "
    "conversation before drawing conclusions.",
    "The extraction coverage check only covers speaker names and quoted titles; a dropped claim without "
    "a name (for example a date or a place) is not detected in code. The invented-claim check is a word "
    "overlap heuristic: it misses invented claims that reuse the answer's words and cannot tell whether "
    "a claim the answer does make was extracted faithfully.",
    "Relative times are resolved in code only for the expressions it recognizes (yesterday, last/next "
    "week, weekend, month, year, weekday names, 'N days/weeks ago'); week-level resolutions are approximate.",
    "Answers get identical factual scores only when their normalized text or their extracted claim sets "
    "match exactly; equivalent answers phrased differently may still be scored differently.",
    "The judge sees a window of turns around the gold evidence and cited turns, and the turns shown to "
    "either answer model, not the whole conversation; a contradiction far from those turns can be missed. "
    "Reports without a generator prompt fall back to gold and cited turns only.",
    "Some LoCoMo reference answers contain details that appear only in shared images (for example a book "
    "title), which no text-only answer can recover.",
    "The pilot has 30 questions, so small differences between runs are not meaningful.",
    "Reference items are split in code only for short comma-separated lists (no digits, no semicolon, no "
    "leading yes/no/likely); everything else is one item and keeps the judge's completeness. Whether an "
    "item is covered is still the judge's call.",
    "The abstention rule only applies when every claim says the information is not available; an answer "
    "that hedges while stating facts (\"plausible but not stated\") is scored by the judge as before. "
    "Likelihood and inference questions are not otherwise treated specially.",
    "The inconsistency check compares normalized claim text: equivalent claims worded differently are not "
    "matched, so some inconsistent pairs are not flagged. Flagged questions are left out of the headline "
    "outcome count only; their scores remain in every other metric.",
)

RUBRIC = {
    "correctness": {
        0: "incorrect: the claims do not give the reference answer, or say the information is unavailable",
        1: "partially correct: part of the essential answer is right, part is wrong, missing, or unresolvable",
        2: "correct: the claims match the reference in meaning (relative dates that resolve to the reference "
           "date count as correct)",
    },
    "grounding (derived from claim verdicts)": {
        0: "a claim is contradicted by the conversation, or no claim is supported",
        1: "some claims are supported, others are not mentioned in the conversation",
        2: "every claim is supported by the conversation",
    },
    "completeness": {
        0: "misses the essential answer (list reference: no item covered)",
        1: "partially complete (list reference: some items covered)",
        2: "complete (list reference: every item covered)",
    },
    "citation accuracy (derived)": {
        CITATION_NONE: "the answer cites no turn",
        CITATION_ACCURATE: "every cited turn exists and is the source of one of the answer's claims "
                           "(independent of whether the claim is correct)",
        CITATION_MIXED: "some cited turns are sources of the claims, others are nonexistent or are not",
        CITATION_INACCURATE: "no cited turn exists and is the source of a claim",
    },
}

EXTRACTION_SYSTEM_PROMPT = """You rewrite ONE answer about a past conversation as a list of short, self-contained factual claims.

You receive the question, excerpts of the conversation with the date of each session, relative time
expressions already resolved against each session date, and the answer.

Rules:
1. Write one claim per fact. Keep every factual claim the answer makes, including negative claims
   ("no other <items> are mentioned") and attributions (who said, did, gave or recommended something).
2. Split modifiers that add a fact about who or what into their own claim. For example "Tom bought a
   car his brother recommended" becomes "Tom bought a car." and "Tom's brother recommended the car
   that Tom bought." Keep a time or date in the same claim as the event it dates ("Tom bought the car
   on 2 May 2023"); never write a claim that only says what a date or "yesterday" was.
3. Make every claim understandable on its own: name the person and the specific thing ("the car that
   Tom bought"), never "it", "the book" or "a book" when the answer refers to a particular one.
4. Every person, title or name the answer mentions must appear in at least one claim.
5. Drop wording about sources, excerpts, citations or the conversation itself ("according to the
   conversation", "the excerpts show", "cited in the provided excerpts"); a claim such as "no other <items>
   are cited in the excerpts" becomes "<person> has no other <items>", with the answer's own person and items.
6. Resolve relative time expressions ("yesterday", "last Fri", "next month") with the resolution shown
   after the turn the fact comes from ("said on ..."), and write the absolute date. Do not add weekday
   names to dates.
7. The answer may quote the conversation in the first person ("I took my kids"); write the claim about
   the speaker of the quoted turn ("Melanie took her kids").
8. Do not add any fact, name, title or detail that the answer does not state. Do not correct the answer.
9. If the answer says the information is not available, write one claim of the form
   "The conversation does not say <what the question asks>".
10. If the answer is empty, return an empty list.

Respond with JSON only, in this shape: {"claims": ["..."]}"""

SCORING_SYSTEM_PROMPT = """You grade the claims made by ONE answer to a question about a past conversation between two people.

You receive the question, the benchmark reference answer, excerpts of the original conversation
(benchmark gold-evidence turns are marked; neighbouring turns and the turns the answer models were shown
are included), and the answer's claims. The claims were extracted from the answer with citations and wording about sources removed.

Dates: after a turn, "(said on <date>: "yesterday" = <date>)" gives that turn's relative time
expressions resolved in code against the turn's own session date. These resolutions are correct. Use
them as they are; do not recompute dates or work out weekdays yourself.

Step 1. For each claim, in the given order, give a verdict based only on the conversation excerpts,
never on the reference answer:
- "supported": the conversation explicitly states it. A paraphrase counts, and a date counts when it
  equals the resolution shown for the source turn (for example, a turn saying "yesterday" with
  "(said on 3 July 2023: "yesterday" = 2 July 2023)" supports a claim that it happened on 2 July 2023).
- "contradicted": the conversation states something incompatible with it, for example a different
  person did or recommended it, it was a different book or event, or it happened on a different date.
  A claim that something is not mentioned, or that there is nothing else, is contradicted when the
  excerpts do mention it.
- "not_mentioned": the excerpts neither state nor contradict it.
For every claim, list in source_ids the turns that state the event or fact the claim is about (the turn
the claim was drawn from), even when the claim gets a detail such as the date or the person wrong. A turn
about a similar but different event (another time, another person, another object) is not a source.
Use an empty list when no turn is about the claim. For a contradicted claim, list in contradicting_ids
the turns that contradict it.

Step 2. For each numbered reference item (R1, R2, ...), in the given order, set covered to true when the
claims give that item (equivalent wording counts) and false otherwise. A single item is the whole reference;
judge it as one fact.

Step 3. Score the claims taken together against the reference:
- correctness: 0 = incorrect, 1 = partially correct, 2 = correct.
- completeness: 0 = misses the essential answer, 1 = partially complete, 2 = complete.
Accept equivalent wording and any valid date format; a date that matches the reference after
resolution is correct (references may be written as "the Friday before 15 July 2023"). When the
reference lists several items, completeness is 2 only if all are given, 1 if some are, 0 if none; grade
against the reference even when an item appears only in a shared image. Do not award correctness for
repeating the reference when the conversation contradicts it. Claims saying the information is not
available get correctness 0 and completeness 0 when the reference gives an answer.

Write a one or two sentence rationale. Respond with JSON only, in this shape:
{"claims": [{"claim": "...", "verdict": "supported", "source_ids": ["D1:3"], "contradicting_ids": []}], "reference_items": [{"item": "...", "covered": true}], "rationale": "", "correctness": 0, "completeness": 0}"""

_SCORE_SCHEMA = {"type": "integer", "enum": [0, 1, 2]}
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["claims"],
}
SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "string", "enum": list(VERDICTS)},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                    "contradicting_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "verdict", "source_ids", "contradicting_ids"],
            },
        },
        "reference_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"item": {"type": "string"}, "covered": {"type": "boolean"}},
                "required": ["item", "covered"],
            },
        },
        "rationale": {"type": "string"},
        "correctness": _SCORE_SCHEMA,
        "completeness": _SCORE_SCHEMA,
    },
    "required": ["claims", "reference_items", "rationale", *SCORED_DIMENSIONS],
}

_EVIDENCE_ID = re.compile(r"D\d+:\d+")
_BRACKET = re.compile(r"\[([^\[\]]{1,80})\]")
_CITATION_MARKER = re.compile(r"\s*\[[^\[\]]{0,80}D\d+:\d+[^\[\]]{0,80}\]")
_META_PHRASE = re.compile(
    r"(?i)\s*,?\s*\b(?:according to|as (?:stated|mentioned|noted|shown|said|described)(?: in)?|based on|as per)"
    r"\s+(?:the\s+|her\s+|his\s+|their\s+)?(?:provided\s+|given\s+|available\s+|original\s+)?"
    r"(?:conversation|excerpts?|messages?|sources?|statements?|evidence|context|passages?|records?|turns?)\b"
)
_SESSION_KEY = re.compile(r"^session_(\d+)$")
_JSON_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)

JudgeCall = Callable[[str, str, dict], str]


class InputError(ValueError):
    """Inputs cannot be read or aligned unambiguously."""


class JudgeResponseError(ValueError):
    """The judge returned something that is not a valid judgement."""


class JudgeTimeoutError(GenerationError):
    pass


# ---- dataset -------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceTurn:
    dia_id: str
    session: int
    session_date: str | None
    session_date_iso: str | None
    speaker: str
    text: str
    caption: str | None = None
    position: int = 0  # order within the session

    @property
    def day(self) -> date | None:
        if not self.session_date_iso:
            return None
        try:
            return date.fromisoformat(self.session_date_iso[:10])
        except ValueError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.dia_id, "session": self.session, "sessionDate": self.session_date,
                "sessionDateIso": self.session_date_iso, "speaker": self.speaker, "text": self.text,
                "imageCaption": self.caption}


@dataclass(frozen=True)
class DatasetQuestion:
    index: int
    question: str
    reference: str  # "" when the record has no answer (e.g. adversarial questions)
    category: Any
    evidence_ids: tuple[str, ...]
    adversarial_answer: str | None = None

    @property
    def qid(self) -> str:
        return f"Q-{self.index + 1:03d}"

    @property
    def has_reference(self) -> bool:
        return bool(normalize_text(self.reference))


@dataclass
class Dataset:
    sample_id: str | None
    questions: list[DatasetQuestion]
    turns: dict[str, EvidenceTurn]

    @property
    def sessions(self) -> dict[int, list[str]]:
        return session_order(self.turns)


def session_order(turns: dict[str, EvidenceTurn]) -> dict[int, list[str]]:
    order: dict[int, list[EvidenceTurn]] = {}
    for turn in turns.values():
        order.setdefault(turn.session, []).append(turn)
    return {s: [t.dia_id for t in sorted(ts, key=lambda t: t.position)] for s, ts in sorted(order.items())}


def load_dataset(path: Path, sample_id: str | None = None) -> Dataset:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read dataset {path}: {exc}") from exc
    records = raw if isinstance(raw, list) else [raw]
    records = [r for r in records if isinstance(r, dict)]
    if sample_id is not None:
        records = [r for r in records if r.get("sample_id") == sample_id]
        if not records:
            raise InputError(f"no LoCoMo record with sample_id={sample_id!r} in {path}")
    if len(records) != 1:
        raise InputError(f"{path} has {len(records)} LoCoMo records; pass --sample-id to choose one")
    record = records[0]
    conversation = record.get("conversation")
    if not isinstance(conversation, dict) or not isinstance(record.get("qa"), list):
        raise InputError(f"{path}: record is missing 'conversation' or 'qa'")
    return Dataset(
        sample_id=record.get("sample_id"),
        questions=[_dataset_question(i, item) for i, item in enumerate(record["qa"])],
        turns=_conversation_turns(conversation),
    )


def _dataset_question(index: int, item: Any) -> DatasetQuestion:
    if not isinstance(item, dict):
        return DatasetQuestion(index, "", "", None, ())
    question = item.get("question") if isinstance(item.get("question"), str) else ""
    answer = item.get("answer")
    reference = "" if answer is None else str(answer)  # same formatting as the report's gold answer
    adversarial = item.get("adversarial_answer")
    return DatasetQuestion(index, question.strip(), reference, item.get("category"),
                           parse_evidence_ids(item.get("evidence")),
                           str(adversarial) if adversarial is not None else None)


def parse_evidence_ids(raw: Any) -> tuple[str, ...]:
    """Dialogue IDs in annotation order; tolerates entries such as ``"D8:6; D9:17"``."""
    if not isinstance(raw, list):
        return ()
    found: list[str] = []
    for item in raw:
        if isinstance(item, str):
            found.extend(_EVIDENCE_ID.findall(item))
    return tuple(dict.fromkeys(found))


def _conversation_turns(conversation: dict[str, Any]) -> dict[str, EvidenceTurn]:
    turns: dict[str, EvidenceTurn] = {}
    for key, value in conversation.items():
        match = _SESSION_KEY.fullmatch(key)
        if not match or not isinstance(value, list):
            continue
        session = int(match.group(1))
        when = conversation.get(f"session_{session}_date_time")
        when = when.strip() if isinstance(when, str) and when.strip() else None
        iso = parse_locomo_datetime(when) if when else None
        for position, turn in enumerate(value):
            if not isinstance(turn, dict) or not isinstance(turn.get("dia_id"), str):
                continue
            caption = turn.get("blip_caption")
            turns[turn["dia_id"].strip()] = EvidenceTurn(
                turn["dia_id"].strip(), session, when, iso if iso != when else None,
                str(turn.get("speaker") or "unknown"), str(turn.get("text") or ""),
                caption.strip() if isinstance(caption, str) and caption.strip() else None,
                position,
            )
    return turns


# ---- answer normalization and citations ------------------------------------------------------


def extract_citations(answer: str) -> tuple[str, ...]:
    """Dialogue IDs cited in square brackets, in order of first appearance."""
    found: list[str] = []
    for group in _BRACKET.findall(answer or ""):
        found.extend(_EVIDENCE_ID.findall(group))
    return tuple(dict.fromkeys(found))


def strip_citations(answer: str) -> str:
    """The answer without bracketed citation markers, whitespace tidied."""
    text = _CITATION_MARKER.sub("", answer or "")
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return " ".join(text.split())


def normalize_answer(answer: str) -> str:
    """Answer text without citation markers or meta wording about sources; what the judge grades."""
    text = _META_PHRASE.sub("", strip_citations(answer))
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"(^|[.!?]\s+)[,;:]\s*", r"\1", text)  # punctuation left where a sentence began with the phrase
    return " ".join(text.split())


def equivalent_answers(first: str, second: str) -> bool:
    return normalize_text(normalize_answer(first)) == normalize_text(normalize_answer(second))


def claims_key(claims: list[str]) -> tuple[str, ...]:
    return tuple(sorted(normalize_text(c) for c in claims if normalize_text(c)))


def derive_grounding(verdicts: list[str]) -> int:
    if not verdicts or VERDICT_CONTRADICTED in verdicts:
        return 0
    supported = verdicts.count(VERDICT_SUPPORTED)
    if supported == len(verdicts):
        return 2
    return 1 if supported else 0


def citation_status(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return CITATION_NONE
    good = sum(1 for c in citations if c["exists"] and c["supports"])
    if good == len(citations):
        return CITATION_ACCURATE
    return CITATION_MIXED if good else CITATION_INACCURATE


def evaluate_citations(cited_ids: list[str], claims: list[dict[str, Any]], turns: dict[str, EvidenceTurn],
                       gold: set[str]) -> list[dict[str, Any]]:
    """A cited turn is accurate when it exists and is a source of one of the answer's claims.

    Citation accuracy asks whether the citation points at the turn the claim comes from. It does not
    depend on the claim's verdict: a claim with a wrong detail can still cite its correct source, and a
    correct claim can cite the wrong turn.
    """
    sources: dict[str, str] = {}
    for claim in claims:
        for sid in claim["sourceIds"]:
            sources.setdefault(sid, claim["claim"])
    out = []
    for cid in cited_ids:
        if cid not in turns:
            out.append({"id": cid, "exists": False, "inGold": False, "supports": False,
                        "reason": "cited turn does not exist in the conversation"})
        elif cid in sources:
            out.append({"id": cid, "exists": True, "inGold": cid in gold, "supports": True,
                        "reason": f"source of: {sources[cid]}"})
        else:
            out.append({"id": cid, "exists": True, "inGold": cid in gold, "supports": False,
                        "reason": "exists, but is not the source of any of the answer's claims"})
    return out


_QUOTED = re.compile(r"[\"“]([^\"“”]{2,80})[\"”]")


def answer_entities(answer: str, names: set[str]) -> list[str]:
    """People named in the answer (from the conversation's speakers) and quoted titles."""
    text = normalize_answer(answer)
    found = [n for n in sorted(names) if n and re.search(rf"\b{re.escape(n)}\b", text, re.I)]
    found += [q.strip() for q in _QUOTED.findall(text)]
    return list(dict.fromkeys(found))


def missing_entities(entities: list[str], claims: list[str]) -> list[str]:
    joined = [f" {normalize_text(c)} " for c in claims]
    return [e for e in entities if not any(f" {normalize_text(e)} " in c for c in joined)]


# Words a faithful claim may add although the answer does not use them: the "does not say" form of rule 9,
# resolved dates (rule 6) and common verbs of possession.
_CLAIM_FRAME_WORDS = frozenset({
    "conversation", "not", "say", "says", "mention", "mentions", "mentioned", "state", "states", "stated",
    "has", "have", "had", "also", "other", "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "week", "weekend", "night", "around", "before", "after",
})


def _substantive(text: str) -> list[str]:
    return [t for t in normalize_text(text).split()
            if len(t) >= 3 and not t.isdigit() and t not in FUNCTION_WORDS and t not in _CLAIM_FRAME_WORDS]


def _word_known(word: str, known: set[str]) -> bool:
    """Same word, or the same stem ("book"/"books", "paint"/"painting")."""
    return word in known or any(min(len(word), len(k)) >= 4 and (word.startswith(k) or k.startswith(word))
                                for k in known)


def invented_claims(item: AlignedQuestion, answer: str, claims: list[str]) -> list[str]:
    """Claims whose substantive words are mostly absent from the answer, the question and the speaker names.

    A conservative heuristic: a flagged claim is very likely not in the answer, but an unflagged claim is
    not proven faithful (it may reuse the answer's words to say something else).
    """
    known = set(_substantive(f"{normalize_answer(answer)} {item.dataset.question} {' '.join(speaker_names(item))}"))
    flagged = []
    for claim in claims:
        words = _substantive(claim)
        absent = [w for w in words if not _word_known(w, known)]
        if len(absent) >= 2 and len(absent) * 2 > len(words):
            flagged.append(claim)
    return flagged


# ---- reference items, derived completeness and overrides ------------------------------------


_QUOTE_CLOSE = {'"': '"', "\u201c": "\u201d"}
# A part opening with one of these is an answer with a justification or a clause, not a list item.
_NOT_LIST_START = frozenset({"yes", "no", "likely", "unlikely", "probably", "maybe", "possibly", "not",
                             "since", "though", "because", "but", "so", "as", "which", "who"})
_MAX_ITEM_WORDS = 4


def _split_commas_outside_quotes(text: str) -> list[str]:
    parts, buf, closer = [], [], None
    for ch in text:
        if closer:
            buf.append(ch)
            closer = None if ch == closer else closer
        elif ch in _QUOTE_CLOSE:
            closer = _QUOTE_CLOSE[ch]
            buf.append(ch)
        elif ch == ",":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def reference_items(reference: str) -> list[str]:
    """Items an answer must give: a short comma-separated list ("pottery, camping, and swimming") is split;
    anything else (a date, a single fact, "Likely no; she ...", a long clause list) is one item."""
    text = " ".join((reference or "").split())
    if not text:
        return []
    if ";" in text or re.search(r"\d", text):
        return [text]
    parts = [re.sub(r"^(?:and|or)\s+", "", p.strip(), flags=re.I).strip() for p in _split_commas_outside_quotes(text)]
    parts = [p for p in parts if p]
    words = [re.findall(r"[\w']+", p.lower()) for p in parts]
    if len(parts) >= 2 and all(1 <= len(w) <= _MAX_ITEM_WORDS and w[0] not in _NOT_LIST_START for w in words):
        return parts
    return [text]


def derive_completeness(items: list[dict[str, Any]], judge_completeness: int) -> tuple[int, str]:
    """(completeness, source): from item coverage for a list reference, else the judge's own score."""
    if len(items) < 2:
        return judge_completeness, "judge"
    covered = sum(1 for i in items if i["covered"])
    return (2 if covered == len(items) else 1 if covered else 0), "reference_items"


_ABSTENTION = re.compile(r"^the conversation does not (?:say|mention|state|specify)\b")


def is_abstention(claims: list[str]) -> bool:
    """Every claim is the rule-9 form "The conversation does not say <what the question asks>"."""
    return bool(claims) and all(_ABSTENTION.match(normalize_text(c)) for c in claims)


def inconsistency(record: dict[str, Any]) -> str | None:
    """Why the two answers' scores disagree although their claims say they should not; None if they agree.

    A score difference is left alone when the answers differ by a claim that is not supported."""
    b, c = record["byRun"][RUN_BASELINE], record["byRun"][RUN_CONTEXT]
    if record.get("equivalentAnswers") or record.get("sharedClaims"):
        diff = [d for d in DIMENSIONS if b[d] != c[d]]
        return f"equal answers or claims scored differently ({', '.join(diff)})" if diff else None
    verdicts = {run: {normalize_text(x["claim"]): x["verdict"] for x in record["byRun"][run]["claims"]} for run in RUNS}
    vb, vc = verdicts[RUN_BASELINE], verdicts[RUN_CONTEXT]
    clash = [k for k in vb if k in vc and vb[k] != vc[k]]
    if clash:
        return f"the same claim got different verdicts ({vb[clash[0]]} vs {vc[clash[0]]}): {clash[0]}"
    for small, big in ((RUN_BASELINE, RUN_CONTEXT), (RUN_CONTEXT, RUN_BASELINE)):
        extra = [k for k in verdicts[big] if k not in verdicts[small]]
        if (verdicts[small] and extra and set(verdicts[small]) <= set(verdicts[big])
                and all(verdicts[big][k] == VERDICT_SUPPORTED for k in extra)
                and record["byRun"][big]["correctness"] < record["byRun"][small]["correctness"]):
            return f"{big} adds only supported claims to {small} but scores lower correctness"
    return None


# ---- relative dates --------------------------------------------------------------------------


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_WEEKDAY_BEFORE_DATE = re.compile(
    rf"\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day,?\s+(?=(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})|(?:{_MONTHS})\s+\d))",
    re.I,
)


def drop_weekdays_before_dates(text: str) -> str:
    """``on Sunday 2 July 2023`` -> ``on 2 July 2023``: weekdays add nothing to a full date and invite
    weekday/date mismatches. A weekday without a date (``on Friday``) is kept."""
    return " ".join(_WEEKDAY_BEFORE_DATE.sub("", text).split())


# ---- alignment -----------------------------------------------------------------------------


@dataclass
class AlignedQuestion:
    qid: str
    dataset: DatasetQuestion
    answers: dict[str, str]
    evidence: list[EvidenceTurn]
    missing_evidence_ids: list[str]
    warnings: list[str] = field(default_factory=list)
    turns: dict[str, EvidenceTurn] = field(default_factory=dict)  # the whole conversation
    prompt_ids: tuple[str, ...] = ()  # existing turns shown to either answer model, in prompt order

    def citations(self, run: str) -> tuple[str, ...]:
        return extract_citations(self.answers[run])

    def cited_turns(self) -> list[EvidenceTurn]:
        """Existing turns cited by either answer that are not gold evidence, in order of appearance."""
        gold = {t.dia_id for t in self.evidence}
        ids = dict.fromkeys(cid for run in RUNS for cid in self.citations(run))
        return [self.turns[cid] for cid in ids if cid in self.turns and cid not in gold]


def _qid_index(qid: str) -> int | None:
    digits = qid.split("-", 1)[-1]
    return int(digits) - 1 if digits.isdigit() else None


def align(dataset: Dataset, baseline: ParsedReport, context: ParsedReport) -> list[AlignedQuestion]:
    """Pair both reports with dataset questions. Raises ``InputError`` listing every problem."""
    errors: list[str] = []
    for name, report in ((RUN_BASELINE, baseline), (RUN_CONTEXT, context)):
        errors += [f"{name} report: {p}" for p in report.problems]
        if not report.sections:
            errors.append(f"{name} report has no question sections")
    only_b = sorted(set(baseline.sections) - set(context.sections))
    only_c = sorted(set(context.sections) - set(baseline.sections))
    if only_b:
        errors.append(f"questions only in the baseline report: {', '.join(only_b)}")
    if only_c:
        errors.append(f"questions only in the context report: {', '.join(only_c)}")

    by_text: dict[str, list[DatasetQuestion]] = {}
    for q in dataset.questions:
        by_text.setdefault(normalize_text(q.question), []).append(q)

    aligned: list[AlignedQuestion] = []
    claimed: dict[int, str] = {}
    for qid in sorted(set(baseline.sections) & set(context.sections), key=lambda q: (_qid_index(q) or 0, q)):
        b, c = baseline.sections[qid], context.sections[qid]
        problems = [f"baseline: {p}" for p in b.problems] + [f"context: {p}" for p in c.problems]
        if b.question is None or c.question is None:
            problems.append("missing 'Question' block")
        if problems:
            errors.append(f"{qid}: " + "; ".join(problems))
            continue
        text = normalize_text(b.question)
        if text != normalize_text(c.question):
            errors.append(f"{qid}: question text differs between reports")
            continue
        candidates = by_text.get(text, [])
        index = _qid_index(qid)
        warnings: list[str] = []
        exact = [q for q in candidates if q.index == index]
        if exact:
            match = exact[0]
        elif len(candidates) == 1:
            match = candidates[0]
            warnings.append(f"report id {qid} matched dataset question {match.qid} by question text")
        elif not candidates:
            errors.append(f"{qid}: question text not found in dataset: {b.question!r}")
            continue
        else:
            errors.append(f"{qid}: question text matches several dataset questions "
                          f"({', '.join(q.qid for q in candidates)}) and none has this id")
            continue
        if match.index in claimed:
            errors.append(f"{qid}: dataset question {match.qid} is already matched by {claimed[match.index]}")
            continue
        claimed[match.index] = qid
        reference = normalize_text(match.reference)
        if normalize_text(b.gold or "") != reference or normalize_text(c.gold or "") != reference:
            errors.append(f"{qid}: report gold answer does not match the dataset answer {match.reference!r}")
            continue
        evidence = [dataset.turns[e] for e in match.evidence_ids if e in dataset.turns]
        missing = [e for e in match.evidence_ids if e not in dataset.turns]
        if missing:
            warnings.append(f"evidence ids not in conversation: {', '.join(missing)}")
        if not match.evidence_ids:
            warnings.append("no gold evidence ids")
        if not match.has_reference:
            warnings.append("no reference answer in dataset")
        for name, sec in ((RUN_BASELINE, b), (RUN_CONTEXT, c)):
            if sec.prompt is None:
                warnings.append(f"{name} report has no generator prompt; judge context omits the turns it was shown")
        shown = dict.fromkeys(i for sec in (b, c) for i in prompt_excerpt_ids(sec.prompt))
        unknown = [i for i in shown if i not in dataset.turns]
        if unknown:
            warnings.append(f"generator prompt ids not in conversation: {', '.join(unknown)}")
        aligned.append(AlignedQuestion(qid, match, {RUN_BASELINE: b.generated or "", RUN_CONTEXT: c.generated or ""},
                                       evidence, missing, warnings, dataset.turns,
                                       tuple(i for i in shown if i in dataset.turns)))
    if errors:
        raise InputError("cannot align reports with the dataset:\n- " + "\n- ".join(errors))
    return aligned


# ---- conversation context and prompts -----------------------------------------------------------


def assign_labels(qid: str, seed: int) -> dict[str, str]:
    """Seeded, per-question A/B assignment: ``{"A": run, "B": run}``. A is graded first."""
    baseline_first = hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).digest()[0] % 2 == 0
    return {"A": RUN_BASELINE, "B": RUN_CONTEXT} if baseline_first else {"A": RUN_CONTEXT, "B": RUN_BASELINE}


_PROMPT_EXCERPT = re.compile(r"^\[(D\d+:\d+)\]", re.M)


def prompt_excerpt_ids(prompt: str | None) -> tuple[str, ...]:
    """Message ids of the excerpts in a generator prompt (one ``[D1:3] ...`` line opens each), in order."""
    return tuple(dict.fromkeys(_PROMPT_EXCERPT.findall(prompt or "")))


def context_turns(item: AlignedQuestion, window: int) -> list[EvidenceTurn]:
    """Gold-evidence and cited turns plus ``window`` neighbours on each side within their session, and the
    turns shown to either answer model (without neighbours). The same turns serve both answers."""
    anchors = [t.dia_id for t in item.evidence] + [t.dia_id for t in item.cited_turns()]
    order = session_order(item.turns)
    selected: set[str] = set(item.prompt_ids)
    for anchor in anchors:
        turn = item.turns[anchor]
        ids = order[turn.session]
        at = ids.index(anchor)
        selected.update(ids[max(0, at - window): at + window + 1])
    return sorted((item.turns[i] for i in selected), key=lambda t: (t.session, t.position))


def _quote(text: str) -> str:
    body = text.strip() or "(empty answer)"
    return "\n".join(f"> {line}" if line else ">" for line in body.splitlines())


def _turn_lines(turn: EvidenceTurn, gold: set[str]) -> list[str]:
    marker = " (gold evidence)" if turn.dia_id in gold else ""
    lines = [f"[{turn.dia_id}]{marker} {turn.speaker}: {turn.text}"]
    if turn.caption:
        lines.append(f"    [shared image: {turn.caption}]")
    day = turn.day
    resolved = resolve_relative_expressions(turn.text, day) if day else []
    if resolved:
        lines.append(f"    (said on {format_date(day)}: "
                     + "; ".join(f'"{phrase}" = {value}' for phrase, value in resolved) + ")")
    return lines


def render_context(item: AlignedQuestion, window: int) -> list[str]:
    turns = context_turns(item, window)
    gold = {t.dia_id for t in item.evidence}
    if not turns:
        return ["(no conversation excerpts are available for this question)"]
    lines: list[str] = []
    session = None
    previous_position = None
    for turn in turns:
        if turn.session != session:
            session, previous_position = turn.session, None
            day = turn.day
            when = f"{format_date(day)} ({turn.session_date})" if day else (turn.session_date or "date unknown")
            if lines:
                lines.append("")
            lines.append(f"Session {session}, {when}:")
        elif previous_position is not None and turn.position != previous_position + 1:
            lines.append("  ...")
        lines += [f"  {line}" for line in _turn_lines(turn, gold)]
        previous_position = turn.position
    return lines


def _context_block(item: AlignedQuestion, window: int) -> list[str]:
    lines = ["Conversation excerpts (gold-evidence turns are marked; neighbouring turns and the turns the answer "
             "models were shown are included; relative times in a turn are resolved in code against that turn's "
             "own session date, shown as \"said on\"):",
             *render_context(item, window)]
    if item.missing_evidence_ids:
        lines.append(f"(annotated evidence ids not found in the conversation: {', '.join(item.missing_evidence_ids)})")
    return lines


def _reference_text(q: DatasetQuestion) -> str:
    if q.has_reference:
        return q.reference
    return ("(none) The benchmark gives no answer for this question. If the conversation does not support the "
            "question's premise, the correct response is to say the information is not available.")


def build_extraction_prompt(item: AlignedQuestion, answer: str, window: int = DEFAULT_CONTEXT_WINDOW,
                            missing: list[str] | None = None, invented: list[str] | None = None) -> str:
    """Claim extraction for one normalized answer. No reference answer, no run name.

    ``missing`` names people or titles a previous extraction left out and ``invented`` lists claims it made
    that the answer does not; the request is repeated once with that feedback.
    """
    lines = [f"Question: {item.dataset.question}", "", *_context_block(item, window), "",
             "Answer:", _quote(normalize_answer(answer)), ""]
    if missing:
        lines += [f"A previous list of claims for this answer left out: {', '.join(missing)}. Every person, "
                  "title or name the answer mentions must appear in a claim, and attributions (who said, did, "
                  "gave or recommended something) must be separate claims.", ""]
    if invented:
        lines += ["A previous list of claims for this answer included claims the answer does not make: "
                  + "; ".join(f'"{c}"' for c in invented) + ". List only claims this answer states.", ""]
    lines.append("List the answer's factual claims. Respond with the JSON object only.")
    return "\n".join(lines)


def speaker_names(item: AlignedQuestion) -> set[str]:
    return {t.speaker for t in item.turns.values() if t.speaker and t.speaker != "unknown"}


def build_scoring_prompt(item: AlignedQuestion, claims: list[str], window: int = DEFAULT_CONTEXT_WINDOW) -> str:
    """Claim scoring. Depends only on the question, the conversation and the claims."""
    numbered = [f"{i}. {claim}" for i, claim in enumerate(claims, 1)] or ["(the answer makes no claim)"]
    items = [f"R{i}. {it}" for i, it in enumerate(reference_items(item.dataset.reference), 1)] or ["(none)"]
    return "\n".join([f"Question: {item.dataset.question}", "", f"Reference answer: {_reference_text(item.dataset)}",
                      "", "Reference items:", *items,
                      "", *_context_block(item, window), "", "Claims made by the answer:", *numbered, "",
                      "Give a verdict for every claim in order, mark every reference item, then score. "
                      "Respond with the JSON object only."])


# ---- judge responses -------------------------------------------------------------------------


def _load_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    fenced = _JSON_FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    if not text:
        raise JudgeResponseError("empty response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeResponseError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise JudgeResponseError("response JSON is not an object")
    return data


def _score(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JudgeResponseError(f"{where} must be an integer 0, 1 or 2, got {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise JudgeResponseError(f"{where} must be an integer 0, 1 or 2, got {value!r}")
        value = int(value)
    if value not in (0, 1, 2):
        raise JudgeResponseError(f"{where} must be 0, 1 or 2, got {value!r}")
    return value


def _text(value: Any, where: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise JudgeResponseError(f"{where} must be a string")
    return value.strip()


def _id_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise JudgeResponseError(f"{where} must be a list of strings")
    return list(dict.fromkeys(m for x in value for m in _EVIDENCE_ID.findall(x)))


def parse_extraction(raw: str) -> list[str]:
    data = _load_json_object(raw)
    claims = data.get("claims")
    if not isinstance(claims, list) or not all(isinstance(c, str) for c in claims):
        raise JudgeResponseError("'claims' must be a list of strings")
    return list(dict.fromkeys(c.strip() for c in claims if c.strip()))


def parse_scoring(raw: str, claims: list[str], known_ids: set[str], items: list[str] = ()) -> dict[str, Any]:
    """Validate claim verdicts (one per claim, in order), reference-item coverage (one per item, in order)
    and the two model-scored dimensions."""
    data = _load_json_object(raw)
    out: dict[str, Any] = {dim: _score(data.get(dim), dim) if dim in data else None for dim in SCORED_DIMENSIONS}
    for dim in SCORED_DIMENSIONS:
        if out[dim] is None:
            raise JudgeResponseError(f"missing '{dim}'")
    entries = data.get("claims")
    if not isinstance(entries, list):
        raise JudgeResponseError("'claims' must be a list")
    if len(entries) != len(claims):
        raise JudgeResponseError(f"expected {len(claims)} claim verdicts, got {len(entries)}")
    verdicts = []
    for i, (entry, claim) in enumerate(zip(entries, claims), 1):
        if not isinstance(entry, dict):
            raise JudgeResponseError(f"claim {i}: verdict entry must be an object")
        verdict = entry.get("verdict")
        if verdict not in VERDICTS:
            raise JudgeResponseError(f"claim {i}: verdict must be one of {', '.join(VERDICTS)}, got {verdict!r}")
        source = _id_list(entry.get("source_ids", []), f"claim {i}: source_ids")
        against = _id_list(entry.get("contradicting_ids", []), f"claim {i}: contradicting_ids")
        verdicts.append({
            "claim": claim,
            "verdict": verdict,
            # Kept for every verdict: the source of a claim is independent of whether the claim is right.
            "sourceIds": [x for x in source if x in known_ids],
            "contradictingIds": [x for x in against if x in known_ids] if verdict == VERDICT_CONTRADICTED else [],
            "unknownIds": [x for x in dict.fromkeys(source + against) if x not in known_ids],
        })
    out["claims"] = verdicts
    coverage = data.get("reference_items", []) if items else []
    if not isinstance(coverage, list) or len(coverage) != len(items):
        raise JudgeResponseError(f"expected {len(items)} reference items, got "
                                 f"{len(coverage) if isinstance(coverage, list) else 'no list'}")
    out["referenceItems"] = []
    for i, (entry, text) in enumerate(zip(coverage, items), 1):
        if not isinstance(entry, dict) or not isinstance(entry.get("covered"), bool):
            raise JudgeResponseError(f"reference item {i}: 'covered' must be true or false")
        out["referenceItems"].append({"item": text, "covered": entry["covered"]})
    out["rationale"] = _text(data.get("rationale"), "rationale")
    return out


# ---- Ollama ----------------------------------------------------------------------------------


def is_local_host(host: str) -> bool:
    return (urlparse(host).hostname or "").lower() in LOCAL_HOSTNAMES


class OllamaJudgeClient:
    """Minimal /api/chat client with structured output; mirrors ``continuator_memory.generation``."""

    def __init__(self, model: str, host: str, *, timeout: float = DEFAULT_TIMEOUT_S, seed: int = DEFAULT_SEED,
                 num_ctx: int = DEFAULT_NUM_CTX, num_predict: int = DEFAULT_NUM_PREDICT, format_mode: str = "schema"):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.seed = seed
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.format_mode = format_mode

    def options(self) -> dict[str, Any]:
        return {"temperature": 0.0, "seed": self.seed, "num_ctx": self.num_ctx, "num_predict": self.num_predict}

    def __call__(self, system: str, prompt: str, schema: dict) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "format": schema if self.format_mode == "schema" else "json",
            "options": self.options(),
        }
        request = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GenerationError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise JudgeTimeoutError(f"Ollama did not answer within {self.timeout:g}s") from exc
            raise GenerationError(f"could not reach Ollama at {self.host} ({exc.reason}); is `ollama serve` running?") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise JudgeTimeoutError(f"Ollama did not answer within {self.timeout:g}s") from exc
        except json.JSONDecodeError as exc:
            raise GenerationError(f"Ollama returned a non-JSON body: {exc}") from exc
        try:
            return body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise GenerationError(f"unexpected Ollama response: {body!r}") from exc


# ---- judging ---------------------------------------------------------------------------------


@dataclass
class CallResult:
    kind: str
    label: str
    status: str
    attempts: int
    latency_ms: float
    raw: str | None = None
    error: str | None = None
    parsed: Any = None
    reused_from: str | None = None

    def reuse(self, label: str) -> CallResult:
        return CallResult(self.kind, label, self.status, 0, 0.0, self.raw, self.error, self.parsed,
                          reused_from=self.label)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "status": self.status, "attempts": self.attempts,
                "latencyMs": self.latency_ms, "rawResponse": self.raw, "error": self.error,
                "reusedFrom": self.reused_from}


def call_judge(call: JudgeCall, system: str, prompt: str, schema: dict, parse: Callable[[str], Any], *,
               kind: str, label: str, retries: int) -> CallResult:
    """One judge request with retries on timeouts and request errors (not on invalid responses)."""
    started = time.perf_counter()
    result = CallResult(kind, label, STATUS_NOT_RUN, 0, 0.0)
    for attempt in range(1, retries + 2):
        result.attempts = attempt
        try:
            raw = call(system, prompt, schema)
        except JudgeTimeoutError as exc:
            result.status, result.error = STATUS_TIMEOUT, str(exc)
            continue
        except GenerationError as exc:
            result.status, result.error = STATUS_REQUEST_ERROR, str(exc)
            continue
        result.raw = raw
        try:
            result.parsed = parse(raw)
        except JudgeResponseError as exc:
            result.status, result.error = STATUS_INVALID_RESPONSE, str(exc)
            break  # deterministic decoding: the same request would repeat the same output
        result.status, result.error = STATUS_OK, None
        break
    result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
    return result


def base_record(item: AlignedQuestion, labels: dict[str, str], window: int = DEFAULT_CONTEXT_WINDOW) -> dict[str, Any]:
    return {
        "qid": item.qid,
        "datasetIndex": item.dataset.index,
        "category": item.dataset.category,
        "question": item.dataset.question,
        "reference": item.dataset.reference,
        "referenceAvailable": item.dataset.has_reference,
        "evidenceIds": list(item.dataset.evidence_ids),
        "evidence": [t.to_dict() for t in item.evidence],
        "missingEvidenceIds": list(item.missing_evidence_ids),
        "citedNonGoldTurns": [t.to_dict() for t in item.cited_turns()],
        "contextTurnIds": [t.dia_id for t in context_turns(item, window)],
        "promptTurnIds": list(item.prompt_ids),
        "warnings": list(item.warnings),
        "labels": labels,
        "answers": dict(item.answers),
        "normalizedAnswers": {run: normalize_answer(item.answers[run]) for run in RUNS},
        "equivalentAnswers": equivalent_answers(item.answers[RUN_BASELINE], item.answers[RUN_CONTEXT]),
        "sharedClaims": False,
        "inconsistent": None,
        "status": STATUS_NOT_RUN,
        "error": None,
        "latencyMs": None,
        "calls": [],
        "byRun": None,
    }


def _fail(record: dict[str, Any], result: CallResult, what: str, started: float) -> dict[str, Any]:
    record.update(status=result.status, error=f"answer {result.label} {what}: {result.error}")
    record["latencyMs"] = round((time.perf_counter() - started) * 1000, 1)
    return record


@dataclass
class Extraction:
    calls: list[CallResult]
    claims: list[str] | None
    coverage: dict[str, Any]

    @property
    def failure(self) -> CallResult | None:
        return next((c for c in self.calls if c.status != STATUS_OK), None) if self.claims is None else None

    def reuse(self, label: str) -> Extraction:
        return Extraction([c.reuse(label) for c in self.calls], self.claims, self.coverage)


def extract_claims(item: AlignedQuestion, answer: str, call: JudgeCall, *, label: str, retries: int,
                   window: int) -> Extraction:
    """Extract claims; repeat once with feedback if a named person or quoted title was dropped or a claim
    looks invented. The repeat is kept only if it fixes a gap without opening another; gaps that remain
    are reported in the coverage, and no claim is removed."""
    entities = answer_entities(answer, speaker_names(item))
    coverage: dict[str, Any] = {"entities": entities, "initiallyMissing": [], "initiallyInvented": [],
                                "repaired": False, "missing": [], "invented": []}
    if not normalize_text(normalize_answer(answer)):
        return Extraction([CallResult("extraction", label, STATUS_OK, 0, 0.0, parsed=[])], [], coverage)
    first = call_judge(call, EXTRACTION_SYSTEM_PROMPT, build_extraction_prompt(item, answer, window),
                       EXTRACTION_SCHEMA, parse_extraction, kind="extraction", label=label, retries=retries)
    if first.status != STATUS_OK:
        return Extraction([first], None, coverage)
    calls, claims = [first], first.parsed
    missing = missing_entities(entities, claims)
    invented = invented_claims(item, answer, claims)
    coverage["initiallyMissing"] = missing
    coverage["initiallyInvented"] = invented
    if missing or invented:
        repair = call_judge(call, EXTRACTION_SYSTEM_PROMPT,
                            build_extraction_prompt(item, answer, window, missing, invented),
                            EXTRACTION_SCHEMA, parse_extraction, kind="extraction_repair", label=label,
                            retries=retries)
        calls.append(repair)
        if repair.status == STATUS_OK and (repair.parsed or not claims):
            still_missing = missing_entities(entities, repair.parsed)
            still_invented = invented_claims(item, answer, repair.parsed)
            if (len(still_missing) <= len(missing) and len(still_invented) <= len(invented)
                    and (len(still_missing) < len(missing) or len(still_invented) < len(invented))):
                claims = repair.parsed
                coverage["repaired"] = True
        missing = missing_entities(entities, claims)
        invented = invented_claims(item, answer, claims)
    coverage["missing"] = missing
    coverage["invented"] = invented
    # A full date already fixes the day; a weekday next to it only invites weekday/date mismatches.
    claims = list(dict.fromkeys(drop_weekdays_before_dates(c) for c in claims))
    return Extraction(calls, claims, coverage)


def judge_question(item: AlignedQuestion, call: JudgeCall, *, seed: int, retries: int = 1,
                   window: int = DEFAULT_CONTEXT_WINDOW) -> dict[str, Any]:
    labels = assign_labels(item.qid, seed)
    record = base_record(item, labels, window)
    gold = {t.dia_id for t in item.evidence}
    known_ids = {t.dia_id for t in context_turns(item, window)}
    items = reference_items(item.dataset.reference)
    started = time.perf_counter()
    extractions: dict[str, Extraction] = {}
    scorings: dict[tuple[str, ...], CallResult] = {}
    claim_keys: dict[str, tuple[str, ...]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    by_run: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        run = labels[label]
        answer = item.answers[run]
        text_key = normalize_text(normalize_answer(answer))
        if text_key in extractions:  # same normalized answer: same claims
            extraction = extractions[text_key].reuse(label)
        else:
            extraction = extract_claims(item, answer, call, label=label, retries=retries, window=window)
            extractions[text_key] = extraction
        record["calls"] += [c.to_dict() for c in extraction.calls if c.attempts or c.reused_from]
        if extraction.claims is None:
            return _fail(record, extraction.failure, "claim extraction", started)
        claims: list[str] = extraction.claims
        coverage[run] = extraction.coverage
        key = claims_key(claims)
        claim_keys[run] = key
        if key in scorings:  # same claims: same verdicts and scores
            scoring = scorings[key].reuse(label)
        elif not claims:
            scoring = CallResult("scoring", label, STATUS_OK, 0, 0.0, parsed={
                "claims": [], "referenceItems": [{"item": it, "covered": False} for it in items],
                "rationale": "The answer makes no factual claim.", "correctness": 0, "completeness": 0})
            scorings[key] = scoring
        else:
            scoring = call_judge(call, SCORING_SYSTEM_PROMPT, build_scoring_prompt(item, claims, window),
                                 SCORING_SCHEMA, lambda raw, cs=claims: parse_scoring(raw, cs, known_ids, items),
                                 kind="scoring", label=label, retries=retries)
            scorings[key] = scoring
        record["calls"].append(scoring.to_dict())
        if scoring.status != STATUS_OK:
            return _fail(record, scoring, "claim scoring", started)
        scored = scoring.parsed
        verdicts = [c["verdict"] for c in scored["claims"]]
        citations = evaluate_citations(list(extract_citations(answer)), scored["claims"], item.turns, gold)
        correctness = scored["correctness"]
        completeness, completeness_source = derive_completeness(scored["referenceItems"], scored["completeness"])
        overrides = []
        if completeness != scored["completeness"]:
            overrides.append({"field": "completeness", "judge": scored["completeness"], "final": completeness,
                              "reason": "derived from reference-item coverage"})
        if item.dataset.has_reference and is_abstention(claims):
            for field_name, value in (("correctness", correctness), ("completeness", completeness)):
                if value:
                    overrides = [o for o in overrides if o["field"] != field_name]
                    overrides.append({"field": field_name, "judge": scored[field_name], "final": 0,
                                      "reason": "the answer only says the information is not available, "
                                                "but the reference gives an answer"})
            correctness = completeness = 0
        by_run[run] = {
            "correctness": correctness,
            "grounding": derive_grounding(verdicts),
            "completeness": completeness,
            "judgeScores": {"correctness": scored["correctness"], "completeness": scored["completeness"]},
            "completenessSource": completeness_source,
            "referenceItems": scored["referenceItems"],
            "overrides": overrides,
            "claims": scored["claims"],
            "unsupported_claims": [c["claim"] for c in scored["claims"] if c["verdict"] != VERDICT_SUPPORTED],
            "contradicted_claims": [c["claim"] for c in scored["claims"] if c["verdict"] == VERDICT_CONTRADICTED],
            "rationale": scored["rationale"],
            "citations": citations,
            "citationStatus": citation_status(citations),
            "citesGold": any(c["inGold"] for c in citations),
            "extractionCoverage": coverage[run],
        }
    record.update(status=STATUS_OK, byRun=by_run, sharedClaims=claim_keys[RUN_BASELINE] == claim_keys[RUN_CONTEXT])
    record["inconsistent"] = inconsistency(record)
    record["latencyMs"] = round((time.perf_counter() - started) * 1000, 1)
    return record


def run_judge(items: list[AlignedQuestion], call: JudgeCall, *, seed: int, retries: int,
              max_consecutive_failures: int = 3, window: int = DEFAULT_CONTEXT_WINDOW,
              progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    consecutive = 0
    for n, item in enumerate(items, 1):
        if max_consecutive_failures and consecutive >= max_consecutive_failures:
            skipped = base_record(item, assign_labels(item.qid, seed), window)
            skipped["error"] = "not run: stopped after repeated judge failures"
            records.append(skipped)
            continue
        record = judge_question(item, call, seed=seed, retries=retries, window=window)
        records.append(record)
        consecutive = 0 if record["status"] in (STATUS_OK, STATUS_INVALID_RESPONSE) else consecutive + 1
        if progress:
            progress(f"[{n}/{len(items)}] {item.qid} {record['status']} {record['latencyMs']:.0f} ms")
    return records


# ---- summary ---------------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(values: list[Any], target: Any) -> float | None:
    return sum(1 for v in values if v == target) / len(values) if values else None


def outcome(record: dict[str, Any]) -> str | None:
    """``context_win``, ``baseline_win`` or ``tie``: correctness first, then grounding (citations excluded)."""
    if record.get("status") != STATUS_OK:
        return None
    key = {run: (record["byRun"][run]["correctness"], record["byRun"][run]["grounding"]) for run in RUNS}
    if key[RUN_CONTEXT] > key[RUN_BASELINE]:
        return "context_win"
    if key[RUN_CONTEXT] < key[RUN_BASELINE]:
        return "baseline_win"
    return "tie"


def _citation_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    cited = [r for r in results if r["citations"]]
    entries = [c for r in results for c in r["citations"]]
    return {
        "answersWithCitations": len(cited),
        "citations": len(entries),
        "nonexistent": sum(1 for c in entries if not c["exists"]),
        "existingNotSupporting": sum(1 for c in entries if c["exists"] and not c["supports"]),
        "supporting": sum(1 for c in entries if c["exists"] and c["supports"]),
        "accurateRate": _rate([r["citationStatus"] for r in cited], CITATION_ACCURATE),
        "citesGoldRate": _rate([r["citesGold"] for r in cited], True),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r["status"] == STATUS_OK]
    runs: dict[str, Any] = {}
    for run in RUNS:
        results = [r["byRun"][run] for r in ok]
        scores = {dim: [x[dim] for x in results] for dim in DIMENSIONS}
        runs[run] = {
            "n": len(ok),
            "correctRate": _rate(scores["correctness"], 2),
            "partialRate": _rate(scores["correctness"], 1),
            "incorrectRate": _rate(scores["correctness"], 0),
            "meanCorrectness": _mean(scores["correctness"]),
            "meanGrounding": _mean(scores["grounding"]),
            "meanCompleteness": _mean(scores["completeness"]),
            "groundingPartialRate": _rate(scores["grounding"], 1),
            "completenessPartialRate": _rate(scores["completeness"], 1),
            "unsupportedRate": _rate(scores["grounding"], 0),
            "answersWithContradictedClaims": sum(1 for x in results if x["contradicted_claims"]),
            "citation": _citation_summary(results),
        }
    excluded = {r["qid"]: reason for r in ok if (reason := inconsistency(r))}
    outcomes = [outcome(r) for r in ok if r["qid"] not in excluded]
    raw_outcomes = [outcome(r) for r in ok]
    flagged = [r["qid"] for r in ok
               if any(r["byRun"][run][dim] < 2 for run in RUNS for dim in DIMENSIONS)
               or any(r["byRun"][run]["citationStatus"] not in (CITATION_NONE, CITATION_ACCURATE) for run in RUNS)
               or r["qid"] in excluded]
    consistent_pairs = [r for r in ok if r["equivalentAnswers"] or r["sharedClaims"]]
    inconsistent = [r["qid"] for r in consistent_pairs
                    if any(r["byRun"][RUN_BASELINE][d] != r["byRun"][RUN_CONTEXT][d] for d in DIMENSIONS)]
    return {
        "nQuestions": len(records),
        "nEvaluated": len(ok),
        "nFailed": len(records) - len(ok),
        "statusCounts": {s: sum(1 for r in records if r["status"] == s) for s in sorted({r["status"] for r in records})},
        "runs": runs,
        "outcomes": {k: outcomes.count(k) for k in ("context_win", "baseline_win", "tie")},
        "outcomesIncludingInconsistent": {k: raw_outcomes.count(k) for k in ("context_win", "baseline_win", "tie")},
        "inconsistentQuestions": excluded,
        "scoreOverrides": sorted({r["qid"] for r in ok for run in RUNS if r["byRun"][run].get("overrides")}),
        "extractionRepairs": sum(1 for r in ok for run in RUNS
                                 if (r["byRun"][run].get("extractionCoverage") or {}).get("repaired")),
        "extractionCoverageGaps": sorted({r["qid"] for r in ok for run in RUNS
                                          if (r["byRun"][run].get("extractionCoverage") or {}).get("missing")}),
        "extractionInventedClaims": sorted({r["qid"] for r in ok for run in RUNS
                                            if (r["byRun"][run].get("extractionCoverage") or {}).get("invented")}),
        "equivalentAnswerPairs": sum(1 for r in ok if r["equivalentAnswers"]),
        "sharedClaimPairs": sum(1 for r in ok if r["sharedClaims"]),
        "inconsistentEquivalentPairs": inconsistent,
        "flaggedQuestions": flagged,
    }


# ---- output ----------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _cell(text: Any, width: int = 70) -> str:
    value = " ".join(str(text if text is not None else "").split()) or "(empty)"
    if len(value) > width:
        value = value[: width - 1] + "…"
    return value.replace("|", "\\|")


def _scores(record: dict[str, Any], run: str) -> str:
    s = record["byRun"][run]
    return f"{s['correctness']}/{s['grounding']}/{s['completeness']}"


def _citation_cell(record: dict[str, Any], run: str) -> str:
    return record["byRun"][run]["citationStatus"].replace("_", " ")


def _citation_detail(citation: dict[str, Any]) -> str:
    if not citation["exists"]:
        verdict = "nonexistent"
    else:
        verdict = "supports" if citation["supports"] else "exists, does not support"
    gold = " (gold)" if citation["inGold"] else ""
    return f"[{citation['id']}]{gold} {verdict}: {_cell(citation['reason'], 160)}"


def render_markdown(payload: dict[str, Any]) -> str:
    meta, summary, records = payload["metadata"], payload["summary"], payload["questions"]
    runs = summary["runs"]
    b, c = runs[RUN_BASELINE], runs[RUN_CONTEXT]
    bc, cc = b["citation"], c["citation"]
    lines = [
        "# LoCoMo conv-26 pilot: LLM-judge answer evaluation",
        "",
        f"> **{DISCLAIMER}**",
        "",
        f"- Judge: `{meta['judge']['model']}` via local Ollama (`{meta['judge']['host']}`), "
        f"temperature {meta['judge']['options']['temperature']}, seed {meta['judge']['options']['seed']}, "
        f"format `{meta['judge']['formatMode']}`, judge version {meta['judgeVersion']}, "
        f"context window ±{meta['contextWindow']} turns",
        f"- Baseline report: `{meta['inputs']['baseline']['path']}`",
        f"- Context-enhanced report: `{meta['inputs']['context']['path']}`",
        f"- Dataset: `{meta['inputs']['dataset']['path']}` sample `{meta['sampleId']}`",
        f"- Created: {meta['createdAt']}",
        "- Method: answers are normalized in code (citations and wording about sources removed), rewritten as "
        "claims without the reference, and the claims are judged against the conversation and the reference. "
        "Grounding and citation accuracy are derived in code from the claim verdicts. Equal normalized answers "
        "or equal claim sets share one judgement. The run name is never shown to the judge.",
        "",
        "## Results",
        "",
        f"- Questions: {summary['nQuestions']} evaluated={summary['nEvaluated']} failed={summary['nFailed']}",
        "- Status counts: " + ", ".join(f"{k}={v}" for k, v in summary["statusCounts"].items()),
        f"- Answer pairs equal after normalization: {summary['equivalentAnswerPairs']}; "
        f"pairs with the same extracted claims: {summary['sharedClaimPairs']}"
        + (f"; inconsistent: {', '.join(summary['inconsistentEquivalentPairs'])}"
           if summary["inconsistentEquivalentPairs"] else ""),
        f"- Claim extractions repeated to recover a dropped name or title: {summary['extractionRepairs']}"
        + (f"; still incomplete: {', '.join(summary['extractionCoverageGaps'])}"
           if summary["extractionCoverageGaps"] else ""),
        *([f"- Extracted claims that still look invented (absent from the answer): "
           f"{', '.join(summary['extractionInventedClaims'])}"] if summary.get("extractionInventedClaims") else []),
        "",
        "### Factual quality (citations excluded)",
        "",
        "| Metric | Baseline | Context-enhanced |",
        "|---|---:|---:|",
        f"| Correct (correctness = 2) | {_pct(b['correctRate'])} | {_pct(c['correctRate'])} |",
        f"| Partially correct (correctness = 1) | {_pct(b['partialRate'])} | {_pct(c['partialRate'])} |",
        f"| Incorrect (correctness = 0) | {_pct(b['incorrectRate'])} | {_pct(c['incorrectRate'])} |",
        f"| Mean correctness (0-2) | {_num(b['meanCorrectness'])} | {_num(c['meanCorrectness'])} |",
        f"| Mean grounding (0-2) | {_num(b['meanGrounding'])} | {_num(c['meanGrounding'])} |",
        f"| Mean completeness (0-2) | {_num(b['meanCompleteness'])} | {_num(c['meanCompleteness'])} |",
        f"| Partial grounding (= 1) | {_pct(b['groundingPartialRate'])} | {_pct(c['groundingPartialRate'])} |",
        f"| Partial completeness (= 1) | {_pct(b['completenessPartialRate'])} | {_pct(c['completenessPartialRate'])} |",
        f"| Unsupported or contradicted (grounding = 0) | {_pct(b['unsupportedRate'])} | {_pct(c['unsupportedRate'])} |",
        f"| Answers with a contradicted claim | {b['answersWithContradictedClaims']} | {c['answersWithContradictedClaims']} |",
        "",
        "Per-question outcome (correctness first, then grounding; citations excluded): "
        f"context-enhanced wins {summary['outcomes']['context_win']}, "
        f"baseline wins {summary['outcomes']['baseline_win']}, ties {summary['outcomes']['tie']}"
        + (f"; {len(summary['inconsistentQuestions'])} question(s) with inconsistent judge scores excluded "
           f"({', '.join(summary['inconsistentQuestions'])}; including them: "
           f"context-enhanced wins {summary['outcomesIncludingInconsistent']['context_win']}, "
           f"baseline wins {summary['outcomesIncludingInconsistent']['baseline_win']}, "
           f"ties {summary['outcomesIncludingInconsistent']['tie']})"
           if summary.get("inconsistentQuestions") else "") + ".",
        *([f"- Scores changed in code (derived completeness or the abstention rule): "
           f"{', '.join(summary['scoreOverrides'])}"] if summary.get("scoreOverrides") else []),
        "",
        "### Citation accuracy (separate from factual quality)",
        "",
        "| Metric | Baseline | Context-enhanced |",
        "|---|---:|---:|",
        f"| Answers with citations | {bc['answersWithCitations']} | {cc['answersWithCitations']} |",
        f"| Citations | {bc['citations']} | {cc['citations']} |",
        f"| Supporting citations | {bc['supporting']} | {cc['supporting']} |",
        f"| Existing turns that do not support the claims | {bc['existingNotSupporting']} | {cc['existingNotSupporting']} |",
        f"| Nonexistent turns | {bc['nonexistent']} | {cc['nonexistent']} |",
        f"| Cited answers with all citations accurate | {_pct(bc['accurateRate'])} | {_pct(cc['accurateRate'])} |",
        f"| Cited answers citing a gold evidence turn | {_pct(bc['citesGoldRate'])} | {_pct(cc['citesGoldRate'])} |",
        "",
        "## Per question",
        "",
        "Scores are correctness/grounding/completeness; citation accuracy is shown separately.",
        "",
        "| Question | Cat | Reference | Baseline | Baseline citations | Context | Context citations | Outcome | Status |",
        "|---|---|---|---:|---|---:|---|---|---|",
    ]
    for r in records:
        ok = r["status"] == STATUS_OK
        lines.append(
            f"| {r['qid']} | {_cell(r['category'], 4)} | {_cell(r['reference'], 50)} | "
            f"{_scores(r, RUN_BASELINE) if ok else '-'} | {_citation_cell(r, RUN_BASELINE) if ok else '-'} | "
            f"{_scores(r, RUN_CONTEXT) if ok else '-'} | {_citation_cell(r, RUN_CONTEXT) if ok else '-'} | "
            f"{(outcome(r) or '-').replace('_', ' ')}{' (excluded: inconsistent)' if r.get('inconsistent') else ''} | "
            f"{r['status']} |"
        )
    lines += ["", "## Questions with a factual or citation issue", ""]
    flagged = [r for r in records if r["qid"] in set(summary["flaggedQuestions"])]
    if not flagged:
        lines.append("None.")
    for r in flagged:
        lines += [f"### {r['qid']}: {_cell(r['question'], 120)}", "", f"- Reference: {_cell(r['reference'], 200)}"]
        if r["equivalentAnswers"]:
            lines.append("- The two answers are equal after removing citations and wording about sources.")
        elif r["sharedClaims"]:
            lines.append("- The two answers make the same extracted claims.")
        if r.get("inconsistent"):
            lines.append(f"- Left out of the headline outcome, inconsistent judge scores: {r['inconsistent']}")
        for run, name in ((RUN_BASELINE, "Baseline"), (RUN_CONTEXT, "Context-enhanced")):
            s = r["byRun"][run]
            problems = [label for dim, label in (("correctness", "incorrect"), ("grounding", "unsupported"),
                                                 ("completeness", "incomplete")) if s[dim] == 0]
            problems += [f"partial {dim}" for dim in DIMENSIONS if s[dim] == 1]
            lines.append(f"- {name} ({_scores(r, run)}; {', '.join(problems) or 'no factual issues'}; "
                         f"citations: {_citation_cell(r, run)}): {_cell(r['answers'][run], 200)}")
            for claim in s["claims"]:
                ids = f"; source {', '.join(claim['sourceIds'])}" if claim["sourceIds"] else ""
                ids += f"; contradicted by {', '.join(claim['contradictingIds'])}" if claim["contradictingIds"] else ""
                lines.append(f"  - Claim ({claim['verdict'].replace('_', ' ')}{ids}): {_cell(claim['claim'], 160)}")
            cov = s.get("extractionCoverage") or {}
            if cov.get("repaired") and cov.get("initiallyMissing"):
                lines.append(f"  - Claim extraction repeated because it left out: {', '.join(cov['initiallyMissing'])}")
            if cov.get("repaired") and cov.get("initiallyInvented"):
                lines.append(f"  - Claim extraction repeated because it invented: {'; '.join(cov['initiallyInvented'])}")
            if cov.get("missing"):
                lines.append(f"  - Claim extraction still leaves out: {', '.join(cov['missing'])}")
            if cov.get("invented"):
                lines.append(f"  - Claim extraction still contains claims not in the answer: {'; '.join(cov['invented'])}")
            if len(s.get("referenceItems") or []) > 1:
                lines.append("  - Reference items: " + "; ".join(
                    f"{i['item']} ({'covered' if i['covered'] else 'not covered'})" for i in s["referenceItems"]))
            for o in s.get("overrides") or []:
                lines.append(f"  - {o['field'].capitalize()} set to {o['final']} in code (judge gave {o['judge']}): "
                             f"{o['reason']}")
            if s["rationale"]:
                lines.append(f"  - Rationale: {_cell(s['rationale'], 300)}")
            for citation in s["citations"]:
                lines.append(f"  - Citation {_citation_detail(citation)}")
        lines.append("")
    failures = [r for r in records if r["status"] != STATUS_OK]
    lines += ["## Failures", ""]
    if not failures:
        lines.append("None.")
    for r in failures:
        lines.append(f"- `{r['qid']}` {r['status']}: {_cell(r['error'], 200)}")
    warned = [r for r in records if r["warnings"]]
    if warned:
        lines += ["", "## Alignment warnings", ""]
        lines += [f"- `{r['qid']}`: {'; '.join(r['warnings'])}" for r in warned]
    lines += ["", "## Scoring rubric", ""]
    for dim, levels in RUBRIC.items():
        lines.append(f"- **{dim[0].upper() + dim[1:]}**: " + "; ".join(f"{k} = {v}" for k, v in levels.items()))
    lines += [
        "",
        "Claim verdicts: supported (the conversation states it), contradicted (the conversation states something "
        "incompatible), not mentioned (neither). Correct means correctness = 2; partial credit means a score of 1. "
        "Relative time expressions are accepted when they resolve, against the session date of the source turn, "
        "to the reference.",
        "",
        "## Limitations",
        "",
        *[f"- {text}" for text in LIMITATIONS],
        "",
        f"_{DISCLAIMER}_",
        "",
    ]
    return "\n".join(lines)


def build_payload(records: list[dict[str, Any]], *, client: OllamaJudgeClient, seed: int, retries: int,
                  window: int, dataset: Dataset, paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "metadata": {
            "disclaimer": DISCLAIMER,
            "limitations": list(LIMITATIONS),
            "judgeVersion": JUDGE_VERSION,
            "createdAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "sampleId": dataset.sample_id,
            "judge": {"provider": "ollama", "model": client.model, "host": client.host,
                      "options": client.options(), "formatMode": client.format_mode,
                      "timeoutS": client.timeout, "retries": retries},
            "contextWindow": window,
            "labelSeed": seed,
            "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
            "rubric": {dim: {str(k): v for k, v in levels.items()} for dim, levels in RUBRIC.items()},
            "systemPrompts": {"extraction": EXTRACTION_SYSTEM_PROMPT, "scoring": SCORING_SYSTEM_PROMPT},
        },
        "summary": summarize(records),
        "questions": records,
    }


def write_outputs(payload: dict[str, Any], json_path: Path, md_path: Path) -> None:
    for path, text in ((json_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n"),
                       (md_path, render_markdown(payload))):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)


# ---- CLI -------------------------------------------------------------------------------------


def load_reports(baseline: Path, context: Path) -> tuple[ParsedReport, ParsedReport]:
    parsed = []
    for name, path in ((RUN_BASELINE, baseline), (RUN_CONTEXT, context)):
        try:
            parsed.append(parse_report(path.read_text(encoding="utf-8")))
        except OSError as exc:
            raise InputError(f"cannot read {name} report {path}: {exc}") from exc
    return parsed[0], parsed[1]


def select_questions(items: list[AlignedQuestion], spec: str | None) -> list[AlignedQuestion]:
    if not spec:
        return items
    wanted = [q.strip() for q in spec.split(",") if q.strip()]
    known = {i.qid for i in items}
    unknown = [q for q in wanted if q not in known]
    if unknown:
        raise InputError(f"--questions names questions not in the reports: {', '.join(unknown)}")
    return [i for i in items if i.qid in set(wanted)]


def main(argv: list[str] | None = None, *, call: JudgeCall | None = None) -> int:
    load_root_env()
    parser = argparse.ArgumentParser(prog="python scripts/judge_locomo_answers.py",
                                     description="Exploratory local LLM-judge for two LoCoMo reports.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-id", help="LoCoMo sample_id when the dataset has several records")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--questions", help="comma-separated question ids to judge (default: all)")
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW,
                        help="neighbouring turns shown on each side of gold and cited turns (default 3)")
    parser.add_argument("--model", default=os.environ.get(JUDGE_MODEL_ENV) or DEFAULT_MODEL,
                        help=f"Ollama judge model (default: ${JUDGE_MODEL_ENV} or {DEFAULT_MODEL})")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST") or DEFAULT_HOST)
    parser.add_argument("--allow-remote-host", action="store_true",
                        help="allow a non-local Ollama host (data would leave this machine)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds per request")
    parser.add_argument("--retries", type=int, default=1, help="retries after a timeout or request error")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    parser.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="judge seed and A/B order seed")
    parser.add_argument("--format-mode", choices=("schema", "json"), default="schema",
                        help="Ollama structured output: JSON schema (default) or plain JSON mode")
    parser.add_argument("--max-consecutive-failures", type=int, default=3,
                        help="stop calling the judge after this many consecutive timeouts/request errors (0 = never)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    parser.add_argument("--dry-run", action="store_true", help="align inputs and print prompts; no model calls")
    args = parser.parse_args(argv)
    if (args.timeout <= 0 or args.retries < 0 or args.num_ctx < 512 or args.num_predict < 64
            or args.context_window < 0):
        parser.error("invalid --timeout/--retries/--num-ctx/--num-predict/--context-window")

    try:
        dataset = load_dataset(args.dataset, args.sample_id)
        baseline, context = load_reports(args.baseline, args.context)
        items = select_questions(align(dataset, baseline, context), args.questions)
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        first = items[0]
        print(f"aligned {len(items)} questions ({items[0].qid}..{items[-1].qid}); no model calls made.")
        for item in items:
            if item.warnings:
                print(f"warning {item.qid}: {'; '.join(item.warnings)}")
        labels = assign_labels(first.qid, args.seed)
        print(f"\n--- extraction system prompt ---\n{EXTRACTION_SYSTEM_PROMPT}")
        for label in LABELS:
            answer = first.answers[labels[label]]
            print(f"\n--- extraction prompt for {first.qid} answer {label} ---\n"
                  f"{build_extraction_prompt(first, answer, args.context_window)}")
        print(f"\n--- scoring system prompt ---\n{SCORING_SYSTEM_PROMPT}")
        print(f"\n--- scoring prompt for {first.qid} (claims shown as placeholders) ---\n"
              f"{build_scoring_prompt(first, ['<claims from extraction>'], args.context_window)}")
        return 0

    if not args.overwrite:
        existing = [p for p in (args.output_json, args.output_md) if p.exists()]
        if existing:
            print(f"error: output exists ({', '.join(map(str, existing))}); pass --overwrite to replace",
                  file=sys.stderr)
            return 2
    if call is None and not is_local_host(args.host) and not args.allow_remote_host:
        print(f"error: judge host {args.host} is not local; pass --allow-remote-host to send data there",
              file=sys.stderr)
        return 2

    client = OllamaJudgeClient(args.model, args.host, timeout=args.timeout, seed=args.seed, num_ctx=args.num_ctx,
                               num_predict=args.num_predict, format_mode=args.format_mode)
    records = run_judge(items, call or client, seed=args.seed, retries=args.retries,
                        max_consecutive_failures=args.max_consecutive_failures, window=args.context_window,
                        progress=lambda line: print(line, file=sys.stderr, flush=True))
    payload = build_payload(records, client=client, seed=args.seed, retries=args.retries,
                            window=args.context_window, dataset=dataset,
                            paths={"dataset": args.dataset, "baseline": args.baseline, "context": args.context})
    summary = payload["summary"]
    if summary["nEvaluated"] == 0:
        print("error: no question was judged successfully; nothing written. First error: "
              f"{records[0]['error'] if records else 'no questions'}", file=sys.stderr)
        return 1
    write_outputs(payload, args.output_json, args.output_md)
    print(f"wrote {args.output_json} and {args.output_md}: evaluated={summary['nEvaluated']} "
          f"failed={summary['nFailed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
