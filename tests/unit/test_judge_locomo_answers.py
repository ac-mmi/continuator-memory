"""Offline LLM-judge script: dataset parsing, alignment, answer normalization, claim extraction and
scoring, derived grounding and citation accuracy, conversation context, and outputs.

Ollama is never called: the judge is a scripted fake, and the HTTP client is tested with a patched
``urlopen``. The audited cases (Q-007, Q-014, Q-017, Q-024, Q-030) use the real conv-26 turns and
the real pilot answers as fixtures; the evaluator itself has no question-specific logic.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import socket
import sys
import urllib.error
from datetime import date
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "judge_locomo_answers.py"
_spec = importlib.util.spec_from_file_location("judge_locomo_answers", _SCRIPT)
judge = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = judge
_spec.loader.exec_module(judge)


@pytest.fixture(autouse=True)
def no_env_file(monkeypatch):
    """Keep the repo .env out of the test process."""
    monkeypatch.setattr(judge, "load_root_env", lambda: ())


# ---- real conv-26 turns and pilot answers ------------------------------------------------------

SESSIONS = {
    1: "1:56 pm on 8 May, 2023",
    2: "1:14 pm on 25 May, 2023",
    4: "10:37 am on 27 June, 2023",
    5: "1:36 pm on 3 July, 2023",
    6: "8:18 pm on 6 July, 2023",
    7: "4:33 pm on 12 July, 2023",
    8: "1:51 pm on 15 July, 2023",
    14: "1:33 pm on 25 August, 2023",
}
# In session order; positions within a session follow this list.
TURNS = [
    ("D1:11", "Caroline", "I'm keen on counseling or working in mental health - I'd love to support those with similar issues.", None),
    ("D2:7", "Melanie", "Thanks, Caroline. It's still a work in progress, but I'm doing my best. My kids are so excited about summer break! We're thinking about going camping next month. Any fun plans for the summer?", None),
    ("D4:11", "Caroline", "Lately, I've been looking into counseling and mental health as a career. I want to help people who have gone through the same things as me.", "a photo of a book shelf with many books on it"),
    ("D4:12", "Melanie", "Sounds great! What kind of counseling and mental health services do you want to persue?", None),
    ("D4:13", "Caroline", "I'm still figuring out the details, but I'm thinking of working with trans people, helping them accept themselves and supporting their mental health.", None),
    ("D5:4", "Melanie", "Wow, Caroline! That's great! I just signed up for a pottery class yesterday. It's like therapy for me, letting me express myself and get creative.", "a photo of a person holding a frisbee in their hand"),
    ("D6:10", "Melanie", "I loved reading \"Charlotte's Web\" as a kid. It was so cool seeing how friendship and compassion can make a difference.", None),
    ("D7:8", "Melanie", "Caroline, so glad you got the support! This book I read last year reminds me to always pursue my dreams, just like you are doing!", "a photography of a book cover with a gold coin on it"),
    ("D7:9", "Caroline", "Thanks so much, Mel! Seeing this pic just made me appreciate my love of reading even more.", None),
    ("D7:10", "Melanie", "Wow, Caroline! Books have such an awesome power! Which one has been your favorite guide?", None),
    ("D7:11", "Caroline", "I loved \"Becoming Nicole\" by Amy Ellis Nutt. It's a real inspiring true story about a trans girl and her family. Highly recommend it for sure!", None),
    ("D7:12", "Melanie", "That sounds awesome! What did you take away from it to use in your life?", None),
    ("D8:2", "Melanie", "Hey Caroline, it's been super busy here. So much since we talked! Last Fri I finally took my kids to a pottery workshop. We all made our own pots, it was fun and therapeutic!", None),
    ("D14:3", "Caroline", "Thanks, Melanie! That plate is awesome! Did you make it?", None),
    ("D14:4", "Melanie", "Yeah, I made it in pottery class yesterday. I love it! Pottery's so relaxing and creative. Have you tried it yet?", None),
]

AUDIT = {
    "Q-007": ("When is Melanie planning on going camping?", "June 2023", ["D2:7"],
              "Melanie is thinking about going camping next month. [D2:7]",
              "Melanie is thinking about going camping next month."),
    "Q-014": ("What career path has Caroline decided to persue?", "counseling or mental health for Transgender people",
              ["D4:13", "D1:11"],
              "Caroline is looking into counseling and mental health as a career. [D4:11]",
              "Caroline is keen on counseling or working in mental health. [D1:11]"),
    "Q-017": ("When did Melanie sign up for a pottery class?", "2 July 2023", ["D5:4"],
              "Melanie signed up for a pottery class yesterday. [D5:4]",
              "Melanie signed up for a pottery class yesterday according to the conversation [D14:4]."),
    "Q-024": ("What books has Melanie read?", "\"Nothing is Impossible\", \"Charlotte's Web\"", ["D7:8", "D6:10"],
              "Melanie mentioned reading a book last year that reminded her to always pursue her dreams [D7:8]. "
              "She also recalled loving \"Charlotte's Web\" as a kid [D6:10].",
              "Melanie mentioned reading a book recommended by Caroline last year, which reminded her to always "
              "pursue her dreams [D7:8]. However, no other specific books read by Melanie are cited in the provided "
              "excerpts."),
    "Q-030": ("When did Melanie go to the pottery workshop?", "The Friday before 15 July 2023", ["D8:2"],
              "Last Fri I finally took my kids to a pottery workshop.\n[D8:2]",
              "Last Fri, Melanie took her kids to a pottery workshop.[D8:2]"),
}


def conversation() -> dict:
    conv: dict = {"speaker_a": "Caroline", "speaker_b": "Melanie"}
    for session, when in SESSIONS.items():
        conv[f"session_{session}_date_time"] = when
        conv[f"session_{session}"] = []
    for dia_id, speaker, text, caption in TURNS:
        session = int(dia_id[1:].split(":")[0])
        turn = {"speaker": speaker, "dia_id": dia_id, "text": text}
        if caption:
            turn["blip_caption"] = caption
        conv[f"session_{session}"].append(turn)
    return conv


TURN_INDEX = judge._conversation_turns(conversation())


def audit_item(qid: str, *, baseline: str | None = None, context: str | None = None) -> judge.AlignedQuestion:
    question, reference, evidence, base_answer, ctx_answer = AUDIT[qid]
    dq = judge.DatasetQuestion(int(qid[2:]) - 1, question, reference, 2, tuple(evidence))
    answers = {"baseline": base_answer if baseline is None else baseline,
               "context": ctx_answer if context is None else context}
    return judge.AlignedQuestion(qid, dq, answers, [TURN_INDEX[e] for e in evidence], [], [], TURN_INDEX)


# ---- scripted judge ---------------------------------------------------------------------------

_EXTRACT_ANSWER = re.compile(r"\nAnswer:\n((?:>.*\n?)+)")
_CLAIM_LINE = re.compile(r"^(\d+)\. (.*)$", re.M)
_GOLD_LINE = re.compile(r"\[(D\d+:\d+)\] \(gold evidence\)")
_ITEM_LINE = re.compile(r"^R\d+\. (.*)$", re.M)


class FakeJudge:
    """``claims`` maps a normalized answer to its claims (default: the answer itself as one claim); a
    tuple of lists gives successive extraction responses (first call, repair call).
    ``verdicts`` maps a claim to (verdict, source ids[, contradicting ids]); default: supported, with the
    gold turns as sources. ``scores`` maps a tuple of claims to (correctness, completeness); default (2, 2).
    ``coverage`` maps a tuple of claims to one bool per reference item; default: every item covered for
    completeness 2, the first only for 1, none for 0."""

    def __init__(self, claims=None, verdicts=None, scores=None, extraction_raw=None, scoring_raw=None,
                 coverage=None):
        self.claims = claims or {}
        self.verdicts = verdicts or {}
        self.scores = scores or {}
        self.coverage = coverage or {}
        self.extraction_raw = extraction_raw
        self.scoring_raw = scoring_raw
        self.calls: list[tuple[str, str]] = []
        self.prompts: list[tuple[str, str]] = []
        self._extracted: dict[str, int] = {}

    def __call__(self, system, prompt, schema):
        if schema is judge.EXTRACTION_SCHEMA:
            assert "Reference answer" not in prompt
            answer = "\n".join(line[2:] for line in _EXTRACT_ANSWER.search(prompt).group(1).strip().splitlines())
            kind = "repair" if "A previous list of claims" in prompt else "extraction"
            self.calls.append((kind, answer))
            self.prompts.append((kind, prompt))
            if self.extraction_raw is not None:
                return self.extraction_raw
            planned = self.claims.get(answer, [answer])
            if isinstance(planned, tuple):
                n = self._extracted.get(answer, 0)
                self._extracted[answer] = n + 1
                planned = planned[min(n, len(planned) - 1)]
            return json.dumps({"claims": planned})
        assert schema is judge.SCORING_SCHEMA
        claims = [m.group(2) for m in _CLAIM_LINE.finditer(prompt.split("Claims made by the answer:")[1])]
        self.calls.append(("scoring", " | ".join(claims)))
        self.prompts.append(("scoring", prompt))
        if self.scoring_raw is not None:
            return self.scoring_raw
        gold = _GOLD_LINE.findall(prompt)
        entries = []
        for claim in claims:
            verdict, source, *against = self.verdicts.get(claim, ("supported", gold))
            entries.append({"claim": claim, "verdict": verdict, "source_ids": list(source),
                            "contradicting_ids": list(against[0]) if against else []})
        correctness, completeness = self.scores.get(tuple(claims), (2, 2))
        items = _ITEM_LINE.findall(prompt.split("Reference items:")[1].split("\n\n")[0])
        covered = self.coverage.get(tuple(claims),
                                    [completeness == 2 or (completeness == 1 and i == 0) for i in range(len(items))])
        return json.dumps({"claims": entries, "reference_items": [{"item": it, "covered": c}
                                                                  for it, c in zip(items, covered)],
                           "rationale": "r", "correctness": correctness, "completeness": completeness})


def kinds(fake: FakeJudge) -> list[str]:
    return [kind for kind, _ in fake.calls]


# ---- normalization -----------------------------------------------------------------------------


def test_extract_and_strip_citations():
    assert judge.extract_citations("a [D1:3, D2:4] b [D1:3]") == ("D1:3", "D2:4")
    assert judge.extract_citations("pots.\n[D8:2]") == ("D8:2",)
    assert judge.extract_citations("memory mem:D1:3:c0 and [not an id]") == ()
    assert judge.strip_citations(AUDIT["Q-030"][3]) == "Last Fri I finally took my kids to a pottery workshop."


def test_normalize_answer_removes_meta_wording_but_keeps_facts():
    assert judge.normalize_answer(AUDIT["Q-017"][4]) == "Melanie signed up for a pottery class yesterday."
    assert judge.normalize_answer("Based on the provided excerpts, she went yesterday.") == "she went yesterday."
    parenthetical = judge.normalize_answer("She went, as mentioned in the conversation, on Friday.")
    assert judge.normalize_text(parenthetical) == "she went on friday"  # a leftover comma is harmless
    assert judge.normalize_answer("She went according to her statement [D1:3].") == "She went."
    assert judge.normalize_answer("She read it, according to Caroline.") == "She read it, according to Caroline."


def test_semantic_equivalence_ignores_citations_and_meta_wording_only():
    assert judge.equivalent_answers(AUDIT["Q-007"][3], AUDIT["Q-007"][4])
    assert judge.equivalent_answers(AUDIT["Q-017"][3], AUDIT["Q-017"][4])
    assert not judge.equivalent_answers(AUDIT["Q-030"][3], AUDIT["Q-030"][4])
    assert not judge.equivalent_answers(AUDIT["Q-024"][3], AUDIT["Q-024"][4])


def test_claims_key_ignores_order_case_and_punctuation():
    assert judge.claims_key(["Melanie went.", "It was June."]) == judge.claims_key(["it was june", "MELANIE WENT"])
    assert judge.claims_key(["", " "]) == ()


@pytest.mark.parametrize("verdicts, expected", [
    ([], 0),
    (["supported"], 2),
    (["supported", "supported"], 2),
    (["supported", "not_mentioned"], 1),
    (["not_mentioned"], 0),
    (["supported", "contradicted"], 0),
])
def test_derive_grounding(verdicts, expected):
    assert judge.derive_grounding(verdicts) == expected


def _claim(text, verdict, source=(), against=()):
    return {"claim": text, "verdict": verdict, "sourceIds": list(source), "contradictingIds": list(against),
            "unknownIds": []}


def test_evaluate_citations_separates_nonexistent_unsupporting_and_supporting():
    claims = [_claim("Melanie signed up on 2 July 2023.", "supported", ["D5:4"]),
              _claim("Caroline recommended it.", "contradicted", [], ["D7:11"])]
    citations = judge.evaluate_citations(["D5:4", "D14:4", "D99:1"], claims, TURN_INDEX, {"D5:4"})
    assert [(c["id"], c["exists"], c["inGold"], c["supports"]) for c in citations] == [
        ("D5:4", True, True, True), ("D14:4", True, False, False), ("D99:1", False, False, False)]
    assert "not the source of any" in citations[1]["reason"]
    assert "does not exist" in citations[2]["reason"]
    assert judge.citation_status(citations) == judge.CITATION_MIXED
    assert judge.citation_status(citations[1:]) == judge.CITATION_INACCURATE
    assert judge.citation_status(citations[:1]) == judge.CITATION_ACCURATE
    assert judge.citation_status([]) == judge.CITATION_NONE


def test_citation_accuracy_does_not_depend_on_the_claim_verdict():
    """Regression: a claim judged wrong (here: contradicted over a date) still has its source turn, so a
    citation of that turn is accurate; a citation of an unrelated turn is not."""
    claims = [_claim("Melanie signed up for a pottery class on 2 July 2023.", "contradicted", ["D5:4"])]
    right, wrong = judge.evaluate_citations(["D5:4", "D14:4"], claims, TURN_INDEX, {"D5:4"})
    assert right["supports"] is True and "source of" in right["reason"]
    assert wrong["supports"] is False


# ---- relative dates ------------------------------------------------------------------------------


@pytest.mark.parametrize("text, day, expected", [
    ("I just signed up for a pottery class yesterday.", date(2023, 7, 3), [("yesterday", "2 July 2023")]),
    ("Last Fri I finally took my kids to a pottery workshop.", date(2023, 7, 15),
     [("Last Fri", "Friday 14 July 2023")]),
    ("We're thinking about going camping next month.", date(2023, 5, 25), [("next month", "June 2023")]),
    ("next month", date(2023, 12, 15), [("next month", "January 2024")]),
    ("last month", date(2023, 1, 10), [("last month", "December 2022")]),
    ("We went last Friday.", date(2023, 7, 14), [("last Friday", "Friday 7 July 2023")]),
    ("This book I read last year", date(2023, 7, 12), [("last year", "2022")]),
    ("the day before yesterday, and yesterday too", date(2023, 7, 3),
     [("the day before yesterday", "1 July 2023"), ("yesterday", "2 July 2023")]),
    ("It was last weekend, not last week.", date(2023, 7, 12),
     [("last weekend", "the weekend of 8 July 2023 to 9 July 2023"), ("last week", "the week before 12 July 2023")]),
    ("two weeks ago and 3 days ago", date(2023, 7, 12),
     [("two weeks ago", "around 28 June 2023"), ("3 days ago", "9 July 2023")]),
    ("Nothing relative here.", date(2023, 7, 12), []),
])
def test_resolve_relative_expressions(text, day, expected):
    assert judge.resolve_relative_expressions(text, day) == expected


def test_only_weekday_expressions_carry_a_weekday():
    """Regression: plain dates carry no weekday, so the judge is not invited to recompute one."""
    for phrase, value in judge.resolve_relative_expressions("yesterday, next month, last year, tomorrow",
                                                            date(2023, 7, 3)):
        assert not any(day in value for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                                                "Saturday", "Sunday")), (phrase, value)
    assert judge.format_day(date(2023, 7, 2)) == "Sunday 2 July 2023"  # the calendar itself is right


@pytest.mark.parametrize("text, expected", [
    ("Melanie signed up on Sunday 2 July 2023.", "Melanie signed up on 2 July 2023."),
    ("On Saturday, 2 July 2023 she went.", "On 2 July 2023 she went."),
    ("It was Friday, July 14, 2023.", "It was July 14, 2023."),
    ("She goes on Friday.", "She goes on Friday."),
])
def test_drop_weekdays_before_dates(text, expected):
    assert judge.drop_weekdays_before_dates(text) == expected


# ---- conversation context ----------------------------------------------------------------------


def test_context_window_includes_surrounding_turns_of_gold_and_cited_turns():
    ids = [t.dia_id for t in judge.context_turns(audit_item("Q-024"), 3)]
    assert ids == ["D6:10", "D7:8", "D7:9", "D7:10", "D7:11"]  # D7:12 is four turns after D7:8
    narrow = [t.dia_id for t in judge.context_turns(audit_item("Q-024"), 1)]
    assert "D7:11" not in narrow and "D7:9" in narrow
    q017 = [t.dia_id for t in judge.context_turns(audit_item("Q-017"), 1)]
    assert q017 == ["D5:4", "D14:3", "D14:4"]  # D14:4 is cited by an answer, so it is shown with context


def _prompt(*excerpts: tuple[str, str]) -> str:
    """A generator prompt as answering._render_prompt writes it."""
    blocks = [f"[{mid}] (message date: 1 May 2023) {text}\nlinked memory mem:{mid}:c0" for mid, text in excerpts]
    return "Source excerpts. Cite using only the square-bracket message id.\n\n" + "\n\n".join(blocks) + \
        "\n\nQuestion: q\nAnswer:"


def test_prompt_excerpt_ids_are_the_excerpt_headers_in_order_without_duplicates():
    prompt = _prompt(("D7:10", "Melanie: Books!\n[image: a shelf]"), ("D2:7", "Melanie: camping"),
                     ("D7:10", "repeat"))
    assert judge.prompt_excerpt_ids(prompt) == ("D7:10", "D2:7")
    assert judge.prompt_excerpt_ids(None) == () and judge.prompt_excerpt_ids("no excerpts") == ()


def test_context_adds_prompt_turns_of_both_runs_once_and_without_neighbours():
    item = audit_item("Q-007")  # gold D2:7 is alone in its session
    item.prompt_ids = ("D7:10", "D2:7", "D14:4")  # D2:7 is also gold
    ids = [t.dia_id for t in judge.context_turns(item, 3)]
    assert ids == ["D2:7", "D7:10", "D14:4"]  # no D7:9/D7:11 or D14:3 neighbours, no duplicate D2:7
    fake = FakeJudge()
    record = judge.judge_question(item, fake, seed=0)
    assert record["contextTurnIds"] == ids and record["promptTurnIds"] == ["D7:10", "D2:7", "D14:4"]
    for kind, prompt in fake.prompts:  # extraction and scoring for both answers see the same turns
        assert all(prompt.count(f"[{i}]") == 1 for i in ids), kind


def test_align_takes_the_union_of_both_reports_prompt_turns(files):
    data = judge.load_dataset(files["dataset"])
    q = AUDIT["Q-024"]
    base = report(section("Q-024", q[0], q[1], q[3], _prompt(("D7:8", "a"), ("D6:10", "b"))))
    ctx = report(section("Q-024", q[0], q[1], q[4], _prompt(("D6:10", "b"), ("D7:12", "c"), ("D99:1", "x"))))
    item, = judge.align(data, judge.parse_report(base), judge.parse_report(ctx))
    assert item.prompt_ids == ("D7:8", "D6:10", "D7:12")
    assert any("D99:1" in w for w in item.warnings)
    without, = judge.align(data, judge.parse_report(report(section("Q-024", q[0], q[1], q[3]))),
                           judge.parse_report(ctx))
    assert without.prompt_ids == ("D6:10", "D7:12")
    assert any("baseline report has no generator prompt" in w for w in without.warnings)


def test_rendered_context_resolves_relative_times_against_each_turns_own_session():
    """Regression: "yesterday" in D5:4 resolves against session 5 (3 July 2023) and "yesterday" in D14:4
    against session 14 (25 August 2023); no weekday is generated for either."""
    text = "\n".join(judge.render_context(audit_item("Q-017"), 1))
    assert "Session 5, 3 July 2023 (1:36 pm on 3 July, 2023):" in text
    d54 = text.index("[D5:4] (gold evidence) Melanie: Wow, Caroline!")
    assert text.index('(said on 3 July 2023: "yesterday" = 2 July 2023)') > d54
    d144 = text.index("[D14:4] Melanie: Yeah, I made it in pottery class yesterday.")
    assert text.index('(said on 25 August 2023: "yesterday" = 24 August 2023)') > d144
    assert "Saturday" not in text and "Sunday" not in text
    assert "Relative times in this session" not in text  # no generic per-session tables


# ---- prompts --------------------------------------------------------------------------------------


def test_extraction_prompt_has_no_reference_run_name_or_citation():
    item = audit_item("Q-017")
    prompts = [judge.build_extraction_prompt(item, item.answers[run]) for run in judge.RUNS]
    assert prompts[0] == prompts[1]  # semantically equal answers give the same prompt
    assert "Reference answer" not in prompts[0] and "2 July 2023" not in prompts[0].split("Answer:")[1]
    assert "> Melanie signed up for a pottery class yesterday." in prompts[0]
    assert "according to the conversation" not in prompts[0]
    assert "baseline" not in prompts[0].lower() and "context-enhanced" not in prompts[0].lower()


def test_scoring_prompt_shows_reference_claims_and_contradicting_context():
    item = audit_item("Q-024")
    prompt = judge.build_scoring_prompt(item, ["Melanie read a book recommended by Caroline last year."])
    assert "Reference answer: \"Nothing is Impossible\", \"Charlotte's Web\"" in prompt
    assert "1. Melanie read a book recommended by Caroline last year." in prompt
    assert "[D7:8] (gold evidence) Melanie: Caroline, so glad you got the support!" in prompt
    assert '(said on 12 July 2023: "last year" = 2022)' in prompt
    assert "[D7:11] Caroline: I loved \"Becoming Nicole\" by Amy Ellis Nutt." in prompt
    assert "[D6:10] (gold evidence) Melanie: I loved reading \"Charlotte's Web\"" in prompt
    rules = " ".join(judge.SCORING_SYSTEM_PROMPT.split())
    assert "When the reference lists several items, completeness is 2 only if all are given" in rules
    assert "is contradicted when the excerpts do mention it" in rules
    assert "even when the claim gets a detail such as the date or the person wrong" in rules


# ---- response validation ---------------------------------------------------------------------------


def test_parse_extraction():
    assert judge.parse_extraction('{"claims": ["a", " a ", "", "b"]}') == ["a", "b"]
    assert judge.parse_extraction('```json\n{"claims": []}\n```') == []
    for raw, message in (("nope", "not valid JSON"), ('{"claims": "a"}', "list of strings"), ("[]", "not an object")):
        with pytest.raises(judge.JudgeResponseError, match=message):
            judge.parse_extraction(raw)


def _scoring(entries, c=2, m=2):
    return json.dumps({"claims": entries, "rationale": "r", "correctness": c, "completeness": m})


def test_parse_scoring_keeps_sources_for_every_verdict():
    claims = ["Melanie signed up on 2 July 2023.", "Caroline recommended the book."]
    raw = _scoring([{"claim": "x", "verdict": "contradicted", "source_ids": ["D5:4", "[D14:4]", "D99:9"],
                     "contradicting_ids": []},
                    {"claim": "y", "verdict": "contradicted", "source_ids": ["D7:8"], "contradicting_ids": ["D7:11"]}])
    parsed = judge.parse_scoring(raw, claims, {"D5:4", "D14:4", "D7:8", "D7:11"})
    first, second = parsed["claims"]
    assert first == {"claim": claims[0], "verdict": "contradicted", "sourceIds": ["D5:4", "D14:4"],
                     "contradictingIds": [], "unknownIds": ["D99:9"]}
    assert second["sourceIds"] == ["D7:8"] and second["contradictingIds"] == ["D7:11"]
    assert (parsed["correctness"], parsed["completeness"]) == (2, 2)
    supported = judge.parse_scoring(_scoring([{"verdict": "supported", "source_ids": ["D5:4"],
                                               "contradicting_ids": ["D14:4"]}]), claims[:1], {"D5:4", "D14:4"})
    assert supported["claims"][0]["contradictingIds"] == []  # only contradicted claims keep contradicting turns
    ok = {"source_ids": [], "contradicting_ids": []}
    cases = [
        (_scoring([{"claim": "x", "verdict": "supported", **ok}]), "expected 2 claim verdicts"),
        (_scoring([{"verdict": "maybe", **ok}, {"verdict": "supported", **ok}]), "verdict must be one of"),
        (_scoring([{"verdict": "supported", "source_ids": "D5:4"}, {"verdict": "supported", **ok}]),
         "source_ids must be a list"),
        (_scoring([{"verdict": "supported", "source_ids": [], "contradicting_ids": "x"},
                   {"verdict": "supported", **ok}]), "contradicting_ids must be a list"),
        (_scoring([{"verdict": "supported", **ok}] * 2, c=3), "must be 0, 1 or 2"),
        (json.dumps({"claims": [], "correctness": 1}), "missing 'completeness'"),
    ]
    for bad, message in cases:
        with pytest.raises(judge.JudgeResponseError, match=message):
            judge.parse_scoring(bad, claims, set())


# ---- judging: semantic equivalence, citations, contradictions ---------------------------------------


def test_semantically_equal_answers_share_one_judgement_and_citations_are_scored_separately():
    """Q-017: same claim, extra meta wording and a citation to a different pottery-class turn."""
    fake = FakeJudge(claims={"Melanie signed up for a pottery class yesterday.":
                             ["Melanie signed up for a pottery class on Sunday 2 July 2023."]})
    record = judge.judge_question(audit_item("Q-017"), fake, seed=0)
    assert fake.calls[1] == ("scoring", "Melanie signed up for a pottery class on 2 July 2023.")  # weekday dropped
    assert record["status"] == "ok" and record["equivalentAnswers"] and record["sharedClaims"]
    assert kinds(fake) == ["extraction", "scoring"]  # the second answer reuses both
    base, ctx = record["byRun"]["baseline"], record["byRun"]["context"]
    assert {d: base[d] for d in judge.DIMENSIONS} == {d: ctx[d] for d in judge.DIMENSIONS} == {
        "correctness": 2, "grounding": 2, "completeness": 2}
    assert base["citations"][0]["supports"] is True and base["citationStatus"] == "accurate"
    [cited] = ctx["citations"]
    assert (cited["id"], cited["exists"], cited["inGold"], cited["supports"]) == ("D14:4", True, False, False)
    assert ctx["citationStatus"] == "inaccurate"
    assert judge.outcome(record) == "tie"
    assert sum(1 for c in record["calls"] if c["reusedFrom"]) == 2


def test_wrong_verdict_does_not_turn_a_correct_citation_into_an_inaccurate_one():
    """Regression (v3): the scorer marked the shared claim contradicted, which emptied its evidence and
    made the exact gold citation [D5:4] look unsupported. Sources are now kept whatever the verdict."""
    claim = "Melanie signed up for a pottery class on 2 July 2023."
    fake = FakeJudge(claims={"Melanie signed up for a pottery class yesterday.": [claim]},
                     verdicts={claim: ("contradicted", ["D5:4"])}, scores={(claim,): (0, 0)})
    record = judge.judge_question(audit_item("Q-017"), fake, seed=0)
    base, ctx = record["byRun"]["baseline"], record["byRun"]["context"]
    assert {d: base[d] for d in judge.DIMENSIONS} == {d: ctx[d] for d in judge.DIMENSIONS}
    assert base["citationStatus"] == "accurate"
    assert ctx["citationStatus"] == "inaccurate"


def test_dropped_attribution_triggers_one_extraction_repair():
    """Regression (v3): "recommended by Caroline" was merged away. A speaker name missing from the claims
    causes one repeat with feedback; the repaired claims keep the attribution as its own claim."""
    answer = judge.normalize_answer(AUDIT["Q-024"][4])
    merged = ["Melanie read a book last year that reminded her to always pursue her dreams.",
              "No other specific books read by Melanie are mentioned."]
    repaired = ["Melanie read a book last year that reminded her to always pursue her dreams.",
                "Caroline recommended the book to Melanie.",
                "No other specific books read by Melanie are mentioned."]
    fake = FakeJudge(claims={answer: (merged, repaired)},
                     verdicts={repaired[1]: ("contradicted", ["D7:8"], ["D7:11"]),
                               repaired[2]: ("contradicted", [], ["D6:10"])})
    record = judge.judge_question(audit_item("Q-024"), fake, seed=0)
    assert kinds(fake).count("repair") == 1
    repair_prompt = next(p for kind, p in fake.prompts if kind == "repair")
    assert "left out: Caroline" in repair_prompt
    ctx = record["byRun"]["context"]
    assert [c["claim"] for c in ctx["claims"]] == repaired
    assert ctx["extractionCoverage"] == {"entities": ["Caroline", "Melanie"], "initiallyMissing": ["Caroline"],
                                         "initiallyInvented": [], "repaired": True, "missing": [], "invented": []}
    assert ctx["contradicted_claims"] == repaired[1:]
    assert ctx["claims"][1]["contradictingIds"] == ["D7:11"] and ctx["claims"][2]["contradictingIds"] == ["D6:10"]
    assert ctx["grounding"] == 0
    assert ctx["citationStatus"] == "accurate"  # [D7:8] is the source of the book claim, even though it is wrong
    assert record["byRun"]["baseline"]["extractionCoverage"]["repaired"] is False
    assert any(c["kind"] == "extraction_repair" for c in record["calls"])


def test_unrepaired_extraction_gap_is_reported():
    answer = judge.normalize_answer(AUDIT["Q-024"][4])
    merged = ["Melanie read a book last year."]
    record = judge.judge_question(audit_item("Q-024"), FakeJudge(claims={answer: (merged, merged)}), seed=0)
    assert record["byRun"]["context"]["extractionCoverage"]["missing"] == ["Caroline"]
    assert judge.summarize([record])["extractionCoverageGaps"] == ["Q-024"]


def test_rule_5_example_names_no_person_or_item_a_model_could_copy():
    assert "read no other books" not in judge.EXTRACTION_SYSTEM_PROMPT
    assert "<person> has no other <items>" in judge.EXTRACTION_SYSTEM_PROMPT


def test_faithful_claims_with_resolved_dates_speakers_and_not_available_form_are_not_flagged():
    q017, q030, q007 = audit_item("Q-017"), audit_item("Q-030"), audit_item("Q-007")
    assert judge.invented_claims(q017, AUDIT["Q-017"][3], ["Melanie signed up for a pottery class on 2 July 2023."]) == []
    assert judge.invented_claims(q030, AUDIT["Q-030"][3],
                                 ["Melanie took her kids to a pottery workshop on Friday 14 July 2023."]) == []
    assert judge.invented_claims(q007, "The excerpts do not say.",
                                 ["The conversation does not say when Melanie is planning on going camping."]) == []
    assert judge.invented_claims(q030, AUDIT["Q-030"][3], ["Melanie has read no other books"]) == [
        "Melanie has read no other books"]


_Q024_BASE = ["Melanie read a book last year that reminded her to always pursue her dreams.",
              "Melanie loved reading \"Charlotte's Web\" as a kid."]
_INVENTED = "Melanie adopted a puppy from the shelter."


def test_invented_claim_triggers_repair_kept_when_entity_count_is_unchanged():
    answer = judge.normalize_answer(AUDIT["Q-024"][3])
    fake = FakeJudge(claims={answer: ([*_Q024_BASE, _INVENTED], _Q024_BASE)})
    record = judge.judge_question(audit_item("Q-024"), fake, seed=0)
    repair_prompt = next(p for kind, p in fake.prompts if kind == "repair")
    assert f'"{_INVENTED}"' in repair_prompt and "left out" not in repair_prompt
    base = record["byRun"]["baseline"]
    assert [c["claim"] for c in base["claims"]] == _Q024_BASE
    cov = base["extractionCoverage"]
    assert cov["missing"] == [] and cov["initiallyMissing"] == []  # entity coverage was complete before and after
    assert cov["initiallyInvented"] == [_INVENTED] and cov["repaired"] is True and cov["invented"] == []
    assert kinds(fake).count("repair") == 1  # the faithful context answer is not repeated


def test_unfixed_invented_claim_is_kept_and_reported():
    answer = judge.normalize_answer(AUDIT["Q-024"][3])
    still = [*_Q024_BASE, _INVENTED]
    for repair in (still, []):  # a repeat that invents again, or one that returns nothing, is not accepted
        fake = FakeJudge(claims={answer: (still, repair)})
        record = judge.judge_question(audit_item("Q-024"), fake, seed=0)
        base = record["byRun"]["baseline"]
        assert [c["claim"] for c in base["claims"]] == still  # nothing is silently dropped
        assert base["extractionCoverage"]["repaired"] is False
        assert base["extractionCoverage"]["invented"] == [_INVENTED]
        assert judge.summarize([record])["extractionInventedClaims"] == ["Q-024"]


def test_repair_that_fixes_a_name_but_invents_a_claim_is_not_accepted():
    answer = judge.normalize_answer(AUDIT["Q-024"][4])
    merged = ["Melanie read a book last year that reminded her to always pursue her dreams."]
    worse = [*merged, "Caroline recommended the book to Melanie.", _INVENTED]
    record = judge.judge_question(audit_item("Q-024"), FakeJudge(claims={answer: (merged, worse)}), seed=0)
    cov = record["byRun"]["context"]["extractionCoverage"]
    assert cov["repaired"] is False and cov["missing"] == ["Caroline"] and cov["invented"] == []


def test_answer_entities_are_speakers_and_quoted_titles():
    names = {"Caroline", "Melanie"}
    assert judge.answer_entities(AUDIT["Q-024"][3], names) == ["Melanie", "Charlotte's Web"]
    assert judge.answer_entities("She said “Becoming Nicole” was good.", names) == ["Becoming Nicole"]
    assert judge.missing_entities(["Caroline"], ["caroline recommended it"]) == []
    assert judge.missing_entities(["Mel"], ["Melanie went"]) == ["Mel"]  # whole words only


def test_different_wording_with_the_same_claims_shares_scoring():
    """Q-030: first-person quote vs third person; same extracted claim, so one scoring call."""
    claim = ["Melanie took her kids to a pottery workshop on Friday 14 July 2023."]
    fake = FakeJudge(claims={judge.normalize_answer(AUDIT["Q-030"][3]): claim,
                             judge.normalize_answer(AUDIT["Q-030"][4]): claim})
    record = judge.judge_question(audit_item("Q-030"), fake, seed=0)
    assert not record["equivalentAnswers"] and record["sharedClaims"]
    assert sorted(kinds(fake)) == ["extraction", "extraction", "scoring"]
    assert record["byRun"]["baseline"]["correctness"] == record["byRun"]["context"]["correctness"]


def test_claim_contradicted_by_surrounding_turns_lowers_grounding():
    """Q-024: "recommended by Caroline" is contradicted by D7:11, a later and different recommendation."""
    base_claims = ["Melanie read a book last year that reminds her to pursue her dreams.",
                   "Melanie loved reading \"Charlotte's Web\" as a kid."]
    ctx_claims = ["Melanie read a book recommended by Caroline last year.",
                  "Melanie has read no other specific books."]
    fake = FakeJudge(
        claims={judge.normalize_answer(AUDIT["Q-024"][3]): base_claims,
                judge.normalize_answer(AUDIT["Q-024"][4]): ctx_claims},
        verdicts={base_claims[0]: ("supported", ["D7:8"]), base_claims[1]: ("supported", ["D6:10"]),
                  ctx_claims[0]: ("contradicted", [], ["D7:11"]), ctx_claims[1]: ("contradicted", [], ["D6:10"])},
        scores={tuple(base_claims): (1, 1), tuple(ctx_claims): (1, 0)},
    )
    record = judge.judge_question(audit_item("Q-024"), fake, seed=0)
    base, ctx = record["byRun"]["baseline"], record["byRun"]["context"]
    assert (base["correctness"], base["grounding"], base["completeness"]) == (1, 2, 1)
    assert (ctx["correctness"], ctx["grounding"], ctx["completeness"]) == (1, 0, 0)
    assert ctx["contradicted_claims"] == ctx_claims
    assert ctx["claims"][0]["contradictingIds"] == ["D7:11"]
    assert judge.outcome(record) == "baseline_win"
    assert base["citationStatus"] == "accurate"
    assert ctx["citationStatus"] == "inaccurate"  # the scorer named no source for its claims
    scoring_prompts = [p for kind, p in fake.prompts if kind == "scoring"]
    assert all("Becoming Nicole" in p for p in scoring_prompts)


def test_supported_claim_not_in_gold_evidence_still_validates_its_citation():
    """Q-014: the baseline cites D4:11, which is not gold evidence but states the claim."""
    claim = "Caroline is looking into counseling and mental health as a career."
    fake = FakeJudge(verdicts={claim: ("supported", ["D4:11", "D1:11"])})
    record = judge.judge_question(audit_item("Q-014"), fake, seed=0)
    [cited] = record["byRun"]["baseline"]["citations"]
    assert (cited["id"], cited["inGold"], cited["supports"]) == ("D4:11", False, True)
    assert "D4:11" in record["contextTurnIds"]


def test_nonexistent_citation_needs_no_model_call():
    fake = FakeJudge()
    item = audit_item("Q-007", context="Melanie is thinking about going camping next month. [D99:1]")
    record = judge.judge_question(item, fake, seed=0)
    assert kinds(fake) == ["extraction", "scoring"]
    [cited] = record["byRun"]["context"]["citations"]
    assert (cited["exists"], cited["supports"]) == (False, False)
    summary = judge.summarize([record])
    assert summary["runs"]["context"]["citation"]["nonexistent"] == 1
    assert summary["runs"]["context"]["citation"]["existingNotSupporting"] == 0


def test_empty_answers_are_scored_without_model_calls():
    fake = FakeJudge()
    record = judge.judge_question(audit_item("Q-007", baseline="", context="  "), fake, seed=0)
    assert fake.calls == []
    for run in judge.RUNS:
        s = record["byRun"][run]
        assert (s["correctness"], s["grounding"], s["completeness"]) == (0, 0, 0) and s["claims"] == []


def test_invalid_responses_and_timeouts():
    bad_extraction = judge.judge_question(audit_item("Q-030"), FakeJudge(extraction_raw="garbage"), seed=0, retries=3)
    assert bad_extraction["status"] == "invalid_response" and "claim extraction" in bad_extraction["error"]
    assert bad_extraction["calls"][0]["attempts"] == 1
    bad_scoring = judge.judge_question(audit_item("Q-030"), FakeJudge(scoring_raw='{"claims": []}'), seed=0)
    assert bad_scoring["status"] == "invalid_response" and "claim scoring" in bad_scoring["error"]

    calls = []

    def flaky(system, prompt, schema):
        calls.append(1)
        if len(calls) == 1:
            raise judge.JudgeTimeoutError("slow")
        return FakeJudge()(system, prompt, schema)

    record = judge.judge_question(audit_item("Q-007"), flaky, seed=0, retries=1)
    assert record["status"] == "ok" and record["calls"][0]["attempts"] == 2


def test_run_judge_stops_after_consecutive_request_errors():
    items = [audit_item(q) for q in AUDIT]
    calls = []

    def down(system, prompt, schema):
        calls.append(1)
        raise judge.GenerationError("connection refused")

    records = judge.run_judge(items, down, seed=0, retries=0, max_consecutive_failures=2)
    assert [r["status"] for r in records] == ["request_error"] * 2 + ["not_run"] * 3
    assert len(calls) == 2


# ---- summary ---------------------------------------------------------------------------------------


def _record(qid, base, ctx, *, status="ok", equivalent=False, shared=False, base_cit=None, ctx_cit=None,
            contradicted=()):
    by_run = None
    if status == "ok":
        def block(s, citations, contra=()):
            citations = citations or []
            return {"correctness": s[0], "grounding": s[1], "completeness": s[2], "claims": [],
                    "unsupported_claims": list(contra), "contradicted_claims": list(contra), "rationale": "",
                    "citations": citations, "citationStatus": judge.citation_status(citations),
                    "citesGold": any(c["inGold"] for c in citations)}
        by_run = {"baseline": block(base, base_cit), "context": block(ctx, ctx_cit, contradicted)}
    return {"qid": qid, "status": status, "byRun": by_run, "equivalentAnswers": equivalent, "sharedClaims": shared}


def test_outcome_ignores_citations():
    bad = [{"id": "D9:9", "exists": False, "inGold": False, "supports": False, "reason": ""}]
    assert judge.outcome(_record("Q", (1, 2, 2), (2, 0, 0))) == "context_win"
    assert judge.outcome(_record("Q", (2, 2, 2), (2, 1, 2))) == "baseline_win"
    assert judge.outcome(_record("Q", (2, 2, 2), (2, 2, 2), ctx_cit=bad)) == "tie"


def test_summarize_rates_citations_contradictions_and_inconsistency():
    good = [{"id": "D5:4", "exists": True, "inGold": True, "supports": True, "reason": ""}]
    wrong = [{"id": "D14:4", "exists": True, "inGold": False, "supports": False, "reason": ""}]
    records = [
        _record("Q-001", (2, 2, 2), (2, 2, 2), base_cit=good, ctx_cit=wrong, equivalent=True),
        _record("Q-002", (2, 2, 1), (1, 0, 0), contradicted=["x"]),
        _record("Q-003", (2, 2, 2), (1, 2, 2), shared=True),
        _record("Q-004", None, None, status="invalid_response"),
    ]
    s = judge.summarize(records)
    assert (s["nQuestions"], s["nEvaluated"], s["nFailed"]) == (4, 3, 1)
    assert s["runs"]["context"]["answersWithContradictedClaims"] == 1
    assert s["runs"]["context"]["citation"]["existingNotSupporting"] == 1
    assert s["runs"]["baseline"]["citation"]["accurateRate"] == 1.0
    # Q-003 makes the same claims in both answers but is scored differently: left out of the headline count
    assert s["outcomes"] == {"context_win": 0, "baseline_win": 1, "tie": 1}
    assert s["outcomesIncludingInconsistent"] == {"context_win": 0, "baseline_win": 2, "tie": 1}
    assert list(s["inconsistentQuestions"]) == ["Q-003"] and "correctness" in s["inconsistentQuestions"]["Q-003"]
    assert s["equivalentAnswerPairs"] == 1 and s["sharedClaimPairs"] == 1
    assert s["inconsistentEquivalentPairs"] == ["Q-003"]
    assert s["flaggedQuestions"] == ["Q-001", "Q-002", "Q-003"]


# ---- reference items, derived completeness, abstentions -------------------------------------------


@pytest.mark.parametrize("reference, expected", [
    ("pottery, camping, painting, swimming", ["pottery", "camping", "painting", "swimming"]),
    ("Her mentors, family, and friends", ["Her mentors", "family", "friends"]),
    ("\"Nothing is Impossible\", \"Charlotte's Web\"", ["\"Nothing is Impossible\"", "\"Charlotte's Web\""]),
    ("\"Love, Simon\", \"Charlotte's Web\"", ["\"Love, Simon\"", "\"Charlotte's Web\""]),  # no split inside quotes
    ("2 July 2023", ["2 July 2023"]),
    ("May 7, 2023", ["May 7, 2023"]),  # a date with a comma is one fact
    ("The Friday before 15 July 2023", ["The Friday before 15 July 2023"]),
    ("Transgender woman", ["Transgender woman"]),
    ("counseling or mental health for Transgender people", ["counseling or mental health for Transgender people"]),
    ("dinosaurs and nature", ["dinosaurs and nature"]),  # no comma: not split
    ("Yes, since she collects classic children's books", ["Yes, since she collects classic children's books"]),
    ("Likely no; though she likes reading, she wants to be a counselor",
     ["Likely no; though she likes reading, she wants to be a counselor"]),
    ("National park; she likes the outdoors", ["National park; she likes the outdoors"]),  # answer; reason
    ("by carving out me-time for activities like running, reading, or playing the violin",
     ["by carving out me-time for activities like running, reading, or playing the violin"]),
    ("", []),
])
def test_reference_items_split_only_short_comma_lists(reference, expected):
    assert judge.reference_items(reference) == expected


def test_derive_completeness_from_items_and_single_item_fallback():
    def items(*covered):
        return [{"item": f"i{n}", "covered": c} for n, c in enumerate(covered)]
    assert judge.derive_completeness(items(True, True, True, True), 0) == (2, "reference_items")
    assert judge.derive_completeness(items(True, False, True, False), 2) == (1, "reference_items")
    assert judge.derive_completeness(items(False, False), 2) == (0, "reference_items")
    for score in (0, 1, 2):  # one item can be partly right (a month without the day): the judge decides
        assert judge.derive_completeness(items(False), score) == (score, "judge")
        assert judge.derive_completeness([], score) == (score, "judge")


def test_list_completeness_is_derived_from_covered_items_and_the_judge_score_is_kept():
    """Q-016 pattern: the judge gives full completeness although only part of the list is given."""
    base_answer, ctx_answer = (judge.normalize_answer(AUDIT["Q-024"][i]) for i in (3, 4))
    base_claims, ctx_claims = ["Melanie read Nothing is Impossible and Charlotte's Web."], ["Melanie read a book."]
    fake = FakeJudge(claims={base_answer: base_claims, ctx_answer: ctx_claims},
                     coverage={tuple(base_claims): [True, True], tuple(ctx_claims): [True, False]})
    record = judge.judge_question(audit_item("Q-024"), fake, seed=0)
    scoring_prompt = next(p for kind, p in fake.prompts if kind == "scoring")
    assert 'R1. "Nothing is Impossible"\nR2. "Charlotte\'s Web"' in scoring_prompt
    base, ctx = record["byRun"]["baseline"], record["byRun"]["context"]
    assert base["completeness"] == 2 and base["overrides"] == []
    assert ctx["completeness"] == 1 and ctx["completenessSource"] == "reference_items"
    assert ctx["judgeScores"] == {"correctness": 2, "completeness": 2}  # the judge's own score is kept
    assert ctx["overrides"] == [{"field": "completeness", "judge": 2, "final": 1,
                                 "reason": "derived from reference-item coverage"}]
    assert [i["covered"] for i in ctx["referenceItems"]] == [True, False]
    none = judge.judge_question(audit_item("Q-024"), FakeJudge(
        claims={base_answer: base_claims, ctx_answer: ctx_claims},
        coverage={tuple(base_claims): [False, False], tuple(ctx_claims): [False, False]}), seed=0)
    assert none["byRun"]["baseline"]["completeness"] == 0


def test_single_fact_reference_is_one_item_and_keeps_the_judge_completeness():
    claims = ["Melanie signed up for a pottery class in July 2023."]
    fake = FakeJudge(claims={judge.normalize_answer(AUDIT["Q-017"][i]): claims for i in (3, 4)},
                     scores={tuple(claims): (1, 1)})
    record = judge.judge_question(audit_item("Q-017"), fake, seed=0)
    scoring_prompt = next(p for kind, p in fake.prompts if kind == "scoring")
    assert "Reference items:\nR1. 2 July 2023\n\n" in scoring_prompt
    base = record["byRun"]["baseline"]
    assert (base["completeness"], base["completenessSource"], base["overrides"]) == (1, "judge", [])


def test_parse_scoring_requires_one_coverage_entry_per_reference_item():
    claims = ["c"]
    entry = [{"claim": "c", "verdict": "supported", "source_ids": [], "contradicting_ids": []}]
    good = json.dumps({"claims": entry, "reference_items": [{"item": "a", "covered": True},
                                                            {"item": "b", "covered": False}],
                       "rationale": "", "correctness": 1, "completeness": 1})
    parsed = judge.parse_scoring(good, claims, set(), ["pottery", "camping"])
    assert parsed["referenceItems"] == [{"item": "pottery", "covered": True}, {"item": "camping", "covered": False}]
    with pytest.raises(judge.JudgeResponseError, match="expected 3 reference items"):
        judge.parse_scoring(good, claims, set(), ["a", "b", "c"])
    with pytest.raises(judge.JudgeResponseError, match="'covered' must be true or false"):
        judge.parse_scoring(good.replace("true", '"yes"'), claims, set(), ["a", "b"])


_NOT_SAID = "The conversation does not say when Melanie is planning on going camping."


def test_abstention_against_a_reference_scores_zero_and_keeps_the_judge_scores():
    answer = "The excerpts do not say when Melanie plans to go camping."
    fake = FakeJudge(claims={answer: [_NOT_SAID]}, scores={(_NOT_SAID,): (2, 2)})
    record = judge.judge_question(audit_item("Q-007", baseline=answer, context=answer), fake, seed=0)
    s = record["byRun"]["baseline"]
    assert (s["correctness"], s["completeness"]) == (0, 0)
    assert s["judgeScores"] == {"correctness": 2, "completeness": 2}
    assert [o["field"] for o in s["overrides"]] == ["correctness", "completeness"]
    assert all("not available" in o["reason"] and o["judge"] == 2 for o in s["overrides"])
    assert s["claims"][0]["verdict"] == "supported"  # the judge's verdict is untouched


def test_abstention_rule_needs_a_reference_and_an_answer_that_only_abstains():
    assert judge.is_abstention([_NOT_SAID]) and not judge.is_abstention([])
    assert not judge.is_abstention([_NOT_SAID, "Melanie has kids."])
    answer = "The excerpts do not say when Melanie plans to go camping."
    item = audit_item("Q-007", baseline=answer, context=answer)
    item.dataset = judge.DatasetQuestion(item.dataset.index, item.dataset.question, "", 5, ())
    fake = FakeJudge(claims={answer: [_NOT_SAID]}, scores={(_NOT_SAID,): (2, 2)})
    s = judge.judge_question(item, fake, seed=0)["byRun"]["baseline"]
    assert (s["correctness"], s["completeness"], s["overrides"]) == (2, 2, [])  # no reference: abstaining is right


# ---- inconsistent scores ---------------------------------------------------------------------------


def _claims_record(base, ctx, base_correct, ctx_correct):
    """``base``/``ctx``: lists of (claim, verdict)."""
    def block(claims, correctness):
        return {"correctness": correctness, "grounding": 2, "completeness": 2,
                "claims": [{"claim": c, "verdict": v} for c, v in claims]}
    return {"qid": "Q", "status": "ok", "equivalentAnswers": False, "sharedClaims": False,
            "byRun": {"baseline": block(base, base_correct), "context": block(ctx, ctx_correct)}}


def test_inconsistency_flags_unexplained_correctness_differences_only():
    a, b = ("Caroline has known the friends for 4 years.", "supported"), ("Caroline moved here.", "supported")
    unsupported = ("Caroline moved from Sweden.", "not_mentioned")
    assert "adds only supported claims" in judge.inconsistency(_claims_record([a], [a, b], 2, 0))
    assert "adds only supported claims" in judge.inconsistency(_claims_record([a, b], [a], 0, 2))
    assert judge.inconsistency(_claims_record([a], [a, unsupported], 2, 0)) is None  # the extra claim explains it
    assert judge.inconsistency(_claims_record([a], [a, b], 0, 2)) is None  # adding a supported item may help
    assert judge.inconsistency(_claims_record([a], [b], 2, 0)) is None  # unrelated claims: not comparable
    clash = judge.inconsistency(_claims_record([a], [(a[0], "not_mentioned"), b], 2, 2))
    assert clash.startswith("the same claim got different verdicts (supported vs not_mentioned)")
    same = _claims_record([a], [a], 2, 0)
    same["sharedClaims"] = True
    assert judge.inconsistency(same) == "equal answers or claims scored differently (correctness)"


def test_subset_with_supported_additions_scored_lower_is_flagged_and_excluded_from_the_headline(files, tmp_path):
    """Q-028 pattern: the longer answer repeats the shorter one's claim, adds a supported one, and scores 0."""
    base_answer, ctx_answer = (judge.normalize_answer(AUDIT["Q-024"][i]) for i in (3, 4))
    shared = "Melanie read a book last year that reminded her to pursue her dreams."
    extra = "Melanie loved reading Charlotte's Web as a kid."
    fake = FakeJudge(claims={base_answer: [shared], ctx_answer: [shared, extra]},
                     scores={(shared,): (2, 2), (shared, extra): (0, 0)})
    assert judge.main(_args(files, tmp_path, "--questions", "Q-024,Q-017"), call=fake) == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    q024 = next(q for q in payload["questions"] if q["qid"] == "Q-024")
    assert q024["inconsistent"] == "context adds only supported claims to baseline but scores lower correctness"
    assert (q024["byRun"]["baseline"]["correctness"], q024["byRun"]["context"]["correctness"]) == (2, 0)  # kept
    summary = payload["summary"]
    assert summary["outcomes"] == {"context_win": 0, "baseline_win": 0, "tie": 1}  # Q-017 only
    assert summary["outcomesIncludingInconsistent"] == {"context_win": 0, "baseline_win": 1, "tie": 1}
    assert list(summary["inconsistentQuestions"]) == ["Q-024"]
    md = (tmp_path / "out.md").read_text(encoding="utf-8")
    assert "1 question(s) with inconsistent judge scores excluded (Q-024" in md
    assert "baseline win (excluded: inconsistent)" in md
    assert "Left out of the headline outcome, inconsistent judge scores" in md


def test_extra_unsupported_claim_keeps_the_question_in_the_headline(files, tmp_path):
    base_answer, ctx_answer = (judge.normalize_answer(AUDIT["Q-024"][i]) for i in (3, 4))
    shared = "Melanie read a book last year that reminded her to pursue her dreams."
    extra = "Caroline recommended the book to Melanie."
    fake = FakeJudge(claims={base_answer: [shared], ctx_answer: [shared, extra]},
                     verdicts={extra: ("contradicted", ["D7:8"], ["D7:11"])},
                     scores={(shared,): (2, 2), (shared, extra): (1, 1)})
    assert judge.main(_args(files, tmp_path, "--questions", "Q-024"), call=fake) == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["questions"][0]["inconsistent"] is None
    assert payload["summary"]["outcomes"]["baseline_win"] == 1 and payload["summary"]["inconsistentQuestions"] == {}


# ---- Ollama client --------------------------------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_client_sends_the_requested_schema_with_deterministic_options(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        seen["url"], seen["timeout"] = request.full_url, timeout
        return _Response(json.dumps({"message": {"content": '{"claims": ["a"]}'}}).encode("utf-8"))

    monkeypatch.setattr(judge.urllib.request, "urlopen", fake_urlopen)
    client = judge.OllamaJudgeClient("qwen2.5:7b", "http://localhost:11434/", timeout=9, seed=0)
    assert judge.parse_extraction(client("sys", "prompt", judge.EXTRACTION_SCHEMA)) == ["a"]
    assert seen["url"] == "http://localhost:11434/api/chat" and seen["timeout"] == 9
    assert seen["payload"]["format"] == judge.EXTRACTION_SCHEMA
    assert seen["payload"]["options"]["temperature"] == 0.0 and seen["payload"]["options"]["seed"] == 0
    judge.OllamaJudgeClient("m", "http://localhost:1", format_mode="json")("s", "p", judge.SCORING_SCHEMA)
    assert seen["payload"]["format"] == "json"


def test_client_maps_timeouts_and_connection_errors(monkeypatch):
    client = judge.OllamaJudgeClient("m", "http://localhost:11434", timeout=1)

    def timeout(request, timeout):
        raise socket.timeout("timed out")

    monkeypatch.setattr(judge.urllib.request, "urlopen", timeout)
    with pytest.raises(judge.JudgeTimeoutError):
        client("s", "p", judge.SCORING_SCHEMA)

    def refused(request, timeout):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(judge.urllib.request, "urlopen", refused)
    with pytest.raises(judge.GenerationError, match="could not reach Ollama") as info:
        client("s", "p", judge.SCORING_SCHEMA)
    assert not isinstance(info.value, judge.JudgeTimeoutError)


def test_local_host_check():
    assert judge.is_local_host("http://localhost:11434") and judge.is_local_host("http://127.0.0.1:11434")
    assert not judge.is_local_host("https://ollama.example.com")


# ---- dataset and alignment --------------------------------------------------------------------------


def fence(text: str) -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}\n{text}\n{ticks}"


def section(qid, question, gold, generated, prompt=None):
    parts = [f"## {qid}", "", "- Category: 2", "- Scorable: True", ""]
    if question is not None:
        parts += ["### Question", "", fence(question), ""]
    parts += ["### Gold answer", "", fence(gold), ""]
    if generated is not None:
        parts += ["### Generated answer", "", fence(generated), ""]
    parts += ["### Retrieved memories", "", fence("## Q-999 inside a fence")]
    if prompt is not None:
        parts += ["", "### Generator prompt", "", fence(prompt)]
    return "\n".join(parts)


def report(*sections):
    return "# LoCoMo conv-26 RAG report\n\n## Metrics\n\n- x\n\n" + "\n\n".join(sections) + "\n"


def dataset_json() -> list:
    qa = [{"question": f"Placeholder {i}?", "answer": "x", "evidence": ["D1:11"], "category": 4} for i in range(30)]
    for qid, (question, reference, evidence, _, _) in AUDIT.items():
        qa[int(qid[2:]) - 1] = {"question": question, "answer": reference, "evidence": evidence, "category": 2}
    qa[0]["evidence"] = ["D8:6; D9:17"]  # malformed annotation, as in the real conv-26 Q-038
    return [{"sample_id": "conv-26", "conversation": conversation(), "qa": qa}]


@pytest.fixture
def files(tmp_path):
    paths = {"dataset": tmp_path / "locomo.json", "baseline": tmp_path / "base.md", "context": tmp_path / "ctx.md"}
    paths["dataset"].write_text(json.dumps(dataset_json()), encoding="utf-8")
    paths["baseline"].write_text(report(*[section(q, a[0], a[1], a[3]) for q, a in AUDIT.items()]), encoding="utf-8")
    paths["context"].write_text(report(*[section(q, a[0], a[1], a[4]) for q, a in reversed(AUDIT.items())]),
                                encoding="utf-8")
    return paths


def test_load_dataset_keeps_session_order_and_aligns(files):
    data = judge.load_dataset(files["dataset"])
    assert data.questions[0].evidence_ids == ("D8:6", "D9:17")
    assert data.sessions[7] == ["D7:8", "D7:9", "D7:10", "D7:11", "D7:12"]
    items = judge.align(data, judge.parse_report(files["baseline"].read_text()),
                        judge.parse_report(files["context"].read_text()))
    assert [i.qid for i in items] == list(AUDIT)
    assert items[-1].answers["context"] == AUDIT["Q-030"][4]


def test_load_dataset_requires_a_single_record(tmp_path):
    path = tmp_path / "two.json"
    record = dataset_json()[0]
    path.write_text(json.dumps([record, {**record, "sample_id": "conv-30"}]), encoding="utf-8")
    with pytest.raises(judge.InputError, match="--sample-id"):
        judge.load_dataset(path)
    assert judge.load_dataset(path, "conv-30").sample_id == "conv-30"


@pytest.mark.parametrize("base, ctx, message", [
    ([section("Q-007", AUDIT["Q-007"][0], "June 2023", "a")], [section("Q-014", AUDIT["Q-014"][0], AUDIT["Q-014"][1], "b")],
     "only in the baseline report"),
    ([section("Q-007", "Unknown?", "x", "a")], [section("Q-007", "Unknown?", "x", "b")], "not found in dataset"),
    ([section("Q-007", AUDIT["Q-007"][0], "July 2023", "a")], [section("Q-007", AUDIT["Q-007"][0], "July 2023", "b")],
     "does not match the dataset answer"),
    ([section("Q-007", AUDIT["Q-007"][0], "June 2023", None)], [section("Q-007", AUDIT["Q-007"][0], "June 2023", "b")],
     "missing 'Generated answer' block"),
    ([section("Q-007", None, "June 2023", "a")], [section("Q-007", None, "June 2023", "b")], "missing 'Question' block"),
])
def test_align_fails_clearly(base, ctx, message):
    data = judge.Dataset("conv-26", [judge._dataset_question(i, q) for i, q in enumerate(dataset_json()[0]["qa"])],
                         TURN_INDEX)
    with pytest.raises(judge.InputError, match=message):
        judge.align(data, judge.parse_report(report(*base)), judge.parse_report(report(*ctx)))


# ---- CLI and outputs ----------------------------------------------------------------------------------


def _args(files, tmp_path, *extra):
    return ["--dataset", str(files["dataset"]), "--baseline", str(files["baseline"]),
            "--context", str(files["context"]), "--output-json", str(tmp_path / "out.json"),
            "--output-md", str(tmp_path / "out.md"), "--host", "http://localhost:11434",
            "--model", "qwen2.5:7b", *extra]


def test_main_writes_claims_derived_grounding_and_citations(files, tmp_path):
    ctx_claim = "Melanie read a book recommended by Caroline last year."
    fake = FakeJudge(claims={judge.normalize_answer(AUDIT["Q-024"][4]): [ctx_claim]},
                     verdicts={ctx_claim: ("contradicted", [])})
    assert judge.main(_args(files, tmp_path), call=fake) == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["judgeVersion"] == "6" and payload["metadata"]["contextWindow"] == 3
    assert any("same model" in text for text in payload["metadata"]["limitations"])
    by_qid = {q["qid"]: q for q in payload["questions"]}
    assert by_qid["Q-017"]["byRun"]["context"]["citationStatus"] == "inaccurate"
    assert by_qid["Q-017"]["byRun"]["context"]["correctness"] == by_qid["Q-017"]["byRun"]["baseline"]["correctness"]
    assert by_qid["Q-024"]["byRun"]["context"]["grounding"] == 0
    assert payload["summary"]["equivalentAnswerPairs"] == 2  # Q-007 and Q-017
    markdown = (tmp_path / "out.md").read_text(encoding="utf-8")
    assert "not a ground-truth evaluation" in markdown and "## Limitations" in markdown
    assert "### Citation accuracy (separate from factual quality)" in markdown
    assert f"Claim (contradicted): {ctx_claim}" in markdown
    assert "Citation [D14:4] exists, does not support" in markdown
    assert "| Answers with a contradicted claim | 0 | 1 |" in markdown


def test_main_question_filter_overwrite_and_failures(files, tmp_path, capsys):
    args = _args(files, tmp_path, "--questions", "Q-017,Q-024")
    assert judge.main(args, call=FakeJudge()) == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert [q["qid"] for q in payload["questions"]] == ["Q-017", "Q-024"]
    assert judge.main(args, call=FakeJudge()) == 2
    assert "pass --overwrite" in capsys.readouterr().err
    assert judge.main([*args, "--overwrite"], call=FakeJudge()) == 0
    assert judge.main(_args(files, tmp_path, "--questions", "Q-099"), call=FakeJudge()) == 2
    assert "Q-099" in capsys.readouterr().err

    def down(system, prompt, schema):
        raise judge.GenerationError("connection refused")

    assert judge.main(_args(files, tmp_path / "x"), call=down) == 1
    assert not (tmp_path / "x").exists()


def test_main_rejects_remote_host_without_calling(files, tmp_path, capsys):
    args = _args(files, tmp_path)
    args[args.index("http://localhost:11434")] = "https://ollama.example.com"
    assert judge.main(args) == 2
    assert "not local" in capsys.readouterr().err


def test_dry_run_prints_prompts_without_writing(files, tmp_path, capsys):
    assert judge.main(_args(files, tmp_path, "--dry-run", "--questions", "Q-024"),
                      call=lambda *a: pytest.fail("judge called")) == 0
    out = capsys.readouterr().out
    assert "extraction prompt for Q-024 answer A" in out and "scoring prompt for Q-024" in out
    assert "Becoming Nicole" in out
    assert not (tmp_path / "out.json").exists()


def test_judge_uses_the_shared_resolver():
    from continuator_memory.v2 import temporal

    assert judge.resolve_relative_expressions is temporal.resolve_relative_expressions
    assert judge.format_date is temporal.format_date
