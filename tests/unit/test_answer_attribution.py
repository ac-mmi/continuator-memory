"""Regression tests for attribution stripping in validate_answer_citations.

The attribution-tail cleanup runs on sentences without a kept citation once any citation in
the answer is rejected. It must remove a trailing source reference only, never claim text.
"""

from __future__ import annotations

from continuator_memory.webapi.answering import PromptExcerpt, validate_answer_citations

D1_16 = PromptExcerpt(
    "D1:16", "user",
    "Rafael: That photo is from when my sailing crew took home first in a local regatta last year. "
    "It was great out on the water! I'm keen to share that with other sailors. "
    "Noor, you ever been in any sailing races or regattas?",
    ("mem:D1:16:c0",),
)
D1_17 = PromptExcerpt(
    "D1:17", "assistant",
    "Noor: I used to compete in a few sailing races and regattas - my best memory was when my "
    "team won first place at a regionals at age fifteen. It was an awesome feeling of accomplishment!",
    ("mem:D1:17:c0",),
)
D1_18 = PromptExcerpt(
    "D1:18", "user", "Rafael: Wow! Winning first place is amazing! What boat were you sailing?",
    ("mem:D1:18:c0",),
)
EXCERPTS = (D1_16, D1_17, D1_18)
QUESTION = "Did Rafael and Noor both participate in sailing races?"


def test_leading_attribution_keeps_claim():
    # Modelled on a real diagnostic answer; the names and scenario are fictional. [D1:18] is rejected.
    raw = (
        "Yes, they did. According to the excerpt [D1:18], Noor mentioned competing in a few sailing "
        "races and regattas, and winning first place at a regionals when she was fifteen. "
        "Additionally, Rafael recalled participating in a local regatta last year where his sailing "
        "crew took home first place [D1:16]."
    )
    display, diag = validate_answer_citations(raw, EXCERPTS, question=QUESTION)
    assert diag["rawAnswer"] == raw
    assert [c["kept"] for c in diag["citations"]] == [False, True]
    assert "[D1:18]" not in display
    assert ("Noor mentioned competing in a few sailing races and regattas, and winning first "
            "place at a regionals when she was fifteen.") in display
    assert "n she was fifteen" not in display.replace("when she was fifteen", "")
    assert display.endswith("took home first place [D1:16].")


def test_short_leading_attribution_sentence_is_not_deleted():
    raw = "According to the excerpt [D9:9], Noor won first place at regionals. Rafael's sailing crew took home first in a local comp [D1:16]."
    display, _ = validate_answer_citations(raw, EXCERPTS, question=QUESTION)
    assert "Noor won first place at regionals" in display


def test_mid_sentence_attribution_keeps_trailing_claim():
    raw = "Noor, according to the excerpt [D9:9], won first place at regionals. Rafael's sailing crew took home first in a local comp [D1:16]."
    display, _ = validate_answer_citations(raw, EXCERPTS, question=QUESTION)
    assert "won first place at regionals" in display


def test_attribution_to_a_person_without_source_reference_is_kept():
    raw = "Rafael trained for months according to Noor [D9:9]. Rafael's sailing crew took home first in a local comp [D1:16]."
    display, _ = validate_answer_citations(raw, EXCERPTS, question=QUESTION)
    assert "Rafael trained for months according to Noor" in display


def test_trailing_source_reference_is_still_removed():
    raw = "Noor won first place at regionals as mentioned in message D1 [D9:9]. Rafael's sailing crew took home first in a local comp [D1:16]."
    display, _ = validate_answer_citations(raw, EXCERPTS, question=QUESTION)
    assert "as mentioned in message" not in display
    assert "Noor won first place at regionals." in display


def test_no_rejected_citation_leaves_text_unchanged():
    raw = "According to the excerpt, Noor won first place at regionals [D1:17]."
    display, diag = validate_answer_citations(raw, EXCERPTS, question=QUESTION)
    assert all(c["kept"] for c in diag["citations"])
    assert display == raw
