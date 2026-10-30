"""Query-aware excerpt trimming and removal of dangling attribution fragments."""

from __future__ import annotations

from types import SimpleNamespace

from continuator_memory.models import Conversation, Message
from continuator_memory.webapi.answering import PromptExcerpt, build_answer_context, validate_answer_citations
from continuator_memory.webapi.prompt_budget import estimate_tokens, trim_to_tokens
from tests.unit._lme_fixtures import (
    MSG_731E37D7,
    Q_731E37D7,
    RAW_2698E78F_PILOT,
    RAW_2698E78F_RERUN,
    RAW_8C8961AE_RERUN,
    S005T000,
    S024T002,
    S038T010,
)

CAP_731E37D7 = 105  # the content cap from the run this regression came from


# ---- trimming -------------------------------------------------------------------------------

def test_731e37d7_dollar_amount_near_the_end_survives_trimming():
    assert len(MSG_731E37D7) == 959 and MSG_731E37D7.index("$20") == 886
    assert "$20" not in trim_to_tokens(MSG_731E37D7, CAP_731E37D7)  # the old head-only behaviour
    out = trim_to_tokens(MSG_731E37D7, CAP_731E37D7, query=Q_731E37D7)
    assert "I paid $20 to attend" in out
    assert estimate_tokens(out) <= CAP_731E37D7
    assert out.startswith("…")  # the omitted opening is marked, not silent


def test_kept_sentences_stay_in_original_order_and_gaps_are_marked():
    text = ("Alpha opens the note. Filler sentence one here. Filler sentence two here. "
            "The boat is painted red. Filler three. Filler four. Filler five. Last line mentions the harbour.")
    out = trim_to_tokens(text, 20, query="What colour is the boat?")
    assert "The boat is painted red." in out
    pieces = [p.strip() for p in out.split("…") if p.strip()]
    positions = [text.index(p) for p in pieces]
    assert positions == sorted(positions)
    assert "…" in out


def test_without_query_overlap_trimming_keeps_the_head():
    text = " ".join(f"word{i}." for i in range(300))
    assert trim_to_tokens(text, 30, query="unrelated question about zebras") == trim_to_tokens(text, 30)


def test_decimal_numbers_are_not_split_into_sentences():
    text = "Intro sentence here. " * 30 + "The ticket cost $5.50 at the workshop. " + "Closing remark. " * 30
    out = trim_to_tokens(text, 40, query="How much did the workshop ticket cost?")
    assert "$5.50" in out


def test_731e37d7_answer_reaches_the_prompt_under_a_full_budget():
    filler = "User: " + "notes about unrelated travel plans and recipes with many words in it. " * 60
    texts = [filler] * 12 + [MSG_731E37D7] + [filler] * 12
    msgs = tuple(Message(f"m{i}", "c", i, "user", t, "2023/02/26 (Sun) 10:00") for i, t in enumerate(texts))
    conv = Conversation("c", msgs)
    ids = [m.message_id for m in msgs]
    matches = [SimpleNamespace(memory_id=f"mem:{mid}", source_message_ids=(mid,), statement="", rank=i, score=1.0,
                               similarity=1.0) for i, mid in enumerate(ids, 1)]
    evidence = SimpleNamespace(selection=SimpleNamespace(selected=tuple(ids), dropped=()), matches=matches)
    ctx = build_answer_context(Q_731E37D7, evidence, conv, seed_ids=[], expanded_ids=[],
                               context_tokens=4096, output_tokens=256)
    assert [e.message_id for e in ctx.excerpts] == ids  # rank order kept, nothing dropped
    assert "m12" in ctx.budget["trimmedMessageIds"]
    assert "I paid $20 to attend" in ctx.prompt


# ---- dangling attribution fragments ----------------------------------------------------------

def _excerpt(mid, text, ts):
    return PromptExcerpt(mid, "user", text, (f"mem:{mid}:c0",), False, ts)


def test_2698e78f_copied_excerpt_leaves_no_message_date_fragment():
    excerpts = (_excerpt("s024t002", S024T002, "2023/04/03 (Mon) 14:18"),)
    display, _ = validate_answer_citations(RAW_2698E78F_RERUN, excerpts,
                                           question="How often do I see my violin teacher, Dr. Okafor?")
    assert display == "You see your violin teacher, Dr. Okafor, every two weeks."


def test_pilot_2698e78f_message_id_stub_is_removed():
    excerpts = (_excerpt("s038t010", S038T010, "2023/11/03 (Fri) 19:56"),)
    display, _ = validate_answer_citations(RAW_2698E78F_PILOT, excerpts,
                                           question="How often do I see my violin teacher, Dr. Okafor?")
    assert display == "You see your violin teacher, Dr. Okafor, every week."


def test_8c8961ae_empty_citation_label_is_removed():
    excerpts = (_excerpt("s005t000", S005T000, "2023/05/20 (Sat) 10:00"),)
    q = "Which trip did I take first, the one to Canada with family or the solo trip to Iceland?"
    display, _ = validate_answer_citations(RAW_8C8961AE_RERUN, excerpts, question=q)
    assert not display.rstrip().endswith("Citation:")
    assert display.endswith("the solo trip to Iceland took place before the family trip to Canada.")


def test_sentences_with_content_are_never_treated_as_orphans():
    excerpts = (_excerpt("s024t002", S024T002, "2023/04/03 (Mon) 14:18"),)
    raw = "The source of the delay was the 3 pm train [s999]. See the excerpt [s999]."
    display, _ = validate_answer_citations(raw, excerpts, question="Why was I late?")
    assert "The source of the delay was the 3 pm train." in display
    assert "See the excerpt" not in display
