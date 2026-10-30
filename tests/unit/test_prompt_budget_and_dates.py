"""Token-aware answer-prompt budget and message timestamp normalisation."""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.temporal import message_day, normalize_timestamp, parse_timestamp
from continuator_memory.webapi.answering import (
    SYSTEM_PROMPT,
    GenerationSettings,
    build_answer_context,
    estimated_prompt_tokens,
    generate_answer,
)
from continuator_memory.webapi.prompt_budget import (
    MIN_EXCERPT_TOKENS,
    PromptBudget,
    allocate,
    estimate_tokens,
    trim_to_tokens,
)

QUESTION = "What colour is the boat?"


def _conversation(texts, timestamp="2023-05-20T10:00:00"):
    msgs = tuple(Message(f"m{i}", "c", i, "assistant" if i % 2 else "user", t, timestamp)
                 for i, t in enumerate(texts))
    return Conversation("c", msgs)


def _evidence(ids):
    matches = [SimpleNamespace(memory_id=f"mem:{mid}", source_message_ids=(mid,), statement="", rank=i,
                               score=1.0, similarity=1.0) for i, mid in enumerate(ids, 1)]
    return SimpleNamespace(selection=SimpleNamespace(selected=tuple(ids), dropped=()), matches=matches)


def _ctx(texts, **kw):
    conv = _conversation(texts)
    ids = [m.message_id for m in conv.messages]
    return build_answer_context(QUESTION, _evidence(ids), conv, seed_ids=[], expanded_ids=[], **kw)


# ---- estimator ------------------------------------------------------------------------------

def test_estimator_counts_digits_individually_and_is_not_a_character_count():
    assert estimate_tokens("2023") == 4
    assert estimate_tokens("the cat sat") == 3
    table = "| --- | --- | --- |\n| 8 am - 4 pm | Admon | 12 |"
    assert estimate_tokens(table) > len(table) / 4  # dense text costs more than chars/4


def test_trim_to_tokens_respects_the_cap_and_word_boundaries():
    text = " ".join(f"word{i}" for i in range(200))
    cut = trim_to_tokens(text, 50)
    assert estimate_tokens(cut) <= 50 and text.startswith(cut) and not cut.endswith(" ")
    assert trim_to_tokens("short", 50) == "short"


# ---- allocation -----------------------------------------------------------------------------

def test_allocate_keeps_everything_when_it_fits():
    assert allocate([5, 5], [10, 10], 100).content_cap is None


def test_allocate_trims_long_excerpts_before_dropping_any():
    plan = allocate([5, 5, 5], [5000, 20, 20], 400)
    assert plan.keep == 3
    assert 20 <= plan.content_cap < 5000
    assert 15 + min(5000, plan.content_cap) + 40 <= 400


def test_allocate_drops_only_lowest_ranked_when_minimum_shares_do_not_fit():
    plan = allocate([10] * 10, [500] * 10, 3 * (10 + MIN_EXCERPT_TOKENS))
    assert plan.keep == 3


# ---- build_answer_context -------------------------------------------------------------------

def test_long_top_excerpt_no_longer_pushes_out_the_answer_bearing_excerpt():
    # Under the old 12,000-character cap the 20,000-character first excerpt consumed the
    # whole budget and m1 (which holds the answer) was dropped.
    long_text = "User: " + ("I like long rambling notes about the weather and travel plans. " * 320)
    texts = [long_text, "Assistant: The boat is painted bright red.", "User: thanks"]
    assert len(long_text) > 12_000
    ctx = _ctx(texts, context_tokens=4096, output_tokens=256)
    ids = [e.message_id for e in ctx.excerpts]
    assert ids == ["m0", "m1", "m2"]  # rank order kept, nothing dropped
    assert "bright red" in ctx.prompt
    assert ctx.excerpts[0].truncated and not ctx.excerpts[1].truncated
    assert ctx.budget["droppedForBudget"] == [] and ctx.budget["trimmedMessageIds"] == ["m0"]


def test_prompt_plus_output_always_fits_the_context_window():
    texts = ["User: " + ("x" * 9000 + " ") * 3 for _ in range(25)]
    for num_ctx, out in ((4096, 256), (2048, 512), (8192, 1024)):
        ctx = _ctx(texts, context_tokens=num_ctx, output_tokens=out)
        total = ctx.budget["templateTokens"] + estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(ctx.prompt)
        assert total == estimated_prompt_tokens(ctx)
        assert total + out <= num_ctx


def test_many_long_excerpts_drop_from_the_bottom_of_the_ranking():
    texts = [f"User: item {i} " + "lorem ipsum dolor sit amet " * 400 for i in range(25)]
    ctx = _ctx(texts, context_tokens=4096, output_tokens=256)
    kept = [e.message_id for e in ctx.excerpts]
    assert kept == [f"m{i}" for i in range(len(kept))]  # a prefix of the ranking
    assert ctx.budget["droppedForBudget"] == [f"m{i}" for i in range(len(kept), 25)]
    assert ctx.truncated


def test_short_excerpts_are_untouched():
    ctx = _ctx(["User: hello", "Assistant: the boat is red"], context_tokens=4096, output_tokens=256)
    assert not ctx.truncated and ctx.budget["contentCapTokens"] is None
    assert [e.content for e in ctx.excerpts] == ["User: hello", "Assistant: the boat is red"]


def test_optional_character_ceiling_still_applies_without_dropping_top_evidence():
    texts = ["User: " + "a" * 5000, "Assistant: The boat is red."]
    ctx = _ctx(texts, max_context_chars=1000, context_tokens=4096, output_tokens=256)
    assert [e.message_id for e in ctx.excerpts] == ["m0", "m1"]
    assert sum(len(e.content) for e in ctx.excerpts) <= 1000 + 2


# ---- generation guard -----------------------------------------------------------------------

class _Gen:
    model_name = "fake"

    def __init__(self, reported):
        self.reported = reported
        self.calls = 0
        self.last_usage = None

    def generate(self, system, prompt, temperature=None, num_predict=None):
        self.calls += 1
        self.last_usage = {"num_ctx": 4096, "prompt_eval_count": self.reported}
        return "Red."


def test_generator_is_not_called_when_the_prompt_cannot_fit():
    ctx = _ctx(["User: the boat is red"], context_tokens=4096, output_tokens=256)
    settings = GenerationSettings(enabled=True, model="fake", num_predict=256, num_ctx=400)
    gen = _Gen(100)
    answer, meta = generate_answer(ctx, settings, gen)
    assert gen.calls == 0 and meta["status"] == "context_overflow"


def test_reported_overflow_after_the_call_is_flagged():
    ctx = _ctx(["User: the boat is red"], context_tokens=4096, output_tokens=256)
    settings = GenerationSettings(enabled=True, model="fake", num_predict=256, num_ctx=4096)
    answer, meta = generate_answer(ctx, settings, _Gen(4000))
    assert meta["status"] == "context_overflow" and meta["rawAnswer"] == "Red."
    answer, meta = generate_answer(ctx, settings, _Gen(500))
    assert meta["status"] == "ok"


# ---- timestamps -----------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("2023/03/04 (Sat) 14:32", datetime(2023, 3, 4, 14, 32)),
    ("2023/05/20 (Sat) 02:21", datetime(2023, 5, 20, 2, 21)),
    ("2023/3/4", datetime(2023, 3, 4)),
    ("2023/03/04 14:32:05", datetime(2023, 3, 4, 14, 32, 5)),
])
def test_longmemeval_timestamps_parse(raw, expected):
    assert parse_timestamp(raw) == expected
    assert normalize_timestamp(raw) == expected.isoformat()
    assert message_day(raw) == expected.date()


@pytest.mark.parametrize("raw, expected_day", [
    ("2023-05-08T13:56:00", date(2023, 5, 8)),
    ("2023-05-08", date(2023, 5, 8)),
    ("2024-03-01T09:00:00Z", date(2024, 3, 1)),
    ("2024-03-01T09:00:00+05:30", date(2024, 3, 1)),
])
def test_iso_timestamps_behave_as_before(raw, expected_day):
    assert message_day(raw) == expected_day
    assert normalize_timestamp(raw) is not None


@pytest.mark.parametrize("raw", [None, "", "   ", "yesterday", "2023/13/40 (Mon) 10:00", "2023/03/04 (Sat) 25:00",
                                 "8 May, 2023"])
def test_unsupported_or_invalid_timestamps_are_none(raw):
    assert parse_timestamp(raw) is None and message_day(raw) is None


def test_longmemeval_dates_now_reach_the_prompt_and_resolve_relative_times():
    conv = Conversation("c", (Message("s001t000", "c", 0, "user",
                                      "I went to the open mic last month.", "2023/05/20 (Sat) 02:21"),))
    ctx = build_answer_context("When?", _evidence(["s001t000"]), conv, seed_ids=[], expanded_ids=[])
    assert "(message date: 20 May 2023)" in ctx.prompt
    assert "last month" in ctx.prompt and "April 2023" in ctx.prompt
