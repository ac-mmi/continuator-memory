"""LoCoMo answer-comparison script: report parsing, normalization, token F1, pairing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from continuator_memory.evaluation import metrics

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "score_locomo_answers.py"
_spec = importlib.util.spec_from_file_location("score_locomo_answers", _SCRIPT)
score = importlib.util.module_from_spec(_spec)
# @dataclass resolves the module through sys.modules, so register it before executing.
sys.modules[_spec.name] = score
_spec.loader.exec_module(score)

HEADER = """# LoCoMo conv-26 RAG report

- Retrieval mode: `hybrid` top-k=15 depth=50

## Metrics

- Questions: 3 processed=3

## Failures and annotation issues

- `Q-031` Would Melanie be considered a member? — annotation; issues: evidence_empty
"""


def fence(text: str) -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}\n{text}\n{ticks}"


def section(qid: str, gold: str | None, generated: str | None, *, error: str | None = None,
            memory: str = "Caroline: hello") -> str:
    parts = [f"## {qid}", "", "- Category: 2", "- Scorable: True"]
    if error:
        parts.append(f"- Error: {error}")
    parts += ["", "### Question", "", fence("When?"), ""]
    if gold is not None:
        parts += ["### Gold answer", "", fence(gold), ""]
    if generated is not None:
        parts += ["### Generated answer", "", fence(generated), ""]
    parts += ["- Gold evidence: D1:3", "", "### Retrieved memories", "",
              "**r1** `mem:D1:3:c0`", "", fence(memory), "", "### Generator prompt", "", fence("Question: When?")]
    return "\n".join(parts)


def report(*sections: str) -> str:
    return HEADER + "\n" + "\n\n".join(sections) + "\n"


# ---- parsing --------------------------------------------------------------------------


def test_parse_extracts_gold_and_generated_by_question_id():
    parsed = score.parse_report(report(
        section("Q-001", "7 May 2023", "Caroline went yesterday. [D1:3]"),
        section("Q-002", "2022", "Line one\nline two"),
    ))
    assert list(parsed.sections) == ["Q-001", "Q-002"]
    assert parsed.sections["Q-001"].gold == "7 May 2023"
    assert parsed.sections["Q-001"].generated == "Caroline went yesterday. [D1:3]"
    assert parsed.sections["Q-002"].generated == "Line one\nline two"
    assert all(not s.problems for s in parsed.sections.values())
    assert parsed.problems == []


def test_headings_inside_fences_are_ignored_and_long_fences_close_correctly():
    tricky_memory = "## Q-999\n### Gold answer\n```\nnot an answer\n```"
    parsed = score.parse_report(report(section("Q-001", "guinea pig", "uses ``` inside", memory=tricky_memory)))
    assert list(parsed.sections) == ["Q-001"]
    assert parsed.sections["Q-001"].generated == "uses ``` inside"
    assert parsed.sections["Q-001"].gold == "guinea pig"
    assert not parsed.sections["Q-001"].problems


def test_question_block_is_captured():
    parsed = score.parse_report(report(section("Q-001", "2022", "2022")))
    assert parsed.sections["Q-001"].question == "When?"


def test_generator_prompt_block_is_captured_and_optional():
    parsed = score.parse_report(report(section("Q-001", "2022", "2022")))
    assert parsed.sections["Q-001"].prompt == "Question: When?"
    bare = "\n".join(["## Q-002", "", "### Gold answer", "", fence("2022"), "", "### Generated answer", "", fence("2022")])
    parsed = score.parse_report(report(bare))
    assert parsed.sections["Q-002"].prompt is None and not parsed.sections["Q-002"].problems


def test_empty_generated_block_is_empty_string_not_missing():
    parsed = score.parse_report(report(section("Q-001", "2022", "")))
    assert parsed.sections["Q-001"].generated == ""
    assert not parsed.sections["Q-001"].problems


def test_missing_and_malformed_blocks_are_reported():
    no_generated = section("Q-001", "2022", None)
    no_fence = "\n".join(["## Q-002", "", "### Gold answer", "", "2022 without a fence", "",
                          "### Generated answer", "", fence("2022")])
    parsed = score.parse_report(report(no_generated, no_fence))
    assert "missing 'Generated answer' block" in parsed.sections["Q-001"].problems
    assert "no fenced block under 'Gold answer'" in parsed.sections["Q-002"].problems
    assert "missing 'Gold answer' block" in parsed.sections["Q-002"].problems


def test_duplicate_section_marks_question_bad():
    parsed = score.parse_report(report(section("Q-001", "a", "a"), section("Q-001", "a", "b")))
    assert "duplicate question section" in parsed.sections["Q-001"].problems
    assert parsed.problems == ["Q-001: duplicate question section"]


def test_row_error_line_is_captured():
    parsed = score.parse_report(report(section("Q-001", "2022", "", error="RuntimeError: boom")))
    assert parsed.sections["Q-001"].error == "RuntimeError: boom"


# ---- normalization and scores ----------------------------------------------------------


def test_uses_project_normalization_helper():
    assert score.normalize_text is metrics.normalize_text
    assert score.exact_match("7 May, 2023!", "7 may 2023")
    assert not score.exact_match("May 2023", "7 May 2023")


@pytest.mark.parametrize("prediction, gold, expected", [
    ("guinea pig", "Guinea pig.", 1.0),
    ("a guinea pig named Oscar", "guinea pig", 2 * (2 / 5) * 1.0 / (2 / 5 + 1.0)),
    ("a a b", "a b b", 2 / 3),
    ("dog", "guinea pig", 0.0),
    ("", "", 1.0),
    ("...", "", 1.0),
    ("", "2022", 0.0),
    ("2022", "", 0.0),
])
def test_token_f1(prediction, gold, expected):
    assert score.token_f1(prediction, gold) == pytest.approx(expected)


# ---- pairing ------------------------------------------------------------------------------


def _pair(baseline_sections, context_sections):
    base = score.parse_report(report(*baseline_sections))
    ctx = score.parse_report(report(*context_sections))
    rows = score.pair_reports(base, ctx)
    return {r.qid: r for r in rows}, rows


def test_pairs_by_question_id_not_file_order():
    by_id, rows = _pair(
        [section("Q-002", "guinea pig", "a dog"), section("Q-001", "2022", "2022")],
        [section("Q-001", "2022", "in 2023"), section("Q-002", "guinea pig", "guinea pig")],
    )
    assert [r.qid for r in rows] == ["Q-001", "Q-002"]
    assert by_id["Q-001"].em_baseline is True and by_id["Q-001"].em_context is False
    assert by_id["Q-002"].f1_baseline == 0.0 and by_id["Q-002"].f1_context == 1.0
    summary = score.summarize(rows)
    assert summary.n_scored == 2
    assert summary.em_baseline == 0.5 and summary.em_context == 0.5
    assert summary.em_delta_pp == pytest.approx(0.0)
    assert summary.f1_delta == pytest.approx((by_id["Q-001"].f1_context + 1.0) / 2 - 0.5)


def test_missing_mismatched_and_ungolded_questions_are_not_scored():
    by_id, rows = _pair(
        [section("Q-001", "2022", "2022"), section("Q-002", "a", "a"),
         section("Q-003", "", ""), section("Q-004", "x", None)],
        [section("Q-002", "b", "b"), section("Q-003", "", ""),
         section("Q-004", "x", "x"), section("Q-005", "y", "y")],
    )
    assert by_id["Q-001"].status == score.STATUS_MISSING_CONTEXT
    assert by_id["Q-005"].status == score.STATUS_MISSING_BASELINE
    assert by_id["Q-002"].status == score.STATUS_GOLD_MISMATCH
    assert by_id["Q-003"].status == score.STATUS_NO_GOLD
    assert by_id["Q-004"].status == score.STATUS_MALFORMED
    assert all(r.f1_baseline is None and r.f1_context is None for r in rows)
    summary = score.summarize(rows)
    assert summary.n_scored == 0 and summary.em_baseline is None and summary.f1_delta is None


def test_empty_and_failed_answers_score_zero_and_are_flagged():
    by_id, rows = _pair(
        [section("Q-001", "2022", "")],
        [section("Q-001", "2022", "The answer model could not be reached. could not reach Ollama")],
    )
    row = by_id["Q-001"]
    assert row.status == score.STATUS_SCORED
    assert row.f1_baseline == 0.0 and row.em_baseline is False
    assert row.f1_context == 0.0
    assert "baseline: empty_answer" in row.notes and "context: generation_error" in row.notes
    assert score.summarize(rows).flag_counts == {"baseline: empty_answer": 1, "context: generation_error": 1}


# ---- CLI ----------------------------------------------------------------------------------


def test_main_prints_aggregates_and_per_question_table(tmp_path, capsys):
    base = tmp_path / "base.md"
    ctx = tmp_path / "ctx.md"
    base.write_text(report(section("Q-001", "guinea pig", "a dog"), section("Q-002", "2022", "2022")), encoding="utf-8")
    ctx.write_text(report(section("Q-001", "guinea pig", "guinea pig | Oscar"), section("Q-002", "2022", "2022")),
                   encoding="utf-8")
    assert score.main(["--baseline", str(base), "--context", str(ctx)]) == 0
    out = capsys.readouterr().out
    assert "| Exact match | 50.0% | 50.0% | +0.0 pp |" in out
    assert "| Mean token F1 | 0.500 |" in out
    assert "| Q-001 | scored | guinea pig | a dog | guinea pig \\| Oscar |" in out
    assert "Scored (paired) questions: 2" in out


def test_main_fails_clearly_on_unreadable_or_empty_reports(tmp_path, capsys):
    empty = tmp_path / "empty.md"
    empty.write_text("# nothing here\n", encoding="utf-8")
    assert score.main(["--baseline", str(tmp_path / "missing.md"), "--context", str(empty)]) == 2
    assert "cannot read baseline report" in capsys.readouterr().err
    assert score.main(["--baseline", str(empty), "--context", str(empty)]) == 2
    assert "no '## Q-NNN' question sections" in capsys.readouterr().err
