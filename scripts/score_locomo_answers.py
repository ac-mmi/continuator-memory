"""Compare generated answers in two LoCoMo Markdown reports (baseline vs contextual).

Reads reports written by ``continuator_memory.webapi.locomo_report``, pairs questions by ID
(``## Q-001``), and scores each generated answer against the report's gold answer with
normalized exact match and token-level F1. Evaluation only: nothing is fed back into retrieval
or generation.

Scoring rules:
- Text is normalized with ``continuator_memory.evaluation.metrics.normalize_text``.
- Token F1 uses whitespace-separated normalized tokens and multiset overlap; both empty is 1.0,
  exactly one empty is 0.0.
- A question is scored only when it is present and well-formed in both reports, both reports
  carry the same non-empty gold answer, and both have a generated-answer block. Everything else
  is listed with its reason and excluded from the aggregates, so a missing answer never counts
  as correct. An empty generated answer is scored (it gets 0 against a non-empty gold) and
  flagged.

Usage:
    python scripts/score_locomo_answers.py --baseline A.md --context B.md
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from continuator_memory.evaluation.metrics import normalize_text

_QUESTION_HEADING = re.compile(r"^## (Q-\d+)\s*$")
_SUBHEADING = re.compile(r"^### (.+?)\s*$")
_FENCE = re.compile(r"^(`{3,})")
_ERROR_LINE = re.compile(r"^- Error: (.*)$")

QUESTION_HEADING = "Question"
GOLD_HEADING = "Gold answer"
GENERATED_HEADING = "Generated answer"
PROMPT_HEADING = "Generator prompt"
# Fenced blocks captured per section, by heading. The question text and generator prompt are optional.
_CAPTURED = {QUESTION_HEADING: "question", GOLD_HEADING: "gold", GENERATED_HEADING: "generated",
             PROMPT_HEADING: "prompt"}
# Answer texts the report runner writes instead of a model answer.
GENERATION_FAILURE_PREFIXES = {
    "The answer model could not be reached": "generation_error",
    "Answer generation is not configured": "generation_not_configured",
}

STATUS_SCORED = "scored"
STATUS_MISSING_BASELINE = "missing_in_baseline"
STATUS_MISSING_CONTEXT = "missing_in_context"
STATUS_MALFORMED = "malformed"
STATUS_GOLD_MISMATCH = "gold_mismatch"
STATUS_NO_GOLD = "no_gold"


@dataclass
class Section:
    qid: str
    gold: str | None = None
    generated: str | None = None
    error: str | None = None
    problems: list[str] = field(default_factory=list)
    question: str | None = None
    prompt: str | None = None  # the prompt the answer model received, when the report includes it


@dataclass
class ParsedReport:
    sections: dict[str, Section]
    problems: list[str]


def parse_report(text: str) -> ParsedReport:
    """Question sections keyed by ID. Headings inside fenced blocks are ignored."""
    sections: dict[str, Section] = {}
    problems: list[str] = []
    current: Section | None = None
    heading: str | None = None
    fence: str | None = None
    capture: list[str] | None = None
    capture_target: str | None = None
    awaiting_fence = False

    def finish_capture() -> None:
        nonlocal capture, capture_target
        if current is not None and capture is not None and capture_target is not None:
            value = "\n".join(capture)
            if getattr(current, capture_target) is not None:
                current.problems.append(f"duplicate '{heading}' block")
            else:
                setattr(current, capture_target, value)
        capture = None
        capture_target = None

    for line in text.splitlines():
        if fence is not None:
            if line == fence:
                fence = None
                finish_capture()
            elif capture is not None:
                capture.append(line)
            continue

        opened = _FENCE.match(line)
        if opened:
            fence = opened.group(1)
            if awaiting_fence and current is not None:
                capture = []
                capture_target = _CAPTURED[heading]
            awaiting_fence = False
            continue

        question = _QUESTION_HEADING.match(line)
        if question:
            _close_section(current, awaiting_fence, heading)
            qid = question.group(1)
            current = Section(qid)
            heading = None
            awaiting_fence = False
            if qid in sections:
                problems.append(f"{qid}: duplicate question section")
                sections[qid].problems.append("duplicate question section")
                current = Section(qid)  # parsed but not stored; the first section is marked bad
            else:
                sections[qid] = current
            continue
        if line.startswith("## "):
            _close_section(current, awaiting_fence, heading)
            current = None
            heading = None
            awaiting_fence = False
            continue
        if current is None:
            continue

        sub = _SUBHEADING.match(line)
        if sub:
            if awaiting_fence:
                current.problems.append(f"no fenced block under '{heading}'")
            heading = sub.group(1)
            awaiting_fence = heading in _CAPTURED
            continue
        error = _ERROR_LINE.match(line)
        if error and heading is None:
            current.error = error.group(1)
            continue
        if awaiting_fence and line.strip():
            current.problems.append(f"no fenced block under '{heading}'")
            awaiting_fence = False

    if fence is not None:
        if current is not None:
            current.problems.append("unterminated fenced block")
        problems.append("report ends inside a fenced block")
    _close_section(current, awaiting_fence, heading)

    for section in sections.values():
        if section.gold is None:
            section.problems.append(f"missing '{GOLD_HEADING}' block")
        if section.generated is None:
            section.problems.append(f"missing '{GENERATED_HEADING}' block")
        section.problems = list(dict.fromkeys(section.problems))
    return ParsedReport(sections, problems)


def _close_section(section: Section | None, awaiting_fence: bool, heading: str | None) -> None:
    if section is not None and awaiting_fence:
        section.problems.append(f"no fenced block under '{heading}'")


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_text(prediction) == normalize_text(gold)


def token_f1(prediction: str, gold: str) -> float:
    pred = normalize_text(prediction).split()
    ref = normalize_text(gold).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def answer_flags(section: Section) -> list[str]:
    flags = []
    if section.error:
        flags.append("row_error")
    answer = section.generated or ""
    if not answer.strip():
        flags.append("empty_answer")
    for prefix, flag in GENERATION_FAILURE_PREFIXES.items():
        if answer.strip().startswith(prefix):
            flags.append(flag)
    return flags


@dataclass
class PairedRow:
    qid: str
    status: str
    gold: str | None
    baseline: str | None
    context: str | None
    em_baseline: bool | None = None
    em_context: bool | None = None
    f1_baseline: float | None = None
    f1_context: float | None = None
    notes: list[str] = field(default_factory=list)


def qid_sort_key(qid: str) -> tuple[int, str]:
    digits = qid.split("-", 1)[-1]
    return (int(digits), qid) if digits.isdigit() else (sys.maxsize, qid)


def pair_reports(baseline: ParsedReport, context: ParsedReport) -> list[PairedRow]:
    rows: list[PairedRow] = []
    for qid in sorted(set(baseline.sections) | set(context.sections), key=qid_sort_key):
        b = baseline.sections.get(qid)
        c = context.sections.get(qid)
        if b is None or c is None:
            present = b or c
            rows.append(PairedRow(qid, STATUS_MISSING_BASELINE if b is None else STATUS_MISSING_CONTEXT,
                                  present.gold if present else None,
                                  b.generated if b else None, c.generated if c else None))
            continue
        row = PairedRow(qid, STATUS_SCORED, b.gold, b.generated, c.generated)
        if b.problems or c.problems:
            row.status = STATUS_MALFORMED
            row.notes += [f"baseline: {p}" for p in b.problems] + [f"context: {p}" for p in c.problems]
            rows.append(row)
            continue
        if (b.gold or "").strip() != (c.gold or "").strip():
            row.status = STATUS_GOLD_MISMATCH
            row.notes.append(f"context gold: {c.gold!r}")
            rows.append(row)
            continue
        if not normalize_text(b.gold or ""):
            row.status = STATUS_NO_GOLD
            rows.append(row)
            continue
        row.notes += [f"baseline: {f}" for f in answer_flags(b)] + [f"context: {f}" for f in answer_flags(c)]
        row.em_baseline = exact_match(b.generated or "", b.gold or "")
        row.em_context = exact_match(c.generated or "", b.gold or "")
        row.f1_baseline = token_f1(b.generated or "", b.gold or "")
        row.f1_context = token_f1(c.generated or "", b.gold or "")
        rows.append(row)
    return rows


@dataclass
class Summary:
    n_scored: int
    em_baseline: float | None
    em_context: float | None
    f1_baseline: float | None
    f1_context: float | None
    status_counts: dict[str, int]
    flag_counts: dict[str, int]

    @property
    def em_delta_pp(self) -> float | None:
        if self.em_baseline is None or self.em_context is None:
            return None
        return (self.em_context - self.em_baseline) * 100

    @property
    def f1_delta(self) -> float | None:
        if self.f1_baseline is None or self.f1_context is None:
            return None
        return self.f1_context - self.f1_baseline


def summarize(rows: list[PairedRow]) -> Summary:
    scored = [r for r in rows if r.status == STATUS_SCORED]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    flags = Counter(note for r in scored for note in r.notes)
    return Summary(
        n_scored=len(scored),
        em_baseline=mean([float(r.em_baseline) for r in scored]),
        em_context=mean([float(r.em_context) for r in scored]),
        f1_baseline=mean([r.f1_baseline for r in scored]),
        f1_context=mean([r.f1_context for r in scored]),
        status_counts=dict(Counter(r.status for r in rows)),
        flag_counts=dict(flags),
    )


def _cell(value: str | None, width: int) -> str:
    if value is None:
        return "(missing)"
    text = " ".join(value.split()) or "(empty)"
    if len(text) > width:
        text = text[: max(1, width - 1)] + "…"
    return text.replace("|", "\\|")


def _num(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def render(rows: list[PairedRow], summary: Summary, baseline: ParsedReport, context: ParsedReport,
           *, baseline_path: str, context_path: str, width: int) -> str:
    lines = [
        "# LoCoMo answer comparison",
        "",
        f"- Baseline: `{baseline_path}` ({len(baseline.sections)} question sections)",
        f"- Context: `{context_path}` ({len(context.sections)} question sections)",
        f"- Scored (paired) questions: {summary.n_scored}",
        "- Status counts: " + (", ".join(f"{k}={v}" for k, v in sorted(summary.status_counts.items())) or "none"),
        "",
        "| Metric | Baseline | Context | Difference |",
        "|---|---:|---:|---:|",
        f"| Exact match | {_pct(summary.em_baseline)} | {_pct(summary.em_context)} | "
        f"{'-' if summary.em_delta_pp is None else f'{summary.em_delta_pp:+.1f} pp'} |",
        f"| Mean token F1 | {_num(summary.f1_baseline)} | {_num(summary.f1_context)} | "
        f"{'-' if summary.f1_delta is None else f'{summary.f1_delta:+.3f}'} |",
    ]
    if summary.flag_counts:
        lines += ["", "Flags on scored questions (scored as written, not excluded): "
                  + ", ".join(f"{k}={v}" for k, v in sorted(summary.flag_counts.items()))]
    report_problems = [f"baseline: {p}" for p in baseline.problems] + [f"context: {p}" for p in context.problems]
    if report_problems:
        lines += ["", "Report-level problems:"] + [f"- {p}" for p in report_problems]
    lines += [
        "",
        "| Question | Status | Gold | Baseline answer | Context answer | F1 base | F1 ctx | Notes |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.qid} | {r.status} | {_cell(r.gold, width)} | {_cell(r.baseline, width)} | "
            f"{_cell(r.context, width)} | {_num(r.f1_baseline)} | {_num(r.f1_context)} | "
            f"{_cell('; '.join(r.notes), width * 2) if r.notes else ''} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/score_locomo_answers.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", type=Path, required=True, help="baseline LoCoMo report (.md)")
    parser.add_argument("--context", type=Path, required=True, help="contextual LoCoMo report (.md)")
    parser.add_argument("--width", type=int, default=60, help="max characters per answer cell (default 60)")
    args = parser.parse_args(argv)
    if args.width < 4:
        parser.error("--width must be >= 4")

    reports = {}
    for name, path in (("baseline", args.baseline), ("context", args.context)):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot read {name} report {path}: {exc}", file=sys.stderr)
            return 2
        parsed = parse_report(text)
        if not parsed.sections:
            print(f"error: no '## Q-NNN' question sections found in {name} report {path}", file=sys.stderr)
            return 2
        reports[name] = parsed

    rows = pair_reports(reports["baseline"], reports["context"])
    summary = summarize(rows)
    sys.stdout.write(render(rows, summary, reports["baseline"], reports["context"],
                            baseline_path=str(args.baseline), context_path=str(args.context), width=args.width))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
