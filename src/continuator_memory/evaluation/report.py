"""Write evaluation results as JSON and a Markdown report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from continuator_memory.evaluation.harness import RETRIEVAL_KS


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _cell(text: str | None, limit: int = 120) -> str:
    if text is None:
        return "—"
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _retrieval_table(summaries: dict[str, dict[str, Any]]) -> list[str]:
    ks = [str(k) for k in RETRIEVAL_KS]
    lines = [
        "| Group | n | " + " | ".join(f"Recall@{k}" for k in ks) + " | "
        + " | ".join(f"Hit@{k}" for k in ks) + " | MRR |",
        "|---" * (3 + 2 * len(ks)) + "|",
    ]
    for name, s in summaries.items():
        if not s["n"]:
            lines.append(f"| {name} | 0 | " + " | ".join("—" for _ in range(2 * len(ks) + 1)) + " |")
            continue
        lines.append(
            f"| {name} | {s['n']} | "
            + " | ".join(_fmt(s["recall_at"][k]) for k in ks)
            + " | "
            + " | ".join(_fmt(s["hit_at"][k]) for k in ks)
            + f" | {_fmt(s['mrr'])} |"
        )
    return lines


def _answer_table(summaries: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Group | n | Correct | Accuracy | False abstention | Stale mention | Errors |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        if not s["n"]:
            lines.append(f"| {name} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {s['n']} | {s['n_correct']} | {_fmt(s['accuracy'])} | "
            f"{_fmt(s['false_abstention_rate'])} | {_fmt(s['stale_mention_rate'])} | "
            f"{s['n_errors']} |"
        )
    return lines


def _diagnostics_section(diag: dict[str, Any] | None) -> list[str]:
    if not diag:
        return []
    cs, ms = diag["current_state"], diag["multi_source"]
    lines = [
        "",
        "### Diagnostics",
        "",
        "Diagnostics only; none of these change correctness. Coverage is the fraction of "
        "expected source messages present in the LLM's context.",
        "",
        "| Metric | Value | n |",
        "|---|---|---|",
        f"| Current-state accuracy (questions with stale values) | {_fmt(cs['accuracy'])} | "
        f"{cs['n']} |",
        f"| Current-state accuracy without stale mention | {_fmt(cs['clean_accuracy'])} | "
        f"{cs['n']} |",
        f"| Multi-source accuracy (≥2 expected sources) | {_fmt(ms['accuracy'])} | {ms['n']} |",
        f"| Multi-source context coverage | {_fmt(ms['context_coverage'])} | {ms['n']} |",
        f"| Context coverage (answerable) | {_fmt(diag['context_coverage'])} | — |",
    ]
    mem = diag.get("memory")
    if mem:
        lines += [
            f"| Memory source coverage (answerable) | {_fmt(mem['source_coverage'])} | "
            f"{mem['n']} |",
            f"| Memory source coverage (multi-source) | {_fmt(mem['multi_source_coverage'])} | "
            f"{ms['n']} |",
            f"| Questions with relationships in context | "
            f"{_fmt(mem['questions_with_relationships'])} | {mem['n']} |",
            f"| Mean memories in context | {_fmt(mem['mean_memories_in_context'], 1)} | "
            f"{mem['n']} |",
            f"| Mean superseded memories in context | "
            f"{_fmt(mem['mean_superseded_in_context'], 1)} | {mem['n']} |",
        ]
    return lines


def _memory_section(memory: dict[str, Any] | None) -> list[str]:
    if not memory:
        return []
    actions = memory["consolidation_actions"]
    fmt_actions = lambda d: ", ".join(f"{k} {v}" for k, v in d.items()) or "—"  # noqa: E731
    return [
        "",
        "## Memory formation",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Extraction model | `{memory['extraction_model']}` |",
        f"| Extraction status (messages) | "
        f"{', '.join(f'{k} {v}' for k, v in sorted(memory['extraction_status'].items()))} |",
        f"| Extracted memories / relationships (after validation) | "
        f"{memory['extracted_memories']} / {memory['extracted_relationships']} |",
        f"| Memory consolidation actions | {fmt_actions(actions['memory'])} |",
        f"| Relationship consolidation actions | {fmt_actions(actions['relationship'])} |",
        f"| Entities | {memory['entities']} |",
        f"| Memories stored (current / superseded) | {memory['memories']} "
        f"({memory['memories_current']} / {memory['memories_superseded']}) |",
        f"| Relationships stored (current / superseded) | {memory['relationships']} "
        f"({memory['relationships_current']} / {memory['relationships_superseded']}) |",
        f"| Extraction latency | {memory['formation_ms']['extraction'] / 1000:.1f} s |",
        f"| Consolidation latency | {memory['formation_ms']['consolidation']:.0f} ms |",
    ]


def render_markdown(results: dict[str, Any]) -> str:
    run = results["run"]
    cfg = run["config"]
    agg = results["aggregate"]
    ret = agg["retrieval"]
    ans = agg["answer"]
    lines = [
        f"# Evaluation report: {run['dataset']['dataset_id']}",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- Started: {run['started_at']}  Finished: {run['finished_at']}",
        f"- Dataset: `{run['dataset']['path']}` (sha256 `{run['dataset']['sha256']}`)",
        f"- Conversations: {run['dataset']['n_conversations']}; questions evaluated: "
        f"{run['dataset']['n_questions_evaluated']} of {run['dataset']['n_questions_total']}",
        f"- Git commit: `{run['environment']['git_commit']}`; package "
        f"{run['environment']['package_version']}; Python {run['environment']['python']}",
        f"- Ingest: {results['timing']['ingest_ms']:.0f} ms; total: "
        f"{results['timing']['total_ms'] / 1000:.1f} s",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | `{v}` |" for k, v in cfg.items()]

    lines += [
        "",
        "## Retrieval",
        "",
        "Scored on answerable questions only. A chunk is relevant if it belongs to the "
        "question's conversation and contains an expected source message. Recall@k is the "
        "fraction of expected source messages covered by the top-k chunks; Hit@k is whether any "
        "is covered; MRR uses the rank of the first relevant chunk.",
        "",
    ]
    lines += _retrieval_table({"**overall**": ret["overall"], **ret["by_category"]})

    lines += ["", "## Answers", ""]
    if cfg.get("retrieval_only"):
        lines.append("Generation skipped (`--retrieval-only`).")
    else:
        o = ans["overall"]
        lines += [
            "Deterministic scoring: answerable questions are correct when the normalized answer "
            "contains an accepted phrase; unanswerable questions are correct when the answer "
            "abstains. Stale mention = the answer mentions a superseded or corrected value "
            "(diagnostic only, not part of correctness).",
            "",
            f"- Accuracy: **{_fmt(o['accuracy'])}** ({o['n_correct']}/{o['n']})",
            f"- Answerable accuracy: {_fmt(o['answerable']['accuracy'])} "
            f"(n={o['answerable']['n']})",
            f"- Abstention accuracy: {_fmt(o['abstention']['accuracy'])} "
            f"(n={o['abstention']['n']})",
        ]
        by_ret = o.get("answerable_by_retrieval")
        if by_ret:
            lines += [
                f"- Answerable accuracy when an expected source was in the top-{by_ret['k']} "
                f"context: {_fmt(by_ret['retrieved']['accuracy'])} "
                f"(n={by_ret['retrieved']['n']}); when not: "
                f"{_fmt(by_ret['not_retrieved']['accuracy'])} (n={by_ret['not_retrieved']['n']})",
            ]
        lines.append("")
        lines += _answer_table({"**overall**": o, **ans["by_category"]})
        lines += _diagnostics_section(agg.get("diagnostics"))

    lines += _memory_section(results.get("memory"))
    lines += [
        "",
        "## Per question",
        "",
        "| ID | Category | Scope | First rank | RR | Recall@5 | Answer correct | Latency ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for q in results["questions"]:
        r, a = q["retrieval"], q["answer"]
        lines.append(
            f"| {q['question_id']} | {q['category']} | {q['scope']} | "
            f"{r['first_expected_rank'] if q['answerable'] else 'n/a'} | "
            f"{_fmt(r['reciprocal_rank']) if q['answerable'] else 'n/a'} | "
            f"{_fmt(r['recall_at'].get('5')) if q['answerable'] else 'n/a'} | "
            f"{'—' if a['correct'] is None else ('yes' if a['correct'] else 'no')} | "
            f"{q['latency_ms']:.0f} |"
        )

    failures = [
        q
        for q in results["questions"]
        if (q["answerable"] and not q["retrieval"]["hit_at"].get("5"))
        or q["answer"]["correct"] is False
    ]
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None.")
    for q in failures:
        r, a = q["retrieval"], q["answer"]
        top = [c["chunk_id"] for c in r["retrieved"][:5]]
        lines += [
            f"### {q['question_id']} ({q['category']})",
            "",
            f"- Question: {_cell(q['question'], 300)}",
            f"- Expected answer: {_cell(a['expected_answer'] or '(abstain)', 300)}",
            f"- Generated answer: {_cell(a['generated_answer'], 300)}",
            f"- Expected sources: {', '.join(r['expected_source_message_ids']) or '—'} "
            f"(conversation `{q['conversation_id']}`)",
            f"- First expected rank: {r['first_expected_rank']}; top-5 chunks: "
            f"{', '.join(top) or '—'}",
        ]
        if a["error"]:
            lines.append(f"- Error: {_cell(a['error'], 300)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_results(results: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = results["run"]["run_id"]
    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(results), encoding="utf-8")
    return json_path, md_path
