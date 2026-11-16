#!/usr/bin/env python3
"""Message-level retrieval baselines: saved V1 results and a raw-chunk control. No Ollama call is made.

Reads the frozen V1 results and gold sets without changing them, and writes to a new directory. ``msgs`` in
gold.json is a list of evidence groups per question: any message satisfies a group, and full evidence coverage
needs every group. A result is a retrieved memory (V1) or a chunk (control), so per-result metrics are also
reported at equal message budgets, because a chunk shows four messages and a V1 memory usually one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from continuator_memory.chunking import chunk_conversation
from continuator_memory.embedding import Embedder, SentenceTransformerEmbedder
from continuator_memory.models import Conversation, parse_conversation
from continuator_memory.retrieval import Retriever, top_k
from continuator_memory.store import Store
from continuator_memory.v2.lexical import fuse_scores, lexical_overlap
from continuator_memory.v2.windowing import build_windows

ROOT = Path.home() / "continuator_eval_artifacts"
V1_DIR = ROOT / "v1_eval/retrieval_eval"
CONVERSATION = Path.home() / "Downloads/continuator_memory_test_conversation.json"
QUESTIONS = Path.home() / "Downloads/continuator_memory_test_questions.md"
RESULT_KS = (5, 10)
MESSAGE_BUDGETS = (4, 8, 12, 20)  # Multiples of 4, so whole chunks fit and no order inside a chunk is assumed.
CHUNK_SIZE = 4
EXPECTED_V1_COVERAGE_AT_5 = {"v1_baseline": 13, "v1_hybrid": 15}  # From the frozen evaluation, as a check.


def message_number(message_id: str) -> int:
    return int(message_id.rsplit(":m", 1)[1][:4])


def score_question(results: list[list[int]], groups: list[list[int]]) -> dict:
    """``results``: message numbers per ranked result, best first."""
    gold = {m for group in groups for m in group}
    out: dict = {"groups": len(groups)}
    for k in RESULT_KS:
        seen = {m for result in results[:k] for m in result}
        hit = sum(any(m in seen for m in group) for group in groups)
        out[f"recall@{k}"] = hit / len(groups)
        out[f"coverage@{k}"] = hit == len(groups)
        out[f"messages@{k}"] = len(seen)
    out["first_rank"] = next((i for i, result in enumerate(results[:max(RESULT_KS)], 1) if gold & set(result)), None)
    out["rr"] = 1 / out["first_rank"] if out["first_rank"] else 0.0
    ordered: list[int] = []
    for result in results:
        ordered += [m for m in result if m not in ordered]
    for budget in MESSAGE_BUDGETS:
        # A result that would pass the budget is left out whole, so nothing depends on order inside a chunk.
        seen_budget: set[int] = set()
        for result in results:
            extra = set(result) - seen_budget
            if len(seen_budget) + len(extra) > budget:
                break
            seen_budget |= extra
        hit = sum(any(m in seen_budget for m in group) for group in groups)
        out[f"budget{budget}_recall"] = hit / len(groups)
        out[f"budget{budget}_coverage"] = hit == len(groups)
        out[f"budget{budget}_short"] = len(seen_budget) < budget and len(ordered) <= len(seen_budget)
    return out


def aggregate(rows: dict[str, dict]) -> dict:
    n = len(rows)
    out: dict = {"questions": n, "mrr@10": sum(r["rr"] for r in rows.values()) / n}
    for k in RESULT_KS:
        out[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows.values()) / n
        out[f"coverage@{k}"] = sum(r[f"coverage@{k}"] for r in rows.values())
        out[f"mean_messages@{k}"] = sum(r[f"messages@{k}"] for r in rows.values()) / n
    for budget in MESSAGE_BUDGETS:
        out[f"budget{budget}_recall"] = sum(r[f"budget{budget}_recall"] for r in rows.values()) / n
        out[f"budget{budget}_coverage"] = sum(r[f"budget{budget}_coverage"] for r in rows.values())
        out[f"budget{budget}_short_questions"] = sum(r[f"budget{budget}_short"] for r in rows.values())
    return out


def control_results(question: str, retriever: Retriever, conversation_id: str, n_chunks: int,
                    lexical_weight: float) -> dict[str, list[dict]]:
    ranked = retriever.retrieve(question, n_chunks, conversation_id)
    fused = [fuse_scores(c.score, lexical_overlap(question, c.text), weight=lexical_weight) for c in ranked]
    order = top_k(np.array(fused), [c.chunk_id for c in ranked], n_chunks)
    row = lambda c, score: {"chunk_id": c.chunk_id, "score": score, "similarity": c.score,
                            "msgs": [message_number(m) for m in c.source_message_ids]}
    return {"control_cosine": [row(c, c.score) for c in ranked],
            "control_hybrid": [row(ranked[i], fused[i]) for i in order]}


def build_control(conversation: Conversation, store: Store, embedder: Embedder) -> tuple[Retriever, int]:
    """Stores 16 non-overlapping chunks. Their message groups must match the extraction windows."""
    chunks = chunk_conversation(conversation, CHUNK_SIZE, 0)
    windows = build_windows(conversation, CHUNK_SIZE, 0)
    if [c.source_message_ids for c in chunks] != [w.target_ids for w in windows]:
        raise SystemExit("chunks do not match the evaluated windows")
    store.save_conversation(conversation, chunks, embedder.embed([c.text for c in chunks]), embedder.model_name)
    return Retriever(store, embedder), len(chunks)


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=ROOT / "knowledge_eval")
    args = parser.parse_args(argv)

    gold = json.loads((V1_DIR / "gold.json").read_text(encoding="utf-8"))
    saved = json.loads((V1_DIR / "retrieval.json").read_text(encoding="utf-8"))
    conversation = parse_conversation(json.loads(CONVERSATION.read_text(encoding="utf-8")))
    if len(conversation.messages) != 62:
        raise SystemExit(f"expected 62 messages, found {len(conversation.messages)}")
    questions = {n: q.strip() for n, q in re.findall(r"^(\d+)\. (.+)$", QUESTIONS.read_text(encoding="utf-8"), re.M)}
    if len(questions) != 20 or any(saved["q"][n]["question"] != q for n, q in questions.items()):
        raise SystemExit("questions do not match the frozen V1 evaluation")

    out = args.output_root / f"baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    settings = saved["settings"]
    embedder = SentenceTransformerEmbedder(settings["model"])
    store = Store(str(out / "control.db"))
    retriever, n_chunks = build_control(conversation, store, embedder)

    def v1_rows(key_all: str) -> dict[str, list[list[int]]]:
        return {n: [[message_number(m) for m in match["msgs"]] for match in saved["q"][n][key_all]["matches"]] for n in questions}

    ranked: dict[str, dict[str, list[list[int]]]] = {"v1_baseline": v1_rows("baseline_all"), "v1_hybrid": v1_rows("hybrid_k10")}
    control: dict[str, dict] = {}
    for n, question in questions.items():
        control[n] = control_results(question, retriever, conversation.conversation_id, n_chunks, settings["lexical"])
    store.close()
    for system in ("control_cosine", "control_hybrid"):
        ranked[system] = {n: [r["msgs"] for r in control[n][system]] for n in questions}

    per_question = {system: {n: score_question(ranked[system][n], gold["q"][n]["msgs"]) for n in questions} for system in ranked}
    summary = {system: aggregate(rows) for system, rows in per_question.items()}
    for system, expected in EXPECTED_V1_COVERAGE_AT_5.items():
        if summary[system]["coverage@5"] != expected:
            raise SystemExit(f"{system} coverage@5 is {summary[system]['coverage@5']}, frozen evaluation found {expected}")

    results = {"inputs": {"gold_sha256": hashlib.sha256((V1_DIR / "gold.json").read_bytes()).hexdigest(),
                          "retrieval_sha256": hashlib.sha256((V1_DIR / "retrieval.json").read_bytes()).hexdigest(),
                          "conversation_sha256": hashlib.sha256(CONVERSATION.read_bytes()).hexdigest()},
               "settings": {**settings, "chunk_size": CHUNK_SIZE, "overlap": 0, "chunks": n_chunks}, "summary": summary,
               "per_question": per_question, "control_rankings": control}
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    shutil.copy2(__file__, out)
    shutil.copy2(V1_DIR / "gold.json", out / "gold_used.json")
    (out / "report.md").write_text(render(results, questions), encoding="utf-8")
    print((out / "report.md").read_text(encoding="utf-8"))
    print(f"ARTIFACTS: {out}")
    return 0


def render(results: dict, questions: dict[str, str]) -> str:
    summary, per = results["summary"], results["per_question"]
    systems = list(summary)
    lines = ["# Message-level retrieval baselines", "", "| Metric | " + " | ".join(systems) + " |", "|---" * (len(systems) + 1) + "|"]
    def row(label, key, fmt):
        lines.append(f"| {label} | " + " | ".join(fmt(summary[s][key]) for s in systems) + " |")
    row("Recall@5", "recall@5", percent)
    row("Recall@10", "recall@10", percent)
    row("MRR@10", "mrr@10", lambda v: f"{v:.3f}")
    row("Coverage@5 (of 20)", "coverage@5", str)
    row("Coverage@10 (of 20)", "coverage@10", str)
    row("Mean messages shown @5", "mean_messages@5", lambda v: f"{v:.1f}")
    row("Mean messages shown @10", "mean_messages@10", lambda v: f"{v:.1f}")
    for budget in MESSAGE_BUDGETS:
        row(f"Recall, {budget} messages", f"budget{budget}_recall", percent)
        row(f"Coverage, {budget} messages", f"budget{budget}_coverage", str)
        row(f"Questions with too few results, {budget} messages", f"budget{budget}_short_questions", str)
    lines += ["", "## Per question: recall@5, then coverage@5 (Y/N), first relevant rank", "", "| # | " + " | ".join(systems) + " |", "|---" * (len(systems) + 1) + "|"]
    for n in questions:
        cells = [f"{per[s][n]['recall@5']:.2f} {'Y' if per[s][n]['coverage@5'] else 'N'} {per[s][n]['first_rank'] or '-'}" for s in systems]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
