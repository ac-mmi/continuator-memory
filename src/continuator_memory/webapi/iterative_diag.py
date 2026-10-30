"""Compare baseline, multi-query, and iterative retrieval on LoCoMo annotations.

Gold ``qa[].evidence`` is read from the JSON file for scoring only. It is never
passed into retrieval, planning, or evaluation.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from continuator_memory.evaluation import metrics
from continuator_memory.webapi.evidence_eval import heuristic_evaluation
from continuator_memory.webapi.workspace import Workspace

DEFAULT_DB = Path("data/locomo-conv26.db")
DEFAULT_LOCOMO = Path("locomo_conv26.json")
CAROLINE_Q = "What fields would Caroline be likely to pursue in her education?"
MODES = ("baseline", "multiquery", "iterative")
KS = (5, 15)


def load_locomo_qa(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    record = raw[0] if isinstance(raw, list) else raw
    rows = []
    for item in record.get("qa") or []:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        evidence = item.get("evidence")
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
            continue
        rows.append({
            "question": question,
            "answer": item.get("answer"),
            "evidence": list(dict.fromkeys(evidence)),
            "category": item.get("category"),
        })
    return rows


def _score(selected: list[str], expected: list[str], conversation_id: str) -> dict[str, Any]:
    class Row:
        def __init__(self, mid: str):
            self.conversation_id = conversation_id
            self.source_message_ids = (mid,)

    ranked = [Row(mid) for mid in selected]
    rank = metrics.first_expected_rank(ranked, conversation_id, expected)
    return {
        "firstExpectedRank": rank,
        "reciprocalRank": metrics.reciprocal_rank(rank),
        "recallAt": {
            "5": metrics.recall_at_k(ranked, conversation_id, expected, 5) if ranked else 0.0,
            "10": metrics.recall_at_k(ranked, conversation_id, expected, min(10, max(len(ranked), 1))),
            "15": metrics.recall_at_k(ranked, conversation_id, expected, min(15, max(len(ranked), 1))),
        },
        "hitAt": {
            "5": metrics.hit_at_k(rank, 5),
            "10": metrics.hit_at_k(rank, 10),
            "15": metrics.hit_at_k(rank, 15),
        },
    }


def _sample_rows(qa: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for row in qa:
        if row["question"].startswith("What fields would Caroline"):
            sample.append(row)
            break
    for row in qa:
        if row in sample:
            continue
        sample.append(row)
        if len(sample) >= 9:
            break
    if not any(row["question"] == CAROLINE_Q for row in sample):
        gold = next((row for row in qa if row["question"].startswith("What fields would Caroline")), None)
        if gold:
            sample.insert(0, {**gold, "question": CAROLINE_Q})
    return sample


def run_modes(workspace: Workspace, question: str) -> dict[str, Any]:
    out = {}
    for mode in MODES:
        started = time.perf_counter()
        result = workspace.ask(question, retrieval_mode=mode, generate=False)
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        evidence = result["answer"]["evidence"]
        diagnostics = evidence.get("diagnostics") or {}
        out[mode] = {
            "selectedMessageIds": list(evidence.get("selectedMessageIds") or []),
            "stopReason": evidence.get("stopReason") or diagnostics.get("stopReason"),
            "roundCount": diagnostics.get("roundCount"),
            "latencyMs": elapsed,
            "diagLatency": (evidence.get("retrievalComparison") or {}).get("latencyMs"),
        }
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_summary(rows: list[dict[str, Any]], k: int) -> None:
    print(f"\n--- summary k={k} n={len(rows)} ---")
    for mode in MODES:
        recalls = [row["scores"][mode]["recallAt"][str(k)] for row in rows]
        rrs = [row["scores"][mode]["reciprocalRank"] for row in rows]
        hits = [row["scores"][mode]["hitAt"][str(k)] for row in rows]
        latencies = [row["runs"][mode]["latencyMs"] for row in rows]
        stops = Counter(row["runs"][mode]["stopReason"] for row in rows)
        rounds = [row["runs"][mode]["roundCount"] for row in rows if row["runs"][mode]["roundCount"] is not None]
        print(
            f"  {mode:10} recall@{k}={_mean(recalls):.2f} "
            f"hit@{k}={_mean(hits):.2f} mrr={_mean(rrs):.3f} "
            f"ms={_mean(latencies):.0f} "
            f"rounds={_mean([float(x) for x in rounds]):.2f} "
            f"stops={dict(stops)}"
        )
    print("  regressions vs baseline (lower recall or worse first rank):")
    found = False
    for row in rows:
        base = row["scores"]["baseline"]
        for mode in ("multiquery", "iterative"):
            other = row["scores"][mode]
            recall_drop = other["recallAt"][str(k)] < base["recallAt"][str(k)]
            rank_base = base["firstExpectedRank"]
            rank_other = other["firstExpectedRank"]
            worse_rank = (
                rank_other is None and rank_base is not None
            ) or (
                rank_other is not None and rank_base is not None and rank_other > rank_base
            )
            if recall_drop or worse_rank:
                found = True
                print(
                    f"    {mode}: {row['question'][:72]} "
                    f"recall {base['recallAt'][str(k)]:.2f}->{other['recallAt'][str(k)]:.2f} "
                    f"first {rank_base}->{rank_other}"
                )
    if not found:
        print("    none")


def main() -> int:
    if not DEFAULT_DB.is_file() or not DEFAULT_LOCOMO.is_file():
        print("missing locomo db or json; skip diagnostic")
        return 0
    qa = load_locomo_qa(DEFAULT_LOCOMO)
    sample = _sample_rows(qa)
    workspace = Workspace.from_sqlite(DEFAULT_DB, k=15)
    workspace.set_evaluator(heuristic_evaluation)
    cid = workspace.conversation.conversation_id
    print(f"conversation={cid} evaluator=heuristic n_qa={len(sample)} ks={list(KS)}")
    print("gold qa evidence is used for scoring only")
    for k in KS:
        workspace.k = k
        print(f"\n========== k={k} ==========")
        scored: list[dict[str, Any]] = []
        for row in sample:
            print(f"\nQ: {row['question']}")
            print(f"  gold_evidence={row['evidence']}  (eval only)")
            runs = run_modes(workspace, row["question"])
            scores = {mode: _score(payload["selectedMessageIds"], row["evidence"], cid) for mode, payload in runs.items()}
            scored.append({"question": row["question"], "runs": runs, "scores": scores})
            for mode, payload in runs.items():
                score = scores[mode]
                print(
                    f"  {mode:10} recall@{k}={score['recallAt'][str(k)]:.2f} "
                    f"first={score['firstExpectedRank']} rr={score['reciprocalRank']:.3f} "
                    f"stop={payload['stopReason']} rounds={payload['roundCount']} "
                    f"ms={payload['latencyMs']} selected={payload['selectedMessageIds'][:8]}"
                )
        _print_summary(scored, k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
