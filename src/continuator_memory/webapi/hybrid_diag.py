"""Compare Baseline and lexical Hybrid on LoCoMo QA.

Gold ``qa[].evidence`` is scoring-only. It is never passed into retrieval.
Hybrid stays a non-default workspace mode. Production ranking is unchanged.

Default sample is every scorable conv-26 question, in file order. Questions are
not chosen by Hybrid success. ``--sample iterative`` keeps the original 10-question
diagnostic subset.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from typing import Any

from continuator_memory.v2.lexical import DEFAULT_LEXICAL_WEIGHT, LEXICAL_WEIGHT_ENV, load_lexical_weight
from continuator_memory.v2.memory_retrieval import RetrievalConfig
from continuator_memory.webapi.iterative_diag import (
    DEFAULT_DB,
    DEFAULT_LOCOMO,
    KS,
    _mean,
    _sample_rows,
    _score,
    load_locomo_qa,
)
from continuator_memory.webapi.multiquery import build_memory_retriever
from continuator_memory.webapi.workspace import Workspace

MODES = ("baseline", "hybrid")
DEPTH = 32
SAMPLES = ("all", "iterative")


def conversation_message_ids(workspace: Workspace) -> set[str]:
    return {message.message_id for message in workspace.conversation.messages}


def qa_coverage(qa: list[dict[str, Any]], message_ids: set[str]) -> dict[str, Any]:
    """Inventory QA before sampling. Exclusion is coverage-only, not retrieval success."""
    empty: list[str] = []
    unmatched: list[str] = []
    scorable: list[dict[str, Any]] = []
    categories: Counter[Any] = Counter()
    for row in qa:
        evidence = [item for item in row["evidence"] if item]
        if not evidence:
            empty.append(row["question"])
            continue
        if not any(item in message_ids for item in evidence):
            unmatched.append(row["question"])
            continue
        scorable.append(row)
        categories[row.get("category")] += 1
    return {
        "total": len(qa),
        "empty_evidence": empty,
        "unmatched_gold": unmatched,
        "scorable": scorable,
        "categories": dict(sorted(categories.items(), key=lambda item: str(item[0]))),
    }


def select_eval_rows(qa: list[dict[str, Any]], coverage: dict[str, Any], sample: str) -> list[dict[str, Any]]:
    if sample == "iterative":
        return _sample_rows(qa)
    if sample != "all":
        raise ValueError("sample must be 'all' or 'iterative'")
    return list(coverage["scorable"])


def gold_ranks(selected: list[str], expected: list[str]) -> dict[str, int | None]:
    """1-based position of each gold id in ``selected``; scoring only."""
    index = {message_id: rank for rank, message_id in enumerate(selected, 1)}
    return {message_id: index.get(message_id) for message_id in expected}


def format_gold_ranks(ranks: dict[str, int | None]) -> str:
    parts = [f"{mid}={rank if rank is not None else 'absent'}" for mid, rank in ranks.items()]
    return " ".join(parts) if parts else "none"


def selected_message_ids(results) -> list[str]:
    ids: list[str] = []
    for result in results:
        if result.source_message_ids:
            ids.append(result.source_message_ids[0])
    return ids


def resolve_lexical_weight(cli_value: float | None) -> float:
    if cli_value is not None:
        if cli_value < 0:
            raise ValueError("lexical_weight must be >= 0")
        os.environ[LEXICAL_WEIGHT_ENV] = str(cli_value)
    return load_lexical_weight()


def search_mode(retriever, question: str, conversation_id: str, *, weight: float, k: int) -> dict[str, Any]:
    retriever.config = RetrievalConfig(lexical_weight=weight)
    started = time.perf_counter()
    results = retriever.search(question, k, conversation_id).results
    elapsed = round((time.perf_counter() - started) * 1000, 3)
    return {"selectedMessageIds": selected_message_ids(results), "latencyMs": elapsed}


def run_modes(retriever, question: str, conversation_id: str, *, hybrid_weight: float, k: int) -> dict[str, Any]:
    return {
        "baseline": search_mode(retriever, question, conversation_id, weight=0.0, k=k),
        "hybrid": search_mode(retriever, question, conversation_id, weight=hybrid_weight, k=k),
    }


def _print_summary(rows: list[dict[str, Any]], k: int) -> None:
    print(f"\n--- summary k={k} n={len(rows)} ---")
    for mode in MODES:
        recalls = [row["scores"][k][mode]["recallAt"][str(k)] for row in rows]
        hits = [row["scores"][k][mode]["hitAt"][str(k)] for row in rows]
        rrs = [row["scores"][k][mode]["reciprocalRank"] for row in rows]
        print(
            f"  {mode:10} recall@{k}={_mean(recalls):.2f} "
            f"hit@{k}={_mean(hits):.2f} mrr={_mean(rrs):.3f}"
        )
    print("  hybrid vs baseline (lower recall or worse first rank):")
    found = False
    for row in rows:
        base = row["scores"][k]["baseline"]
        other = row["scores"][k]["hybrid"]
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
                f"    {row['question'][:72]} "
                f"recall {base['recallAt'][str(k)]:.2f}->{other['recallAt'][str(k)]:.2f} "
                f"first {rank_base}->{rank_other}"
            )
    if not found:
        print("    none")


def _print_category_summary(rows: list[dict[str, Any]], k: int) -> None:
    by_cat: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_cat.setdefault(row.get("category"), []).append(row)
    print(f"\n--- by category k={k} ---")
    for category in sorted(by_cat, key=lambda item: str(item)):
        group = by_cat[category]
        print(
            f"  cat={category} n={len(group)} "
            + " ".join(
                f"{mode} recall={_mean([r['scores'][k][mode]['recallAt'][str(k)] for r in group]):.2f} "
                f"hit={_mean([r['scores'][k][mode]['hitAt'][str(k)] for r in group]):.2f} "
                f"mrr={_mean([r['scores'][k][mode]['reciprocalRank'] for r in group]):.3f}"
                for mode in MODES
            )
        )


def _print_latency(init_ms: float, rows: list[dict[str, Any]]) -> None:
    print("\n--- latency ---")
    print(f"  init_ms={init_ms:.1f}  (workspace + embedder load + memory index)")
    for mode in MODES:
        all_ms = [row["runs"][mode]["latencyMs"] for row in rows]
        steady = all_ms[1:] if len(all_ms) > 1 else all_ms
        print(
            f"  {mode:10} query_ms={_mean(all_ms):.1f} "
            f"steady_ms={_mean(steady):.1f} "
            f"first_ms={all_ms[0] if all_ms else 0:.1f} n={len(all_ms)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m continuator_memory.webapi.hybrid_diag")
    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=DEFAULT_LEXICAL_WEIGHT,
        help=f"hybrid fusion weight (default: {DEFAULT_LEXICAL_WEIGHT})",
    )
    parser.add_argument(
        "--sample",
        choices=SAMPLES,
        default="all",
        help="all = every scorable LoCoMo QA row in file order; iterative = original 10-question subset",
    )
    args = parser.parse_args(argv)
    if not DEFAULT_DB.is_file() or not DEFAULT_LOCOMO.is_file():
        print("missing locomo db or json; skip hybrid diagnostic")
        return 0
    weight = resolve_lexical_weight(args.lexical_weight)
    qa = load_locomo_qa(DEFAULT_LOCOMO)
    started = time.perf_counter()
    workspace = Workspace.from_sqlite(DEFAULT_DB, k=15)
    message_ids = conversation_message_ids(workspace)
    coverage = qa_coverage(qa, message_ids)
    sample = select_eval_rows(qa, coverage, args.sample)
    retriever = build_memory_retriever(
        workspace.conversation,
        workspace.claims,
        workspace.relations,
        workspace.embedder(),
    )
    init_ms = round((time.perf_counter() - started) * 1000, 3)
    cid = workspace.conversation.conversation_id
    print(
        f"conversation={cid} lexical_weight={weight} sample={args.sample} "
        f"n_qa={len(sample)} depth={DEPTH} ks={list(KS)} modes={list(MODES)}"
    )
    print(
        f"inventory qa_total={coverage['total']} scorable={len(coverage['scorable'])} "
        f"empty_evidence={len(coverage['empty_evidence'])} "
        f"unmatched_gold={len(coverage['unmatched_gold'])} "
        f"categories={coverage['categories']}"
    )
    print("selection=file-order scorable rows; not chosen by Hybrid success")
    print("gold qa evidence is used for scoring only")
    print("fusion: score = similarity + adjustment + lexical_weight * overlap")
    print(f"init_ms={init_ms} n_memories={len(retriever)}")

    scored: list[dict[str, Any]] = []
    for row in sample:
        print(f"\nQ: {row['question']}")
        print(f"  category={row.get('category')} gold_evidence={row['evidence']}  (eval only)")
        runs = run_modes(retriever, row["question"], cid, hybrid_weight=weight, k=DEPTH)
        scores = {
            k: {
                mode: _score(payload["selectedMessageIds"][:k], row["evidence"], cid)
                for mode, payload in runs.items()
            }
            for k in KS
        }
        scored.append({
            "question": row["question"],
            "category": row.get("category"),
            "runs": runs,
            "scores": scores,
        })
        for mode, payload in runs.items():
            ranks = gold_ranks(payload["selectedMessageIds"], row["evidence"])
            bits = [
                f"{mode:10} gold_ranks {format_gold_ranks(ranks)} ms={payload['latencyMs']}"
            ]
            for k in KS:
                score = scores[k][mode]
                bits.append(
                    f"k={k} recall={score['recallAt'][str(k)]:.2f} "
                    f"hit={score['hitAt'][str(k)]:.2f} "
                    f"first={score['firstExpectedRank']} rr={score['reciprocalRank']:.3f}"
                )
            print("  " + " | ".join(bits))
    for k in KS:
        _print_summary(scored, k)
        _print_category_summary(scored, k)
    _print_latency(init_ms, scored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
