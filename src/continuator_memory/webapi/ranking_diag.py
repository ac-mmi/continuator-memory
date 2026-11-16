"""Per-query ranking dump for LoCoMo multi-query / iterative regressions.

Gold ``qa[].evidence`` is used for scoring only. It is never passed into
planning, retrieval, or evaluation. Default RRF weight stays 1; the
original-query weight comparison is evaluation-only.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

from continuator_memory.webapi.evidence_eval import heuristic_evaluation
from continuator_memory.webapi.iterative import retrieve_iterative_evidence
from continuator_memory.webapi.iterative_diag import (
    CAROLINE_Q,
    DEFAULT_DB,
    DEFAULT_LOCOMO,
    KS,
    _mean,
    _sample_rows,
    _score,
    load_locomo_qa,
)
from continuator_memory.webapi.multiquery import (
    DEFAULT_PER_QUERY_K,
    build_memory_retriever,
    evidence_from_rankings,
    rrf_fusion,
    search_queries,
)
from continuator_memory.webapi.query_plan import resolve_plan
from continuator_memory.webapi.workspace import Workspace
from continuator_memory.v2.hybrid_retrieval import DEFAULT_RRF_C

EXPERIMENTAL_WEIGHT = 4.0
FOCUS = (
    "When did Caroline go to the LGBTQ support group?",
    "When did Melanie run a charity race?",
    "When is Melanie planning on going camping?",
)


def _msg_rank(results, message_id: str) -> dict[str, Any] | None:
    for result in results:
        if message_id in result.source_message_ids:
            return {
                "rank": result.rank,
                "memoryId": result.memory_id,
                "similarity": round(result.similarity, 4),
            }
    return None


def _fused_msg_rank(fused, message_id: str) -> dict[str, Any] | None:
    for index, item in enumerate(fused, 1):
        if message_id in item.source_message_ids:
            return {
                "rank": index,
                "memoryId": item.memory_id,
                "fusion": round(item.fusion, 6),
                "perQueryRanks": list(item.ranks),
            }
    return None


def _top_ids(results, limit: int = 8) -> list[str]:
    out = []
    for result in results[:limit]:
        mid = result.source_message_ids[0] if result.source_message_ids else result.memory_id
        out.append(f"{mid}#{result.rank}")
    return out


def _print_gold_table(label: str, gold: list[str], lookup) -> None:
    parts = []
    for mid in gold:
        row = lookup(mid)
        if row is None:
            parts.append(f"{mid}=absent")
        else:
            extra = ""
            if "perQueryRanks" in row:
                extra = f" ranks={row['perQueryRanks']} fusion={row['fusion']}"
            elif "similarity" in row:
                extra = f" sim={row['similarity']}"
            parts.append(f"{mid}=r{row['rank']}{extra}")
    print(f"    {label}: {', '.join(parts)}")


def main() -> int:
    if not DEFAULT_DB.is_file() or not DEFAULT_LOCOMO.is_file():
        print("missing locomo db or json; skip ranking diagnostic")
        return 0
    qa = load_locomo_qa(DEFAULT_LOCOMO)
    sample = _sample_rows(qa)
    workspace = Workspace.from_sqlite(DEFAULT_DB, k=15)
    workspace.set_evaluator(heuristic_evaluation)
    retriever = build_memory_retriever(
        workspace.conversation,
        workspace.claims,
        workspace.relations,
        workspace.embedder(),
    )
    cid = workspace.conversation.conversation_id
    print(f"conversation={cid} rrf_c={DEFAULT_RRF_C} experimental_weight={EXPERIMENTAL_WEIGHT}")
    print("gold qa evidence is used for scoring only")
    summaries: dict[int, list[dict[str, Any]]] = {k: [] for k in KS}

    for row in sample:
        question = row["question"]
        gold = row["evidence"]
        focus = question in FOCUS or question.startswith("What fields would Caroline")
        print(f"\n======== Q: {question} ========")
        print(f"  gold={gold}  (eval only)")
        plan = resolve_plan(question, None)
        print(f"  plan.status={plan.status} queries={list(plan.queries)}")
        depth = max(15, DEFAULT_PER_QUERY_K)
        started = time.perf_counter()
        outcomes = search_queries(retriever, plan.queries, cid, depth)
        retrieve_ms = round((time.perf_counter() - started) * 1000, 3)
        original = outcomes[0].results
        print(f"  original top-8: {_top_ids(original)}")
        _print_gold_table("original-query", gold, lambda mid, src=original: _msg_rank(src, mid))
        for query, outcome in zip(plan.queries[1:], outcomes[1:]):
            print(f"  paraphrase {query!r} top-8: {_top_ids(outcome.results)}")
            _print_gold_table(
                f"paraphrase {query!r}",
                gold,
                lambda mid, src=outcome.results: _msg_rank(src, mid),
            )

        equal = rrf_fusion([o.results for o in outcomes], rrf_c=DEFAULT_RRF_C, original_weight=1.0)
        weighted = rrf_fusion(
            [o.results for o in outcomes], rrf_c=DEFAULT_RRF_C, original_weight=EXPERIMENTAL_WEIGHT,
        )
        print(f"  equal-RRF top-8: {[item.source_message_ids[0] for item in equal[:8]]}")
        _print_gold_table("equal-RRF fused", gold, lambda mid, src=equal: _fused_msg_rank(src, mid))
        print(f"  w{EXPERIMENTAL_WEIGHT:g}-RRF top-8: {[item.source_message_ids[0] for item in weighted[:8]]}")
        _print_gold_table(
            f"w{EXPERIMENTAL_WEIGHT:g}-RRF fused",
            gold,
            lambda mid, src=weighted: _fused_msg_rank(src, mid),
        )

        for k in KS:
            eq_ids = [
                item.source_message_ids[0]
                for item in equal[:k]
                if item.source_message_ids
            ]
            wt_ids = [
                item.source_message_ids[0]
                for item in weighted[:k]
                if item.source_message_ids
            ]
            base_ids = [
                r.source_message_ids[0]
                for r in original[:k]
                if r.source_message_ids
            ]
            protected = {r.memory_id for r in original[:k]}
            protect_ids = [
                item.source_message_ids[0]
                for item in equal
                if item.memory_id in protected and item.source_message_ids
            ][:k]
            evidence, _, selected = evidence_from_rankings(
                question,
                workspace.conversation,
                [o.results for o in outcomes],
                k=k,
                budget_chars=workspace.budget_chars,
                rrf_c=DEFAULT_RRF_C,
                original_outcome=outcomes[0],
                original_weight=1.0,
            )
            dropped = list(evidence.selection.dropped)
            print(
                f"  k={k} budget_dropped={dropped[:6] or '[]'} "
                f"selected={list(evidence.selection.selected)[:8]}"
            )
            it_started = time.perf_counter()
            it_ev, it_diag = retrieve_iterative_evidence(
                question,
                workspace.conversation,
                workspace.claims,
                workspace.relations,
                workspace.embedder(),
                k,
                workspace.budget_chars,
                planner=None,
                evaluator=heuristic_evaluation,
                retriever=retriever,
                original_weight=1.0,
            )
            it_ms = round((time.perf_counter() - it_started) * 1000, 3)
            it_wt_started = time.perf_counter()
            it_wt_ev, it_wt_diag = retrieve_iterative_evidence(
                question,
                workspace.conversation,
                workspace.claims,
                workspace.relations,
                workspace.embedder(),
                k,
                workspace.budget_chars,
                planner=None,
                evaluator=heuristic_evaluation,
                retriever=retriever,
                original_weight=EXPERIMENTAL_WEIGHT,
            )
            it_wt_ms = round((time.perf_counter() - it_wt_started) * 1000, 3)
            it_ids = list(it_ev.selection.selected)
            it_wt_ids = list(it_wt_ev.selection.selected)
            scores = {
                "baseline": _score(base_ids, gold, cid),
                "multiquery": _score(eq_ids, gold, cid),
                "multiquery_w": _score(wt_ids, gold, cid),
                "multiquery_protect": _score(protect_ids, gold, cid),
                "iterative": _score(it_ids, gold, cid),
                "iterative_w": _score(it_wt_ids, gold, cid),
            }
            summaries[k].append({
                "question": question,
                "scores": scores,
                "latency": {"plan_retrieve": retrieve_ms, "iterative": it_ms, "iterative_w": it_wt_ms},
                "iterative": {
                    "stop": it_diag.get("stopReason"),
                    "rounds": it_diag.get("roundCount"),
                },
                "iterative_w": {
                    "stop": it_wt_diag.get("stopReason"),
                    "rounds": it_wt_diag.get("roundCount"),
                },
            })
            print(
                f"  k={k} baseline first={scores['baseline']['firstExpectedRank']} "
                f"recall={scores['baseline']['recallAt'][str(k)]:.2f} "
                f"| MQ {scores['multiquery']['firstExpectedRank']}/"
                f"{scores['multiquery']['recallAt'][str(k)]:.2f} "
                f"| MQ-w{EXPERIMENTAL_WEIGHT:g} {scores['multiquery_w']['firstExpectedRank']}/"
                f"{scores['multiquery_w']['recallAt'][str(k)]:.2f} "
                f"| MQ-protect {scores['multiquery_protect']['firstExpectedRank']}/"
                f"{scores['multiquery_protect']['recallAt'][str(k)]:.2f} "
                f"| IT {scores['iterative']['firstExpectedRank']}/"
                f"{scores['iterative']['recallAt'][str(k)]:.2f} "
                f"stop={it_diag.get('stopReason')} rounds={it_diag.get('roundCount')} "
                f"| IT-w {scores['iterative_w']['firstExpectedRank']}/"
                f"{scores['iterative_w']['recallAt'][str(k)]:.2f} "
                f"stop={it_wt_diag.get('stopReason')}"
            )
            if focus:
                for rno, round_row in enumerate(it_diag.get("iterativeRounds") or [], 1):
                    ev = round_row.get("evaluation") or {}
                    print(
                        f"    iter r{rno} q={round_row.get('queries')} "
                        f"status={ev.get('status')} stop_gap={ev.get('missingInformation')!r} "
                        f"next={ev.get('nextQueries')} selected={round_row.get('selectedMessageIds')}"
                    )

    for k, rows in summaries.items():
        print(f"\n--- summary k={k} ---")
        for mode in (
            "baseline", "multiquery", "multiquery_w", "multiquery_protect",
            "iterative", "iterative_w",
        ):
            recalls = [r["scores"][mode]["recallAt"][str(k)] for r in rows]
            hits = [r["scores"][mode]["hitAt"][str(k)] for r in rows]
            rrs = [r["scores"][mode]["reciprocalRank"] for r in rows]
            print(
                f"  {mode:14} recall@{k}={_mean(recalls):.2f} "
                f"hit@{k}={_mean(hits):.2f} mrr={_mean(rrs):.3f}"
            )
        print("  regressions vs baseline (current MQ/IT):")
        for row in rows:
            base = row["scores"]["baseline"]
            for mode in ("multiquery", "iterative"):
                other = row["scores"][mode]
                if (
                    other["recallAt"][str(k)] < base["recallAt"][str(k)]
                    or (
                        other["firstExpectedRank"]
                        and base["firstExpectedRank"]
                        and other["firstExpectedRank"] > base["firstExpectedRank"]
                    )
                    or (other["firstExpectedRank"] is None and base["firstExpectedRank"] is not None)
                ):
                    print(
                        f"    {mode}: {row['question'][:70]} "
                        f"recall {base['recallAt'][str(k)]:.2f}->{other['recallAt'][str(k)]:.2f} "
                        f"first {base['firstExpectedRank']}->{other['firstExpectedRank']}"
                    )
        print(f"  experimental original_weight={EXPERIMENTAL_WEIGHT:g} vs current MQ/IT:")
        recovered = 0
        leftover = 0
        for row in rows:
            for current, weighted in (("multiquery", "multiquery_w"), ("iterative", "iterative_w")):
                cur = row["scores"][current]
                wtd = row["scores"][weighted]
                base = row["scores"]["baseline"]
                cur_reg = (
                    cur["recallAt"][str(k)] < base["recallAt"][str(k)]
                    or (
                        cur["firstExpectedRank"]
                        and base["firstExpectedRank"]
                        and cur["firstExpectedRank"] > base["firstExpectedRank"]
                    )
                    or (cur["firstExpectedRank"] is None and base["firstExpectedRank"] is not None)
                )
                wtd_reg = (
                    wtd["recallAt"][str(k)] < base["recallAt"][str(k)]
                    or (
                        wtd["firstExpectedRank"]
                        and base["firstExpectedRank"]
                        and wtd["firstExpectedRank"] > base["firstExpectedRank"]
                    )
                    or (wtd["firstExpectedRank"] is None and base["firstExpectedRank"] is not None)
                )
                if cur_reg and not wtd_reg:
                    recovered += 1
                    print(f"    recovered {current}: {row['question'][:70]}")
                elif cur_reg and wtd_reg:
                    leftover += 1
        print(f"    recovered={recovered} still_regressed={leftover}")
        stops = Counter(r["iterative"]["stop"] for r in rows)
        print(f"  iterative stops={dict(stops)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
