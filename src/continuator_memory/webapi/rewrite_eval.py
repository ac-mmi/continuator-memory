"""Compare baseline, legacy generic multi-query, and intent-preserving multi-query.

Gold ``qa[].evidence`` is scoring-only. Fusion stays equal-weight RRF.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from continuator_memory.webapi.iterative_diag import (
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
    search_queries,
)
from continuator_memory.webapi.query_plan import (
    EDUCATION_CUES,
    entities_from_question,
    resolve_plan,
    tokens,
)
from continuator_memory.webapi.workspace import Workspace
from continuator_memory.v2.hybrid_retrieval import DEFAULT_RRF_C

REGRESSIONS = (
    "When did Caroline go to the LGBTQ support group?",
    "When did Melanie run a charity race?",
    "When is Melanie planning on going camping?",
)


def legacy_queries(question: str) -> tuple[str, ...]:
    """Previous fallback: generic speaker paraphrases and school→career routing."""
    name = (entities_from_question(question) or ("they",))[0]
    toks = set(tokens(question))
    if toks & EDUCATION_CUES:
        return (
            question,
            f"What career is {name} interested in?",
            f"What kind of work does {name} want to do?",
            f"What does {name} plan to study or train for?",
        )
    return (
        question,
        f"What did {name} say that answers this question?",
        f"Which stated facts about {name} are relevant?",
    )


def _selected_ids(question, conversation, retriever, queries, k: int, budget_chars: int) -> list[str]:
    depth = max(k, DEFAULT_PER_QUERY_K)
    outcomes = search_queries(retriever, queries, conversation.conversation_id, depth)
    evidence, _, _ = evidence_from_rankings(
        question,
        conversation,
        [item.results for item in outcomes],
        k=k,
        budget_chars=budget_chars,
        rrf_c=DEFAULT_RRF_C,
        original_outcome=outcomes[0],
        original_weight=1.0,
    )
    return list(evidence.selection.selected)


def main() -> int:
    if not DEFAULT_DB.is_file() or not DEFAULT_LOCOMO.is_file():
        print("missing locomo db or json; skip rewrite eval")
        return 0
    qa = load_locomo_qa(DEFAULT_LOCOMO)
    sample = _sample_rows(qa)
    workspace = Workspace.from_sqlite(DEFAULT_DB, k=15)
    retriever = build_memory_retriever(
        workspace.conversation,
        workspace.claims,
        workspace.relations,
        workspace.embedder(),
    )
    cid = workspace.conversation.conversation_id
    print(f"conversation={cid} fusion=equal-weight-rrf n={len(sample)}")
    print("gold qa evidence is used for scoring only")
    summaries: dict[int, list[dict[str, Any]]] = {k: [] for k in KS}

    for row in sample:
        question = row["question"]
        gold = row["evidence"]
        improved = resolve_plan(question, None)
        legacy = legacy_queries(question)
        print(f"\nQ: {question}")
        print(f"  gold={gold}  (eval only)")
        print(f"  legacy_q={list(legacy)}")
        print(f"  improved_q={list(improved.queries)}")
        for k in KS:
            started = time.perf_counter()
            base_ids = _selected_ids(
                question, workspace.conversation, retriever, (question,), k, workspace.budget_chars,
            )
            base_ms = round((time.perf_counter() - started) * 1000, 3)
            started = time.perf_counter()
            legacy_ids = _selected_ids(
                question, workspace.conversation, retriever, legacy, k, workspace.budget_chars,
            )
            legacy_ms = round((time.perf_counter() - started) * 1000, 3)
            started = time.perf_counter()
            new_ids = _selected_ids(
                question, workspace.conversation, retriever, improved.queries, k, workspace.budget_chars,
            )
            new_ms = round((time.perf_counter() - started) * 1000, 3)
            scores = {
                "baseline": _score(base_ids, gold, cid),
                "legacy_mq": _score(legacy_ids, gold, cid),
                "improved_mq": _score(new_ids, gold, cid),
            }
            summaries[k].append({"question": question, "scores": scores, "ms": {
                "baseline": base_ms, "legacy_mq": legacy_ms, "improved_mq": new_ms,
            }})
            print(
                f"  k={k} "
                f"base first={scores['baseline']['firstExpectedRank']} "
                f"recall={scores['baseline']['recallAt'][str(k)]:.2f} {base_ms}ms "
                f"| legacy {scores['legacy_mq']['firstExpectedRank']}/"
                f"{scores['legacy_mq']['recallAt'][str(k)]:.2f} {legacy_ms}ms "
                f"| improved {scores['improved_mq']['firstExpectedRank']}/"
                f"{scores['improved_mq']['recallAt'][str(k)]:.2f} {new_ms}ms "
                f"selected={new_ids[:8]}"
            )

    for k, rows in summaries.items():
        print(f"\n--- summary k={k} ---")
        for mode in ("baseline", "legacy_mq", "improved_mq"):
            print(
                f"  {mode:12} recall@{k}={_mean([r['scores'][mode]['recallAt'][str(k)] for r in rows]):.2f} "
                f"hit@{k}={_mean([r['scores'][mode]['hitAt'][str(k)] for r in rows]):.2f} "
                f"mrr={_mean([r['scores'][mode]['reciprocalRank'] for r in rows]):.3f} "
                f"ms={_mean([r['ms'][mode] for r in rows]):.0f}"
            )
        print("  prior regressions (legacy vs improved vs baseline):")
        for row in rows:
            if row["question"] not in REGRESSIONS:
                continue
            print(
                f"    {row['question'][:64]} "
                f"base={row['scores']['baseline']['firstExpectedRank']}/"
                f"{row['scores']['baseline']['recallAt'][str(k)]:.2f} "
                f"legacy={row['scores']['legacy_mq']['firstExpectedRank']}/"
                f"{row['scores']['legacy_mq']['recallAt'][str(k)]:.2f} "
                f"improved={row['scores']['improved_mq']['firstExpectedRank']}/"
                f"{row['scores']['improved_mq']['recallAt'][str(k)]:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
