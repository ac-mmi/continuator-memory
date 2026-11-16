"""Compare Baseline cosine ranking with experimental lexical hybrid ranking.

Gold LoCoMo evidence IDs are used for scoring only. They are never passed into
retrieval. Baseline ranking is unchanged (lexical_weight=0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from continuator_memory.v2.lexical import (
    DEFAULT_LEXICAL_WEIGHT,
    lexical_overlap,
    load_lexical_weight,
)
from continuator_memory.v2.memory_retrieval import RetrievalConfig
from continuator_memory.webapi.iterative_diag import DEFAULT_DB, DEFAULT_LOCOMO, load_locomo_qa
from continuator_memory.webapi.multiquery import build_memory_retriever
from continuator_memory.webapi.workspace import Workspace

QUESTION = "What did Caroline research?"
DEPTH = 50
WATCH = ("D2:8", "D1:9", "D2:12")


def _row(results, message_id: str) -> dict[str, Any] | None:
    for result in results:
        if message_id in result.source_message_ids:
            lexical = lexical_overlap(QUESTION, result.content)
            return {
                "rank": result.rank,
                "memoryId": result.memory_id,
                "similarity": round(result.similarity, 4),
                "adjustment": round(result.adjustment, 4),
                "lexical": round(lexical, 4),
                "score": round(result.score, 4),
                "inTop15": result.rank <= 15,
            }
    return None


def _print_watch(label: str, results, gold: list[str], weight: float) -> None:
    print(f"  {label}:")
    seen = list(dict.fromkeys([*gold, *WATCH]))
    for mid in seen:
        row = _row(results, mid)
        if row is None:
            print(f"    {mid}=absent (beyond depth {len(results)})")
            continue
        print(
            f"    {mid}=r{row['rank']} sim={row['similarity']} adj={row['adjustment']} "
            f"lex={row['lexical']} score={row['score']} top15={row['inTop15']}"
        )
    top = []
    for result in results[:15]:
        mid = result.source_message_ids[0] if result.source_message_ids else result.memory_id
        top.append(f"{mid}#{result.rank}")
    print(f"    top-15: {top}")
    print(f"    weight={weight}  score = similarity + adjustment + {weight} * lexical")


def main() -> int:
    if not DEFAULT_DB.is_file():
        print("missing locomo db; skip lexical diagnostic")
        return 0
    qa = load_locomo_qa(DEFAULT_LOCOMO) if DEFAULT_LOCOMO.is_file() else []
    gold = next((row["evidence"] for row in qa if row["question"] == QUESTION), [])
    workspace = Workspace.from_sqlite(DEFAULT_DB, k=15)
    retriever = build_memory_retriever(
        workspace.conversation,
        workspace.claims,
        workspace.relations,
        workspace.embedder(),
    )
    cid = workspace.conversation.conversation_id
    weight = load_lexical_weight()
    print(f"conversation={cid} question={QUESTION!r}")
    print(f"gold={gold or WATCH[:1]}  (eval only)")
    print(
        f"fusion: score = similarity + adjustment + lexical_weight * overlap; "
        f"default_weight={DEFAULT_LEXICAL_WEIGHT} env_weight={weight}"
    )
    print("overlap = |question_content_stems ∩ memory_stems| / |question_content_stems|")

    baseline = retriever.search(QUESTION, DEPTH, cid).results
    retriever.config = RetrievalConfig(lexical_weight=weight)
    hybrid = retriever.search(QUESTION, DEPTH, cid).results
    _print_watch("baseline lexical_weight=0", baseline, gold, 0.0)
    _print_watch(f"hybrid lexical_weight={weight}", hybrid, gold, weight)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
