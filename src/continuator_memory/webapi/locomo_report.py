"""LoCoMo full-conversation RAG evaluation report.

Runs the existing retrieve-and-generate path. Gold answers and evidence IDs are
scoring-only and are never passed into retrieval or generation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from continuator_memory.envfile import load_root_env
from continuator_memory.evaluation import metrics
from continuator_memory.evaluation.answer_scoring import score_answer
from continuator_memory.v2.lexical import load_lexical_weight
from continuator_memory.v2.memory_retrieval import RetrievalConfig, RetrievalOutcome, question_intent
from continuator_memory.v2.turn_context import load_context_weight
from continuator_memory.webapi.answering import (
    MAX_ANSWER_TOP_K,
    build_answer_context,
    generate_answer,
    load_answer_top_k,
    load_generation_settings,
    make_ollama_generator,
    parse_generation_overrides,
)
from continuator_memory.webapi.iterative import retrieve_iterative_evidence
from continuator_memory.webapi.iterative_diag import DEFAULT_DB, DEFAULT_LOCOMO
from continuator_memory.webapi.multiquery import build_memory_retriever, load_original_weight, retrieve_multiquery_evidence
from continuator_memory.webapi.workspace import RETRIEVAL_MODES, Workspace
from continuator_memory.v2.graph_index import GraphIndex

from continuator_memory.v2.bprime_links import BPrimeConfig
from continuator_memory.v2.source_retrieval import (
    SourceEvidence,
    resolve,
    select_sources,
    source_chunks,
)

RETRIEVAL_KS = (5, 15, 50)
DEFAULT_OUTPUT = Path("reports/locomo_conv26_report.md")
DEFAULT_DEPTH = 50
ISSUE_NOT_LIST = "evidence_not_list"
ISSUE_EMPTY = "evidence_empty"
ISSUE_NON_STRING = "evidence_non_string"
ISSUE_UNMATCHED = "evidence_unmatched"


@dataclass
class EvidenceAnnotation:
    gold_ids: list[str]
    issues: list[str]
    scorable: bool


@dataclass
class LocomoQuestion:
    index: int
    question: str
    gold_answer: Any
    category: Any
    raw_evidence: Any
    annotation: EvidenceAnnotation
    sample_id: str | None = None
    # Evaluation-only. LoCoMo category 5 carries a trap answer instead of ``answer``.
    adversarial_answer: Any = None
    # Evaluation-only. None means: LoCoMo category 5 expects an abstention.
    expects_abstention: bool | None = None

    @property
    def abstention_expected(self) -> bool:
        if self.expects_abstention is not None:
            return self.expects_abstention
        return str(self.category) == "5"


def load_locomo_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        records = [item for item in raw if isinstance(item, dict)]
        if not records:
            raise ValueError(f"{path}: LoCoMo JSON list has no conversation objects")
        return records
    if isinstance(raw, dict):
        return [raw]
    raise ValueError(f"{path}: LoCoMo JSON must be an object or a list")


def select_record(records: list[dict[str, Any]], *, sample_id: str | None, index: int) -> dict[str, Any]:
    if sample_id:
        for record in records:
            if record.get("sample_id") == sample_id:
                return record
        raise ValueError(f"no LoCoMo conversation with sample_id={sample_id!r}")
    if index < 0 or index >= len(records):
        raise ValueError(f"conversation index {index} is out of range 0..{len(records) - 1}")
    return records[index]


def classify_evidence(raw: Any, message_ids: set[str]) -> EvidenceAnnotation:
    issues: list[str] = []
    if not isinstance(raw, list):
        return EvidenceAnnotation([], [ISSUE_NOT_LIST], False)
    gold: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            issues.append(ISSUE_NON_STRING)
            continue
        text = item.strip()
        if not text:
            continue
        if text not in gold:
            gold.append(text)
    if not gold:
        issues.append(ISSUE_EMPTY)
        return EvidenceAnnotation([], issues, False)
    if not any(item in message_ids for item in gold):
        issues.append(ISSUE_UNMATCHED)
        return EvidenceAnnotation(gold, issues, False)
    return EvidenceAnnotation(gold, issues, True)


def load_locomo_questions(record: dict[str, Any], message_ids: set[str]) -> list[LocomoQuestion]:
    sample_id = record.get("sample_id") if isinstance(record.get("sample_id"), str) else None
    rows: list[LocomoQuestion] = []
    for index, item in enumerate(record.get("qa") or []):
        if not isinstance(item, dict):
            rows.append(LocomoQuestion(
                index, "", None, None, item,
                EvidenceAnnotation([], [ISSUE_NOT_LIST], False), sample_id,
            ))
            continue
        question = item.get("question")
        rows.append(LocomoQuestion(
            index=index,
            question=question.strip() if isinstance(question, str) else "",
            gold_answer=item.get("answer"),
            category=item.get("category"),
            raw_evidence=item.get("evidence"),
            annotation=classify_evidence(item.get("evidence"), message_ids),
            sample_id=sample_id,
            adversarial_answer=item.get("adversarial_answer"),
        ))
    return rows


def gold_ranks(selected: list[str], expected: list[str]) -> dict[str, int | None]:
    index = {message_id: rank for rank, message_id in enumerate(selected, 1)}
    return {message_id: index.get(message_id) for message_id in expected}


def score_retrieval(selected: list[str], expected: list[str], conversation_id: str) -> dict[str, Any]:
    class Row:
        def __init__(self, mid: str):
            self.conversation_id = conversation_id
            self.source_message_ids = (mid,)

    ranked = [Row(mid) for mid in selected]
    rank = metrics.first_expected_rank(ranked, conversation_id, expected) if expected else None
    recall: dict[str, float | None] = {}
    hit: dict[str, bool | None] = {}
    for k in RETRIEVAL_KS:
        if not expected:
            recall[str(k)] = None
            hit[str(k)] = None
        elif not ranked:
            recall[str(k)] = 0.0
            hit[str(k)] = False
        else:
            recall[str(k)] = metrics.recall_at_k(ranked, conversation_id, expected, k)
            hit[str(k)] = metrics.hit_at_k(rank, k)
    return {
        "firstExpectedRank": rank,
        "reciprocalRank": metrics.reciprocal_rank(rank) if expected else None,
        "recallAt": recall,
        "hitAt": hit,
        "goldRanks": gold_ranks(selected, expected),
        "availableDepth": len(selected),
    }


def ranked_message_ids(results) -> list[str]:
    ids: list[str] = []
    for result in results:
        for message_id in result.source_message_ids:
            if message_id not in ids:
                ids.append(message_id)
    return ids


def md_escape(text: Any) -> str:
    value = "" if text is None else str(text)
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")


def md_fence(text: Any, info: str = "") -> str:
    value = "" if text is None else str(text)
    ticks = "```"
    while ticks in value:
        ticks += "`"
    return f"{ticks}{info}\n{value}\n{ticks}"


def format_gold_answer(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# Experimental graph retrieval (evaluation harness only; the web workspace modes are unchanged).
GRAPH_MODES = frozenset({"graph", "graph_hybrid"})
EVAL_RETRIEVAL_MODES = RETRIEVAL_MODES | GRAPH_MODES
GRAPH_RRF_C = 60  # standard reciprocal-rank-fusion constant; not tuned


def retrieval_config(mode: str) -> RetrievalConfig | None:
    # The graph modes reuse the unchanged hybrid configuration for their baseline ranking.
    if mode == "hybrid" or mode in GRAPH_MODES:
        return RetrievalConfig(lexical_weight=load_lexical_weight(), context_weight=load_context_weight())
    return RetrievalConfig()


def evidence_for_generation(question: str, conversation, outcome: RetrievalOutcome, *, k: int,
                            budget_chars: int) -> SourceEvidence:
    cut = outcome.results[:k]
    cut_outcome = RetrievalOutcome(outcome.question, outcome.intent, cut, outcome.excluded)
    matches = tuple(resolve(result, conversation) for result in cut)
    selection = select_sources(matches, budget_chars)
    return SourceEvidence(
        question, cut_outcome, matches, selection,
        source_chunks(selection, matches, conversation),
        n_seed_matches=len(matches),
    )


def graph_index_for(workspace: Workspace) -> GraphIndex:
    graph = getattr(workspace, "_eval_graph", None)
    if graph is None:
        graph = GraphIndex(workspace.embedder())
        graph.add_messages(workspace.conversation.messages)
        workspace._eval_graph = graph
    return graph


def _graph_retrieve(workspace: Workspace, question: str, mode: str, *, depth: int, top_k: int):
    """Graph-only or graph+baseline (RRF) ranking over the same memories; evidence assembly is shared."""
    retriever = getattr(workspace, "_eval_retriever", None)
    if retriever is None:
        raise ValueError("graph retrieval needs the evaluation retriever")
    cid = workspace.conversation.conversation_id
    base_config = retrieval_config("hybrid")
    retriever.config = base_config
    baseline = retriever.search(question, max(1, len(retriever)), cid)
    # Result objects for every memory, including ones below the similarity floor, so a turn the
    # graph reaches can be returned through the normal evidence path.
    retriever.config = replace(base_config, min_similarity=None)
    try:
        everything = retriever.search(question, max(1, len(retriever)), cid)
    finally:
        retriever.config = base_config
    by_message: dict[str, Any] = {}
    for result in everything.results:
        for mid in result.source_message_ids:
            by_message.setdefault(mid, result)
    graph = graph_index_for(workspace)
    found = graph.search(question, None, cid)
    graph_order: list[Any] = []
    seen: set[str] = set()
    for hit in found.hits:
        result = by_message.get(hit.message_id)
        if result is not None and result.memory_id not in seen:
            seen.add(result.memory_id)
            graph_order.append(result)
    graph_score = {r.memory_id: hit.score for hit in found.hits
                   for r in [by_message.get(hit.message_id)] if r is not None}
    if mode == "graph":
        ranked = [(r, graph_score[r.memory_id]) for r in graph_order]
    else:
        base_rank = {r.memory_id: i for i, r in enumerate(baseline.results, 1)}
        graph_rank = {r.memory_id: i for i, r in enumerate(graph_order, 1)}
        pool = {r.memory_id: r for r in [*baseline.results, *graph_order]}
        fused = {
            mid: (1 / (GRAPH_RRF_C + base_rank[mid]) if mid in base_rank else 0.0)
            + (1 / (GRAPH_RRF_C + graph_rank[mid]) if mid in graph_rank else 0.0)
            for mid in pool
        }
        missing = 1 << 30
        order = sorted(pool, key=lambda mid: (-round(fused[mid], 12), base_rank.get(mid, missing),
                                              graph_rank.get(mid, missing), mid))
        ranked = [(pool[mid], fused[mid]) for mid in order]
    results = tuple(replace(r, rank=i, score=score) for i, (r, score) in enumerate(ranked[:depth], 1))
    outcome = RetrievalOutcome(question, baseline.intent, results)
    evidence = evidence_for_generation(question, workspace.conversation, outcome, k=top_k,
                                       budget_chars=workspace.budget_chars)
    hits = {hit.message_id: hit for hit in found.hits}
    trace = {
        "mode": mode,
        "seeds": found.seeds,
        "links": {k: list(v) for k, v in found.links.items()},
        "unmatched": list(found.unmatched),
        "graphHits": len(found.hits),
        "turns": [
            {"messageId": mid, "graphRank": hits[mid].rank, "graphScore": hits[mid].score,
             "hops": hits[mid].hops, "matchedSeeds": list(hits[mid].matched_seeds),
             "speaker": hits[mid].speaker, "timestamp": hits[mid].timestamp}
            for r in results for mid in r.source_message_ids if mid in hits
        ],
    }
    extra = {
        "rankedMessageIds": ranked_message_ids(results),
        "retrievedMemories": _memory_rows(results),
        "trace": trace,
    }
    return evidence, extra


def _retrieve(workspace: Workspace, question: str, mode: str, *, depth: int, top_k: int):
    extra: dict[str, Any] = {}
    if mode == "multiquery":
        evidence, extra = retrieve_multiquery_evidence(
            question, workspace.conversation, workspace.claims, workspace.relations,
            workspace.embedder(), top_k, workspace.budget_chars,
            planner=workspace._resolve_planner(load_generation_settings()),
            original_weight=load_original_weight(),
        )
        return evidence, extra
    if mode == "iterative":
        evidence, extra = retrieve_iterative_evidence(
            question, workspace.conversation, workspace.claims, workspace.relations,
            workspace.embedder(), top_k, workspace.budget_chars,
            planner=workspace._resolve_planner(load_generation_settings()),
            evaluator=workspace._resolve_evaluator(load_generation_settings()),
            original_weight=load_original_weight(),
        )
        return evidence, extra
    if mode in GRAPH_MODES:
        return _graph_retrieve(workspace, question, mode, depth=depth, top_k=top_k)
    bprime = None
    if mode == "bprime":
        bprime = BPrimeConfig(
            enabled=True,
            classifications_path=workspace.classifications_path,
            max_hops=1,
        )
    retriever = getattr(workspace, "_eval_retriever", None)
    if retriever is None or mode == "bprime":
        from continuator_memory.v2.source_retrieval import retrieve_source_evidence
        evidence = retrieve_source_evidence(
            question, workspace.conversation, workspace.claims, workspace.relations,
            workspace.embedder(), depth, workspace.budget_chars,
            config=retrieval_config(mode), bprime=bprime,
        )
        return evidence, extra
    retriever.config = retrieval_config(mode) or RetrievalConfig()
    outcome = retriever.search(question, depth, workspace.conversation.conversation_id)
    evidence = evidence_for_generation(
        question, workspace.conversation, outcome, k=top_k, budget_chars=workspace.budget_chars,
    )
    extra["rankedMessageIds"] = ranked_message_ids(outcome.results)
    extra["retrievedMemories"] = [
        {
            "rank": result.rank,
            "memoryId": result.memory_id,
            "score": result.score,
            "similarity": result.similarity,
            "adjustment": result.adjustment,
            "content": result.content,
            "sourceMessageIds": list(result.source_message_ids),
        }
        for result in outcome.results
    ]
    extra["fullOutcome"] = outcome
    return evidence, extra


CONDITION_NORMAL = "normal"
CONDITION_ORACLE = "oracle"
CONDITION_NO_MEMORY = "no_memory"
CONDITIONS = (CONDITION_NORMAL, CONDITION_ORACLE, CONDITION_NO_MEMORY)


def _memory_rows(results) -> list[dict[str, Any]]:
    return [
        {
            "rank": r.rank,
            "memoryId": r.memory_id,
            "score": r.score,
            "similarity": r.similarity,
            "adjustment": r.adjustment,
            "content": r.content,
            "sourceMessageIds": list(r.source_message_ids),
        }
        for r in results
    ]


def _condition_evidence(workspace: Workspace, question: str, condition: str, mode: str,
                        oracle_message_ids: list[str] | None):
    """Evidence for the oracle and no-memory conditions. Prompt assembly and generation are shared
    with the normal condition; only the evidence differs."""
    missing: list[str] = []
    if condition == CONDITION_NO_MEMORY:
        outcome = RetrievalOutcome(question, question_intent(question), ())
    else:
        retriever = getattr(workspace, "_eval_retriever", None)
        if retriever is None:
            raise ValueError("the oracle condition needs the evaluation retriever")
        wanted = list(dict.fromkeys(oracle_message_ids or ()))
        if not wanted:
            raise ValueError("the oracle condition needs oracle message ids")
        # Oracle evidence is chosen by message id, so the similarity floor must not drop it.
        base = retrieval_config(mode) or RetrievalConfig()
        retriever.config = replace(base, min_similarity=None)
        try:
            full = retriever.search(question, max(1, len(retriever)), workspace.conversation.conversation_id)
        finally:
            retriever.config = base
        position = {m.message_id: m.position for m in workspace.conversation.messages}
        chosen = sorted(
            (r for r in full.results if set(wanted) & set(r.source_message_ids)),
            key=lambda r: min(position.get(mid, 1 << 30) for mid in r.source_message_ids),
        )
        covered = {mid for r in chosen for mid in r.source_message_ids}
        missing = [mid for mid in wanted if mid not in covered]
        outcome = RetrievalOutcome(
            question, full.intent, tuple(replace(r, rank=i) for i, r in enumerate(chosen, 1)),
        )
    evidence = evidence_for_generation(
        question, workspace.conversation, outcome, k=len(outcome.results), budget_chars=workspace.budget_chars,
    )
    extra = {
        "rankedMessageIds": ranked_message_ids(outcome.results),
        "retrievedMemories": _memory_rows(outcome.results),
        "oracleMissingMessageIds": missing,
    }
    return evidence, extra


def _answer_scores(item: LocomoQuestion, final: str, raw: str | None) -> dict[str, Any]:
    reference = format_gold_answer(item.gold_answer)
    trap = None if item.adversarial_answer is None else str(item.adversarial_answer)
    scores = {"final": score_answer(final, reference, expects_abstention=item.abstention_expected,
                                    adversarial_answer=trap)}
    if raw is not None:
        scores["raw"] = score_answer(raw, reference, expects_abstention=item.abstention_expected,
                                     adversarial_answer=trap)
    return scores


def evaluate_question(
    item: LocomoQuestion,
    workspace: Workspace,
    *,
    mode: str,
    top_k: int,
    depth: int,
    generate: bool,
    condition: str = CONDITION_NORMAL,
    oracle_message_ids: list[str] | None = None,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    row: dict[str, Any] = {
        "index": item.index,
        "questionId": f"Q-{item.index + 1:03d}",
        "question": item.question,
        "category": item.category,
        "condition": condition,
        "goldAnswer": format_gold_answer(item.gold_answer),
        "goldEvidence": list(item.annotation.gold_ids),
        "evidenceIssues": list(item.annotation.issues),
        "scorable": item.annotation.scorable,
        "sampleId": item.sample_id,
        "error": None,
        "generatedAnswer": "",
        "rawAnswer": None,
        "prompt": None,
        "promptMessageIds": [],
        "droppedMessageIds": [],
        "promptChars": None,
        "promptTruncated": None,
        "promptBudget": {},
        "oracleMissingMessageIds": [],
        "retrievedMemories": [],
        "rankedMessageIds": [],
        "metrics": score_retrieval([], item.annotation.gold_ids, workspace.conversation.conversation_id),
        "answerScore": None,
        "latencyMs": {"retrieval": None, "generation": None, "total": None},
        "generation": {},
    }
    if not item.question:
        row["error"] = "empty_question"
        return row
    started = time.perf_counter()
    try:
        retrieve_started = time.perf_counter()
        if condition == CONDITION_NORMAL:
            evidence, extra = _retrieve(workspace, item.question, mode, depth=depth, top_k=top_k)
        else:
            evidence, extra = _condition_evidence(workspace, item.question, condition, mode, oracle_message_ids)
        retrieval_ms = round((time.perf_counter() - retrieve_started) * 1000, 3)
        ranked = extra.get("rankedMessageIds") or ranked_message_ids(evidence.outcome.results)
        memories = extra.get("retrievedMemories")
        if memories is None:
            memories = [
                {
                    "rank": match.rank,
                    "memoryId": match.memory_id,
                    "score": match.score,
                    "similarity": match.similarity,
                    "adjustment": 0.0,
                    "content": match.statement,
                    "sourceMessageIds": list(match.source_message_ids),
                }
                for match in evidence.matches
            ]
        settings = parse_generation_overrides(None, load_generation_settings())
        ctx = build_answer_context(
            item.question, evidence, workspace.conversation,
            seed_ids=[match.memory_id for match in evidence.matches[: evidence.n_seed_matches]],
            expanded_ids=list(evidence.bprime.get("expanded_memory_ids") or []),
            max_context_chars=settings.max_context_chars,
            context_tokens=settings.num_ctx,
            output_tokens=settings.num_predict,
        )
        generation_ms = None
        if generate:
            generator = workspace._resolve_generator(settings)
            if generator is not None and hasattr(generator, "last_usage"):
                generator.last_usage = None
            generate_started = time.perf_counter()
            answer_text, generation_meta = generate_answer(ctx, settings, generator)
            generation_ms = round((time.perf_counter() - generate_started) * 1000, 3)
            usage = getattr(generator, "last_usage", None)
            if usage:
                generation_meta = {**generation_meta, "usage": dict(usage)}
        else:
            answer_text = ""
            generation_meta = {**settings.public_dict(), "status": "not_run"}
        answered = generate and generation_meta.get("status") == "ok"
        row.update({
            "generatedAnswer": answer_text,
            "rawAnswer": generation_meta.get("rawAnswer"),
            "prompt": ctx.prompt,
            "promptMessageIds": [excerpt.message_id for excerpt in ctx.excerpts],
            "droppedMessageIds": list(ctx.dropped_message_ids),
            "promptChars": ctx.prompt_chars,
            "promptTruncated": ctx.truncated,
            "promptBudget": dict(ctx.budget),
            "oracleMissingMessageIds": list(extra.get("oracleMissingMessageIds") or []),
            "retrievedMemories": memories,
            "retrievalTrace": extra.get("trace"),
            "rankedMessageIds": ranked,
            "metrics": score_retrieval(
                ranked, item.annotation.gold_ids, workspace.conversation.conversation_id,
            ),
            "answerScore": _answer_scores(item, answer_text, generation_meta.get("rawAnswer")) if answered else None,
            "generation": generation_meta,
            "latencyMs": {
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "total": round((time.perf_counter() - started) * 1000, 3),
            },
        })
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["latencyMs"]["total"] = round((time.perf_counter() - started) * 1000, 3)
    return row


def context_window_risk(usage: dict[str, Any] | None) -> bool | None:
    """True when the prompt plus the output budget may not fit Ollama's context window."""
    if not usage or usage.get("prompt_eval_count") is None or not usage.get("num_ctx"):
        return None
    return usage["prompt_eval_count"] + (usage.get("num_predict") or 0) >= usage["num_ctx"]


def question_record(
    row: dict[str, Any],
    run: dict[str, Any],
    *,
    provenance=None,
    known_issue: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One structured, JSON-serialisable record per evaluated question.

    Fields under ``evaluationOnly`` are labels and scores; none of them reaches retrieval or
    generation. ``provenance`` maps a message id to its source (session, turn, date).
    """
    generation = row.get("generation") or {}
    in_prompt = set(row.get("promptMessageIds") or [])
    retrieved = []
    for mem in row.get("retrievedMemories") or []:
        sources = list(mem.get("sourceMessageIds") or [])
        entry = {
            "rank": mem.get("rank"),
            "memoryId": mem.get("memoryId"),
            "score": mem.get("score"),
            "similarity": mem.get("similarity"),
            "adjustment": mem.get("adjustment"),
            "sourceMessageIds": sources,
            "inPrompt": any(mid in in_prompt for mid in sources),
        }
        if provenance is not None:
            entry["provenance"] = [provenance(mid) for mid in sources]
        retrieved.append(entry)
    usage = generation.get("usage")
    return {
        "dataset": run.get("dataset"),
        "sampleId": row.get("sampleId") or run.get("sampleId"),
        "conversationId": run.get("conversationId"),
        "questionId": row["questionId"],
        "questionType": row.get("category"),
        "condition": row.get("condition"),
        "question": row.get("question"),
        "rawAnswer": row.get("rawAnswer"),
        "finalAnswer": row.get("generatedAnswer"),
        "retrieved": retrieved,
        "promptMessageIds": list(row.get("promptMessageIds") or []),
        "droppedMessageIds": list(row.get("droppedMessageIds") or []),
        "promptChars": row.get("promptChars"),
        "promptTruncated": row.get("promptTruncated"),
        "promptBudget": row.get("promptBudget") or {},
        "prompt": row.get("prompt"),
        "oracleMissingMessageIds": list(row.get("oracleMissingMessageIds") or []),
        "retrievalTrace": row.get("retrievalTrace"),
        "generation": {
            "status": generation.get("status"),
            "model": generation.get("model"),
            "error": generation.get("error"),
            "usage": usage,
            "contextWindowRisk": context_window_risk(usage),
            "strippedCitations": len(generation.get("strippedCitations") or []),
        },
        "latencyMs": row.get("latencyMs"),
        "error": row.get("error"),
        "evaluationOnly": {
            "referenceAnswer": row.get("goldAnswer"),
            "goldEvidence": list(row.get("goldEvidence") or []),
            "evidenceIssues": list(row.get("evidenceIssues") or []),
            "scorable": row.get("scorable"),
            "retrievalMetrics": row.get("metrics"),
            "answerScore": row.get("answerScore"),
            "knownAnnotationIssue": known_issue,
        },
    }


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_known_issues(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Evaluation-only annotation manifest keyed by (sampleId, questionId). Gold labels are not edited."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(issue["sampleId"], issue["questionId"]): issue for issue in data.get("issues") or []}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    processed = [row for row in rows if row.get("error") != "empty_question"]
    errors = [row for row in rows if row.get("error")]
    scorable = [row for row in rows if row["scorable"] and not row.get("error")]
    summary = {
        "totalQuestions": len(rows),
        "processed": len(processed),
        "errors": len(errors),
        "scorable": len(scorable),
        "unscorable": len(rows) - len([row for row in rows if row["scorable"]]),
        "recallAt": {},
        "hitAt": {},
        "mrr": None,
        "latencyMs": {
            "retrieval": _mean([row["latencyMs"]["retrieval"] for row in rows if row["latencyMs"]["retrieval"] is not None]),
            "generation": _mean([row["latencyMs"]["generation"] for row in rows if row["latencyMs"]["generation"] is not None]),
            "total": _mean([row["latencyMs"]["total"] for row in rows if row["latencyMs"]["total"] is not None]),
        },
        "byCategory": {},
        "failures": [
            {
                "questionId": row["questionId"],
                "question": row["question"],
                "error": row.get("error"),
                "evidenceIssues": row.get("evidenceIssues") or [],
            }
            for row in rows
            if row.get("error") or row.get("evidenceIssues")
        ],
    }
    if scorable:
        for k in RETRIEVAL_KS:
            key = str(k)
            summary["recallAt"][key] = _mean([row["metrics"]["recallAt"][key] for row in scorable])
            summary["hitAt"][key] = _mean([float(row["metrics"]["hitAt"][key]) for row in scorable])
        summary["mrr"] = _mean([row["metrics"]["reciprocalRank"] for row in scorable])
    by_cat: dict[Any, list[dict[str, Any]]] = {}
    for row in scorable:
        by_cat.setdefault(row.get("category"), []).append(row)
    for category, group in sorted(by_cat.items(), key=lambda item: str(item[0])):
        summary["byCategory"][str(category)] = {
            "n": len(group),
            "recallAt": {str(k): _mean([row["metrics"]["recallAt"][str(k)] for row in group]) for k in RETRIEVAL_KS},
            "hitAt": {str(k): _mean([float(row["metrics"]["hitAt"][str(k)]) for row in group]) for k in RETRIEVAL_KS},
            "mrr": _mean([row["metrics"]["reciprocalRank"] for row in group]),
        }
    summary["answers"] = answer_summary(rows)
    return summary


def answer_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer-level counts. Abstention questions are scored, not skipped; token F1 is auxiliary."""
    answered = [row for row in rows if row.get("answerScore")]
    expected = [row for row in answered if row["answerScore"]["final"]["expectsAbstention"]]
    f1 = [row["answerScore"]["final"]["tokenF1"] for row in answered
          if row["answerScore"]["final"]["tokenF1"] is not None and not row["answerScore"]["final"]["expectsAbstention"]]
    return {
        "generated": len(answered),
        "rawDiffersFromFinal": sum(
            1 for row in answered if (row.get("rawAnswer") or "").strip() != (row.get("generatedAnswer") or "").strip()
        ),
        "abstentionExpected": len(expected),
        "abstentionCorrect": sum(1 for row in expected if row["answerScore"]["final"]["abstentionCorrect"]),
        "abstainedWhenAnswerExpected": sum(
            1 for row in answered
            if not row["answerScore"]["final"]["expectsAbstention"] and row["answerScore"]["final"]["abstained"]
        ),
        "adversarialAnswerMatched": sum(
            1 for row in answered if row["answerScore"]["final"]["adversarialAnswerMatched"]
        ),
        "promptTruncated": sum(1 for row in rows if row.get("promptTruncated")),
        "contextWindowRisk": sum(
            1 for row in rows if context_window_risk((row.get("generation") or {}).get("usage"))
        ),
        "meanTokenF1Auxiliary": _mean(f1) if f1 else None,
        "semanticCorrectness": "not judged here; use the records with a judge or human labels",
    }


def render_report(run: dict[str, Any]) -> str:
    summary = run["summary"]
    lines = [
        "# LoCoMo conv-26 RAG report",
        "",
        f"- Dataset: `{md_escape(run.get('dataset'))}`",
        f"- Conversation: `{md_escape(run.get('conversationId'))}` sample `{md_escape(run.get('sampleId'))}`",
        f"- Retrieval mode: `{md_escape(run.get('retrievalMode'))}` top-k={run.get('topK')} depth={run.get('depth')}",
        f"- Lexical weight: {run.get('lexicalWeight')}",
        f"- Context weight: {run.get('contextWeight', 0.0)}",
        f"- Generation: `{md_escape((run.get('generation') or {}).get('model'))}` status `{(run.get('generation') or {}).get('status')}`",
        f"- Gold answers and evidence IDs are evaluation-only.",
        "",
        "## Metrics",
        "",
        "Recall@k is the fraction of gold evidence IDs found in the top-k retrieved messages. "
        "A question with two gold IDs scores 0.5 if only one is retrieved. "
        "Hit@k is 1 if any gold ID is in the top k. "
        "MRR is the mean of 1 / rank of the first gold ID (0 if none). "
        "Unscorable questions (missing or malformed gold evidence) are processed when possible "
        "but excluded from Recall, Hit, and MRR.",
        "",
        f"- Questions: {summary['totalQuestions']} processed={summary['processed']} "
        f"scorable={summary['scorable']} unscorable={summary['unscorable']} errors={summary['errors']}",
    ]
    if summary["scorable"]:
        rec = summary["recallAt"]
        hit = summary["hitAt"]
        lines.append(
            f"- Recall@5={rec['5']:.3f} Recall@15={rec['15']:.3f} Recall@50={rec['50']:.3f}"
        )
        lines.append(
            f"- Hit@5={hit['5']:.3f} Hit@15={hit['15']:.3f} Hit@50={hit['50']:.3f} "
            f"MRR={summary['mrr']:.3f}"
        )
    else:
        lines.append("- Retrieval metrics: none (no scorable gold evidence)")
    lat = summary["latencyMs"]
    lines.append(
        f"- Latency ms: retrieval={lat['retrieval']:.1f} generation={lat['generation']:.1f} "
        f"total={lat['total']:.1f}"
    )
    if summary["byCategory"]:
        lines += ["", "| Category | n | Recall@5 | Recall@15 | Recall@50 | Hit@5 | Hit@15 | Hit@50 | MRR |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for category, row in summary["byCategory"].items():
            lines.append(
                f"| {md_escape(category)} | {row['n']} | "
                f"{row['recallAt']['5']:.3f} | {row['recallAt']['15']:.3f} | {row['recallAt']['50']:.3f} | "
                f"{row['hitAt']['5']:.3f} | {row['hitAt']['15']:.3f} | {row['hitAt']['50']:.3f} | "
                f"{row['mrr']:.3f} |"
            )
    answers = summary.get("answers") or {}
    if answers.get("generated"):
        lines += [
            "",
            f"- Condition: `{md_escape(run.get('condition', CONDITION_NORMAL))}`",
            f"- Answers generated={answers['generated']} rawDiffersFromFinal={answers['rawDiffersFromFinal']} "
            f"promptTruncated={answers['promptTruncated']} contextWindowRisk={answers['contextWindowRisk']}",
            f"- Abstention expected={answers['abstentionExpected']} correct={answers['abstentionCorrect']}; "
            f"abstained when an answer was expected={answers['abstainedWhenAnswerExpected']}; "
            f"adversarial answer given={answers['adversarialAnswerMatched']}",
            f"- Token F1 (auxiliary, not correctness)={answers['meanTokenF1Auxiliary']}",
        ]
    lines += ["", "## Failures and annotation issues", ""]
    if not summary["failures"]:
        lines.append("None.")
    else:
        for failure in summary["failures"]:
            issues = ", ".join(failure.get("evidenceIssues") or []) or "none"
            err = failure.get("error") or "annotation"
            lines.append(
                f"- `{failure['questionId']}` {md_escape(failure.get('question') or '(empty)')} "
                f"— {md_escape(err)}; issues: {md_escape(issues)}"
            )
    for row in run["questions"]:
        lines += [
            "",
            f"## {row['questionId']}",
            "",
            f"- Category: {md_escape(row.get('category'))}",
            f"- Scorable: {row.get('scorable')}",
        ]
        if row.get("error"):
            lines.append(f"- Error: {md_escape(row['error'])}")
        if row.get("evidenceIssues"):
            lines.append(f"- Evidence issues: {md_escape(', '.join(row['evidenceIssues']))}")
        m = row["metrics"]
        ranks = " ".join(
            f"{mid}={rank if rank is not None else 'absent'}"
            for mid, rank in (m.get("goldRanks") or {}).items()
        ) or "none"
        lines += [
            "",
            "### Question",
            "",
            md_fence(row.get("question") or ""),
            "",
            "### Gold answer",
            "",
            md_fence(row.get("goldAnswer") or ""),
            "",
            "### Generated answer",
            "",
            md_fence(row.get("generatedAnswer") or ""),
            "",
        ]
        if row.get("rawAnswer") is not None:
            lines += ["### Raw answer", "", md_fence(row.get("rawAnswer") or ""), ""]
        usage = (row.get("generation") or {}).get("usage") or {}
        score = (row.get("answerScore") or {}).get("final") or {}
        lines += [
            f"- Condition: {md_escape(row.get('condition', CONDITION_NORMAL))}",
            f"- Prompt: excerpts={len(row.get('promptMessageIds') or [])} chars={row.get('promptChars')} "
            f"truncated={row.get('promptTruncated')} promptTokens={usage.get('prompt_eval_count')} "
            f"numCtx={usage.get('num_ctx')}",
            f"- Answer score: abstained={score.get('abstained')} expectsAbstention={score.get('expectsAbstention')} "
            f"abstentionCorrect={score.get('abstentionCorrect')} tokenF1(aux)={score.get('tokenF1')}",
            f"- Gold evidence: {md_escape(', '.join(row.get('goldEvidence') or []) or 'none')}",
            f"- Gold ranks: {md_escape(ranks)}",
            f"- Recall@5={m['recallAt']['5']} Recall@15={m['recallAt']['15']} Recall@50={m['recallAt']['50']}",
            f"- Hit@5={m['hitAt']['5']} Hit@15={m['hitAt']['15']} Hit@50={m['hitAt']['50']} "
            f"first={m['firstExpectedRank']} MRR={m['reciprocalRank']}",
            f"- Latency ms: retrieval={row['latencyMs']['retrieval']} "
            f"generation={row['latencyMs']['generation']} total={row['latencyMs']['total']}",
            "",
            "### Retrieved memories",
            "",
        ]
        if not row.get("retrievedMemories"):
            lines.append("None.")
        else:
            lines += [
                "| Rank | Memory | Messages | Score | Similarity |",
                "|---:|---|---|---:|---:|",
            ]
            for mem in row["retrievedMemories"]:
                lines.append(
                    f"| {mem.get('rank')} | `{md_escape(mem.get('memoryId'))}` | "
                    f"{md_escape(', '.join(mem.get('sourceMessageIds') or []))} | "
                    f"{mem.get('score'):.4f} | {mem.get('similarity'):.4f} |"
                )
            for mem in row["retrievedMemories"]:
                lines += [
                    "",
                    f"**r{mem.get('rank')}** `{md_escape(mem.get('memoryId'))}`",
                    "",
                    md_fence(mem.get("content") or ""),
                ]
        lines += ["", "### Generator prompt", "", md_fence(row.get("prompt") or "")]
    lines.append("")
    return "\n".join(lines)


def write_report(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def run_evaluation(
    *,
    dataset: Path,
    db: Path,
    output: Path,
    retrieval_mode: str,
    top_k: int,
    depth: int,
    conversation_id: str | None,
    sample_id: str | None,
    conversation_index: int,
    generate: bool,
    limit: int | None = None,
    workspace: Workspace | None = None,
    condition: str = CONDITION_NORMAL,
    records_path: Path | None = None,
    known_issues: Path | None = None,
) -> dict[str, Any]:
    if retrieval_mode not in EVAL_RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode must be one of {sorted(EVAL_RETRIEVAL_MODES)}")
    records = load_locomo_records(dataset)
    record = select_record(records, sample_id=sample_id, index=conversation_index)
    owned = workspace
    if workspace is None:
        workspace = Workspace.from_sqlite(db, conversation_id=conversation_id, k=min(top_k, MAX_ANSWER_TOP_K))
        workspace._eval_retriever = build_memory_retriever(
            workspace.conversation, workspace.claims, workspace.relations,
            workspace.embedder(), retrieval_config(retrieval_mode),
        )
    else:
        if getattr(workspace, "_eval_retriever", None) is None:
            workspace._eval_retriever = build_memory_retriever(
                workspace.conversation, workspace.claims, workspace.relations,
                workspace.embedder(), retrieval_config(retrieval_mode),
            )
    message_ids = {message.message_id for message in workspace.conversation.messages}
    items = load_locomo_questions(record, message_ids)
    if limit is not None:
        items = items[:limit]
    # The oracle condition is the only place gold evidence IDs choose evidence, and only as message ids.
    rows = [
        evaluate_question(
            item, workspace, mode=retrieval_mode, top_k=top_k, depth=depth, generate=generate,
            condition=condition,
            oracle_message_ids=[mid for mid in item.annotation.gold_ids if mid in message_ids]
            if condition == CONDITION_ORACLE else None,
        )
        for item in items
    ]
    settings = load_generation_settings()
    run = {
        "dataset": str(dataset),
        "conversationId": workspace.conversation.conversation_id,
        "sampleId": record.get("sample_id"),
        "retrievalMode": retrieval_mode,
        "condition": condition,
        "topK": top_k,
        "depth": depth,
        "lexicalWeight": load_lexical_weight() if retrieval_mode == "hybrid" or retrieval_mode in GRAPH_MODES else 0.0,
        "contextWeight": load_context_weight() if retrieval_mode == "hybrid" or retrieval_mode in GRAPH_MODES else 0.0,
        "generation": settings.public_dict(),
        "questions": rows,
        "summary": aggregate(rows),
    }
    write_report(output, render_report(run))
    if records_path is not None:
        issues = load_known_issues(known_issues)
        write_records(records_path, [
            question_record(row, run, known_issue=issues.get((run["sampleId"], row["questionId"])))
            for row in rows
        ])
    if owned is None:
        store = getattr(workspace, "_store", None)
        if store is not None:
            store.close()
    return run


def main(argv: list[str] | None = None) -> int:
    load_root_env()
    parser = argparse.ArgumentParser(prog="python -m continuator_memory.webapi.locomo_report")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_LOCOMO)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retrieval-mode", default="baseline", choices=sorted(EVAL_RETRIEVAL_MODES))
    parser.add_argument("--top-k", type=int, default=None,
                        help="memories sent to the generator (default: CONTINUATOR_ANSWER_TOP_K or 25)")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="ranking depth for Recall@k")
    parser.add_argument("--conversation", help="SQLite conversation id")
    parser.add_argument("--sample-id", help="LoCoMo sample_id when the JSON has several conversations")
    parser.add_argument("--conversation-index", type=int, default=0)
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    parser.add_argument("--no-generate", action="store_true", help="retrieval-only (not the default)")
    parser.add_argument("--condition", default=CONDITION_NORMAL, choices=CONDITIONS,
                        help="evidence given to the generator: retrieved (normal), gold turns (oracle), none")
    parser.add_argument("--records", type=Path, help="write one structured JSON record per question (JSONL)")
    parser.add_argument("--known-issues", type=Path,
                        help="evaluation-only annotation manifest; flags records, never edits gold labels")
    args = parser.parse_args(argv)
    if (args.top_k is not None and args.top_k < 1) or args.depth < 1:
        raise SystemExit("top-k and depth must be >= 1")
    try:
        top_k = load_answer_top_k(args.top_k)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run = run_evaluation(
        dataset=args.dataset,
        db=args.db,
        output=args.output,
        retrieval_mode=args.retrieval_mode,
        top_k=top_k,
        depth=args.depth,
        conversation_id=args.conversation,
        sample_id=args.sample_id,
        conversation_index=args.conversation_index,
        generate=not args.no_generate,
        limit=args.limit,
        condition=args.condition,
        records_path=args.records,
        known_issues=args.known_issues,
    )
    summary = run["summary"]
    print(
        f"wrote {args.output} questions={summary['totalQuestions']} "
        f"scorable={summary['scorable']} errors={summary['errors']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
