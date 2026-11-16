"""LoCoMo report runner helpers. No MiniLM, Ollama, or the live locomo DB."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from continuator_memory.webapi.locomo_report import (
    ISSUE_EMPTY,
    ISSUE_NOT_LIST,
    ISSUE_NON_STRING,
    ISSUE_UNMATCHED,
    LocomoQuestion,
    aggregate,
    classify_evidence,
    evaluate_question,
    load_locomo_questions,
    load_locomo_records,
    md_escape,
    md_fence,
    render_report,
    retrieval_config,
    run_evaluation,
    score_retrieval,
    select_record,
)
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder

SEED_MSG = "m-commute"


def test_classify_evidence_malformed_and_unmatched():
    known = {"D1:3", "D2:8"}
    empty = classify_evidence([], known)
    assert empty.scorable is False and ISSUE_EMPTY in empty.issues
    assert classify_evidence(None, known).issues == [ISSUE_NOT_LIST]
    mixed = classify_evidence(["D1:3", 12, ""], known)
    assert mixed.gold_ids == ["D1:3"]
    assert ISSUE_NON_STRING in mixed.issues
    assert mixed.scorable is True
    unmatched = classify_evidence(["D8:6; D9:17"], known)
    assert unmatched.scorable is False
    assert ISSUE_UNMATCHED in unmatched.issues
    assert unmatched.gold_ids == ["D8:6; D9:17"]


def test_score_retrieval_fractional_recall_and_hit():
    scored = score_retrieval(["m-a", "m-gold", "m-b"], ["m-gold", "m-missing"], "conv")
    assert scored["firstExpectedRank"] == 2
    assert scored["reciprocalRank"] == pytest.approx(0.5)
    assert scored["recallAt"]["5"] == pytest.approx(0.5)
    assert scored["recallAt"]["15"] == pytest.approx(0.5)
    assert scored["recallAt"]["50"] == pytest.approx(0.5)
    assert scored["hitAt"]["5"] is True
    assert scored["goldRanks"] == {"m-gold": 2, "m-missing": None}


def test_score_retrieval_unscorable_gold_is_none():
    scored = score_retrieval(["m-a"], [], "conv")
    assert scored["recallAt"]["5"] is None
    assert scored["hitAt"]["5"] is None
    assert scored["reciprocalRank"] is None


def test_load_questions_preserves_index_and_int_answers(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(
        '[{"sample_id":"conv-26","qa":['
        '{"question":"When?","answer":2022,"evidence":["D1:12"],"category":2},'
        '{"question":"Bad","answer":"x","evidence":[],"category":3}'
        "]}]",
        encoding="utf-8",
    )
    record = select_record(load_locomo_records(path), sample_id="conv-26", index=0)
    items = load_locomo_questions(record, {"D1:12"})
    assert items[0].index == 0
    assert items[0].gold_answer == 2022
    assert items[0].annotation.scorable is True
    assert items[1].index == 1
    assert items[1].annotation.scorable is False


def test_markdown_escaping_and_fences():
    assert "\\|" in md_escape("a | b")
    fenced = md_fence("hello ``` world")
    assert fenced.startswith("````")
    assert "hello ``` world" in fenced


def test_render_report_separates_gold_and_generated():
    run = {
        "dataset": "locomo_conv26.json",
        "conversationId": "imported",
        "sampleId": "conv-26",
        "retrievalMode": "baseline",
        "topK": 15,
        "depth": 50,
        "lexicalWeight": 0.0,
        "generation": {"model": "qwen2.5:7b", "status": "ok"},
        "questions": [
            {
                "questionId": "Q-001",
                "index": 0,
                "question": "What pet does Caroline have?",
                "category": 4,
                "goldAnswer": "guinea pig",
                "generatedAnswer": "A dog [D1:1]",
                "goldEvidence": ["D13:3"],
                "evidenceIssues": [],
                "scorable": True,
                "error": None,
                "prompt": "Question: What pet does Caroline have?\nAnswer:",
                "retrievedMemories": [
                    {
                        "rank": 1,
                        "memoryId": "mem:D13:3:c0",
                        "score": 0.5,
                        "similarity": 0.5,
                        "content": "Oscar, my guinea pig",
                        "sourceMessageIds": ["D13:3"],
                    }
                ],
                "metrics": score_retrieval(["D13:3"], ["D13:3"], "imported"),
                "latencyMs": {"retrieval": 1.0, "generation": 2.0, "total": 3.0},
            },
            {
                "questionId": "Q-002",
                "index": 1,
                "question": "Would Melanie be considered an ally?",
                "category": 3,
                "goldAnswer": "",
                "generatedAnswer": "unknown",
                "goldEvidence": [],
                "evidenceIssues": [ISSUE_EMPTY],
                "scorable": False,
                "error": None,
                "prompt": "",
                "retrievedMemories": [],
                "metrics": score_retrieval([], [], "imported"),
                "latencyMs": {"retrieval": 1.0, "generation": 1.0, "total": 2.0},
            },
        ],
    }
    run["summary"] = aggregate(run["questions"])
    markdown = render_report(run)
    assert markdown.startswith("# LoCoMo conv-26 RAG report")
    assert "## Q-001" in markdown
    assert "### Gold answer" in markdown
    assert "guinea pig" in markdown
    assert "A dog [D1:1]" in markdown
    assert markdown.index("### Gold answer") < markdown.index("### Generated answer")
    assert "## Failures and annotation issues" in markdown
    assert "evidence_empty" in markdown
    assert run["summary"]["scorable"] == 1
    assert run["summary"]["unscorable"] == 1
    assert run["summary"]["recallAt"]["5"] == 1.0
    assert "- Context weight: 0.0" in markdown
    run["contextWeight"] = 0.5
    assert "- Context weight: 0.5" in render_report(run)


def test_context_weight_applies_to_hybrid_only(monkeypatch):
    monkeypatch.setenv("CONTINUATOR_CONTEXT_WEIGHT", "0.5")
    assert retrieval_config("hybrid").context_weight == 0.5
    assert retrieval_config("baseline").context_weight == 0.0
    assert retrieval_config("bprime").context_weight == 0.0
    monkeypatch.delenv("CONTINUATOR_CONTEXT_WEIGHT")
    assert retrieval_config("hybrid").context_weight == 0.0


def test_evaluate_question_records_error_and_continues():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    empty = LocomoQuestion(0, "", "a", 1, [], classify_evidence([], set()), None)
    failed = evaluate_question(empty, workspace, mode="baseline", top_k=1, depth=5, generate=False)
    assert failed["error"] == "empty_question"
    ok = LocomoQuestion(
        1, "weekday bicycle", "rides", 1, [SEED_MSG],
        classify_evidence([SEED_MSG], {SEED_MSG}), None,
    )
    result = evaluate_question(ok, workspace, mode="baseline", top_k=1, depth=5, generate=False)
    assert result["error"] is None
    assert result["generation"]["status"] == "not_run"
    assert result["question"] == "weekday bicycle"
    assert "rides" == result["goldAnswer"]


def test_run_evaluation_writes_report(tmp_path):
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    dataset = tmp_path / "locomo.json"
    dataset.write_text(
        '[{"sample_id":"fixture","qa":['
        f'{{"question":"weekday bicycle","answer":"rides","evidence":["{SEED_MSG}"],"category":1}},'
        '{"question":"no gold","answer":"x","evidence":[],"category":3},'
        '{"question":"missing id","answer":"x","evidence":["D8:6; D9:17"],"category":1}'
        "]}]",
        encoding="utf-8",
    )
    output = tmp_path / "report.md"
    run = run_evaluation(
        dataset=dataset,
        db=tmp_path / "unused.db",
        output=output,
        retrieval_mode="baseline",
        top_k=1,
        depth=5,
        conversation_id=None,
        sample_id=None,
        conversation_index=0,
        generate=False,
        workspace=workspace,
    )
    text = output.read_text(encoding="utf-8")
    assert output.is_file()
    assert run["summary"]["totalQuestions"] == 3
    assert run["summary"]["scorable"] == 1
    assert run["summary"]["unscorable"] == 2
    assert "## Q-001" in text and "## Q-002" in text and "## Q-003" in text
    assert "rides" in text
    assert "evidence_empty" in text
    assert "evidence_unmatched" in text
    assert run["questions"][0]["goldAnswer"] == "rides"
    assert run["questions"][0]["generatedAnswer"] == ""


def test_gold_is_not_passed_into_retrieval_or_generation():
    source = inspect.getsource(evaluate_question)
    assert "_retrieve(workspace, item.question" in source
    assert "gold_answer" not in source.split("_retrieve")[1].split("\n")[0]
    assert "goldEvidence" not in inspect.getsource(evaluate_question).split("build_answer_context")[1].split("\n")[0]


# ---- answer evidence count: selection is a prefix of the unchanged ranking -----------------------

def _ranked_dated_outcome(n: int = 30):
    from continuator_memory.models import Conversation, Message
    from continuator_memory.v2.memory import MemorySource
    from continuator_memory.v2.memory_retrieval import RetrievalOutcome, RetrievalResult, question_intent

    msgs = tuple(Message(f"T{i}", "dated-k", i, "user", f"Sam: I visited place {i} yesterday.",
                         f"2024-03-{(i % 28) + 1:02d}T10:00:00") for i in range(n))
    conv = Conversation("dated-k", msgs)
    order = list(reversed(range(n)))  # ranking deliberately differs from message order
    results = tuple(
        RetrievalResult(rank, f"mem:T{i}", "dated-k", msgs[i].content, 1.0 - rank / 100, 0.0, 1.0 - rank / 100,
                        "current", "static", None, None, None, None,
                        (MemorySource(f"mem:T{i}", f"c:T{i}", f"T{i}", i, msgs[i].timestamp),), ())
        for rank, i in enumerate(order, 1))
    question = "When did Sam visit place 3?"
    return question, conv, RetrievalOutcome(question, question_intent(question), results)


@pytest.mark.parametrize("k", [5, 15, 25])
def test_answer_evidence_is_the_top_k_prefix_with_ids_text_dates_and_notes(k):
    from continuator_memory.webapi.answering import build_answer_context, validate_answer_citations
    from continuator_memory.webapi.locomo_report import evidence_for_generation, ranked_message_ids

    question, conv, outcome = _ranked_dated_outcome()
    full_ranking = ranked_message_ids(outcome.results)
    evidence = evidence_for_generation(question, conv, outcome, k=k, budget_chars=50_000)
    ctx = build_answer_context(question, evidence, conv, seed_ids=[], expanded_ids=[], max_context_chars=12_000)
    assert ranked_message_ids(outcome.results) == full_ranking  # the ranking itself is untouched
    ids = [e.message_id for e in ctx.excerpts]
    assert ids == full_ranking[:k] and not ctx.truncated
    by_id = {m.message_id: m for m in conv.messages}
    for e in ctx.excerpts:
        msg = by_id[e.message_id]
        assert e.content == msg.content and e.timestamp == msg.timestamp
        assert e.date_note and "yesterday" in e.date_note
        assert f"[{e.message_id}] (message date: {e.message_date}) {e.content}" in ctx.prompt
    # a date claim cites its own excerpt and keeps the citation; a citation outside the top k is stripped
    first = ctx.excerpts[0]
    day = first.date_resolutions[0].value
    outside = full_ranking[k] if k < len(full_ranking) else None
    answer = f"Sam visited place {first.message_id[1:]} on {day} [{first.message_id}]."
    if outside:
        answer += f" Also [{outside}]."
    _, meta = validate_answer_citations(answer, ctx.excerpts, question=question)
    kept = {c["id"]: c["kept"] for c in meta["citations"]}
    assert kept[first.message_id] is True
    if outside:
        assert kept[outside] is False and meta["citations"][1]["reason"] == "not_in_prompt"


def test_multiquery_and_iterative_score_fused_ranking_not_original_outcome(monkeypatch):
    """Eval metrics must use diagnostics rankedMessageIds (fused), not evidence.outcome (original)."""
    from continuator_memory.v2.source_retrieval import SourceEvidence, SourceSelection
    from continuator_memory.v2.memory_retrieval import RetrievalOutcome, question_intent

    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=2)
    gold = [SEED_MSG]
    item = LocomoQuestion(
        0, "weekday bicycle", "rides", 1, gold,
        classify_evidence(gold, {m.message_id for m in workspace.conversation.messages}),
    )
    question = item.question
    original = RetrievalOutcome(question, question_intent(question), ())
    # Empty matches: scoring must come solely from rankedMessageIds in extra.
    empty_selection = SourceSelection(workspace.budget_chars, (), (), (), 0, {})
    evidence = SourceEvidence(question, original, (), empty_selection, (), n_seed_matches=0)

    def fake_mq(*_a, depth=None, **_k):
        assert depth == 50
        return evidence, {"rankedMessageIds": [SEED_MSG, "other"], "perQueryK": depth}

    def fake_it(*_a, depth=None, **_k):
        assert depth == 50
        return evidence, {"rankedMessageIds": [SEED_MSG], "perQueryK": depth, "stopReason": "sufficient"}

    monkeypatch.setattr(
        "continuator_memory.webapi.locomo_report.retrieve_multiquery_evidence", fake_mq
    )
    mq = evaluate_question(item, workspace, mode="multiquery", top_k=15, depth=50, generate=False)
    assert mq["error"] is None
    assert mq["rankedMessageIds"] == [SEED_MSG, "other"]
    assert mq["metrics"]["hitAt"]["5"] is True
    assert mq["metrics"]["availableDepth"] == 2

    monkeypatch.setattr(
        "continuator_memory.webapi.locomo_report.retrieve_iterative_evidence", fake_it
    )
    it = evaluate_question(item, workspace, mode="iterative", top_k=15, depth=50, generate=False)
    assert it["error"] is None
    assert it["rankedMessageIds"] == [SEED_MSG]
    assert it["metrics"]["hitAt"]["5"] is True


def test_hybrid_answer_top_k_15_to_25_changes_only_the_prompt(monkeypatch, tmp_path):
    """--top-k only cuts the already-ranked list for the answerer; depth-50 hybrid retrieval is unchanged."""
    from continuator_memory.webapi.locomo_report import build_memory_retriever

    monkeypatch.setenv("CONTINUATOR_LEXICAL_WEIGHT", "0.25")
    monkeypatch.setenv("CONTINUATOR_CONTEXT_WEIGHT", "0.5")
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=5)
    workspace.import_text("\n\n".join(f"You: I visited museum number {i} with my sister" for i in range(40)),
                          filename="c.txt", extract="messages", db_path=tmp_path / "w.db")
    workspace._eval_retriever = build_memory_retriever(
        workspace.conversation, workspace.claims, workspace.relations, workspace.embedder(),
        retrieval_config("hybrid"))
    gold = [workspace.conversation.messages[7].message_id]
    item = LocomoQuestion(0, "Which museum did I visit with my sister?", None, 1, gold,
                          classify_evidence(gold, {m.message_id for m in workspace.conversation.messages}))
    a = evaluate_question(item, workspace, mode="hybrid", top_k=15, depth=50, generate=False)
    b = evaluate_question(item, workspace, mode="hybrid", top_k=25, depth=50, generate=False)
    assert a["error"] is None and b["error"] is None
    assert len(a["rankedMessageIds"]) == 40
    for key in ("rankedMessageIds", "retrievedMemories", "metrics"):
        assert a[key] == b[key], key

    def prompt_ids(row):
        return [line.split("]")[0][1:] for line in row["prompt"].splitlines() if line.startswith("[imported:")]

    assert prompt_ids(a) == a["rankedMessageIds"][:15]
    assert prompt_ids(b) == b["rankedMessageIds"][:25]


def test_report_cli_resolves_top_k_default_and_explicit(monkeypatch, tmp_path):
    from continuator_memory.webapi import locomo_report as lr

    seen = {}
    monkeypatch.setattr(lr, "run_evaluation", lambda **kw: seen.update(kw) or {
        "summary": {"totalQuestions": 0, "scorable": 0, "errors": 0}})
    monkeypatch.setattr(lr, "load_root_env", lambda: None)
    monkeypatch.delenv("CONTINUATOR_ANSWER_TOP_K", raising=False)
    out = str(tmp_path / "r.md")
    assert lr.main(["--output", out]) == 0 and seen["top_k"] == 25 and seen["depth"] == 50
    assert lr.main(["--output", out, "--top-k", "15"]) == 0 and seen["top_k"] == 15
    monkeypatch.setenv("CONTINUATOR_ANSWER_TOP_K", "10")
    assert lr.main(["--output", out]) == 0 and seen["top_k"] == 10
    assert lr.main(["--output", out, "--top-k", "3"]) == 0 and seen["top_k"] == 3
    with pytest.raises(SystemExit):
        lr.main(["--output", out, "--top-k", "40"])
