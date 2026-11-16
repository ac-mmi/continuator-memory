"""Focused tests for contextual_lexical_rerank (no LoCoMo full eval)."""

from __future__ import annotations

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.memory_retrieval import QuestionIntent, RetrievalConfig, RetrievalOutcome, RetrievalResult
from continuator_memory.v2.memory import MemorySource
from continuator_memory.v2.models import Claim, Provenance
from continuator_memory.v2.rerank import RerankerLoadError, passage_for_result, rerank
from continuator_memory.webapi.contextual_rerank import (
    MODE_CONTEXTUAL_LEXICAL_RERANK,
    apply_retrieval_content_rerank,
    evidence_from_outcome,
    retrieve_contextual_lexical_rerank_evidence,
)
from continuator_memory.webapi.locomo_report import ranked_message_ids, retrieval_config
from continuator_memory.webapi.multiquery import build_memory_retriever
from continuator_memory.webapi.workspace import (
    CONTEXTUAL_ENRICHMENT_MODES,
    RETRIEVAL_MODES,
    Workspace,
    normalize_retrieval_mode,
)
from tests.helpers import HashingEmbedder


def _msg(mid: str, position: int, text: str) -> Message:
    return Message(mid, "conv", position, "user", text, timestamp="2023-01-01T10:00:00")


def _conversation() -> Conversation:
    return Conversation(
        "conv",
        (
            _msg("D1:1", 0, "Alice: previous mention of hiking"),
            _msg("D1:2", 1, "Bob: target swimming with kids"),
            _msg("D1:3", 2, "Alice: next acknowledgement"),
        ),
        title="t",
    )


class DictScorer:
    def __init__(self, by_substr: dict[str, float]):
        self.by_substr = by_substr
        self.seen: list[list[str]] = []

    def score(self, question: str, passages: list[str]) -> list[float]:
        self.seen.append(list(passages))
        out = []
        for p in passages:
            score = 0.0
            for key, value in self.by_substr.items():
                if key in p:
                    score = max(score, value)
            out.append(score)
        return out

    def metadata(self) -> dict:
        return {"model": "dict-scorer", "device": "cpu"}


def _result(rank: int, mid: str, content: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        memory_id=f"mem:{mid}:c0",
        conversation_id="conv",
        content=content,
        similarity=score,
        adjustment=0.0,
        score=score,
        status="current",
        time_class="unspecified",
        event_time=None,
        valid_from=None,
        valid_until=None,
        superseded_by=None,
        sources=(MemorySource(f"mem:{mid}:c0", f"{mid}:c0", mid, rank),),
        links=(),
    )


def test_mode_registration_and_config():
    assert MODE_CONTEXTUAL_LEXICAL_RERANK in RETRIEVAL_MODES
    assert MODE_CONTEXTUAL_LEXICAL_RERANK in CONTEXTUAL_ENRICHMENT_MODES
    assert normalize_retrieval_mode("contextual_lexical_rerank") == "contextual_lexical_rerank"
    cfg = retrieval_config("contextual_lexical_rerank")
    assert cfg is not None
    assert cfg.context_weight == 0.0
    assert cfg.lexical_weight == retrieval_config("contextual_lexical").lexical_weight


def test_candidate_set_preserved_and_ordered_by_score():
    before = RetrievalOutcome(
        "q",
        QuestionIntent(),
        (
            _result(1, "D1:1", "[TARGET] Alice: previous mention of hiking", 0.9),
            _result(2, "D1:2", "[PREV] x\n\n[TARGET] Bob: target swimming with kids\n\n[NEXT] y", 0.8),
            _result(3, "D1:3", "[TARGET] Alice: next acknowledgement", 0.7),
        ),
    )
    scorer = DictScorer({"swimming": 5.0, "hiking": 1.0, "acknowledgement": 2.0})
    after, rows = rerank(
        "swimming?",
        before,
        _conversation(),
        scorer,
        passage_source="retrieval_content",
    )
    assert [r.memory_id for r in after.results] == [
        "mem:D1:2:c0",
        "mem:D1:3:c0",
        "mem:D1:1:c0",
    ]
    assert sorted(r.memory_id for r in after.results) == sorted(r.memory_id for r in before.results)
    assert [r.rank for r in after.results] == [1, 2, 3]
    assert rows[0]["rerankScore"] == 5.0
    assert rows[0]["sourceMessageIds"] == ["D1:2"]
    assert rows[0]["passageSource"] == "retrieval_content"
    # Passages were enriched retrieval content, not raw isolated messages alone.
    assert any("[TARGET]" in p for p in scorer.seen[0])


def test_source_identity_preserved_with_contextual_passages():
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]
    consolidator = consolidate("conv", claims)
    original_sources = {
        mid: tuple(s.message_id for s in consolidator.sources if s.memory_id == mid)
        for mid in consolidator.memories
    }
    retriever = build_memory_retriever(
        conversation,
        claims,
        (),
        HashingEmbedder(),
        RetrievalConfig(lexical_weight=0.25, context_weight=0.0),
        consolidator=consolidator,
        contextual_enrichment=True,
    )
    outcome = retriever.search("swimming", k=3, conversation_id="conv")
    assert outcome.results
    for result in outcome.results:
        assert "[TARGET]" in result.content
        assert result.source_message_ids == (consolidator.memories[result.memory_id].message_id,)

    scorer = DictScorer({"swimming": 9.0, "hiking": 1.0, "acknowledgement": 2.0})
    evidence = evidence_from_outcome("swimming", conversation, outcome, budget_chars=10_000)
    rebuilt, rows = apply_retrieval_content_rerank(
        "swimming", conversation, evidence, scorer, budget_chars=10_000,
    )
    assert ranked_message_ids(rebuilt.outcome.results) == [
        mid for r in rebuilt.outcome.results for mid in r.source_message_ids
    ]
    for result in rebuilt.outcome.results:
        assert result.source_message_ids == original_sources[result.memory_id]
        assert len(result.source_message_ids) == 1
    assert {tuple(row["sourceMessageIds"]) for row in rows} == {
        result.source_message_ids for result in rebuilt.outcome.results
    }


def test_passage_source_retrieval_content_vs_source_message():
    result = _result(
        1,
        "D1:2",
        "[PREV] Alice: previous mention of hiking\n\n[TARGET] Bob: target swimming with kids\n\n[NEXT] Alice: next acknowledgement",
        0.5,
    )
    messages = {m.message_id: m.content for m in _conversation().messages}
    assert "hiking" in passage_for_result(result, messages, passage_source="retrieval_content")
    assert passage_for_result(result, messages, passage_source="source_message") == (
        "Bob: target swimming with kids"
    )


def test_existing_modes_unaffected():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    baseline = workspace.ask("weekday bicycle", retrieval_mode="baseline", generate=False)
    lexical = workspace.ask("weekday bicycle", retrieval_mode="lexical", generate=False)
    contextual = workspace.ask("weekday bicycle", retrieval_mode="contextual_lexical", generate=False)
    assert baseline["answer"]["evidence"]["diagnostics"]["mode"] == "baseline"
    assert lexical["answer"]["evidence"]["diagnostics"]["mode"] == "lexical"
    assert contextual["answer"]["evidence"]["diagnostics"]["mode"] == "contextual_lexical"
    assert "rerank" not in contextual["answer"]["evidence"]["diagnostics"]


def test_workspace_rerank_mode_with_fake_scorer():
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]
    evidence, meta = retrieve_contextual_lexical_rerank_evidence(
        "swimming",
        conversation,
        claims,
        (),
        HashingEmbedder(),
        k=3,
        budget_chars=10_000,
        scorer=DictScorer({"swimming": 5.0}),
    )
    assert meta["rerank"]["passageSource"] == "retrieval_content"
    assert meta["rerank"]["candidatePool"] == len(evidence.outcome.results)
    assert all(len(r.source_message_ids) == 1 for r in evidence.outcome.results)

    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=2)
    workspace._eval_reranker = DictScorer({"bicycle": 10.0, "weekday": 1.0})
    result = workspace.ask("weekday bicycle", retrieval_mode="contextual_lexical_rerank", generate=False)
    diagnostics = result["answer"]["evidence"]["diagnostics"]
    assert diagnostics["mode"] == "contextual_lexical_rerank"
    assert diagnostics["contextualEnrichment"] is True
    assert diagnostics["rerank"]["passageSource"] == "retrieval_content"


def test_missing_reranker_fails_closed(monkeypatch):
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]

    def boom(env=None):
        raise RerankerLoadError("cross-encoder unavailable")

    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_rerank.load_cross_encoder_reranker",
        boom,
    )
    with pytest.raises(RerankerLoadError, match="unavailable"):
        retrieve_contextual_lexical_rerank_evidence(
            "swimming",
            conversation,
            claims,
            (),
            HashingEmbedder(),
            k=3,
            budget_chars=10_000,
            scorer=None,
        )
