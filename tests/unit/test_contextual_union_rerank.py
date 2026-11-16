"""Focused tests for contextual_union_rerank (no LoCoMo full eval)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.memory import MemorySource
from continuator_memory.v2.memory_retrieval import RetrievalResult
from continuator_memory.v2.models import Claim, Provenance
from continuator_memory.v2.rerank import RerankerLoadError
from continuator_memory.webapi.contextual_union_rerank import (
    DEFAULT_CHANNEL_TOP_N,
    MODE_CONTEXTUAL_UNION_RERANK,
    candidate_dedupe_key,
    retrieve_contextual_union_rerank_evidence,
    stable_union_by_memory_id,
    stable_union_by_target_message_id,
    target_message_id,
)
from continuator_memory.webapi.locomo_report import ranked_message_ids
from continuator_memory.webapi.locomo_report import retrieval_config
from continuator_memory.webapi.workspace import (
    CONTEXTUAL_ENRICHMENT_MODES,
    RETRIEVAL_MODES,
    Workspace,
    normalize_retrieval_mode,
)
from tests.helpers import HashingEmbedder
from tests.unit.test_contextual_lexical_rerank import DictScorer


def _msg(mid: str, position: int, text: str) -> Message:
    return Message(mid, "conv", position, "user", text, timestamp="2023-01-01T10:00:00")


def _conversation() -> Conversation:
    return Conversation(
        "conv",
        (
            _msg("D1:1", 0, "Alice: previous mention of hiking"),
            _msg("D1:2", 1, "Bob: rareterm xylophone swimming with kids"),
            _msg("D1:3", 2, "Alice: next acknowledgement"),
            _msg("D1:4", 3, "Bob: unrelated gardening tomatoes"),
        ),
        title="t",
    )


def _result(
    rank: int,
    mid: str,
    content: str,
    score: float = 0.5,
    *,
    sources: tuple[MemorySource, ...] | None = None,
    memory_id: str | None = None,
) -> RetrievalResult:
    mem_id = memory_id or f"mem:{mid}:c0"
    src = sources or (MemorySource(mem_id, f"{mid}:c0", mid, rank),)
    return RetrievalResult(
        rank=rank,
        memory_id=mem_id,
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
        sources=src,
        links=(),
    )


def test_mode_registration():
    assert MODE_CONTEXTUAL_UNION_RERANK in RETRIEVAL_MODES
    assert MODE_CONTEXTUAL_UNION_RERANK in CONTEXTUAL_ENRICHMENT_MODES
    assert normalize_retrieval_mode("contextual_union_rerank") == "contextual_union_rerank"
    assert DEFAULT_CHANNEL_TOP_N == 50
    cfg = retrieval_config("contextual_union_rerank")
    assert cfg is not None
    assert cfg.context_weight == 0.0
    assert cfg.lexical_weight == retrieval_config("contextual_lexical").lexical_weight


def test_stable_dedupe_lexical_then_novel_bm25():
    lexical = [
        _result(1, "A", "[TARGET] A"),
        _result(2, "B", "[TARGET] B"),
        _result(3, "C", "[TARGET] C"),
    ]
    bm25 = [
        _result(1, "B", "[TARGET] B from bm25"),
        _result(2, "D", "[TARGET] D"),
        _result(3, "E", "[TARGET] E"),
    ]
    union = stable_union_by_target_message_id(lexical, bm25)
    assert [target_message_id(r) for r in union] == ["A", "B", "C", "D", "E"]
    # Keep the first (lexical) instance of B, not the BM25 duplicate text.
    assert union[1].content == "[TARGET] B"
    assert [r.rank for r in union] == [1, 2, 3, 4, 5]


def test_target_identity_helper():
    enriched = _result(
        1,
        "D1:2",
        "[PREV] A\n\n[TARGET] B\n\n[NEXT] C",
    )
    assert target_message_id(enriched) == "D1:2"
    assert enriched.source_message_ids == ("D1:2",)


def test_union_rerank_uses_both_channels_and_complete_pool(monkeypatch):
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]

    calls = {"lexical": 0, "bm25": 0}

    real_lexical = __import__(
        "continuator_memory.webapi.contextual_union_rerank", fromlist=["retrieve_source_evidence"]
    ).retrieve_source_evidence
    real_bm25 = __import__(
        "continuator_memory.webapi.contextual_bm25", fromlist=["retrieve_contextual_bm25_evidence"]
    ).retrieve_contextual_bm25_evidence

    def wrap_lexical(*args, **kwargs):
        calls["lexical"] += 1
        assert kwargs.get("contextual_enrichment") is True
        assert args[5] == 3  # channel_top_n passed as k
        return real_lexical(*args, **kwargs)

    def wrap_bm25(*args, **kwargs):
        calls["bm25"] += 1
        assert args[4] == 3
        return real_bm25(*args, **kwargs)

    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_union_rerank.retrieve_source_evidence",
        wrap_lexical,
    )
    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_union_rerank.retrieve_contextual_bm25_evidence",
        wrap_bm25,
    )

    scorer = DictScorer({"xylophone": 9.0, "gardening": 5.0, "hiking": 1.0, "acknowledgement": 2.0})
    evidence, meta = retrieve_contextual_union_rerank_evidence(
        "rareterm xylophone",
        conversation,
        claims,
        (),
        HashingEmbedder(),
        budget_chars=10_000,
        scorer=scorer,
        channel_top_n=3,
    )
    assert calls["lexical"] == 1
    assert calls["bm25"] == 1
    assert meta["union"]["channelTopN"] == 3
    assert meta["union"]["truncatedBeforeRerank"] is False
    assert meta["rerank"]["passageSource"] == "retrieval_content"
    assert meta["rerank"]["candidatePool"] == len(evidence.outcome.results)
    # CE must see every unique pre-rerank candidate (complete union).
    assert len(scorer.seen[0]) == meta["union"]["unionSize"]
    assert meta["union"]["unionSize"] == meta["rerank"]["candidatePool"]
    for result in evidence.outcome.results:
        assert len(result.source_message_ids) == 1
        assert "[TARGET]" in result.content or result.content


def test_reranker_reorders_complete_union():
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]
    # Force a known pre-rerank order via a scorer that prefers gardening (D1:4).
    scorer = DictScorer({"gardening": 10.0, "xylophone": 1.0, "hiking": 0.5, "acknowledgement": 0.2})
    evidence, meta = retrieve_contextual_union_rerank_evidence(
        "gardening tomatoes",
        conversation,
        claims,
        (),
        HashingEmbedder(),
        budget_chars=10_000,
        scorer=scorer,
        channel_top_n=4,
    )
    assert evidence.outcome.results
    top = evidence.outcome.results[0]
    assert top.source_message_ids == ("D1:4",)
    # Ordering follows CE scores, not channel append order.
    assert [r.rank for r in evidence.outcome.results] == list(
        range(1, len(evidence.outcome.results) + 1)
    )
    assert meta["rerank"]["scores"][0]["sourceMessageIds"] == ["D1:4"]


def test_missing_reranker_fails_closed(monkeypatch):
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]

    def boom(env=None):
        raise RerankerLoadError("cross-encoder unavailable")

    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_union_rerank.load_cross_encoder_reranker",
        boom,
    )
    with pytest.raises(RerankerLoadError, match="unavailable"):
        retrieve_contextual_union_rerank_evidence(
            "swimming",
            conversation,
            claims,
            (),
            HashingEmbedder(),
            budget_chars=10_000,
            scorer=None,
            channel_top_n=3,
        )


def test_workspace_mode_and_existing_modes_unchanged():
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    baseline = workspace.ask("weekday bicycle", retrieval_mode="baseline", generate=False)
    lexical = workspace.ask("weekday bicycle", retrieval_mode="lexical", generate=False)
    contextual = workspace.ask("weekday bicycle", retrieval_mode="contextual_lexical", generate=False)
    bm25 = workspace.ask("weekday bicycle", retrieval_mode="contextual_bm25", generate=False)
    assert baseline["answer"]["evidence"]["diagnostics"]["mode"] == "baseline"
    assert lexical["answer"]["evidence"]["diagnostics"]["mode"] == "lexical"
    assert contextual["answer"]["evidence"]["diagnostics"]["mode"] == "contextual_lexical"
    assert bm25["answer"]["evidence"]["diagnostics"]["mode"] == "contextual_bm25"

    workspace._eval_reranker = DictScorer({"bicycle": 10.0, "weekday": 1.0})
    union = workspace.ask("weekday bicycle", retrieval_mode="contextual_union_rerank", generate=False)
    diagnostics = union["answer"]["evidence"]["diagnostics"]
    assert diagnostics["mode"] == "contextual_union_rerank"
    assert diagnostics["contextualEnrichment"] is True
    assert diagnostics["candidateUnion"] is True
    assert diagnostics["channelTopN"] == DEFAULT_CHANNEL_TOP_N
    assert diagnostics["rerank"]["passageSource"] == "retrieval_content"
    assert diagnostics["union"]["dedupeKey"] == "memory_id"
    assert diagnostics["union"]["multiSourceProvenanceAllowed"] is True


def test_dedupe_by_memory_id_ignores_retrieval_text_differences():
    left = _result(1, "D1:2", "[PREV] x\n\n[TARGET] Bob swimming\n\n[NEXT] y")
    right = _result(1, "D1:2", "[TARGET] Bob swimming only")  # same memory_id
    union = stable_union_by_memory_id([left], [right])
    assert len(union) == 1
    assert union[0].memory_id == left.memory_id
    assert candidate_dedupe_key(left) == "mem:D1:2:c0"


def test_multi_source_restatement_memory_is_accepted_and_fully_credited():
    """Regression for conv-47/48: consolidator KEEP attaches a second source message id."""
    # Exact pattern from failing artifacts: mem:D16:16:c0 → (D16:16, D17:37)
    multi = _result(
        1,
        "D16:16",
        "[TARGET] John: Take care, bye!",
        sources=(
            MemorySource("mem:D16:16:c0", "D16:16:c0", "D16:16", 363),
            MemorySource("mem:D16:16:c0", "D17:37:c0", "D17:37", 400),
        ),
    )
    single = _result(2, "D1:2", "[TARGET] Bob swimming")
    # Same multi-source memory from both channels must dedupe once without dropping IDs.
    bm_dup = replace(multi, rank=1, content="[TARGET] John: Take care, bye! (bm25)")
    union = stable_union_by_target_message_id([multi, single], [bm_dup])
    assert len(union) == 2
    kept = next(r for r in union if r.memory_id == "mem:D16:16:c0")
    assert kept.source_message_ids == ("D16:16", "D17:37")
    ranked = ranked_message_ids(union)
    assert ranked.count("D16:16") == 1
    assert "D17:37" in ranked
    assert "D1:2" in ranked


def test_retrieve_union_rerank_preserves_multi_source_provenance(monkeypatch):
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]

    multi = _result(
        1,
        "D1:2",
        "[PREV] hiking\n\n[TARGET] Bob: rareterm xylophone swimming with kids\n\n[NEXT] next",
        sources=(
            MemorySource("mem:D1:2:c0", "D1:2:c0", "D1:2", 1),
            MemorySource("mem:D1:2:c0", "D1:4:c0", "D1:4", 3),
        ),
    )
    other = _result(2, "D1:1", "[TARGET] Alice: previous mention of hiking")

    class LexEv:
        def __init__(self):
            from continuator_memory.v2.memory_retrieval import QuestionIntent, RetrievalOutcome

            self.outcome = RetrievalOutcome("q", QuestionIntent(), (multi, other))

    class BmEv:
        def __init__(self):
            from continuator_memory.v2.memory_retrieval import QuestionIntent, RetrievalOutcome

            self.outcome = RetrievalOutcome("q", QuestionIntent(), (multi,))

    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_union_rerank.retrieve_source_evidence",
        lambda *a, **k: LexEv(),
    )
    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_union_rerank.retrieve_contextual_bm25_evidence",
        lambda *a, **k: (BmEv(), {"bm25": {"algorithm": "okapi"}}),
    )

    scorer = DictScorer({"xylophone": 9.0, "hiking": 1.0})
    evidence, meta = retrieve_contextual_union_rerank_evidence(
        "xylophone",
        conversation,
        claims,
        (),
        HashingEmbedder(),
        budget_chars=10_000,
        scorer=scorer,
        channel_top_n=2,
    )
    assert meta["union"]["dedupeKey"] == "memory_id"
    assert meta["union"]["multiSourceProvenanceAllowed"] is True
    assert meta["union"]["unionSize"] == 2
    multi_hit = next(r for r in evidence.outcome.results if r.memory_id == "mem:D1:2:c0")
    assert multi_hit.source_message_ids == ("D1:2", "D1:4")
    # Evaluation expansion still credits both provenance ids.
    assert set(ranked_message_ids(evidence.outcome.results)) >= {"D1:2", "D1:4", "D1:1"}
