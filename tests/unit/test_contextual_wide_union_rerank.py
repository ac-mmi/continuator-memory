"""Focused tests for wide-union + reduction ablation modes (no LoCoMo full eval)."""

from __future__ import annotations

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.memory import MemorySource
from continuator_memory.v2.memory_retrieval import RetrievalResult
from continuator_memory.v2.models import Claim, Provenance
from continuator_memory.webapi.contextual_union_rerank import (
    DEFAULT_CHANNEL_TOP_N,
    MODE_CONTEXTUAL_UNION_RERANK,
    retrieve_contextual_union_rerank_evidence,
)
from continuator_memory.webapi.contextual_wide_union_rerank import (
    DEFAULT_QUOTA_PER_CHANNEL,
    DEFAULT_REDUCED_TOP_N,
    DEFAULT_WIDE_CHANNEL_TOP_N,
    DEFAULT_WIDE_RRF_C,
    MODE_CONTEXTUAL_WIDE_UNION_QUOTA_RERANK,
    MODE_CONTEXTUAL_WIDE_UNION_RERANK,
    MODE_CONTEXTUAL_WIDE_UNION_RRF_RERANK,
    MODE_TO_REDUCER,
    WIDE_UNION_RERANK_MODES,
    apply_reduction,
    retrieve_contextual_wide_union_rerank_evidence,
    retrieve_for_mode,
)
from continuator_memory.webapi.locomo_report import ranked_message_ids, retrieval_config
from continuator_memory.webapi.workspace import (
    CONTEXTUAL_ENRICHMENT_MODES,
    RETRIEVAL_MODES,
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
            _msg("D1:5", 4, "Alice: piano recital notes"),
            _msg("D1:6", 5, "Bob: cooking pasta primavera"),
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


def test_mode_registration_and_defaults():
    for mode in WIDE_UNION_RERANK_MODES:
        assert mode in RETRIEVAL_MODES
        assert mode in CONTEXTUAL_ENRICHMENT_MODES
        assert normalize_retrieval_mode(mode) == mode
        cfg = retrieval_config(mode)
        assert cfg is not None
        assert cfg.context_weight == 0.0
    assert MODE_TO_REDUCER[MODE_CONTEXTUAL_WIDE_UNION_RERANK] == "none"
    assert MODE_TO_REDUCER[MODE_CONTEXTUAL_WIDE_UNION_RRF_RERANK] == "rrf"
    assert MODE_TO_REDUCER[MODE_CONTEXTUAL_WIDE_UNION_QUOTA_RERANK] == "channel_quota"
    assert DEFAULT_WIDE_CHANNEL_TOP_N == 200
    assert DEFAULT_REDUCED_TOP_N == 100
    assert DEFAULT_WIDE_RRF_C == 60
    assert DEFAULT_QUOTA_PER_CHANNEL == 50
    # Existing depth-50 union mode unchanged.
    assert MODE_CONTEXTUAL_UNION_RERANK in RETRIEVAL_MODES
    assert DEFAULT_CHANNEL_TOP_N == 50


def test_existing_union_mode_still_requests_depth_50(monkeypatch):
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]
    depths = []

    real_lex = __import__(
        "continuator_memory.webapi.contextual_union_rerank", fromlist=["retrieve_source_evidence"]
    ).retrieve_source_evidence
    real_bm = __import__(
        "continuator_memory.webapi.contextual_bm25", fromlist=["retrieve_contextual_bm25_evidence"]
    ).retrieve_contextual_bm25_evidence

    def wrap_lex(*args, **kwargs):
        depths.append(("lex", args[5]))
        return real_lex(*args, **kwargs)

    def wrap_bm(*args, **kwargs):
        depths.append(("bm", args[4]))
        return real_bm(*args, **kwargs)

    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_union_rerank.retrieve_source_evidence", wrap_lex,
    )
    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_union_rerank.retrieve_contextual_bm25_evidence",
        wrap_bm,
    )
    retrieve_contextual_union_rerank_evidence(
        "xylophone",
        conversation,
        claims,
        (),
        HashingEmbedder(),
        budget_chars=10_000,
        scorer=DictScorer({"xylophone": 1.0}),
        channel_top_n=DEFAULT_CHANNEL_TOP_N,
    )
    assert depths == [("lex", 50), ("bm", 50)]


def test_wide_retrieval_requests_200_from_each_channel(monkeypatch):
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]
    depths = []

    real_lex = __import__(
        "continuator_memory.webapi.contextual_wide_union_rerank", fromlist=["retrieve_source_evidence"]
    ).retrieve_source_evidence
    real_bm = __import__(
        "continuator_memory.webapi.contextual_bm25", fromlist=["retrieve_contextual_bm25_evidence"]
    ).retrieve_contextual_bm25_evidence

    def wrap_lex(*args, **kwargs):
        depths.append(("lex", args[5]))
        assert kwargs.get("contextual_enrichment") is True
        return real_lex(*args, **kwargs)

    def wrap_bm(*args, **kwargs):
        depths.append(("bm", args[4]))
        return real_bm(*args, **kwargs)

    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_wide_union_rerank.retrieve_source_evidence",
        wrap_lex,
    )
    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_wide_union_rerank.retrieve_contextual_bm25_evidence",
        wrap_bm,
    )
    evidence, meta = retrieve_contextual_wide_union_rerank_evidence(
        "xylophone",
        conversation,
        claims,
        (),
        HashingEmbedder(),
        budget_chars=10_000,
        scorer=DictScorer({"xylophone": 9.0, "gardening": 1.0}),
        channel_top_n=DEFAULT_WIDE_CHANNEL_TOP_N,
        reducer="none",
    )
    assert depths == [("lex", 200), ("bm", 200)]
    assert meta["phase1"]["channelTopN"] == 200
    assert meta["reduction"]["strategy"] == "none"
    assert meta["reduction"]["droppedCount"] == 0
    assert meta["rerank"]["candidatesIn"] == meta["union"]["unionSize"]
    assert meta["rerank"]["candidatesOut"] == len(evidence.outcome.results)


def test_stable_union_lexical_first_dedupe_memory_id():
    lexical = [
        _result(1, "A", "[TARGET] A"),
        _result(2, "B", "[TARGET] B lexical"),
        _result(3, "C", "[TARGET] C"),
    ]
    bm25 = [
        _result(1, "B", "[TARGET] B bm25"),
        _result(2, "D", "[TARGET] D"),
        _result(3, "E", "[TARGET] E"),
    ]
    # Import alias from wide module path via contextual_union helper used inside.
    from continuator_memory.webapi.contextual_union_rerank import stable_union_by_memory_id as union

    merged = union(lexical, bm25)
    assert [r.memory_id for r in merged] == [
        "mem:A:c0", "mem:B:c0", "mem:C:c0", "mem:D:c0", "mem:E:c0",
    ]
    assert merged[1].content == "[TARGET] B lexical"


def test_multi_source_preserved_through_wide_none(monkeypatch):
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]

    multi = _result(
        1,
        "D1:2",
        "[TARGET] rareterm xylophone swimming with kids",
        sources=(
            MemorySource("mem:D1:2:c0", "D1:2:c0", "D1:2", 0),
            MemorySource("mem:D1:2:c0", "D1:2:c1", "D1:5", 1),
        ),
    )
    other = _result(2, "D1:4", "[TARGET] gardening tomatoes")

    class LexEv:
        def __init__(self):
            from continuator_memory.v2.memory_retrieval import RetrievalOutcome, question_intent

            self.outcome = RetrievalOutcome("q", question_intent("q"), (multi, other))

    class BmEv:
        def __init__(self):
            from continuator_memory.v2.memory_retrieval import RetrievalOutcome, question_intent

            self.outcome = RetrievalOutcome("q", question_intent("q"), (other,))

    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_wide_union_rerank.retrieve_source_evidence",
        lambda *a, **k: LexEv(),
    )
    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_wide_union_rerank.retrieve_contextual_bm25_evidence",
        lambda *a, **k: (BmEv(), {"bm25": {"algorithm": "okapi"}}),
    )
    evidence, meta = retrieve_contextual_wide_union_rerank_evidence(
        "xylophone",
        conversation,
        claims,
        (),
        HashingEmbedder(),
        budget_chars=10_000,
        scorer=DictScorer({"xylophone": 5.0, "gardening": 1.0}),
        reducer="none",
        channel_top_n=10,
    )
    assert meta["union"]["multiSourceProvenanceAllowed"] is True
    multi_out = next(r for r in evidence.outcome.results if r.memory_id == "mem:D1:2:c0")
    assert multi_out.source_message_ids == ("D1:2", "D1:5")
    assert set(ranked_message_ids(evidence.outcome.results)) >= {"D1:2", "D1:5", "D1:4"}


def test_no_reduction_full_union_enters_ce():
    lexical = [_result(i, mid, f"[TARGET] {mid}") for i, mid in enumerate(["A", "B", "C"], 1)]
    bm25 = [_result(i, mid, f"[TARGET] {mid}") for i, mid in enumerate(["B", "D", "E"], 1)]
    from continuator_memory.webapi.contextual_union_rerank import stable_union_by_memory_id

    union = stable_union_by_memory_id(lexical, bm25)
    kept, dropped, meta = apply_reduction(
        "none", union, lexical, bm25,
        reduced_top_n=2, rrf_c=60, quota_per_channel=1,
    )
    assert len(kept) == len(union) == 5
    assert dropped == ()
    assert meta["droppedCount"] == 0
    assert meta["beforeCount"] == meta["afterCount"]


def test_rrf_reduction_deterministic_and_traced():
    lexical = [
        _result(1, "A", "[TARGET] A"),
        _result(2, "B", "[TARGET] B"),
        _result(3, "C", "[TARGET] C"),
        _result(4, "D", "[TARGET] D"),
    ]
    bm25 = [
        _result(1, "E", "[TARGET] E"),
        _result(2, "B", "[TARGET] B"),
        _result(3, "F", "[TARGET] F"),
        _result(4, "C", "[TARGET] C"),
    ]
    from continuator_memory.webapi.contextual_union_rerank import stable_union_by_memory_id

    union = stable_union_by_memory_id(lexical, bm25)
    kept1, dropped1, meta1 = apply_reduction(
        "rrf", union, lexical, bm25,
        reduced_top_n=3, rrf_c=DEFAULT_WIDE_RRF_C, quota_per_channel=1,
    )
    kept2, dropped2, meta2 = apply_reduction(
        "rrf", union, lexical, bm25,
        reduced_top_n=3, rrf_c=DEFAULT_WIDE_RRF_C, quota_per_channel=1,
    )
    assert [r.memory_id for r in kept1] == [r.memory_id for r in kept2]
    assert dropped1 == dropped2
    assert meta1["strategy"] == "rrf"
    assert meta1["afterCount"] == 3
    assert meta1["droppedCount"] == len(union) - 3
    assert meta1["beforeCount"] == meta1["afterCount"] + meta1["droppedCount"]
    assert set(meta1["droppedMemoryIds"]) == set(dropped1)
    assert len(kept1) + len(dropped1) == len(union)
    # Explicit reduction only — every kept id was in the union.
    assert set(r.memory_id for r in kept1) <= set(r.memory_id for r in union)


def test_quota_reduction_preserves_both_channels_deterministically():
    # Lexical-only deep candidate vs BM25-only deep candidate must both be reserved.
    lexical = [
        _result(1, "L1", "[TARGET] L1"),
        _result(2, "L2", "[TARGET] L2"),
        _result(3, "SHARED", "[TARGET] SHARED"),
        _result(4, "L_DEEP", "[TARGET] L_DEEP"),
    ]
    bm25 = [
        _result(1, "B1", "[TARGET] B1"),
        _result(2, "SHARED", "[TARGET] SHARED"),
        _result(3, "B_DEEP", "[TARGET] B_DEEP"),
        _result(4, "B2", "[TARGET] B2"),
    ]
    from continuator_memory.webapi.contextual_union_rerank import stable_union_by_memory_id

    union = stable_union_by_memory_id(lexical, bm25)
    kept, dropped, meta = apply_reduction(
        "channel_quota",
        union,
        lexical,
        bm25,
        reduced_top_n=4,
        rrf_c=60,
        quota_per_channel=2,
    )
    kept_ids = [r.memory_id for r in kept]
    assert meta["strategy"] == "channel_quota"
    assert meta["quotaPerChannel"] == 2
    assert len(kept_ids) == 4
    # Top-2 lexical novel: L1, L2; top-2 bm25 novel after lex: B1, SHARED(already)/B_DEEP
    assert "mem:L1:c0" in kept_ids
    assert "mem:L2:c0" in kept_ids
    assert "mem:B1:c0" in kept_ids
    # Deterministic second call.
    kept_b, _, _ = apply_reduction(
        "channel_quota", union, lexical, bm25,
        reduced_top_n=4, rrf_c=60, quota_per_channel=2,
    )
    assert [r.memory_id for r in kept] == [r.memory_id for r in kept_b]
    assert meta["beforeCount"] == meta["afterCount"] + meta["droppedCount"]


def test_cross_encoder_reorder_only_no_provenance_mutation(monkeypatch):
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]

    a = _result(
        1,
        "D1:2",
        "[TARGET] rareterm xylophone",
        sources=(
            MemorySource("mem:D1:2:c0", "c0", "D1:2", 0),
            MemorySource("mem:D1:2:c0", "c1", "D1:6", 1),
        ),
    )
    b = _result(2, "D1:4", "[TARGET] gardening tomatoes")

    class LexEv:
        def __init__(self):
            from continuator_memory.v2.memory_retrieval import RetrievalOutcome, question_intent

            self.outcome = RetrievalOutcome("q", question_intent("q"), (a, b))

    class BmEv:
        def __init__(self):
            from continuator_memory.v2.memory_retrieval import RetrievalOutcome, question_intent

            self.outcome = RetrievalOutcome("q", question_intent("q"), (b,))

    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_wide_union_rerank.retrieve_source_evidence",
        lambda *a, **k: LexEv(),
    )
    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_wide_union_rerank.retrieve_contextual_bm25_evidence",
        lambda *a, **k: (BmEv(), {"bm25": {}}),
    )
    # Prefer gardening so CE reorders.
    scorer = DictScorer({"gardening": 10.0, "xylophone": 1.0})
    evidence, meta = retrieve_contextual_wide_union_rerank_evidence(
        "gardening",
        conversation,
        claims,
        (),
        HashingEmbedder(),
        budget_chars=10_000,
        scorer=scorer,
        reducer="none",
        channel_top_n=10,
    )
    assert evidence.outcome.results[0].memory_id == "mem:D1:4:c0"
    multi = next(r for r in evidence.outcome.results if r.memory_id == "mem:D1:2:c0")
    assert multi.source_message_ids == ("D1:2", "D1:6")
    assert meta["rerank"]["candidatesIn"] == meta["rerank"]["candidatesOut"]
    assert meta["rerank"]["candidatePool"] == len(evidence.outcome.results)


def test_retrieve_for_mode_dispatches_reducers(monkeypatch):
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]
    seen = []

    def fake_retrieve(*args, **kwargs):
        seen.append(kwargs.get("reducer"))
        # Minimal valid return via real path with tiny channel depth would work;
        # short-circuit by calling real with mocked channels below.
        raise AssertionError("should be patched at retrieve_contextual_wide_union_rerank_evidence")

    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_wide_union_rerank.retrieve_contextual_wide_union_rerank_evidence",
        lambda *a, **k: (_DummyEvidence(), {"reduction": {"strategy": k["reducer"]}}),
    )

    class _DummyEvidence:
        pass

    for mode, expected in MODE_TO_REDUCER.items():
        _, meta = retrieve_for_mode(
            mode,
            "q",
            conversation,
            claims,
            (),
            HashingEmbedder(),
            1000,
        )
        assert meta["reduction"]["strategy"] == expected


def test_candidate_accounting_rrf_end_to_end(monkeypatch):
    conversation = _conversation()
    claims = [Claim(f"{m.message_id}:c0", m.content, Provenance.of(m)) for m in conversation.messages]
    lex = tuple(
        _result(i, mid, f"[TARGET] {mid}")
        for i, mid in enumerate(["A", "B", "C", "D", "E"], 1)
    )
    bm = tuple(
        _result(i, mid, f"[TARGET] {mid}")
        for i, mid in enumerate(["E", "F", "G", "B"], 1)
    )

    class LexEv:
        def __init__(self):
            from continuator_memory.v2.memory_retrieval import RetrievalOutcome, question_intent

            self.outcome = RetrievalOutcome("q", question_intent("q"), lex)

    class BmEv:
        def __init__(self):
            from continuator_memory.v2.memory_retrieval import RetrievalOutcome, question_intent

            self.outcome = RetrievalOutcome("q", question_intent("q"), bm)

    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_wide_union_rerank.retrieve_source_evidence",
        lambda *a, **k: LexEv(),
    )
    monkeypatch.setattr(
        "continuator_memory.webapi.contextual_wide_union_rerank.retrieve_contextual_bm25_evidence",
        lambda *a, **k: (BmEv(), {"bm25": {}}),
    )
    evidence, meta = retrieve_contextual_wide_union_rerank_evidence(
        "q",
        conversation,
        claims,
        (),
        HashingEmbedder(),
        budget_chars=10_000,
        scorer=DictScorer({}),
        reducer="rrf",
        channel_top_n=10,
        reduced_top_n=3,
    )
    assert meta["phase1"]["lexicalCandidates"] == 5
    assert meta["phase1"]["bm25Candidates"] == 4
    assert meta["union"]["unionSize"] == 7
    assert meta["reduction"]["beforeCount"] == 7
    assert meta["reduction"]["afterCount"] == 3
    assert meta["reduction"]["droppedCount"] == 4
    assert meta["rerank"]["candidatesIn"] == 3
    assert meta["rerank"]["candidatesOut"] == 3
    assert len(evidence.outcome.results) == 3


def test_no_gold_labels_in_wide_module_source():
    from pathlib import Path

    src = Path("src/continuator_memory/webapi/contextual_wide_union_rerank.py").read_text()
    for forbidden in ("goldEvidence", "gold_ids", "locomo10", "answer_category"):
        assert forbidden not in src
