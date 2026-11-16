"""Focused tests for general pretrained reranker model ablation (config + invariants).

Does not run LoCoMo evaluation and does not select models using LoCoMo labels.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.memory import MemorySource
from continuator_memory.v2.memory_retrieval import QuestionIntent, RetrievalOutcome, RetrievalResult
from continuator_memory.v2.rerank import (
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANKER_LABEL,
    KNOWN_RERANKER_LABELS,
    RERANK_MODEL_ENV,
    CrossEncoderReranker,
    RerankerLoadError,
    load_cross_encoder_reranker,
    load_rerank_model_name,
    load_reranker_label,
    rerank,
    reranker_label_for_model,
)


def _result(rank: int, mid: str, score: float, *, sources: tuple[str, ...] | None = None) -> RetrievalResult:
    sids = sources or (mid,)
    mem_id = f"mem-{mid}"
    return RetrievalResult(
        rank=rank,
        memory_id=mem_id,
        conversation_id="c",
        content=f"content {mid}",
        similarity=score,
        adjustment=0.0,
        score=score,
        status="active",
        time_class="past",
        event_time=None,
        valid_from=None,
        valid_until=None,
        superseded_by=None,
        sources=tuple(MemorySource(mem_id, f"claim-{s}", s, i) for i, s in enumerate(sids, 1)),
        links=(),
    )


CONVERSATION = Conversation(
    "c",
    tuple(
        Message(mid, "c", i, "user", text, timestamp="2023-05-20")
        for i, (mid, text) in enumerate(
            [
                ("m1", "I adopted a cat."),
                ("m2", "I moved to Lisbon."),
                ("m3", "Lisbon rent is high."),
                ("m4", "Weather chat."),
            ]
        )
    ),
)


class DictScorer:
    def __init__(self, scores_by_passage: dict[str, float], *, model: str = "custom", label: str = "custom"):
        self.scores_by_passage = scores_by_passage
        self.model_name = model
        self.label = label

    def score(self, question: str, passages: list[str]) -> list[float]:
        return [self.scores_by_passage.get(p, 0.0) for p in passages]

    def metadata(self) -> dict:
        return {
            "model": self.model_name,
            "rerankerModel": self.model_name,
            "rerankerLabel": self.label,
        }


def test_default_model_and_label():
    assert load_rerank_model_name({}) == DEFAULT_RERANK_MODEL
    assert load_reranker_label({}) == DEFAULT_RERANKER_LABEL
    assert DEFAULT_RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert DEFAULT_RERANKER_LABEL == "baseline_minilm"
    assert KNOWN_RERANKER_LABELS[DEFAULT_RERANK_MODEL] == "baseline_minilm"


def test_model_configuration_selection(monkeypatch):
    monkeypatch.setenv(RERANK_MODEL_ENV, "cross-encoder/ms-marco-MiniLM-L-12-v2")
    assert load_rerank_model_name() == "cross-encoder/ms-marco-MiniLM-L-12-v2"
    assert load_reranker_label() == "minilm_l12"

    env = {RERANK_MODEL_ENV: "BAAI/bge-reranker-base"}
    assert load_rerank_model_name(env) == "BAAI/bge-reranker-base"
    assert load_reranker_label(env) == "bge_reranker_base"


def test_unknown_model_gets_deterministic_custom_label_not_another_known_model():
    label = reranker_label_for_model("org/some-other-reranker")
    assert label == "custom:org__some-other-reranker"
    assert label not in {"baseline_minilm", "minilm_l12", "bge_reranker_base"}


def test_load_cross_encoder_uses_configured_model_identity(monkeypatch):
    """Requested model is selected; metadata records model + label (no silent remap)."""
    monkeypatch.setenv(RERANK_MODEL_ENV, "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = load_cross_encoder_reranker()
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert reranker.label == "baseline_minilm"
    meta = reranker.metadata()
    assert meta["rerankerModel"] == reranker.model_name
    assert meta["rerankerLabel"] == "baseline_minilm"
    assert meta["model"] == reranker.model_name


def test_candidate_identity_preservation_across_reorder():
    before = RetrievalOutcome(
        "Where did I move?",
        QuestionIntent(),
        (
            _result(1, "m1", 0.9, sources=("m1", "m4")),
            _result(2, "m3", 0.8),
            _result(3, "m2", 0.7),
        ),
        (("x", "superseded"),),
    )
    scorer = DictScorer(
        {
            "I adopted a cat.\nWeather chat.": 1.0,
            "Lisbon rent is high.": 2.0,
            "I moved to Lisbon.": 9.0,
        },
        model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        label="minilm_l12",
    )
    after, rows = rerank("Where did I move?", before, CONVERSATION, scorer)
    assert len(after.results) == len(before.results) == 3
    assert {r.memory_id for r in after.results} == {r.memory_id for r in before.results}
    by_before = {r.memory_id: r for r in before.results}
    for result in after.results:
        original = by_before[result.memory_id]
        assert result.source_message_ids == original.source_message_ids
        assert replace(result, rank=original.rank) == original
    assert [r.memory_id for r in after.results] == ["mem-m2", "mem-m3", "mem-m1"]
    assert [row["sourceMessageIds"] for row in rows] == [
        list(by_before[mid].source_message_ids) for mid in ("mem-m2", "mem-m3", "mem-m1")
    ]


def test_failed_model_load_does_not_silently_fallback(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv(RERANK_MODEL_ENV, "continuator-test/no-such-cross-encoder-ablation")
    with pytest.raises(RerankerLoadError, match="(?s)not available offline.*continuator-test"):
        load_cross_encoder_reranker()
    # Still points at the requested (missing) model, not the default.
    assert load_rerank_model_name() == "continuator-test/no-such-cross-encoder-ablation"
    assert load_reranker_label() == "custom:continuator-test__no-such-cross-encoder-ablation"


def test_baseline_minilm_label_and_default_unchanged():
    assert reranker_label_for_model("cross-encoder/ms-marco-MiniLM-L-6-v2") == "baseline_minilm"
    assert load_rerank_model_name({RERANK_MODEL_ENV: ""}) == DEFAULT_RERANK_MODEL
    assert load_rerank_model_name({}) == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_question_record_includes_reranker_identity():
    from continuator_memory.webapi.locomo_report import question_record

    run = {
        "dataset": "locomo10.json",
        "sampleId": "conv-26",
        "conversationId": "c",
        "rerankModel": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "rerankerModel": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "rerankerLabel": "minilm_l12",
    }
    row = {
        "questionId": "q1",
        "category": 1,
        "condition": "standard",
        "sampleId": "conv-26",
        "question": "Where?",
        "retrievedMemories": [],
        "promptMessageIds": [],
        "goldEvidence": [],
        "scorable": True,
        "metrics": {},
        "generation": {"status": "not_run"},
    }
    record = question_record(row, run)
    assert record["rerankerModel"] == "cross-encoder/ms-marco-MiniLM-L-12-v2"
    assert record["rerankerLabel"] == "minilm_l12"


def test_smoke_ablation_models_same_candidates_different_scores():
    """Same passages scored by each ablation model; identities unchanged, scores may differ."""
    models = [
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "baseline_minilm"),
        ("cross-encoder/ms-marco-MiniLM-L-12-v2", "minilm_l12"),
        ("BAAI/bge-reranker-base", "bge_reranker_base"),
    ]
    question = "Where did Bob go on holiday?"
    passages = [
        "Bob flew to Paris for two weeks.",
        "Alice adopted a cat named Miso.",
        "The weather in Lisbon was rainy.",
    ]
    before = RetrievalOutcome(
        question,
        QuestionIntent(),
        tuple(_result(i, mid, 1.0 - 0.1 * i) for i, mid in enumerate(["m2", "m1", "m3"], 1)),
        (),
    )
    # Map retrieval contents to conversation source texts used by default passage mode.
    expected_ids = [r.memory_id for r in before.results]
    expected_sources = [list(r.source_message_ids) for r in before.results]

    scored: dict[str, list[float]] = {}
    for model_name, label in models:
        try:
            reranker = CrossEncoderReranker(model_name, device="cpu", batch_size=4)
        except RerankerLoadError as exc:
            pytest.fail(f"smoke load failed for {model_name}: {exc}")
        assert reranker.model_name == model_name
        assert reranker.label == label
        after, rows = rerank(question, before, CONVERSATION, reranker)
        assert [r.memory_id for r in after.results]  # non-empty
        assert len(after.results) == len(before.results)
        assert {r.memory_id for r in after.results} == set(expected_ids)
        by_id = {r.memory_id: r for r in after.results}
        for mid, sids in zip(expected_ids, expected_sources, strict=True):
            assert list(by_id[mid].source_message_ids) == sids
        scored[label] = [float(row["rerankScore"]) for row in rows]
        assert len(scored[label]) == 3

    # At least two models should not produce identical score vectors (capability difference).
    vectors = list(scored.values())
    assert not all(v == vectors[0] for v in vectors[1:]), scored
