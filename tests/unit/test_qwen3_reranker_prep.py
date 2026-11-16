"""Prepare-only tests for Qwen3-Reranker-0.6B evidence ablation config.

Does NOT download weights, call snapshot_download, or run LoCoMo evaluation.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.memory import MemorySource
from continuator_memory.v2.memory_retrieval import QuestionIntent, RetrievalOutcome, RetrievalResult
from continuator_memory.v2.rerank import (
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANKER_LABEL,
    KNOWN_RERANKER_LABELS,
    QWEN3_EVIDENCE_INSTRUCTION,
    QWEN3_EVIDENCE_LABEL,
    QWEN3_EVIDENCE_PROMPT_NAME,
    QWEN3_RERANKER_0_6B,
    CrossEncoderReranker,
    RerankerLoadError,
    cross_encoder_load_kwargs,
    is_qwen3_reranker_model,
    model_is_locally_available,
    rerank,
    reranker_label_for_model,
    require_local_reranker_model,
)


def test_qwen_label_and_model_configuration():
    assert QWEN3_RERANKER_0_6B == "Qwen/Qwen3-Reranker-0.6B"
    assert QWEN3_EVIDENCE_LABEL == "qwen3_evidence"
    assert KNOWN_RERANKER_LABELS[QWEN3_RERANKER_0_6B] == "qwen3_evidence"
    assert reranker_label_for_model(QWEN3_RERANKER_0_6B) == "qwen3_evidence"
    assert is_qwen3_reranker_model(QWEN3_RERANKER_0_6B) is True
    assert is_qwen3_reranker_model(DEFAULT_RERANK_MODEL) is False


def test_existing_minilm_configuration_unchanged():
    assert DEFAULT_RERANK_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert DEFAULT_RERANKER_LABEL == "baseline_minilm"
    assert KNOWN_RERANKER_LABELS[DEFAULT_RERANK_MODEL] == "baseline_minilm"
    assert cross_encoder_load_kwargs(DEFAULT_RERANK_MODEL) == {}
    assert cross_encoder_load_kwargs("cross-encoder/ms-marco-MiniLM-L-12-v2") == {}
    assert cross_encoder_load_kwargs("BAAI/bge-reranker-base") == {}


def test_evidence_instruction_selected_for_qwen_not_web_search():
    kwargs = cross_encoder_load_kwargs(QWEN3_RERANKER_0_6B)
    assert kwargs["default_prompt_name"] == QWEN3_EVIDENCE_PROMPT_NAME
    assert kwargs["prompts"][QWEN3_EVIDENCE_PROMPT_NAME] == QWEN3_EVIDENCE_INSTRUCTION
    assert "web search query" not in QWEN3_EVIDENCE_INSTRUCTION.lower()
    assert "information need" in QWEN3_EVIDENCE_INSTRUCTION
    assert "useful evidence" in QWEN3_EVIDENCE_INSTRUCTION


def test_missing_local_qwen_model_fails_closed_without_download(monkeypatch):
    monkeypatch.setattr(
        "continuator_memory.v2.rerank.model_is_locally_available",
        lambda _name: False,
    )
    download_calls: list[str] = []

    def _forbid_snapshot(*_a, **_k):
        download_calls.append("snapshot_download")
        raise AssertionError("snapshot_download must not be called")

    def _forbid_from_pretrained(*_a, **_k):
        download_calls.append("from_pretrained")
        raise AssertionError("from_pretrained must not be called")

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        _forbid_snapshot,
        raising=False,
    )
    # Even if CrossEncoder were reached, forbid weight loads.
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        _forbid_from_pretrained,
        raising=False,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        _forbid_from_pretrained,
        raising=False,
    )

    with pytest.raises(RerankerLoadError, match="(?s)not available locally.*No download was attempted"):
        require_local_reranker_model(QWEN3_RERANKER_0_6B)
    with pytest.raises(RerankerLoadError, match="(?s)not available locally.*No download was attempted"):
        CrossEncoderReranker(QWEN3_RERANKER_0_6B)
    assert download_calls == []


def test_local_files_only_true_required_when_loading(monkeypatch):
    monkeypatch.setattr(
        "continuator_memory.v2.rerank.model_is_locally_available",
        lambda _name: True,
    )
    captured: dict[str, object] = {}

    class FakeCrossEncoder:
        def __init__(self, model_name_or_path, **kwargs):
            captured["model"] = model_name_or_path
            captured["kwargs"] = kwargs
            self.max_seq_length = 8192

        def predict(self, pairs, **kwargs):
            return [0.0] * len(pairs)

    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        FakeCrossEncoder,
    )
    # Avoid revision cache probe side effects.
    monkeypatch.setattr(
        "continuator_memory.v2.rerank._cached_revision",
        lambda _name: None,
    )

    reranker = CrossEncoderReranker(QWEN3_RERANKER_0_6B, device="cpu", batch_size=4)
    assert captured["model"] == QWEN3_RERANKER_0_6B
    assert captured["kwargs"]["local_files_only"] is True
    assert captured["kwargs"]["default_prompt_name"] == QWEN3_EVIDENCE_PROMPT_NAME
    assert captured["kwargs"]["prompts"][QWEN3_EVIDENCE_PROMPT_NAME] == QWEN3_EVIDENCE_INSTRUCTION
    assert reranker.label == "qwen3_evidence"
    assert reranker.metadata()["localFilesOnly"] is True
    assert reranker.metadata()["instruction"] == QWEN3_EVIDENCE_INSTRUCTION


def test_no_download_helpers_invoked_on_missing_cache(monkeypatch):
    """Requesting Qwen when absent must not touch HF download APIs."""
    monkeypatch.setattr(
        "continuator_memory.v2.rerank.model_is_locally_available",
        lambda _name: False,
    )
    sentinel = MagicMock(side_effect=AssertionError("download API called"))
    for target in (
        "huggingface_hub.snapshot_download",
        "huggingface_hub.hf_hub_download",
    ):
        monkeypatch.setattr(target, sentinel, raising=False)

    with pytest.raises(RerankerLoadError, match="No download was attempted"):
        CrossEncoderReranker(QWEN3_RERANKER_0_6B)
    assert sentinel.call_count == 0


def test_candidate_provenance_invariants_unchanged_with_qwen_scorer_metadata():
    """PairScorer reorder-only path is independent of which CE backend is configured."""

    def _result(rank: int, mid: str, score: float) -> RetrievalResult:
        return RetrievalResult(
            rank=rank,
            memory_id=f"mem-{mid}",
            conversation_id="c",
            content=f"normalized {mid}",
            similarity=score,
            adjustment=0.0,
            score=score,
            status="active",
            time_class="past",
            event_time=None,
            valid_from=None,
            valid_until=None,
            superseded_by=None,
            sources=(MemorySource(f"mem-{mid}", f"claim-{mid}", mid, rank),),
            links=(),
        )

    conversation = Conversation(
        "c",
        tuple(
            Message(mid, "c", i, "user", text, timestamp="2023-05-20")
            for i, (mid, text) in enumerate(
                [("m1", "I adopted a cat."), ("m2", "I moved to Lisbon."), ("m3", "Lisbon rent is high.")]
            )
        ),
    )
    before = RetrievalOutcome(
        "Where did I move?",
        QuestionIntent(),
        (_result(1, "m1", 0.9), _result(2, "m3", 0.8), _result(3, "m2", 0.7)),
        (),
    )

    class EvidenceAwareScorer:
        model_name = QWEN3_RERANKER_0_6B
        label = QWEN3_EVIDENCE_LABEL

        def score(self, question, passages):
            # Prefer Lisbon move passage.
            out = []
            for p in passages:
                out.append(5.0 if "Lisbon" in p and "moved" in p else 0.1)
            return out

        def metadata(self):
            return {
                "rerankerModel": self.model_name,
                "rerankerLabel": self.label,
                "instruction": QWEN3_EVIDENCE_INSTRUCTION,
                "localFilesOnly": True,
            }

    after, rows = rerank("Where did I move?", before, conversation, EvidenceAwareScorer())
    assert [r.memory_id for r in after.results] == ["mem-m2", "mem-m1", "mem-m3"]
    by_before = {r.memory_id: r for r in before.results}
    for result in after.results:
        original = by_before[result.memory_id]
        assert result.source_message_ids == original.source_message_ids
        assert replace(result, rank=original.rank) == original
    assert len(after.results) == len(before.results)
    assert {r.memory_id for r in after.results} == {r.memory_id for r in before.results}


def test_model_is_locally_available_reports_false_for_uncached_qwen():
    # Live cache inspection only — must not download. Model is expected absent in this env.
    assert model_is_locally_available(QWEN3_RERANKER_0_6B) is False
