"""Optional cross-encoder reranking of retrieved memories (experimental).

Off unless a model is attached or a dedicated mode loads one. Candidates from an unchanged
retrieval pool are scored as ``(question, passage)`` pairs and reordered by that score; ties
keep the retrieval order. Memories, message ids, provenance and retrieval scores are kept;
only ``rank`` changes. The model is loaded from the local Hugging Face cache only: nothing is
downloaded here.

Passage text modes
------------------
* ``source_message`` — original conversation message text (legacy harness default).
* ``retrieval_content`` — the text that was indexed/scored by the retriever (e.g. contextual
  ``[PREV]/[TARGET]/[NEXT]`` units). Evaluation identity still comes from
  ``source_message_ids`` (target-only for contextual modes).
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, Protocol

from continuator_memory.models import Conversation
from continuator_memory.v2.memory_retrieval import RetrievalOutcome

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_LABEL = "baseline_minilm"
RERANK_MODEL_ENV = "CONTINUATOR_RERANK_MODEL"
RERANK_DEVICE_ENV = "CONTINUATOR_RERANK_DEVICE"
RERANK_BATCH_SIZE_ENV = "CONTINUATOR_RERANK_BATCH_SIZE"
PassageSource = Literal["source_message", "retrieval_content"]

# Prepared for future cached-pool ablation. Weights are never downloaded here.
QWEN3_RERANKER_0_6B = "Qwen/Qwen3-Reranker-0.6B"
QWEN3_EVIDENCE_LABEL = "qwen3_evidence"
QWEN3_EVIDENCE_PROMPT_NAME = "evidence"

# Task-agnostic evidence instruction for Qwen3-Reranker. Do **not** substitute the
# model's default web-search instruction. Not LoCoMo-specific.
QWEN3_EVIDENCE_INSTRUCTION = (
    "Given an information need and a candidate text, determine whether "
    "the text contains useful evidence for satisfying that information need.\n"
    "\n"
    "Relevant evidence can be direct or partial. It may contain a factual "
    "statement, entity, attribute, date, relationship, explanation, quote, "
    "code detail, or one component of a larger answer.\n"
    "\n"
    "The candidate does not need to independently answer the entire query.\n"
    "\n"
    "Do not reward a passage merely because it discusses the same topic. "
    "Prefer concrete information that materially supports answering "
    "the information need."
)

# Explicit labels for the general pretrained reranker ablation.
# Selection remains CONTINUATOR_RERANK_MODEL; labels are identity metadata only.
KNOWN_RERANKER_LABELS: dict[str, str] = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "baseline_minilm",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "minilm_l12",
    "BAAI/bge-reranker-base": "bge_reranker_base",
    QWEN3_RERANKER_0_6B: QWEN3_EVIDENCE_LABEL,
}


class RerankerLoadError(RuntimeError):
    """The cross-encoder could not be loaded from the local cache."""


class PairScorer(Protocol):
    def score(self, question: str, passages: list[str]) -> list[float]:
        ...


def is_qwen3_reranker_model(model_name: str) -> bool:
    name = (model_name or "").strip()
    return name == QWEN3_RERANKER_0_6B or name.startswith("Qwen/Qwen3-Reranker")


def model_is_locally_available(model_name: str) -> bool:
    """True iff ``model_name`` resolves from a local path or the HF cache (no download)."""
    name = (model_name or "").strip()
    if not name:
        return False
    if Path(name).exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        found = try_to_load_from_cache(name, "config.json")
    except Exception:
        return False
    return isinstance(found, str)


def require_local_reranker_model(model_name: str) -> None:
    """Fail closed before any Hugging Face load/download attempt for Qwen3 rerankers."""
    if not is_qwen3_reranker_model(model_name):
        return
    if model_is_locally_available(model_name):
        return
    raise RerankerLoadError(
        f"{model_name} is not available locally.\nNo download was attempted."
    )


def cross_encoder_load_kwargs(model_name: str) -> dict[str, Any]:
    """Extra CrossEncoder constructor kwargs for model-specific formatting.

    Always pairs with ``local_files_only=True`` at the call site. MiniLM/BGE need no
    prompts; Qwen3-Reranker requires an explicit evidence instruction so the default
    web-search prompt is never used silently.
    """
    if is_qwen3_reranker_model(model_name):
        return {
            "prompts": {QWEN3_EVIDENCE_PROMPT_NAME: QWEN3_EVIDENCE_INSTRUCTION},
            "default_prompt_name": QWEN3_EVIDENCE_PROMPT_NAME,
        }
    return {}


class CrossEncoderReranker:
    """A Sentence Transformers cross-encoder, loaded with ``local_files_only`` (no downloads)."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, device: str = "cpu", batch_size: int = 16):
        # Strict pre-check for Qwen3: never reach CrossEncoder/from_pretrained if missing.
        require_local_reranker_model(model_name)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerLoadError(f"sentence-transformers is not installed: {exc}") from exc
        try:
            from torch import nn
            # Raw logits for every model: a sigmoid head would squash scores into ties near 0 or 1.
            # Compatible with MS MARCO MiniLM, BGE CrossEncoder-format, and Qwen3-Reranker
            # (ST CrossEncoder API; instruction via prompts= for Qwen only).
            load_kwargs = cross_encoder_load_kwargs(model_name)
            local_files_only = True
            if local_files_only is not True:
                raise RerankerLoadError("local_files_only must be True; downloads are forbidden")
            self._model = CrossEncoder(
                model_name,
                device=device,
                local_files_only=local_files_only,
                activation_fn=nn.Identity(),
                **load_kwargs,
            )
        except RerankerLoadError:
            raise
        except Exception as exc:
            raise RerankerLoadError(
                f"cross-encoder {model_name!r} is not available offline ({type(exc).__name__}: {exc}). "
                f"Download it once, separately, with: "
                f"hf download {model_name}  (or huggingface_hub.snapshot_download({model_name!r}))"
            ) from exc
        self.model_name = model_name
        self.label = reranker_label_for_model(model_name)
        self.device = device
        self.batch_size = batch_size
        self.prompt_name = (
            QWEN3_EVIDENCE_PROMPT_NAME if is_qwen3_reranker_model(model_name) else None
        )
        self.instruction = (
            QWEN3_EVIDENCE_INSTRUCTION if is_qwen3_reranker_model(model_name) else None
        )
        self.max_length = int(
            getattr(self._model, "max_seq_length", None) or self._model.max_length
        )
        self.revision = _cached_revision(model_name)

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict([(question, p) for p in passages], batch_size=self.batch_size,
                                     show_progress_bar=False, convert_to_numpy=True)
        return [float(s) for s in scores]

    def metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "model": self.model_name,
            "rerankerModel": self.model_name,
            "rerankerLabel": self.label,
            "revision": self.revision,
            "device": self.device,
            "maxLength": self.max_length,
            "batchSize": self.batch_size,
            "scores": "raw logits",
            "localFilesOnly": True,
        }
        if self.prompt_name is not None:
            meta["promptName"] = self.prompt_name
            meta["instruction"] = self.instruction
        return meta


def reranker_label_for_model(model_name: str) -> str:
    """Stable experiment label for a Hugging Face / local model id."""
    name = (model_name or "").strip()
    if not name:
        return DEFAULT_RERANKER_LABEL
    if name in KNOWN_RERANKER_LABELS:
        return KNOWN_RERANKER_LABELS[name]
    # Deterministic fallback for unexpected ids (never remaps to another known model).
    return "custom:" + name.replace("/", "__")


def load_rerank_model_name(env: dict[str, str] | None = None) -> str:
    """Configurable pretrained model id; default is a general MS MARCO MiniLM cross-encoder."""
    raw = ((env if env is not None else os.environ).get(RERANK_MODEL_ENV) or "").strip()
    return raw or DEFAULT_RERANK_MODEL


def load_reranker_label(env: dict[str, str] | None = None) -> str:
    """Label for the configured model (identity metadata; does not change scoring)."""
    return reranker_label_for_model(load_rerank_model_name(env))


def load_rerank_device(env: dict[str, str] | None = None) -> str:
    raw = ((env if env is not None else os.environ).get(RERANK_DEVICE_ENV) or "").strip()
    return raw or "cpu"


def load_rerank_batch_size(env: dict[str, str] | None = None) -> int:
    raw = ((env if env is not None else os.environ).get(RERANK_BATCH_SIZE_ENV) or "").strip()
    if not raw:
        return 16
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{RERANK_BATCH_SIZE_ENV} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{RERANK_BATCH_SIZE_ENV} must be >= 1")
    return value


def load_cross_encoder_reranker(env: dict[str, str] | None = None) -> CrossEncoderReranker:
    """Construct the configured pretrained reranker, or raise ``RerankerLoadError``.

    Failures are deterministic: there is no silent fallback to another model.
    """
    return CrossEncoderReranker(
        load_rerank_model_name(env),
        device=load_rerank_device(env),
        batch_size=load_rerank_batch_size(env),
    )


def _cached_revision(model_name: str) -> str | None:
    """Commit hash of the cached snapshot (None for a local path or an unknown cache layout)."""
    if Path(model_name).exists():
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        found = try_to_load_from_cache(model_name, "config.json")
    except Exception:
        return None
    return Path(found).parent.name if isinstance(found, str) else None


def candidate_text(result, messages: dict[str, str]) -> str:
    """The original source message text of a retrieved memory (its content if no source resolves)."""
    texts = [messages[mid] for mid in result.source_message_ids if mid in messages]
    return "\n".join(texts) if texts else result.content


def passage_for_result(
    result,
    messages: dict[str, str],
    *,
    passage_source: PassageSource = "source_message",
) -> str:
    if passage_source == "retrieval_content":
        return result.content
    if passage_source == "source_message":
        return candidate_text(result, messages)
    raise ValueError(f"unknown passage_source {passage_source!r}")


def rerank(
    question: str,
    outcome: RetrievalOutcome,
    conversation: Conversation,
    scorer: PairScorer,
    *,
    passage_source: PassageSource = "source_message",
) -> tuple[RetrievalOutcome, list[dict[str, Any]]]:
    """Reorder ``outcome`` by cross-encoder score without adding/removing candidates.

    Returns the new outcome and one score row per candidate. ``source_message_ids`` are
    preserved exactly; only ``rank`` (and row order) change.
    """
    messages = {m.message_id: m.content for m in conversation.messages}
    candidates = outcome.results
    passages = [
        passage_for_result(r, messages, passage_source=passage_source) for r in candidates
    ]
    scores = scorer.score(question, passages)
    if len(scores) != len(candidates):
        raise ValueError(f"reranker returned {len(scores)} scores for {len(candidates)} candidates")
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    results = tuple(replace(candidates[i], rank=rank) for rank, i in enumerate(order, 1))
    rows = [
        {
            "rank": rank,
            "memoryId": candidates[i].memory_id,
            "retrievalRank": candidates[i].rank,
            "retrievalScore": candidates[i].score,
            "rerankScore": scores[i],
            "sourceMessageIds": list(candidates[i].source_message_ids),
            "passageSource": passage_source,
        }
        for rank, i in enumerate(order, 1)
    ]
    return RetrievalOutcome(outcome.question, outcome.intent, results, outcome.excluded), rows
