"""Local text embedding."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

_LOG = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim)."""
        ...


class EmbeddingLoadError(RuntimeError):
    """Raised when the embedding model cannot be loaded for retrieval."""

    def __init__(self, message: str, *, reason: str = "unavailable"):
        super().__init__(message)
        self.reason = reason  # unavailable | network | configuration | dependency


def classify_embedding_load_failure(model_name: str, exc: BaseException) -> EmbeddingLoadError:
    """Map a raw loader exception to an actionable :class:`EmbeddingLoadError`."""
    text = f"{type(exc).__name__}: {exc}".lower()
    detail = str(exc).strip() or type(exc).__name__

    if isinstance(exc, ModuleNotFoundError) or "sentence_transformers" in text or "no module named" in text:
        return EmbeddingLoadError(
            "embedding dependency is not installed (sentence-transformers). "
            "Install Continuator with its default dependencies: pip install -e .",
            reason="dependency",
        )
    if any(
        token in text
        for token in (
            "403",
            "401",
            "forbidden",
            "offline",
            "connection",
            "timed out",
            "timeout",
            "name or service not known",
            "nodename nor servname",
            "failed to establish",
            "network is unreachable",
            "temporary failure in name resolution",
            "huggingface_hub.errors",
            "localentrynotfound",
            "repository not found",
        )
    ):
        return EmbeddingLoadError(
            f"embedding model {model_name!r} is not available (network or Hub access failed: {detail}). "
            "Ensure network access for the first download, or pre-cache the model "
            f"(for example: hf download {model_name}). "
            "Override with CONTINUATOR_EMBEDDING_MODEL if you use a different local model.",
            reason="network",
        )
    if any(token in text for token in ("invalid", "config", "not a valid model", "oserror")):
        return EmbeddingLoadError(
            f"embedding model {model_name!r} could not be loaded ({detail}). "
            "Check CONTINUATOR_EMBEDDING_MODEL or clear a corrupt Hugging Face cache entry.",
            reason="configuration",
        )
    return EmbeddingLoadError(
        f"embedding model {model_name!r} is unavailable ({detail}). "
        "Download or cache the model, then retry. "
        f"Example: hf download {model_name}",
        reason="unavailable",
    )


class SentenceTransformerEmbedder:
    """Sentence Transformers model run on CPU, producing L2-normalized vectors.

    The model is downloaded/loaded in ``__init__`` (first use). Failures raise
    :class:`EmbeddingLoadError` with an actionable message — callers must not
    fall back to a different retrieval algorithm.
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise classify_embedding_load_failure(model_name, exc) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except EmbeddingLoadError:
            raise
        except Exception as exc:
            _LOG.exception("failed to load embedding model %s", model_name)
            raise classify_embedding_load_failure(model_name, exc) from exc

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
