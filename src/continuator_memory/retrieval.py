"""Brute-force cosine-similarity retrieval over locally stored embeddings."""

from __future__ import annotations

import numpy as np

from continuator_memory.embedding import Embedder
from continuator_memory.models import RetrievedChunk
from continuator_memory.store import Store


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a (d,) query and each row of an (n, d) matrix.

    Zero vectors get similarity 0.
    """
    query = np.asarray(query, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
        raise ValueError(f"shape mismatch: query {query.shape}, matrix {matrix.shape}")
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)


def top_k(scores: np.ndarray, ids: list[str], k: int) -> list[int]:
    """Indices of the ``k`` highest scores; ties are broken by ascending id."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(scores) != len(ids):
        raise ValueError("scores and ids must have the same length")
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
    return order[:k]


class Retriever:
    def __init__(self, store: Store, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self, question: str, k: int, conversation_id: str | None = None
    ) -> list[RetrievedChunk]:
        chunk_ids, matrix = self.store.load_embeddings(self.embedder.model_name, conversation_id)
        if not chunk_ids:
            return []
        query = self.embedder.embed([question])[0]
        scores = cosine_similarity(query, matrix)
        selected = top_k(scores, chunk_ids, k)
        chunks = {c.chunk_id: c for c in self.store.get_chunks([chunk_ids[i] for i in selected])}
        results = []
        for rank, i in enumerate(selected, start=1):
            chunk = chunks[chunk_ids[i]]
            results.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    conversation_id=chunk.conversation_id,
                    source_message_ids=chunk.source_message_ids,
                    score=float(scores[i]),
                    text=chunk.text,
                )
            )
        return results
