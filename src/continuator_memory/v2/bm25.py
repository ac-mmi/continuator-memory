"""General-purpose Okapi BM25 sparse retrieval over arbitrary retrieval units.

This is true sparse BM25 — not dense cosine + lexical-overlap bonus (product ``lexical``).

Formula and defaults match the common ``rank_bm25.BM25Okapi`` convention:

* ``k1 = 1.5``
* ``b = 0.75``
* ``epsilon = 0.25`` (floor for negative IDF terms)

Tokenization is general English-oriented (lowercase alphanumerics via ``v2.lexical.tokenize``).
It does not depend on benchmark labels, gold evidence, or dataset-specific vocabularies.

Units are opaque ``RetrievalUnit`` records::

    id                  — opaque unit identity (e.g. memory_id)
    retrieval_content   — text indexed by BM25
    source_message_ids  — evaluation/provenance ids (caller-defined; BM25 never invents them)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from continuator_memory.v2.lexical import tokenize

# Okapi BM25 defaults aligned with rank_bm25.BM25Okapi (not tuned on any benchmark).
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_EPSILON = 0.25


@dataclass(frozen=True)
class RetrievalUnit:
    """One sparse-retrieval document with caller-owned provenance."""

    id: str
    retrieval_content: str
    source_message_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BM25Hit:
    unit: RetrievalUnit
    score: float
    rank: int


def sparse_tokenize(text: str) -> list[str]:
    """General-purpose sparse tokenizer (same surface forms as product lexical tokenization)."""
    return tokenize(text or "")


class BM25Index:
    """In-memory Okapi BM25 index over ``RetrievalUnit`` documents."""

    def __init__(
        self,
        units: Sequence[RetrievalUnit],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        epsilon: float = DEFAULT_EPSILON,
    ):
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.units: tuple[RetrievalUnit, ...] = tuple(units)
        self._corpus_tokens: list[list[str]] = [sparse_tokenize(u.retrieval_content) for u in self.units]
        self._doc_len = [len(toks) for toks in self._corpus_tokens]
        self.n_docs = len(self.units)
        self.avgdl = (sum(self._doc_len) / self.n_docs) if self.n_docs else 0.0
        self._df: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []
        for toks in self._corpus_tokens:
            counts: dict[str, int] = {}
            for tok in toks:
                counts[tok] = counts.get(tok, 0) + 1
            self._tf.append(counts)
            for tok in counts:
                self._df[tok] = self._df.get(tok, 0) + 1
        self._idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        """IDF with rank_bm25-style negative-IDF floor (``epsilon`` * mean positive IDF)."""
        idf: dict[str, float] = {}
        n = self.n_docs
        for term, df in self._df.items():
            # log((N - df + 0.5) / (df + 0.5) + 1) — same as rank_bm25.BM25Okapi
            idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        positive = [v for v in idf.values() if v > 0]
        average = (sum(positive) / len(positive)) if positive else 0.0
        eps = self.epsilon * average
        return {term: (val if val > 0 else eps) for term, val in idf.items()}

    def score_document(self, query_tokens: Sequence[str], doc_index: int) -> float:
        if self.n_docs == 0 or not query_tokens:
            return 0.0
        tf = self._tf[doc_index]
        dl = self._doc_len[doc_index]
        score = 0.0
        avgdl = self.avgdl or 1.0
        for term in query_tokens:
            freq = tf.get(term)
            if not freq:
                continue
            idf = self._idf.get(term, 0.0)
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            score += idf * (freq * (self.k1 + 1.0) / denom)
        return score

    def search(self, query: str, k: int) -> list[BM25Hit]:
        """Return top-``k`` units by BM25 score (stable ties: unit id)."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if self.n_docs == 0:
            return []
        q_tokens = sparse_tokenize(query)
        if not q_tokens:
            return []
        scored: list[tuple[float, str, int]] = []
        for i in range(self.n_docs):
            score = self.score_document(q_tokens, i)
            if score <= 0.0:
                continue
            scored.append((score, self.units[i].id, i))
        scored.sort(key=lambda row: (-row[0], row[1]))
        hits: list[BM25Hit] = []
        for rank, (score, _uid, idx) in enumerate(scored[:k], 1):
            hits.append(BM25Hit(unit=self.units[idx], score=score, rank=rank))
        return hits


def build_index(units: Iterable[RetrievalUnit], **kwargs) -> BM25Index:
    return BM25Index(list(units), **kwargs)
