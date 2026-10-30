"""Lexical overlap helpers and experimental hybrid ranking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from continuator_memory.v2.lexical import (
    DEFAULT_LEXICAL_WEIGHT,
    content_stems,
    fuse_scores,
    lexical_overlap,
    load_lexical_weight,
    question_stems,
    stem,
    tokenize,
)
from continuator_memory.v2.memory import STATUS_CURRENT, Memory, MemorySource
from continuator_memory.v2.memory_retrieval import MemoryRetriever, RetrievalConfig

QUESTION = "What did Alex research?"
RELEVANT = "Alex: Researching adoption agencies — it's been a dream to have a family."
EDUCATION = "Alex: I am considering education and career options after school."
CONVERSATIONAL = "Alex: Cool! What did it look like?"


class ScriptedEmbedder:
    """Assigns known unit vectors so cosine values are exact."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = {text: np.asarray(vec, dtype=np.float64) for text, vec in vectors.items()}

    def embed(self, texts):
        missing = [text for text in texts if text not in self.vectors]
        if missing:
            raise KeyError(f"no scripted vector for {missing}")
        return np.stack([self.vectors[text] for text in texts])


def _unit(cosine: float) -> np.ndarray:
    return np.array([cosine, math.sqrt(1.0 - cosine * cosine)], dtype=np.float64)


def _memory(n: int, content: str) -> Memory:
    return Memory(
        f"mem:m{n}:c0", "conv", content, STATUS_CURRENT, f"m{n}:c0", f"m{n}", n,
        time_class="present",
    )


def test_tokenize_lowercases_and_splits():
    assert tokenize("What did Alex research?") == ["what", "did", "alex", "research"]


def test_stem_matches_research_and_researching():
    assert stem("research") == "research"
    assert stem("Researching") == "research"
    assert stem("agencies") == "agency"


def test_stop_words_and_question_entities_are_ignored():
    stems = question_stems(QUESTION)
    assert "what" not in stems
    assert "did" not in stems
    assert "alex" not in stems
    assert stems == {"research"}


def test_content_stems_keep_memory_words():
    assert "research" in content_stems(RELEVANT)
    assert "research" not in content_stems(EDUCATION)
    assert "research" not in content_stems(CONVERSATIONAL)


def test_lexical_overlap_is_fraction_of_question_stems():
    assert lexical_overlap(QUESTION, RELEVANT) == 1.0
    assert lexical_overlap(QUESTION, EDUCATION) == 0.0
    assert lexical_overlap(QUESTION, CONVERSATIONAL) == 0.0
    assert lexical_overlap("What did she do?", RELEVANT) == 0.0


def test_fuse_scores_keeps_baseline_when_weight_is_zero():
    assert fuse_scores(0.459, 1.0, weight=0.0) == pytest.approx(0.459)
    assert fuse_scores(0.459, 1.0, weight=0.25, adjustment=0.0) == pytest.approx(0.709)
    with pytest.raises(ValueError, match="lexical weight"):
        fuse_scores(0.5, 1.0, weight=-0.1)


def test_load_lexical_weight_defaults_and_reads_env(monkeypatch):
    monkeypatch.delenv("CONTINUATOR_LEXICAL_WEIGHT", raising=False)
    assert load_lexical_weight() == DEFAULT_LEXICAL_WEIGHT
    monkeypatch.setenv("CONTINUATOR_LEXICAL_WEIGHT", "0.4")
    assert load_lexical_weight() == 0.4


def test_hybrid_ranking_promotes_researching_over_distractors():
    query = _unit(1.0)
    vectors = {
        QUESTION: query,
        RELEVANT: _unit(0.459),
        EDUCATION: _unit(0.504),
        CONVERSATIONAL: _unit(0.594),
    }
    memories = [_memory(1, EDUCATION), _memory(2, CONVERSATIONAL), _memory(3, RELEVANT)]
    sources = [MemorySource(m.memory_id, m.claim_id, m.message_id, m.position, None)
               for m in memories]

    baseline = MemoryRetriever(ScriptedEmbedder(vectors), RetrievalConfig(k=3, lexical_weight=0.0))
    baseline.index(memories, sources)
    base = baseline.retrieve(QUESTION)
    assert [r.content for r in base] == [CONVERSATIONAL, EDUCATION, RELEVANT]
    assert [round(r.similarity, 3) for r in base] == [0.594, 0.504, 0.459]
    assert [round(r.score, 3) for r in base] == [0.594, 0.504, 0.459]

    hybrid = MemoryRetriever(
        ScriptedEmbedder(vectors), RetrievalConfig(k=3, lexical_weight=DEFAULT_LEXICAL_WEIGHT),
    )
    hybrid.index(memories, sources)
    fused = hybrid.retrieve(QUESTION)
    assert fused[0].content == RELEVANT
    assert fused[0].rank == 1
    assert fused[0].score == pytest.approx(0.459 + DEFAULT_LEXICAL_WEIGHT)
    assert [r.content for r in fused] == [RELEVANT, CONVERSATIONAL, EDUCATION]
    assert [r.content for r in fused[1:]] == [CONVERSATIONAL, EDUCATION]
