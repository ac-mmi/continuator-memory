import numpy as np
import pytest

from continuator_memory.models import load_conversation, parse_conversation
from continuator_memory.pipeline import Pipeline
from continuator_memory.retrieval import cosine_similarity, top_k
from tests.helpers import EXAMPLE_CONVERSATION, HashingEmbedder


def test_cosine_similarity_known_values():
    matrix = np.array([[1, 0], [0, 1], [-1, 0], [1, 1], [3, 0]], dtype=np.float32)
    scores = cosine_similarity(np.array([1, 0], dtype=np.float32), matrix)
    np.testing.assert_allclose(scores, [1.0, 0.0, -1.0, 1 / np.sqrt(2), 1.0], atol=1e-12)


def test_cosine_similarity_zero_vectors():
    matrix = np.array([[0, 0], [1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(cosine_similarity(np.array([1.0, 0.0]), matrix), [0.0, 1.0])
    np.testing.assert_array_equal(cosine_similarity(np.zeros(2), matrix), [0.0, 0.0])


def test_cosine_similarity_shape_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3), np.ones((2, 2)))


def test_top_k_orders_by_score_desc():
    assert top_k(np.array([0.1, 0.9, 0.5, 0.7]), ["a", "b", "c", "d"], 3) == [1, 3, 2]


def test_top_k_breaks_ties_by_id():
    assert top_k(np.array([0.5, 0.5, 0.9, 0.5]), ["z", "b", "x", "a"], 4) == [2, 3, 1, 0]


def test_top_k_larger_than_n():
    assert top_k(np.array([0.2, 0.1]), ["a", "b"], 10) == [0, 1]


def test_top_k_invalid():
    with pytest.raises(ValueError):
        top_k(np.array([0.1]), ["a"], 0)
    with pytest.raises(ValueError):
        top_k(np.array([0.1, 0.2]), ["a"], 1)


def test_retriever_returns_best_matching_chunk(store, embedder):
    pipeline = Pipeline(store, embedder, window_size=4, overlap=2)
    pipeline.ingest(load_conversation(EXAMPLE_CONVERSATION))
    results = pipeline.retrieve("shellfish allergic Priya", k=3)
    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score >= results[1].score >= results[2].score
    assert "m007" in results[0].source_message_ids
    assert "shellfish" in results[0].text


def test_retriever_respects_k_and_conversation_filter(store, embedder):
    pipeline = Pipeline(store, embedder, window_size=2, overlap=0)
    for cid, word in (("a", "apples"), ("b", "bananas")):
        pipeline.ingest(
            parse_conversation(
                {
                    "conversation_id": cid,
                    "messages": [
                        {"role": "user", "content": f"I like {word}"},
                        {"role": "assistant", "content": f"{word} are great"},
                        {"role": "user", "content": "thanks"},
                    ],
                }
            )
        )
    assert len(pipeline.retrieve("apples", k=1)) == 1
    everything = pipeline.retrieve("apples", k=100)
    assert len(everything) == 4
    assert everything[0].conversation_id == "a"
    only_b = pipeline.retrieve("apples", k=100, conversation_id="b")
    assert {r.conversation_id for r in only_b} == {"b"}


def test_retriever_only_uses_matching_embedding_model(store):
    conv = load_conversation(EXAMPLE_CONVERSATION)
    Pipeline(store, HashingEmbedder(model_name="model-a")).ingest(conv)
    assert Pipeline(store, HashingEmbedder(model_name="model-b")).retrieve("Sintra", k=5) == []


def test_retrieval_is_deterministic(store, embedder):
    pipeline = Pipeline(store, embedder)
    pipeline.ingest(load_conversation(EXAMPLE_CONVERSATION))
    assert pipeline.retrieve("When is fado?", k=5) == pipeline.retrieve("When is fado?", k=5)
