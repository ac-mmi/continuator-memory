"""Retrieval with the real local Sentence Transformers model.

Skipped when the model cannot be loaded (e.g. not downloaded and offline).
"""

import json

import pytest

from continuator_memory.config import DEFAULT_EMBEDDING_MODEL
from continuator_memory.models import load_conversation
from continuator_memory.pipeline import Pipeline
from tests.helpers import EXAMPLE_CONVERSATION, EXAMPLE_QUESTIONS

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_embedder():
    try:
        from continuator_memory.embedding import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(DEFAULT_EMBEDDING_MODEL)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"embedding model unavailable: {exc}")


def test_embeddings_are_normalized_and_deterministic(real_embedder):
    a = real_embedder.embed(["hello world", "lisbon trip"])
    b = real_embedder.embed(["hello world", "lisbon trip"])
    assert a.shape == (2, 384)
    assert (abs((a**2).sum(axis=1) - 1) < 1e-5).all()
    assert (a == b).all()


def test_expected_messages_in_top_k(store, real_embedder):
    pipeline = Pipeline(store, real_embedder)
    pipeline.ingest(load_conversation(EXAMPLE_CONVERSATION))
    questions = json.loads(EXAMPLE_QUESTIONS.read_text(encoding="utf-8"))["questions"]
    for q in questions:
        retrieved = pipeline.retrieve(q["question"], k=3)
        found = {mid for r in retrieved for mid in r.source_message_ids}
        assert set(q["expected_message_ids"]) & found, q["id"]
