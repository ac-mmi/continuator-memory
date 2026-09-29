import numpy as np
import pytest

from continuator_memory.chunking import chunk_conversation
from continuator_memory.models import parse_conversation
from continuator_memory.store import Store


def _conv(cid="c1", n=5, prefix="msg"):
    return parse_conversation(
        {
            "conversation_id": cid,
            "title": f"title {cid}",
            "messages": [
                {
                    "id": f"m{i}",
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"{prefix} {i}",
                    "timestamp": f"2026-01-01T00:00:0{i}Z",
                }
                for i in range(n)
            ],
        }
    )


def _save(store, conv, model="model-a", dim=8):
    chunks = chunk_conversation(conv, 3, 1)
    vectors = np.arange(len(chunks) * dim, dtype=np.float32).reshape(len(chunks), dim)
    store.save_conversation(conv, chunks, vectors, model)
    return chunks, vectors


def test_conversation_roundtrip(store):
    conv = _conv()
    _save(store, conv)
    assert store.get_conversation("c1") == conv
    assert store.list_conversations() == [("c1", "title c1")]
    assert store.get_conversation("missing") is None


def test_chunks_and_metadata_roundtrip(store):
    conv = _conv()
    chunks, _ = _save(store, conv)
    assert store.get_chunks(conversation_id="c1") == chunks
    assert store.get_chunks([chunks[1].chunk_id]) == [chunks[1]]
    assert store.get_chunks([]) == []


def test_embeddings_roundtrip_exactly(store):
    chunks, vectors = _save(store, _conv())
    ids, matrix = store.load_embeddings("model-a")
    assert ids == [c.chunk_id for c in chunks]
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix, vectors)


def test_embeddings_filtered_by_model_and_conversation(store):
    _save(store, _conv("c1"))
    _save(store, _conv("c2"), model="model-b")
    assert store.load_embeddings("model-a")[0] == ["c1:c0000", "c1:c0001"]
    assert store.load_embeddings("model-b", "c1")[0] == []
    assert store.load_embeddings("model-b", "c2")[0] == ["c2:c0000", "c2:c0001"]


def test_persists_across_connections(tmp_path):
    path = tmp_path / "sub" / "db.sqlite"
    conv = _conv()
    with Store(path) as s:
        chunks, vectors = _save(s, conv)
    with Store(path) as s:
        assert s.get_conversation("c1") == conv
        assert s.get_chunks(conversation_id="c1") == chunks
        np.testing.assert_array_equal(s.load_embeddings("model-a")[1], vectors)


def test_reingest_replaces_conversation(store):
    _save(store, _conv(n=7))
    new = _conv(n=3, prefix="new")
    chunks, _ = _save(store, new)
    assert store.get_conversation("c1") == new
    assert store.get_chunks(conversation_id="c1") == chunks
    assert len(store.load_embeddings("model-a")[0]) == len(chunks)


def test_same_message_ids_in_different_conversations(store):
    _save(store, _conv("c1", prefix="one"))
    _save(store, _conv("c2", prefix="two"))
    assert store.get_messages("c1", ["m0"])[0].content == "one 0"
    assert store.get_messages("c2", ["m0"])[0].content == "two 0"


def test_get_messages_preserves_requested_order(store):
    _save(store, _conv())
    assert [m.message_id for m in store.get_messages("c1", ["m3", "m0", "m2"])] == [
        "m3",
        "m0",
        "m2",
    ]


def test_embedding_count_mismatch_rejected(store):
    conv = _conv()
    chunks = chunk_conversation(conv, 3, 1)
    with pytest.raises(ValueError):
        store.save_conversation(conv, chunks, np.zeros((1, 4), np.float32), "m")
    assert store.get_conversation("c1") is None
