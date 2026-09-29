import json

from continuator_memory.chunking import format_message
from continuator_memory.models import load_conversation
from continuator_memory.pipeline import Pipeline
from tests.helpers import EXAMPLE_CONVERSATION, EXAMPLE_QUESTIONS


def _ingested(store, embedder, window=4, overlap=2):
    pipeline = Pipeline(store, embedder, window_size=window, overlap=overlap)
    pipeline.ingest(load_conversation(EXAMPLE_CONVERSATION))
    return pipeline


def test_retrieved_chunks_trace_back_to_original_messages(store, embedder):
    pipeline = _ingested(store, embedder)
    original = {m.message_id: m for m in load_conversation(EXAMPLE_CONVERSATION).messages}
    for result in pipeline.retrieve("Sintra train Rossio", k=100):
        messages = store.get_messages(result.conversation_id, result.source_message_ids)
        assert [m.message_id for m in messages] == list(result.source_message_ids)
        assert all(m == original[m.message_id] for m in messages)
        assert result.text == "\n".join(format_message(m) for m in messages)
        positions = [m.position for m in messages]
        assert positions == list(range(positions[0], positions[0] + len(positions)))


def test_all_messages_reachable_from_chunks(store, embedder):
    pipeline = _ingested(store, embedder, window=3, overlap=1)
    retrieved = pipeline.retrieve("anything", k=1000)
    covered = {mid for r in retrieved for mid in r.source_message_ids}
    assert covered == {m.message_id for m in load_conversation(EXAMPLE_CONVERSATION).messages}


def test_retrieved_chunk_fields(store, embedder):
    result = _ingested(store, embedder).retrieve("return flight time", k=1)[0]
    d = result.to_dict()
    assert set(d) == {"rank", "chunk_id", "conversation_id", "source_message_ids", "score", "text"}
    assert d["conversation_id"] == "trip-planning-001"
    assert d["chunk_id"].startswith("trip-planning-001:c")
    assert isinstance(d["score"], float)
    json.dumps(d)


def test_question_file_references_existing_messages():
    ids = {m.message_id for m in load_conversation(EXAMPLE_CONVERSATION).messages}
    questions = json.loads(EXAMPLE_QUESTIONS.read_text(encoding="utf-8"))["questions"]
    for q in questions:
        assert set(q["expected_message_ids"]) <= ids
