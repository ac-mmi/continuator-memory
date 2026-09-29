import pytest

from continuator_memory.chunking import chunk_conversation
from continuator_memory.models import parse_conversation


def _conv(n: int):
    return parse_conversation(
        {
            "conversation_id": "c",
            "messages": [
                {"id": f"m{i}", "role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
                for i in range(n)
            ],
        }
    )


def _ids(chunks):
    return [list(c.source_message_ids) for c in chunks]


def test_sliding_window_with_overlap():
    chunks = chunk_conversation(_conv(7), window_size=4, overlap=2)
    assert _ids(chunks) == [
        ["m0", "m1", "m2", "m3"],
        ["m2", "m3", "m4", "m5"],
        ["m4", "m5", "m6"],
    ]
    assert [c.chunk_id for c in chunks] == ["c:c0000", "c:c0001", "c:c0002"]
    assert [c.position for c in chunks] == [0, 1, 2]


def test_exact_fit_does_not_emit_trailing_subset():
    assert _ids(chunk_conversation(_conv(6), window_size=4, overlap=2)) == [
        ["m0", "m1", "m2", "m3"],
        ["m2", "m3", "m4", "m5"],
    ]


def test_no_overlap():
    assert _ids(chunk_conversation(_conv(5), window_size=2, overlap=0)) == [
        ["m0", "m1"],
        ["m2", "m3"],
        ["m4"],
    ]


def test_conversation_shorter_than_window():
    chunks = chunk_conversation(_conv(2), window_size=4, overlap=2)
    assert _ids(chunks) == [["m0", "m1"]]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
@pytest.mark.parametrize("window,overlap", [(1, 0), (3, 1), (4, 2), (4, 3)])
def test_every_message_is_covered_in_order(n, window, overlap):
    chunks = chunk_conversation(_conv(n), window, overlap)
    covered = [mid for c in chunks for mid in c.source_message_ids]
    assert set(covered) == {f"m{i}" for i in range(n)}
    assert all(len(c.source_message_ids) <= window for c in chunks)


def test_chunk_text_and_metadata():
    chunk = chunk_conversation(_conv(3), window_size=2, overlap=1)[1]
    assert chunk.text == "assistant: msg 1\nuser: msg 2"
    assert chunk.metadata == {
        "chunker": "message_window_v1",
        "window_size": 2,
        "overlap": 1,
        "start_position": 1,
        "end_position": 2,
    }


def test_chunking_is_deterministic():
    assert chunk_conversation(_conv(9), 4, 2) == chunk_conversation(_conv(9), 4, 2)


@pytest.mark.parametrize("window,overlap", [(0, 0), (3, 3), (3, -1)])
def test_invalid_parameters(window, overlap):
    with pytest.raises(ValueError):
        chunk_conversation(_conv(3), window, overlap)
