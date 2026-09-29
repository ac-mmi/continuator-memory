import json

import pytest

from continuator_memory.models import ParseError, load_conversation, parse_conversation
from tests.helpers import EXAMPLE_CONVERSATION


def _conv(**overrides):
    data = {
        "conversation_id": "c1",
        "title": "Test",
        "messages": [
            {"id": "a", "role": "user", "content": "hello", "timestamp": "2026-01-01T00:00:00Z"},
            {"id": "b", "role": "assistant", "content": "hi there"},
        ],
    }
    data.update(overrides)
    return data


def test_parse_valid_conversation():
    conv = parse_conversation(_conv())
    assert conv.conversation_id == "c1"
    assert conv.title == "Test"
    assert [m.message_id for m in conv.messages] == ["a", "b"]
    assert [m.position for m in conv.messages] == [0, 1]
    assert all(m.conversation_id == "c1" for m in conv.messages)
    assert conv.messages[0].timestamp == "2026-01-01T00:00:00Z"
    assert conv.messages[1].timestamp is None


def test_missing_message_ids_are_derived_from_position():
    conv = parse_conversation(
        _conv(messages=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}])
    )
    assert [m.message_id for m in conv.messages] == ["c1:m0000", "c1:m0001"]


@pytest.mark.parametrize(
    "data, match",
    [
        ([], "JSON object"),
        ({"messages": [{"role": "user", "content": "x"}]}, "conversation_id"),
        (_conv(messages=[]), "non-empty list"),
        (_conv(messages=[{"role": "robot", "content": "x"}]), "role"),
        (_conv(messages=[{"role": "user", "content": "   "}]), "content"),
        (_conv(messages=[{"role": "user"}]), "content"),
        (_conv(messages=["not an object"]), "JSON object"),
        (
            _conv(
                messages=[
                    {"id": "a", "role": "user", "content": "x"},
                    {"id": "a", "role": "assistant", "content": "y"},
                ]
            ),
            "duplicate",
        ),
    ],
)
def test_parse_rejects_invalid_input(data, match):
    with pytest.raises(ParseError, match=match):
        parse_conversation(data)


def test_load_conversation_from_file(tmp_path):
    path = tmp_path / "conv.json"
    path.write_text(json.dumps(_conv()), encoding="utf-8")
    assert load_conversation(path).conversation_id == "c1"


def test_load_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ParseError, match="invalid JSON"):
        load_conversation(path)


def test_example_dataset_parses():
    conv = load_conversation(EXAMPLE_CONVERSATION)
    assert conv.conversation_id == "trip-planning-001"
    assert len(conv.messages) == 18
