"""Previous-turn context texts for experimarine contextual retrieval."""

from __future__ import annotations

import pytest

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.memory import STATUS_CURRENT, Memory
from continuator_memory.v2.turn_context import (
    CONTEXT_WEIGHT_ENV,
    DEFAULT_CONTEXT_WEIGHT,
    load_context_weight,
    previous_turn_contexts,
)

CONV = "imported"
SESSION_1 = "2023-05-08T13:56:00"
SESSION_2 = "2023-05-25T13:14:00"


def _message(position: int, message_id: str, content: str, timestamp: str | None) -> Message:
    return Message(message_id, CONV, position, "user", content, timestamp)


def _memory(message: Message, conversation_id: str = CONV) -> Memory:
    return Memory(f"mem:{message.message_id}:c0", conversation_id, message.content, STATUS_CURRENT,
                  f"{message.message_id}:c0", message.message_id, message.position)


MESSAGES = (
    _message(0, "D1:1", "Priya: Hey Lena!", SESSION_1),
    _message(1, "D1:2", "Lena: What pet do you have?", SESSION_1),
    _message(2, "D1:3", "Priya: Biscuit, my tortoise.", SESSION_1),
    _message(3, "D2:1", "Priya: Hi again!", SESSION_2),
)
CONVERSATION = Conversation(CONV, MESSAGES)


def test_previous_turn_in_same_session_is_prepended():
    contexts = previous_turn_contexts(CONVERSATION, [_memory(MESSAGES[2])])
    assert contexts == {
        (CONV, "mem:D1:3:c0"): "Lena: What pet do you have?\nPriya: Biscuit, my tortoise.",
    }


def test_first_turn_of_conversation_and_of_session_have_no_context():
    contexts = previous_turn_contexts(CONVERSATION, [_memory(m) for m in MESSAGES])
    assert set(contexts) == {(CONV, "mem:D1:2:c0"), (CONV, "mem:D1:3:c0")}


def test_missing_timestamps_or_message_give_no_context():
    untimed = Conversation(CONV, (_message(0, "a", "A: one", None), _message(1, "b", "B: two", None)))
    assert previous_turn_contexts(untimed, [_memory(untimed.messages[1])]) == {}
    orphan = Memory("mem:zz:c0", CONV, "Z: gone", STATUS_CURRENT, "zz:c0", "zz", 9)
    assert previous_turn_contexts(CONVERSATION, [orphan]) == {}


def test_memory_of_another_conversation_is_ignored():
    assert previous_turn_contexts(CONVERSATION, [_memory(MESSAGES[2], conversation_id="other")]) == {}


def test_order_follows_message_position_not_tuple_order():
    shuffled = Conversation(CONV, (MESSAGES[2], MESSAGES[0], MESSAGES[1]))
    contexts = previous_turn_contexts(shuffled, [_memory(MESSAGES[2])])
    assert contexts[(CONV, "mem:D1:3:c0")].startswith("Lena: What pet do you have?\n")


def test_load_context_weight_defaults_off_and_validates():
    assert DEFAULT_CONTEXT_WEIGHT == 0.0
    assert load_context_weight({}) == 0.0
    assert load_context_weight({CONTEXT_WEIGHT_ENV: ""}) == 0.0
    assert load_context_weight({CONTEXT_WEIGHT_ENV: "0.5"}) == 0.5
    with pytest.raises(ValueError, match="number"):
        load_context_weight({CONTEXT_WEIGHT_ENV: "half"})
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_context_weight({CONTEXT_WEIGHT_ENV: "1.5"})


def test_load_context_weight_reads_process_env(monkeypatch):
    monkeypatch.delenv(CONTEXT_WEIGHT_ENV, raising=False)
    assert load_context_weight() == DEFAULT_CONTEXT_WEIGHT
    monkeypatch.setenv(CONTEXT_WEIGHT_ENV, "0.5")
    assert load_context_weight() == 0.5
