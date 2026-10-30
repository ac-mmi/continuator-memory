import pytest

from continuator_memory.models import parse_conversation
from continuator_memory.v2.windowing import build_windows


def conversation(n: int):
    return parse_conversation({"conversation_id": "c", "messages": [
        {"id": f"m{i:02d}", "role": "user" if i % 2 else "assistant", "content": f"message {i}"}
        for i in range(1, n + 1)]})


def test_every_message_is_a_target_exactly_once_in_order():
    windows = build_windows(conversation(10), targets_per_window=4, context_before=3)
    targets = [mid for w in windows for mid in w.target_ids]
    assert targets == [f"m{i:02d}" for i in range(1, 11)]
    assert [w.index for w in windows] == [0, 1, 2]


def test_context_is_bounded_and_precedes_the_targets():
    windows = build_windows(conversation(10), targets_per_window=4, context_before=3)
    assert windows[0].context_ids == ()
    assert windows[1].context_ids == ("m02", "m03", "m04")
    assert windows[2].target_ids == ("m09", "m10")
    assert windows[2].context_ids == ("m06", "m07", "m08")
    assert all(len(w.messages) <= 7 for w in windows)


def test_windows_are_deterministic():
    conv = conversation(9)
    assert build_windows(conv, 3, 2) == build_windows(conv, 3, 2)


def test_preceding_includes_context_and_earlier_targets_only():
    w = build_windows(conversation(8), targets_per_window=4, context_before=2)[1]
    target = w.targets[1]
    assert [m.message_id for m in w.preceding(target)] == ["m03", "m04", "m05"]


@pytest.mark.parametrize("targets,context", [(0, 2), (2, -1)])
def test_invalid_window_parameters(targets, context):
    with pytest.raises(ValueError):
        build_windows(conversation(3), targets, context)
