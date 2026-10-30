import json

import pytest

from continuator_memory.models import ParseError
from continuator_memory.webapi.transcript import parse_transcript


def test_labeled_you_chatgpt_preserves_order():
    text = "You: I ride a weekday bicycle.\n\nChatGPT: You mentioned a Fender Stratocaster.\n"
    conv = parse_transcript(text, filename="chat.txt")
    assert conv.title == "Imported · chat.txt"
    assert [m.role for m in conv.messages] == ["user", "assistant"]
    assert conv.messages[0].content == "I ride a weekday bicycle."
    assert "Fender" in conv.messages[1].content


def test_json_conversation_schema():
    text = """
    {"conversation_id": "x", "messages": [
      {"role": "user", "content": "hello"},
      {"role": "assistant", "content": "hi"}
    ]}
    """
    conv = parse_transcript(text, filename="x.json")
    assert [m.role for m in conv.messages] == ["user", "assistant"]
    assert conv.messages[0].content == "hello"


def test_unlabeled_keeps_assistant_internal_blank_lines():
    text = (
        "what should my ferry trip look like\n"
        "\n"
        "\n"
        "Here is a schedule.\n"
        "\n"
        "6:00 AM\n"
        "Pack a sandwich\n"
        "\n"
        "Bring a raincoat.\n"
        "\n"
        "\n"
        "no my ferry is 7 40 to 6 10\n"
        "\n"
        "\n"
        "Got it, here is a revised plan.\n"
    )
    conv = parse_transcript(text, filename="conv.txt")
    assert [m.role for m in conv.messages] == ["user", "assistant", "user", "assistant"]
    assert "Bring a raincoat" in conv.messages[1].content
    assert conv.messages[2].content.startswith("no my ferry")


def test_empty_file_errors():
    try:
        parse_transcript("   \n")
    except ParseError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("expected ParseError")


def _roles(text):
    return [(m.role, m.content) for m in parse_transcript(text).messages]


def test_uppercase_user_paragraphs_are_kept_when_two_blank_lines_separate_turns():
    text = (
        "Should I move to Lisbon?\n\n\n"
        "Lisbon suits many people.\n\nHere are some things to consider.\n\n\n"
        "I work at a bakery there already.\n\n\n"
        "That makes the move easier."
    )
    assert _roles(text) == [
        ("user", "Should I move to Lisbon?"),
        ("assistant", "Lisbon suits many people.\n\nHere are some things to consider."),
        ("user", "I work at a bakery there already."),
        ("assistant", "That makes the move easier."),
    ]


def test_lowercase_user_paragraphs_with_single_blank_lines():
    text = "where should i go\n\nTry the coast.\n\nno i meant in winter\n\nThen try the south."
    assert _roles(text) == [
        ("user", "where should i go"),
        ("assistant", "Try the coast."),
        ("user", "no i meant in winter"),
        ("assistant", "Then try the south."),
    ]


def test_ordinary_paragraphs_do_not_become_user_messages():
    text = (
        "plan my week\n\n"
        "Here is a plan for your week.\n\n"
        "Morning routine\n\n"
        "1. Rinse the pan\n\n"
        "Rest in the afternoon, then review your notes before dinner."
    )
    roles = _roles(text)
    assert [role for role, _ in roles] == ["user", "assistant"]
    assert "Morning routine" in roles[1][1] and "review your notes" in roles[1][1]


def test_headings_stop_alternation_so_they_stay_in_the_assistant_turn():
    text = "plan my week\n\n\nHere is a plan.\n\n\n1. Rinse the pan\n\n\nRest afterwards."
    assert [role for role, _ in _roles(text)] == ["user", "assistant"]


def test_lowercase_block_in_an_assistant_slot_stops_alternation():
    text = "Plan my week\n\n\nHere is a plan.\n\n\nMore of the plan.\n\n\nno thanks\n\n\nAll right."
    assert _roles(text) == [
        ("user", "Plan my week"),
        ("assistant", "Here is a plan.\n\nMore of the plan."),
        ("user", "no thanks"),
        ("assistant", "All right."),
    ]


def test_uppercase_user_paragraphs_with_single_blank_lines_stay_in_the_assistant_turn():
    # Known limitation: nothing structural separates the turns, so headers are needed for correct roles.
    text = "Should I move to Lisbon?\n\nLisbon suits many people.\n\nI work at a bakery there already."
    assert [role for role, _ in _roles(text)] == ["user", "assistant"]


@pytest.mark.parametrize("user, assistant", [
    ("You", "ChatGPT"),
    ("You said", "ChatGPT said"),
    ("User", "Assistant"),
    ("Human", "GPT"),
    ("**You**", "**ChatGPT**"),
])
def test_explicit_headers_set_roles_for_any_capitalisation_of_the_body(user, assistant):
    text = f"{user}: I work at a bakery.\n\n{assistant}: Noted.\n\nMore detail here.\n\n{user}: thanks"
    assert _roles(text) == [
        ("user", "I work at a bakery."),
        ("assistant", "Noted.\n\nMore detail here."),
        ("user", "thanks"),
    ]


def test_json_message_list_and_single_block_formats():
    rows = [{"message_id": "a", "role": "user", "content": "I work at a bakery."},
            {"message_id": "b", "role": "assistant", "content": "Noted."}]
    assert _roles(json.dumps(rows)) == [("user", "I work at a bakery."), ("assistant", "Noted.")]
    assert _roles("I work at a bakery.") == [("user", "I work at a bakery.")]
