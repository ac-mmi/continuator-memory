"""Small in-memory conversation + claims + one eligible LINK. No extraction, no Ollama."""

from __future__ import annotations

import json
from pathlib import Path

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.memory import memory_id
from continuator_memory.v2.models import Claim, Provenance

SAMPLE_CONVERSATION_ID = "web-sample"
SAMPLE_SOURCE_TITLE = "Sample conversation"

# Distinct tokens so HashingEmbedder / MiniLM can separate commute (seed) from guitar (gold).
_ITEMS: tuple[tuple[str, str, str], ...] = (
    ("m-commute", "seed", "Alex commutes on a weekday bicycle."),
    ("m-remote", "remote", "Alex works remotely on Tuesday and Thursday."),
    ("m-coffee", "coffee", "Alex drinks coffee with oat milk."),
    ("m-guitar", "gold", "Alex owns a Fender Stratocaster electric guitar."),
    ("m-uke", "uke", "Alex browsed a Cordoba ukulele listing and did not buy it."),
    ("m-cat", "cat", "Alex adopted a tabby cat named Miso."),
)


def sample_conversation() -> Conversation:
    messages = tuple(
        Message(message_id, SAMPLE_CONVERSATION_ID, i, "user", text, f"2024-03-{(i + 1):02d}")
        for i, (message_id, _claim_id, text) in enumerate(_ITEMS)
    )
    return Conversation(SAMPLE_CONVERSATION_ID, messages, title=SAMPLE_SOURCE_TITLE)


def sample_claims(conversation: Conversation) -> tuple[Claim, ...]:
    by_id = {m.message_id: m for m in conversation.messages}
    return tuple(
        Claim(claim_id, text, Provenance.of(by_id[message_id]))
        for message_id, claim_id, text in _ITEMS
    )


def write_sample_classifications(path: Path) -> Path:
    """One eligible RELATED link: commute seed → guitar gold."""
    row = {
        "pair_id": "P01",
        "decision": "LINK",
        "parse_ok": True,
        "postcheck_ok": True,
        "source_memory_id": memory_id("seed"),
        "candidate_memory_id": memory_id("gold"),
        "relation": "RELATED",
        "direction": "undirected",
        "rationale": "The commute memory is a retrieved seed; the guitar memory states an owned instrument.",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path
