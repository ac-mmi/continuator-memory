"""Data models and JSON parsing for conversations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_ROLES = frozenset({"system", "user", "assistant"})


class ParseError(ValueError):
    """Raised when conversation input does not match the expected schema."""


@dataclass(frozen=True)
class Message:
    message_id: str
    conversation_id: str
    position: int
    role: str
    content: str
    timestamp: str | None = None


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    messages: tuple[Message, ...]
    title: str | None = None


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    conversation_id: str
    position: int
    text: str
    source_message_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    chunk_id: str
    conversation_id: str
    source_message_ids: tuple[str, ...]
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "conversation_id": self.conversation_id,
            "source_message_ids": list(self.source_message_ids),
            "score": self.score,
            "text": self.text,
        }


def _require_str(obj: dict[str, Any], key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ParseError(f"{where}: '{key}' must be a non-empty string")
    return value


def parse_conversation(data: Any) -> Conversation:
    """Parse a conversation dict.

    Expected shape::

        {"conversation_id": "...", "title": "...",  # title optional
         "messages": [{"id": "...", "role": "user", "content": "...",
                       "timestamp": "..."}]}       # id, timestamp optional

    Messages without an ``id`` get ``{conversation_id}:m{position:04d}``.
    """
    if not isinstance(data, dict):
        raise ParseError("conversation must be a JSON object")
    conversation_id = _require_str(data, "conversation_id", "conversation")
    title = data.get("title")
    if title is not None and not isinstance(title, str):
        raise ParseError("conversation: 'title' must be a string")

    raw_messages = data.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ParseError("conversation: 'messages' must be a non-empty list")

    messages: list[Message] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_messages):
        where = f"message[{position}]"
        if not isinstance(raw, dict):
            raise ParseError(f"{where}: must be a JSON object")
        message_id = raw.get("id")
        if message_id is None:
            message_id = f"{conversation_id}:m{position:04d}"
        elif not isinstance(message_id, str) or not message_id.strip():
            raise ParseError(f"{where}: 'id' must be a non-empty string")
        if message_id in seen_ids:
            raise ParseError(f"{where}: duplicate message id '{message_id}'")
        seen_ids.add(message_id)

        role = _require_str(raw, "role", where)
        if role not in VALID_ROLES:
            raise ParseError(f"{where}: role '{role}' not in {sorted(VALID_ROLES)}")
        content = _require_str(raw, "content", where)
        timestamp = raw.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, str):
            raise ParseError(f"{where}: 'timestamp' must be a string")

        messages.append(
            Message(
                message_id=message_id,
                conversation_id=conversation_id,
                position=position,
                role=role,
                content=content,
                timestamp=timestamp,
            )
        )

    return Conversation(conversation_id=conversation_id, messages=tuple(messages), title=title)


def load_conversation(path: str | Path) -> Conversation:
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ParseError(f"{path}: invalid JSON: {exc}") from exc
    return parse_conversation(data)
