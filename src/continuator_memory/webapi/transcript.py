"""Parse a pasted ChatGPT transcript into the existing Conversation model."""

from __future__ import annotations

import json
import re
from pathlib import Path

from continuator_memory.models import Conversation, Message, ParseError, parse_conversation

IMPORTED_CONVERSATION_ID = "imported"
MAX_IMPORT_CHARS = 2_000_000
MAX_IMPORT_MESSAGES = 2_000

_ROLE = {
    "you": "user",
    "you said": "user",
    "user": "user",
    "human": "user",
    "chatgpt": "assistant",
    "chatgpt said": "assistant",
    "assistant": "assistant",
    "gpt": "assistant",
    "system": "system",
}

_HEADER = re.compile(
    r"^(?:\*\*|__)?(you said|chatgpt said|you|user|human|chatgpt|assistant|gpt|system)"
    r"(?:\*\*|__)?\s*:\s*(.*)$",
    re.IGNORECASE,
)
_MULTI_BLANK = re.compile(r"(?:\n[ \t\u00a0]*){2,}")
_TURN_BREAK = re.compile(r"(?:\n[ \t\u00a0]*){3,}")
_ASSISTANT_START = re.compile(
    r"^(got it,|what i |why might|the important|and remember|and if |"
    r"but [a-z]+,|i think |when you start|you'?re |this is the |what i want |"
    r"since the |try smaller|two things |one question|one thing i |optional |"
    r"take your time|the fact that|what might be happening|but i also|but i wouldn)",
    re.IGNORECASE,
)
_HEADING = re.compile(
    r"^(\d+\.\s|\d{1,2}:\d{2}\b|[A-D]\.\s|what i think|the important part)",
    re.IGNORECASE,
)


def parse_transcript(text: str, *, filename: str = "conversation.txt") -> Conversation:
    if not isinstance(text, str):
        raise ParseError("transcript must be a string")
    if len(text) > MAX_IMPORT_CHARS:
        raise ParseError(f"file exceeds {MAX_IMPORT_CHARS} characters")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        raise ParseError("file is empty")
    rows = _parse_body(normalized)
    if len(rows) > MAX_IMPORT_MESSAGES:
        raise ParseError(f"conversation has more than {MAX_IMPORT_MESSAGES} messages")
    title = _title_from(filename)
    messages = tuple(
        Message(
            message_id=f"{IMPORTED_CONVERSATION_ID}:m{i:04d}",
            conversation_id=IMPORTED_CONVERSATION_ID,
            position=i,
            role=role,
            content=content,
        )
        for i, (role, content) in enumerate(rows)
    )
    return Conversation(IMPORTED_CONVERSATION_ID, messages, title=title)


def _parse_body(text: str) -> list[tuple[str, str]]:
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            return _rows_from_conversation(_parse_json(stripped))
        except (ParseError, json.JSONDecodeError):
            pass
    labeled = _parse_labeled(text)
    if labeled is not None:
        return labeled
    return _parse_unlabeled(text)


def _parse_json(raw: str) -> Conversation:
    data = json.loads(raw)
    if isinstance(data, list):
        data = {"conversation_id": IMPORTED_CONVERSATION_ID, "messages": data}
    if not isinstance(data, dict):
        raise ParseError("JSON transcript must be an object or a list of messages")
    if "messages" in data and "conversation_id" not in data:
        data = {**data, "conversation_id": IMPORTED_CONVERSATION_ID}
    return parse_conversation(data)


def _rows_from_conversation(conversation: Conversation) -> list[tuple[str, str]]:
    rows = [(m.role, m.content.strip()) for m in conversation.messages if m.content.strip()]
    if not rows:
        raise ParseError("conversation has no messages")
    return rows


def _parse_labeled(text: str) -> list[tuple[str, str]] | None:
    hits = 0
    rows: list[tuple[str, str]] = []
    role: str | None = None
    buf: list[str] = []
    for line in text.split("\n"):
        match = _HEADER.match(line.strip())
        if match:
            hits += 1
            if role is not None:
                content = "\n".join(buf).strip()
                if content:
                    rows.append((role, content))
            role = _ROLE[match.group(1).lower()]
            buf = [match.group(2)] if match.group(2) else []
            continue
        if role is not None:
            buf.append(line)
    if role is not None:
        content = "\n".join(buf).strip()
        if content:
            rows.append((role, content))
    if hits < 2 or len(rows) < 2:
        return None
    return rows


def _parse_unlabeled(text: str) -> list[tuple[str, str]]:
    alternating = _alternating_turns(text)
    if alternating is not None:
        return alternating
    blocks = [part.strip(" \t\n\u00a0") for part in _MULTI_BLANK.split(text) if part.strip(" \t\n\u00a0")]
    if not blocks:
        raise ParseError("could not find any messages in the file")
    if len(blocks) == 1:
        return [("user", blocks[0])]
    user_at = {i for i, block in enumerate(blocks) if _looks_like_user(block, i == 0)}
    if 0 not in user_at:
        user_at.add(0)
    turns: list[tuple[str, str]] = []
    starts = sorted(user_at)
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(blocks)
        user = blocks[start].strip()
        assistant = "\n\n".join(blocks[start + 1 : end]).strip()
        if user:
            turns.append(("user", user))
        if assistant:
            turns.append(("assistant", assistant))
    if not turns:
        raise ParseError("could not parse user/assistant turns")
    return turns


def _alternating_turns(text: str) -> list[tuple[str, str]] | None:
    """Two or more blank lines separate turns when every block fits strict user/assistant alternation."""
    blocks = [part.strip(" \t\n\u00a0") for part in _TURN_BREAK.split(text) if part.strip(" \t\n\u00a0")]
    if len(blocks) < 2:
        return None
    if any(_looks_like_user(block, False) for block in blocks[1::2]):
        return None
    if any(_looks_like_assistant(block) for block in blocks[2::2]):
        return None
    return [("assistant" if i % 2 else "user", block) for i, block in enumerate(blocks)]


def _looks_like_assistant(block: str) -> bool:
    return bool(_ASSISTANT_START.match(block) or _HEADING.match(block.split("\n", 1)[0]))


def _looks_like_user(block: str, is_first: bool) -> bool:
    if is_first:
        return True
    text = block.strip()
    if not text or len(text) > 900:
        return False
    first = text.split("\n", 1)[0]
    if re.match(r"^[A-D]\.\s+\"", text) or text.count("\n") >= 3 and re.search(r"^[A-D]\.\s", text, re.M):
        return False
    if _looks_like_assistant(text):
        return False
    if re.match(r"^[A-Z][A-Za-z0-9 .·'’-]{0,70}$", first) and len(text) < 90:
        return False
    if text[0].islower():
        return True
    return bool(re.search(r"\b(i don't|i dont|you know|also i don't|also i dont)\b", text, re.I))


def _title_from(filename: str) -> str:
    name = Path(filename or "conversation.txt").name.strip() or "conversation.txt"
    return f"Imported · {name}"
