"""Conversation-only LoCoMo adapter for manual UI testing.

Reads the official LoCoMo JSON and builds a Continuator ``Conversation``.
QA keys, summaries, and observations are never copied into messages or memories.
``dia_id`` is kept as ``message_id``; session dates become ``Message.timestamp``.
Speaker names are prefixed in content because Continuator roles are only
user/assistant/system.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from continuator_memory.models import Conversation, Message, ParseError
from continuator_memory.store import Store
from continuator_memory.webapi.extract import ingest_imported
from continuator_memory.webapi.transcript import IMPORTED_CONVERSATION_ID

_SESSION = re.compile(r"^session_(\d+)$")
_DATE_FORMATS = ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y")
DEFAULT_DB = Path("data/locomo-conv26.db")


def parse_locomo_datetime(raw: str) -> str:
    """Convert LoCoMo session clocks to ISO-8601; keep the original if unparsable."""
    text = raw.strip()
    if not text:
        return text
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            continue
    return text


def conversation_from_locomo(data: Any, *, conversation_id: str = IMPORTED_CONVERSATION_ID) -> Conversation:
    record = _record(data)
    raw = record.get("conversation")
    if not isinstance(raw, dict):
        raise ParseError("LoCoMo record is missing a conversation object")
    speaker_a = raw.get("speaker_a") if isinstance(raw.get("speaker_a"), str) else "speaker_a"
    speaker_b = raw.get("speaker_b") if isinstance(raw.get("speaker_b"), str) else "speaker_b"
    sample = record.get("sample_id") if isinstance(record.get("sample_id"), str) else "locomo"
    messages: list[Message] = []
    seen: set[str] = set()
    for session_index in _session_indices(raw):
        turns = raw.get(f"session_{session_index}")
        if not isinstance(turns, list) or not turns:
            continue
        date_raw = raw.get(f"session_{session_index}_date_time")
        timestamp = parse_locomo_datetime(date_raw) if isinstance(date_raw, str) else None
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            message = _message(
                turn,
                conversation_id=conversation_id,
                position=len(messages),
                timestamp=timestamp,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
            )
            if message is None:
                continue
            if message.message_id in seen:
                raise ParseError(f"duplicate dia_id {message.message_id!r}")
            seen.add(message.message_id)
            messages.append(message)
    if not messages:
        raise ParseError("LoCoMo conversation has no dialogue turns")
    title = f"Imported · {sample} · {speaker_a} and {speaker_b}"
    return Conversation(conversation_id, tuple(messages), title=title)


def load_locomo_conversation(path: str | Path, *, conversation_id: str = IMPORTED_CONVERSATION_ID) -> Conversation:
    payload = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ParseError(f"{path}: invalid JSON: {exc}") from exc
    return conversation_from_locomo(data, conversation_id=conversation_id)


def ingest_locomo(
    path: str | Path,
    db_path: str | Path = DEFAULT_DB,
    *,
    conversation_id: str = IMPORTED_CONVERSATION_ID,
) -> dict[str, Any]:
    """Write conversation-only memories into ``db_path``. Does not touch other SQLite files."""
    destination = Path(db_path)
    if destination.resolve() in {
        Path("data/continuator.db").resolve(),
        Path("data/continuator-web-import.db").resolve(),
    }:
        raise ValueError("refusing to ingest LoCoMo into the default Continuator databases")
    conversation = load_locomo_conversation(path, conversation_id=conversation_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    store = Store(str(destination))
    try:
        report = ingest_imported(conversation, store, extract="messages")
    except Exception:
        store.close()
        raise
    store.close()
    return {
        "ok": True,
        "db": str(destination),
        "conversationId": conversation.conversation_id,
        "title": conversation.title,
        "nMessages": len(conversation.messages),
        "nClaims": report.claims,
        "nMemories": report.claims,
        "extract": "messages",
    }


def _record(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        if not data or not isinstance(data[0], dict):
            raise ParseError("LoCoMo JSON must be an object or a non-empty list of objects")
        return data[0]
    if isinstance(data, dict):
        if "conversation" in data:
            return data
        raise ParseError("LoCoMo JSON object is missing 'conversation'")
    raise ParseError("LoCoMo JSON must be an object or a list")


def _session_indices(conversation: dict[str, Any]) -> list[int]:
    found = []
    for key, value in conversation.items():
        match = _SESSION.fullmatch(key)
        if match and isinstance(value, list):
            found.append(int(match.group(1)))
    return sorted(found)


def _message(
    turn: dict[str, Any],
    *,
    conversation_id: str,
    position: int,
    timestamp: str | None,
    speaker_a: str,
    speaker_b: str,
) -> Message | None:
    dia_id = turn.get("dia_id")
    if not isinstance(dia_id, str) or not dia_id.strip():
        return None
    speaker = turn.get("speaker") if isinstance(turn.get("speaker"), str) else "unknown"
    text = turn.get("text") if isinstance(turn.get("text"), str) else ""
    caption = turn.get("blip_caption") if isinstance(turn.get("blip_caption"), str) else ""
    body = text.strip()
    if caption.strip():
        body = f"{body}\n[image: {caption.strip()}]".strip()
    if not body:
        return None
    if speaker == speaker_a:
        role = "user"
    elif speaker == speaker_b:
        role = "assistant"
    else:
        role = "user"
    return Message(
        message_id=dia_id.strip(),
        conversation_id=conversation_id,
        position=position,
        role=role,
        content=f"{speaker}: {body}",
        timestamp=timestamp,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m continuator_memory.webapi.locomo")
    parser.add_argument("path", help="LoCoMo JSON (conversation + evaluation-only fields)")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help=f"dedicated SQLite path (default: {DEFAULT_DB}; never the default Continuator DBs)",
    )
    args = parser.parse_args(argv)
    result = ingest_locomo(args.path, args.db)
    print(
        f"locomo {result['conversationId']} → {result['db']} "
        f"({result['nMessages']} messages, {result['nMemories']} memories)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
