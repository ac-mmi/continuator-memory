"""Ephemeral context-enriched retrieval text for experimental indexing.

Used by the ``contextual_lexical`` retrieval mode. Builds a search representation that
includes same-session neighbour turns around a **target** memory, without mutating stored
``Memory.content``, SQLite rows, or ``MemorySource`` links.

Session membership matches ``turn_context.py``: equal, non-empty ``Message.timestamp`` values
(the LoCoMo adapter stamps every turn of a session with the session date).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from continuator_memory.models import Conversation, Message
from continuator_memory.v2.memory import Memory

DEFAULT_PREVIOUS_MESSAGES = 1
DEFAULT_NEXT_MESSAGES = 1


def same_session(left: Message, right: Message) -> bool:
    """True when both messages share a non-empty timestamp (LoCoMo session stamp)."""
    return bool(left.timestamp) and left.timestamp == right.timestamp


def _ordered_messages(conversation: Conversation) -> list[Message]:
    return sorted(conversation.messages, key=lambda m: m.position)


def neighbour_messages(
    conversation: Conversation,
    target_message_id: str,
    *,
    previous_messages: int = DEFAULT_PREVIOUS_MESSAGES,
    next_messages: int = DEFAULT_NEXT_MESSAGES,
    same_session_only: bool = True,
) -> tuple[list[Message], Message | None, list[Message]]:
    """Return ``(prev_msgs, target_or_none, next_msgs)`` around ``target_message_id``."""
    if previous_messages < 0 or next_messages < 0:
        raise ValueError("previous_messages and next_messages must be >= 0")
    ordered = _ordered_messages(conversation)
    index = {m.message_id: i for i, m in enumerate(ordered)}
    at = index.get(target_message_id)
    if at is None:
        return [], None, []
    target = ordered[at]
    prev: list[Message] = []
    for offset in range(1, previous_messages + 1):
        i = at - offset
        if i < 0:
            break
        candidate = ordered[i]
        if same_session_only and not same_session(target, candidate):
            break
        prev.append(candidate)
    prev.reverse()
    nxt: list[Message] = []
    for offset in range(1, next_messages + 1):
        i = at + offset
        if i >= len(ordered):
            break
        candidate = ordered[i]
        if same_session_only and not same_session(target, candidate):
            break
        nxt.append(candidate)
    return prev, target, nxt


def build_contextual_retrieval_text(
    conversation: Conversation,
    memory: Memory,
    *,
    previous_messages: int = DEFAULT_PREVIOUS_MESSAGES,
    next_messages: int = DEFAULT_NEXT_MESSAGES,
    same_session_only: bool = True,
) -> str:
    """Build ephemeral retrieval text for ``memory``; always includes a ``[TARGET]`` block.

    Neighbour blocks use conversation message content. The target block uses ``memory.content``
    so claim/memory wording stays aligned with the indexed unit even if it ever diverged from
    the raw message (passthrough LoCoMo keeps them equal).
    """
    prev, target_msg, nxt = neighbour_messages(
        conversation,
        memory.message_id,
        previous_messages=previous_messages,
        next_messages=next_messages,
        same_session_only=same_session_only,
    )
    parts: list[str] = []
    for message in prev:
        parts.append(f"[PREV] {message.content}")
    # Prefer memory.content; fall back to conversation message if somehow empty.
    target_body = memory.content.strip() or (target_msg.content if target_msg is not None else "")
    parts.append(f"[TARGET] {target_body}")
    for message in nxt:
        parts.append(f"[NEXT] {message.content}")
    return "\n\n".join(parts)


def memories_with_contextual_retrieval_text(
    conversation: Conversation,
    memories: Iterable[Memory],
    *,
    previous_messages: int = DEFAULT_PREVIOUS_MESSAGES,
    next_messages: int = DEFAULT_NEXT_MESSAGES,
    same_session_only: bool = True,
) -> list[Memory]:
    """Return **new** Memory objects whose ``content`` is contextual retrieval text.

    Original memories are not mutated. Callers must keep original ``MemorySource`` rows so
    evaluation still credits only the target ``message_id``.
    """
    return [
        replace(
            memory,
            content=build_contextual_retrieval_text(
                conversation,
                memory,
                previous_messages=previous_messages,
                next_messages=next_messages,
                same_session_only=same_session_only,
            ),
        )
        for memory in memories
    ]
