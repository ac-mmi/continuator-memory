"""Previous-turn context for the experimental contextual retrieval option.

A memory's context text is the dialogue turn just before the memory's own message, followed by
the memory content::

    context = previous_message.content + "\\n" + memory.content

The previous turn is used only when it is in the same session, which is decided by equal,
non-empty message timestamps (the LoCoMo adapter stamps every turn of a session with the session
date). The first turn of a conversation or session, a memory whose message is not in the
conversation, and messages without timestamps get no context. The memory itself is unchanged;
the context text is only an extra embedding used for ranking.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from continuator_memory.models import Conversation
from continuator_memory.v2.memory import Memory

DEFAULT_CONTEXT_WEIGHT = 0.0
CONTEXT_WEIGHT_ENV = "CONTINUATOR_CONTEXT_WEIGHT"


def load_context_weight(env: dict[str, str] | None = None) -> float:
    """Experimental. Unset env keeps contextual retrieval off (weight 0)."""
    raw = ((env if env is not None else os.environ).get(CONTEXT_WEIGHT_ENV) or "").strip()
    if not raw:
        return DEFAULT_CONTEXT_WEIGHT
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{CONTEXT_WEIGHT_ENV} must be a number") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{CONTEXT_WEIGHT_ENV} must be between 0 and 1")
    return value


def previous_turn_contexts(conversation: Conversation,
                           memories: Iterable[Memory]) -> dict[tuple[str, str], str]:
    """Context text per ``(conversation_id, memory_id)``; memories without context are omitted."""
    ordered = sorted(conversation.messages, key=lambda m: m.position)
    index = {m.message_id: i for i, m in enumerate(ordered)}
    contexts: dict[tuple[str, str], str] = {}
    for memory in memories:
        if memory.conversation_id != conversation.conversation_id:
            continue
        i = index.get(memory.message_id)
        if not i:  # missing message, or the first message of the conversation
            continue
        current, previous = ordered[i], ordered[i - 1]
        if not current.timestamp or current.timestamp != previous.timestamp:
            continue
        contexts[(memory.conversation_id, memory.memory_id)] = f"{previous.content}\n{memory.content}"
    return contexts
