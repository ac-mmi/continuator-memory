"""Deterministic local context windows for atomization.

Consecutive messages are grouped into windows of at most ``targets_per_window`` TARGET
messages, each preceded by up to ``context_before`` earlier messages shown as CONTEXT only.
Every message is a target in exactly one window and is never split across calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from continuator_memory.models import Conversation, Message

DEFAULT_TARGETS_PER_WINDOW = 4
DEFAULT_CONTEXT_BEFORE = 4


@dataclass(frozen=True)
class ContextWindow:
    conversation_id: str
    index: int
    context: tuple[Message, ...]
    targets: tuple[Message, ...]

    @property
    def messages(self) -> tuple[Message, ...]:
        return self.context + self.targets

    @property
    def target_ids(self) -> tuple[str, ...]:
        return tuple(m.message_id for m in self.targets)

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(m.message_id for m in self.context)

    def message(self, message_id: str) -> Message | None:
        return next((m for m in self.messages if m.message_id == message_id), None)

    def preceding(self, target: Message) -> tuple[Message, ...]:
        """Context and earlier targets: the messages that may resolve references in ``target``."""
        return tuple(m for m in self.messages if m.position < target.position)


def build_windows(
    conversation: Conversation,
    targets_per_window: int = DEFAULT_TARGETS_PER_WINDOW,
    context_before: int = DEFAULT_CONTEXT_BEFORE,
) -> list[ContextWindow]:
    if targets_per_window < 1:
        raise ValueError("targets_per_window must be >= 1")
    if context_before < 0:
        raise ValueError("context_before must be >= 0")
    messages = conversation.messages
    windows = []
    for index, start in enumerate(range(0, len(messages), targets_per_window)):
        windows.append(ContextWindow(
            conversation_id=conversation.conversation_id,
            index=index,
            context=messages[max(0, start - context_before):start],
            targets=messages[start:start + targets_per_window],
        ))
    return windows
