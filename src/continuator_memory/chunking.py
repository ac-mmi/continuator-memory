"""Deterministic sliding-window chunking over whole messages."""

from __future__ import annotations

from continuator_memory.models import Chunk, Conversation, Message

CHUNKER_NAME = "message_window_v1"


def format_message(message: Message) -> str:
    return f"{message.role}: {message.content}"


def chunk_conversation(
    conversation: Conversation, window_size: int = 4, overlap: int = 2
) -> list[Chunk]:
    """Split a conversation into windows of ``window_size`` consecutive messages.

    Consecutive windows share ``overlap`` messages. Windows are emitted until one
    contains the final message, so every message is covered and the last window
    may be shorter. Output depends only on the inputs.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if not 0 <= overlap < window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")

    messages = conversation.messages
    stride = window_size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(messages), stride):
        window = messages[start : start + window_size]
        position = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{conversation.conversation_id}:c{position:04d}",
                conversation_id=conversation.conversation_id,
                position=position,
                text="\n".join(format_message(m) for m in window),
                source_message_ids=tuple(m.message_id for m in window),
                metadata={
                    "chunker": CHUNKER_NAME,
                    "window_size": window_size,
                    "overlap": overlap,
                    "start_position": window[0].position,
                    "end_position": window[-1].position,
                },
            )
        )
        if start + window_size >= len(messages):
            break
    return chunks
