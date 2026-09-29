"""Orchestration: ingest -> chunk -> embed -> store, and retrieve -> generate."""

from __future__ import annotations

from dataclasses import dataclass

from continuator_memory.chunking import chunk_conversation
from continuator_memory.embedding import Embedder
from continuator_memory.generation import SYSTEM_PROMPT, Generator, build_prompt
from continuator_memory.models import Chunk, Conversation, RetrievedChunk
from continuator_memory.retrieval import Retriever
from continuator_memory.store import Store


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    model_name: str
    prompt: str
    retrieved: list[RetrievedChunk]


class Pipeline:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        generator: Generator | None = None,
        window_size: int = 4,
        overlap: int = 2,
    ):
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.window_size = window_size
        self.overlap = overlap
        self.retriever = Retriever(store, embedder)

    def ingest(self, conversation: Conversation) -> list[Chunk]:
        chunks = chunk_conversation(conversation, self.window_size, self.overlap)
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.save_conversation(conversation, chunks, embeddings, self.embedder.model_name)
        return chunks

    def retrieve(
        self, question: str, k: int, conversation_id: str | None = None
    ) -> list[RetrievedChunk]:
        return self.retriever.retrieve(question, k, conversation_id)

    def ask(self, question: str, k: int, conversation_id: str | None = None) -> Answer:
        if self.generator is None:
            raise RuntimeError("Pipeline was created without a generator")
        retrieved = self.retrieve(question, k, conversation_id)
        prompt = build_prompt(question, retrieved)
        text = self.generator.generate(SYSTEM_PROMPT, prompt)
        return Answer(
            question=question,
            answer=text,
            model_name=self.generator.model_name,
            prompt=prompt,
            retrieved=retrieved,
        )
