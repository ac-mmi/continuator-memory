"""Phase 3 pipeline: Phase 1 ingestion + memory formation; memory-aware retrieval and answers.

Exposes the same ``ingest`` / ``retrieve`` / ``ask`` interface as the Phase 1
``Pipeline`` so the Phase 2 harness can evaluate either one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from continuator_memory.embedding import Embedder
from continuator_memory.generation import Generator
from continuator_memory.memory.context import MEMORY_SYSTEM_PROMPT, build_memory_prompt
from continuator_memory.memory.formation import Extractor, FormationReport, MemoryFormation
from continuator_memory.memory.retrieval import MemoryRetrieval, MemoryRetriever
from continuator_memory.memory.store import MemoryStore
from continuator_memory.models import Chunk, Conversation, RetrievedChunk
from continuator_memory.pipeline import Answer, Pipeline
from continuator_memory.store import Store


@dataclass(frozen=True)
class MemoryAnswer(Answer):
    memory_trace: dict[str, Any] = field(default_factory=dict)


class MemoryPipeline:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        generator: Generator | None,
        extractor: Extractor,
        window_size: int = 4,
        overlap: int = 2,
    ):
        self.base = Pipeline(store, embedder, generator, window_size, overlap)
        self.generator = generator
        self.memory = MemoryStore(store.connection)
        self.formation = MemoryFormation(self.memory, extractor)
        self.retriever = MemoryRetriever(store, self.memory, self.base.retriever)
        self.reports: list[FormationReport] = []

    def ingest(self, conversation: Conversation) -> list[Chunk]:
        chunks = self.base.ingest(conversation)
        self.reports.append(self.formation.form(conversation))
        return chunks

    def retrieve_memory(
        self, question: str, k: int, conversation_id: str | None = None
    ) -> MemoryRetrieval:
        return self.retriever.retrieve(question, k, conversation_id)

    def retrieve(
        self, question: str, k: int, conversation_id: str | None = None
    ) -> list[RetrievedChunk]:
        return self.retrieve_memory(question, k, conversation_id).chunks

    def ask(self, question: str, k: int, conversation_id: str | None = None) -> MemoryAnswer:
        if self.generator is None:
            raise RuntimeError("MemoryPipeline was created without a generator")
        retrieval = self.retrieve_memory(question, k, conversation_id)
        prompt = build_memory_prompt(question, retrieval)
        text = self.generator.generate(MEMORY_SYSTEM_PROMPT, prompt)
        return MemoryAnswer(
            question=question,
            answer=text,
            model_name=self.generator.model_name,
            prompt=prompt,
            retrieved=retrieval.chunks,
            memory_trace=retrieval.trace(),
        )

    def memory_diagnostics(self) -> dict[str, Any]:
        stats = self.memory.stats()
        stats["memories_superseded"] = stats["memories"] - stats["memories_current"]
        stats["relationships_superseded"] = (
            stats["relationships"] - stats["relationships_current"]
        )
        stats["extraction_model"] = self.formation.extractor.model_name
        grounding: Counter = Counter()
        for report in self.reports:
            grounding.update(report.grounding)
        stats["grounding"] = dict(sorted(grounding.items()))
        stats["formation_ms"] = {
            "extraction": round(sum(r.extraction_ms for r in self.reports), 3),
            "consolidation": round(sum(r.consolidation_ms for r in self.reports), 3),
        }
        stats["per_conversation"] = [r.to_dict() for r in self.reports]
        return stats
