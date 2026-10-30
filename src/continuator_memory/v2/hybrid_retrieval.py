"""V2 hybrid retrieval (phase 3B): raw conversation evidence fused with V2 memory evidence.

    question --> raw retriever (V1 Retriever, unchanged) --> chunks --> their messages
             +-> memory retriever (phase 3A, unchanged) --> memories --> their source messages
                                        |
                  one evidence item per (conversation_id, message_id), carrying every path
                  that reached it, then deterministic reciprocal-rank fusion and final top K

The evidence unit is a raw message: raw chunks are expanded to the messages they contain and
memories to the messages of their own sources (``MemorySource``). Nothing else is expanded: no
neighbouring messages, no links followed, no second search. Memories phase 3A excluded are
never reintroduced through the memory path; their messages appear only if raw retrieval found
them. Disagreements between raw text and memory state are left for the answer layer.

Fusion, per message::

    fusion = 1 / (rrf_c + raw_rank)       if raw retrieval reached the message
           + 1 / (rrf_c + memory_rank)    if a retrieved memory has it as a source

``raw_rank`` is the rank of the best retrieved chunk containing the message; ``memory_rank``
is the best phase 3A rank among memories sourced from it. Only ranks are fused, because raw
chunk cosine and phase 3A's status-adjusted memory score are on different scales. A message
found by both paths gets both terms; there is no other bonus. Ties: fusion, raw rank, memory
rank (missing ranks last), conversation_id, message position, message_id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from continuator_memory.models import Message, RetrievedChunk
from continuator_memory.v2.memory_retrieval import MemoryRetriever, RetrievalOutcome, RetrievalResult, VersionLink

RAW = "raw"
MEMORY = "memory"
# Evidence types: which retrieval paths reached the message.
EVIDENCE_RAW = "raw"
EVIDENCE_MEMORY_SOURCE = "memory_source"
EVIDENCE_RAW_AND_MEMORY = "raw+memory"

DEFAULT_RAW_K = 5
DEFAULT_MEMORY_K = 5
DEFAULT_FINAL_K = 5
# Standard reciprocal-rank-fusion constant (Cormack et al., 2009); it only flattens the rank
# curve and is the same for both paths.
DEFAULT_RRF_C = 60
_SCORE_DIGITS = 12


class RawRetriever(Protocol):
    def retrieve(self, question: str, k: int, conversation_id: str | None = None) -> list[RetrievedChunk]: ...


class MessageSource(Protocol):
    def get_messages(self, conversation_id: str, message_ids: list[str] | tuple[str, ...]) -> list[Message]: ...


@dataclass(frozen=True)
class HybridConfig:
    raw_k: int = DEFAULT_RAW_K
    memory_k: int = DEFAULT_MEMORY_K
    final_k: int = DEFAULT_FINAL_K
    rrf_c: int = DEFAULT_RRF_C

    def __post_init__(self) -> None:
        if min(self.raw_k, self.memory_k, self.final_k) < 1:
            raise ValueError("raw_k, memory_k and final_k must be >= 1")
        if self.rrf_c < 0:
            raise ValueError("rrf_c must be >= 0")


@dataclass(frozen=True)
class MemoryEvidence:
    """A phase 3A result that has this message as a source, with the claims linking them."""

    memory_id: str
    content: str
    status: str
    time_class: str
    rank: int  # phase 3A rank
    similarity: float
    score: float  # phase 3A score (similarity + status adjustment)
    claim_ids: tuple[str, ...]  # this memory's source claims from this message
    event_time: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by: str | None = None
    links: tuple[VersionLink, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"memory_id": self.memory_id, "content": self.content, "status": self.status,
                "time_class": self.time_class, "rank": self.rank, "similarity": self.similarity,
                "score": self.score, "claim_ids": list(self.claim_ids), "event_time": self.event_time,
                "valid_from": self.valid_from, "valid_until": self.valid_until,
                "superseded_by": self.superseded_by, "links": [l.to_dict() for l in self.links]}


@dataclass(frozen=True)
class HybridRetrievalResult:
    rank: int
    conversation_id: str
    message_id: str
    position: int | None
    role: str | None
    timestamp: str | None
    text: str | None  # original message content; None only if the message is not stored
    sources: tuple[str, ...]  # retrieval paths: "raw", "memory"
    fusion_score: float
    raw_rank: int | None = None
    raw_score: float | None = None  # cosine of the best retrieved chunk containing the message
    chunk_ids: tuple[str, ...] = ()  # retrieved chunks containing the message, best first
    memories: tuple[MemoryEvidence, ...] = ()  # best phase 3A rank first

    @property
    def evidence_type(self) -> str:
        if RAW in self.sources and MEMORY in self.sources:
            return EVIDENCE_RAW_AND_MEMORY
        return EVIDENCE_RAW if RAW in self.sources else EVIDENCE_MEMORY_SOURCE

    @property
    def memory_rank(self) -> int | None:
        return self.memories[0].rank if self.memories else None

    @property
    def memory_similarity(self) -> float | None:
        return self.memories[0].similarity if self.memories else None

    @property
    def memory_score(self) -> float | None:
        return self.memories[0].score if self.memories else None

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(m.memory_id for m in self.memories)

    @property
    def claim_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c for m in self.memories for c in m.claim_ids))

    def debug(self) -> dict[str, Any]:
        return {"rank": self.rank, "message_id": self.message_id, "fusion_score": round(self.fusion_score, 6),
                "raw_rank": self.raw_rank, "raw_score": None if self.raw_score is None else round(self.raw_score, 6),
                "memory_rank": self.memory_rank,
                "memory_score": None if self.memory_score is None else round(self.memory_score, 6),
                "sources": list(self.sources), "memory_ids": list(self.memory_ids),
                "status": [m.status for m in self.memories]}

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "conversation_id": self.conversation_id, "message_id": self.message_id,
                "position": self.position, "role": self.role, "timestamp": self.timestamp, "text": self.text,
                "sources": list(self.sources), "evidence_type": self.evidence_type,
                "fusion_score": self.fusion_score, "raw_rank": self.raw_rank, "raw_score": self.raw_score,
                "chunk_ids": list(self.chunk_ids), "memory_rank": self.memory_rank,
                "memory_similarity": self.memory_similarity, "memory_score": self.memory_score,
                "memories": [m.to_dict() for m in self.memories]}


@dataclass(frozen=True)
class HybridOutcome:
    question: str
    results: tuple[HybridRetrievalResult, ...]
    raw: tuple[RetrievedChunk, ...]  # raw retrieval output, as returned
    memory: RetrievalOutcome  # phase 3A output, as returned
    candidates: int  # distinct messages before the final cut

    def debug(self) -> dict[str, Any]:
        return {"question": self.question, "candidates": self.candidates,
                "raw": [{"rank": c.rank, "chunk_id": c.chunk_id, "score": round(c.score, 6),
                         "message_ids": list(c.source_message_ids)} for c in self.raw],
                "memory": [r.debug() for r in self.memory.results],
                "memory_excluded": [{"memory_id": mid, "reason": why} for mid, why in self.memory.excluded],
                "results": [r.debug() for r in self.results]}


class HybridRetriever:
    def __init__(self, raw_retriever: RawRetriever, memory_retriever: MemoryRetriever, messages: MessageSource,
                 config: HybridConfig | None = None):
        self.raw_retriever = raw_retriever
        self.memory_retriever = memory_retriever
        self.messages = messages
        self.config = config or HybridConfig()

    def retrieve(self, question: str, k: int | None = None,
                 conversation_id: str | None = None) -> list[HybridRetrievalResult]:
        return list(self.search(question, k, conversation_id).results)

    def search(self, question: str, k: int | None = None, conversation_id: str | None = None) -> HybridOutcome:
        k = self.config.final_k if k is None else k
        if k < 1:
            raise ValueError("k must be >= 1")
        raw = tuple(self.raw_retriever.retrieve(question, self.config.raw_k, conversation_id))
        memory = self.memory_retriever.search(question, self.config.memory_k, conversation_id)

        candidates: dict[tuple[str, str], _Candidate] = {}
        for chunk in raw:
            for mid in dict.fromkeys(chunk.source_message_ids):
                candidates.setdefault((chunk.conversation_id, mid), _Candidate()).add_chunk(chunk)
        for result in memory.results:
            for (cid, mid), claim_ids in _messages_of(result).items():
                candidates.setdefault((cid, mid), _Candidate()).memories.append(_memory_evidence(result, claim_ids))

        messages = self._load_messages(candidates)
        c = self.config.rrf_c
        scored = []
        for key, cand in candidates.items():
            fusion = (1 / (c + cand.raw_rank) if cand.raw_rank is not None else 0.0) + \
                     (1 / (c + cand.memory_rank) if cand.memories else 0.0)
            scored.append((key, cand, fusion, messages.get(key)))
        missing = 1 << 30
        scored.sort(key=lambda s: (-round(s[2], _SCORE_DIGITS),
                                   s[1].raw_rank if s[1].raw_rank is not None else missing,
                                   s[1].memory_rank if s[1].memories else missing,
                                   s[0][0], s[3].position if s[3] else missing, s[0][1]))
        results = tuple(_result(rank, key, cand, fusion, message)
                        for rank, (key, cand, fusion, message) in enumerate(scored[:k], 1))
        return HybridOutcome(question, results, raw, memory, len(candidates))

    def _load_messages(self, candidates: dict[tuple[str, str], _Candidate]) -> dict[tuple[str, str], Message]:
        by_conversation: dict[str, list[str]] = {}
        for cid, mid in sorted(candidates):
            by_conversation.setdefault(cid, []).append(mid)
        return {(cid, m.message_id): m for cid, mids in by_conversation.items()
                for m in self.messages.get_messages(cid, mids)}


class _Candidate:
    def __init__(self) -> None:
        self.chunks: list[RetrievedChunk] = []
        self.memories: list[MemoryEvidence] = []

    def add_chunk(self, chunk: RetrievedChunk) -> None:
        if chunk.chunk_id not in {c.chunk_id for c in self.chunks}:
            self.chunks.append(chunk)

    @property
    def raw_rank(self) -> int | None:
        return min((c.rank for c in self.chunks), default=None)

    @property
    def memory_rank(self) -> int:
        return min(m.rank for m in self.memories)


def _messages_of(result: RetrievalResult) -> dict[tuple[str, str], tuple[str, ...]]:
    """The memory's own source messages, each with the source claims from that message."""
    out: dict[tuple[str, str], list[str]] = {}
    for s in result.sources:
        out.setdefault((result.conversation_id, s.message_id), []).append(s.claim_id)
    return {key: tuple(dict.fromkeys(claims)) for key, claims in out.items()}


def _memory_evidence(r: RetrievalResult, claim_ids: tuple[str, ...]) -> MemoryEvidence:
    return MemoryEvidence(r.memory_id, r.content, r.status, r.time_class, r.rank, r.similarity, r.score, claim_ids,
                          r.event_time, r.valid_from, r.valid_until, r.superseded_by, r.links)


def _result(rank: int, key: tuple[str, str], cand: _Candidate, fusion: float,
            message: Message | None) -> HybridRetrievalResult:
    chunks = sorted(cand.chunks, key=lambda c: (c.rank, c.chunk_id))
    memories = tuple(sorted(cand.memories, key=lambda m: (m.rank, m.memory_id)))
    sources = tuple(s for s, present in ((RAW, bool(chunks)), (MEMORY, bool(memories))) if present)
    return HybridRetrievalResult(
        rank, key[0], key[1], message.position if message else None, message.role if message else None,
        message.timestamp if message else None, message.content if message else None, sources, fusion,
        chunks[0].rank if chunks else None, float(chunks[0].score) if chunks else None,
        tuple(c.chunk_id for c in chunks), memories)
