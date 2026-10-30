"""V2 memory retrieval (phase 3A): rank consolidated memories for a question.

Retrieves memories only, never raw chunks or messages, and answers nothing. Each result keeps
its sources (claim -> message) and the direct version links of its memory.

Embeddings come from the project's ``Embedder`` (the pinned Sentence Transformers model in
production) and are built by ``MemoryRetriever.index``, never during consolidation.

Ranking rule, per eligible memory::

    score = similarity + adjustment + lexical_weight * lexical
    adjustment = 0                         if the memory's status is aligned with the question
               = -RetrievalConfig.state_margin  otherwise
    lexical   = fraction of question content stems found in the memory (0 when unused)

``lexical_weight`` defaults to 0, so Baseline ranking is ``similarity + adjustment``.

Experimental contextual retrieval: when ``context_weight`` (w) is positive and the memory was
indexed with a context text (see ``continuator_memory.v2.turn_context``), ``similarity`` in the
score above is replaced by ``(1 - w) * similarity + w * context_similarity``. ``similarity``
stays the raw memory cosine (used for ``min_similarity``) and the result is still the memory
itself. ``context_weight`` defaults to 0, which skips context embedding and scoring entirely.

``current`` and ``contested`` are always aligned. ``historical`` and ``superseded`` are aligned
when the question has a past cue; ``future`` and ``conditional`` when it has a future or
conditional cue. Nothing is boosted. Ties: score, then status preference (statuses the question
asks for, then other aligned, then the rest), then conversation position, then memory_id.

Eligibility: every status is eligible except a ``superseded`` memory whose replacement is in the
index, which is excluded unless the question has a past cue. Results also need
``similarity > min_similarity`` (default 0.0: no positive semantic overlap, no result).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from continuator_memory.embedding import Embedder
from continuator_memory.retrieval import cosine_similarity
from continuator_memory.v2.lexical import fuse_scores, lexical_overlap
from continuator_memory.v2.memory import (
    LINK_SUPERSEDES,
    LINK_UPDATES,
    STATUS_CONDITIONAL,
    STATUS_CONTESTED,
    STATUS_CURRENT,
    STATUS_FUTURE,
    STATUS_HISTORICAL,
    STATUS_SUPERSEDED,
    Memory,
    MemoryLink,
    MemorySource,
)

DEFAULT_K = 5
# Similarity must exceed this; 0.0 drops only memories with no positive similarity at all.
DEFAULT_MIN_SIMILARITY = 0.0
# Penalty for a status the question did not ask for; small enough to reorder only near-ties.
DEFAULT_STATE_MARGIN = 0.05

# Lexical temporal cues in the question. Only used to unlock statuses and pick the penalty;
# a question without cues is ranked by similarity with the default (current-state) alignment.
PAST_CUE = re.compile(
    r"\b(before|previous(?:ly)?|used to|formerly|former|earlier|originally|in the past|past|"
    r"history|did|was|were|had|ago|anymore|any more)\b")
FUTURE_CUE = re.compile(
    r"\b(will|going to|gonna|plan|plans|planned|planning|intend\w*|next|upcoming|future|soon|"
    r"later|schedul\w*)\b")
CONDITIONAL_CUE = re.compile(
    r"\b(if|might|could|would|consider\w*|thinking about|thinking of|possibl\w*|maybe|perhaps|"
    r"whether|may (?:i|we|you|he|she|they|it))\b")
CURRENT_CUE = re.compile(r"\b(now|currently|current|these days|nowadays|still|at the moment|presently|today)\b")

_ALWAYS_ALIGNED = (STATUS_CURRENT, STATUS_CONTESTED)
_PAST_STATUSES = (STATUS_HISTORICAL, STATUS_SUPERSEDED)
_PROSPECTIVE_STATUSES = (STATUS_FUTURE, STATUS_CONDITIONAL)
_STATUS_TIEBREAK = {s: i for i, s in enumerate((STATUS_CURRENT, STATUS_CONTESTED, STATUS_HISTORICAL,
                                                 STATUS_SUPERSEDED, STATUS_FUTURE, STATUS_CONDITIONAL))}
_REPLACING_LINKS = (LINK_UPDATES, LINK_SUPERSEDES)
_SCORE_DIGITS = 9  # float noise below this never reorders results

EXCLUDED_SUPERSEDED = "superseded_replacement_indexed"
EXCLUDED_LOW_SIMILARITY = "below_min_similarity"


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = DEFAULT_K
    min_similarity: float | None = DEFAULT_MIN_SIMILARITY  # None keeps every eligible memory
    state_margin: float = DEFAULT_STATE_MARGIN
    lexical_weight: float = 0.0  # 0 keeps Baseline; hybrid uses a positive weight
    context_weight: float = 0.0  # 0 keeps context off; experimental contextual retrieval uses 0.5

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.state_margin < 0:
            raise ValueError("state_margin must be >= 0")
        if self.lexical_weight < 0:
            raise ValueError("lexical_weight must be >= 0")
        if not 0.0 <= self.context_weight <= 1.0:
            raise ValueError("context_weight must be between 0 and 1")


@dataclass(frozen=True)
class QuestionIntent:
    past: bool = False
    future: bool = False
    conditional: bool = False
    current: bool = False
    cues: tuple[str, ...] = ()

    @property
    def prospective(self) -> bool:
        return self.future or self.conditional

    @property
    def unspecified(self) -> bool:
        return not (self.past or self.prospective or self.current)

    def aligned(self, status: str) -> bool:
        return (status in _ALWAYS_ALIGNED or (self.past and status in _PAST_STATUSES)
                or (self.prospective and status in _PROSPECTIVE_STATUSES))

    def preference(self, status: str) -> int:
        """0 for statuses the question asks for, 1 for other aligned statuses, 2 otherwise."""
        asked = ((self.past and status in _PAST_STATUSES) or (self.prospective and status in _PROSPECTIVE_STATUSES)
                 or (not (self.past or self.prospective) and status in _ALWAYS_ALIGNED))
        return 0 if asked else 1 if self.aligned(status) else 2

    def to_dict(self) -> dict[str, Any]:
        return {"past": self.past, "future": self.future, "conditional": self.conditional,
                "current": self.current, "cues": list(self.cues)}


def question_intent(question: str) -> QuestionIntent:
    text = " ".join(question.lower().replace("\u2019", "'").split())
    found = {name: [m.group(0) for m in rx.finditer(text)]
             for name, rx in (("past", PAST_CUE), ("future", FUTURE_CUE), ("conditional", CONDITIONAL_CUE),
                              ("current", CURRENT_CUE))}
    cues = tuple(dict.fromkeys(c for name in found for c in found[name]))
    return QuestionIntent(bool(found["past"]), bool(found["future"]), bool(found["conditional"]),
                          bool(found["current"]), cues)


@dataclass(frozen=True)
class VersionLink:
    """A direct stored link of the result's memory; ``role`` is the other memory's side."""

    type: str
    role: str  # "newer" (other memory replaced/contradicts/refers to this) or "older"
    memory_id: str
    relation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "role": self.role, "memory_id": self.memory_id, "relation_id": self.relation_id}


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    memory_id: str
    conversation_id: str
    content: str
    similarity: float
    adjustment: float
    score: float
    status: str
    time_class: str
    event_time: str | None
    valid_from: str | None
    valid_until: str | None
    superseded_by: str | None
    sources: tuple[MemorySource, ...]
    links: tuple[VersionLink, ...]
    context_similarity: float | None = None  # set only when contextual retrieval scored the memory

    @property
    def source_claim_ids(self) -> tuple[str, ...]:
        return tuple(s.claim_id for s in self.sources)

    @property
    def source_message_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.message_id for s in self.sources))

    def debug(self) -> dict[str, Any]:
        return {"rank": self.rank, "memory_id": self.memory_id, "status": self.status,
                "similarity": round(self.similarity, 6), "adjustment": round(self.adjustment, 6),
                "score": round(self.score, 6), "source_claim_ids": list(self.source_claim_ids),
                "source_message_ids": list(self.source_message_ids),
                "links": [f"{l.type} {l.role} {l.memory_id}" for l in self.links]}

    def to_dict(self) -> dict[str, Any]:
        out = {"rank": self.rank, "memory_id": self.memory_id, "conversation_id": self.conversation_id,
               "content": self.content, "similarity": self.similarity, "adjustment": self.adjustment,
               "score": self.score, "status": self.status, "time_class": self.time_class,
               "event_time": self.event_time, "valid_from": self.valid_from, "valid_until": self.valid_until,
               "superseded_by": self.superseded_by, "sources": [s.to_dict() for s in self.sources],
               "links": [l.to_dict() for l in self.links]}
        if self.context_similarity is not None:
            out["context_similarity"] = self.context_similarity
        return out


@dataclass(frozen=True)
class RetrievalOutcome:
    question: str
    intent: QuestionIntent
    results: tuple[RetrievalResult, ...]
    excluded: tuple[tuple[str, str], ...] = ()  # (memory_id, reason), in index order

    def debug(self) -> dict[str, Any]:
        return {"question": self.question, "intent": self.intent.to_dict(),
                "results": [r.debug() for r in self.results],
                "excluded": [{"memory_id": mid, "reason": why} for mid, why in self.excluded]}


@dataclass(frozen=True)
class IndexStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0


@dataclass
class _Entry:
    memory: Memory
    vector: np.ndarray
    sources: tuple[MemorySource, ...] = ()
    links: tuple[VersionLink, ...] = field(default_factory=tuple)
    context: str | None = None
    context_vector: np.ndarray | None = None


class MemoryRetriever:
    """In-memory embedding index over consolidated V2 memories, keyed by (conversation, memory)."""

    def __init__(self, embedder: Embedder, config: RetrievalConfig | None = None):
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self._entries: dict[tuple[str, str], _Entry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    # ---- indexing ---------------------------------------------------------

    def index(self, memories: Iterable[Memory], sources: Iterable[MemorySource] = (),
              links: Iterable[MemoryLink] = (), conversation_id: str | None = None,
              contexts: Mapping[tuple[str, str], str] | None = None) -> IndexStats:
        """Sync the index to a snapshot of each conversation's memories.

        Memories of a synced conversation that are absent from the snapshot are dropped; content
        is re-embedded only when it changed. Pass ``conversation_id`` to sync an empty snapshot.
        ``contexts`` maps ``(conversation_id, memory_id)`` to an optional context text, embedded
        separately and only when it changed; memories without one have no context vector.
        """
        contexts = contexts or {}
        memories = sorted(memories, key=lambda m: (m.conversation_id, m.position, m.memory_id))
        synced = {m.conversation_id for m in memories} | ({conversation_id} if conversation_id else set())
        by_memory: dict[str, list[MemorySource]] = {}
        for s in sources:
            by_memory.setdefault(s.memory_id, []).append(s)
        link_list = list(links)

        stale = [key for key in self._entries if key[0] in synced]
        keep = {(m.conversation_id, m.memory_id) for m in memories}
        removed = 0
        for key in stale:
            if key not in keep:
                del self._entries[key]
                removed += 1

        to_embed = [m for m in memories
                    if (m.conversation_id, m.memory_id) not in self._entries
                    or self._entries[(m.conversation_id, m.memory_id)].memory.content != m.content]
        vectors = self.embedder.embed([m.content for m in to_embed]) if to_embed else []
        fresh = {(m.conversation_id, m.memory_id): np.asarray(v, dtype=np.float32) for m, v in zip(to_embed, vectors)}

        context_embed = [key for key in ((m.conversation_id, m.memory_id) for m in memories)
                         if contexts.get(key) and (key not in self._entries
                                                   or self._entries[key].context != contexts[key]
                                                   or self._entries[key].context_vector is None)]
        context_vectors = self.embedder.embed([contexts[key] for key in context_embed]) if context_embed else []
        fresh_context = {key: np.asarray(v, dtype=np.float32) for key, v in zip(context_embed, context_vectors)}

        added = updated = 0
        for m in memories:
            key = (m.conversation_id, m.memory_id)
            if key in fresh:
                added, updated = (added + 1, updated) if key not in self._entries else (added, updated + 1)
                vector = fresh[key]
            else:
                vector = self._entries[key].vector
            context = contexts.get(key) or None
            if context is None:
                context_vector = None
            elif key in fresh_context:
                context_vector = fresh_context[key]
            else:
                context_vector = self._entries[key].context_vector
            own = sorted(by_memory.get(m.memory_id, ()), key=lambda s: (s.position, s.claim_id))
            self._entries[key] = _Entry(m, vector, tuple(own), _direct_links(m.memory_id, link_list),
                                        context, context_vector)
        return IndexStats(added, updated, len(memories) - added - updated, removed)

    def index_consolidator(self, consolidator,
                           contexts: Mapping[tuple[str, str], str] | None = None) -> IndexStats:
        return self.index(consolidator.memories.values(), consolidator.sources, consolidator.links,
                          consolidator.conversation_id, contexts)

    def index_store(self, memory_store, conversation_id: str) -> IndexStats:
        return self.index(memory_store.get_memories(conversation_id), memory_store.get_sources(conversation_id),
                          memory_store.get_links(conversation_id), conversation_id)

    # ---- retrieval --------------------------------------------------------

    def retrieve(self, question: str, k: int | None = None, conversation_id: str | None = None) -> list[RetrievalResult]:
        return list(self.search(question, k, conversation_id).results)

    def search(self, question: str, k: int | None = None, conversation_id: str | None = None) -> RetrievalOutcome:
        k = self.config.k if k is None else k
        if k < 1:
            raise ValueError("k must be >= 1")
        intent = question_intent(question)
        entries = sorted((e for key, e in self._entries.items() if conversation_id in (None, key[0])),
                         key=lambda e: (e.memory.conversation_id, e.memory.position, e.memory.memory_id))
        if not entries:
            return RetrievalOutcome(question, intent, ())
        query = self.embedder.embed([question])[0]
        similarities = cosine_similarity(query, np.stack([e.vector for e in entries]))
        weight = self.config.context_weight
        context_similarities: dict[int, float] = {}
        if weight:
            with_context = [i for i, e in enumerate(entries) if e.context_vector is not None]
            if with_context:
                values = cosine_similarity(query, np.stack([entries[i].context_vector for i in with_context]))
                context_similarities = {i: float(v) for i, v in zip(with_context, values)}

        excluded: list[tuple[str, str]] = []
        candidates = []
        for i, (entry, similarity) in enumerate(zip(entries, similarities)):
            m, similarity = entry.memory, float(similarity)
            if m.status == STATUS_SUPERSEDED and not intent.past and self._replacement_indexed(entry):
                excluded.append((m.memory_id, EXCLUDED_SUPERSEDED))
                continue
            if self.config.min_similarity is not None and similarity <= self.config.min_similarity:
                excluded.append((m.memory_id, EXCLUDED_LOW_SIMILARITY))
                continue
            adjustment = 0.0 if intent.aligned(m.status) else -self.config.state_margin
            lexical = (
                lexical_overlap(question, m.content) if self.config.lexical_weight else 0.0
            )
            context_similarity = context_similarities.get(i)
            ranking_similarity = (similarity if context_similarity is None
                                  else (1.0 - weight) * similarity + weight * context_similarity)
            score = fuse_scores(
                ranking_similarity, lexical, weight=self.config.lexical_weight, adjustment=adjustment,
            )
            candidates.append((entry, similarity, adjustment, score, context_similarity))

        candidates.sort(key=lambda c: (-round(c[3], _SCORE_DIGITS), intent.preference(c[0].memory.status),
                                       _STATUS_TIEBREAK.get(c[0].memory.status, len(_STATUS_TIEBREAK)),
                                       c[0].memory.conversation_id, c[0].memory.position, c[0].memory.memory_id))
        results = tuple(_result(rank, entry, similarity, adjustment, score, context_similarity)
                        for rank, (entry, similarity, adjustment, score, context_similarity)
                        in enumerate(candidates[:k], 1))
        return RetrievalOutcome(question, intent, results, tuple(excluded))

    def _replacement_indexed(self, entry: _Entry) -> bool:
        cid = entry.memory.conversation_id
        newer = {entry.memory.superseded_by} | {l.memory_id for l in entry.links
                                                if l.role == "newer" and l.type in _REPLACING_LINKS}
        return any((cid, mid) in self._entries for mid in newer if mid)


def _direct_links(mid: str, links: list[MemoryLink]) -> tuple[VersionLink, ...]:
    out = []
    for link in links:
        if link.older_memory_id == mid:
            out.append(VersionLink(link.type, "newer", link.newer_memory_id, link.relation_id))
        elif link.newer_memory_id == mid:
            out.append(VersionLink(link.type, "older", link.older_memory_id, link.relation_id))
    return tuple(dict.fromkeys(out))


def _result(rank: int, entry: _Entry, similarity: float, adjustment: float,
            score: float | None = None, context_similarity: float | None = None) -> RetrievalResult:
    m = entry.memory
    return RetrievalResult(rank, m.memory_id, m.conversation_id, m.content, similarity, adjustment,
                           similarity + adjustment if score is None else score,
                           m.status, m.time_class, m.event_time, m.valid_from,
                           m.valid_until, m.superseded_by, entry.sources, entry.links, context_similarity)
