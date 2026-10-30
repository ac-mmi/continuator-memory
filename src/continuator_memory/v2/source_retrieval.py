"""Semantic memories as an index into the original conversation (memory -> provenance -> source).

    question
      -> V2 memory retrieval: ``MemoryRetriever.search`` over ``consolidate``d V2 claims (unchanged V2 code)
      -> each retrieved memory's stored provenance (``MemorySource.message_id``, one per claim it holds)
      -> the original messages of the conversation, verbatim
      -> deduplicated sources in ranked-memory order, cut at a character budget
      -> one ``RetrievedChunk`` per source whose text is the original message

The memory statement is retrieval metadata only: it is kept beside its sources for diagnostics and
never becomes evidence text. A provenance reference that does not name a message of the
conversation (other conversation, unknown message, position mismatch) or a memory without any
provenance is recorded as unresolved; no source is ever inferred for it.

Ordering is deterministic: memories in ``MemoryRetriever`` rank order; within a memory, its sources
in stored order (conversation position, claim ID); a source reached again through a later memory
keeps its first place. Budget: walk the unique resolved sources in that order and stop at the first
whose content would exceed the budget; it and every later source are dropped (recorded).

Inputs are a question string, a conversation, its V2 claims and relations, an embedder, the number
of memories and a character budget. Nothing here reads answers, labels or data-source identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from continuator_memory.embedding import Embedder
from continuator_memory.models import Conversation, Message, RetrievedChunk
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.memory_retrieval import MemoryRetriever, RetrievalConfig, RetrievalOutcome, RetrievalResult
from continuator_memory.v2.models import Claim, Relation

from continuator_memory.v2.bprime_links import (
    ALLOWED_MAX_HOPS,
    BPrimeConfig,
    ExpansionHop,
    empty_trace,
    filter_eligible,
    load_classifications,
    neighbours_for_seeds,
    sha256_file,
)

NO_PROVENANCE = "no_provenance"
OTHER_CONVERSATION = "other_conversation"
MESSAGE_NOT_FOUND = "message_not_in_conversation"
POSITION_MISMATCH = "position_mismatch"
ORIGIN_RETRIEVED = "retrieved"
ORIGIN_BPRIME = "bprime_expanded"


@dataclass(frozen=True)
class SourceMessage:
    """An original message named by a memory's provenance, with its verbatim content."""

    message_id: str
    claim_id: str
    position: int
    role: str
    content: str
    timestamp: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"message_id": self.message_id, "claim_id": self.claim_id, "position": self.position,
                "role": self.role, "chars": len(self.content), "timestamp": self.timestamp}


@dataclass(frozen=True)
class UnresolvedSource:
    claim_id: str | None
    message_id: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"claim_id": self.claim_id, "message_id": self.message_id, "reason": self.reason}


@dataclass(frozen=True)
class MemoryMatch:
    """A retrieved memory (the reason for retrieval) and the sources its provenance resolves to."""

    rank: int
    memory_id: str
    statement: str
    score: float
    similarity: float
    status: str
    sources: tuple[SourceMessage, ...]
    unresolved: tuple[UnresolvedSource, ...]

    @property
    def provenance_message_ids(self) -> tuple[str, ...]:
        ids = [s.message_id for s in self.sources] + [u.message_id for u in self.unresolved if u.message_id]
        return tuple(dict.fromkeys(ids))

    @property
    def source_message_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.message_id for s in self.sources))

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "memory_id": self.memory_id, "statement": self.statement,
                "score": self.score, "similarity": self.similarity, "status": self.status,
                "provenance_message_ids": list(self.provenance_message_ids),
                "sources": [s.to_dict() for s in self.sources],
                "unresolved": [u.to_dict() for u in self.unresolved]}


@dataclass(frozen=True)
class SourceSelection:
    budget_chars: int
    unique_sources: tuple[str, ...]  # every resolved source, deduplicated, in ranked-memory order
    selected: tuple[str, ...]
    dropped: tuple[str, ...]  # beyond the budget
    used_chars: int
    first_memory: dict[str, str]  # source message ID -> memory that first reached it

    def to_dict(self) -> dict[str, Any]:
        return {"budget_chars": self.budget_chars, "unique_sources": list(self.unique_sources),
                "selected": list(self.selected), "dropped_by_budget": list(self.dropped),
                "used_chars": self.used_chars, "first_memory": dict(self.first_memory)}


@dataclass(frozen=True)
class SourceEvidence:
    question: str
    outcome: RetrievalOutcome
    matches: tuple[MemoryMatch, ...]
    selection: SourceSelection
    chunks: tuple[RetrievedChunk, ...]
    bprime: dict[str, Any] = field(default_factory=lambda: empty_trace(enabled=False))
    n_seed_matches: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.outcome.intent.to_dict(),
            "memories": [m.to_dict() for m in self.matches],
            "selection": self.selection.to_dict(),
            "evidence_chunk_ids": [c.chunk_id for c in self.chunks],
            "n_seed_matches": self.n_seed_matches,
            "bprime": self.bprime,
        }


def memory_index(embedder: Embedder, conversation_id: str, claims: Iterable[Claim],
                 relations: Iterable[Relation] = (), config: RetrievalConfig | None = None) -> MemoryRetriever:
    """The V2 memory index exactly as Phase 4 builds it: consolidate the claims, index the consolidator."""
    retriever = MemoryRetriever(embedder, config or RetrievalConfig())
    retriever.index_consolidator(consolidate(conversation_id, claims, relations))
    return retriever


def _result_from_consolidator(consolidator: Any, memory_id: str, rank: int) -> RetrievalResult | None:
    """Build a RetrievalResult for a consolidator memory so ``resolve`` can attach provenance."""
    mem = consolidator.memories.get(memory_id)
    if mem is None:
        return None
    sources = tuple(consolidator.sources_of(memory_id))
    return RetrievalResult(
        rank=rank,
        memory_id=mem.memory_id,
        conversation_id=mem.conversation_id,
        content=mem.content,
        similarity=0.0,
        adjustment=0.0,
        score=0.0,
        status=mem.status,
        time_class=getattr(mem, "time_class", "") or "",
        event_time=getattr(mem, "event_time", None),
        valid_from=mem.valid_from,
        valid_until=mem.valid_until,
        superseded_by=mem.superseded_by,
        sources=sources,
        links=(),
    )


def _apply_expansion_hops(
    hops: list[ExpansionHop],
    consolidator: Any,
    conversation: Conversation,
    *,
    next_rank: int,
    blocked: set[str],
    skipped: list[dict[str, Any]],
) -> tuple[list[MemoryMatch], list[dict[str, Any]], list[dict[str, Any]], list[str], int]:
    """Resolve and append hop neighbours. Skips duplicates already in ``blocked``."""
    expanded: list[MemoryMatch] = []
    applied: list[dict[str, Any]] = []
    unresolved_neighbours: list[dict[str, Any]] = []
    resolved_message_ids: list[str] = []
    rank = next_rank
    for hop in hops:
        if hop.neighbour_memory_id in blocked:
            skipped.append(
                {
                    "pair_id": hop.pair_id,
                    "reason": "neighbour_already_expanded_or_seed",
                    "neighbour_memory_id": hop.neighbour_memory_id,
                    "hop_depth": hop.hop_depth,
                    "origin_memory_id": hop.seed_memory_id,
                }
            )
            continue
        result = _result_from_consolidator(consolidator, hop.neighbour_memory_id, rank)
        if result is None:
            skipped.append(
                {
                    "pair_id": hop.pair_id,
                    "reason": "neighbour_missing_from_consolidator",
                    "neighbour_memory_id": hop.neighbour_memory_id,
                    "hop_depth": hop.hop_depth,
                }
            )
            continue
        match = resolve(result, conversation)
        if not match.sources and match.unresolved:
            unresolved_neighbours.append(
                {
                    "pair_id": hop.pair_id,
                    "neighbour_memory_id": hop.neighbour_memory_id,
                    "hop_depth": hop.hop_depth,
                    "unresolved": [u.to_dict() for u in match.unresolved],
                }
            )
        expanded.append(match)
        blocked.add(hop.neighbour_memory_id)
        applied.append(
            {
                **hop.to_dict(),
                "origin": ORIGIN_BPRIME,
                "hop_depth": hop.hop_depth,
                "resolved_message_ids": list(match.source_message_ids),
            }
        )
        resolved_message_ids.extend(match.source_message_ids)
        rank += 1
    return expanded, applied, unresolved_neighbours, resolved_message_ids, rank


def expand_matches_with_bprime(
    seed_matches: tuple[MemoryMatch, ...],
    consolidator: Any,
    conversation: Conversation,
    bprime: BPrimeConfig,
) -> tuple[tuple[MemoryMatch, ...], dict[str, Any]]:
    """Append B′ neighbour matches after seeds (one hop, or two when ``max_hops=2``).

    Hop 1 expands from retrieved seeds only. Hop 2 expands only from hop-1 expanded
    memories (fixed depth; no further recursion). Seed order and ranks are unchanged;
    expanded matches are appended in hop order then deterministic neighbour order.
    """
    path = bprime.resolved_path()
    assert path is not None
    max_hops = bprime.effective_max_hops()
    if max_hops not in ALLOWED_MAX_HOPS:
        raise ValueError(f"B′ max_hops must be 1 or 2 when enabled, got {max_hops}")
    rows = load_classifications(path)
    memory_ids = frozenset(consolidator.memories)
    eligible, skipped = filter_eligible(rows, memory_ids)
    seed_ids = [m.memory_id for m in seed_matches]
    seed_ranks = {m.memory_id: m.rank for m in seed_matches}
    blocked: set[str] = set(seed_ids)

    hops1 = neighbours_for_seeds(seed_ids, eligible, seed_ranks=seed_ranks, hop_depth=1)
    expanded1, applied1, unresolved1, resolved1, next_rank = _apply_expansion_hops(
        hops1, consolidator, conversation, next_rank=len(seed_matches) + 1, blocked=blocked, skipped=skipped
    )

    expanded2: list[MemoryMatch] = []
    applied2: list[dict[str, Any]] = []
    unresolved2: list[dict[str, Any]] = []
    resolved2: list[str] = []
    if max_hops >= 2 and expanded1:
        hop1_ids = [m.memory_id for m in expanded1]
        hop1_ranks = {m.memory_id: m.rank for m in expanded1}
        hops2 = neighbours_for_seeds(
            hop1_ids,
            eligible,
            seed_ranks=hop1_ranks,
            exclude_neighbour_ids=blocked,
            hop_depth=2,
        )
        expanded2, applied2, unresolved2, resolved2, next_rank = _apply_expansion_hops(
            hops2, consolidator, conversation, next_rank=next_rank, blocked=blocked, skipped=skipped
        )

    expanded = expanded1 + expanded2
    applied = applied1 + applied2
    unresolved_neighbours = unresolved1 + unresolved2
    resolved_message_ids = list(dict.fromkeys(resolved1 + resolved2))

    trace = {
        "enabled": True,
        "max_hops": max_hops,
        "classifications_path": str(path),
        "classifications_sha256": sha256_file(path),
        "n_rows_loaded": len(rows),
        "eligible_links": [e.to_dict() for e in eligible],
        "skipped": skipped,
        "applied_hops": applied,
        "expanded_memory_ids": [m.memory_id for m in expanded],
        "expanded_memory_ids_by_hop": {
            "1": [m.memory_id for m in expanded1],
            "2": [m.memory_id for m in expanded2],
        },
        "resolved_message_ids": resolved_message_ids,
        "unresolved_neighbours": unresolved_neighbours,
        "seed_memory_ids": seed_ids,
        "seed_origins": [{"memory_id": m.memory_id, "rank": m.rank, "origin": ORIGIN_RETRIEVED} for m in seed_matches],
    }
    return seed_matches + tuple(expanded), trace


def resolve(result: RetrievalResult, conversation: Conversation) -> MemoryMatch:
    """Resolve one retrieved memory's stored provenance against the conversation's messages."""
    messages = {m.message_id: m for m in conversation.messages}
    sources: list[SourceMessage] = []
    unresolved: list[UnresolvedSource] = []
    for s in result.sources:
        message: Message | None = messages.get(s.message_id)
        if result.conversation_id != conversation.conversation_id:
            unresolved.append(UnresolvedSource(s.claim_id, s.message_id, OTHER_CONVERSATION))
        elif message is None:
            unresolved.append(UnresolvedSource(s.claim_id, s.message_id, MESSAGE_NOT_FOUND))
        elif message.position != s.position:
            unresolved.append(UnresolvedSource(s.claim_id, s.message_id, POSITION_MISMATCH))
        else:
            sources.append(SourceMessage(message.message_id, s.claim_id, message.position, message.role,
                                         message.content, message.timestamp))
    if not result.sources:
        unresolved.append(UnresolvedSource(None, None, NO_PROVENANCE))
    return MemoryMatch(result.rank, result.memory_id, result.content, result.score, result.similarity,
                       result.status, tuple(sources), tuple(unresolved))


def select_sources(matches: Iterable[MemoryMatch], budget_chars: int) -> SourceSelection:
    if budget_chars < 0:
        raise ValueError("budget_chars must be >= 0")
    content: dict[str, str] = {}
    first: dict[str, str] = {}
    for m in matches:
        for s in m.sources:
            if s.message_id not in first:
                first[s.message_id] = m.memory_id
                content[s.message_id] = s.content
    unique = tuple(first)
    selected: list[str] = []
    used = 0
    for mid in unique:
        if used + len(content[mid]) > budget_chars:
            break
        selected.append(mid)
        used += len(content[mid])
    return SourceSelection(budget_chars, unique, tuple(selected), unique[len(selected):], used, first)


def source_chunks(selection: SourceSelection, matches: Iterable[MemoryMatch],
                  conversation: Conversation) -> tuple[RetrievedChunk, ...]:
    """One item per selected source; its text is the original message content."""
    messages = {m.message_id: m for m in conversation.messages}
    score = {m.memory_id: m.score for m in matches}
    return tuple(RetrievedChunk(rank, f"{selection.first_memory[mid]}->{mid}", conversation.conversation_id, (mid,),
                                score[selection.first_memory[mid]], messages[mid].content)
                 for rank, mid in enumerate(selection.selected, 1))


def retrieve_source_evidence(
    question: str,
    conversation: Conversation,
    claims: Iterable[Claim],
    relations: Iterable[Relation],
    embedder: Embedder,
    k: int,
    budget_chars: int,
    config: RetrievalConfig | None = None,
    bprime: BPrimeConfig | None = None,
) -> SourceEvidence:
    """Retrieve memories, optionally expand via B′, then budget sources.

    When ``bprime`` is None or disabled, the classifier file is not read and behaviour
    matches the pre-B′ Arm A path (seed matches only). When enabled with ``max_hops=1``
    (default ``--bprime``), only seed neighbours are appended. With ``max_hops=2``
    (``--bprime-two-hop``), hop-1 expansions are also expanded once; depth is fixed.
    """
    consolidator = consolidate(conversation.conversation_id, claims, relations)
    retriever = MemoryRetriever(embedder, config or RetrievalConfig())
    retriever.index_consolidator(consolidator)
    outcome = retriever.search(question, k, conversation.conversation_id)
    seed_matches = tuple(resolve(r, conversation) for r in outcome.results)
    n_seeds = len(seed_matches)

    cfg = bprime or BPrimeConfig(enabled=False)
    if cfg.enabled:
        matches, bprime_trace = expand_matches_with_bprime(seed_matches, consolidator, conversation, cfg)
    else:
        matches = seed_matches
        bprime_trace = empty_trace(enabled=False)

    selection = select_sources(matches, budget_chars)
    bprime_trace = {
        **bprime_trace,
        "final_selected": list(selection.selected),
        "final_dropped": list(selection.dropped),
        "budget_chars": selection.budget_chars,
        "used_chars": selection.used_chars,
    }
    return SourceEvidence(
        question,
        outcome,
        matches,
        selection,
        source_chunks(selection, matches, conversation),
        bprime=bprime_trace,
        n_seed_matches=n_seeds,
    )
