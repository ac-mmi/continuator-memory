"""Query-time relationship expansion and context assembly (phase 5D, experimental).

    question -> existing retrieval (unchanged) -> seed memories
             -> one-hop expansion over the phase 5C relationship graph
             -> phase 5A source windows of every final memory -> ExpandedContext

Nothing here retrieves, ranks by similarity, calls a model or creates memories or edges. The
seeds are an existing retriever's results, kept in their order; expansion only appends.

Expansion is one hop (``max_depth`` must be 1): for each seed, its direct edges in both
directions (``RelationshipGraph.outgoing`` / ``incoming``; the graph stays directed). Policy per
(type, direction from the seed), with its priority (lower first):

  1  UPDATES / SUPERSEDES   incoming   the newer, replacing memory (the current state)
  2  CONTRADICTS            both       the conflicting memory, marked "conflict"
  3  DERIVES                outgoing   each memory supporting the derived seed
  4  EXTENDS                both       the memory adding to / extended by the seed
  5  REFERS_TO              outgoing   the memory the seed refers to (incoming: never)
  6  UPDATES / SUPERSEDES   outgoing   the older version, only as history: when the question
                                       has a past cue or the older memory was not superseded
  7  DERIVES                incoming   the conclusion derived from the seed

Expanded memories are ordered by their best path: seed rank, priority, direction (outgoing
first), the memory's own rank in the full retrieved list (if it was retrieved below the
seeds), conversation position, then memory_id; at most ``max_expanded_memories`` are added.
A memory reached by several paths, or that is already a seed, appears once with every path
recorded. ``max_context_items`` (optional) truncates the final list from the end.

Source windows are deduplicated by window ID in final-memory order; each message carries its
role and the speaker name the conversation established at or before it (phase 5A).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from continuator_memory.models import Message
from continuator_memory.v2.consolidation import _ordinal
from continuator_memory.v2.context import (
    ROLE_LABELS,
    SPEAKER_NAMED,
    ContextualMemory,
    SourceWindow,
    SpeakerRegistry,
    resolve_window,
)
from continuator_memory.v2.memory import (
    LINK_CONTRADICTS,
    LINK_REFERS_TO,
    LINK_SUPERSEDES,
    LINK_UPDATES,
    STATUS_SUPERSEDED,
)
from continuator_memory.v2.memory_retrieval import question_intent
from continuator_memory.v2.relationships import LINK_DERIVES, LINK_EXTENDS, RelationshipGraph

OUTGOING, INCOMING = "outgoing", "incoming"
ORIGIN_RETRIEVED, ORIGIN_EXPANDED = "retrieved", "expanded"
ALWAYS, HISTORY, NEVER = "always", "history", "never"


@dataclass(frozen=True)
class PolicyRule:
    include: str  # ALWAYS, HISTORY (past cue or older not superseded) or NEVER
    priority: int
    role: str  # what the expanded memory is for the seed


POLICY: dict[tuple[str, str], PolicyRule] = {
    (LINK_UPDATES, INCOMING): PolicyRule(ALWAYS, 1, "current_state"),
    (LINK_SUPERSEDES, INCOMING): PolicyRule(ALWAYS, 1, "current_state"),
    (LINK_CONTRADICTS, OUTGOING): PolicyRule(ALWAYS, 2, "conflict"),
    (LINK_CONTRADICTS, INCOMING): PolicyRule(ALWAYS, 2, "conflict"),
    (LINK_DERIVES, OUTGOING): PolicyRule(ALWAYS, 3, "support"),
    (LINK_EXTENDS, OUTGOING): PolicyRule(ALWAYS, 4, "extended"),
    (LINK_EXTENDS, INCOMING): PolicyRule(ALWAYS, 4, "extension"),
    (LINK_REFERS_TO, OUTGOING): PolicyRule(ALWAYS, 5, "reference"),
    (LINK_UPDATES, OUTGOING): PolicyRule(HISTORY, 6, "history"),
    (LINK_SUPERSEDES, OUTGOING): PolicyRule(HISTORY, 6, "history"),
    (LINK_DERIVES, INCOMING): PolicyRule(ALWAYS, 7, "conclusion"),
    (LINK_REFERS_TO, INCOMING): PolicyRule(NEVER, 8, "referrer"),
}
EXPANDABLE_TYPES = (LINK_UPDATES, LINK_SUPERSEDES, LINK_CONTRADICTS, LINK_EXTENDS, LINK_DERIVES, LINK_REFERS_TO)

SKIP_POLICY_NEVER = "policy_never"
SKIP_HISTORY_NOT_REQUESTED = "history_not_requested"
SKIP_TYPE_DISABLED = "type_not_expandable"
SKIP_OVER_BUDGET = "over_budget"
SKIP_NO_KNOWLEDGE = "seed_without_knowledge"
SKIP_MISSING_MEMORY = "missing_memory"
SKIP_CROSS_CONVERSATION = "cross_conversation"

DEFAULT_MAX_EXPANDED = 5  # the Phase 4 answer context holds 5 items; expansion may at most double it


@dataclass(frozen=True)
class ExpansionConfig:
    max_depth: int = 1
    max_expanded_memories: int = DEFAULT_MAX_EXPANDED
    max_context_items: int | None = None  # None: seeds + expansions, no truncation
    types: tuple[str, ...] = EXPANDABLE_TYPES

    def __post_init__(self) -> None:
        if self.max_depth != 1:
            raise ValueError("phase 5D expands exactly one hop (max_depth=1)")
        if self.max_expanded_memories < 0:
            raise ValueError("max_expanded_memories must be >= 0")
        if self.max_context_items is not None and self.max_context_items < 1:
            raise ValueError("max_context_items must be >= 1 or None")
        unknown = set(self.types) - set(EXPANDABLE_TYPES)
        if unknown:
            raise ValueError(f"unknown relationship types {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return {"max_depth": self.max_depth, "max_expanded_memories": self.max_expanded_memories,
                "max_context_items": self.max_context_items, "types": list(self.types)}


class SeedLike(Protocol):
    """An existing retriever's result (``memory_retrieval.RetrievalResult``)."""

    rank: int
    memory_id: str
    conversation_id: str
    score: float


# ---- knowledge ------------------------------------------------------------------------------


def speaker_label(role: str, name: str | None) -> str:
    return f"{name} ({role})" if name else ROLE_LABELS.get(role, role).removeprefix("the ").capitalize()


class _Messages:
    def __init__(self, conversation_id: str, messages: Iterable[Message]):
        self._by_id = {(conversation_id, m.message_id): m for m in messages}

    def get_messages(self, conversation_id: str, message_ids: Sequence[str]) -> list[Message]:
        return [self._by_id[(conversation_id, mid)] for mid in message_ids if (conversation_id, mid) in self._by_id]


@dataclass
class ConversationKnowledge:
    """Everything phase 5A-5C stored for one conversation, as query-time expansion reads it."""

    conversation_id: str
    graph: RelationshipGraph
    memories: dict[str, ContextualMemory]  # memory_id -> contextual memory (with entity references)
    messages: tuple[Message, ...]
    entities: dict[str, Any] = field(default_factory=dict)  # entity_id -> phase 5B Entity

    def __post_init__(self) -> None:
        if self.graph.conversation_id != self.conversation_id or set(self.graph.memories) != set(self.memories):
            raise ValueError(f"{self.conversation_id}: graph and contextual memories do not match")
        if any(m.memory.conversation_id != self.conversation_id for m in self.memories.values()):
            raise ValueError(f"{self.conversation_id}: memory from another conversation")
        self._lookup = _Messages(self.conversation_id, self.messages)
        registry = SpeakerRegistry(self.messages)
        self._speakers = {m.message_id: registry.speaker(m.role, m.position) for m in self.messages}

    def evidence_message(self, message: Message) -> EvidenceMessage:
        sp = self._speakers[message.message_id]
        name = sp.name if sp.status == SPEAKER_NAMED else None
        return EvidenceMessage(message.message_id, message.position, message.role, name,
                               speaker_label(message.role, name), message.content, message.timestamp)

    def source(self, window: SourceWindow) -> tuple[EvidenceMessage, ...]:
        return tuple(self.evidence_message(m) for m in resolve_window(window, self._lookup).messages)


# ---- result ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpansionPath:
    seed_memory_id: str
    seed_rank: int
    relationship_id: str
    type: str
    direction: str  # from the seed: OUTGOING (seed is the edge source) or INCOMING
    role: str
    priority: int
    source_memory_id: str
    target_memory_id: str
    depth: int = 1

    @property
    def edge(self) -> str:
        return f"{self.source_memory_id} --{self.type}--> {self.target_memory_id}"

    def to_dict(self) -> dict[str, Any]:
        return {"seed_memory_id": self.seed_memory_id, "seed_rank": self.seed_rank, "edge": self.edge,
                "relationship_id": self.relationship_id, "type": self.type, "direction": self.direction,
                "role": self.role, "priority": self.priority, "depth": self.depth}


@dataclass(frozen=True)
class ContextItem:
    """One memory of the assembled context."""

    rank: int
    origin: str  # ORIGIN_RETRIEVED (a seed) or ORIGIN_EXPANDED
    depth: int  # 0 for seeds, 1 for expanded memories
    retrieval_rank: int | None
    retrieval_score: float | None
    memory: ContextualMemory
    paths: tuple[ExpansionPath, ...]  # expanded: every path that reached it; seed: paths from other seeds
    window_ids: tuple[str, ...]

    @property
    def conversation_id(self) -> str:
        return self.memory.memory.conversation_id

    @property
    def memory_id(self) -> str:
        return self.memory.memory.memory_id

    @property
    def global_id(self) -> str:
        return self.memory.global_id

    @property
    def source_message_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.message_id for s in self.memory.sources))

    def format(self) -> str:
        m = self.memory
        via = ("retrieved" if self.origin == ORIGIN_RETRIEVED else
               "; ".join(f"{p.role} via {p.edge}" for p in self.paths))
        return (f"[{self.global_id}] {m.contextual_content} (status: {m.memory.status}; asserted by "
                f"{speaker_label(m.speaker.role, m.speaker.name)}; {via})")

    def to_dict(self) -> dict[str, Any]:
        m = self.memory
        return {"rank": self.rank, "global_id": self.global_id, "conversation_id": self.conversation_id,
                "memory_id": self.memory_id, "origin": self.origin, "depth": self.depth,
                "retrieval_rank": self.retrieval_rank,
                "retrieval_score": None if self.retrieval_score is None else round(self.retrieval_score, 6),
                "content": m.memory.content, "contextual_content": m.contextual_content, "status": m.memory.status,
                "speaker": m.speaker.to_dict(), "source_claim_ids": [s.claim_id for s in m.sources],
                "source_message_ids": list(self.source_message_ids), "window_ids": list(self.window_ids),
                "entity_refs": [{"role": r.role, "entity_id": r.entity_id, "name": r.name, "status": r.status}
                                for r in m.entity_refs],
                "paths": [p.to_dict() for p in self.paths]}


@dataclass(frozen=True)
class EvidenceMessage:
    message_id: str
    position: int
    role: str
    speaker_name: str | None
    speaker: str  # "Sam (user)", or "User" / "Assistant" when no name is established
    content: str
    timestamp: str | None

    def to_dict(self) -> dict[str, Any]:
        return {"message_id": self.message_id, "position": self.position, "role": self.role,
                "speaker_name": self.speaker_name, "speaker": self.speaker, "content": self.content,
                "timestamp": self.timestamp}


@dataclass(frozen=True)
class SourceEvidence:
    window: SourceWindow
    messages: tuple[EvidenceMessage, ...]
    memory_ids: tuple[str, ...]  # global IDs of the context memories citing this window

    def format(self) -> str:
        return "\n".join([f"source {self.window.window_id}:",
                          *(f"[{m.message_id}] {m.speaker}: {m.content}" for m in self.messages)])

    def to_dict(self) -> dict[str, Any]:
        return {"window": self.window.to_dict(), "memory_ids": list(self.memory_ids),
                "messages": [m.to_dict() for m in self.messages]}


@dataclass(frozen=True)
class ExpandedContext:
    question: str
    config: ExpansionConfig
    memories: tuple[ContextItem, ...]
    source_windows: tuple[SourceEvidence, ...]
    relationships: tuple[ExpansionPath, ...]  # every traversed edge kept by the policy, in traversal order
    entities: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]

    @property
    def seeds(self) -> tuple[ContextItem, ...]:
        return tuple(i for i in self.memories if i.origin == ORIGIN_RETRIEVED)

    @property
    def expanded(self) -> tuple[ContextItem, ...]:
        return tuple(i for i in self.memories if i.origin == ORIGIN_EXPANDED)

    def format(self) -> str:
        return "\n\n".join(["MEMORIES:", *(i.format() for i in self.memories), "SOURCES:",
                            *(s.format() for s in self.source_windows)])

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "config": self.config.to_dict(),
                "memories": [i.to_dict() for i in self.memories],
                "source_windows": [s.to_dict() for s in self.source_windows],
                "relationships": [p.to_dict() for p in self.relationships],
                "entities": list(self.entities), "diagnostics": self.diagnostics}


# ---- expansion ------------------------------------------------------------------------------


def _memory_key(memory: ContextualMemory) -> tuple[int, int, str]:
    m = memory.memory
    return m.position, _ordinal(m.claim_id), m.memory_id


def expand(question: str, seeds: Sequence[SeedLike], knowledge: Mapping[str, ConversationKnowledge],
           config: ExpansionConfig | None = None, retrieved: Iterable[SeedLike] = ()) -> ExpandedContext:
    """Assemble the context for ``question`` from ``seeds`` (kept, in order) and their one-hop
    neighbours. ``retrieved`` is the full retrieved list, used only to order expanded memories
    that the retriever also ranked below the seeds."""
    config = config or ExpansionConfig()
    intent = question_intent(question)
    seeds = sorted(seeds, key=lambda s: s.rank)
    ranked = {(r.conversation_id, r.memory_id): r for r in retrieved}
    seed_keys = {(s.conversation_id, s.memory_id): s for s in seeds}
    skipped: dict[str, int] = {}
    skipped_paths: list[dict[str, Any]] = []
    violations: list[str] = []
    traversed: list[ExpansionPath] = []
    seed_paths: dict[tuple[str, str], list[ExpansionPath]] = {}
    candidates: dict[tuple[str, str], list[ExpansionPath]] = {}
    edges_examined = 0

    def skip(reason: str, detail: dict[str, Any]) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1
        skipped_paths.append({"reason": reason, **detail})

    valid_seeds = []
    for seed in seeds:
        kb = knowledge.get(seed.conversation_id)
        if kb is None:
            skip(SKIP_NO_KNOWLEDGE, {"seed": seed.memory_id, "conversation_id": seed.conversation_id})
            continue
        if seed.memory_id not in kb.memories:
            skip(SKIP_MISSING_MEMORY, {"seed": seed.memory_id, "conversation_id": seed.conversation_id})
            continue
        valid_seeds.append(seed)
        edges = [(OUTGOING, e) for e in kb.graph.outgoing(seed.memory_id)] + [
            (INCOMING, e) for e in kb.graph.incoming(seed.memory_id)]
        for direction, e in edges:
            edges_examined += 1
            neighbour = e.target_memory_id if direction == OUTGOING else e.source_memory_id
            detail = {"seed": seed.memory_id, "edge": f"{e.source_memory_id} --{e.type}--> {e.target_memory_id}",
                      "direction": direction}
            if e.conversation_id != seed.conversation_id or neighbour not in kb.memories:
                reason = SKIP_CROSS_CONVERSATION if e.conversation_id != seed.conversation_id else SKIP_MISSING_MEMORY
                violations.append(f"{seed.conversation_id}:{detail['edge']}")
                skip(reason, detail)
                continue
            if e.type not in config.types:
                skip(SKIP_TYPE_DISABLED, detail)
                continue
            rule = POLICY[(e.type, direction)]
            if rule.include == NEVER:
                skip(SKIP_POLICY_NEVER, detail)
                continue
            if rule.include == HISTORY and not (intent.past or kb.memories[neighbour].memory.status != STATUS_SUPERSEDED):
                skip(SKIP_HISTORY_NOT_REQUESTED, detail)
                continue
            path = ExpansionPath(seed.memory_id, seed.rank, e.relationship_id, e.type, direction, rule.role,
                                 rule.priority, e.source_memory_id, e.target_memory_id)
            traversed.append(path)
            key = (seed.conversation_id, neighbour)
            (seed_paths if key in seed_keys else candidates).setdefault(key, []).append(path)

    def path_key(key: tuple[str, str], p: ExpansionPath) -> tuple:
        own = ranked.get(key)
        return (p.seed_rank, p.priority, p.direction != OUTGOING, own.rank if own else float("inf"),
                *_memory_key(knowledge[key[0]].memories[key[1]]))

    for key, paths in candidates.items():
        paths.sort(key=lambda p: path_key(key, p))
    order = sorted(candidates, key=lambda k: path_key(k, candidates[k][0]))
    chosen, over = order[:config.max_expanded_memories], order[config.max_expanded_memories:]
    for key in over:
        skip(SKIP_OVER_BUDGET, {"memory": key[1], "conversation_id": key[0], "best_path": candidates[key][0].edge})

    items: list[ContextItem] = []
    for seed in valid_seeds:
        key = (seed.conversation_id, seed.memory_id)
        cm = knowledge[seed.conversation_id].memories[seed.memory_id]
        items.append(ContextItem(0, ORIGIN_RETRIEVED, 0, seed.rank, seed.score, cm,
                                 tuple(sorted(seed_paths.get(key, ()), key=lambda p: path_key(key, p))),
                                 tuple(dict.fromkeys(s.window.window_id for s in cm.sources))))
    for key in chosen:
        cm = knowledge[key[0]].memories[key[1]]
        own = ranked.get(key)
        items.append(ContextItem(0, ORIGIN_EXPANDED, 1, own.rank if own else None, own.score if own else None, cm,
                                 tuple(candidates[key]), tuple(dict.fromkeys(s.window.window_id for s in cm.sources))))
    truncated: list[str] = []
    if config.max_context_items is not None and len(items) > config.max_context_items:
        truncated = [i.global_id for i in items[config.max_context_items:]]
        items = items[:config.max_context_items]
    items = [ContextItem(rank, *(getattr(i, f) for f in ("origin", "depth", "retrieval_rank", "retrieval_score",
                                                         "memory", "paths", "window_ids")))
             for rank, i in enumerate(items, 1)]

    windows: dict[str, tuple[SourceWindow, list[str]]] = {}
    for item in items:
        for s in item.memory.sources:
            windows.setdefault(s.window.window_id, (s.window, []))
            if item.global_id not in windows[s.window.window_id][1]:
                windows[s.window.window_id][1].append(item.global_id)
    sources = tuple(SourceEvidence(w, knowledge[w.conversation_id].source(w), tuple(ids)) for w, ids in windows.values())

    entities: dict[str, dict[str, Any]] = {}
    for item in items:
        for r in item.memory.entity_refs:
            if not r.entity_id:
                continue
            e = knowledge[item.conversation_id].entities.get(r.entity_id)
            entry = entities.setdefault(r.entity_id, {
                "entity_id": r.entity_id, "conversation_id": item.conversation_id,
                "canonical_name": e.canonical_name if e else r.name, "entity_type": e.entity_type if e else None,
                "status": e.status if e else None, "aliases": list(e.aliases) if e else [], "memories": []})
            if item.global_id not in entry["memories"]:
                entry["memories"].append(item.global_id)

    candidate_paths = sum(len(p) for p in candidates.values()) + sum(len(p) for p in seed_paths.values())
    reached = len(candidates) + len(seed_paths)
    diagnostics = {
        "seed_memories": len(valid_seeds), "expanded_memories": sum(i.origin == ORIGIN_EXPANDED for i in items),
        "final_memories": len(items), "source_windows": len(sources),
        "source_messages": len({(s.window.conversation_id, m.message_id) for s in sources for m in s.messages}),
        "relationships_traversed": len(traversed), "edges_examined": edges_examined,
        "candidate_paths": candidate_paths, "duplicate_paths": candidate_paths - reached,
        "paths_to_seeds": sum(len(p) for p in seed_paths.values()),
        "added": [i.global_id for i in items if i.origin == ORIGIN_EXPANDED],
        "max_expanded_memories": config.max_expanded_memories, "max_context_items": config.max_context_items,
        "truncated": bool(truncated), "truncated_memories": truncated,
        "skipped": dict(sorted(skipped.items())), "skipped_paths": skipped_paths,
        "cross_conversation_violations": violations, "question_intent": intent.to_dict(),
    }
    return ExpandedContext(question, config, tuple(items), sources, tuple(traversed),
                           tuple(entities.values()), diagnostics)
