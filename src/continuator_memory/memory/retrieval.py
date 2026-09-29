"""Memory-aware retrieval.

question
  -> Phase 1 semantic chunk retrieval (unchanged)
  -> score memories/relationships: relevance = lexical overlap (IDF-weighted) +
     entity match, multiplied up when a source message was semantically
     retrieved; superseded records are down-weighted unless the question has
     historical intent. Records with zero relevance are never seeds.
  -> seed entities from the best records, expanded one hop along relationships
     and along memory values that name another entity
  -> context: entity -> predicate slots (current value + history), relationships
  -> supporting chunks: the semantic ranking is the baseline. Its top
     PROTECTED_SEMANTIC chunks keep their ranks; then up to MAX_MEMORY_ADDITIONS
     chunks are inserted for context records with positive relevance (best record
     first) whose source messages are not yet covered; the remaining semantic
     chunks follow in their original order.
  -> MEMORY block: a precision filter over the candidate context, applied after
     fusion so the supporting chunks do not depend on it. Evidence is the overlap
     between the question and a record's predicate/value, excluding the record's
     own entity name. A record enters with at least MIN_DIRECT_EVIDENCE evidence
     tokens, or with one evidence token when its entity is named in the question.
     When the question names entities, records from conversations without any of
     them are excluded. Memories of entities linked by a selected relationship
     enter as relationship support; a selected slot keeps its whole state history.
     Constraint pivots extend this: when the question identifies an unnamed entity
     by a condition ("the person who handles the budget"), a candidate record whose
     predicate carries the condition's head identifies a pivot entity (only if the
     best match is unique). The pivot contributes the constraint slot plus slots
     that provide the requested attribute, by token overlap or by the question's
     stranded preposition preceding the value in a source message ("from Taipei").
     Without a usable constraint, selection is exactly the rule above.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from continuator_memory.memory.models import (
    Entity,
    Memory,
    Relationship,
    normalize_name,
    normalize_value,
)
from continuator_memory.memory.query import LexicalQuestionAnalyzer, QuestionAnalysis, QuestionAnalyzer
from continuator_memory.memory.store import MemoryStore
from continuator_memory.memory.text import content_tokens, predicate_tokens
from continuator_memory.models import Chunk, RetrievedChunk
from continuator_memory.retrieval import Retriever
from continuator_memory.store import Store

PROTECTED_SEMANTIC = 3
MAX_MEMORY_ADDITIONS = 2
SUPERSEDED_WEIGHT = 0.5
ENTITY_MATCH_BONUS = 1.0
SEMANTIC_SUPPORT_WEIGHT = 2.0
MIN_DIRECT_EVIDENCE = 2

DIRECT_QUESTION_MATCH = "DIRECT_QUESTION_MATCH"
ENTITY_PLUS_ATTRIBUTE_MATCH = "ENTITY_PLUS_ATTRIBUTE_MATCH"
RELATIONSHIP_SUPPORT = "RELATIONSHIP_SUPPORT"
STATE_HISTORY = "STATE_HISTORY"
CONSTRAINT_MATCH = "CONSTRAINT_MATCH"
CONSTRAINT_PIVOT = "CONSTRAINT_PIVOT"
ENTITY_NAMED = "ENTITY_NAMED"


@dataclass
class EntityGroup:
    entity: Entity
    slots: list[list[Memory]]  # each slot: one predicate, current values first


@dataclass
class MemoryRetrieval:
    chunks: list[RetrievedChunk]
    analysis: QuestionAnalysis
    groups: list[EntityGroup] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    superseded_by: dict[str, Memory] = field(default_factory=dict)
    candidate_groups: list[EntityGroup] = field(default_factory=list)
    candidate_relationships: list[Relationship] = field(default_factory=list)
    selection: list[dict[str, Any]] = field(default_factory=list)
    understanding: dict[str, Any] = field(default_factory=dict)

    def trace(self) -> dict[str, Any]:
        memories = [m for g in self.groups for slot in g.slots for m in slot]
        candidates = [m.memory_id for g in self.candidate_groups for slot in g.slots for m in slot]
        sources = sorted(
            {(m.conversation_id, mid) for m in memories for mid in m.source_message_ids}
            | {(r.conversation_id, mid) for r in self.relationships for mid in r.source_message_ids}
        )
        return {
            "temporal_scope": self.analysis.temporal_scope,
            "matched_entities": [e.name for e in self.analysis.entities],
            "entities": [g.entity.name for g in self.groups],
            "memory_ids": [m.memory_id for m in memories],
            "relationship_ids": [r.relationship_id for r in self.relationships],
            "n_memories": len(memories),
            "n_current": sum(m.is_current for m in memories),
            "n_superseded": sum(not m.is_current for m in memories),
            "n_relationships": len(self.relationships),
            "source_messages": [list(s) for s in sources],
            "candidate_memory_ids": candidates,
            "candidate_relationship_ids": [
                r.relationship_id for r in self.candidate_relationships
            ],
            "n_candidates": len(candidates),
            "context_selection": self.selection,
            "question_understanding": self.understanding,
        }


def _entry(record_id: str, entity: str, predicate: str, value: str, is_current: bool,
           score: float, reason: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "entity": entity,
        "predicate": predicate,
        "value": value,
        "is_current": is_current,
        "score": round(score, 4),
        "reason": reason,
        "evidence": evidence,
    }


class MemoryRetriever:
    def __init__(
        self,
        store: Store,
        memory: MemoryStore,
        semantic: Retriever,
        analyzer: QuestionAnalyzer | None = None,
        semantic_depth: int = 10,
        top_records: int = 8,
        max_entities: int = 8,
        max_slots_per_entity: int = 8,
        max_relationships: int = 12,
    ):
        self.store = store
        self.memory = memory
        self.semantic = semantic
        self.analyzer = analyzer or LexicalQuestionAnalyzer()
        self.semantic_depth = semantic_depth
        self.top_records = top_records
        self.max_entities = max_entities
        self.max_slots_per_entity = max_slots_per_entity
        self.max_relationships = max_relationships

    def _chunks_by_message(self, conversation_id: str | None) -> dict[tuple[str, str], list[Chunk]]:
        index: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
        for chunk in self.store.get_chunks(conversation_id=conversation_id):
            for mid in chunk.source_message_ids:
                index[(chunk.conversation_id, mid)].append(chunk)
        return index

    def retrieve(self, question: str, k: int, conversation_id: str | None = None) -> MemoryRetrieval:
        semantic = self.semantic.retrieve(question, max(k, self.semantic_depth), conversation_id)
        memories = self.memory.memories(conversation_id)
        relationships = self.memory.relationships(conversation_id)
        entities = self.memory.entities(conversation_id)
        analysis = self.analyzer.analyze(
            question, entities, {m.predicate for m in memories}, {r.predicate for r in relationships}
        )
        if not memories and not relationships:
            return MemoryRetrieval(chunks=semantic[:k], analysis=analysis)

        scores = self._score(analysis, semantic, memories, relationships)
        ent_by_id = {e.entity_id: e for e in entities}
        seeds = self._seed_entities(analysis, scores, memories, relationships, ent_by_id)
        groups = self._groups(seeds, memories, scores, ent_by_id)
        selected_ids = {g.entity.entity_id for g in groups}
        rels = sorted(
            (r for r in relationships
             if r.subject_entity_id in selected_ids or r.object_entity_id in selected_ids),
            key=lambda r: (-scores[r.relationship_id], r.relationship_id),
        )[: self.max_relationships]

        superseded_by = {
            m.supersedes_memory_id: m for m in memories if m.supersedes_memory_id is not None
        }
        chunks = self._fuse(semantic, groups, rels, scores, k, conversation_id)
        context_groups, context_rels, selection, understanding = self._select_context(
            analysis, groups, rels, scores, ent_by_id
        )
        return MemoryRetrieval(chunks, analysis, context_groups, context_rels, superseded_by,
                               groups, rels, selection, understanding)

    # ---- scoring ----------------------------------------------------------

    def _score(
        self,
        analysis: QuestionAnalysis,
        semantic: list[RetrievedChunk],
        memories: list[Memory],
        relationships: list[Relationship],
    ) -> dict[str, float]:
        message_rank: dict[tuple[str, str], int] = {}
        for chunk in semantic:
            for mid in chunk.source_message_ids:
                message_rank.setdefault((chunk.conversation_id, mid), chunk.rank)

        docs: list[tuple[str, set[str], set[str], str, tuple[str, ...], bool]] = []
        for m in memories:
            tokens = content_tokens(m.subject) | predicate_tokens(m.predicate) | content_tokens(m.value)
            docs.append((m.memory_id, tokens, {m.entity_id}, m.conversation_id,
                         m.source_message_ids, m.is_current))
        for r in relationships:
            tokens = content_tokens(r.subject) | predicate_tokens(r.predicate) | content_tokens(r.object)
            docs.append((r.relationship_id, tokens, {r.subject_entity_id, r.object_entity_id},
                         r.conversation_id, r.source_message_ids, r.is_current))

        df: dict[str, int] = defaultdict(int)
        for _, tokens, *_ in docs:
            for t in tokens:
                df[t] += 1
        n = len(docs)
        matched = {e.entity_id for e in analysis.entities}

        scores: dict[str, float] = {}
        for record_id, tokens, endpoints, conv, sources, is_current in docs:
            lexical = sum(math.log(1 + n / df[t]) for t in analysis.tokens & tokens)
            ranks = [message_rank[(conv, mid)] for mid in sources if (conv, mid) in message_rank]
            support = 1.0 / min(ranks) if ranks else 0.0
            relevance = lexical + (ENTITY_MATCH_BONUS if endpoints & matched else 0.0)
            score = relevance * (1.0 + SEMANTIC_SUPPORT_WEIGHT * support)
            if not is_current and analysis.temporal_scope == "current":
                score *= SUPERSEDED_WEIGHT
            scores[record_id] = score
        return scores

    def _seed_entities(
        self,
        analysis: QuestionAnalysis,
        scores: dict[str, float],
        memories: list[Memory],
        relationships: list[Relationship],
        ent_by_id: dict[str, Entity],
    ) -> list[str]:
        records: list[tuple[float, str, list[str], Memory | None]] = []
        for m in memories:
            records.append((scores[m.memory_id], m.memory_id, [m.entity_id], m))
        for r in relationships:
            records.append((scores[r.relationship_id], r.relationship_id,
                            [r.subject_entity_id, r.object_entity_id], None))
        records.sort(key=lambda x: (-x[0], x[1]))

        by_name: dict[tuple[str, str], str] = {
            (e.conversation_id, e.normalized_name): e.entity_id for e in ent_by_id.values()
        }
        seeds: list[str] = [e.entity_id for e in analysis.entities]
        for score, _, endpoints, mem in records[: self.top_records]:
            if score <= 0:
                break
            for eid in endpoints:
                if eid not in seeds:
                    seeds.append(eid)
            if mem is not None:
                linked = by_name.get((mem.conversation_id, normalize_name(mem.value)))
                if linked and linked not in seeds:
                    seeds.append(linked)

        expanded = list(seeds)
        for eid in seeds:
            for rel, other in self.memory.neighbors(eid, current_only=True):
                if other not in expanded:
                    expanded.append(other)
        return expanded[: self.max_entities]

    def _groups(
        self,
        entity_ids: list[str],
        memories: list[Memory],
        scores: dict[str, float],
        ent_by_id: dict[str, Entity],
    ) -> list[EntityGroup]:
        by_slot: dict[tuple[str, str], list[Memory]] = defaultdict(list)
        for m in memories:
            by_slot[(m.entity_id, m.predicate)].append(m)
        groups = []
        for eid in entity_ids:
            slots = [
                sorted(ms, key=lambda m: (not m.is_current, -m.observed_position, m.memory_id))
                for (e, _), ms in by_slot.items()
                if e == eid
            ]
            if not slots:
                continue
            slots.sort(key=lambda s: (-max(scores[m.memory_id] for m in s), s[0].memory_id))
            groups.append(EntityGroup(ent_by_id[eid], slots[: self.max_slots_per_entity]))
        return groups

    # ---- MEMORY block selection ------------------------------------------

    def _select_context(
        self,
        analysis: QuestionAnalysis,
        groups: list[EntityGroup],
        relationships: list[Relationship],
        scores: dict[str, float],
        ent_by_id: dict[str, Entity],
    ) -> tuple[list[EntityGroup], list[Relationship], list[dict[str, Any]], dict[str, Any]]:
        matched = {e.entity_id for e in analysis.entities}
        conversations = {e.conversation_id for e in analysis.entities}

        def judge(endpoints: set[str], conversation_id: str, tokens: set[str],
                  own_names: set[str]) -> tuple[str, list[str]] | None:
            if conversations and conversation_id not in conversations:
                return None
            evidence = sorted((tokens & analysis.tokens) - own_names)
            if len(evidence) >= MIN_DIRECT_EVIDENCE:
                return DIRECT_QUESTION_MATCH, evidence
            if evidence and endpoints & matched:
                return ENTITY_PLUS_ATTRIBUTE_MATCH, evidence
            return None

        def name_tokens(eid: str, *names: str) -> set[str]:
            entity = ent_by_id.get(eid)
            out = content_tokens(entity.name) if entity else set()
            for name in names:
                out |= content_tokens(name)
            return out

        context_rels: list[Relationship] = []
        rel_entries: list[dict[str, Any]] = []
        for r in relationships:
            verdict = judge(
                {r.subject_entity_id, r.object_entity_id}, r.conversation_id,
                content_tokens(r.subject) | predicate_tokens(r.predicate) | content_tokens(r.object),
                name_tokens(r.subject_entity_id, r.subject) | name_tokens(r.object_entity_id, r.object),
            )
            if verdict:
                context_rels.append(r)
                rel_entries.append(_entry(r.relationship_id, r.subject, r.predicate, r.object,
                                          r.is_current, scores[r.relationship_id], *verdict))
        linked = {eid for r in context_rels for eid in (r.subject_entity_id, r.object_entity_id)}

        context_groups: list[EntityGroup] = []
        entries: list[dict[str, Any]] = []
        for group in groups:
            eid = group.entity.entity_id
            kept: list[list[Memory]] = []
            for slot in group.slots:
                verdicts = [
                    judge({m.entity_id}, m.conversation_id,
                          predicate_tokens(m.predicate) | content_tokens(m.value),
                          name_tokens(m.entity_id, m.subject))
                    for m in slot
                ]
                if any(verdicts):
                    fallback = (STATE_HISTORY, [])
                elif eid in linked:
                    fallback = (RELATIONSHIP_SUPPORT, [])
                else:
                    continue
                kept.append(slot)
                for m, verdict in zip(slot, verdicts):
                    entries.append(_entry(m.memory_id, group.entity.name, m.predicate, m.value,
                                          m.is_current, scores[m.memory_id],
                                          *(verdict or fallback)))
            if kept:
                context_groups.append(EntityGroup(group.entity, kept))

        understanding = analysis.to_dict()
        understanding["entity_roles"] = {e.name: ENTITY_NAMED for e in analysis.targets}
        understanding["constraint_matches"] = []
        understanding["pivots"] = []
        for constraint in analysis.constraints:
            pivot = self._constraint_pivot(constraint, groups, conversations, name_tokens,
                                           understanding)
            if pivot is None:
                continue
            understanding["entity_roles"].setdefault(pivot.entity.name, CONSTRAINT_PIVOT)
            target = next((g for g in context_groups if g.entity.entity_id == pivot.entity.entity_id),
                          None)
            present = {(s[0].entity_id, s[0].predicate) for g in context_groups for s in g.slots}
            added: list[str] = []
            for slot, reason, evidence in self._pivot_slots(pivot, constraint, analysis,
                                                            name_tokens):
                if (slot[0].entity_id, slot[0].predicate) in present:
                    continue
                if target is None:
                    target = EntityGroup(pivot.entity, [])
                    context_groups.append(target)
                target.slots.append(slot)
                for m in slot:
                    own = reason if reason == CONSTRAINT_MATCH or m.memory_id in evidence \
                        else STATE_HISTORY
                    entries.append({**_entry(m.memory_id, pivot.entity.name, m.predicate, m.value,
                                             m.is_current, scores[m.memory_id], own,
                                             evidence.get(m.memory_id, [])),
                                    "pivot": pivot.entity.name})
                    added.append(m.memory_id)
            understanding["pivots"].append({"entity": pivot.entity.name, "constraint": constraint.text,
                                            "added_memory_ids": added})
        return context_groups, context_rels, entries + rel_entries, understanding

    def _constraint_pivot(self, constraint, groups, conversations, name_tokens,
                          understanding) -> EntityGroup | None:
        """The unique candidate entity whose record carries the constraint's relation."""
        matches: list[tuple[int, Memory, EntityGroup]] = []
        for group in groups:
            for slot in group.slots:
                for m in slot:
                    if conversations and m.conversation_id not in conversations:
                        continue
                    own = name_tokens(m.entity_id, m.subject)
                    if constraint.head not in predicate_tokens(m.predicate) - own:
                        continue
                    terms = (predicate_tokens(m.predicate) | content_tokens(m.value)) - own
                    matches.append((len(terms & constraint.terms), m, group))
        if not matches:
            return None
        best = max(n for n, _, _ in matches)
        top = [(m, g) for n, m, g in matches if n == best]
        pivots = {g.entity.entity_id for _, g in top}
        understanding["constraint_matches"] += [
            {"record_id": m.memory_id, "entity": g.entity.name, "predicate": m.predicate,
             "value": m.value, "overlap": best, "constraint": constraint.text}
            for m, g in top
        ]
        if len(pivots) != 1:
            return None
        return top[0][1]

    def _pivot_slots(self, pivot: EntityGroup, constraint, analysis: QuestionAnalysis,
                     name_tokens) -> list[tuple[list[Memory], str, dict[str, list[str]]]]:
        """Constraint slot(s) plus slots providing the requested attribute."""
        requested = analysis.requested_attribute
        out = []
        for slot in pivot.slots:
            if any(constraint.head in predicate_tokens(m.predicate) for m in slot):
                out.append((slot, CONSTRAINT_MATCH, {}))
                continue
            if requested is None:
                continue
            evidence: dict[str, list[str]] = {}
            for m in slot:
                own = name_tokens(m.entity_id, m.subject)
                overlap = (predicate_tokens(m.predicate) | content_tokens(m.value)) - own
                if overlap & requested.tokens:
                    evidence[m.memory_id] = sorted(overlap & requested.tokens)
                elif requested.frame and self._framed(m, requested.frame):
                    evidence[m.memory_id] = [f"frame:{requested.frame} {m.value}"]
            if evidence:
                out.append((slot, CONSTRAINT_PIVOT, evidence))
        return out

    def _framed(self, memory: Memory, frame: str) -> bool:
        """A source message says '<frame> <value>' (e.g. 'from Taipei')."""
        phrase = [frame] + normalize_value(memory.value).split()
        for message in self.store.get_messages(memory.conversation_id, memory.source_message_ids):
            words = normalize_value(message.content).split()
            if any(words[i:i + len(phrase)] == phrase for i in range(len(words))):
                return True
        return False

    def _fuse(
        self,
        semantic: list[RetrievedChunk],
        groups: list[EntityGroup],
        relationships: list[Relationship],
        scores: dict[str, float],
        k: int,
        conversation_id: str | None,
    ) -> list[RetrievedChunk]:
        """Semantic ranking with memory additions; independent of ``k`` apart from the cut."""
        records: list[tuple[float, str, str, tuple[str, ...]]] = [
            (scores[m.memory_id], m.memory_id, m.conversation_id, m.source_message_ids)
            for g in groups for slot in g.slots for m in slot
        ] + [
            (scores[r.relationship_id], r.relationship_id, r.conversation_id, r.source_message_ids)
            for r in relationships
        ]
        records = sorted((r for r in records if r[0] > 0), key=lambda x: (-x[0], x[1]))

        head = semantic[:PROTECTED_SEMANTIC]
        taken = {c.chunk_id for c in head}
        covered = {(c.conversation_id, mid) for c in head for mid in c.source_message_ids}
        semantic_rank = {c.chunk_id: c.rank for c in semantic}
        semantic_score = {c.chunk_id: c.score for c in semantic}
        index = self._chunks_by_message(conversation_id)
        additions: list[Chunk] = []
        for _, _, conv, sources in records:
            if len(additions) >= MAX_MEMORY_ADDITIONS:
                break
            missing = {mid for mid in sources if (conv, mid) not in covered}
            candidates = {
                c.chunk_id: c for mid in missing for c in index.get((conv, mid), [])
                if c.chunk_id not in taken
            }
            if not candidates:
                continue
            best = min(candidates.values(), key=lambda c: (
                -len(missing & set(c.source_message_ids)),
                semantic_rank.get(c.chunk_id, math.inf),
                c.chunk_id,
            ))
            additions.append(best)
            taken.add(best.chunk_id)
            covered |= {(best.conversation_id, mid) for mid in best.source_message_ids}

        order = [(c.chunk_id, c.conversation_id, c.source_message_ids, c.text) for c in head]
        order += [(c.chunk_id, c.conversation_id, c.source_message_ids, c.text) for c in additions]
        order += [(c.chunk_id, c.conversation_id, c.source_message_ids, c.text)
                  for c in semantic[PROTECTED_SEMANTIC:] if c.chunk_id not in taken]
        return [
            RetrievedChunk(rank, cid, conv, sources, semantic_score.get(cid, 0.0), text)
            for rank, (cid, conv, sources, text) in enumerate(order[:k], start=1)
        ]
