"""Memory formation: extract -> validate -> consolidate -> persist, one message at a time."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from continuator_memory.memory.consolidation import Decision, decide_memory, decide_relationship
from continuator_memory.memory.extraction import (
    ExtractedMemory,
    ExtractedRelationship,
    ExtractionError,
    ExtractionRequestError,
    ExtractionResult,
    KnownFact,
)
from continuator_memory.memory.models import Action, normalize_predicate, normalize_value
from continuator_memory.memory.store import MemoryStore
from continuator_memory.memory.text import content_tokens
from continuator_memory.models import Conversation, Message


class Extractor(Protocol):
    model_name: str

    def extract(
        self, message: Message, recent: list[Message], known_facts: list[KnownFact]
    ) -> tuple[ExtractionResult, str]: ...


@dataclass
class FormationReport:
    conversation_id: str
    messages: int = 0
    extraction_status: Counter = field(default_factory=Counter)
    extracted_memories: int = 0
    extracted_relationships: int = 0
    rejected_items: int = 0
    grounding: Counter = field(default_factory=Counter)
    actions: Counter = field(default_factory=Counter)
    extraction_ms: float = 0.0
    consolidation_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "messages": self.messages,
            "extraction_status": dict(sorted(self.extraction_status.items())),
            "extracted_memories": self.extracted_memories,
            "extracted_relationships": self.extracted_relationships,
            "rejected_items": self.rejected_items,
            "grounding": dict(sorted(self.grounding.items())),
            "actions": dict(sorted(self.actions.items())),
            "extraction_ms": round(self.extraction_ms, 3),
            "consolidation_ms": round(self.consolidation_ms, 3),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryFormation:
    def __init__(
        self,
        memory: MemoryStore,
        extractor: Extractor,
        recent_window: int = 3,
        max_known_facts: int = 40,
        clock: Callable[[], str] = _utc_now,
    ):
        self.memory = memory
        self.extractor = extractor
        self.recent_window = recent_window
        self.max_known_facts = max_known_facts
        self.clock = clock

    def known_facts(self, conversation_id: str) -> list[KnownFact]:
        """Current memories and relationships, most recent last, as extractor context."""
        rows = [
            (m.observed_position, KnownFact(m.subject, m.predicate, m.value, m.source_message_ids))
            for m in self.memory.memories(conversation_id, current_only=True)
        ] + [
            (r.observed_position, KnownFact(r.subject, r.predicate, r.object, r.source_message_ids))
            for r in self.memory.relationships(conversation_id, current_only=True)
        ]
        rows.sort(key=lambda x: x[0])
        return [fact for _, fact in rows[-self.max_known_facts :]]

    def form(self, conversation: Conversation) -> FormationReport:
        report = FormationReport(conversation.conversation_id)
        messages = conversation.messages
        by_id = {m.message_id: m for m in messages}
        for i, message in enumerate(messages):
            report.messages += 1
            recent = list(messages[max(0, i - self.recent_window) : i])
            known = self.known_facts(conversation.conversation_id)
            t0 = time.perf_counter()
            try:
                result, raw = self.extractor.extract(message, recent, known)
                status, error = "ok", None
            except ExtractionRequestError as exc:
                result, raw, status, error = ExtractionResult(), None, "error", str(exc)
            except ExtractionError as exc:
                result, raw, status, error = ExtractionResult(), None, "invalid", str(exc)
            latency = (time.perf_counter() - t0) * 1000
            report.extraction_ms += latency
            report.extraction_status[status] += 1
            report.extracted_memories += len(result.memories)
            report.extracted_relationships += len(result.relationships)
            report.rejected_items += len(result.rejected)
            report.grounding.update(g["grounding"] for g in result.grounding)

            t1 = time.perf_counter()
            with self.memory.conn:
                self.memory.log_extraction(
                    conversation.conversation_id, message.message_id, self.extractor.model_name,
                    status, len(result.memories), len(result.relationships), result.rejected,
                    raw, error, round(latency, 3), result.grounding if status == "ok" else None,
                )
                for cand in result.memories:
                    source = message
                    if cand.evidence_message_ids:
                        source = by_id.get(cand.evidence_message_ids[0], message)
                    target, action = self._apply_fact(conversation.conversation_id, source, cand)
                    report.actions[f"{target}:{action.value}"] += 1
                for rel in result.relationships:
                    action = self._apply_relationship(conversation.conversation_id, message, rel)
                    report.actions[f"relationship:{action.value}"] += 1
            report.consolidation_ms += (time.perf_counter() - t1) * 1000
        return report

    # ---- applying decisions ----------------------------------------------

    def _apply_fact(
        self, conv: str, message: Message, cand: ExtractedMemory
    ) -> tuple[str, Action]:
        """Route a subject/predicate/value fact: a value naming another known entity
        makes it a relationship; anything else is an attribute memory."""
        subject = self.memory.resolve_entity(conv, cand.subject)
        target = self.memory.find_entity(conv, cand.value)
        if target is not None and target.entity_id != subject.entity_id:
            rel = ExtractedRelationship(
                cand.subject, cand.predicate, cand.value, cand.kind, cand.previous_value,
                cand.valid_from,
            )
            return "relationship", self._apply_relationship(conv, message, rel)
        return "memory", self._apply_memory(conv, message, cand)

    def _update_target(self, conv: str, entity, predicate: str, cand: ExtractedMemory):
        """If a change names a previous value that is current for exactly one other entity
        under the same predicate, that entity's slot is the one being changed."""
        if not cand.previous_value or cand.kind == "reference":
            return entity
        if any(m.is_current for m in self.memory.memories(entity_id=entity.entity_id,
                                                          predicate=predicate)):
            return entity
        prev = normalize_value(cand.previous_value)
        matches = [
            m for m in self.memory.memories(conv, predicate=predicate, current_only=True)
            if m.entity_id != entity.entity_id and normalize_value(m.value) == prev
        ]
        if len(matches) == 1:
            return self.memory.get_entity(matches[0].entity_id)
        return entity

    def _apply_memory(self, conv: str, message: Message, cand: ExtractedMemory) -> Action:
        ms = self.memory
        predicate = normalize_predicate(cand.predicate)
        entity = self._update_target(conv, ms.resolve_entity(conv, cand.subject), predicate, cand)
        slot = ms.memories(entity_id=entity.entity_id, predicate=predicate)
        decision = decide_memory(cand, slot, message.message_id, message.content)
        valid_from = cand.valid_from or message.timestamp
        now = self.clock()
        record_id = decision.existing_id
        superseded = list(decision.supersede_ids)

        if decision.action is Action.UPDATE:
            ms.add_memory_source(decision.existing_id, conv, message.message_id)
        elif decision.action in (Action.CREATE, Action.SUPERSEDE):
            if decision.create_previous:
                prev_id = ms.insert_memory(
                    conv, entity, predicate, decision.create_previous, "state",
                    message.message_id, message.position, now, None, is_current=False,
                )
                ms.close_memory(prev_id, valid_from)
                superseded.append(prev_id)
            latest = None
            if superseded:
                by_pos = sorted(
                    (m for m in ms.memories(entity_id=entity.entity_id, predicate=predicate)
                     if m.memory_id in superseded),
                    key=lambda m: (m.observed_position, m.memory_id),
                )
                latest = by_pos[-1].memory_id if by_pos else None
            record_id = ms.insert_memory(
                conv, entity, predicate, cand.value, cand.kind, message.message_id,
                message.position, now, valid_from, is_current=True, supersedes_memory_id=latest,
            )
            for old in decision.supersede_ids:
                ms.close_memory(old, valid_from)
        self._log(conv, message, "memory", decision, record_id, superseded)
        return decision.action

    def _apply_relationship(
        self, conv: str, message: Message, cand: ExtractedRelationship
    ) -> Action:
        ms = self.memory
        subject = ms.resolve_entity(conv, cand.subject)
        obj = ms.resolve_entity(conv, cand.object)
        predicate = normalize_predicate(cand.predicate)
        previous = ms.find_entity(conv, cand.previous) if cand.previous else None
        slot = [
            r
            for r in ms.relationships(conv, predicate=predicate)
            if {r.subject_entity_id, r.object_entity_id} & {subject.entity_id, obj.entity_id}
        ]
        decision = decide_relationship(
            subject.entity_id, obj.entity_id, cand.kind,
            previous.entity_id if previous else None, slot, message.message_id,
            previous_stated=bool(
                cand.previous and content_tokens(cand.previous)
                and content_tokens(cand.previous) <= content_tokens(message.content)
            ),
        )
        valid_from = cand.valid_from or message.timestamp
        record_id = decision.existing_id

        if decision.action is Action.UPDATE:
            ms.add_relationship_source(decision.existing_id, conv, message.message_id)
        elif decision.action in (Action.CREATE, Action.SUPERSEDE):
            latest = None
            if decision.supersede_ids:
                olds = sorted(
                    (r for r in slot if r.relationship_id in decision.supersede_ids),
                    key=lambda r: (r.observed_position, r.relationship_id),
                )
                latest = olds[-1].relationship_id
            record_id = ms.insert_relationship(
                conv, subject, predicate, obj, cand.kind, message.message_id, message.position,
                self.clock(), valid_from, is_current=True, supersedes_relationship_id=latest,
            )
            for old in decision.supersede_ids:
                ms.close_relationship(old, valid_from)
        self._log(conv, message, "relationship", decision, record_id, list(decision.supersede_ids))
        return decision.action

    def _log(
        self, conv: str, message: Message, target: str, decision: Decision,
        record_id: str | None, superseded: list[str],
    ) -> None:
        self.memory.log_consolidation(
            conv, message.message_id, target, decision.action.value, record_id, superseded,
            decision.reason,
        )
