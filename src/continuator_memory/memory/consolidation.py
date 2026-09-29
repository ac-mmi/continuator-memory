"""Consolidation: decide how a candidate fact changes the memory state.

Pure functions over existing records, so every state transition can be tested
and explained without an LLM or a database.

Rules (memories; a *slot* is one entity + predicate):

- ``reference`` assertions never change state                       -> IGNORE
- same value as a current memory, message already recorded          -> KEEP
- same value as a current memory, new message                       -> UPDATE (add evidence)
- ``state`` that names a current value of the slot as its previous value is treated as
  an ``update`` (the explicit previous value is stronger evidence than the kind label)
- ``update``/``correction`` back to a value the slot already held    -> IGNORE unless the
  message states the value it replaces (mentions of old values are usually references)
- ``update``/``correction`` with a different value                  -> SUPERSEDE the slot's
  current values (only the one matching ``previous_value`` if named); if nothing is
  current but the message names the previous value, that value is recorded as history
- ``update``/``correction`` with nothing to replace                  -> CREATE
- ``state`` repeating a value that was already superseded           -> IGNORE (no resurrection
  without change language)
- ``state`` with a new value                                        -> CREATE; existing current
  values are kept, because a plain statement is not evidence of change

Relationships use the same rules; their slot is "same predicate, sharing the
subject or the object".
"""

from __future__ import annotations

from dataclasses import dataclass

from continuator_memory.memory.extraction import ExtractedMemory
from continuator_memory.memory.text import content_tokens
from continuator_memory.memory.models import (
    CHANGE_KINDS,
    Action,
    Memory,
    Relationship,
    normalize_value,
)


@dataclass(frozen=True)
class Decision:
    action: Action
    existing_id: str | None = None
    supersede_ids: tuple[str, ...] = ()
    create_previous: str | None = None
    reason: str = ""


def _stated_in(phrase: str | None, message_text: str) -> bool:
    tokens = content_tokens(phrase or "")
    return bool(tokens) and tokens <= content_tokens(message_text)


def decide_memory(
    candidate: ExtractedMemory, slot: list[Memory], message_id: str, message_text: str = ""
) -> Decision:
    """``slot``: every memory (current and historical) with the candidate's entity and predicate."""
    if candidate.kind == "reference":
        return Decision(Action.IGNORE, reason="reference, not an assertion")

    value = normalize_value(candidate.value)
    current = [m for m in slot if m.is_current]
    same = [m for m in current if normalize_value(m.value) == value]
    if same:
        existing = same[0]
        if message_id in existing.source_message_ids:
            return Decision(Action.KEEP, existing.memory_id, reason="already recorded")
        return Decision(Action.UPDATE, existing.memory_id, reason="restates current value")

    kind = candidate.kind
    if kind == "state" and candidate.previous_value:
        prev = normalize_value(candidate.previous_value)
        if any(normalize_value(m.value) == prev for m in current):
            kind = "update"

    if kind in CHANGE_KINDS:
        reverts = any(normalize_value(m.value) == value for m in slot if not m.is_current)
        if reverts and not _stated_in(candidate.previous_value, message_text):
            return Decision(
                Action.IGNORE,
                reason="mentions a superseded value; reversion needs the replaced value stated",
            )
        targets = current
        if candidate.previous_value:
            prev = normalize_value(candidate.previous_value)
            named = [m for m in current if normalize_value(m.value) == prev]
            if named:
                targets = named
            elif not current and not any(normalize_value(m.value) == prev for m in slot):
                return Decision(
                    Action.SUPERSEDE,
                    create_previous=candidate.previous_value,
                    reason=f"{kind} of a previously unrecorded value",
                )
        if targets:
            return Decision(
                Action.SUPERSEDE,
                supersede_ids=tuple(m.memory_id for m in targets),
                reason=f"{kind} replaces current value",
            )
        return Decision(Action.CREATE, reason=f"{kind} with no prior value")

    if any(normalize_value(m.value) == value for m in slot):
        return Decision(Action.IGNORE, reason="restates a superseded value without change")
    if current:
        return Decision(Action.CREATE, reason="additional value; no change asserted")
    return Decision(Action.CREATE, reason="new fact")


def decide_relationship(
    subject_id: str,
    object_id: str,
    kind: str,
    previous_entity_id: str | None,
    slot: list[Relationship],
    message_id: str,
    previous_stated: bool = False,
) -> Decision:
    """``slot``: relationships with the same predicate that share the subject or the object."""
    if kind == "reference":
        return Decision(Action.IGNORE, reason="reference, not an assertion")

    def identical(r: Relationship) -> bool:
        return r.subject_entity_id == subject_id and r.object_entity_id == object_id

    current = [r for r in slot if r.is_current]
    same = [r for r in current if identical(r)]
    if same:
        existing = same[0]
        if message_id in existing.source_message_ids:
            return Decision(Action.KEEP, existing.relationship_id, reason="already recorded")
        return Decision(Action.UPDATE, existing.relationship_id, reason="restates relationship")

    if kind in CHANGE_KINDS:
        if any(identical(r) for r in slot if not r.is_current) and not previous_stated:
            return Decision(
                Action.IGNORE,
                reason="mentions a superseded relationship; reversion needs the replaced "
                "value stated",
            )
        targets = current
        if previous_entity_id is not None:
            named = [
                r
                for r in current
                if previous_entity_id in (r.subject_entity_id, r.object_entity_id)
            ]
            if named:
                targets = named
        if targets:
            return Decision(
                Action.SUPERSEDE,
                supersede_ids=tuple(r.relationship_id for r in targets),
                reason=f"{kind} replaces current relationship",
            )
        return Decision(Action.CREATE, reason=f"{kind} with no prior relationship")

    if any(identical(r) for r in slot):
        return Decision(Action.IGNORE, reason="restates a superseded relationship without change")
    return Decision(Action.CREATE, reason="new relationship")
