"""Memory data model: entities, attribute memories, relationships."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

ASSERTION_KINDS = ("state", "update", "correction", "reference")
CHANGE_KINDS = frozenset({"update", "correction"})
FIRST_PERSON = frozenset({"i", "me", "my", "myself", "user", "the user", "speaker"})


class Action(str, Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"
    KEEP = "KEEP"
    IGNORE = "IGNORE"


@dataclass(frozen=True)
class Entity:
    entity_id: str
    conversation_id: str
    name: str
    normalized_name: str


@dataclass(frozen=True)
class Memory:
    """An attribute of an entity: ``subject.predicate = value``."""

    memory_id: str
    conversation_id: str
    entity_id: str
    subject: str
    predicate: str
    value: str
    kind: str
    source_message_ids: tuple[str, ...]
    observed_position: int
    created_at: str
    valid_from: str | None
    valid_until: str | None
    is_current: bool
    supersedes_memory_id: str | None


@dataclass(frozen=True)
class Relationship:
    """A directed edge ``subject --predicate--> object`` between two entities."""

    relationship_id: str
    conversation_id: str
    subject_entity_id: str
    subject: str
    predicate: str
    object_entity_id: str
    object: str
    kind: str
    source_message_ids: tuple[str, ...]
    observed_position: int
    created_at: str
    valid_from: str | None
    valid_until: str | None
    is_current: bool
    supersedes_relationship_id: str | None


# ---- normalization ----------------------------------------------------------


def normalize_value(text: str) -> str:
    """Comparison key for values and names: casefold, strip accents/punctuation/ordinals."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", stripped)
    stripped = re.sub(r"(\d),(\d{3})\b", r"\1\2", stripped)
    return " ".join(re.sub(r"[^0-9a-z]+", " ", stripped).split())


def normalize_name(name: str) -> str:
    key = normalize_value(name)
    if key in FIRST_PERSON:
        return "user"
    if key.startswith("the "):
        key = key[4:]
    return key


def normalize_predicate(predicate: str) -> str:
    key = normalize_value(predicate).replace(" ", "_")
    return key[:64]
