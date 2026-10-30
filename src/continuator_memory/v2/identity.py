"""Conservative entity and slot identity for V2 consolidation (phase 2C).

Identity comes only from what the claims already carry. No fuzzy matching, edit distance,
embeddings or semantic equivalence: "Sam" and "Samuel" are different entities unless a claim
says otherwise, and "team lead" is not "manager".

Signals, strongest first (all optional ``Claim.metadata`` keys except names and entities):
  subject_id    explicit ID of the claim's subject                      -> identity "id:<id>"
  entity_ids    {name: id} for entities named in the claim               -> identity "id:<id>"
  aliases       {alias: canonical name} declared by the claim ("Samuel, who I call Sam")
  subject       explicit subject name (resolved through aliases and IDs)
  entities      exact normalized names (Phase 1 already resolved pronouns into them)
  position      the Phase 2A rule: content opens with "the user"/"the assistant" or one of the
                claim's own entities, not in the possessive

Declarations accumulate per conversation in claim order and apply from that claim onwards
(never retroactively). A name that has been declared an alias of, or given an ID for, more
than one entity is ambiguous and resolves to nothing.

A slot is (subject identity, attribute). The attribute is ``metadata["attribute"]`` if given,
otherwise the claim's ``claim_type`` (Phase 2A), normalized structurally only (Unicode, case,
punctuation, whitespace); with ``claim_type_slots=False`` only an explicit attribute counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from continuator_memory.memory.models import FIRST_PERSON, normalize_value
from continuator_memory.v2.memory import IDENTITY_AMBIGUOUS
from continuator_memory.v2.models import Claim

SPEAKERS = ("user", "assistant")


def entity_key(name: str) -> str:
    """Exact normalized identity of an entity name; first-person references become "user"."""
    key = normalize_value(name)
    if key in FIRST_PERSON:
        return "user"
    if key in ("assistant", "the assistant"):
        return "assistant"
    return key[4:] if key.startswith("the ") else key


def entity_keys(claim: Claim) -> tuple[str, ...]:
    keys: list[str] = []
    for e in claim.entities:
        key = entity_key(e)
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def explicit_subject(claim: Claim) -> str | None:
    value = _text(claim.metadata.get("subject"))
    return entity_key(value) if value else None


def claim_subject(claim: Claim) -> str | None:
    """The claim's subject name if it can be read off deterministically, else None.

    An explicit ``metadata["subject"]`` wins. Otherwise the content must open with "the user",
    "the assistant" or one of the claim's own entities, not in the possessive ("The user's
    sister ..." is about the sister, whose identity is not known)."""
    explicit = explicit_subject(claim)
    if explicit:
        return explicit
    text = normalize_value(claim.content)
    for key in sorted({*SPEAKERS, *entity_keys(claim)}, key=lambda k: (-len(k), k)):
        for prefix in (key, f"the {key}"):
            if text == prefix or (text.startswith(prefix + " ") and not text.startswith(prefix + " s ")):
                return key
    return None


def _mapping(value: Any) -> dict[str, str]:
    """Normalized {name: value} from a metadata mapping; malformed entries are ignored."""
    if not isinstance(value, dict):
        return {}
    out = {}
    for k, v in value.items():
        name, target = _text(k), _text(v)
        if name and target and entity_key(name):
            out[entity_key(name)] = target
    return out


def claim_attribute(claim: Claim, claim_type_slots: bool = True) -> str | None:
    explicit = _text(claim.metadata.get("attribute"))
    if explicit:
        return normalize_value(explicit) or None
    if claim_type_slots and claim.claim_type:
        return normalize_value(claim.claim_type) or None
    return None


@dataclass(frozen=True)
class ClaimIdentity:
    subject: str | None  # resolved identity of the claim's subject
    explicit_subject: str | None  # resolved identity from metadata subject / subject_id only
    attribute: str | None
    entities: tuple[str, ...]  # resolved identities of the claim's entities
    problems: tuple[str, ...] = ()


class IdentityRegistry:
    """Explicit alias and ID declarations seen so far in one conversation."""

    def __init__(self) -> None:
        self._aliases: dict[str, set[str]] = {}  # alias key -> canonical name keys
        self._ids: dict[str, set[str]] = {}  # name key -> explicit IDs

    def register(self, claim: Claim) -> None:
        for alias, canonical in _mapping(claim.metadata.get("aliases")).items():
            canonical_key = entity_key(canonical)
            if canonical_key and canonical_key != alias:
                self._aliases.setdefault(alias, set()).add(canonical_key)
        for name, eid in _mapping(claim.metadata.get("entity_ids")).items():
            self._ids.setdefault(name, set()).add(eid)

    def resolve(self, name: str, claim: Claim) -> tuple[str | None, str | None]:
        """(identity, problem) for a name in ``claim``; the claim's own declarations win."""
        key = entity_key(name)
        if not key or key in SPEAKERS:
            return key or None, None
        local_ids = _mapping(claim.metadata.get("entity_ids"))
        local_aliases = {a: entity_key(c) for a, c in _mapping(claim.metadata.get("aliases")).items()}
        if key in local_ids:
            return f"id:{local_ids[key]}", None
        canonical = local_aliases.get(key)
        if canonical is None:
            declared = self._aliases.get(key, set())
            if len(declared) > 1:
                return None, f"{IDENTITY_AMBIGUOUS}: {key!r} is declared an alias of {sorted(declared)}"
            canonical = next(iter(declared)) if declared else key
        if canonical in local_ids:
            return f"id:{local_ids[canonical]}", None
        ids = self._ids.get(canonical, set())
        if len(ids) > 1:
            return None, f"{IDENTITY_AMBIGUOUS}: {canonical!r} names several entities {sorted(ids)}"
        return (f"id:{next(iter(ids))}" if ids else canonical), None

    def identify(self, claim: Claim, claim_type_slots: bool = True) -> ClaimIdentity:
        """Register the claim's declarations, then resolve its subject, entities and slot."""
        self.register(claim)
        problems: list[str] = []

        def resolve(name: str) -> str | None:
            identity, problem = self.resolve(name, claim)
            if problem and problem not in problems:
                problems.append(problem)
            return identity

        subject_id = _text(claim.metadata.get("subject_id"))
        subject_name = _text(claim.metadata.get("subject"))
        if subject_id:
            explicit = subject = f"id:{subject_id}"
        elif subject_name:
            explicit = subject = resolve(subject_name)
        else:
            explicit = None
            positional = claim_subject(claim)
            subject = resolve(positional) if positional else None
        entities = tuple(dict.fromkeys(i for i in (resolve(e) for e in claim.entities) if i))
        return ClaimIdentity(subject, explicit, claim_attribute(claim, claim_type_slots), entities, tuple(problems))
