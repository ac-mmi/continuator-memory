"""Build the LLM prompt from structured memory plus source evidence."""

from __future__ import annotations

from continuator_memory.generation import build_prompt
from continuator_memory.memory.models import Memory, Relationship, normalize_value
from continuator_memory.memory.query import AUXILIARIES
from continuator_memory.memory.retrieval import CONSTRAINT_PIVOT, MemoryRetrieval

ABSTAIN_ANSWER = "I don't know; that information is not in the conversation."

MEMORY_SYSTEM_PROMPT = f"""\
You answer questions about a past conversation using the MEMORY and the SOURCE EVIDENCE below.

- MEMORY lists facts extracted from the conversation, grouped by the entity they describe.
- A value marked [CURRENT] is the latest valid value. A value marked [SUPERSEDED] was later \
changed or corrected: never present it as the current value. Use superseded values only when \
the question asks about the past, such as an original or previous value.
- A fact belongs only to the entity it is listed under. Do not transfer facts between entities.
- For questions that connect several facts, combine MEMORY entries and RELATIONSHIPS step by step.
- If neither MEMORY nor SOURCE EVIDENCE states the requested fact, answer exactly: \
"{ABSTAIN_ANSWER}"
- Answer in one or two short sentences.
"""


def _sources(ids: tuple[str, ...]) -> str:
    return f"(messages: {', '.join(ids)})"


def _validity(valid_from: str | None, valid_until: str | None) -> str:
    parts = []
    if valid_from:
        parts.append(f"from {valid_from}")
    if valid_until:
        parts.append(f"until {valid_until}")
    return f", valid {' '.join(parts)}" if parts else ""


def _status(m: Memory, superseded_by: dict[str, Memory]) -> str:
    if m.is_current:
        status = "CURRENT"
    else:
        newer = superseded_by.get(m.memory_id)
        status = f"SUPERSEDED by {newer.value}" if newer else "SUPERSEDED"
    return f"[{status}{_validity(m.valid_from, m.valid_until)}]"


def _memory_line(m: Memory, superseded_by: dict[str, Memory]) -> str:
    return f"- {m.predicate}: {m.value} {_status(m, superseded_by)} {_sources(m.source_message_ids)}"


def _constraint_phrase(question: str, marker: str, clause: str) -> str:
    """The constraint as worded in the question: antecedent + relative word + clause."""
    words = normalize_value(question).split()
    tail = [marker] + clause.split()
    start = next((i for i in range(len(words)) if words[i:i + len(tail)] == tail), None)
    if start is None:
        return " ".join(tail)
    head = start
    while head > 0 and words[head - 1] not in AUXILIARIES:
        head -= 1
    return " ".join(words[head:start] + tail)


def _resolution_lines(retrieval: MemoryRetrieval) -> list[str]:
    """Render constraint pivots already established by memory selection."""
    understanding = retrieval.understanding or {}
    shown = {m.memory_id: m for g in retrieval.groups for slot in g.slots for m in slot}
    constraints = {c["text"]: c for c in understanding.get("constraints", [])}
    requested = understanding.get("requested_attribute") or {}
    lines: list[str] = []
    for pivot in understanding.get("pivots", []):
        entity, clause = pivot["entity"], pivot["constraint"]
        matches = [
            f"{m['predicate']}: {m['value']}" for m in understanding.get("constraint_matches", [])
            if m["entity"] == entity and m["constraint"] == clause and m["record_id"] in shown
        ]
        answers = [
            shown[e["record_id"]] for e in retrieval.selection
            if e.get("pivot") == entity and e["reason"] == CONSTRAINT_PIVOT
            and e["record_id"] in shown
        ]
        if not matches:
            continue
        constraint = constraints.get(clause)
        phrase = _constraint_phrase(understanding.get("question", ""), constraint["marker"],
                                    clause) if constraint else clause
        lines.append(f'- "{phrase}" is {entity} ({"; ".join(matches)})')
        for m in answers:
            lines.append(
                f'- requested "{requested.get("text", "")}" of {entity}: '
                f"{m.predicate}: {m.value} {_status(m, retrieval.superseded_by)}"
            )
    return lines


def _relationship_line(r: Relationship) -> str:
    status = "CURRENT" if r.is_current else "SUPERSEDED"
    return (
        f"- {r.subject} -> {r.predicate} -> {r.object} "
        f"[{status}{_validity(r.valid_from, r.valid_until)}] {_sources(r.source_message_ids)}"
    )


def render_memory_context(retrieval: MemoryRetrieval) -> str:
    if not retrieval.groups and not retrieval.relationships:
        return "MEMORY:\n(no stored facts match this question)"
    conversations = {g.entity.conversation_id for g in retrieval.groups} | {
        r.conversation_id for r in retrieval.relationships
    }
    label_conversations = len(conversations) > 1
    lines = ["MEMORY:"]
    for group in retrieval.groups:
        header = group.entity.name
        if label_conversations:
            header += f" (conversation {group.entity.conversation_id})"
        lines.append(header)
        for slot in group.slots:
            lines += [_memory_line(m, retrieval.superseded_by) for m in slot]
    if retrieval.relationships:
        lines += ["", "RELATIONSHIPS:"]
        lines += [_relationship_line(r) for r in retrieval.relationships]
    resolution = _resolution_lines(retrieval)
    if resolution:
        lines += ["", "RESOLUTION:"] + resolution
    return "\n".join(lines)


def build_memory_prompt(question: str, retrieval: MemoryRetrieval) -> str:
    evidence = build_prompt(question, retrieval.chunks).replace(
        "Conversation excerpts:", "SOURCE EVIDENCE (conversation excerpts):", 1
    )
    return f"{render_memory_context(retrieval)}\n\n{evidence}"
