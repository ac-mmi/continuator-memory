"""V1 memory-extraction protocol: durable, independently useful memories per TARGET message.

Same windows, validation and consolidation as the semantic atomizer (``continuator_memory.v2.prompt``);
only the instruction changes from "atomize every proposition" to "extract durable memories", and each
memory carries a ``memory_type`` (stored in ``Claim.metadata``, never in ``claim_type``). Memories keep
the speaker's own wording because ``validation.classify_support`` rejects claims whose words are not in
the source. That check is lexical; the durability and speaker rules below are instructions to the model,
not guarantees enforced by code. Scope is English-only. The examples are fictional, written for this
module, and share no people, places, organizations or events with the LoCoMo or synthetic evaluation data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from continuator_memory.models import Message
from continuator_memory.v2.models import MEMORY_TYPES, MODEL_STATUSES, POLARITIES, RELATION_TYPES, Claim
from continuator_memory.v2.prompt import _message_line, _prior_line
from continuator_memory.v2.windowing import ContextWindow

MEMORY_PROTOCOL_VERSION = "memories-v1"

MEMORY_SYSTEM_PROMPT = """\
You extract long-term memories from a conversation. A memory is a durable, independently useful \
statement worth recalling in a later conversation: a person's preferences, personal facts, goals, \
projects, events, decisions, relationships or procedures.

Rules:
1. SOURCE EVIDENCE: every memory must be stated by its own TARGET message. CONTEXT messages and \
PRIOR STATE only resolve who or what a reference means; they are never evidence.
2. Extract only what is explicitly stated. Do not infer, guess, summarize the conversation or add \
plausible facts.
3. Keep the speaker's own words. Each memory is one self-contained sentence reusing the message's \
wording. Only replace I/me/my/we with the speaker: use the name shown at the start of the message \
("Tavi: ..."), otherwise "the user" or "the assistant" by role; replace other pronouns with the \
names they refer to. Do not paraphrase.
4. Durable only. Skip greetings, thanks, small talk, questions, reactions, compliments, filler and \
generic advice. If a TARGET message states nothing durable, return status "no_durable_assertion" \
with no claims. Never return status "claims" with an empty list. One message may give several memories.
5. One fact per memory. Split independent facts; keep a cause or condition with the fact it qualifies.
6. Speaker attribution: suggestions, explanations or speculation by the assistant are never facts \
about the user. Record an assistant statement only if it is a durable commitment or decision, \
attributed to the assistant.
7. memory_type: preference, personal_fact, goal, project, event, decision, relationship or procedural.
8. TEMPORAL: copy temporal expressions exactly into temporal.expression; fill temporal.resolved_date \
(YYYY-MM-DD) only when the message timestamp makes it certain; never invent dates. A later statement \
does not replace an earlier one merely by coming later; keep past states ("used to ...") as stated.
9. Negation: keep it in the content and set polarity "negative"; otherwise "positive".
10. RELATIONS: UPDATES / CONTRADICTS only when the TARGET message itself states a change or a \
correction, from the newer memory to the older memory's claim ID (PRIOR STATE or earlier). CAUSED_BY \
only with explicit causal words. "evidence" is the exact source words expressing the relation.
11. Claim IDs: <source_message_id>:c1, :c2, ... in order. entities: names or things as written. \
qualifiers: modifiers copied from the source (quantities, durations, frequency, conditions, hedges).

Output: {"messages": [{"source_message_id", "status", "claims", "explicit_relations"}]}, exactly one \
entry per TARGET message.
"""


def build_memory_prompt(window: ContextWindow, prior: Sequence[Claim] = ()) -> str:
    prior_text = "\n".join(_prior_line(c) for c in prior) or "(none)"
    context = "\n".join(_message_line(m) for m in window.context) or "(none)"
    targets = "\n".join(_message_line(m) for m in window.targets)
    return (f"PRIOR STATE (identity/reference context only, NOT evidence):\n{prior_text}\n\n"
            f"CONTEXT MESSAGES (do not extract):\n{context}\n\n"
            f"TARGET MESSAGES (extract durable memories from each):\n{targets}\n\nJSON:")


def memory_output_schema(window: ContextWindow) -> dict[str, Any]:
    text = {"type": "string"}
    ids = {"type": "string", "enum": list(window.target_ids)}
    claim = {"type": "object",
             "required": ["claim_id", "content", "memory_type", "entities", "qualifiers", "temporal", "polarity",
                          "source_message_id"],
             "properties": {"claim_id": text, "content": text,
                            "memory_type": {"type": "string", "enum": list(MEMORY_TYPES)},
                            "entities": {"type": "array", "items": text},
                            "qualifiers": {"type": "array", "items": text},
                            "temporal": {"type": "object", "required": ["expression", "resolved_date"],
                                         "properties": {"expression": text, "resolved_date": text}},
                            "polarity": {"type": "string", "enum": list(POLARITIES)},
                            "source_message_id": ids}}
    relation = {"type": "object", "required": ["from_claim_id", "to_claim_id", "type", "source_message_id", "evidence"],
                "properties": {"from_claim_id": text, "to_claim_id": text,
                               "type": {"type": "string", "enum": list(RELATION_TYPES)},
                               "source_message_id": ids, "evidence": text}}
    entry = {"type": "object", "required": ["source_message_id", "status", "claims", "explicit_relations"],
             "properties": {"source_message_id": ids, "status": {"type": "string", "enum": list(MODEL_STATUSES)},
                            "claims": {"type": "array", "items": claim},
                            "explicit_relations": {"type": "array", "items": relation}}}
    return {"type": "object", "required": ["messages"], "properties": {"messages": {"type": "array", "items": entry}}}


# ---- examples (invented) ------------------------------------------------------------------------

def _m(mid: str, role: str, content: str, position: int) -> Message:
    return Message(mid, "example", position, role, content, None)


def _mem(cid: str, content: str, memory_type: str, entities=(), qualifiers=(), temporal: str = "",
         polarity: str = "positive") -> dict[str, Any]:
    return {"claim_id": cid, "content": content, "memory_type": memory_type, "entities": list(entities),
            "qualifiers": list(qualifiers), "temporal": {"expression": temporal, "resolved_date": ""},
            "polarity": polarity, "source_message_id": cid.split(":c")[0]}


def _entry(mid: str, claims=(), relations=()) -> dict[str, Any]:
    return {"source_message_id": mid, "status": "claims" if claims else "no_durable_assertion",
            "claims": list(claims), "explicit_relations": list(relations)}


_EX1 = ContextWindow("example", 0, (), (
    _m("x1", "user", "Hi! I'm Priya. I moved to Lyon in March and I work as a nurse at Saint-Luc hospital. "
                     "I hate early shifts.", 0),
    _m("x2", "assistant", "Welcome, Priya! You could try the Croix-Rousse market on Sundays.", 1),
    _m("x3", "user", "Thanks! Actually I'm not a nurse anymore, I switched to teaching last week.", 2),
))
_EX1_OUT = {"messages": [
    _entry("x1", [
        _mem("x1:c1", "The user is Priya.", "personal_fact", ["Priya"]),
        _mem("x1:c2", "Priya moved to Lyon in March.", "event", ["Priya", "Lyon"], temporal="in March"),
        _mem("x1:c3", "Priya works as a nurse at Saint-Luc hospital.", "personal_fact", ["Priya", "Saint-Luc hospital"]),
        _mem("x1:c4", "Priya hates early shifts.", "preference", ["Priya"]),
    ]),
    _entry("x2"),
    _entry("x3", [
        _mem("x3:c1", "Priya is not a nurse anymore.", "personal_fact", ["Priya"], polarity="negative"),
        _mem("x3:c2", "Priya switched to teaching last week.", "event", ["Priya"], temporal="last week"),
    ], [{"from_claim_id": "x3:c1", "to_claim_id": "x1:c3", "type": "UPDATES", "source_message_id": "x3",
         "evidence": "Actually I'm not a nurse anymore"}]),
]}

_EX2 = ContextWindow("example", 1, (_m("y1", "user", "Tavi: Hey Odalys, how's the week going?", 0),), (
    _m("y2", "assistant", "Odalys: Busy! I used to drive trams in Ghent. I opened a bicycle repair workshop last "
                          "month and I love it.", 1),
    _m("y3", "user", "Tavi: That's lovely! I'm learning Welsh for a move to Cardiff next year.", 2),
    _m("y4", "assistant", "Odalys: You could try the evening classes at the town library.", 3),
))
_EX2_OUT = {"messages": [
    _entry("y2", [
        _mem("y2:c1", "Odalys used to drive trams in Ghent.", "personal_fact", ["Odalys", "Ghent"]),
        _mem("y2:c2", "Odalys opened a bicycle repair workshop last month.", "project", ["Odalys"], temporal="last month"),
        _mem("y2:c3", "Odalys loves the bicycle repair workshop.", "preference", ["Odalys"]),
    ]),
    _entry("y3", [
        _mem("y3:c1", "Tavi is learning Welsh for a move to Cardiff next year.", "goal", ["Tavi", "Cardiff"],
             temporal="next year"),
    ]),
    _entry("y4"),
]}

MEMORY_EXAMPLES: tuple[tuple[str, str], ...] = (
    (build_memory_prompt(_EX1), json.dumps(_EX1_OUT)),
    (build_memory_prompt(_EX2), json.dumps(_EX2_OUT)),
)
