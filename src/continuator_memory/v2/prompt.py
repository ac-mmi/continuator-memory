"""Atomizer prompt, few-shot examples and output schema.

The examples use invented text (not benchmark data). Together they cover: two independent
facts in one sentence, a secondary proposition, an explicit cause, an explicit update of a
prior claim, an explicit negation, pronoun resolution, a temporal statement, two facts joined
by "and" that must not become causal, a condition and a message with no durable assertion.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from continuator_memory.models import Message
from continuator_memory.v2.models import MODEL_STATUSES, POLARITIES, RELATION_TYPES, Claim, Provenance
from continuator_memory.v2.windowing import ContextWindow

SYSTEM_PROMPT = """\
You are preserving semantic information from conversation, not summarizing it.

Extract every distinct proposition explicitly asserted by each TARGET message. Preserve \
secondary propositions, modifiers, quantities, conditions, temporal information, negations \
and explicit causal, reference and update relationships. Do not decide what is important: \
keep everything the message asserts.

Rules:
1. SOURCE EVIDENCE: a claim may only assert information supported by its own TARGET message. \
CONTEXT messages and PRIOR STATE are context, not evidence, and are never atomized.
2. IDENTITY CONTEXT: prior state and context may only resolve who or what a pronoun or \
reference means ("She moved to Bangalore." with prior "Leah ..." -> "Leah moved to \
Bangalore."). They never add a fact the target message does not state. If a referent is \
unclear, keep the original wording.
3. Do not infer facts merely because they are plausible. Do not add explanations or helpful \
inferred facts.
4. Each claim is one self-contained sentence in "content". Write "the user" for the user's \
I/me/my and "the assistant" for the assistant's. Do not merge independent propositions into \
one claim, and do not split a cause or condition away from its meaning without a relation.
5. TEMPORAL: copy temporal expressions exactly as stated into temporal.expression. Fill \
temporal.resolved_date (YYYY-MM-DD) only when the message timestamp makes it certain; never \
invent dates. A later statement does not supersede an earlier one merely by coming later.
6. Negation: keep it in the content and set polarity "negative"; otherwise "positive".
7. RELATIONS: only when the source explicitly expresses them or grammar clearly establishes \
them. CAUSED_BY goes from effect to cause and needs explicit causal wording (because, so, \
since, due to, as a result). UPDATES / CONTRADICTS go from the newer claim to the older claim \
it changes or corrects, only when the source states a change or correction. SUPPORTS goes \
from evidence to the claim it supports; REFERS_TO from a claim to the claim it refers back to. \
Never infer causality from two statements occurring together (joined by "and" or adjacent), \
and never relate claims only because they are similar. Relations may point to PRIOR STATE \
claim IDs. "evidence" is the exact source words expressing the relation.
8. Claim IDs: <source_message_id>:c1, <source_message_id>:c2, ... in order within each message.
9. entities: names or things the claim is about, as written (may be empty). qualifiers: \
modifiers copied from the source such as quantities, durations, frequency, conditions or \
hedges (may be empty). claim_type: leave "" unless clearly useful.
10. Return exactly one entry per TARGET message. If a message contains no durable assertion \
(greeting, acknowledgement, question, filler), return status "no_durable_assertion" with no \
claims. Never return status "claims" with an empty list.

Output: {"messages": [{"source_message_id", "status", "claims", "explicit_relations"}]}.
"""


def _message_line(m: Message) -> str:
    ts = f" @ {m.timestamp}" if m.timestamp else ""
    return f"[{m.message_id}] {m.role}{ts}: {m.content}"


def _prior_line(c: Claim) -> str:
    entities = f" (entities: {', '.join(c.entities)})" if c.entities else ""
    return f"- [{c.claim_id}] {c.content}{entities}"


def build_prompt(window: ContextWindow, prior: Sequence[Claim] = ()) -> str:
    prior_text = "\n".join(_prior_line(c) for c in prior) or "(none)"
    context = "\n".join(_message_line(m) for m in window.context) or "(none)"
    targets = "\n".join(_message_line(m) for m in window.targets)
    return (f"PRIOR STATE (identity/reference context only, NOT evidence):\n{prior_text}\n\n"
            f"CONTEXT MESSAGES (do not atomize):\n{context}\n\n"
            f"TARGET MESSAGES (atomize every one):\n{targets}\n\nJSON:")


def output_schema(window: ContextWindow) -> dict[str, Any]:
    text = {"type": "string"}
    ids = {"type": "string", "enum": list(window.target_ids)}
    claim = {"type": "object", "required": ["claim_id", "content", "entities", "qualifiers", "temporal",
                                            "polarity", "claim_type", "source_message_id"],
             "properties": {"claim_id": text, "content": text,
                            "entities": {"type": "array", "items": text},
                            "qualifiers": {"type": "array", "items": text},
                            "temporal": {"type": "object", "required": ["expression", "resolved_date"],
                                         "properties": {"expression": text, "resolved_date": text}},
                            "polarity": {"type": "string", "enum": list(POLARITIES)},
                            "claim_type": text, "source_message_id": ids}}
    relation = {"type": "object", "required": ["from_claim_id", "to_claim_id", "type", "source_message_id", "evidence"],
                "properties": {"from_claim_id": text, "to_claim_id": text,
                               "type": {"type": "string", "enum": list(RELATION_TYPES)},
                               "source_message_id": ids, "evidence": text}}
    entry = {"type": "object", "required": ["source_message_id", "status", "claims", "explicit_relations"],
             "properties": {"source_message_id": ids, "status": {"type": "string", "enum": list(MODEL_STATUSES)},
                            "claims": {"type": "array", "items": claim},
                            "explicit_relations": {"type": "array", "items": relation}}}
    return {"type": "object", "required": ["messages"],
            "properties": {"messages": {"type": "array", "items": entry}}}


# ---- examples ---------------------------------------------------------------


def _claim(cid: str, content: str, entities=(), qualifiers=(), temporal: str = "", polarity: str = "positive") -> dict:
    return {"claim_id": cid, "content": content, "entities": list(entities), "qualifiers": list(qualifiers),
            "temporal": {"expression": temporal, "resolved_date": ""}, "polarity": polarity, "claim_type": "",
            "source_message_id": cid.rsplit(":c", 1)[0]}


def _rel(src: str, dst: str, rtype: str, evidence: str) -> dict:
    return {"from_claim_id": src, "to_claim_id": dst, "type": rtype,
            "source_message_id": src.rsplit(":c", 1)[0], "evidence": evidence}


def _window(name: str, context: Sequence[tuple[str, str, str]], targets: Sequence[tuple[str, str, str]]) -> ContextWindow:
    msgs = [Message(mid, name, i, role, content) for i, (mid, role, content) in enumerate([*context, *targets])]
    return ContextWindow(name, 0, tuple(msgs[:len(context)]), tuple(msgs[len(context):]))


def _prior(name: str, cid: str, content: str, entities: Sequence[str], position: int) -> Claim:
    mid = cid.rsplit(":c", 1)[0]
    return Claim(cid, content, Provenance(name, mid, position, "user"), tuple(entities))


EXAMPLE_CASES: tuple[tuple[ContextWindow, tuple[Claim, ...], dict], ...] = (
    # Two independent facts, a secondary proposition (the brother relation) and a pronoun;
    # the facts are joined by "and" and must not become causal.
    (_window("ex-a", [], [("a1", "user", "My brother Karim is a pilot and he lives in Oslo.")]), (), {"messages": [
        {"source_message_id": "a1", "status": "claims", "claims": [
            _claim("a1:c1", "Karim is the user's brother.", ["Karim"]),
            _claim("a1:c2", "Karim is a pilot.", ["Karim"]),
            _claim("a1:c3", "Karim lives in Oslo.", ["Karim", "Oslo"])],
         "explicit_relations": []}]}),
    # Explicit causes, a temporal expression and a negation.
    (_window("ex-b", [], [("b1", "user", "I sold my car last spring because parking got too expensive, "
                                         "so I don't drive to work anymore.")]), (), {"messages": [
        {"source_message_id": "b1", "status": "claims", "claims": [
            _claim("b1:c1", "The user sold their car last spring.", temporal="last spring"),
            _claim("b1:c2", "Parking got too expensive for the user.", qualifiers=["too expensive"]),
            _claim("b1:c3", "The user doesn't drive to work anymore.", polarity="negative")],
         "explicit_relations": [_rel("b1:c1", "b1:c2", "CAUSED_BY", "because"),
                                _rel("b1:c3", "b1:c1", "CAUSED_BY", "so")]}]}),
    # Explicit update of a prior claim; "it" resolved from prior state; a temporal qualifier.
    (_window("ex-c", [("c1", "assistant", "Noted.")],
             [("c2", "user", "Change of plan: it meets on Fridays now, starting next week.")]),
     (_prior("ex-c", "c0:c1", "The user's book club meets on Wednesdays.", ["book club"], -1),),
     {"messages": [
         {"source_message_id": "c2", "status": "claims", "claims": [
             _claim("c2:c1", "The user's book club meets on Fridays now, starting next week.", ["book club"],
                    temporal="starting next week")],
          "explicit_relations": [_rel("c2:c1", "c0:c1", "UPDATES", "Change of plan")]}]}),
    # A question with no durable assertion, then a condition kept inside its claim.
    (_window("ex-d", [], [("d1", "assistant", "Sounds good! Where will the picnic be?"),
                          ("d2", "user", "In Hillside Park, but if it rains, the picnic moves indoors to the library.")]),
     (), {"messages": [
         {"source_message_id": "d1", "status": "no_durable_assertion", "claims": [], "explicit_relations": []},
         {"source_message_id": "d2", "status": "claims", "claims": [
             _claim("d2:c1", "The picnic will be in Hillside Park.", ["picnic", "Hillside Park"]),
             _claim("d2:c2", "If it rains, the picnic moves indoors to the library.", ["picnic", "library"],
                    qualifiers=["if it rains"])],
          "explicit_relations": []}]}),
)

EXAMPLES: tuple[tuple[str, str], ...] = tuple(
    (build_prompt(window, prior), json.dumps(output)) for window, prior, output in EXAMPLE_CASES)
