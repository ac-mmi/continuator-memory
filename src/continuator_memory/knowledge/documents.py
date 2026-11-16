"""The document contract: the prompt, local message labels, and structural parsing of the Markdown.

Parsing is structural only. It finds headings, bullets and citation markers, and checks that a citation names a
message shown to the model. It does not compare a claim's words with its source, so a valid citation shows where
a claim says it came from, not that the source supports it. Every claim therefore reports
``semantic_support: not_verified``, and only manual review can establish precision.

No claim is dropped here. A structural problem marks the claim ``needs_review`` and keeps it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from continuator_memory.models import Message

PROMPT_VERSION = "kd-prompt-v1"
FORMAT_VERSION = "kd-markdown-v1"
NOTHING_DURABLE = "NOTHING_DURABLE"

SECTIONS = ("People and identities", "Projects and activities", "Facts and preferences", "Decisions and reasons",
            "Plans and timelines", "Changes and corrections", "Relationships", "Unresolved questions")

REVIEW_EVIDENCE_CITED = "evidence_cited"
REVIEW_NEEDED = "needs_review"
SEMANTIC_NOT_VERIFIED = "not_verified"

MISSING_CITATION = "MISSING_CITATION"
UNKNOWN_CITATION = "UNKNOWN_CITATION"
CONTEXT_ONLY_CITATION = "CONTEXT_ONLY_CITATION"
OUTSIDE_SECTION = "OUTSIDE_SECTION"
UNKNOWN_SECTION = "UNKNOWN_SECTION"
NOT_A_BULLET = "NOT_A_BULLET"
TRUNCATED_DOCUMENT = "TRUNCATED_DOCUMENT"

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_BULLET = re.compile(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+(.*)$")
_CONTINUATION = re.compile(r"^\s{2,}\S")
_CITATION = re.compile(r"\s*\[\s*([MC]\d+(?:\s*,\s*[MC]\d+)*)\s*\]")
_FENCE = re.compile(r"^\s*```[\w-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)

SYSTEM_PROMPT = """\
You write a knowledge document from one part of a conversation, so it can be found and understood later without \
the conversation. Read it the way a careful person would.

M1, M2, ... are the messages to document. C1, C2 are the messages just before them, shown only so you can \
understand the M messages, such as the question that a "Yes" answers. Document what the M messages say. Do not \
document something only because a C message says it.

Write Markdown under these headings, using only the ones you have something to put under, in this order. Never \
add a heading just to fill it in:
## People and identities
## Projects and activities
## Facts and preferences
## Decisions and reasons
## Plans and timelines
## Changes and corrections
## Relationships
## Unresolved questions

Rules:
- Each statement is one bullet: a complete sentence that makes sense alone, naming people and things rather than \
using I, you, it or they. Do not assume anyone's gender. Keep separate facts in separate bullets.
- End every bullet with the messages it comes from, such as [M2] or [M2, C1]. Never cite a message you were not \
shown. Cite a C message only together with the M message that relies on it.
- Write only what the conversation says. Do not add outside knowledge, or details, dates or reasons nobody stated.
- Cover every durable statement in the M messages, including short answers such as "Yes" and facts stated in \
passing. Say what was decided and the stated reason.
- Keep certainty as stated. A possibility stays a possibility ("Possibly: ...", "might"). A question goes under \
Unresolved questions unless a message answers it; a question does not establish a fact. A suggestion is not a \
decision. A plan is not a completed event. Keep negation.
- Attribute what one person says about another to the person who said it.
- Keep dates and time words as stated. A message's date is when it was said, not when a fact applies.
- If someone corrects or changes something, state both, under Changes and corrections ("now X; earlier said Y").
- You see only part of the conversation, so never call a fact outdated or the latest.

If the M messages contain nothing durable, reply with exactly: NOTHING_DURABLE

Example.
Messages:
[C1] user @ 2024-03-05: Ines: Rafi, are you still at Harbour Print?
[M1] assistant @ 2024-03-05: Rafi: Yes, but I might leave in June. I chose Tallow over Ember because it is cheaper.
[M2] user @ 2024-03-05: Ines: Nice! Actually it's Thursday I swim, not Tuesday. Do you still cycle?

Document:
## Facts and preferences
- Rafi is at Harbour Print. [M1, C1]
- Ines swims on Thursday. [M2]

## Decisions and reasons
- Rafi chose Tallow over Ember because Tallow is cheaper. [M1]

## Plans and timelines
- Possibly: Rafi might leave in June. [M1]

## Changes and corrections
- Ines now says she swims on Thursday; she earlier said Tuesday. [M2]

## Unresolved questions
- Ines asked whether Rafi still cycles. [M2]
"""


def labels(targets: Sequence[Message], context: Sequence[Message]) -> dict[str, Message]:
    out = {f"M{i}": m for i, m in enumerate(targets, 1)}
    out.update({f"C{i}": m for i, m in enumerate(context, 1)})
    return out


def _line(label: str, message: Message) -> str:
    stamp = f" @ {message.timestamp}" if message.timestamp else ""
    return f"[{label}] {message.role}{stamp}: {message.content}"


def build_prompt(targets: Sequence[Message], context: Sequence[Message] = ()) -> str:
    named = labels(targets, context)
    lines = [_line(k, m) for k, m in named.items() if k.startswith("C")] + [_line(k, m) for k, m in named.items() if k.startswith("M")]
    return "Messages:\n" + "\n".join(lines) + "\n\nDocument:"


@dataclass(frozen=True)
class DocumentClaim:
    claim_index: int
    section: str | None
    text: str
    citations: tuple[str, ...]
    message_ids: tuple[str, ...]
    review_reasons: tuple[str, ...] = ()

    @property
    def review_status(self) -> str:
        return REVIEW_NEEDED if self.review_reasons else REVIEW_EVIDENCE_CITED

    def to_dict(self) -> dict[str, Any]:
        return {"claim_index": self.claim_index, "section": self.section, "text": self.text, "citations": list(self.citations),
                "message_ids": list(self.message_ids), "review_status": self.review_status,
                "review_reasons": list(self.review_reasons), "semantic_support": SEMANTIC_NOT_VERIFIED}


@dataclass
class ParsedDocument:
    claims: list[DocumentClaim] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    nothing_durable: bool = False
    content: str = ""
    uncited_target_ids: tuple[str, ...] = ()


def _claim(index: int, section: str | None, known_section: bool, raw: str, named: dict[str, Message], is_bullet: bool) -> DocumentClaim:
    found = [label.strip() for group in _CITATION.findall(raw) for label in group.split(",")]
    text = " ".join(_CITATION.sub("", raw).split())
    cited = tuple(dict.fromkeys(found))
    known = [c for c in cited if c in named]
    reasons: list[str] = []
    if section is None:
        reasons.append(OUTSIDE_SECTION)
    elif not known_section:
        reasons.append(UNKNOWN_SECTION)
    if not is_bullet:
        reasons.append(NOT_A_BULLET)
    if not cited:
        reasons.append(MISSING_CITATION)
    if len(known) != len(cited):
        reasons.append(UNKNOWN_CITATION)
    if known and not any(c.startswith("M") for c in known):
        reasons.append(CONTEXT_ONLY_CITATION)
    return DocumentClaim(index, section, text, cited, tuple(dict.fromkeys(named[c].message_id for c in known)), tuple(reasons))


def parse_document(raw: str, targets: Sequence[Message], context: Sequence[Message] = ()) -> ParsedDocument:
    """Every bullet or stray paragraph becomes a claim, kept whatever its problems."""
    named = labels(targets, context)
    content = raw.strip()
    fence = _FENCE.match(content)
    parsed = ParsedDocument()
    if fence:
        content = fence.group(1).strip()
        parsed.issues.append("WRAPPED_IN_CODE_FENCE")
    parsed.content = content
    if content == NOTHING_DURABLE:
        parsed.nothing_durable = True
        parsed.uncited_target_ids = tuple(m.message_id for m in targets)
        return parsed

    section: str | None = None
    pending: tuple[str | None, bool, list[str], bool] | None = None

    def flush() -> None:
        nonlocal pending
        if pending:
            owner, known_section, parts, is_bullet = pending
            parsed.claims.append(_claim(len(parsed.claims), owner, known_section, " ".join(parts), named, is_bullet))
        pending = None

    known_section = False
    for line in content.splitlines():
        if not line.strip():
            flush()
            continue
        heading = _HEADING.match(line)
        if heading:
            flush()
            section = heading.group(1).strip()
            known_section = section in SECTIONS
            continue
        bullet = _BULLET.match(line)
        if bullet:
            flush()
            pending = (section, known_section, [bullet.group(1)], True)
        elif pending and (_CONTINUATION.match(line) or not pending[3]):
            pending[2].append(line.strip())
        else:
            flush()
            pending = (section, known_section, [line.strip()], False)
    flush()
    cited_ids = {mid for c in parsed.claims for mid in c.message_ids}
    parsed.uncited_target_ids = tuple(m.message_id for m in targets if m.message_id not in cited_ids)
    return parsed
