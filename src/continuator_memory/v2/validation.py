"""Deterministic validation of atomizer output.

Nothing the model returns is stored without passing ``validate_atomization``. Problems are
recorded as ``ValidationIssue``s: REJECTED items are dropped, FLAGGED items are kept (with the
offending annotation removed where applicable).

Source support. Every claim is classified by comparing its content tokens with its source
message:
  SOURCE            every content token occurs in the source message.
  IDENTITY_CONTEXT  the remaining tokens only resolve a reference: the source contains a
                    referring expression and the extra tokens are (a) an entity listed in the
                    claim's entities that appears as a capitalized name in the window's preceding
                    messages, or as a name or listed entity of a prior-state claim, or (b) at most
                    ``MAX_REFERENT_TOKENS`` tokens of a referent from a preceding message
                    (requires a pronoun or demonstrative). Prior state can only supply entities.
  otherwise         rejected as UNSUPPORTED_CLAIM: context or prior state would be acting as
                    evidence for a new assertion.
  not comparable    rejected as UNVERIFIABLE_CLAIM: the source has no comparable tokens, the claim has
                    none beyond FRAME_TOKENS, or a word of the claim is lost by tokenization (text
                    outside the supported Latin-script scope).

This is a lexical check on an unordered set of English word stems, not semantic entailment. It
rejects faithful paraphrases whose words differ from the source, and it does not detect changed
negation, word order, speaker or certainty. Those rules are instructions to the model only.

Relations: the cited message must contain one endpoint, both endpoints must come from this
window (its targets or its conversation's prior state), and "evidence" must be an exact
substring of the cited message.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from continuator_memory.memory.models import normalize_value
from continuator_memory.memory.text import content_tokens
from continuator_memory.models import Message
from continuator_memory.v2.models import (
    MEMORY_TYPES,
    MODEL_STATUSES,
    POLARITIES,
    RELATION_TYPES,
    STATUS_CLAIMS,
    STATUS_FAILED,
    STATUS_MISSING,
    STATUS_NO_DURABLE,
    STATUS_REJECTED,
    SUPPORT_IDENTITY,
    SUPPORT_SOURCE,
    Claim,
    Provenance,
    Relation,
    SemanticMessage,
    Temporal,
    claim_id,
    relation_id,
)
from continuator_memory.v2.windowing import ContextWindow

REJECTED, FLAGGED = "rejected", "flagged"

MAX_REFERENT_TOKENS = 2
MAX_CLAIM_TYPE = 40

# Words a faithful restatement may add without new meaning: speaker labels, contraction
# fragments, negation and causal connectives.
FRAME_TOKENS = frozenset({
    "user", "assistant", "speaker", "m", "ll", "re", "ve", "d", "don", "doesn", "didn", "isn",
    "wasn", "aren", "weren", "won", "wouldn", "couldn", "shouldn", "haven", "hasn", "hadn",
    "cannot", "not", "no", "never", "longer", "anymore", "because", "since", "due", "therefore",
    "result",
})
PRONOUNS = frozenset(
    "i me my mine myself we us our ours you your yours he him his she her hers they them their "
    "theirs it its this that these those there then here".split())
# "<effect> because <cause>" (or sentence-initial "Because <cause>, <effect>") and
# "<cause>, so <effect>". Longer cues come first within each group.
CAUSAL_CUE = re.compile(
    r"\b(?:(?P<back>because of|because|since|due to|caused by|thanks to|owing to|as a result of)|"
    r"(?P<fwd>as a result|so|therefore|thus|hence|which is why|that's why|that is why|led to|leads to|"
    r"caus(?:es|ed)|delay(?:s|ed)|forc(?:es|ed)|prevent(?:s|ed)))\b",
    re.IGNORECASE)
CLAUSE_BREAK = re.compile(r"[.!?;:,]|\s[-\u2013\u2014]\s|\b(?:and|but)\b", re.IGNORECASE)
CHANGE_CUE = re.compile(
    r"\b(now|actually|instead|anymore|no longer|chang(?:e|es|ed)|mov(?:e|es|ed)|update[ds]?|"
    r"correct(?:ion|ed)?|not|switch(?:ed|es)?|replac(?:e|es|ed)|after all|rather|takes? over|"
    r"took over|wrong|mistake|mixed up|from now)\b",
    re.IGNORECASE)
NAME_SPAN = re.compile(r"\b[A-Z][\w'-]*(?:\s+(?:(?:de|del|da|di|van|von|der|la|le)\s+)?[A-Z][\w'-]*)*")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    action: str
    detail: str
    message_id: str | None = None
    ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "action": self.action, "detail": self.detail,
                "message_id": self.message_id, "ref": self.ref}


@dataclass
class ValidatedWindow:
    window: ContextWindow
    messages: list[SemanticMessage]
    issues: list[ValidationIssue] = field(default_factory=list)
    raw: str | None = None
    error: str | None = None

    @property
    def claims(self) -> list[Claim]:
        return [c for m in self.messages for c in m.claims]

    @property
    def relations(self) -> list[Relation]:
        return [r for m in self.messages for r in m.explicit_relations]

    @property
    def status(self) -> str:
        if self.error:
            return "error"
        return "invalid" if any(i.action == REJECTED for i in self.issues) else "ok"


# ---- tokens and support -----------------------------------------------------


def _root(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) - len(suffix) >= 3 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    return token[:-1] if len(token) > 3 and token.endswith("e") else token


def tokens(text: str) -> set[str]:
    return {_root(t) for t in content_tokens(text)}


def _comparable(text: str) -> bool:
    """False if tokenization would silently drop a whole word, leaving it unchecked."""
    return all(normalize_value(word) or not any(ch.isalnum() for ch in word) for word in text.split())


def _name_keys(text: str) -> set[str]:
    """Normalized capitalized spans of ``text`` and their contiguous word sub-spans."""
    keys = set()
    for span in NAME_SPAN.findall(text):
        words = normalize_value(span).split()
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                keys.add(" ".join(words[i:j]))
    return keys


def _has_referring_expression(text: str, entities: Sequence[str]) -> bool:
    words = set(normalize_value(text).split())
    if words & PRONOUNS or "the" in words:
        return True
    return any(words & set(normalize_value(e).split()) for e in entities)


def classify_support(
    content: str,
    entities: Sequence[str],
    source: Message,
    preceding: Sequence[Message],
    prior: Sequence[Claim],
) -> tuple[str | None, set[str], tuple[str, ...]]:
    """Return (support, unsupported tokens, resolved_from IDs); support is None if unsupported.
    No unsupported tokens with support None means the comparison could not run."""
    claim_tokens = tokens(content) - FRAME_TOKENS
    source_tokens = tokens(source.content)
    if not claim_tokens or not source_tokens or not _comparable(content):
        return None, set(), ()
    missing = claim_tokens - source_tokens
    if not missing:
        return SUPPORT_SOURCE, set(), ()
    if not _has_referring_expression(source.content, entities):
        return None, missing, ()
    resolved_from: list[str] = []
    for entity in entities:
        key = normalize_value(entity)
        if not key or not tokens(entity) & missing:
            continue
        origins = [m.message_id for m in preceding if key in _name_keys(m.content)]
        origins += [c.claim_id for c in prior
                    if key in _name_keys(c.content) or any(key == normalize_value(e) for e in c.entities)]
        if origins:
            missing -= tokens(entity)
            resolved_from += [o for o in origins if o not in resolved_from]
    if missing:
        words = set(normalize_value(source.content).split())
        origins = [m.message_id for m in preceding if missing <= tokens(m.content)]
        if len(missing) > MAX_REFERENT_TOKENS or not (words & PRONOUNS) or not origins:
            return None, missing, ()
        resolved_from += [o for o in origins[-1:] if o not in resolved_from]
    return SUPPORT_IDENTITY, set(), tuple(resolved_from)


# ---- causal links -----------------------------------------------------------


def _clause_before(text: str, end: int) -> str:
    segment = text[:end].rstrip(" ,")
    breaks = list(CLAUSE_BREAK.finditer(segment))
    return segment[breaks[-1].end():] if breaks else segment


def _clause_after(text: str, start: int) -> str:
    segment = text[start:].lstrip(" ,")
    found = CLAUSE_BREAK.search(segment)
    return segment[:found.start()] if found else segment


def _claim_matches_clause(claim: str, clause: str, source_tokens: set[str]) -> bool:
    """At least half of the claim's source-grounded tokens occur in the clause."""
    claim_tokens = tokens(claim) - FRAME_TOKENS
    overlap = claim_tokens & tokens(clause)
    return bool(overlap) and 2 * len(overlap) >= len(claim_tokens & source_tokens)


def causal_link_supported(source_text: str, effect: str, cause: str,
                          evidence_span: tuple[int, int] | None = None) -> bool:
    """True if a causal connective in ``source_text`` joins the clause stating ``effect`` with the
    clause stating ``cause``. Only the clauses adjacent to the connective count, so a cue in an
    unrelated clause does not link two other propositions. With an evidence span containing a
    connective, only connectives inside that span are considered."""
    source_tokens = tokens(source_text) - FRAME_TOKENS
    cues = list(CAUSAL_CUE.finditer(source_text))
    if evidence_span and any(evidence_span[0] <= m.start() and m.end() <= evidence_span[1] for m in cues):
        cues = [m for m in cues if evidence_span[0] <= m.start() and m.end() <= evidence_span[1]]
    for cue in cues:
        before, after = _clause_before(source_text, cue.start()), _clause_after(source_text, cue.end())
        if cue.group("back"):
            if before.strip():
                effect_side, cause_side = before, after
            else:
                rest = source_text[cue.end():].lstrip(" ,")
                found = CLAUSE_BREAK.search(rest)
                effect_side, cause_side = (_clause_after(rest, found.end()) if found else ""), after
        else:
            effect_side, cause_side = after, before
        if (_claim_matches_clause(effect, effect_side, source_tokens)
                and _claim_matches_clause(cause, cause_side, source_tokens)):
            return True
    return False


# ---- annotations ------------------------------------------------------------


def _strings(value: Any, name: str, issues: list[ValidationIssue], mid: str, ref: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        issues.append(ValidationIssue("INVALID_SCHEMA", FLAGGED, f"'{name}' is not a list; dropped", mid, ref))
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            if item.strip() not in out:
                out.append(item.strip())
        elif item not in ("", None):
            issues.append(ValidationIssue("INVALID_SCHEMA", FLAGGED, f"non-string in '{name}' dropped", mid, ref))
    return out


def _temporal(value: Any, source: Message, issues: list[ValidationIssue], ref: str) -> Temporal | None:
    mid = source.message_id
    if value in (None, "", {}):
        return None
    if not isinstance(value, dict):
        issues.append(ValidationIssue("MALFORMED_TEMPORAL", FLAGGED, "temporal is not an object; dropped", mid, ref))
        return None
    expression, resolved = value.get("expression"), value.get("resolved_date")
    if expression in (None, "") and resolved in (None, ""):
        return None
    if not isinstance(expression, str) or not expression.strip():
        issues.append(ValidationIssue("MALFORMED_TEMPORAL", FLAGGED, "temporal has no expression; dropped", mid, ref))
        return None
    expression = expression.strip()
    normalized = normalize_value(expression)
    if not normalized or normalized not in normalize_value(source.content):
        issues.append(ValidationIssue("TEMPORAL_NOT_IN_SOURCE", FLAGGED,
                                      f"temporal expression {expression!r} is not in the source; dropped", mid, ref))
        return None
    if resolved in (None, ""):
        return Temporal(expression)
    if not isinstance(resolved, str) or not ISO_DATE.match(resolved.strip()):
        issues.append(ValidationIssue("MALFORMED_TEMPORAL", FLAGGED,
                                      f"resolved_date {resolved!r} is not YYYY-MM-DD; dropped", mid, ref))
        return Temporal(expression)
    if not source.timestamp:
        issues.append(ValidationIssue("RESOLVED_WITHOUT_TIMESTAMP", FLAGGED,
                                      f"resolved_date {resolved!r} but the source has no timestamp; dropped", mid, ref))
        return Temporal(expression)
    return Temporal(expression, resolved.strip())


def _optional_enum(value: Any, allowed: Sequence[str], code: str, issues: list[ValidationIssue],
                   mid: str, ref: str) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    issues.append(ValidationIssue(code, FLAGGED, f"{value!r} not in {list(allowed)}; dropped", mid, ref))
    return None


# ---- window validation ------------------------------------------------------


def failed_window(window: ContextWindow, raw: str | None, error: str, code: str = "REQUEST_FAILED") -> ValidatedWindow:
    messages = [SemanticMessage(Provenance.of(m), m.content, STATUS_FAILED) for m in window.targets]
    return ValidatedWindow(window, messages, [ValidationIssue(code, REJECTED, error)], raw, error)


def _message_of_ref(ref: str) -> str:
    return ref.rsplit(":c", 1)[0] if ":c" in ref else ""


def validate_atomization(raw: str | None, window: ContextWindow, prior: Sequence[Claim] = ()) -> ValidatedWindow:
    try:
        data = json.loads(raw or "")
    except (json.JSONDecodeError, TypeError) as exc:
        return failed_window(window, raw, f"invalid JSON: {exc}", "INVALID_JSON")
    if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
        return failed_window(window, raw, "top level must be an object with a 'messages' list", "INVALID_SCHEMA")

    issues: list[ValidationIssue] = []
    targets = {m.message_id: m for m in window.targets}
    context_ids = set(window.context_ids)
    entries: dict[str, dict[str, Any]] = {}
    for i, entry in enumerate(data["messages"]):
        sid = entry.get("source_message_id") if isinstance(entry, dict) else None
        if not isinstance(entry, dict):
            issues.append(ValidationIssue("INVALID_SCHEMA", REJECTED, f"messages[{i}] is not an object"))
        elif sid in context_ids:
            issues.append(ValidationIssue("IMPOSSIBLE_SOURCE", REJECTED,
                                          f"messages[{i}] is a CONTEXT message, not a target", sid))
        elif sid not in targets:
            issues.append(ValidationIssue("INVALID_SOURCE_MESSAGE", REJECTED,
                                          f"messages[{i}] has unknown source_message_id {sid!r}", str(sid)))
        elif sid in entries:
            issues.append(ValidationIssue("DUPLICATE_MESSAGE_ENTRY", REJECTED,
                                          f"messages[{i}] repeats {sid}", sid))
        else:
            entries[sid] = entry

    claims: dict[str, list[Claim]] = {mid: [] for mid in targets}
    statuses: dict[str, str] = {}
    model_ids: dict[str, list[str]] = {}  # model claim id -> canonical ids
    rejected_refs: set[str] = set()
    for mid, message in targets.items():
        entry = entries.get(mid)
        if entry is None:
            issues.append(ValidationIssue("MISSING_MESSAGE", FLAGGED, "no entry for target message", mid))
            statuses[mid] = STATUS_MISSING
            continue
        status = entry.get("status")
        raw_claims = entry.get("claims", [])
        if status not in MODEL_STATUSES:
            issues.append(ValidationIssue("INVALID_SCHEMA", FLAGGED, f"invalid status {status!r}", mid))
            status = STATUS_CLAIMS if raw_claims else STATUS_NO_DURABLE
        if not isinstance(raw_claims, list):
            issues.append(ValidationIssue("INVALID_SCHEMA", REJECTED, "'claims' is not a list", mid))
            raw_claims = []
        if status == STATUS_CLAIMS and not raw_claims:
            issues.append(ValidationIssue("EMPTY_CLAIMS_STATUS", FLAGGED, "status 'claims' with no claims", mid))
        if status == STATUS_NO_DURABLE and raw_claims:
            issues.append(ValidationIssue("NO_DURABLE_WITH_CLAIMS", FLAGGED,
                                          "status 'no_durable_assertion' but claims given; claims validated", mid))
        preceding = window.preceding(message)
        seen_ids: set[str] = set()
        for j, item in enumerate(raw_claims):
            ref = f"{mid}/claims[{j}]"
            if not isinstance(item, dict):
                issues.append(ValidationIssue("INVALID_SCHEMA", REJECTED, "claim is not an object", mid, ref))
                continue
            model_id = item.get("claim_id")
            model_id = model_id.strip() if isinstance(model_id, str) and model_id.strip() else None
            if model_id and model_id in seen_ids:
                issues.append(ValidationIssue("DUPLICATE_CLAIM_ID", REJECTED, f"claim id {model_id!r} repeated", mid, model_id))
                continue
            if model_id:
                seen_ids.add(model_id)
            source_id = item.get("source_message_id", mid)
            if source_id != mid:
                code = ("IMPOSSIBLE_SOURCE" if source_id in context_ids or source_id in targets
                        else "INVALID_SOURCE_MESSAGE")
                issues.append(ValidationIssue(code, REJECTED,
                                              f"claim attributed to {source_id!r} inside the entry for {mid}", mid, model_id or ref))
                if model_id:
                    rejected_refs.add(model_id)
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                issues.append(ValidationIssue("EMPTY_CLAIM", REJECTED, "empty claim content", mid, model_id or ref))
                if model_id:
                    rejected_refs.add(model_id)
                continue
            content = " ".join(content.split())
            entities = _strings(item.get("entities"), "entities", issues, mid, ref)
            support, unsupported, resolved_from = classify_support(content, entities, message, preceding, prior)
            if support is None:
                detail = (f"{content!r}: not supported by the source ({sorted(unsupported)})" if unsupported
                          else f"{content!r}: cannot be compared with the source; rejected, not assumed supported")
                issues.append(ValidationIssue("UNSUPPORTED_CLAIM" if unsupported else "UNVERIFIABLE_CLAIM", REJECTED,
                                              detail, mid, model_id or ref))
                if model_id:
                    rejected_refs.add(model_id)
                continue
            allowed = tokens(message.content) | FRAME_TOKENS
            for p in preceding:
                allowed |= tokens(p.content)
            for c in prior:
                allowed |= tokens(c.content) | {t for e in c.entities for t in tokens(e)}
            kept_entities = []
            for e in entities:
                if tokens(e) <= allowed and _comparable(e):
                    kept_entities.append(e)
                else:
                    issues.append(ValidationIssue("UNSUPPORTED_ENTITY", FLAGGED, f"entity {e!r} not in source/context; dropped", mid, ref))
            qualifiers = []
            for q in _strings(item.get("qualifiers"), "qualifiers", issues, mid, ref):
                if tokens(q) <= tokens(message.content) | FRAME_TOKENS and _comparable(q):
                    qualifiers.append(q)
                else:
                    issues.append(ValidationIssue("UNSUPPORTED_QUALIFIER", FLAGGED, f"qualifier {q!r} not in source; dropped", mid, ref))
            claim_type = item.get("claim_type")
            if claim_type in ("", None) or not isinstance(claim_type, str):
                claim_type = None
            elif len(claim_type) > MAX_CLAIM_TYPE:
                issues.append(ValidationIssue("INVALID_SCHEMA", FLAGGED, "claim_type too long; dropped", mid, ref))
                claim_type = None
            metadata = {}
            memory_type = item.get("memory_type")
            if memory_type not in (None, ""):
                if memory_type in MEMORY_TYPES:
                    metadata["memory_type"] = memory_type
                else:
                    issues.append(ValidationIssue("INVALID_MEMORY_TYPE", FLAGGED,
                                                  f"memory_type {memory_type!r} unknown; dropped", mid, ref))
            cid = claim_id(mid, len(claims[mid]) + 1)
            claims[mid].append(Claim(
                claim_id=cid, content=content, provenance=Provenance.of(message),
                entities=tuple(kept_entities), qualifiers=tuple(qualifiers),
                temporal=_temporal(item.get("temporal"), message, issues, cid),
                polarity=_optional_enum(item.get("polarity"), POLARITIES, "INVALID_POLARITY", issues, mid, cid),
                claim_type=claim_type.strip() if claim_type else None,
                support=support, resolved_from=resolved_from, metadata=metadata))
            model_ids.setdefault(model_id or cid, []).append(cid)
        if claims[mid]:
            statuses[mid] = STATUS_CLAIMS
        elif raw_claims:
            statuses[mid] = STATUS_REJECTED
        else:
            statuses[mid] = STATUS_NO_DURABLE if status == STATUS_NO_DURABLE else STATUS_CLAIMS

    relations = _validate_relations(entries, window, claims, model_ids, rejected_refs, prior, issues)
    messages = [SemanticMessage(Provenance.of(m), m.content, statuses[m.message_id], tuple(claims[m.message_id]),
                                tuple(relations[m.message_id])) for m in window.targets]
    return ValidatedWindow(window, messages, issues, raw)


def _validate_relations(
    entries: dict[str, dict[str, Any]],
    window: ContextWindow,
    claims: dict[str, list[Claim]],
    model_ids: dict[str, list[str]],
    rejected_refs: set[str],
    prior: Sequence[Claim],
    issues: list[ValidationIssue],
) -> dict[str, list[Relation]]:
    available: dict[str, Claim] = {c.claim_id: c for c in prior}
    for cs in claims.values():
        available.update({c.claim_id: c for c in cs})
    prior_messages = {c.source_message_id for c in prior}
    targets = {m.message_id: m for m in window.targets}
    out: dict[str, list[Relation]] = {mid: [] for mid in targets}
    seen: set[tuple[str, str, str]] = set()

    def resolve(ref: Any, mid: str, rref: str) -> Claim | None:
        if not isinstance(ref, str) or not ref.strip():
            issues.append(ValidationIssue("INVALID_SCHEMA", REJECTED, "relation endpoint is not a claim id", mid, rref))
            return None
        ref = ref.strip()
        canonical = model_ids.get(ref, [])
        if len(canonical) > 1:
            issues.append(ValidationIssue("AMBIGUOUS_CLAIM_REF", REJECTED, f"{ref!r} names several claims", mid, rref))
            return None
        if canonical:
            return available[canonical[0]]
        if ref in available and ref not in rejected_refs:
            return available[ref]
        owner = _message_of_ref(ref)
        if ref in rejected_refs:
            code, detail = "NONEXISTENT_CLAIM", f"{ref!r} was rejected by validation"
        elif owner in window.context_ids or (owner and owner not in targets and owner not in prior_messages):
            code, detail = "UNAVAILABLE_SOURCE", f"{ref!r} belongs to a message whose claims are not available here"
        else:
            code, detail = "NONEXISTENT_CLAIM", f"{ref!r} does not exist"
        issues.append(ValidationIssue(code, REJECTED, detail, mid, rref))
        return None

    for mid, entry in entries.items():
        raw_relations = entry.get("explicit_relations", [])
        if not isinstance(raw_relations, list):
            issues.append(ValidationIssue("INVALID_SCHEMA", REJECTED, "'explicit_relations' is not a list", mid))
            continue
        for k, item in enumerate(raw_relations):
            rref = f"{mid}/explicit_relations[{k}]"
            if not isinstance(item, dict):
                issues.append(ValidationIssue("INVALID_SCHEMA", REJECTED, "relation is not an object", mid, rref))
                continue
            rtype = item.get("type")
            if rtype not in RELATION_TYPES:
                issues.append(ValidationIssue("INVALID_RELATION_TYPE", REJECTED, f"type {rtype!r}", mid, rref))
                continue
            source_id = item.get("source_message_id", mid)
            if source_id not in targets:
                code = "IMPOSSIBLE_SOURCE" if source_id in window.context_ids else "INVALID_SOURCE_MESSAGE"
                issues.append(ValidationIssue(code, REJECTED, f"relation source {source_id!r} is not a target", mid, rref))
                continue
            src, dst = resolve(item.get("from_claim_id"), mid, rref), resolve(item.get("to_claim_id"), mid, rref)
            if src is None or dst is None:
                continue
            if src.claim_id == dst.claim_id:
                issues.append(ValidationIssue("SELF_LOOP", REJECTED, f"{src.claim_id} -> itself", mid, rref))
                continue
            if source_id not in (src.source_message_id, dst.source_message_id):
                issues.append(ValidationIssue("RELATION_SOURCE_MISMATCH", REJECTED,
                                              f"{source_id} contains neither endpoint", mid, rref))
                continue
            if any(c.provenance.conversation_id != window.conversation_id for c in (src, dst)):
                issues.append(ValidationIssue("ENDPOINT_OUTSIDE_WINDOW", REJECTED,
                                              "relation endpoint is not from this window's conversation", mid, rref))
                continue
            if rtype in ("UPDATES", "CONTRADICTS", "REFERS_TO") and src.provenance.position < dst.provenance.position:
                issues.append(ValidationIssue("UNSUPPORTED_DIRECTION", REJECTED,
                                              f"{rtype} must point from the newer claim to the older one", mid, rref))
                continue
            key = (src.claim_id, dst.claim_id, rtype)
            if key in seen:
                issues.append(ValidationIssue("DUPLICATE_RELATION", REJECTED, f"{key} repeated", mid, rref))
                continue
            if rtype in ("CAUSED_BY", "SUPPORTS") and (dst.claim_id, src.claim_id, rtype) in seen:
                issues.append(ValidationIssue("UNSUPPORTED_DIRECTION", REJECTED,
                                              f"{rtype} in both directions between {src.claim_id} and {dst.claim_id}", mid, rref))
                continue
            source_text = targets[source_id].content
            if rtype == "CAUSED_BY":
                ev = item.get("evidence")
                found = (re.search(rf"(?<!\w){re.escape(ev.strip())}(?!\w)", source_text, re.IGNORECASE)
                         if isinstance(ev, str) and ev.strip() else None)
                if not causal_link_supported(source_text, src.content, dst.content, found.span() if found else None):
                    issues.append(ValidationIssue("UNSUPPORTED_RELATION", REJECTED,
                                                  "CAUSED_BY but no causal connective in the source joins these two claims",
                                                  mid, rref))
                    continue
            evidence = item.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip() or evidence.strip() not in source_text:
                issues.append(ValidationIssue("EVIDENCE_NOT_IN_SOURCE", REJECTED,
                                              f"evidence {evidence!r} is not an exact substring of {source_id}", mid, rref))
                continue
            if rtype in ("UPDATES", "CONTRADICTS") and not CHANGE_CUE.search(source_text):
                issues.append(ValidationIssue("NO_CHANGE_CUE", FLAGGED,
                                              f"{rtype} but the source has no explicit change/correction wording", mid, rref))
            evidence = evidence.strip()
            start = source_text.index(evidence)
            span = (start, start + len(evidence))
            seen.add(key)
            out[source_id].append(Relation(relation_id(source_id, len(out[source_id]) + 1), src.claim_id,
                                           dst.claim_id, rtype, source_id, evidence, span))
    return out
