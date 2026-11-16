"""Product Context Pack: retrieved memories + source evidence for any downstream LLM.

A Context Pack is a view over existing ``SourceEvidence`` and ``AnswerContext`` data.
It does not introduce a second memory schema. Continuator's answering layer consumes the
same pack; external applications can use ``formatted_context`` or the structured fields
without calling Continuator's answer generator.

Document EvidenceUnit hits (BM25) are an additive parallel field and are not folded into
message-shaped ``evidence`` / ``AnswerContext``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from continuator_memory.evidence.models import EvidenceHit
from continuator_memory.webapi.answering import AnswerContext, context_to_evidence_fields
from continuator_memory.v2.source_retrieval import SourceEvidence

# Public Context Pack JSON schema identifier. Additive optional fields may appear
# within v1; renaming/removing fields or changing types requires a new version.
CONTEXT_PACK_SCHEMA_VERSION = "1"

# Top-level keys always present on ContextPack.to_dict() (schema v1).
CONTEXT_PACK_REQUIRED_KEYS = frozenset({
    "schemaVersion",
    "query",
    "mode",
    "conversationId",
    "memories",
    "evidence",
    "citations",
    "documentEvidence",
    "documentCitations",
    "formattedContext",
    "selection",
    "seedMemoryIds",
    "expandedMemoryIds",
    "hops",
    "diagnostics",
    "queryPlan",
    "answerPrompt",
})


def document_evidence_row(hit: EvidenceHit) -> dict[str, Any]:
    """Public camelCase row for one document EvidenceUnit hit."""
    return {
        "evidenceId": hit.unit.evidence_id,
        "documentId": hit.document.document_id,
        "sourceUri": hit.document.source_uri,
        "sourceType": hit.document.source_type,
        "title": hit.document.title,
        "headingPath": list(hit.unit.heading_path),
        "sectionId": hit.unit.section_id,
        "charStart": hit.unit.char_start,
        "charEnd": hit.unit.char_end,
        "text": hit.unit.text,
        "rank": hit.rank,
        "score": hit.score,
    }


def document_citation_row(hit: EvidenceHit) -> dict[str, Any]:
    """Public camelCase provenance citation for one document EvidenceUnit hit."""
    return {
        "evidenceId": hit.unit.evidence_id,
        "documentId": hit.document.document_id,
        "sourceUri": hit.document.source_uri,
        "sectionId": hit.unit.section_id,
        "headingPath": list(hit.unit.heading_path),
    }


@dataclass(frozen=True)
class ContextPack:
    """Canonical product retrieval result: structured data + deterministic text context."""

    query: str
    mode: str
    evidence: SourceEvidence
    answer_context: AnswerContext
    conversation_id: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    query_diagnostics: dict[str, Any] = field(default_factory=dict)
    seed_memory_ids: tuple[str, ...] = ()
    expanded_memory_ids: tuple[str, ...] = ()
    hops: tuple[dict[str, Any], ...] = ()
    document_hits: tuple[EvidenceHit, ...] = ()

    @property
    def formatted_context(self) -> str:
        """Deterministic text block suitable for injection into an external LLM prompt."""
        return format_context_pack(self)

    def to_dict(self) -> dict[str, Any]:
        """Public JSON shape for ``POST /context`` (and nested under ask diagnostics)."""
        memories = [
            {
                "id": m.memory_id,
                "text": m.statement,
                "rank": m.rank,
                "score": m.score,
                "similarity": m.similarity,
                "status": m.status,
                "sourceMessageIds": list(m.source_message_ids),
                "sources": [
                    {
                        "messageId": s.message_id,
                        "claimId": s.claim_id,
                        "role": s.role,
                        "text": s.content,
                        "timestamp": s.timestamp,
                    }
                    for s in m.sources
                ],
            }
            for m in self.evidence.matches
        ]
        evidence_rows = []
        for excerpt in self.answer_context.excerpts:
            evidence_rows.append({
                "messageId": excerpt.message_id,
                "role": excerpt.role,
                "text": excerpt.content,
                "memoryIds": list(excerpt.memory_ids),
                "truncated": excerpt.truncated,
                "timestamp": excerpt.timestamp,
            })
        # Include selected sources that were budget-dropped from the answerer prompt but still retrieved.
        in_prompt = {e.message_id for e in self.answer_context.excerpts}
        by_match_source = {}
        for match in self.evidence.matches:
            for src in match.sources:
                by_match_source.setdefault(src.message_id, src)
        for mid in self.evidence.selection.selected:
            if mid in in_prompt or mid not in by_match_source:
                continue
            src = by_match_source[mid]
            owners = [
                m.memory_id for m in self.evidence.matches if mid in m.source_message_ids
            ]
            evidence_rows.append({
                "messageId": src.message_id,
                "role": src.role,
                "text": src.content,
                "memoryIds": owners,
                "truncated": False,
                "timestamp": src.timestamp,
                "inAnswerPrompt": False,
            })
        citations = [
            {
                "memoryId": m.memory_id,
                "messageIds": list(m.source_message_ids),
                "unresolved": [u.to_dict() for u in m.unresolved],
            }
            for m in self.evidence.matches
        ]
        extras = {
            k: v for k, v in self.query_diagnostics.items()
            if k not in {"queryPlan"} and v is not None
        }
        return {
            "schemaVersion": CONTEXT_PACK_SCHEMA_VERSION,
            "query": self.query,
            "mode": self.mode,
            "conversationId": self.conversation_id or None,
            "memories": memories,
            "evidence": evidence_rows,
            "citations": citations,
            "documentEvidence": [document_evidence_row(h) for h in self.document_hits],
            "documentCitations": [document_citation_row(h) for h in self.document_hits],
            "formattedContext": self.formatted_context,
            "selection": self.evidence.selection.to_dict(),
            "seedMemoryIds": list(self.seed_memory_ids),
            "expandedMemoryIds": list(self.expanded_memory_ids),
            "hops": list(self.hops),
            "diagnostics": dict(self.diagnostics),
            "queryPlan": self.query_diagnostics.get("queryPlan"),
            "answerPrompt": {
                "system": self.answer_context.system,
                "prompt": self.answer_context.prompt,
                "promptMemoryIds": list(self.answer_context.prompt_memory_ids),
                "droppedMemoryIds": list(self.answer_context.dropped_memory_ids),
                "droppedMessageIds": list(self.answer_context.dropped_message_ids),
            },
            **extras,
        }


def format_context_pack(pack: ContextPack) -> str:
    """Build a deterministic, model-agnostic context string from retrieved memories + evidence."""
    lines = [
        "RELEVANT LONG-TERM MEMORY",
        f"Query: {pack.query}",
        f"Retrieval mode: {pack.mode}",
        "",
    ]
    if not pack.evidence.matches:
        lines.append("(no memories retrieved)")
    else:
        for match in pack.evidence.matches:
            lines.append(f"[{match.rank}] memory {match.memory_id}")
            lines.append(
                f"status={match.status} score={match.score:.6f} similarity={match.similarity:.6f}"
            )
            lines.append(f"Statement: {match.statement}")
            if match.sources:
                lines.append("Supporting evidence:")
                for src in match.sources:
                    ts = f" @ {src.timestamp}" if src.timestamp else ""
                    lines.append(f"  [{src.message_id}] {src.role}{ts}: {src.content}")
            elif match.unresolved:
                lines.append("Supporting evidence: (unresolved provenance)")
                for u in match.unresolved:
                    lines.append(
                        f"  unresolved claim={u.claim_id} message={u.message_id} reason={u.reason}"
                    )
            else:
                lines.append("Supporting evidence: (none)")
            lines.append("")
        selected = list(pack.evidence.selection.selected)
        if selected:
            lines.append(f"Selected source message ids (budget order): {', '.join(selected)}")

    if pack.document_hits:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("RELEVANT DOCUMENT EVIDENCE")
        lines.append("")
        for hit in pack.document_hits:
            lines.append(f"[{hit.rank}] evidence {hit.unit.evidence_id}")
            lines.append(
                f"document={hit.document.document_id} score={hit.score:.6f} "
                f"sourceType={hit.document.source_type}"
            )
            lines.append(f"sourceUri={hit.document.source_uri}")
            if hit.unit.heading_path:
                lines.append(f"headingPath={' > '.join(hit.unit.heading_path)}")
            lines.append(f"sectionId={hit.unit.section_id}")
            lines.append(hit.unit.text)
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def assemble_context_pack(
    *,
    query: str,
    mode: str,
    evidence: SourceEvidence,
    answer_context: AnswerContext,
    conversation_id: str = "",
    diagnostics: dict[str, Any] | None = None,
    query_diagnostics: dict[str, Any] | None = None,
    seed_memory_ids: list[str] | tuple[str, ...] = (),
    expanded_memory_ids: list[str] | tuple[str, ...] = (),
    hops: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    document_hits: list[EvidenceHit] | tuple[EvidenceHit, ...] = (),
) -> ContextPack:
    """Assemble a ContextPack from the canonical retrieval + answer-context builders."""
    return ContextPack(
        query=query,
        mode=mode,
        evidence=evidence,
        answer_context=answer_context,
        conversation_id=conversation_id,
        diagnostics=dict(diagnostics or {}),
        query_diagnostics=dict(query_diagnostics or {}),
        seed_memory_ids=tuple(seed_memory_ids),
        expanded_memory_ids=tuple(expanded_memory_ids),
        hops=tuple(hops),
        document_hits=tuple(document_hits),
    )


def context_pack_evidence_fields(pack: ContextPack) -> dict[str, Any]:
    """Fields historically nested under ask evidence, derived from the Context Pack."""
    selected = list(pack.evidence.selection.selected)
    return {
        "memoryIds": [m.memory_id for m in pack.evidence.matches],
        "sourceIds": [pack.conversation_id] if selected and pack.conversation_id else [],
        "expandedMemoryIds": list(pack.expanded_memory_ids),
        "hops": list(pack.hops),
        "diagnostics": dict(pack.diagnostics),
        "selectedMessageIds": selected,
        "seedMemoryIds": list(pack.seed_memory_ids),
        "queryPlan": pack.query_diagnostics.get("queryPlan"),
        "queryRankings": pack.query_diagnostics.get("queryRankings"),
        "fusedRanking": pack.query_diagnostics.get("fusedRanking"),
        "retrievalComparison": pack.query_diagnostics.get("retrievalComparison"),
        "iterativeRounds": pack.query_diagnostics.get("iterativeRounds"),
        "stopReason": pack.query_diagnostics.get("stopReason"),
        "contextPack": {
            "formattedContext": pack.formatted_context,
            "citations": [
                {"memoryId": m.memory_id, "messageIds": list(m.source_message_ids)}
                for m in pack.evidence.matches
            ],
        },
        **context_to_evidence_fields(pack.answer_context),
    }
