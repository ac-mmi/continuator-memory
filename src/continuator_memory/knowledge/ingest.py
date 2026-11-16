"""One sequential model call per non-overlapping chunk, stored with passages and embeddings.

Documents are independent: a chunk sees up to two earlier raw messages as read-only context, and never an earlier
document or summary. Answer evidence stays the original messages; this module neither retrieves nor answers.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

from continuator_memory.chunking import chunk_conversation
from continuator_memory.embedding import Embedder
from continuator_memory.generation import GenerationError
from continuator_memory.knowledge.documents import (
    FORMAT_VERSION,
    PROMPT_VERSION,
    REVIEW_EVIDENCE_CITED,
    REVIEW_NEEDED,
    SYSTEM_PROMPT,
    TRUNCATED_DOCUMENT,
    DocumentClaim,
    ParsedDocument,
    build_prompt,
    parse_document,
)
from continuator_memory.knowledge.store import DocumentRow, KnowledgeStore, PassageRow
from continuator_memory.models import Chunk, Conversation
from continuator_memory.store import Store

CHUNK_SIZE = 4
CONTEXT_MESSAGES = 2
NUM_PREDICT = 4096
NUM_CTX = 8192
# MiniLM reads about 256 word pieces and ignores the rest. Characters / 4 is an estimate, kept well under that.
MAX_PASSAGE_TOKENS = 180
CHARS_PER_TOKEN = 4

STATUS_OK = "ok"
STATUS_NOTHING_DURABLE = "nothing_durable"
STATUS_TRUNCATED = "truncated"
STATUS_FAILED = "failed"
CITED = "cited"
CHUNK_FALLBACK = "chunk_fallback"


class TextClient(Protocol):
    model_name: str
    last_usage: dict | None

    def generate(self, system: str, prompt: str, *, temperature: float | None = None, num_predict: int | None = None) -> str: ...


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _day_range(chunk: Chunk, stamps: dict[str, str | None]) -> str:
    days = [stamps[m] for m in chunk.source_message_ids if stamps.get(m)]
    if not days:
        return ""
    return min(days) if min(days) == max(days) else f"{min(days)} to {max(days)}"


def build_passages(document_id: str, claims: list[DocumentClaim], chunk: Chunk, dates: str,
                   max_tokens: int = MAX_PASSAGE_TOKENS) -> list[PassageRow]:
    """Split by section, then between claims, never inside one. A claim with no usable citation gets its own
    passage pointing at the whole chunk, so it cannot borrow a neighbour's messages."""
    groups: list[list[DocumentClaim]] = []
    current: list[DocumentClaim] = []
    size = 0
    for claim in claims:
        lone = not claim.message_ids
        cost = estimate_tokens(claim.text) + 1
        if current and (lone or not current[-1].message_ids or claim.section != current[-1].section or size + cost > max_tokens):
            groups.append(current)
            current, size = [], 0
        current.append(claim)
        size += cost
    if current:
        groups.append(current)
    out: list[PassageRow] = []
    for index, group in enumerate(groups):
        heading = group[0].section or "(no section)"
        content = "\n".join(f"- {c.text}" for c in group)
        prefix = f"{heading} | {dates}" if dates else heading
        embedded = f"{prefix}\n{content}"
        message_ids = tuple(dict.fromkeys(m for c in group for m in c.message_ids))
        out.append(PassageRow(
            f"{document_id}:p{index:03d}", index, heading, content, embedded, tuple(c.claim_index for c in group),
            message_ids or chunk.source_message_ids, CITED if message_ids else CHUNK_FALLBACK,
            REVIEW_NEEDED if any(c.review_reasons for c in group) else REVIEW_EVIDENCE_CITED, estimate_tokens(embedded)))
    return out


def _mark_truncated(parsed: ParsedDocument) -> list[DocumentClaim]:
    return [DocumentClaim(c.claim_index, c.section, c.text, c.citations, c.message_ids, (*c.review_reasons, TRUNCATED_DOCUMENT))
            for c in parsed.claims]


@dataclass
class IngestReport:
    conversation_id: str
    documents: int = 0
    model_calls: int = 0
    ok: int = 0
    nothing_durable: int = 0
    truncated: int = 0
    failed: int = 0
    claims: int = 0
    claims_needing_review: int = 0
    passages: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def prepare_source(conversation: Conversation, store: Store, embedder: Embedder) -> list[Chunk]:
    """Store the raw conversation only if absent. An existing one is compared and never rewritten."""
    chunks = chunk_conversation(conversation, CHUNK_SIZE, 0)
    existing = store.get_conversation(conversation.conversation_id)
    if existing is None:
        store.save_conversation(conversation, chunks, embedder.embed([c.text for c in chunks]), embedder.model_name)
        return chunks
    if existing.messages != conversation.messages:
        raise ValueError(f"conversation {conversation.conversation_id!r} is already stored with different messages")
    stored = store.get_chunks(conversation_id=conversation.conversation_id)
    if [(c.chunk_id, c.source_message_ids, c.text) for c in stored] != [(c.chunk_id, c.source_message_ids, c.text) for c in chunks]:
        raise ValueError(f"conversation {conversation.conversation_id!r} is stored with different chunks; use a fresh database")
    return chunks


def ingest_documents(
    conversation: Conversation,
    store: Store,
    client: TextClient,
    embedder: Embedder,
    *,
    replace: bool = False,
    on_document: Callable[[Chunk, str, DocumentRow, list[PassageRow]], None] | None = None,
    clock: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat(),
) -> IngestReport:
    """Each document is saved as it finishes, so a failure or interruption keeps earlier ones."""
    cid = conversation.conversation_id
    chunks = prepare_source(conversation, store, embedder)
    knowledge = KnowledgeStore(store.connection)
    if knowledge.has_documents(cid, PROMPT_VERSION, client.model_name):
        if not replace:
            raise ValueError(f"conversation {cid!r} already has documents for this prompt and model; pass replace=True")
        knowledge.delete_documents(cid, PROMPT_VERSION, client.model_name)
    by_id = {m.message_id: m for m in conversation.messages}
    stamps = {m.message_id: m.timestamp for m in conversation.messages}
    report = IngestReport(cid)
    for chunk in chunks:
        targets = [by_id[m] for m in chunk.source_message_ids]
        start = targets[0].position
        context = list(conversation.messages[max(0, start - CONTEXT_MESSAGES):start])
        prompt = build_prompt(targets, context)
        if hasattr(client, "last_usage"):
            client.last_usage = None
        t0 = time.perf_counter()
        raw, error = None, None
        try:
            raw = client.generate(SYSTEM_PROMPT, prompt, temperature=0.0, num_predict=NUM_PREDICT)
        except (GenerationError, OSError, ValueError) as exc:
            error = str(exc)
        latency = round((time.perf_counter() - t0) * 1000, 3)
        usage = getattr(client, "last_usage", None)
        report.model_calls += 1

        issues: list[str] = []
        claims: list[DocumentClaim] = []
        uncited: tuple[str, ...] = chunk.source_message_ids
        content = None
        if error is not None:
            status = STATUS_FAILED
        elif raw is None or not raw.strip():
            status, error = STATUS_FAILED, "the model returned no text"
        else:
            parsed = parse_document(raw, targets, context)
            issues, claims, uncited, content = list(parsed.issues), parsed.claims, parsed.uncited_target_ids, parsed.content
            finish = (usage or {}).get("done_reason")
            if finish == "length":
                status, error = STATUS_TRUNCATED, "the model hit its output limit (done_reason=length)"
                claims = _mark_truncated(parsed)
            elif parsed.nothing_durable:
                status = STATUS_NOTHING_DURABLE
            else:
                status = STATUS_OK
            if finish is None:
                issues.append("DONE_REASON_MISSING")

        document_id = f"{chunk.chunk_id}:{PROMPT_VERSION}"
        passages = build_passages(document_id, claims, chunk, _day_range(chunk, stamps)) if status == STATUS_OK else []
        vectors = embedder.embed([p.embedded_text for p in passages]) if passages else None
        row = DocumentRow(document_id, cid, chunk.chunk_id, chunk.position, FORMAT_VERSION, PROMPT_VERSION, client.model_name, status,
                          content, sha256_text(content) if content else None, sha256_text(chunk.text),
                          tuple(m.message_id for m in context), tuple(c.to_dict() for c in claims), tuple(issues), tuple(uncited),
                          error, usage, latency, clock())
        knowledge.save_document(row, passages, vectors, embedder.model_name if passages else None)
        report.documents += 1
        report.ok += status == STATUS_OK
        report.nothing_durable += status == STATUS_NOTHING_DURABLE
        report.truncated += status == STATUS_TRUNCATED
        report.failed += status == STATUS_FAILED
        report.claims += len(claims)
        report.claims_needing_review += sum(bool(c.review_reasons) for c in claims)
        report.passages += len(passages)
        report.prompt_tokens += (usage or {}).get("prompt_eval_count") or 0
        report.output_tokens += (usage or {}).get("eval_count") or 0
        report.latency_ms += latency
        if error:
            report.errors.append(f"{chunk.chunk_id}: {error}")
        if on_document:
            on_document(chunk, prompt, row, passages)
    return report
