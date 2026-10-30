"""V2 ingestion: raw storage -> context windows -> atomization -> validation -> semantic store.

No consolidation or versioning happens here; the stored semantic records are the input to
that later phase. Raw messages (and, with an embedder, the V1 chunk/vector index) are written
by the unchanged V1 code path.

A conversation with no stored claims has nothing to protect: its windows are written as they
complete, so partial results survive a failure. If claims are already stored, nothing is written
until every window has been extracted and validated, then raw and semantic rows are replaced in
one transaction, so an exception or interruption leaves the database untouched. Such a run does not
replace the stored rows (``V2IngestReport.kept_existing``) if a window failed, or if it produced no
claims for any reason other than the model reporting nothing durable in every message. Claims the
validator rejects from a run that still produced claims are normal and do not block replacement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from continuator_memory.chunking import chunk_conversation
from continuator_memory.embedding import Embedder
from continuator_memory.models import Chunk, Conversation
from continuator_memory.store import Store
from continuator_memory.v2.atomizer import DEFAULT_PRIOR_LIMIT, SemanticAtomizer
from continuator_memory.v2.models import STATUS_NO_DURABLE, Claim, SemanticMessage
from continuator_memory.v2.store import SemanticStore
from continuator_memory.v2.validation import ValidatedWindow
from continuator_memory.v2.windowing import DEFAULT_CONTEXT_BEFORE, DEFAULT_TARGETS_PER_WINDOW, build_windows


@dataclass
class V2IngestReport:
    conversation_id: str
    windows: int = 0
    messages: int = 0
    claims: int = 0
    relations: int = 0
    failed_windows: int = 0
    statuses: Counter = field(default_factory=Counter)
    issues: Counter = field(default_factory=Counter)
    latency_ms: float = 0.0
    model_calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    kept_existing: bool = False  # the run was discarded and previously stored rows left in place

    def to_dict(self) -> dict[str, Any]:
        return {"conversation_id": self.conversation_id, "windows": self.windows, "messages": self.messages,
                "claims": self.claims, "relations": self.relations, "failed_windows": self.failed_windows,
                "statuses": dict(sorted(self.statuses.items())), "issues": dict(sorted(self.issues.items())),
                "latency_ms": round(self.latency_ms, 3), "model_calls": self.model_calls,
                "prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
                "kept_existing": self.kept_existing}


class V2IngestionPipeline:
    def __init__(
        self,
        store: Store,
        atomizer: SemanticAtomizer,
        embedder: Embedder | None = None,
        targets_per_window: int = DEFAULT_TARGETS_PER_WINDOW,
        context_before: int = DEFAULT_CONTEXT_BEFORE,
        prior_limit: int = DEFAULT_PRIOR_LIMIT,
        chunk_window: int = 4,
        chunk_overlap: int = 2,
    ):
        self.store = store
        self.atomizer = atomizer
        self.embedder = embedder
        self.targets_per_window = targets_per_window
        self.context_before = context_before
        self.prior_limit = prior_limit
        self.chunk_window = chunk_window
        self.chunk_overlap = chunk_overlap
        self.semantic = SemanticStore(store.connection)

    def _raw_rows(self, conversation: Conversation) -> tuple[list[Chunk], np.ndarray, str]:
        if self.embedder is None:
            return [], np.zeros((0, 0), dtype=np.float32), "none"
        chunks = chunk_conversation(conversation, self.chunk_window, self.chunk_overlap)
        return chunks, self.embedder.embed([c.text for c in chunks]), self.embedder.model_name

    def ingest(self, conversation: Conversation) -> V2IngestReport:
        cid = conversation.conversation_id
        protect_existing = self.semantic.has_claims(cid)
        if not protect_existing:
            self.store.save_conversation(conversation, *self._raw_rows(conversation))
            self.semantic.delete_conversation(cid)
        report = V2IngestReport(cid)
        staged: list[tuple[ValidatedWindow, float, list[str]]] = []
        prior: list[Claim] = []
        for window in build_windows(conversation, self.targets_per_window, self.context_before):
            window_prior = prior[-self.prior_limit:] if self.prior_limit else []
            result = self.atomizer.atomize(window, window_prior)
            validated = result.validated
            prior_ids = [c.claim_id for c in window_prior]
            if protect_existing:
                staged.append((validated, result.latency_ms, prior_ids))
            else:
                self.semantic.save_window(validated, self.atomizer.model_name, result.latency_ms, prior_ids)
            prior.extend(validated.claims)
            report.windows += 1
            report.messages += len(validated.messages)
            report.claims += len(validated.claims)
            report.relations += len(validated.relations)
            report.failed_windows += validated.error is not None
            report.statuses.update(m.status for m in validated.messages)
            report.issues.update(f"{i.code}:{i.action}" for i in validated.issues)
            report.latency_ms += result.latency_ms
            usage = getattr(result, "usage", None) or {}
            report.model_calls += 1 if usage or result.prompt else 0
            report.prompt_tokens += usage.get("prompt_eval_count") or 0
            report.output_tokens += usage.get("eval_count") or 0
        if not protect_existing:
            return report
        if report.failed_windows or (not report.claims and set(report.statuses) - {STATUS_NO_DURABLE}):
            report.kept_existing = True
            return report
        raw_rows = self._raw_rows(conversation)
        with self.store.connection:
            self.store.write_conversation(conversation, *raw_rows)
            self.semantic.clear_conversation(cid)
            for validated, latency_ms, prior_ids in staged:
                self.semantic.write_window(validated, self.atomizer.model_name, latency_ms, prior_ids)
        return report

    def semantic_records(self, conversation_id: str) -> list[SemanticMessage]:
        return self.semantic.get_semantic_messages(conversation_id)
