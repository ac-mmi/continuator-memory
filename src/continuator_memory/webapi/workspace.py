"""Product Workspace: V2 memories over a conversation, plus retrieval and answering.

Product path: persisted (or fallback in-memory) consolidation → memory retrieval →
source evidence → ``ContextPack`` → optional Ollama answer. Embeddings are created on
first ``context`` / ``ask``. Extraction runs only on ``import_text`` (or offline ingest).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from continuator_memory.config import (
    Settings,
    legacy_import_db_path,
    resolve_product_db_path,
)
from continuator_memory.embedding import Embedder, EmbeddingLoadError
from continuator_memory.envfile import load_root_env
from continuator_memory.evidence.models import Document, EvidenceHit, EvidenceUnit
from continuator_memory.evidence.search import resolve_provenance, search_evidence
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from continuator_memory.v2.consolidation import Consolidator, consolidate
from continuator_memory.v2.context_store import ContextStore
from continuator_memory.v2.entity_store import EntityStore
from continuator_memory.v2.lexical import load_lexical_weight
from continuator_memory.v2.memory_retrieval import RetrievalConfig
from continuator_memory.v2.memory_store import MemoryStore, resolve_consolidator
from continuator_memory.v2.memory import CURRENT_STATUSES, STATUS_SUPERSEDED
from continuator_memory.v2.models import Claim, Relation
from continuator_memory.v2.relationship_store import RelationshipStore
from continuator_memory.v2.store import SemanticStore

from continuator_memory.generation import OllamaGenerator
from continuator_memory.webapi.answering import (
    MAX_ANSWER_TOP_K,
    GenerationSettings,
    build_answer_context,
    generate_answer,
    load_generation_settings,
    make_ollama_generator,
    parse_generation_overrides,
)
from continuator_memory.webapi.context_pack import (
    ContextPack,
    assemble_context_pack,
    context_pack_evidence_fields,
)
from continuator_memory.webapi.extract import ingest_imported
from continuator_memory.webapi.contextual_bm25 import (
    MODE_CONTEXTUAL_BM25,
    retrieve_contextual_bm25_evidence,
)
from continuator_memory.webapi.contextual_rerank import (
    MODE_CONTEXTUAL_LEXICAL_RERANK,
    retrieve_contextual_lexical_rerank_evidence,
)
from continuator_memory.webapi.contextual_union_rerank import (
    DEFAULT_CHANNEL_TOP_N,
    MODE_CONTEXTUAL_UNION_RERANK,
    retrieve_contextual_union_rerank_evidence,
)
from continuator_memory.webapi.contextual_wide_union_rerank import (
    DEFAULT_QUOTA_PER_CHANNEL,
    DEFAULT_REDUCED_TOP_N,
    DEFAULT_WIDE_CHANNEL_TOP_N,
    DEFAULT_WIDE_RRF_C,
    MODE_CONTEXTUAL_WIDE_UNION_QUOTA_RERANK,
    MODE_CONTEXTUAL_WIDE_UNION_RERANK,
    MODE_CONTEXTUAL_WIDE_UNION_RRF_RERANK,
    MODE_TO_REDUCER,
    WIDE_UNION_RERANK_MODES,
    retrieve_for_mode as retrieve_contextual_wide_union_for_mode,
)
from continuator_memory.webapi.hybrid_rrf import (
    HYBRID_RRF_MODES,
    retrieve_hybrid_rrf_evidence,
)
from continuator_memory.webapi.iterative import retrieve_iterative_evidence
from continuator_memory.webapi.multiquery import load_original_weight, retrieve_multiquery_evidence
from continuator_memory.webapi.query_plan import load_planner_model, plan_from_generator
from continuator_memory.webapi.sample import (
    SAMPLE_CONVERSATION_ID,
    sample_claims,
    sample_conversation,
    write_sample_classifications,
)
from continuator_memory.webapi.transcript import IMPORTED_CONVERSATION_ID, parse_transcript

from continuator_memory.v2.bprime_links import BPrimeConfig
from continuator_memory.v2.source_retrieval import SourceEvidence, retrieve_source_evidence

MAX_QUESTION_CHARS = 4000
MAX_K = MAX_ANSWER_TOP_K  # k is also the answer evidence count here, so it shares the answerer's ceiling
DEFAULT_SAMPLE_K = 1
DEFAULT_BUDGET_CHARS = 50_000
DEFAULT_DOCUMENT_TOP_K = 10
MEMORY_TYPES = frozenset({"person", "project", "fact", "preference", "event"})
_LOG = logging.getLogger(__name__)
# Display bucket for each V1 memory_type; the exact value is also returned as memoryType.
_V1_DISPLAY_TYPES = {
    "preference": "preference",
    "project": "project",
    "event": "event",
    "relationship": "person",
    "personal_fact": "fact",
    "goal": "fact",
    "decision": "fact",
    "procedural": "fact",
}
# Canonical product modes. ``hybrid`` is a deprecated alias of ``lexical`` (semantic +
# lexical overlap). It is not ``v2.hybrid_retrieval.HybridRetriever`` (raw∪memory RRF).
# ``lexical_semantic_rrf`` / ``lexical_multiquery_rrf`` are controlled RRF experiments.
# ``contextual_lexical`` is Option B: same scoring as ``lexical``, but ephemeral ±1
# same-session context is included in the indexed retrieval text only.
# ``contextual_lexical_rerank`` = contextual_lexical candidates + pretrained cross-encoder
# reorder (same candidate set; target-only source ids).
# ``contextual_bm25`` = true Okapi BM25 over the same contextual units (no dense score).
# ``contextual_union_rerank`` = lexical∪BM25 top-N union (target-id dedupe) + CE reorder.
# Wide-union experiments (depth 200/channel; optional reduction) are separate modes.
RETRIEVAL_MODES = frozenset({
    "baseline",
    "lexical",
    "contextual_lexical",
    "contextual_lexical_rerank",
    "contextual_bm25",
    "contextual_union_rerank",
    "contextual_wide_union_rerank",
    "contextual_wide_union_rrf_rerank",
    "contextual_wide_union_quota_rerank",
    "bprime",
    "multiquery",
    "iterative",
    "lexical_semantic_rrf",
    "lexical_multiquery_rrf",
})
RETRIEVAL_MODE_ALIASES = {"hybrid": "lexical"}
CONTEXTUAL_ENRICHMENT_MODES = frozenset({
    "contextual_lexical",
    "contextual_lexical_rerank",
    "contextual_bm25",
    "contextual_union_rerank",
    "contextual_wide_union_rerank",
    "contextual_wide_union_rrf_rerank",
    "contextual_wide_union_quota_rerank",
})


def normalize_retrieval_mode(mode: str) -> str:
    """Map a client mode string to a canonical product retrieval mode."""
    key = mode.strip().lower()
    canonical = RETRIEVAL_MODE_ALIASES.get(key, key)
    if canonical not in RETRIEVAL_MODES:
        raise ValueError(
            "retrieval_mode must be 'baseline', 'lexical' (alias 'hybrid'), "
            "'contextual_lexical', 'contextual_lexical_rerank', 'contextual_bm25', "
            "'contextual_union_rerank', 'contextual_wide_union_rerank', "
            "'contextual_wide_union_rrf_rerank', 'contextual_wide_union_quota_rerank', "
            "'bprime', 'multiquery', 'iterative', "
            "'lexical_semantic_rrf', or 'lexical_multiquery_rrf'."
        )
    return canonical


def _iso(value: str | None) -> str:
    if value:
        return value if "T" in value or value.endswith("Z") else f"{value}T00:00:00Z"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _title(text: str, limit: int = 48) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _memory_type(claim_type: str | None) -> str:
    if claim_type and claim_type in MEMORY_TYPES:
        return claim_type
    return "fact"


def _frontend_status(status: str) -> str:
    return "superseded" if status == STATUS_SUPERSEDED else ("current" if status in CURRENT_STATUSES else "superseded")


class Workspace:
    """In-process Continuator surface for the web UI."""

    def __init__(
        self,
        *,
        conversation: Conversation,
        claims: tuple[Claim, ...],
        relations: tuple[Relation, ...] = (),
        classifications_path: Path | None,
        embedder: Embedder | None = None,
        k: int = DEFAULT_SAMPLE_K,
        budget_chars: int = DEFAULT_BUDGET_CHARS,
        sample_id: str = "fixture",
        source_title: str | None = None,
        document_top_k: int = DEFAULT_DOCUMENT_TOP_K,
    ):
        if k < 1 or k > MAX_K:
            raise ValueError(f"k must be 1..{MAX_K}")
        if document_top_k < 1:
            raise ValueError("document_top_k must be >= 1")
        self.conversation = conversation
        self.claims = claims
        self._claims_by_id = {c.claim_id: c for c in claims}
        self.relations = relations
        self.classifications_path = Path(classifications_path) if classifications_path else None
        self._embedder = embedder
        self.k = k
        self.budget_chars = budget_chars
        self.document_top_k = document_top_k
        self.sample_id = sample_id
        self.source_title = source_title or conversation.title or conversation.conversation_id
        self.extraction = "fixture" if sample_id == "fixture" else "sqlite"
        self.imported_filename: str | None = None
        # Fixture / no-DB workspaces consolidate in RAM. SQLite workspaces prefer
        # MemoryStore snapshots (see from_sqlite / import_text).
        self.consolidator = consolidate(conversation.conversation_id, claims, relations)
        # Serializes product SQLite access and corpus in-memory mutations for this Workspace.
        # When a Store is attached, ``_db_lock`` is rebound to ``Store.db_lock`` (same object
        # if the Store was constructed with ``db_lock=self._db_lock``).
        self._db_lock = threading.RLock()
        # Serializes cold embedder construction only — never nest with DB work that
        # waits on this lock while holding ``_db_lock`` (avoid deadlock).
        self._embedder_init_lock = threading.Lock()
        self._importing = False
        self._generator = None
        self._planner = None
        self._evaluator = None
        self._generation = load_generation_settings()
        self._chats: dict[str, dict[str, Any]] = {}
        self._classifications: dict[str, dict[str, Any]] = {}
        self._evidence_store_cache: EvidenceStore | None = None
        self._corpus_deleted = False
        self._deleted_conversation_id: str | None = None
        if self.classifications_path and self.classifications_path.is_file():
            self._classifications = _load_classifications(self.classifications_path)

    def _attach_store(self, store: Store) -> None:
        """Bind the product Store and share its connection lock."""
        self._store = store
        self._db_lock = store.db_lock
        self._evidence_store_cache = None

    def _evidence_store(self) -> EvidenceStore | None:
        """Lazy EvidenceStore over the product SQLite connection, or None if no DB.

        Caller must already hold ``_db_lock`` when a product store is present.
        """
        store = getattr(self, "_store", None)
        if store is None:
            self._evidence_store_cache = None
            return None
        cached = self._evidence_store_cache
        if cached is not None and cached.connection is store.connection:
            return cached
        evidence_store = EvidenceStore(store.connection)
        self._evidence_store_cache = evidence_store
        return evidence_store

    def _search_document_evidence(self, query: str) -> tuple[tuple[EvidenceHit, ...], str]:
        """BM25 over EvidenceUnits when a product store exists; never raises to callers.

        Returns ``(hits, status)`` where status is ``skipped`` | ``bm25`` | ``error``.
        Caller must hold ``_db_lock`` when a product store is present.
        """
        evidence_store = self._evidence_store()
        if evidence_store is None:
            return (), "skipped"
        try:
            hits = search_evidence(
                evidence_store, query, top_k=self.document_top_k,
            )
            return tuple(hits), "bm25"
        except Exception:
            _LOG.exception("document evidence search failed")
            return (), "error"

    def _require_product_store(self) -> Store:
        store = getattr(self, "_store", None)
        if store is None:
            raise ValueError("a product SQLite store is required (fixture workspaces have none)")
        return store

    def _ensure_corpus_available(self) -> None:
        if self._corpus_deleted:
            raise ValueError(
                "corpus conversation has been deleted; import or reload a conversation "
                "before memory retrieval"
            )

    def _invalidate_corpus_state(self, conversation_id: str) -> None:
        """Clear in-memory conversation/memory state after corpus deletion; keep document DB.

        Caller must hold ``_db_lock``.
        """
        self._corpus_deleted = True
        self._deleted_conversation_id = conversation_id
        self.conversation = Conversation(conversation_id, ())
        self.claims = ()
        self.relations = ()
        self._claims_by_id = {}
        self.consolidator = Consolidator(conversation_id)
        self.extraction = "deleted"
        self.imported_filename = None
        self._chats.clear()

    def _ensure_corpus_schemas(self, connection) -> tuple[
        RelationshipStore, EntityStore, ContextStore, MemoryStore, SemanticStore
    ]:
        """Create conversation-scoped schemas outside any deletion transaction."""
        semantic = SemanticStore(connection)
        memories = MemoryStore(connection)
        relationships = RelationshipStore(connection)
        entities = EntityStore(connection)
        contexts = ContextStore(connection)
        return relationships, entities, contexts, memories, semantic

    @staticmethod
    def _document_summary(document: Document, *, unit_count: int) -> dict[str, Any]:
        return {
            "documentId": document.document_id,
            "sourceUri": document.source_uri,
            "sourceType": document.source_type,
            "title": document.title,
            "contentHash": document.content_hash,
            "createdAt": document.created_at,
            "metadata": dict(document.metadata),
            "unitCount": unit_count,
        }

    @staticmethod
    def _evidence_unit_json(unit: EvidenceUnit) -> dict[str, Any]:
        return {
            "evidenceId": unit.evidence_id,
            "documentId": unit.document_id,
            "ordinal": unit.ordinal,
            "text": unit.text,
            "retrievalText": unit.retrieval_text,
            "headingPath": list(unit.heading_path),
            "sectionId": unit.section_id,
            "charStart": unit.char_start,
            "charEnd": unit.char_end,
            "metadata": dict(unit.metadata),
        }

    def list_documents(self) -> list[dict[str, Any]]:
        with self._db_lock:
            self._require_product_store()
            evidence_store = self._evidence_store()
            assert evidence_store is not None
            return [
                self._document_summary(doc, unit_count=evidence_store.unit_count(doc.document_id))
                for doc in evidence_store.list_documents()
            ]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            self._require_product_store()
            evidence_store = self._evidence_store()
            assert evidence_store is not None
            document = evidence_store.get_document(document_id)
            if document is None:
                return None
            units = evidence_store.get_units(document_id)
            return {
                "document": self._document_summary(document, unit_count=len(units)),
                "units": [self._evidence_unit_json(u) for u in units],
            }

    def delete_document(self, document_id: str) -> dict[str, Any]:
        with self._db_lock:
            self._require_product_store()
            evidence_store = self._evidence_store()
            assert evidence_store is not None
            deleted, units_removed = evidence_store.delete_by_document_id(document_id)
            return {
                "ok": True,
                "deleted": deleted,
                "documentId": document_id,
                "unitsRemoved": units_removed,
            }

    def ingest_markdown_text(
        self,
        text: str,
        *,
        filename: str = "document.md",
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        """Ingest Markdown text into the product EvidenceStore (same DB as /documents).

        When ``source_uri`` is omitted, a content-addressed ``inline://md-…`` URI is used
        so filename alone cannot collide across different documents. Pass an explicit
        ``source_uri`` for stable identity across content edits (snapshot replace).
        """
        from continuator_memory.evidence.ingest import content_hash_for, ingest_markdown

        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")
        if source_uri is not None and (not isinstance(source_uri, str) or not source_uri.strip()):
            raise ValueError("sourceUri must be a non-empty string when provided")

        name = filename.strip()
        uri = source_uri.strip() if source_uri else f"inline://md-{content_hash_for(text)[:24]}"

        with self._db_lock:
            self._require_product_store()
            evidence_store = self._evidence_store()
            assert evidence_store is not None
            ingested = ingest_markdown(
                name,
                evidence_store,
                text=text,
                source_uri=uri,
            )
            summary = self._document_summary(
                ingested.document, unit_count=len(ingested.units),
            )
            return {
                "ok": True,
                "replaced": ingested.replaced,
                "document": summary,
            }

    def get_evidence_unit(self, evidence_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            self._require_product_store()
            evidence_store = self._evidence_store()
            assert evidence_store is not None
            resolved = resolve_provenance(evidence_store, evidence_id)
            if resolved is None:
                return None
            unit, document = resolved
            return {
                "evidence": self._evidence_unit_json(unit),
                "document": self._document_summary(
                    document, unit_count=evidence_store.unit_count(document.document_id),
                ),
            }

    def delete_corpus_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Delete the active Workspace corpus conversation from its product SQLite store.

        One atomic SQLite transaction covers all conversation-scoped cleanup and the
        conversation row. Preserves ``ev_documents`` / ``ev_units``. Idempotent when
        already absent. In-memory corpus state is invalidated only after commit.
        """
        with self._db_lock:
            store = self._require_product_store()
            if conversation_id != self.conversation.conversation_id and not (
                self._corpus_deleted and conversation_id == self._deleted_conversation_id
            ):
                raise ValueError(
                    "conversation_id must match the active Workspace corpus conversation"
                )
            if self._corpus_deleted and conversation_id == self._deleted_conversation_id:
                return {
                    "ok": True,
                    "deleted": False,
                    "conversationId": conversation_id,
                }

            conn = store.connection
            conn.execute("PRAGMA foreign_keys = ON")
            # Schema DDL (executescript) commits; run before BEGIN.
            relationships, entities, contexts, memories, semantic = self._ensure_corpus_schemas(conn)

            existing = store.get_conversation(conversation_id)
            if existing is None:
                self._invalidate_corpus_state(conversation_id)
                return {
                    "ok": True,
                    "deleted": False,
                    "conversationId": conversation_id,
                }

            fail_after = getattr(self, "_deletion_fail_after", None)
            try:
                # Schema DDL / lookups may leave an implicit transaction open.
                if conn.in_transaction:
                    conn.commit()
                conn.execute("BEGIN")
                relationships.clear_conversation(conversation_id)
                if fail_after == "relationships":
                    raise RuntimeError("injected failure after relationship cleanup")
                entities.clear_conversation(conversation_id)
                if fail_after == "entities":
                    raise RuntimeError("injected failure after entity cleanup")
                contexts.clear_conversation(conversation_id)
                memories.clear_conversation(conversation_id)
                if fail_after == "memories":
                    raise RuntimeError("injected failure after memory cleanup")
                semantic.clear_conversation(conversation_id)
                _clear_optional_conversation_tables(conn, conversation_id)
                if fail_after == "optional":
                    raise RuntimeError("injected failure during optional-table cleanup")
                if fail_after == "conversation":
                    raise RuntimeError("injected failure during conversation DELETE")
                conn.execute(
                    "DELETE FROM conversations WHERE conversation_id = ?",
                    (conversation_id,),
                )
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    _LOG.exception("rollback after corpus deletion failure also failed")
                _LOG.exception(
                    "atomic corpus conversation deletion failed for %s", conversation_id,
                )
                raise

            self._invalidate_corpus_state(conversation_id)
            return {
                "ok": True,
                "deleted": True,
                "conversationId": conversation_id,
            }

    def _set_consolidator_from_store(self, store: Store, *, persist_if_missing: bool = True) -> None:
        self.consolidator = resolve_consolidator(
            store.connection,
            self.conversation.conversation_id,
            self.claims,
            self.relations,
            persist_if_missing=persist_if_missing,
        )

    @classmethod
    def from_sample(cls, embedder: Embedder | None = None, k: int = DEFAULT_SAMPLE_K) -> Workspace:
        conversation = sample_conversation()
        claims = sample_claims(conversation)
        tmp = Path(tempfile.mkdtemp(prefix="continuator-web-")) / "classifications.jsonl"
        write_sample_classifications(tmp)
        return cls(
            conversation=conversation,
            claims=claims,
            classifications_path=tmp,
            embedder=embedder,
            k=k,
            sample_id="fixture",
        )

    @classmethod
    def from_sqlite(
        cls,
        db_path: Path,
        *,
        conversation_id: str | None = None,
        classifications_path: Path | None = None,
        embedder: Embedder | None = None,
        k: int | None = None,
        budget_chars: int = DEFAULT_BUDGET_CHARS,
    ) -> Workspace:
        settings = Settings.from_env()
        store = Store(str(db_path))
        try:
            cid = _choose_conversation_id(store, conversation_id)
            conversation = store.get_conversation(cid)
            if conversation is None:
                raise FileNotFoundError(f"conversation {cid!r} not in {db_path}")
            semantic = SemanticStore(store.connection)
            claims = tuple(semantic.get_claims(cid))
            relations = tuple(semantic.get_relations(cid))
        except Exception:
            store.close()
            raise
        imported = cid == IMPORTED_CONVERSATION_ID
        if imported and classifications_path is None:
            empty_links = Path(tempfile.mkdtemp(prefix="continuator-import-")) / "classifications.jsonl"
            empty_links.write_text("", encoding="utf-8")
            classifications_path = empty_links
        workspace = cls(
            conversation=conversation,
            claims=claims,
            relations=relations,
            classifications_path=classifications_path,
            embedder=embedder,
            k=k if k is not None else settings.top_k,
            budget_chars=budget_chars,
            sample_id="imported" if imported else "sqlite",
            source_title=conversation.title or cid,
        )
        workspace._attach_store(store)
        with workspace._db_lock:
            workspace._set_consolidator_from_store(store, persist_if_missing=True)
        workspace.extraction = "message_claims" if imported else "sqlite"
        if imported:
            workspace.imported_filename = _imported_filename(conversation.title)
        return workspace

    def import_text(
        self,
        text: str,
        *,
        filename: str = "conversation.txt",
        extract: str = "messages",
        atomizer=None,
        db_path: Path | None = None,
    ) -> dict[str, Any]:
        if extract not in {"v2", "messages", "memories"}:
            raise ValueError(
                "extract must be 'v2' (LLM atomizer), 'memories' (durable memory extraction) "
                "or 'messages' (one memory per message). Backend defaults to Ollama; "
                "override with CONTINUATOR_V2_EXTRACTOR or pass atomizer=..."
            )
        with self._db_lock:
            if self._importing:
                raise ValueError("an import is already running")
            self._importing = True
        try:
            conversation = parse_transcript(text, filename=filename)
            store_path = resolve_product_db_path(explicit=db_path)
            # Share Workspace lock so ingest + attach cannot race other product connection users.
            store = Store(str(store_path), db_lock=self._db_lock)
            attached = False
            try:
                with self._db_lock:
                    report = ingest_imported(
                        conversation, store, extract=extract, atomizer=atomizer,
                    )
                    if report.kept_existing:
                        raise ValueError(
                            f"extraction did not complete ({report.failed_windows} of {report.windows} windows failed, "
                            f"{report.claims} memories produced). The previously stored conversation and memories were kept."
                        )
                    semantic = SemanticStore(store.connection)
                    claims = tuple(semantic.get_claims(conversation.conversation_id))
                    relations = tuple(semantic.get_relations(conversation.conversation_id))
                    if not claims:
                        raise ValueError(
                            "import stored the conversation but extraction produced no memories "
                            f"({report.failed_windows} of {report.windows} windows failed; if windows failed, "
                            "check that Ollama is running and the configured model is available). "
                            "The workspace still shows the previous conversation until it is reloaded."
                        )
                    empty_links = Path(tempfile.mkdtemp(prefix="continuator-import-")) / "classifications.jsonl"
                    empty_links.write_text("", encoding="utf-8")
                    settings = Settings.from_env()
                    old_store = getattr(self, "_store", None)
                    self.conversation = conversation
                    self.claims = claims
                    self._claims_by_id = {c.claim_id: c for c in claims}
                    self.relations = relations
                    self.classifications_path = empty_links
                    self._classifications = {}
                    self.sample_id = "imported"
                    self.extraction = {
                        "v2": "v2_atomizer", "memories": "memory_extraction",
                    }.get(extract, "message_claims")
                    self.imported_filename = Path(filename).name
                    self.source_title = conversation.title or f"Imported · {self.imported_filename}"
                    self.k = settings.top_k if self.k == DEFAULT_SAMPLE_K else self.k
                    self._attach_store(store)
                    attached = True
                    self._corpus_deleted = False
                    self._deleted_conversation_id = None
                    # Ingest already wrote the MemoryStore snapshot; load it (fallback consolidates).
                    self._set_consolidator_from_store(store, persist_if_missing=True)
                    self._chats.clear()
                    if old_store is not None and old_store is not store:
                        try:
                            old_store.close()
                        except Exception:
                            pass
                    # health() re-enters _db_lock (RLock).
                    health = self.health()
                    source = self._source_summary()
                    imported_filename = self.imported_filename
                    extraction = self.extraction
            except Exception:
                if not attached:
                    try:
                        store.close()
                    except Exception:
                        pass
                raise
            return {
                "ok": True,
                "source": source,
                "report": {
                    "filename": imported_filename,
                    "nMessages": len(conversation.messages),
                    "nClaims": len(claims),
                    "nMemories": health["nMemories"],
                    "nRelations": len(relations),
                    "failedWindows": report.failed_windows,
                    "windows": report.windows,
                    "extract": extraction,
                    "bprimeLinks": 0,
                },
                "health": health,
                "note": (
                    "B′ retrieval is available but has no classified links for this import, "
                    "so expansion will match baseline until a classifications JSONL is supplied."
                ),
            }
        finally:
            with self._db_lock:
                self._importing = False

    def embedder(self) -> Embedder:
        """Return the shared embedder, lazily constructing it once.

        Construction is guarded by ``_embedder_init_lock`` and is safe to call
        **outside** ``_db_lock`` so cold model load does not block other API work.
        Callers that already hold ``_db_lock`` must only hit the fast path (already
        initialized) or risk holding the DB lock during load — prefer warming via
        ``ensure_embedder()`` before acquiring ``_db_lock``.
        """
        cached = self._embedder
        if cached is not None:
            return cached
        with self._embedder_init_lock:
            if self._embedder is not None:
                return self._embedder
            from continuator_memory.embedding import SentenceTransformerEmbedder

            model_name = Settings.from_env().embedding_model
            try:
                self._embedder = SentenceTransformerEmbedder(model_name)
            except EmbeddingLoadError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                from continuator_memory.embedding import classify_embedding_load_failure

                raise classify_embedding_load_failure(model_name, exc) from exc
            return self._embedder

    def ensure_embedder(self) -> Embedder:
        """Warm the embedder outside the DB lock (thread-safe lazy init)."""
        return self.embedder()

    def health(self) -> dict[str, Any]:
        with self._db_lock:
            payload = {
                "ok": True,
                "sampleId": self.sample_id,
                "conversationId": self.conversation.conversation_id,
                "nMemories": len(self.consolidator.memories),
                "nClaims": len(self.claims),
                "nMessages": len(self.conversation.messages),
                "k": self.k,
                "budgetChars": self.budget_chars,
                "bprimeClassifications": str(self.classifications_path) if self.classifications_path else None,
                "answerGeneration": (
                    "disabled" if not self._generation.enabled
                    else ("configured" if self._generation.configured else "not_configured")
                ),
                "generation": self._generation.public_dict(),
                "origin": self.sample_id,
                "extraction": self.extraction,
                "importedFilename": self.imported_filename,
                "corpusDeleted": self._corpus_deleted,
            }
            if self._corpus_deleted:
                payload["deletedConversationId"] = self._deleted_conversation_id
            return payload

    def list_memories(self, *, text: str = "", types: list[str] | None = None,
                      include_superseded: bool = True) -> list[dict[str, Any]]:
        with self._db_lock:
            self._ensure_corpus_available()
            needle = text.strip().lower()
            out = []
            for mem in self.consolidator.memories.values():
                row = self._memory_json(mem)
                if not include_superseded and row["status"] != "current":
                    continue
                if types and row["type"] not in types:
                    continue
                blob = f"{row['title']} {row['statement']} {row['subject']}".lower()
                if needle and needle not in blob:
                    continue
                out.append(row)
            out.sort(key=lambda m: (m["updatedAt"], m["id"]), reverse=True)
            return out

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            self._ensure_corpus_available()
            mem = self.consolidator.memories.get(memory_id)
            if mem is None:
                return None
            related = []
            for link in self.consolidator.links:
                other = None
                if link.newer_memory_id == memory_id:
                    other = self.consolidator.memories.get(link.older_memory_id)
                elif link.older_memory_id == memory_id:
                    other = self.consolidator.memories.get(link.newer_memory_id)
                if other is None:
                    continue
                related.append({
                    "memory": self._memory_json(other),
                    "relationship": {
                        "id": f"{link.type}:{link.newer_memory_id}:{link.older_memory_id}",
                        "fromId": link.newer_memory_id,
                        "toId": link.older_memory_id,
                        "label": link.type,
                        "status": "current",
                    },
                })
            evidence = []
            for src in self.consolidator.sources_of(memory_id):
                message = next((m for m in self.conversation.messages if m.message_id == src.message_id), None)
                if message is None:
                    continue
                evidence.append({
                    "source": self._source_summary(),
                    "message": _message_json(message),
                })
            return {
                "memory": self._memory_json(mem),
                "related": related,
                "evidence": evidence,
                "supersedes": None,
                "supersededBy": self._memory_json(self.consolidator.memories[mem.superseded_by])
                if mem.superseded_by and mem.superseded_by in self.consolidator.memories
                else None,
            }

    def list_sources(self) -> list[dict[str, Any]]:
        with self._db_lock:
            self._ensure_corpus_available()
            return [self._source_summary()]

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            self._ensure_corpus_available()
            if source_id != self.conversation.conversation_id:
                return None
            memories = []
            for mem in self.consolidator.memories.values():
                mids = [s.message_id for s in self.consolidator.sources_of(mem.memory_id)]
                if mids:
                    memories.append({"memory": self._memory_json(mem), "messageIds": mids})
            return {
                "source": {
                    "id": self.conversation.conversation_id,
                    "title": self.source_title,
                    "kind": "conversation",
                    "createdAt": _iso(self.conversation.messages[0].timestamp if self.conversation.messages else None),
                    "messages": [_message_json(m) for m in self.conversation.messages],
                },
                "memories": memories,
            }

    def get_graph(self) -> dict[str, Any]:
        with self._db_lock:
            self._ensure_corpus_available()
            relationships = [
                {
                    "id": f"{link.type}:{link.newer_memory_id}:{link.older_memory_id}",
                    "fromId": link.newer_memory_id,
                    "toId": link.older_memory_id,
                    "label": link.type,
                    "status": "current",
                }
                for link in self.consolidator.links
            ]
            return {
                "memories": [self._memory_json(m) for m in self.consolidator.memories.values()],
                "relationships": relationships,
            }

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._db_lock:
            rows = list(self._chats.values())
        return [
            {
                "id": c["id"],
                "title": c["title"],
                "updatedAt": c["updatedAt"],
                "messageCount": len(c["messages"]),
            }
            for c in sorted(rows, key=lambda c: c["updatedAt"], reverse=True)
        ]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._db_lock:
            chat = self._chats.get(conversation_id)
            return dict(chat) if chat else None

    def set_generator(self, generator) -> None:
        self._generator = generator

    def set_planner(self, planner) -> None:
        self._planner = planner

    def set_evaluator(self, evaluator) -> None:
        self._evaluator = evaluator

    def _resolve_generator(self, settings: GenerationSettings):
        current = self._generator
        if isinstance(current, OllamaGenerator) and (current.model_name != settings.model
                                                     or current.num_ctx != settings.num_ctx):
            current = None
        if current is None and settings.configured:
            current = make_ollama_generator(settings)
            self._generator = current
        return current

    def _resolve_planner(self, settings: GenerationSettings):
        if self._planner is not None:
            return self._planner
        model = load_planner_model() or settings.model
        if not model.strip():
            return None
        generator = self._resolve_generator(
            GenerationSettings(
                enabled=settings.enabled,
                model=model,
                temperature=0.0,
                num_predict=settings.num_predict,
                max_context_chars=settings.max_context_chars,
                provider=settings.provider,
            )
        )
        if generator is None:
            return None
        return lambda question: plan_from_generator(question, generator)

    def _resolve_evaluator(self, settings: GenerationSettings):
        if self._evaluator is not None:
            return self._evaluator
        generator = self._resolve_generator(settings)
        if generator is None:
            return None
        return generator

    def _validate_question(self, question: str) -> str:
        text = question.strip()
        if not text:
            raise ValueError("Question is empty.")
        if len(text) > MAX_QUESTION_CHARS:
            raise ValueError(f"Question exceeds {MAX_QUESTION_CHARS} characters.")
        return text

    def _retrieve_evidence(
        self,
        question: str,
        mode: str,
        *,
        settings: GenerationSettings,
        k: int | None = None,
    ) -> tuple[SourceEvidence, dict[str, Any]]:
        """Canonical product retrieval: memories → source evidence (no answering LLM)."""
        if mode == "bprime" and not (self.classifications_path and self.classifications_path.is_file()):
            raise ValueError(
                "B′ retrieval needs a classifications JSONL file "
                "(set CONTINUATOR_BPRIME_CLASSIFICATIONS or use the built-in sample)."
            )
        top_k = self.k if k is None else k
        if top_k < 1 or top_k > MAX_K:
            raise ValueError(f"k must be 1..{MAX_K}")
        query_diagnostics: dict[str, Any] = {}
        original_weight = load_original_weight()
        consolidator = self.consolidator
        if mode == "multiquery":
            evidence, query_diagnostics = retrieve_multiquery_evidence(
                question,
                self.conversation,
                self.claims,
                self.relations,
                self.embedder(),
                top_k,
                self.budget_chars,
                planner=self._resolve_planner(settings),
                original_weight=original_weight,
                consolidator=consolidator,
            )
        elif mode in HYBRID_RRF_MODES:
            hybrid_kwargs: dict[str, Any] = {"consolidator": consolidator}
            if mode == "lexical_multiquery_rrf":
                hybrid_kwargs["planner"] = self._resolve_planner(settings)
                hybrid_kwargs["original_weight"] = original_weight
            evidence, query_diagnostics = retrieve_hybrid_rrf_evidence(
                mode,
                question,
                self.conversation,
                self.claims,
                self.relations,
                self.embedder(),
                top_k,
                self.budget_chars,
                **hybrid_kwargs,
            )
        elif mode == "iterative":
            evidence, query_diagnostics = retrieve_iterative_evidence(
                question,
                self.conversation,
                self.claims,
                self.relations,
                self.embedder(),
                top_k,
                self.budget_chars,
                planner=self._resolve_planner(settings),
                evaluator=self._resolve_evaluator(settings),
                original_weight=original_weight,
                consolidator=consolidator,
            )
        else:
            bprime = None
            if mode == "bprime":
                bprime = BPrimeConfig(
                    enabled=True,
                    classifications_path=self.classifications_path,
                    max_hops=1,
                )
            if mode == MODE_CONTEXTUAL_LEXICAL_RERANK:
                evidence, query_diagnostics = retrieve_contextual_lexical_rerank_evidence(
                    question,
                    self.conversation,
                    self.claims,
                    self.relations,
                    self.embedder(),
                    top_k,
                    self.budget_chars,
                    consolidator=consolidator,
                    scorer=getattr(self, "_eval_reranker", None),
                )
            elif mode == MODE_CONTEXTUAL_UNION_RERANK:
                from continuator_memory.webapi.contextual_bm25 import build_contextual_bm25_index

                cached_bm25 = getattr(self, "_eval_bm25_index", None)
                if cached_bm25 is None:
                    self._eval_bm25_index, _ = build_contextual_bm25_index(
                        self.conversation,
                        self.claims,
                        self.relations,
                        consolidator=consolidator,
                    )
                    cached_bm25 = self._eval_bm25_index
                evidence, query_diagnostics = retrieve_contextual_union_rerank_evidence(
                    question,
                    self.conversation,
                    self.claims,
                    self.relations,
                    self.embedder(),
                    self.budget_chars,
                    consolidator=consolidator,
                    scorer=getattr(self, "_eval_reranker", None),
                    channel_top_n=DEFAULT_CHANNEL_TOP_N,
                    bm25_index=cached_bm25,
                )
            elif mode in WIDE_UNION_RERANK_MODES:
                from continuator_memory.webapi.contextual_bm25 import build_contextual_bm25_index

                cached_bm25 = getattr(self, "_eval_bm25_index", None)
                if cached_bm25 is None:
                    self._eval_bm25_index, _ = build_contextual_bm25_index(
                        self.conversation,
                        self.claims,
                        self.relations,
                        consolidator=consolidator,
                    )
                    cached_bm25 = self._eval_bm25_index
                evidence, query_diagnostics = retrieve_contextual_wide_union_for_mode(
                    mode,
                    question,
                    self.conversation,
                    self.claims,
                    self.relations,
                    self.embedder(),
                    self.budget_chars,
                    consolidator=consolidator,
                    scorer=getattr(self, "_eval_reranker", None),
                    channel_top_n=DEFAULT_WIDE_CHANNEL_TOP_N,
                    reduced_top_n=DEFAULT_REDUCED_TOP_N,
                    rrf_c=DEFAULT_WIDE_RRF_C,
                    quota_per_channel=DEFAULT_QUOTA_PER_CHANNEL,
                    bm25_index=cached_bm25,
                )
            elif mode == MODE_CONTEXTUAL_BM25:
                from continuator_memory.webapi.contextual_bm25 import build_contextual_bm25_index

                cached = getattr(self, "_eval_bm25_index", None)
                if cached is None:
                    self._eval_bm25_index, _ = build_contextual_bm25_index(
                        self.conversation,
                        self.claims,
                        self.relations,
                        consolidator=consolidator,
                    )
                    cached = self._eval_bm25_index
                evidence, query_diagnostics = retrieve_contextual_bm25_evidence(
                    question,
                    self.conversation,
                    self.claims,
                    self.relations,
                    top_k,
                    self.budget_chars,
                    consolidator=consolidator,
                    index=cached,
                )
            else:
                retrieval_config = None
                contextual_enrichment = False
                if mode == "lexical":
                    retrieval_config = RetrievalConfig(lexical_weight=load_lexical_weight())
                elif mode == "contextual_lexical":
                    # Same lexical scoring as product lexical; context_weight forced off.
                    retrieval_config = RetrievalConfig(
                        lexical_weight=load_lexical_weight(),
                        context_weight=0.0,
                    )
                    contextual_enrichment = True
                evidence = retrieve_source_evidence(
                    question,
                    self.conversation,
                    self.claims,
                    self.relations,
                    self.embedder(),
                    top_k,
                    self.budget_chars,
                    config=retrieval_config,
                    bprime=bprime,
                    consolidator=consolidator,
                    contextual_enrichment=contextual_enrichment,
                )
        return evidence, query_diagnostics

    def _bprime_hops(self, evidence: SourceEvidence) -> list[dict[str, Any]]:
        hops = []
        for hop in evidence.bprime.get("applied_hops") or []:
            pair_id = hop.get("pair_id")
            row = self._classifications.get(pair_id, {})
            hops.append({
                "pairId": pair_id,
                "seedMemoryId": hop.get("seed_memory_id"),
                "neighbourMemoryId": hop.get("neighbour_memory_id"),
                "relation": hop.get("relation"),
                "direction": hop.get("direction"),
                "hopDepth": hop.get("hop_depth"),
                "rationale": row.get("rationale"),
                "resolvedMessageIds": list(hop.get("resolved_message_ids") or []),
            })
        return hops

    def context(
        self,
        question: str,
        *,
        retrieval_mode: str = "baseline",
        generation: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> ContextPack:
        """Retrieve memories and source evidence into a Context Pack (no answering LLM).

        Multiquery / iterative modes may still call a planner/evaluator model for retrieval
        only; they do not generate a final answer.
        """
        # Validate and warm the embedder before taking the DB lock so cold MiniLM
        # download/load does not stall concurrent health/import/document work.
        text = self._validate_question(question)
        mode = normalize_retrieval_mode(retrieval_mode)
        settings = parse_generation_overrides(generation, self._generation)
        top_k = self.k if k is None else k
        self.ensure_embedder()
        with self._db_lock:
            self._ensure_corpus_available()
            evidence, query_diagnostics = self._retrieve_evidence(
                text, mode, settings=settings, k=top_k,
            )
            document_hits, document_search = self._search_document_evidence(text)

            seed_ids = [m.memory_id for m in evidence.matches[: evidence.n_seed_matches]]
            expanded_ids = list(evidence.bprime.get("expanded_memory_ids") or [])
            hops = self._bprime_hops(evidence)
            answer_ctx = build_answer_context(
                text,
                evidence,
                self.conversation,
                seed_ids=seed_ids,
                expanded_ids=expanded_ids,
                max_context_chars=settings.max_context_chars,
                context_tokens=settings.num_ctx,
                output_tokens=settings.num_predict,
            )
            selected = list(evidence.selection.selected)
            diagnostics: dict[str, Any] = {
                "mode": mode,
                "k": top_k,
                "seedCount": evidence.n_seed_matches,
                "expandedCount": len(expanded_ids),
                "sourceCount": len(evidence.selection.unique_sources),
                "selectedSourceCount": len(selected),
                "droppedSourceCount": len(evidence.selection.dropped),
                "budgetChars": evidence.selection.budget_chars,
                "usedChars": evidence.selection.used_chars,
                "promptMemoryCount": len(answer_ctx.prompt_memory_ids),
                "retrievedMemoryCount": len(answer_ctx.retrieved_memory_ids),
                "answerGeneration": "not_run",
                "documentSearch": document_search,
                "documentEvidenceCount": len(document_hits),
                "documentTopK": self.document_top_k,
            }
            plan = query_diagnostics.get("queryPlan") or {}
            if mode in {"multiquery", "lexical_multiquery_rrf"} and plan.get("queries"):
                diagnostics["queryCount"] = len(plan["queries"])
                diagnostics["planStatus"] = plan.get("status")
            if query_diagnostics.get("retrievalComparison", {}).get("latencyMs"):
                diagnostics["latencyMs"] = query_diagnostics["retrievalComparison"]["latencyMs"]
            if mode == "iterative":
                diagnostics["roundCount"] = query_diagnostics.get("roundCount")
                diagnostics["stopReason"] = query_diagnostics.get("stopReason")
            if (
                mode in {
                    "lexical",
                    "contextual_lexical",
                    MODE_CONTEXTUAL_LEXICAL_RERANK,
                    MODE_CONTEXTUAL_UNION_RERANK,
                    *WIDE_UNION_RERANK_MODES,
                }
                or mode in HYBRID_RRF_MODES
            ):
                diagnostics["lexicalWeight"] = load_lexical_weight()
            if mode in CONTEXTUAL_ENRICHMENT_MODES:
                diagnostics["contextualEnrichment"] = True
                diagnostics["contextWeight"] = 0.0
            if mode == MODE_CONTEXTUAL_BM25:
                diagnostics["sparseRetrieval"] = "bm25_okapi"
                diagnostics["denseRetrieval"] = False
                if query_diagnostics.get("bm25"):
                    diagnostics["bm25"] = {
                        k: v for k, v in query_diagnostics["bm25"].items()
                        if k in {"algorithm", "k1", "b", "nDocuments", "tokenizer", "representation"}
                    }
            if mode == MODE_CONTEXTUAL_UNION_RERANK:
                diagnostics["candidateUnion"] = True
                diagnostics["channelTopN"] = DEFAULT_CHANNEL_TOP_N
                if query_diagnostics.get("union"):
                    diagnostics["union"] = dict(query_diagnostics["union"])
            if mode in WIDE_UNION_RERANK_MODES:
                diagnostics["candidateUnion"] = True
                diagnostics["channelTopN"] = DEFAULT_WIDE_CHANNEL_TOP_N
                diagnostics["wideUnionReducer"] = MODE_TO_REDUCER[mode]
                if query_diagnostics.get("phase1"):
                    diagnostics["phase1"] = dict(query_diagnostics["phase1"])
                if query_diagnostics.get("union"):
                    diagnostics["union"] = dict(query_diagnostics["union"])
                if query_diagnostics.get("reduction"):
                    reduction = dict(query_diagnostics["reduction"])
                    # Compact diagnostics: keep counts, drop bulky id lists.
                    reduction.pop("droppedMemoryIds", None)
                    diagnostics["reduction"] = reduction
            if mode in {
                MODE_CONTEXTUAL_LEXICAL_RERANK,
                MODE_CONTEXTUAL_UNION_RERANK,
                *WIDE_UNION_RERANK_MODES,
            }:
                diagnostics["rerank"] = (query_diagnostics.get("rerank") or {})
                # Drop bulky per-candidate score rows from the compact diagnostics surface.
                if isinstance(diagnostics["rerank"], dict):
                    diagnostics["rerank"] = {
                        k: v for k, v in diagnostics["rerank"].items() if k != "scores"
                    }
            if mode in HYBRID_RRF_MODES:
                diagnostics["rrfC"] = query_diagnostics.get("rrfC")
                if query_diagnostics.get("lexicalQueryCount") is not None:
                    diagnostics["lexicalQueryCount"] = query_diagnostics["lexicalQueryCount"]
                if query_diagnostics.get("semanticQueryCount") is not None:
                    diagnostics["semanticQueryCount"] = query_diagnostics["semanticQueryCount"]
            return assemble_context_pack(
                query=text,
                mode=mode,
                evidence=evidence,
                answer_context=answer_ctx,
                conversation_id=self.conversation.conversation_id,
                diagnostics=diagnostics,
                query_diagnostics=query_diagnostics,
                seed_memory_ids=seed_ids,
                expanded_memory_ids=expanded_ids,
                hops=hops,
                document_hits=document_hits,
            )

    def ask(self, question: str, *, conversation_id: str | None = None,
            retrieval_mode: str = "baseline", generate: bool | None = None,
            generation: dict[str, Any] | None = None) -> dict[str, Any]:
        settings = parse_generation_overrides(generation, self._generation)
        pack = self.context(question, retrieval_mode=retrieval_mode, generation=generation)
        should_generate = self._generation.enabled if generate is None else bool(generate)
        payload = self._ask_payload_from_pack(pack, settings=settings, generate=should_generate)
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        with self._db_lock:
            chat = self._chats.get(conversation_id) if conversation_id else None
            if chat is None:
                chat = {
                    "id": f"chat-{uuid.uuid4().hex[:10]}",
                    "title": _title(pack.query),
                    "updatedAt": now,
                    "messages": [],
                }
                self._chats[chat["id"]] = chat
            user_msg = {
                "id": f"{chat['id']}-q{len(chat['messages']) + 1}",
                "role": "user",
                "content": pack.query,
                "createdAt": now,
            }
            answer_msg = {
                "id": f"{chat['id']}-a{len(chat['messages']) + 2}",
                "role": "assistant",
                "content": payload["answerText"],
                "createdAt": now,
                "evidence": payload["evidence"],
            }
            chat["messages"] = [*chat["messages"], user_msg, answer_msg]
            chat["updatedAt"] = now
            conversation = dict(chat)
        return {"conversation": conversation, "answer": answer_msg}

    def _ask_payload_from_pack(
        self,
        pack: ContextPack,
        *,
        settings: GenerationSettings,
        generate: bool,
    ) -> dict[str, Any]:
        selected = list(pack.evidence.selection.selected)
        by_msg = {m.message_id: m for m in self.conversation.messages}
        selected_messages = [_message_json(by_msg[mid]) for mid in selected if mid in by_msg]
        diagnostics = dict(pack.diagnostics)
        if generate:
            answer_text, generation_meta = generate_answer(
                pack.answer_context, settings, self._resolve_generator(settings),
            )
            gen_status = generation_meta.get("status", "ok")
        else:
            answer_text = _format_evidence_answer(
                pack.query, pack.mode,
                {
                    "k": diagnostics.get("k", self.k),
                    "seedCount": pack.evidence.n_seed_matches,
                    "expandedCount": len(pack.expanded_memory_ids),
                    "selectedSourceCount": len(selected),
                    "sourceCount": len(pack.evidence.selection.unique_sources),
                    "usedChars": pack.evidence.selection.used_chars,
                    "budgetChars": pack.evidence.selection.budget_chars,
                },
                pack.evidence, list(pack.hops), self.consolidator,
                query_plan=(pack.query_diagnostics.get("queryPlan")),
            )
            generation_meta = {**settings.public_dict(), "status": "not_run"}
            gen_status = "not_run"
        diagnostics["answerGeneration"] = gen_status
        updated = assemble_context_pack(
            query=pack.query,
            mode=pack.mode,
            evidence=pack.evidence,
            answer_context=pack.answer_context,
            conversation_id=pack.conversation_id,
            diagnostics=diagnostics,
            query_diagnostics=pack.query_diagnostics,
            seed_memory_ids=pack.seed_memory_ids,
            expanded_memory_ids=pack.expanded_memory_ids,
            hops=pack.hops,
            document_hits=pack.document_hits,
        )
        evidence = context_pack_evidence_fields(updated)
        evidence["selectedMessages"] = selected_messages
        evidence["generation"] = generation_meta
        return {"answerText": answer_text, "evidence": evidence}

    def _memory_json(self, mem) -> dict[str, Any]:
        sources = [
            {"sourceId": self.conversation.conversation_id, "messageId": s.message_id}
            for s in self.consolidator.sources_of(mem.memory_id)
        ]
        created = _iso(mem.created_at or mem.valid_from)
        v1_type = self._v1_memory_type(mem)
        return {
            "id": mem.memory_id,
            "title": _title(mem.content, 40),
            "statement": mem.content,
            "type": _V1_DISPLAY_TYPES.get(v1_type, "fact") if v1_type else _memory_type(mem.claim_type),
            "memoryType": v1_type,
            "status": _frontend_status(mem.status),
            "subject": mem.subject or "",
            "createdAt": created,
            "updatedAt": created,
            "validFrom": mem.valid_from,
            "validUntil": mem.valid_until,
            "supersedesId": None,
            "supersededById": mem.superseded_by,
            "sources": sources,
            "history": [],
        }

    def _v1_memory_type(self, mem) -> str | None:
        """V1 extraction keeps memory_type in claim metadata, same lookup order as memory_records."""
        claim_ids = [mem.claim_id, *(s.claim_id for s in self.consolidator.sources_of(mem.memory_id))]
        for cid in claim_ids:
            claim = self._claims_by_id.get(cid)
            if claim and claim.metadata.get("memory_type"):
                return claim.metadata["memory_type"]
        return None

    def _source_summary(self) -> dict[str, Any]:
        preview = self.conversation.messages[0].content if self.conversation.messages else ""
        n_mem = sum(1 for _ in self.consolidator.memories)
        return {
            "id": self.conversation.conversation_id,
            "title": self.source_title,
            "kind": "conversation",
            "createdAt": _iso(self.conversation.messages[0].timestamp if self.conversation.messages else None),
            "preview": _title(preview, 120),
            "messageCount": len(self.conversation.messages),
            "memoryCount": n_mem,
            "origin": self.sample_id,
        }


def _message_json(message: Message) -> dict[str, Any]:
    role = message.role if message.role in {"user", "assistant", "system"} else "user"
    return {
        "id": message.message_id,
        "role": role,
        "content": message.content,
        "timestamp": _iso(message.timestamp),
    }


def _load_classifications(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pair_id = obj.get("pair_id")
            if isinstance(pair_id, str):
                rows[pair_id] = obj
    return rows


def _format_evidence_answer(question: str, mode: str, diagnostics: dict[str, Any],
                            evidence, hops: list[dict[str, Any]], consolidator,
                            query_plan: dict[str, Any] | None = None) -> str:
    lines = [
        f"Retrieved evidence ({mode}, k={diagnostics['k']}).",
        "Answer generation is not run in this API (no Ollama reader). The excerpts below are the selected source messages.",
        "",
        f"Seeds: {diagnostics['seedCount']}  ·  B′ expanded: {diagnostics['expandedCount']}  ·  "
        f"Sources selected: {diagnostics['selectedSourceCount']} / {diagnostics['sourceCount']}  ·  "
        f"Budget: {diagnostics['usedChars']}/{diagnostics['budgetChars']} chars",
        "",
    ]
    if query_plan:
        lines.append("Query plan")
        lines.append(f"- need: {query_plan.get('informationNeed') or ''}")
        if query_plan.get("entities"):
            lines.append(f"- entities: {', '.join(query_plan['entities'])}")
        if query_plan.get("facets"):
            lines.append(f"- facets: {', '.join(query_plan['facets'])}")
        for item in query_plan.get("queries") or []:
            lines.append(f"- query: {item}")
        lines.append("")
    seed_ids = set(m.memory_id for m in evidence.matches[: evidence.n_seed_matches])
    lines.append("Memories")
    for match in evidence.matches:
        tag = "seed" if match.memory_id in seed_ids else "bprime"
        statement = consolidator.memories[match.memory_id].content if match.memory_id in consolidator.memories else match.statement
        src = ", ".join(match.source_message_ids) or "(unresolved)"
        lines.append(f"- [{tag}] {match.memory_id} ← {src}")
        lines.append(f"  {statement}")
    if hops:
        lines.append("")
        lines.append("B′ hops")
        for hop in hops:
            reason = hop.get("rationale") or "(no rationale stored)"
            lines.append(
                f"- {hop['pairId']}: {hop['seedMemoryId']} → {hop['neighbourMemoryId']} "
                f"({hop.get('relation')}/{hop.get('direction')})"
            )
            lines.append(f"  {reason}")
    lines.append("")
    lines.append("Selected source messages")
    by_id = {}
    for match in evidence.matches:
        for src in match.sources:
            by_id.setdefault(src.message_id, src.content)
    if not evidence.selection.selected:
        lines.append("(none)")
    for mid in evidence.selection.selected:
        content = by_id.get(mid, "")
        excerpt = content if len(content) <= 280 else content[:279] + "…"
        lines.append(f"- {mid}: {excerpt}")
    lines.append("")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def _choose_conversation_id(store: Store, requested: str | None) -> str:
    listed = store.list_conversations()
    if not listed:
        raise FileNotFoundError("no conversations in the SQLite database")
    if requested:
        return requested
    semantic = SemanticStore(store.connection)
    for cid, _title in listed:
        if semantic.get_claims(cid):
            return cid
    return listed[0][0]


def _imported_filename(title: str | None) -> str | None:
    if title and title.startswith("Imported · "):
        return title[len("Imported · "):]
    return None


_OPTIONAL_CONVERSATION_TABLES = (
    "kd_documents",
    "sx_message_outcomes",
    "sx_records",
    "sx_windows",
    "sx_rolling_summaries",
    "relationship_sources",
    "memory_sources",
    "relationships",
    "memories",
    "entities",
    "extraction_log",
    "consolidation_log",
)


def _clear_optional_conversation_tables(connection, conversation_id: str) -> None:
    """Clear optional conversation-scoped tables inside the caller's open transaction.

    Does not commit. Tables that are absent are skipped.
    """
    existing = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    for table in _OPTIONAL_CONVERSATION_TABLES:
        if table in existing:
            connection.execute(
                f"DELETE FROM {table} WHERE conversation_id = ?",
                (conversation_id,),
            )


def _close_workspace_store(workspace: Workspace) -> None:
    store = getattr(workspace, "_store", None)
    if store is not None:
        try:
            store.close()
        except Exception:
            pass


def open_workspace(
    *,
    sample: str = "fixture",
    db_path: Path | None = None,
    conversation_id: str | None = None,
    classifications_path: Path | None = None,
    embedder: Embedder | None = None,
    k: int | None = None,
    budget_chars: int | None = None,
) -> Workspace:
    load_root_env()
    budget = DEFAULT_BUDGET_CHARS if budget_chars is None else budget_chars
    if sample == "fixture":
        return Workspace.from_sample(embedder=embedder, k=k or DEFAULT_SAMPLE_K)
    if sample == "db":
        if db_path is None:
            raise ValueError("db sample requires a SQLite path (CONTINUATOR_DB_PATH or --db)")
        workspace = Workspace.from_sqlite(
            db_path,
            conversation_id=conversation_id,
            classifications_path=classifications_path,
            embedder=embedder,
            k=k,
            budget_chars=budget,
        )
        if conversation_id or workspace.claims:
            return workspace
        # An explicit dedicated --db is the corpus. Never substitute another SQLite file.
        canonical = resolve_product_db_path()
        try:
            dedicated = Path(db_path).resolve() != canonical.resolve()
        except OSError:
            dedicated = True
        if dedicated:
            return workspace
        # Legacy: empty default DB may fall back to an explicitly set CONTINUATOR_IMPORT_DB.
        alt = legacy_import_db_path()
        if alt is None:
            return workspace
        try:
            same = alt.resolve() == Path(db_path).resolve()
        except OSError:
            same = False
        if not alt.is_file() or same:
            return workspace
        try:
            imported = Workspace.from_sqlite(
                alt,
                classifications_path=classifications_path,
                embedder=embedder,
                k=k,
                budget_chars=budget,
            )
        except FileNotFoundError:
            return workspace
        if not imported.claims:
            _close_workspace_store(imported)
            return workspace
        _close_workspace_store(workspace)
        return imported
    raise ValueError("sample must be 'fixture' or 'db'")
