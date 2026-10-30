"""Load a sample or existing V2 SQLite cache and run baseline / B′ retrieval.

Does not extract, classify, or call Ollama. Embeddings are created on first ask.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from continuator_memory.config import Settings
from continuator_memory.embedding import Embedder
from continuator_memory.envfile import load_root_env
from continuator_memory.models import Conversation, Message
from continuator_memory.store import Store
from continuator_memory.v2.consolidation import consolidate
from continuator_memory.v2.lexical import load_lexical_weight
from continuator_memory.v2.memory_retrieval import RetrievalConfig
from continuator_memory.v2.memory import CURRENT_STATUSES, STATUS_SUPERSEDED
from continuator_memory.v2.models import Claim, Relation
from continuator_memory.v2.store import SemanticStore

from continuator_memory.generation import OllamaGenerator
from continuator_memory.webapi.answering import (
    MAX_ANSWER_TOP_K,
    GenerationSettings,
    build_answer_context,
    context_to_evidence_fields,
    generate_answer,
    load_generation_settings,
    make_ollama_generator,
    parse_generation_overrides,
)
from continuator_memory.webapi.extract import ingest_imported
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
from continuator_memory.v2.source_retrieval import retrieve_source_evidence

MAX_QUESTION_CHARS = 4000
MAX_K = MAX_ANSWER_TOP_K  # k is also the answer evidence count here, so it shares the answerer's ceiling
DEFAULT_SAMPLE_K = 1
DEFAULT_BUDGET_CHARS = 50_000
MEMORY_TYPES = frozenset({"person", "project", "fact", "preference", "event"})
RETRIEVAL_MODES = frozenset({"baseline", "bprime", "multiquery", "iterative", "hybrid"})


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
    ):
        if k < 1 or k > MAX_K:
            raise ValueError(f"k must be 1..{MAX_K}")
        self.conversation = conversation
        self.claims = claims
        self.relations = relations
        self.classifications_path = Path(classifications_path) if classifications_path else None
        self._embedder = embedder
        self.k = k
        self.budget_chars = budget_chars
        self.sample_id = sample_id
        self.source_title = source_title or conversation.title or conversation.conversation_id
        self.extraction = "fixture" if sample_id == "fixture" else "sqlite"
        self.imported_filename: str | None = None
        self.consolidator = consolidate(conversation.conversation_id, claims, relations)
        self._lock = threading.Lock()
        self._importing = False
        self._generator = None
        self._planner = None
        self._evaluator = None
        self._generation = load_generation_settings()
        self._chats: dict[str, dict[str, Any]] = {}
        self._classifications: dict[str, dict[str, Any]] = {}
        if self.classifications_path and self.classifications_path.is_file():
            self._classifications = _load_classifications(self.classifications_path)

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
        workspace._store = store  # noqa: SLF001 — keep DB handle alive
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
            raise ValueError("extract must be 'v2' (Ollama atomizer), 'memories' (Ollama memory extraction) "
                             "or 'messages' (one memory per message).")
        with self._lock:
            if self._importing:
                raise ValueError("an import is already running")
            self._importing = True
        try:
            conversation = parse_transcript(text, filename=filename)
            store_path = db_path or Path(os.environ.get("CONTINUATOR_IMPORT_DB", "data/continuator-web-import.db"))
            store = Store(str(store_path))
            try:
                report = ingest_imported(conversation, store, extract=extract, atomizer=atomizer)
            except Exception:
                store.close()
                raise
            if report.kept_existing:
                store.close()
                raise ValueError(
                    f"extraction did not complete ({report.failed_windows} of {report.windows} windows failed, "
                    f"{report.claims} memories produced). The previously stored conversation and memories were kept."
                )
            semantic = SemanticStore(store.connection)
            claims = tuple(semantic.get_claims(conversation.conversation_id))
            relations = tuple(semantic.get_relations(conversation.conversation_id))
            if not claims:
                store.close()
                raise ValueError(
                    "import stored the conversation but extraction produced no memories. "
                    "The workspace still shows the previous conversation until it is reloaded."
                )
            empty_links = Path(tempfile.mkdtemp(prefix="continuator-import-")) / "classifications.jsonl"
            empty_links.write_text("", encoding="utf-8")
            settings = Settings.from_env()
            old_store = getattr(self, "_store", None)
            with self._lock:
                self.conversation = conversation
                self.claims = claims
                self.relations = relations
                self.classifications_path = empty_links
                self._classifications = {}
                self.sample_id = "imported"
                self.extraction = {"v2": "v2_atomizer", "memories": "memory_extraction"}.get(extract, "message_claims")
                self.imported_filename = Path(filename).name
                self.source_title = conversation.title or f"Imported · {self.imported_filename}"
                self.k = settings.top_k if self.k == DEFAULT_SAMPLE_K else self.k
                self.consolidator = consolidate(conversation.conversation_id, claims, relations)
                self._chats.clear()
                self._store = store
            if old_store is not None and old_store is not store:
                try:
                    old_store.close()
                except Exception:
                    pass
            health = self.health()
            return {
                "ok": True,
                "source": self._source_summary(),
                "report": {
                    "filename": self.imported_filename,
                    "nMessages": len(conversation.messages),
                    "nClaims": len(claims),
                    "nMemories": health["nMemories"],
                    "nRelations": len(relations),
                    "failedWindows": report.failed_windows,
                    "windows": report.windows,
                    "extract": self.extraction,
                    "bprimeLinks": 0,
                },
                "health": health,
                "note": (
                    "B′ retrieval is available but has no classified links for this import, "
                    "so expansion will match baseline until a classifications JSONL is supplied."
                ),
            }
        finally:
            with self._lock:
                self._importing = False

    def embedder(self) -> Embedder:
        if self._embedder is None:
            from continuator_memory.embedding import SentenceTransformerEmbedder
            from continuator_memory.config import Settings

            self._embedder = SentenceTransformerEmbedder(Settings.from_env().embedding_model)
        return self._embedder

    def health(self) -> dict[str, Any]:
        return {
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
        }

    def list_memories(self, *, text: str = "", types: list[str] | None = None,
                      include_superseded: bool = True) -> list[dict[str, Any]]:
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
        return [self._source_summary()]

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        if source_id != self.conversation.conversation_id:
            return None
        by_message: dict[str, list[str]] = {}
        for mem in self.consolidator.memories.values():
            for src in self.consolidator.sources_of(mem.memory_id):
                by_message.setdefault(src.message_id, []).append(mem.memory_id)
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
        with self._lock:
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
        with self._lock:
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

    def ask(self, question: str, *, conversation_id: str | None = None,
            retrieval_mode: str = "baseline", generate: bool | None = None,
            generation: dict[str, Any] | None = None) -> dict[str, Any]:
        text = question.strip()
        if not text:
            raise ValueError("Question is empty.")
        if len(text) > MAX_QUESTION_CHARS:
            raise ValueError(f"Question exceeds {MAX_QUESTION_CHARS} characters.")
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(
                "retrieval_mode must be 'baseline', 'bprime', 'multiquery', 'iterative', or 'hybrid'."
            )
        if retrieval_mode == "bprime" and not (self.classifications_path and self.classifications_path.is_file()):
            raise ValueError(
                "B′ retrieval needs a classifications JSONL file "
                "(set CONTINUATOR_BPRIME_CLASSIFICATIONS or use the built-in sample)."
            )

        settings = parse_generation_overrides(generation, self._generation)
        query_diagnostics: dict[str, Any] = {}
        original_weight = load_original_weight()
        if retrieval_mode == "multiquery":
            evidence, query_diagnostics = retrieve_multiquery_evidence(
                text,
                self.conversation,
                self.claims,
                self.relations,
                self.embedder(),
                self.k,
                self.budget_chars,
                planner=self._resolve_planner(settings),
                original_weight=original_weight,
            )
        elif retrieval_mode == "iterative":
            evidence, query_diagnostics = retrieve_iterative_evidence(
                text,
                self.conversation,
                self.claims,
                self.relations,
                self.embedder(),
                self.k,
                self.budget_chars,
                planner=self._resolve_planner(settings),
                evaluator=self._resolve_evaluator(settings),
                original_weight=original_weight,
            )
        else:
            bprime = None
            if retrieval_mode == "bprime":
                bprime = BPrimeConfig(
                    enabled=True,
                    classifications_path=self.classifications_path,
                    max_hops=1,
                )
            retrieval_config = None
            if retrieval_mode == "hybrid":
                retrieval_config = RetrievalConfig(lexical_weight=load_lexical_weight())
            evidence = retrieve_source_evidence(
                text,
                self.conversation,
                self.claims,
                self.relations,
                self.embedder(),
                self.k,
                self.budget_chars,
                config=retrieval_config,
                bprime=bprime,
            )
        should_generate = self._generation.enabled if generate is None else bool(generate)
        payload = self._ask_payload(
            text, retrieval_mode, evidence, settings=settings, generate=should_generate,
            query_diagnostics=query_diagnostics,
        )
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        with self._lock:
            chat = self._chats.get(conversation_id) if conversation_id else None
            if chat is None:
                chat = {
                    "id": f"chat-{uuid.uuid4().hex[:10]}",
                    "title": _title(text),
                    "updatedAt": now,
                    "messages": [],
                }
                self._chats[chat["id"]] = chat
            user_msg = {
                "id": f"{chat['id']}-q{len(chat['messages']) + 1}",
                "role": "user",
                "content": text,
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

    def _ask_payload(self, question: str, mode: str, evidence, *,
                     settings: GenerationSettings, generate: bool,
                     query_diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
        seed_ids = [m.memory_id for m in evidence.matches[: evidence.n_seed_matches]]
        expanded_ids = list(evidence.bprime.get("expanded_memory_ids") or [])
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
        selected = list(evidence.selection.selected)
        memory_ids = [m.memory_id for m in evidence.matches]
        source_ids = [self.conversation.conversation_id] if selected else []
        by_msg = {m.message_id: m for m in self.conversation.messages}
        selected_messages = [_message_json(by_msg[mid]) for mid in selected if mid in by_msg]
        extra = query_diagnostics or {}
        ctx = build_answer_context(
            question,
            evidence,
            self.conversation,
            seed_ids=seed_ids,
            expanded_ids=expanded_ids,
            max_context_chars=settings.max_context_chars,
            context_tokens=settings.num_ctx,
            output_tokens=settings.num_predict,
        )
        if generate:
            answer_text, generation_meta = generate_answer(
                ctx, settings, self._resolve_generator(settings),
            )
            gen_status = generation_meta.get("status", "ok")
        else:
            answer_text = _format_evidence_answer(
                question, mode,
                {"k": self.k, "seedCount": evidence.n_seed_matches,
                 "expandedCount": len(expanded_ids),
                 "selectedSourceCount": len(selected),
                 "sourceCount": len(evidence.selection.unique_sources),
                 "usedChars": evidence.selection.used_chars,
                 "budgetChars": evidence.selection.budget_chars},
                evidence, hops, self.consolidator,
                query_plan=extra.get("queryPlan"),
            )
            generation_meta = {**settings.public_dict(), "status": "not_run"}
            gen_status = "not_run"
        diagnostics = {
            "mode": mode,
            "k": self.k,
            "seedCount": evidence.n_seed_matches,
            "expandedCount": len(expanded_ids),
            "sourceCount": len(evidence.selection.unique_sources),
            "selectedSourceCount": len(selected),
            "droppedSourceCount": len(evidence.selection.dropped),
            "budgetChars": evidence.selection.budget_chars,
            "usedChars": evidence.selection.used_chars,
            "answerGeneration": gen_status,
            "promptMemoryCount": len(ctx.prompt_memory_ids),
            "retrievedMemoryCount": len(ctx.retrieved_memory_ids),
        }
        plan = extra.get("queryPlan") or {}
        if mode == "multiquery" and plan.get("queries"):
            diagnostics["queryCount"] = len(plan["queries"])
            diagnostics["planStatus"] = plan.get("status")
        if extra.get("retrievalComparison", {}).get("latencyMs"):
            diagnostics["latencyMs"] = extra["retrievalComparison"]["latencyMs"]
        if mode == "iterative":
            diagnostics["roundCount"] = extra.get("roundCount")
            diagnostics["stopReason"] = extra.get("stopReason")
        if mode == "hybrid":
            diagnostics["lexicalWeight"] = load_lexical_weight()
        return {
            "answerText": answer_text,
            "evidence": {
                "memoryIds": memory_ids,
                "sourceIds": source_ids,
                "expandedMemoryIds": expanded_ids,
                "hops": hops,
                "diagnostics": diagnostics,
                "selectedMessageIds": selected,
                "selectedMessages": selected_messages,
                "seedMemoryIds": seed_ids,
                "generation": generation_meta,
                "queryPlan": extra.get("queryPlan"),
                "queryRankings": extra.get("queryRankings"),
                "fusedRanking": extra.get("fusedRanking"),
                "retrievalComparison": extra.get("retrievalComparison"),
                "iterativeRounds": extra.get("iterativeRounds"),
                "stopReason": extra.get("stopReason"),
                **context_to_evidence_fields(ctx),
            },
        }

    def _memory_json(self, mem) -> dict[str, Any]:
        sources = [
            {"sourceId": self.conversation.conversation_id, "messageId": s.message_id}
            for s in self.consolidator.sources_of(mem.memory_id)
        ]
        created = _iso(mem.created_at or mem.valid_from)
        return {
            "id": mem.memory_id,
            "title": _title(mem.content, 40),
            "statement": mem.content,
            "type": _memory_type(mem.claim_type),
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


def _close_workspace_store(workspace: Workspace) -> None:
    store = getattr(workspace, "_store", None)
    if store is not None:
        try:
            store.close()
        except Exception:
            pass


def _import_db_path() -> Path:
    return Path(os.environ.get("CONTINUATOR_IMPORT_DB", "data/continuator-web-import.db"))


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
        default_db = Path(os.environ.get("CONTINUATOR_DB_PATH", "data/continuator.db"))
        try:
            dedicated = Path(db_path).resolve() != default_db.resolve()
        except OSError:
            dedicated = True
        if dedicated:
            return workspace
        alt = _import_db_path()
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
