# Continuator architecture

This document defines the product boundaries for the repository. It reflects the
code as of the V2 product-convergence milestone. Extend product functionality
along the **product spine** below; treat other packages as baseline, legacy, or
offline research unless a change explicitly requires them.

---

## Product spine

```text
src/continuator_memory/v2/
src/continuator_memory/webapi/
web/
```

**This is the primary implementation going forward.**

The intended product is a **model-agnostic long-term memory infrastructure** for
AI applications. The live product path is:

```text
Raw conversation / transcript
        ↓
V2 ingestion          (v2/pipeline.py, webapi/extract.py)
        ↓
Extraction            (v2/extractors.MemoryExtractor)
                      backends: ollama (SemanticAtomizer+OllamaChatClient), passthrough
                      protocols: atomization-v2 | memories-v1
        ↓
Validation            (v2/validation.py)
        ↓
Claims + relations    (v2/store.py → SemanticStore)
        ↓
Consolidation         (v2/consolidation.py; deterministic, no LLM)
        ↓
Persisted memories    (v2/memory_store.py → MemoryStore)
        ↓
Memory retrieval      (v2/memory_retrieval.py)
        ↓
Source evidence       (v2/source_retrieval.py)
        ↓
Context Pack          (webapi/context_pack.py ← Workspace.context)
        ↓
Optional answering    (webapi/answering.py via Workspace.ask / POST /ask)
```

Continuator **provides context**. Answering with Continuator's Ollama path is optional;
any external application can consume the Context Pack's structured fields or
`formattedContext` with its own model.

Entry points:

| Surface | Module | Role |
| --- | --- | --- |
| HTTP API | `webapi/server.py` | `/import`, `/context`, `/ask`, `POST /documents/ingest`, memory/source/document/evidence listing |
| Python SDK | `continuator_memory.sdk` | `ContinuatorClient` — stdlib HTTP client for the product API |
| Workspace | `webapi/workspace.py` | In-process product façade; `context()` then optional `ask()` |
| Context Pack | `webapi/context_pack.py` | Canonical retrieval view (structured + deterministic text); public contract **schema v1** — see `docs/context-pack-schema.md` |
| External LLM example | `examples/external_llm/` | App-owned generation using `formattedContext` + Continuator provenance |
| Web UI | `web/` | React client against the HTTP API |
| Import helper | `webapi/extract.py` | `ingest_imported` → `V2IngestionPipeline` |

### Context Pack API

Canonical product query path (no answering LLM):

```text
Query
  ↓
Workspace.context / POST /context
  ↓
Memory retrieval → source evidence → ContextPack
  ├── memories, evidence, citations (structured)
  └── formattedContext (deterministic prompt text)
```

Optional answer path reuses the same pack:

```text
Workspace.ask / POST /ask
  ↓
Workspace.context → ContextPack
  ↓
answering.generate_answer(pack.answer_context, …)
```

When adding product features (new extract modes, retrieval modes, persistence,
context assembly), prefer changes under `v2/` and `webapi/`.

### Extraction interface

The pipeline depends on `v2/extractors.MemoryExtractor`, not on Ollama types:

```text
V2IngestionPipeline
        ↓
MemoryExtractor.atomize(window, prior) → AtomizationResult
        ↓
validate_atomization (inside SemanticAtomizer) / passthrough ValidatedWindow
        ↓
SemanticStore → consolidate_stored
```

| Knob | Meaning | Env / API |
| --- | --- | --- |
| Extractor backend | `ollama` or `passthrough` | `CONTINUATOR_V2_EXTRACTOR` |
| Model id | Ollama tag (or future HF id) | `CONTINUATOR_V2_ATOMIZER_MODEL` |
| Import mode | instruction protocol | `extract=messages\|v2\|memories` |

Inject any compatible extractor: `V2IngestionPipeline(store, my_extractor)` or
`ingest_imported(..., atomizer=my_extractor)`.

---

## Phase 1 — raw conversational RAG baseline

```text
pipeline.py, chunking.py, embedding.py, retrieval.py, generation.py, store.py
```

**Role:** deterministic conversational RAG over sliding-window chunks, plus the
shared SQLite raw-message store and MiniLM embedding utilities that V2 still
uses for provenance and optional chunk indexes.

**Not** the main memory product. Phase 1 remains the control baseline for
evaluation and the storage layer for raw messages (`Store`).

CLI without `--memory` uses Phase 1 only (`Pipeline`).

---

## Phase 3 — legacy SPO memory (`memory/`)

```text
src/continuator_memory/memory/
```

**Role:** legacy / experimental subject–predicate–object memory formation
(LLM fact extraction into `entities` / `memories` / `relationships` tables).

Enabled only via CLI `--memory`. It is a **parallel** schema, not an ancestor of
V2 claims. Do not extend it for product work; do not delete it in this milestone.
Benchmarks and older reports that call `--memory` remain valid research paths.

---

## Offline research and evaluation

| Path | Role |
| --- | --- |
| `benchmarks/` | Datasets, experiment harnesses, frozen result artifacts |
| `experiments/` | Offline extractor bakeoffs, fine-tuning design (not wired to product) |
| `reports/` | Written experiment and audit reports |
| `knowledge/` | Experimental document-claim ingest; not on the default ask path |
| `ARCHITECTURE_CRIME_SCENE.md` | Historical LongMemEval cost/architecture autopsy |

Changing product code should not silently rewrite benchmark harnesses to match
new behaviour. If a product change intentionally alters a measured path, update
or note the affected experiment explicitly.

---

## Retrieval modes (product)

Product modes accepted by `Workspace.context` / `Workspace.ask` / `POST /context` / `POST /ask`:

| Mode | Behaviour |
| --- | --- |
| `baseline` | Semantic search over consolidated memories (MiniLM + status adjustment) |
| `lexical` | Same as baseline, plus lexical overlap weighting (`CONTINUATOR_LEXICAL_WEIGHT`) |
| `bprime` | Baseline seeds + optional classified neighbour expansion |
| `multiquery` | Experimental: plan alternate queries, RRF-fuse memory rankings |
| `iterative` | Experimental: multi-round query planning with coverage evaluation |

`hybrid` is accepted as a **deprecated alias** of `lexical`. It must not be
confused with `v2/hybrid_retrieval.py` (`HybridRetriever`), which is a separate
benchmark algorithm that RRF-fuses Phase 1 raw chunks with memory sources.

---

## Persistence model

- **Raw evidence:** `messages` (and optional Phase 1 chunks/embeddings) in SQLite.
- **Extracted evidence:** `v2_claims`, `v2_relations`, `v2_semantic_messages`.
- **Derived state:** consolidated rows in `v2_memories`, `v2_memory_sources`,
  `v2_memory_versions`, `v2_memory_decisions`.

Consolidation is deterministic and idempotent for a given claim stream. A
snapshot is replaced on successful re-ingest. Failed protected re-ingests leave
previous claims **and** previous memory snapshots untouched (`kept_existing`).

Product retrieval **prefers** the persisted snapshot. If none exists (older DBs,
fixture workspaces without a store), it falls back to in-memory `consolidate()`
and may persist the snapshot when a DB connection is available.

---

## What a contributor should extend

| Goal | Extend |
| --- | --- |
| Extraction / validation / consolidation / memory retrieval | `v2/` |
| HTTP API, import, answer prompts, product retrieval wiring | `webapi/` |
| UI | `web/` |
| Shared embeddings / raw message SQLite | Phase 1 modules (`store.py`, `embedding.py`, …) |
| Offline eval only | `benchmarks/`, `experiments/` |
| Publishable LoCoMo harness | `benchmarks/publish/` (wraps `webapi/locomo_report`; see PROTOCOL.md) |
| Legacy SPO experiments | `memory/` (avoid for product) |
