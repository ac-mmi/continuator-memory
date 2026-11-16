# Continuator Universal Indexing Architecture Audit

**Scope:** read-only inspection of the repository as of this audit.  
**No production code changes. No benchmark runs. No model downloads.**

Label legend used below:

- **MEASURED / OBSERVED** — confirmed in source
- **INFERRED** — reasonable from code structure, not exhaustively runtime-verified
- **ARCHITECTURAL RECOMMENDATION** — engineering judgment from evidence
- **NOT VERIFIED** — not exercised in this audit

---

## 1. Executive summary

### MEASURED / OBSERVED

Continuator’s **product spine** (`ARCHITECTURE.md`, `src/continuator_memory/v2/`, `webapi/`) is a **conversation-first long-term memory system**:

```text
transcript / LoCoMo / Conversation JSON
  → Message sequence
  → ContextWindow atomization (passthrough or LLM)
  → Claims (+ relations)
  → Consolidator → Memory (+ MemorySource + MemoryLink)
  → SQLite MemoryStore
  → MemoryRetriever (dense) / BM25 over contextual units / CE rerank
```

The **canonical indexed/retrievable object in the product path is the V2 `Memory`**, keyed by `memory_id`, with provenance via `MemorySource` → `message_id`.

There is **no product-path adapter** for PDF, HTML, Markdown documents, or code repositories as first-class sources. Non-conversation input is forced through conversation parsers (`parse_transcript`, LoCoMo adapter, JSON conversation schema).

A **parallel research/knowledge path** (`knowledge/ingest.py`) builds LLM “documents” from **message windows** and stores passages separately from V2 memories — not the universal evidence layer described in the target architecture.

### ARCHITECTURAL RECOMMENDATION

**Choose OPTION D (hybrid):** keep the conversation memory pipeline; introduce a **new universal canonical evidence layer beside it**, with source adapters mapping into that layer. Retrieval/ranking work can continue on conversations, but a universal product will hit the conversation/`Message` ceiling without an explicit evidence-unit model.

---

## 2. Current ingestion flow

### 2.1 Product entry points (MEASURED)

| Entry | File | What it accepts |
|---|---|---|
| Web/UI import | `webapi/workspace.py` `Workspace.import_text` | Paste/transcript text → `parse_transcript` |
| LoCoMo ingest | `webapi/locomo.py` `ingest_locomo` | LoCoMo JSON → `Conversation` |
| Product extract helper | `webapi/extract.py` `ingest_imported` | Any `Conversation` → V2 pipeline |
| CLI ingest (legacy V1) | `cli.py` → `pipeline.Pipeline.ingest` | Conversation JSON → chunks+embeddings |
| Knowledge ingest (research) | `knowledge/ingest.py` `ingest_documents` | Conversation → chunk windows → LLM docs |

### 2.2 Stage table — V2 product path (MEASURED)

| Stage | Actual implementation | Input | Output | File/function |
|---|---|---|---|---|
| Raw input | Transcript / LoCoMo / JSON conversation | bytes/text/JSON | `Conversation` + `Message`s | `webapi/transcript.py` `parse_transcript`; `webapi/locomo.py` `conversation_from_locomo`; `models.py` `parse_conversation` |
| Parsing | Role/header heuristics; LoCoMo session/dia_id mapping | text/JSON | ordered messages | same |
| Normalization | Role map; speaker prefix in LoCoMo content; ISO timestamps | messages | normalized `Message` | `locomo.py` `_message`; `transcript.py` |
| Segmentation (extraction windows) | Fixed target-message windows + prior context | `Conversation` | `ContextWindow` | `v2/windowing.py` `build_windows` (default 4 targets, 4 context) |
| Extraction | Passthrough or Ollama SemanticAtomizer | window + prior claims | `ValidatedWindow` / claims | `v2/extractors.py`; `v2/atomizer.py` |
| Memory/evidence creation | Deterministic consolidation | claims + relations | `Memory`, `MemorySource`, `MemoryLink` | `v2/consolidation.py`; `v2/memory.py` |
| Persistence | SQLite | conversation + claims + memories | tables under `Store` / `SemanticStore` / `MemoryStore` | `store.py`; `v2/store.py`; `v2/memory_store.py` |
| Optional raw chunk index | Message sliding windows + embeddings | conversation | `chunks` + `chunk_embeddings` | `chunking.py` `chunk_conversation`; `v2/pipeline.py` `_raw_rows` (only if embedder set) |
| Dense index | In-process vectors over `Memory.content` | memories | `MemoryRetriever` entries | `v2/memory_retrieval.py` `MemoryRetriever.index` |
| Lexical/BM25 index | Ephemeral Okapi BM25 over contextual unit text | memories + conversation | `BM25Index` | `v2/bm25.py`; `webapi/contextual_bm25.py` |
| Retrieval availability | Workspace modes | question | ranked memories / evidence | `webapi/workspace.py` |

### 2.3 Flow diagram (MEASURED)

```text
SOURCE (chat transcript | LoCoMo | Conversation JSON)
        ↓  source-specific parser
Conversation { messages: Message[] }
        ↓  build_windows (message windows)
ContextWindow
        ↓  MemoryExtractor (passthrough | ollama)
Claims / Relations  → SemanticStore
        ↓  consolidate_stored
Memory + MemorySource + MemoryLink  → MemoryStore
        ↓
MemoryRetriever.index(memory.content)     [dense]
BM25(RetrievalUnit(retrieval_content=PREV/TARGET/NEXT text))  [sparse, ephemeral]
CrossEncoder(passage)                     [rerank, experimental]
```

---

## 3. Current canonical indexed unit

### Answers (MEASURED)

1. **What is the canonical unit today?**  
   Product retrieval: **`Memory`** (`v2/memory.py`), identified by `memory_id` (typically `mem:{claim_id}`).  
   Legacy V1 CLI retrieve: **`Chunk`** (`models.py`).  
   Knowledge research path: **`PassageRow` / Document** (`knowledge/`).

2. **Stable identity?**  
   **Yes for memories** within a conversation: `memory_id` primary key in `v2_memories`.  
   Claim→memory mapping is deterministic from consolidation.

3. **Provenance preserved?**  
   **Partially.** `MemorySource` stores `(memory_id, claim_id, message_id, position, observed_at)`. Raw message content lives in `messages`. No page/byte offset/span for arbitrary documents.

4. **Can one indexed unit map to multiple original source messages?**  
   **Yes.** `RetrievalResult.source_message_ids` aggregates all `MemorySource.message_id`s for that memory (`memory_retrieval.py`). Consolidation can attach multiple sources over time (`consolidation.py` appends `MemorySource`).

5. **Can one original message map to multiple indexed units?**  
   **Yes.** Passthrough creates one claim/memory per message; LLM extraction can emit multiple claims per message; supersession creates new memory rows while keeping older ones.

6. **Are source offsets / locations preserved?**  
   **Message-level only** (`message_id`, conversation `position`, optional `timestamp`). **No** character offsets, page numbers, heading paths, or file:line for code. **MEASURED.**

7. **Parent-child hierarchy?**  
   **Memory version links** (`MemoryLink`: UPDATES / SUPERSEDES / CONTRADICTS / REFERS_TO) — versioning, not document structure.  
   **No persisted** document→section→paragraph hierarchy. Conversation structure is implicit in message order + timestamps.

---

## 4. Chunking / segmentation audit

### Mechanisms found (MEASURED)

#### A. V1 / raw message window chunker — `chunking.py` `chunk_conversation`

```text
Purpose:          Build overlapping windows of consecutive messages for V1 chunk embeddings / knowledge prep
Trigger:          Pipeline ingest with embedder; knowledge prepare_source; V2 _raw_rows when embedder set
Boundary logic:   Whole messages only; never splits a message
Size limits:      window_size messages (default 4 product V2 raw path; knowledge uses 4)
Overlap:          message count overlap (V2 default 2; knowledge ingest uses overlap 0)
Identity:         chunk_id = "{conversation_id}:c{position:04d}"
Provenance:       source_message_ids = message ids in window
```

**Fixed-size?** Message-count fixed window — **not token-fixed** for this chunker.  
**Structure-aware?** Message boundaries only; no headings/paragraphs/code AST.

#### B. Atomization windows — `v2/windowing.py` `build_windows`

```text
Purpose:          Batch messages for extraction calls
Trigger:          Every V2 ingest
Boundary logic:   Non-overlapping target groups; prior messages as CONTEXT only
Size limits:      targets_per_window=4, context_before=4 (defaults)
Overlap:          Targets do not overlap; context may repeat across windows
Identity:         ContextWindow.index
Provenance:       Target message ids preserved into claims via Provenance
```

#### C. Knowledge passage packing — `knowledge/ingest.py` `build_passages`

```text
Purpose:          Pack LLM claims into embedding-sized passages
Trigger:          After document parse per chunk
Boundary logic:   Split by section / citation / estimated token budget
Size limits:      MAX_PASSAGE_TOKENS ≈ 180 (chars/4 estimate)
Overlap:          None between passages
Identity:         "{document_id}:p{index:03d}"
Provenance:       cited message_ids or whole-chunk fallback
```

#### D. Contextual PREV/TARGET/NEXT — `v2/contextual_units.py`

```text
Purpose:          Ephemeral retrieval text enrichment for BM25 / CE / contextual lexical
Trigger:          Query-time (or index build for BM25 units); not ingestion persistence of enriched text
Boundary logic:   ±1 same-session neighbour message around memory.message_id
Size limits:      neighbour count (default 1 prev, 1 next)
Overlap:          N/A (generated view)
Identity:         Still the same memory_id
Provenance:       Explicitly target-only in BM25 units (PREV/NEXT not credited)
```

### Evidence boundaries that can be lost (MEASURED / INFERRED)

| Risk | Mechanism | Effect |
|---|---|---|
| Multi-fact message → one memory | Passthrough `PassThroughExtractor` | Entire message becomes one unit; finer facts not separated |
| LLM claim rewrite | SemanticAtomizer | Indexed `Memory.content` may differ from raw message wording |
| Window merge (V1 chunks) | `chunk_conversation` | Multiple turns share one chunk embedding (legacy path) |
| Session stamp coupling | LoCoMo timestamps | “Same session” for PREV/NEXT depends on equal timestamps — brittle for generic corpora |
| No document structure | No MD/PDF adapters | Headings/pages never become units |

---

## 5. Structural relationship audit

| Relationship | Supported? | How represented? | Stable across ingestion? |
|---|---|---|---|
| previous / next message | Partial | Implicit via `Message.position`; neighbours reconstructed in `neighbour_messages` | Yes for stored messages |
| same conversation | Yes | `conversation_id` | Yes |
| same session | Partial | Equal non-empty `Message.timestamp` (`same_session`) | Only if adapter stamps timestamps |
| memory version parent/child | Yes | `MemoryLink` / `v2_memory_versions` | Yes (persisted) |
| claim→message | Yes | `MemorySource` / claim provenance | Yes |
| document→section→paragraph | **No** (product) | — | — |
| file→symbol (code) | **No** | — | — |
| PREV/TARGET/NEXT as persisted structure | **No** | Generated at retrieval/index-build time in `contextual_units.py`; **not** stored as edges | Regenerated from messages |

### MEASURED clarification

`[PREV]/[TARGET]/[NEXT]` is **dynamic retrieval text**, not a persisted graph. BM25 stores the enriched string only in an **in-memory** `RetrievalUnit`, not as the canonical SQLite memory content.

---

## 6. Representation audit

| Representation | Source | Generated when | Persisted? | Used by |
|---|---|---|---|---|
| Original message content | `Message.content` | ingest/parse | Yes (`messages`) | provenance, display, neighbour text |
| Claim content | extractor | atomize | Yes (`SemanticStore`) | consolidation input |
| Memory content | claim (usually unchanged) | consolidate | Yes (`v2_memories.content`) | dense embedding text (default) |
| MemorySource provenance | consolidation | consolidate | Yes | credit / `source_message_ids` |
| V1 chunk text | joined `role: content` windows | optional raw path | Yes (`chunks`) | legacy retrieve |
| Chunk embedding | chunk text | with embedder | Yes (`chunk_embeddings`) | legacy dense |
| Memory dense vector | `Memory.content` | `MemoryRetriever.index` | **In-process only** (not a vector DB table for V2 memories) | dense / hybrid lexical |
| Optional turn-context vector | `turn_context` dual-vector experiment | index with contexts map | In-process | contextual_weight retrieval |
| Contextual retrieval text | PREV+TARGET+NEXT | query/BM25 build | **No** (ephemeral) | contextual_bm25, CE passages, contextual_lexical |
| CE passage variants | reconstructed for ablation | offline cached runner | No (experiment artifacts) | `rerank_cached_wide_pool` |
| Knowledge passage embedded text | claim packing | knowledge ingest | Yes (`KnowledgeStore`) | knowledge retrieval research |

### Critical question

> One canonical evidence object with multiple representations, or different retrieval objects?

**MEASURED answer:** Closest to **one canonical `Memory`**, with **derived views**:

- Dense indexes `Memory.content` (unless dual-context vectors).
- BM25/CE often score **ephemeral enriched text** while identity remains `memory_id`.
- Legacy **chunks** and **knowledge passages** are **separate objects**, not views of `Memory`.

So: product retrieval is mostly “one memory, multiple scoring texts,” but the repo also contains parallel object types that are not unified.

---

## 7. Indexing audit

### Dense (MEASURED)

| Aspect | Reality |
|---|---|
| Embedding generation | `SentenceTransformerEmbedder` (`embedding.py`) over memory contents |
| Stored vector | `MemoryRetriever._entries` in RAM; rebuilt when Workspace loads consolidator |
| Vector DB | **No** dedicated V2 vector table observed; V1 has `chunk_embeddings` BLOB table |
| Identity mapping | `(conversation_id, memory_id)` |
| Update behavior | Re-embed when content changes; drop missing memories on sync (`IndexStats`) |

### Lexical (MEASURED)

| Aspect | Reality |
|---|---|
| Product “lexical” mode | Dense cosine + **lexical overlap bonus** (`v2/lexical.py`) — not BM25 |
| True BM25 | `v2/bm25.py` Okapi BM25 over `RetrievalUnit.retrieval_content` |
| Indexed text (contextual_bm25) | Ephemeral PREV/TARGET/NEXT string |
| Tokenization | `v2/lexical.tokenize` / `sparse_tokenize` |
| Identity | `RetrievalUnit.id` = `memory_id` |

### Other indexes (MEASURED)

| Index | Exists? | Notes |
|---|---|---|
| Metadata filters | Limited | status/time_class used in ranking eligibility; not a general metadata index |
| Graph | Experimental | `v2/graph_index.py` `GraphIndex` over messages (eval harness) |
| B′ links | Optional | classification JSONL expansion (`bprime_links.py`) |
| Parent/section index | No | — |

---

## 8. Provenance and evidence integrity

### Can Continuator answer “exactly where did this come from?”

| Trace target | Conversation product path | Universal document path |
|---|---|---|
| Original source file | Filename only on import (`imported_filename`) | **Not modeled** |
| Conversation / document | `conversation_id` | Document id exists only in knowledge path |
| Original message / paragraph | `message_id` | No paragraph/span id |
| Location (page/offset/time) | optional `timestamp` | **No page/offset** |

### Is `memory_id` + `source_message_ids` enough for a universal document system?

**MEASURED limitation:** Sufficient for **message-granular conversational evidence**. Insufficient for:

- PDF page + char span
- Markdown heading path
- HTML DOM path
- Code `path:symbol:line`
- Stable re-ingest when source text is edited mid-document

Multi-source memories help conversational merging, but provenance remains **message-id based**, not **span based**.

---

## 9. Universal architecture gap analysis

| Capability | Current state | Evidence/files | Gap | Priority |
|---|---|---|---|---|
| 1. Source-specific adapters | Conversation/transcript/LoCoMo only | `transcript.py`, `locomo.py` | No PDF/MD/HTML/code adapters | **P0** |
| 2. Canonical document model | `Conversation` only (product) | `models.py` | No Document/Section abstraction | **P0** |
| 3. Canonical evidence unit | `Memory` (claim-derived) | `v2/memory.py` | Tied to claims/conversations; not generic evidence | **P0** |
| 4. Structure-aware segmentation | Message windows only | `windowing.py`, `chunking.py` | No heading/paragraph/AST split | **P0** |
| 5. Hierarchical relationships | Version links only | `MemoryLink` | No parent section tree | **P1** |
| 6. Stable evidence identity | Yes within conversation | `memory_id` PK | No cross-corpus content-addressed evidence id | **P1** |
| 7. Full provenance | Message-level | `MemorySource` | No offsets/pages/paths | **P0** |
| 8. Multiple representations per unit | Partial (ephemeral texts) | `contextual_units.py`, retriever | Not a first-class representation store | **P1** |
| 9. Dense indexing | Yes (in-memory) | `memory_retrieval.py` | No durable vector index for V2 memories | **P2** |
| 10. Lexical indexing | Yes (overlap + BM25) | `lexical.py`, `bm25.py` | BM25 ephemeral; rebuild each load | **P2** |
| 11. Metadata filtering/indexing | Minimal | status/time in ranker | No general filters | **P2** |
| 12. Context expansion after retrieval | Partial | PREV/NEXT text; B′; iterative modes | Not structure-native expand | **P1** |
| 13. Incremental re-indexing | Partial | retriever sync; protect_existing ingest | No document-patch updates | **P1** |
| 14. Source updates/deletions | Conversation replace semantics | `V2IngestionPipeline` | No fine-grained doc update API | **P1** |
| 15. Reproducible ingestion/versioning | Partial | prompt/model versions in knowledge; V2 statuses | No corpus version snapshot for universal sources | **P2** |

---

## 10. Incremental evolution assessment

| Target capability | Class | Notes |
|---|---|---|
| Conversation adapters | **A** | Reuse as-is |
| Memory + MemorySource identity | **A** | Strong foundation for evidence identity |
| Dense MemoryRetriever | **A/B** | Reuse; may need durable vector store later |
| BM25 RetrievalUnit abstraction | **A** | Already opaque id + content + provenance |
| CE reorder-only rerank | **A** | Representation-agnostic |
| Ephemeral contextual views | **B** | Extend as representation strategy, not structure |
| Claim extraction for chats | **A/B** | Keep for conversational memory product |
| Source adapters (PDF/MD/HTML/code) | **C** | Add beside conversation parsers |
| Canonical Document/Evidence model | **C** → risk of **D** if forced into Message | Can start as parallel tables; **D** if Message is mandatory forever |
| Span-level provenance | **C** | Additive fields/tables |
| Structure hierarchy edges | **C** | New relation types or edge table |
| Making Message the only unit for all corpora | **D** | Forces fake “messages” for pages/functions; loses native structure |

### Why some items are near-D

Forcing PDFs/code into `Message`/`Conversation` without a new model would:

- invent synthetic roles/positions for non-dialogue content (**INFERRED** misuse of `transcript.py` / `Message.role`)
- overload `timestamp` for session grouping (`contextual_units.same_session`)
- block page/line provenance (**MEASURED** absence)

That is an architectural ceiling, not a retrieval bug.

---

## 11. Risks and architectural limitations

1. **Conversation ontology leakage** into all retrieval (roles, sessions, dia_id).  
2. **Claim/memory semantics** may be wrong for raw document evidence (not every corpus wants LLM atomization).  
3. **Dual object systems** (V1 chunks, V2 memories, knowledge passages) risk diverging identity.  
4. **Ephemeral BM25/CE text** can change ranking without changing stored evidence (representation drift).  
5. **In-memory dense index** complicates multi-process / large-corpus deployment.  
6. **LoCoMo-centric session heuristics** may not generalize.

---

## 12. Recommended path

### OPTION D — Hybrid

```text
Preserve existing conversation pipeline while introducing
a new universal canonical ingestion layer alongside it.
```

**Why not A:** Universal corpora are not supported; limitation is ingestion model, not only ranking.  
**Why not B alone:** Incremental patches on `Message` can work short-term but encode the wrong ontology.  
**Why not C rewrite:** Conversation memory + consolidator + multi-channel retrieval are valuable and should not be discarded.

**Practical sequencing (ARCHITECTURAL RECOMMENDATION):**

1. Keep V2 conversation path for chat memory product.  
2. Define `Document` / `EvidenceUnit` / `EvidenceSource` (span-capable) parallel to Memory.  
3. Add adapters (MD → sections; PDF text → pages/blocks; code → symbols).  
4. Map evidence units into existing `RetrievalUnit` + dense index interfaces.  
5. Continue ranking experiments on conversations without blocking universal ingest.

---

## Direct answers

### 1. What exactly is Continuator indexing today?

**MEASURED:** Primarily **consolidated V2 memories** (text = claim/memory content) into an in-process dense index, plus optional ephemeral BM25 over contextualized memory text, plus legacy message-window **chunks** (and a separate knowledge-passage store).

### 2. What exactly is the current evidence unit?

**MEASURED:** Product evidence unit = **`Memory`** with provenance **`MemorySource` → `message_id`**. Evaluation credit uses full `source_message_ids` tuples on retrieval results.

### 3. Is the current chunking/indexing architecture suitable for a universal evidence retrieval product?

**Partially.** Suitable as a **conversation evidence** backbone and as a pattern for multi-channel retrieval. **Not sufficient** as-is for PDFs/docs/code without new adapters and a span-aware evidence model.

### 4. What is the single biggest architectural limitation?

**Conversation/`Message` is the only first-class source ontology in the product spine**, so non-dialogue structure and locations cannot be represented faithfully.

### 5. Can we reach strong universal evidence retrieval without changing ingestion?

**No (for non-conversation corpora).** Ranking improvements help within the current candidate pools, but cannot invent missing document structure or span provenance.

### 6. What should be changed first, if anything?

**Introduce a parallel canonical evidence/document model + first adapter (e.g. Markdown sections)** while freezing conversation ingest. Do not rip out Memory retrieval.

### 7. Which parts must NOT be thrown away?

- `Memory` / `MemorySource` / consolidation versioning  
- `MemoryRetriever` + BM25 `RetrievalUnit` interfaces  
- Reorder-only CE / multi-channel candidate generation patterns  
- SQLite conversation + claim stores for the chat product  
- Provenance discipline (`source_message_ids` full tuple credit)

---

## Decision

```text
INTRODUCE A NEW CANONICAL EVIDENCE LAYER ALONGSIDE
THE EXISTING CONVERSATION PIPELINE  (OPTION D)
```
