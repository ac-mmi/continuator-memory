# Canonical Evidence Adapter Architecture Audit

**Scope:** Audit only — propose the smallest safe format-adapter contract.  
**Not in scope:** HTML / DOCX / PDF implementation, retrieval changes, Memory/Workspace changes, embeddings/rerankers, model downloads.

---

## 1. Current architecture map

```text
Markdown file / text
        │
        ▼
ingest_markdown()                    [evidence/ingest.py]
        │
        ├── canonicalize_source_uri → document_id
        ├── parse_markdown_sections  [evidence/markdown.py]
        │         └── MarkdownSection (in-memory only)
        ├── split_section_text (optional)
        ├── EvidenceUnit[]           [evidence/models.py]
        └── EvidenceStore.save_document
                  ├── ev_documents
                  └── ev_units

Search path (unchanged conversation Memory path):

EvidenceStore.list_units / get_units
        │
        ▼
units_to_retrieval_units → RetrievalUnit
        │
        ▼
BM25Index.search → EvidenceHit
```

**Parallel experiment paths (not production ingest):**

| Path | Role |
|---|---|
| SciFact `adapter.py` | Builds `Document` + sentence `EvidenceUnit`s from abstracts (`source_type=text`) |
| Semantic corpus cache / CE ablation | Read already-built EvidenceUnits; do not redefine ingest |

**Conversation path remains isolated:** `Memory` / `MemorySource` / Workspace modes are untouched.

---

## 2. Current Document and EvidenceUnit models

### 2.1 `Document` (persisted: `ev_documents`)

| Field | Meaning |
|---|---|
| `document_id` | `doc:{sha256(source_uri)[:24]}` |
| `source_uri` | Stable identity (e.g. `file://…`, `scifact://…`) |
| `source_type` | Constants: `markdown`, `text`, `pdf`, `html`, `code`, `conversation` — **only markdown implemented** |
| `content_hash` | SHA-256 of source text |
| `title` | Optional |
| `metadata` | Free-form dict (`adapter`, paths, etc.) |
| `created_at` | Ingest timestamp |

### 2.2 `EvidenceUnit` (persisted: `ev_units`)

| Field | Meaning |
|---|---|
| `evidence_id` | `ev:{document_id}:u{ordinal:04d}` |
| `document_id` | Parent document |
| `ordinal` | Structural order after parse |
| `text` | Body / evidence content |
| `retrieval_text` | Indexed string (Markdown: `"H1 > H2\n\nbody"`; SciFact: `title\n\nsentence`) |
| `heading_path` | Tuple ancestry (may be empty) |
| `section_id` | Stable section key (`sec:…` or SciFact `sec:sNNNN`) |
| `char_start` / `char_end` | Source spans when known |
| `metadata` | e.g. `heading_level`, piece indices, SciFact ids |

### 2.3 Related types

| Type | Persisted? | Role |
|---|---|---|
| `MarkdownSection` | **No** | In-memory MD parse node: `heading_path`, `text`, spans, `heading_level` |
| `IngestedDocument` | No | Return wrapper: document + units + replaced flag |
| `EvidenceHit` | No | Search result |

---

## 3. Is there already a Section concept?

**Yes, implicitly — not as a first-class persisted model.**

| Mechanism | What it encodes |
|---|---|
| `MarkdownSection` | Explicit in-memory section during MD parse |
| `EvidenceUnit.heading_path` | Ancestry of titles from root → leaf |
| `EvidenceUnit.section_id` | Slugified path (or preamble / piece suffix) |
| `metadata["heading_level"]` | Leaf heading depth (MD only) |

There is **no** `ev_sections` table, no `Section` dataclass in `models.py`, and no parent/child section foreign keys.

Hierarchy today is **denormalized onto each EvidenceUnit** via `heading_path` / `section_id`.

---

## 4. Can current models represent hierarchical structure cleanly?

### What works today

- Markdown heading stack → ordered sections → units with ancestry in `heading_path`.
- Retrieval stays format-agnostic: only `retrieval_text` + `evidence_id` matter to BM25.
- Evaluation can gold-match at document / section_id / evidence_id / char span.
- Empty `heading_path` already supports “no hierarchy” (preamble, SciFact sentences).

### Limits

1. **No section tree as an object** — sibling/parent sections cannot be queried without reconstructing from unit paths.
2. **`heading_path` is title-based, not id-based** — duplicate heading titles in one doc collide in `section_id` slugs.
3. **No typed provenance beyond char spans** — PDF page/bbox, HTML CSS selectors, DOCX paragraph ids have nowhere first-class to live (only free-form `metadata`).
4. **No adapter boundary** — `ingest_markdown` mixes URI identity, MD parse, unit materialization, and store write. HTML/DOCX/PDF would copy that tangle unless an adapter contract exists.
5. **`retrieval_text` policy is adapter-local** — MD uses ancestry join; SciFact uses title+sentence. That is fine if documented as adapter-owned, but needs a shared contract so search never embeds format logic.

**Verdict:** Hierarchy is *representable enough for retrieval* via `heading_path`, but *not clean enough as a multi-format parse intermediate*. The gap is the missing adapter → canonical parse step, not missing retrieval schema.

---

## 5. What HTML / DOCX / PDF need that current models cannot preserve cleanly

| Need | Markdown today | HTML | DOCX | Digital text PDF |
|---|---|---|---|---|
| Source identity / URI | Yes (`source_uri`) | Yes | Yes | Yes |
| Title | Yes | Often (`<title>` / H1) | Often (core props / first heading) | Weak / heuristic |
| Heading hierarchy | Explicit `#` | Often (`h1–h6`, sometimes `<section>`) | Often (Heading styles) | Often **absent** |
| Ordered blocks | Line/section order | DOM order | Paragraph order | Reading-order heuristic |
| Char offsets in original bytes | Yes (MD text) | Fragile (normalized text ≠ raw HTML) | Not char-in-file; para/run ids | Page + bbox more natural |
| Page / layout | N/A | N/A | Limited | Common |
| Semantic vs layout structure | Structure = headings | Mixed | Mixed | Layout ≠ semantics |
| Empty / flat hierarchy | Preamble `()` | Possible | Possible | **Default case** |

**Cannot preserve cleanly without either:**

- stuffing format-specific location into `metadata` (acceptable short-term), or
- a small typed provenance extension later (page, bbox, xpath, paragraph_id) — **not required for the first adapter contract**.

**Must not force:** fake headings for PDFs when extraction gives only pages/paragraphs.

---

## 6. Section design decision

### Options

| Option | Meaning |
|---|---|
| **A** | First-class **persisted** `Section` rows |
| **B** | **In-memory** canonical parse structure (adapters → sections → units) |
| **C** | Remain **only** `heading_path` / metadata (no Section type at all) |
| **D** | Something else |

### Recommendation: **B** (primary), keep **C** as the persistence encoding

**Choose B for the adapter boundary; keep C for storage.**

Rationale:

1. **Smallest safe refactor** — no `ev_sections` migration; retrieval and SciFact eval keep working.
2. **Matches existing MD reality** — `MarkdownSection` already is B; generalize it.
3. **Graceful degradation** — PDF adapter can emit one flat “body” section with `heading_path=()`; no fake tree.
4. **Avoids A for now** — persisted sections add join complexity and dual-write risk before a second format exists.
5. **Rejects pure C alone** — without an explicit parse Section type, each format will reinvent MD’s stack logic inside ingest functions.

**Conceptual shape:**

```text
CanonicalParse (in-memory, adapter output)
  DocumentFields (uri, type, title, hash, metadata)
  └── Section*          ← ordered; optional children OR flat list with path
        └── body blocks → EvidenceUnit*   (materialized for store/search)

Persisted (unchanged schema):
  Document
  └── EvidenceUnit*     (heading_path / section_id encode ancestry when present)
```

**Not required as persisted:**

```text
Document → Section* → EvidenceUnit*   (A)
```

That tree is the **parse** model, not the **store** model, until a concrete consumer needs section-level queries independent of units.

---

## 7. Recommended canonical representation

### 7.1 Persisted (keep)

`Document` + `EvidenceUnit` as today.

Invariants:

- Stable `source_uri` → `document_id`
- Deterministic `evidence_id` from `(document_id, ordinal)` unless an adapter documents a stronger bijection (SciFact sentence index)
- `retrieval_text` is the only text scored by BM25
- Empty `heading_path` is valid

### 7.2 In-memory canonical parse (add)

Minimal types (names illustrative):

```text
CanonicalSection
  section_id: str                 # stable within document
  heading_path: tuple[str, ...]   # may be ()
  heading_level: int              # 0 = no heading / flat
  text: str                       # section body (pre-unit-split)
  char_start / char_end: optional # when meaningful for this format
  parent_section_id: str | None   # optional; None for flat/PDF
  metadata: dict                  # format-specific provenance

CanonicalDocument
  source_uri, source_type, title, content_hash, metadata
  sections: tuple[CanonicalSection, ...]   # document order
```

Adapters **must** produce `CanonicalDocument`.  
A shared **materializer** turns sections → `EvidenceUnit`s (optional size split, `retrieval_text` policy).

### 7.3 Is EvidenceUnit + heading_path sufficient for retrieval?

**Yes** for BM25 / semantic / CE experiments already in-repo.

**No** as the sole *parse* API for multi-format ingest — adapters need an intermediate section list so they do not invent units ad hoc.

---

## 8. Minimal adapter contract

```text
FormatAdapter
  name: str                      # "markdown" | "html" | …
  source_types: frozenset[str]   # which Document.source_type values

  supports(path_or_uri, *, sniff?) -> bool

  parse(source_bytes_or_text, *, source_uri, metadata?) -> CanonicalDocument
      # Format-specific. No store writes. No retrieval.
      # May return flat sections when hierarchy is unavailable.

Materializer (shared, not format-specific)
  materialize(CanonicalDocument, *, max_section_chars?) -> list[EvidenceUnit]
      # ordinal assignment, evidence_id, retrieval_text, spans

Ingest orchestration (thin)
  ingest(adapter, source, store) -> IngestedDocument
      # parse → materialize → Document row → store.save_document
```

### Contract responsibilities

| Concern | Owner |
|---|---|
| Bytes/text decode, format tree walk | Adapter |
| Heading / style / DOM → `CanonicalSection` | Adapter |
| Fake hierarchy when absent | **Forbidden** — use flat section(s) |
| `retrieval_text` construction | Materializer (policy knobs OK; default = ancestry join like MD) |
| Persistence | Orchestrator + `EvidenceStore` |
| BM25 / embeddings / CE | Existing search / experiment code only |

### Provenance policy (minimal)

- Prefer `char_start`/`char_end` when the adapter’s text is a faithful linearization of the source.
- Otherwise leave spans `None` and put format locators in `CanonicalSection.metadata` / unit `metadata` (e.g. `page`, `paragraph_id`) without schema change.

---

## 9. Compatibility with existing Markdown ingestion

| Behavior | After adapter extraction |
|---|---|
| `parse_markdown_sections` | Becomes Markdown adapter internals (or maps `MarkdownSection` → `CanonicalSection`) |
| `ingest_markdown` | Thin wrapper: MarkdownAdapter.parse → materialize → store |
| `ev_*` schema | **Unchanged** |
| Unit ids for same MD content | **Unchanged** if ordinalization preserved |
| CLI `ingest-markdown` | Keep name; call new orchestrator |
| SciFact adapter | Remains a specialized path; optionally later implement `FormatAdapter` for `text` — **not required now** |
| Conversation Memory | Untouched |

Risk to avoid: changing `build_retrieval_text` or ordinal rules while extracting the adapter — would invalidate eval identity.

---

## 10. Gaps summary

| Gap | Severity | Fix in this refactor? |
|---|---|---|
| No FormatAdapter interface | High for multi-format | **Yes** |
| Section only as MD-private dataclass | Medium | **Yes** (canonicalize in-memory) |
| No persisted Section | Low for retrieval | **No** (defer) |
| Typed page/bbox/xpath provenance | Medium for PDF/HTML UX | **No** (metadata for now) |
| Duplicate heading slug collisions | Low–medium | Document; fix later if needed |
| SOURCE_TYPE_* reserved but unused | Low | Keep; enable per adapter |

---

## 11. Migration plan — smallest steps

1. **Audit** (this document) — done.
2. **Introduce** `CanonicalSection` / `CanonicalDocument` + `FormatAdapter` protocol + shared materializer **without** new formats.
3. **Refactor Markdown** to implement the adapter; keep `ingest_markdown` as compatibility facade; golden tests for unit ids / heading_path / retrieval_text.
4. **Add HTML adapter** (first non-MD format with recoverable headings).
5. **Add DOCX adapter** (heading styles → path).
6. **Add PDF digital-text adapter** (flat/page-aware sections; no fake headings).
7. **Only if needed:** persist `Section` or typed provenance columns.

---

## 12. Risks and invariants

### Invariants

1. Conversation Memory ingest/search paths never import format adapters for chat.
2. `EvidenceStore` schema remains Document + EvidenceUnit until a measured need for `ev_sections`.
3. Adapters must not call search, embeddings, or downloads.
4. Empty hierarchy is first-class (`heading_path=()`).
5. Re-ingest same `source_uri` still replaces by URI; `document_id` stable.
6. Materializer ordinal order = document reading order.

### Risks

| Risk | Mitigation |
|---|---|
| Accidental change to MD unit identity | Golden tests on fixture `.md` before/after |
| Over-abstracting “universal chunker” | Materializer stays dumb; adapters own structure |
| PDF forced into fake H1/H2 | Explicit flat-section policy in contract tests |
| Retrieval coupled to format | Search continues to see only EvidenceUnits |

---

## 13. Recommended implementation order

1. Adapter contract + in-memory canonical section types + materializer  
2. Markdown adapter migration (facade-compatible)  
3. HTML adapter  
4. DOCX adapter  
5. PDF digital-text adapter (flat-first)  
6. Revisit persisted Section / typed provenance **only after** two+ formats share pain

---

## 14. ONE recommended next implementation step

**Implement the in-memory `CanonicalDocument` / `CanonicalSection` types, a `FormatAdapter` protocol, a shared EvidenceUnit materializer, and migrate Markdown to that contract behind the existing `ingest_markdown` facade — with golden tests proving EvidenceUnit identity and `heading_path` / `retrieval_text` are unchanged. Do not add HTML/DOCX/PDF yet; do not change `ev_*` schema or retrieval.**
