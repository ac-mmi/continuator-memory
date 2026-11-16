# Canonical Evidence Adapter — Phase 1

**Scope:** Introduce in-memory canonical parse types, a minimal `FormatAdapter` contract, a shared materializer, and migrate Markdown behind that contract.  
**Not in scope:** HTML / DOCX / PDF adapters, schema migration, retrieval changes, conversation Memory, embeddings / rerankers / LLMs.

---

## 1. Architecture before

```text
Markdown file / text
        │
        ▼
ingest_markdown()
        ├── canonicalize_source_uri → document_id
        ├── parse_markdown_sections → MarkdownSection[]
        ├── split_section_text / build_retrieval_text / section_id_for
        ├── EvidenceUnit[]  (built inline in ingest.py)
        └── EvidenceStore.save_document → Document + EvidenceUnit
```

Markdown parsing, unit identity, and persistence were coupled inside `ingest.py`.

---

## 2. Architecture after

```text
Markdown file / text
        │
        ▼
ingest_markdown()   (compatibility facade)
        │
        ▼
MarkdownAdapter.parse(...)
        │
        ▼
CanonicalDocument
        └── CanonicalSection*   (flat list for Markdown; children supported)
        │
        ▼
materialize(...)
        ├── Document
        └── EvidenceUnit[]
        │
        ▼
EvidenceStore.save_document   (unchanged schema)
```

Retrieval continues to see only `EvidenceUnit` + `Document`. Conversation Memory is untouched.

---

## 3. CanonicalDocument design

In-memory only (`evidence/canonical.py`):

| Field | Role |
|---|---|
| `source_uri` | Stable source identity |
| `source_type` | e.g. `markdown` |
| `content_hash` | SHA-256 of source text |
| `title` | Optional document title |
| `metadata` | Adapter / ingest metadata (persisted on Document) |
| `sections` | Ordered top-level `CanonicalSection` tuple |

---

## 4. CanonicalSection design

In-memory only; **not** persisted as rows.

| Field | Role |
|---|---|
| `section_id` | Stable within-document key (`sec:…`) |
| `heading_path` | Ancestry titles; may be `()` |
| `title` | Leaf heading title when present |
| `heading_level` | `0` = preamble / no heading |
| `text` | Section body before unit split |
| `char_start` / `char_end` | Provenance when linearization is faithful |
| `parent_section_id` | Optional parent key |
| `children` | Nested sections (DFS materialization) |
| `metadata` | Format-specific provenance |

Markdown emits a **flat** ordered list (`children=()`) with `heading_path` encoding ancestry — identical order to Phase-1 `MarkdownSection` output. Nested `children` are supported for future adapters and covered by a DFS golden test.

Empty hierarchy is first-class (no fake headings).

---

## 5. FormatAdapter contract

`evidence/adapter.py` — minimal `Protocol`:

```text
FormatAdapter
  name: str
  source_types: frozenset[str]
  supports(path_or_uri, *, sniff?) -> bool
  parse(source, *, source_uri, metadata?) -> CanonicalDocument
```

Adapters must not write to the store, call search, download models, or invent hierarchy when absent.

`MarkdownAdapter` (`evidence/adapters/markdown.py`) is the only implementation.

---

## 6. Materializer responsibilities

`evidence/materialize.py`:

- Depth-first walk of `CanonicalSection` (flat or nested)
- Optional `max_chars` split via existing `split_section_text`
- Deterministic `evidence_id` = `ev:{document_id}:u{ordinal:04d}`
- `retrieval_text` via existing `build_retrieval_text`
- `section_id` + `#pNN` piece suffixes when split
- Char spans only for unsplit sections
- Unit metadata: `heading_level`, `piece_index`, `piece_count`
- Build persisted `Document` from canonical fields + `created_at`

---

## 7. Markdown migration path

| Before | After |
|---|---|
| `parse_markdown_sections` used directly in ingest | Still used inside `MarkdownAdapter` |
| `build_units_from_markdown` built units inline | Adapter.parse → `materialize_units` |
| `ingest_markdown` built Document + units inline | Adapter.parse → `materialize` → store |
| CLI `ingest-markdown` | Unchanged entry point |

Public APIs (`ingest_markdown`, `build_units_from_markdown`, ID helpers) remain import-compatible.

---

## 8. Exact compatibility invariants

1. Same `source_uri` → same `document_id`
2. Same structure → same evidence ordinals and `evidence_id`s
3. Same `heading_path`, `section_id`, `text`, `retrieval_text`
4. Same char spans when unsplit; `None` spans when size-split
5. Same unit metadata keys/values for Markdown (`heading_level`, `piece_*`)
6. Nested ATX headings, preamble, and no-heading docs behave as before
7. `max_section_chars` splitting unchanged
8. Re-ingest same URI → `replaced=True`, snapshot replace, stable ids
9. Document metadata still includes `adapter=markdown` and `original_path`
10. Empty `heading_path` remains valid (PDF-ready flat path)

Golden fixtures in `tests/unit/test_canonical_evidence_adapter_phase1.py` encode Phase-1 unit dicts captured before migration; drift fails loudly.

---

## 9. Tests run and results

```text
PYTHONPATH=src python -m pytest \
  tests/unit/test_canonical_evidence_adapter_phase1.py \
  tests/unit/test_evidence_markdown.py \
  tests/unit/test_evidence_eval.py -v
```

| Suite | Result |
|---|---|
| `test_canonical_evidence_adapter_phase1.py` | 11 passed |
| `test_evidence_markdown.py` | 10 passed |
| `test_evidence_eval.py` | 9 passed |
| **Total** | **30 passed** |

SciFact / retrieval ablations / LoCoMo were not run (out of scope).

---

## 10. What did NOT change

- `ev_documents` / `ev_units` schema (no Section table)
- BM25 / `search_evidence` / `RetrievalUnit` mapping
- Conversation Memory / Workspace ingestion
- Embeddings, rerankers, LLMs, model downloads
- HTML / DOCX / PDF ingestion
- CLI command names
- Markdown heading-stack rules in `parse_markdown_sections`
- Deterministic ID formulas and retrieval_text policy

---

## Status checklist

| Question | Answer |
|---|---|
| Markdown persisted identities changed? | **NO** |
| Retrieval behavior intentionally changed? | **NO** |
| Database schema changed? | **NO** |

---

## ONE recommended next step

**Implement an HTML `FormatAdapter` that parses recoverable `h1–h6` / semantic section structure into `CanonicalDocument` (flat or nested), reuses the shared materializer, and adds focused golden tests — without changing schema or retrieval.**
