# Canonical Evidence HTML Adapter — Phase 2

**Scope:** HTML → `HTMLAdapter` → `CanonicalDocument` → shared materializer → existing `Document` + `EvidenceUnit` store.  
**Not in scope:** DOCX / PDF, schema changes, retrieval changes, conversation Memory, embeddings / rerankers / LLMs.

---

## 1. HTML parsing approach

- Library: **BeautifulSoup4** with the stdlib backend **`html.parser`** (no lxml).
- Declared in `pyproject.toml` as `beautifulsoup4>=4.12`.
- Adapter: `src/continuator_memory/evidence/adapters/html.py` (`HTMLAdapter`).
- Facade: `ingest_html(...)` / `build_units_from_html(...)` in `evidence/ingest.py`.
- CLI: `ingest-html` (mirrors `ingest-markdown`).

Flow:

```text
HTML source
    → HTMLAdapter.parse
    → CanonicalDocument (nested CanonicalSection*)
    → materialize
    → Document + EvidenceUnit
    → EvidenceStore
```

---

## 2. Heading hierarchy algorithm

Primary structure signal: **`h1`–`h6`**.

Stack rules (Markdown-equivalent):

1. Walk the chosen content root in document order.
2. On heading at level **N** with title **T**:
   - close open sections with level ≥ N
   - open a new section with `heading_path = parent_path + (T,)`
   - attach as a **child** of the new parent (or as a root section)
3. Body events accumulate on the current open section (or preamble).

HTML emits a **nested** `CanonicalSection` tree (`children` populated). The shared materializer DFS-flattens to EvidenceUnits in reading order.

---

## 3. Content before headings

Text in `p` / `li` before the first heading becomes a preamble section:

- `heading_path=()`
- `heading_level=0`
- `section_id=sec:preamble`

No invented headings.

---

## 4. No-heading HTML

If the document has no `h1`–`h6`, all collected `p` / `li` text becomes one flat preamble section. Ingest succeeds; hierarchy is empty.

---

## 5. Ignored / noise elements

Decomposed before structure walk:

- `script`
- `style`
- `noscript`
- `template`

No LLM cleaner; no general boilerplate heuristics.

Readable body tags collected: **`p`**, **`li`**.

---

## 6. Metadata / provenance preserved

| Field | Source |
|---|---|
| `source_uri` | ingest / adapter |
| `source_type` | `html` |
| `content_hash` | SHA-256 of raw HTML text |
| `title` | `<title>` if present, else first `h1`, else path stem |
| Document `metadata.adapter` | `html` |
| Document `metadata.original_path` | ingest path |
| Document `metadata.content_root` | `article` / `main` / `body` / `document` |
| Document `metadata.parser` | `beautifulsoup4:html.parser` |
| Document `metadata.html_title` | raw `<title>` when present |
| Section `metadata.html_tag` / `html_heading_level` | heading tag info |
| Unit spans | `char_start` / `char_end` are **`None`** (HTML is not a faithful char-linear source like Markdown) |

Semantic `article` / `main` may be chosen as content root; they do **not** create a second heading hierarchy. Arbitrary nested `<section>` / `<div>` are not used as structural parents.

---

## 7. Relationship to CanonicalDocument / CanonicalSection

- No change to canonical field definitions or materializer identity rules.
- HTML uses nested `children`; Markdown remains flat-with-`heading_path`.
- Both produce the same persisted shape: `Document` + `EvidenceUnit*` with `heading_path` / `section_id` / `retrieval_text`.

---

## 8. Tests run and results

```text
PYTHONPATH=src python -m pytest \
  tests/unit/test_evidence_html_adapter.py \
  tests/unit/test_canonical_evidence_adapter_phase1.py \
  tests/unit/test_evidence_markdown.py -v
```

| Suite | Result |
|---|---|
| `test_evidence_html_adapter.py` | 11 passed |
| `test_canonical_evidence_adapter_phase1.py` | 11 passed |
| `test_evidence_markdown.py` | 10 passed |
| **Total** | **32 passed** |

Coverage includes: single heading, nested h1→h2→h3, preamble, no-heading, lists, noise tags, re-ingest identity, ordering, MD↔HTML structural equivalence, Markdown unchanged.

---

## 9. What did NOT change

- `ev_documents` / `ev_units` schema (no Section table)
- Shared materializer identity / retrieval_text policy (except consuming HTML sections)
- MarkdownAdapter behavior and Markdown persisted identities
- BM25 / semantic / RRF / CE / SciFact paths
- Conversation Memory / Workspace ingestion
- Embeddings, rerankers, LLMs, model downloads

---

## Status checklist

| Question | Answer |
|---|---|
| Database schema changed? | **NO** |
| Retrieval changed? | **NO** |
| Markdown behavior changed? | **NO** |

---

## ONE recommended next step

**Implement a DOCX `FormatAdapter` that maps Word heading styles + paragraphs into `CanonicalDocument` (nested or flat with `heading_path`), reuses the shared materializer, and adds focused golden tests — without changing schema or retrieval.**
