# Canonical Evidence DOCX Adapter — Phase 3

**Scope:** DOCX → `DOCXAdapter` → `CanonicalDocument` → shared materializer → existing `Document` + `EvidenceUnit` store.  
**Not in scope:** PDF, schema changes, retrieval changes, Markdown/HTML behavior changes, embeddings / rerankers / LLMs.

---

## 1. Architecture findings

Existing path already supports nested or flat `CanonicalSection` trees. HTML emits nested children; Markdown emits flat lists with `heading_path`. DOCX follows the **HTML nested** pattern and reuses the shared materializer unchanged.

```text
DOCX (.docx bytes)
    → DOCXAdapter.parse
    → CanonicalDocument (nested CanonicalSection*)
    → materialize
    → Document + EvidenceUnit
    → EvidenceStore
```

Binary note: DOCX is a ZIP package. `ingest_docx` reads `path.read_bytes()` (not UTF-8 text).

---

## 2. DOCX parsing behavior

- Library: **python-docx** (`python-docx>=1.1` in `pyproject.toml`)
- Adapter: `src/continuator_memory/evidence/adapters/docx.py`
- Facade: `ingest_docx(...)` / `build_units_from_docx(...)`
- CLI: `ingest-docx`
- Source type: `docx` (`SOURCE_TYPE_DOCX`, included in `IMPLEMENTED_SOURCE_TYPES`)
- Title: core properties title → first Heading 1 → first paragraph → path stem
- Content hash: SHA-256 of raw `.docx` bytes
- Spans: `char_start` / `char_end` are `None` (not a linear text source)

Body blocks are walked in document order via `w:p` / `w:tbl` children (so tables interleave correctly with paragraphs).

---

## 3. Heading hierarchy behavior

Primary signal: paragraph styles **Heading 1–6** (case-insensitive match).

Stack rules (Markdown/HTML-equivalent):

1. Heading at level N closes open sections with level ≥ N
2. New section path = parent path + (title,)
3. Attached as child of parent, or as a document root section
4. Sibling Heading 2 under the same Heading 1 do not nest into each other

---

## 4. Preamble / no-heading behavior

- Paragraphs (and tables) before the first heading → preamble section (`heading_path=()`, `section_id=sec:preamble`)
- No Heading styles → one flat preamble section; ingest succeeds
- No fake headings invented

---

## 5. List / table behavior and limitations

**Lists:** Item text is preserved as ordinary paragraph body text in reading order. No special list schema, numbering reconstruction, or nesting model.

**Tables:** Each table is serialized into readable ordered text:

```text
cell | cell | cell
cell | cell | cell
```

Merged-cell duplicate text from python-docx is lightly collapsed when adjacent cells repeat. No table schema, no cell provenance model.

**Out of scope this phase:** tracked changes, comments, revisions, images/OCR, headers/footers, fields, layout/page coordinates.

---

## 6. Tests

```text
PYTHONPATH=src python -m pytest \
  tests/unit/test_evidence_docx_adapter.py \
  tests/unit/test_evidence_html_adapter.py \
  tests/unit/test_canonical_evidence_adapter_phase1.py \
  tests/unit/test_evidence_markdown.py -v
```

| Suite | Result |
|---|---|
| `test_evidence_docx_adapter.py` | 13 passed |
| `test_evidence_html_adapter.py` | 11 passed |
| `test_canonical_evidence_adapter_phase1.py` | 11 passed |
| `test_evidence_markdown.py` | 10 passed |
| **Total** | **45 passed** |

---

## 7. What did NOT change

- `ev_documents` / `ev_units` schema
- Shared materializer identity / retrieval_text policy
- MarkdownAdapter and HTMLAdapter behavior
- BM25 / semantic / RRF / CE / SciFact
- Conversation Memory pipeline
- Embeddings / rerankers / LLMs
- PDF ingestion (not implemented)

---

## ONE recommended next step

**Implement a digital-text PDF `FormatAdapter` that extracts readable text in reading order into flat (or weakly structured) `CanonicalDocument` sections without inventing fake headings, reuses the shared materializer, and adds focused tests — without changing schema or retrieval.**
