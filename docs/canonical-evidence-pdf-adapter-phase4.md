# Canonical Evidence PDF Adapter — Phase 4 (Digital Text)

**Scope:** Digital-text PDF → `PDFAdapter` → `CanonicalDocument` → shared materializer → existing `Document` + `EvidenceUnit` store.  
**Not in scope:** OCR, scanned PDFs, typography-based heading inference, layout/multi-column reconstruction, schema/retrieval changes, Markdown/HTML/DOCX behavior changes.

---

## 1. Architecture findings

`CanonicalSection` already supports flat, heading-less content (`heading_path=()`, `heading_level=0`). PDF maps **one section per page with extractable text**, preserving page provenance in section/unit metadata — no schema change and no fake headings.

```text
PDF bytes
    → PDFAdapter.parse
    → CanonicalDocument
         └── CanonicalSection*   (flat; one per non-empty page)
    → materialize
    → Document + EvidenceUnit
```

---

## 2. PDF library

**Selected: `pypdf` (`pypdf>=5.0`)**

Why:

- Lightweight and maintained (PyPDF2 successor)
- Page-level `extract_text()` fits digital-text + page provenance
- No OCR / vision stack
- Single library for this phase (fixtures are handcrafted PDFs in tests)

---

## 3. Extraction behavior

- Binary PDF input (path bytes or raw bytes)
- Extract text **page by page** via `pypdf`
- Light whitespace normalization only (collapse runs of spaces; keep useful newlines)
- Empty / whitespace-only pages are **skipped** as sections and listed in document metadata `empty_pages`
- Content hash = SHA-256 of raw PDF bytes
- Title = PDF `/Title` metadata when present, else path stem
- `char_start` / `char_end` remain `None`

---

## 4. Page provenance

Each non-empty page section:

| Field | Value |
|---|---|
| `section_id` | `sec:page-{N}` (1-based) |
| `heading_path` | `()` |
| `heading_level` | `0` |
| `metadata.page_number` | 1-based page number |
| `metadata.pdf_role` | `page` |

Document metadata includes `parser=pypdf`, `page_count`, `pages_with_text`, and optional `empty_pages`.

---

## 5. Empty / scanned PDF behavior

If **no** page yields extractable digital text, parse raises:

`PDFNoExtractableTextError` — *"PDF has no extractable digital text (scanned/image-only PDFs require OCR, which is not supported)"*

CLI `ingest-pdf` prints the error and exits non-zero. No silent OCR pretend.

---

## 6. Known limitations

- **Multi-column / complex layout:** uses `pypdf`'s default text order; not a layout reconstructor. Reading order may be wrong on complex pages.
- **No heading inference** from font size/bold/position (intentional).
- **No OCR**, images, tables-as-structure, headers/footers as separate objects.
- Weak structure (page buckets) is preferred over false semantic hierarchy.

---

## 7. Tests

```text
PYTHONPATH=src python -m pytest \
  tests/unit/test_evidence_pdf_adapter.py \
  tests/unit/test_evidence_docx_adapter.py \
  tests/unit/test_evidence_html_adapter.py \
  tests/unit/test_canonical_evidence_adapter_phase1.py \
  tests/unit/test_evidence_markdown.py -v
```

| Suite | Result |
|---|---|
| `test_evidence_pdf_adapter.py` | 11 passed |
| `test_evidence_docx_adapter.py` | 13 passed |
| `test_evidence_html_adapter.py` | 11 passed |
| `test_canonical_evidence_adapter_phase1.py` | 11 passed |
| `test_evidence_markdown.py` | 10 passed |
| **Total** | **56 passed** |

---

## 8. What did NOT change

- Database schema
- Retrieval / BM25 / semantic / embeddings / reranking
- Markdown / HTML / DOCX adapters and identities
- Shared materializer identity rules (PDF only adds page metadata on units)

---

## ONE recommended next step

**Add a thin multi-format ingest dispatcher (path/extension → Markdown/HTML/DOCX/PDF adapter) with focused routing tests — without changing retrieval or schema — so callers need not pick the facade manually.**
