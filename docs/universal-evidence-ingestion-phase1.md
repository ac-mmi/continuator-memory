# Universal Evidence Ingestion — Phase 1

**Scope:** Markdown-only parallel evidence layer.  
**Constraint honored:** conversation `Memory` pipeline unchanged; no Memory migration; no PDF/HTML/code; no LlamaIndex; no model downloads.

---

## 1. Existing architecture inspected

From the audit (`docs/universal-indexing-architecture-audit.md`) and direct code inspection:

| Area | Observed | Implication for Phase 1 |
|---|---|---|
| Product spine | `Conversation` → claims → `Memory` + `MemorySource` | Do not route Markdown through Message/Memory |
| Dense retrieval | `MemoryRetriever` hard-coded to memories | Do not force Markdown into MemoryRetriever yet |
| Sparse retrieval | `RetrievalUnit` + `BM25Index` (`v2/bm25.py`) — opaque ids | **Best integration point** |
| Persistence | Product `Store` + additive stores (`KnowledgeStore` `kd_*`, `MemoryStore` `v2_*`) | Add additive `ev_*` tables the same way |
| Knowledge path | LLM Markdown docs from chat windows (`knowledge/`) | Isolated experiment; **not** reused as the universal model |
| Workspace modes | Memory-only `_retrieve_evidence` | Leave modes alone; expose parallel CLI/API |

---

## 2. What was added

New package: `src/continuator_memory/evidence/`

| Module | Role |
|---|---|
| `models.py` | `Document`, `EvidenceUnit`, `IngestedDocument`, `EvidenceHit` |
| `markdown.py` | Structure-first ATX heading parse + retrieval text + optional size split |
| `store.py` | `EvidenceStore` — `ev_documents` / `ev_units` |
| `ingest.py` | `ingest_markdown(...)` |
| `search.py` | `search_evidence(...)` via BM25 `RetrievalUnit` |
| `__init__.py` | Public exports |

Also:

- CLI: `ingest-markdown`, `search-evidence`
- Sample: `docs/example.md`
- Tests: `tests/unit/test_evidence_markdown.py`

---

## 3. Why the integration point was chosen

**Chosen:** map `EvidenceUnit` → existing `RetrievalUnit` and score with `BM25Index`.

Reasons:

1. `RetrievalUnit` is already content-agnostic (`id`, `retrieval_content`, caller-owned provenance ids).
2. Avoids rewriting `MemoryRetriever` or Workspace mode dispatch.
3. Avoids a second unrelated retrieval library.
4. Dense embedding over EvidenceUnits can reuse the same unit identity later without schema churn.

**Not chosen now:** folding Markdown into `Memory` / contextual PREV/TARGET/NEXT (conversation-specific).

---

## 4. Canonical Document model

```text
Document
  document_id   — doc:{sha256(source_uri)[:24]}
  source_uri    — file:// absolute URI (or provided URI)
  source_type   — "markdown" (Phase 1); constants reserved for text/pdf/html/code/conversation
  title         — first H1, else filename stem
  content_hash  — sha256 of raw source bytes/text
  metadata      — adapter + original_path (+ caller extras)
  created_at    — ingest timestamp
```

---

## 5. Canonical EvidenceUnit model

```text
EvidenceUnit
  evidence_id     — ev:{document_id}:u{ordinal:04d}
  document_id
  ordinal         — structural order after parse
  text            — section body (evidence content)
  retrieval_text  — deterministic "H1 > H2\n\nbody"
  heading_path    — tuple ancestry
  section_id      — sec:slug/... for section-level gold
  char_start/end  — source spans when unsplit
  metadata        — heading_level, piece_index/count
```

Future evaluation can gold-match at:

- document: `document_id` / `source_uri`
- section: `section_id` / `heading_path`
- unit/span: `evidence_id` + `char_start`/`char_end`

---

## 6. Markdown structural ingestion flow

```text
source .md
  → read text
  → canonicalize source_uri → document_id
  → parse_markdown_sections (heading stack)
  → optional split_section_text(max_chars)   # default: None (no split)
  → EvidenceUnit per piece
  → EvidenceStore.save_document (replace by source_uri)
```

Example ancestry preserved:

```text
Project Overview
└── Authentication
    └── Token Refresh   ← "Access tokens expire after one hour."
```

---

## 7. Provenance model

Every unit resolves via stable IDs (not display strings alone):

```text
evidence_id
  → EvidenceUnit (text, heading_path, section_id, char spans)
  → document_id
  → Document (source_uri, source_type, content_hash, title)
```

Helper: `resolve_provenance(store, evidence_id)`.

BM25 `RetrievalUnit.source_message_ids` carries `(document_id, section_id, evidence_id)` as opaque caller-owned ids for future scoring — not conversation message ids.

---

## 8. Identity / re-ingestion behavior

| Case | Behavior |
|---|---|
| Same `source_uri` re-ingested | Snapshot **replace** (delete prior document+units, insert fresh). `replaced=True` |
| `document_id` | Stable for a given canonical `source_uri` |
| `evidence_id` | Deterministic from `(document_id, ordinal)`. Identical content+structure → identical ids |
| Structure change | Ordinals/ids after the change point may shift; prior snapshot is removed |

No silent random UUIDs.

---

## 9. Retrieval integration now

| Capability | Status |
|---|---|
| Persist Document + EvidenceUnit | Yes |
| BM25 search via `RetrievalUnit` | Yes (`search_evidence` / CLI `search-evidence`) |
| Workspace conversation modes | Unchanged |
| Dense MemoryRetriever for evidence | Not wired (next step candidate) |
| CE rerank over evidence | Not wired |
| Union of Memory + Evidence candidates | Not wired |

**Smallest next integration step:** add a Workspace (or API) method that unions Memory BM25/dense hits with Evidence BM25 hits by opaque candidate id, without migrating Memory rows.

---

## 10. Intentionally out of scope

- PDF / HTML / code / OCR adapters
- LlamaIndex or plugin framework
- Migrating `Memory` → `EvidenceUnit`
- Changing LoCoMo / conversation benchmarks
- Embedding downloads / dense evidence index
- Sophisticated recursive chunkers
- Full document-level evaluation harness

Extension point already present: `split_section_text(..., max_chars=...)`.

---

## 11. Tests run and results

```text
PYTHONPATH=src python -m pytest tests/unit/test_evidence_markdown.py -q
→ 10 passed

PYTHONPATH=src python -m pytest tests/unit/test_v2_memory_store.py tests/unit/test_contextual_bm25.py -q
→ 14 passed
```

CLI smoke:

```text
PYTHONPATH=src python -m continuator_memory.cli --db /tmp/continuator-evidence-phase1.db ingest-markdown docs/example.md
→ 4 evidence units; Token Refresh ancestry preserved

PYTHONPATH=src python -m continuator_memory.cli --db /tmp/continuator-evidence-phase1.db search-evidence "How do access tokens expire?" -k 3
→ #1 Token Refresh / Access tokens expire after one hour.
```

---

## Commands

```bash
# Ingest
PYTHONPATH=src python -m continuator_memory.cli --db /tmp/evidence.db ingest-markdown docs/example.md

# Search (BM25, no model download)
PYTHONPATH=src python -m continuator_memory.cli --db /tmp/evidence.db search-evidence "How do access tokens expire?" -k 5 --json
```

Python API:

```python
from continuator_memory.evidence import EvidenceStore, ingest_markdown, search_evidence
from continuator_memory.store import Store

with Store("evidence.db") as product:
    store = EvidenceStore(product.connection)
    doc = ingest_markdown("docs/example.md", store)
    hits = search_evidence(store, "How do access tokens expire?", top_k=10)
```
