# Hierarchical Document → Section → Evidence Experiment

**Status:** experiment-only. Production `search_evidence` and DB schema are unchanged.

## Question this experiment answers

Does **Document → Section → Evidence** retrieval improve evidence discovery/ranking
compared with the existing flat pipeline on documents that have real section
hierarchy — and **at which stage** do failures occur?

---

## Phase A — Benchmark audit

### What already exists in-repo

| Asset | Hierarchy? | Query→evidence gold? | Verdict |
|---|---|---|---|
| `benchmarks/evidence_retrieval/corpus/` (4 Markdown docs, 40 questions) | Yes — nested `heading_path` / `section_id` after ingest | Yes — `document_key` + `heading_path` → exact `EvidenceUnit` | **Use for this experiment** |
| SciFact (`claims_dev` / abstracts) | No reusable sections (unique `section_id` per sentence) | Yes (claim→sentence) | Unsuitable for a true section stage |
| `hard-evidence-retrieval-benchmark-plan.md` | Proposed, not authored | Planned | Not available yet |
| Adapter unit fixtures (MD/HTML/DOCX/PDF) | Structure tests only | No retrieval Q/A set | Architecture unit tests only |

**Conclusion:** Do **not** invent a new public dataset. The checked-in Phase-2 corpus
already satisfies:

1. Real section hierarchy (nested headings → CanonicalSection → EvidenceUnit)
2. Queries mapped to evidence via stable gold (`document_key` + `heading_path`)
3. Hierarchy preserved as Document → Section → EvidenceUnit

### Separation of concerns

| Track | Purpose | Material |
|---|---|---|
| **Architecture / unit testing** | Enforce stage scoping, identity preservation, empty/flat/nested edge cases | Synthetic in-memory stores in `hierarchical_dse/tests/` |
| **Realistic retrieval evaluation (small)** | Measure stage recalls + compare to flat BM25→CE on sectioned docs | Existing 4-doc / 40-Q corpus |
| **Large-scale / hard retrieval** | Stress competing siblings, multi-evidence, etc. | Out of scope here (see hard-evidence plan); requires authored fixtures later |

SciFact remains the right fixture for **flat** retrieval / expansion experiments, not for validating a section stage.

---

## Phase B — Experimental pipeline

```text
Query
  ↓
Global Document Retrieval          (BM25 over document pseudo-texts)
  ↓
Top-K Documents
  ↓
Section Retrieval                  (BM25 over section pseudo-texts
  ↓                                 only within selected documents)
Top-K Sections
  ↓
Evidence Retrieval                 (BM25 over EvidenceUnits
  ↓                                 only within selected sections)
Candidate pool
  ↓
Existing Cross-Encoder reranker    (fixed-pool; no new candidates)
  ↓
Final Top-K EvidenceUnits
```

### Flat baseline (same queries / gold)

```text
Query → BM25 over all EvidenceUnits → candidate pool → same CE → Top-K
```

This mirrors the local corpus’s validated flat BM25 path, with the same CE used
in SciFact union→CE experiments (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

### Constraints respected

- Reuses `Document`, `EvidenceUnit`, `heading_path`, `section_id`
- No schema changes
- No production `search_evidence` changes (hierarchical stages use experiment-local `BM25Index`)
- No duplicated ingestion — uses existing `ingest_markdown` / corpus ingest helpers
- Isolated under `benchmarks/evidence_retrieval/hierarchical_dse/`
- Evidence identities preserved end-to-end (CE cannot add/drop ids)
- No query rewriting, HyDE, LLM agents, synonym dictionaries, multi-hop, or production wiring

---

## Required diagnostics (per query)

| Field | Meaning |
|---|---|
| `gold_document_discovered` | Gold `document_id` in Top-K documents |
| `gold_section_discovered` | Gold `(document_id, section_id)` in Top-K sections |
| `gold_evidence_discovered` | Gold `evidence_id` in hierarchical candidate pool |
| `gold_evidence_candidate_pool_rank` | 1-based rank in candidate pool |
| `final_ce_rank` | 1-based rank after CE |

### Aggregate metrics

- Document Recall@K  
- Section Recall@K  
- Evidence Candidate Recall@K  
- Final Evidence Hit@K (hierarchical vs flat)

Plus failure-locus buckets: document miss → section miss after doc → evidence miss after section → rerank miss after candidate.

---

## How to run

```bash
python -m benchmarks.evidence_retrieval.hierarchical_dse.run \
  -o benchmarks/results/evidence_retrieval/hierarchical_dse
```

Optional: `--no-cross-encoder` keeps evidence BM25 order (identity passthrough).

Artifacts:

- `benchmarks/results/evidence_retrieval/hierarchical_dse/summary.md`
- `benchmarks/results/evidence_retrieval/hierarchical_dse/full.json`

---

## Interpretation guide

1. If **Document Recall** is low → first-stage doc selection is the bottleneck.  
2. If doc is found but **Section Recall** is low → section scoring/locality fails.  
3. If section is found but evidence candidate miss → unit indexing inside the section.  
4. If candidates contain gold but **Final Hit@K** fails → reranker issue (same as flat CE).  
5. Compare hierarchical vs flat Hit@K on the **same** 40 questions to see whether locality helps when flat distractors dominate.

This corpus often has ~1 EvidenceUnit per leaf section, so Evidence Candidate Recall
closely tracks Section Recall — still a valid architecture check for the cascade.

---

## Initial run (defaults: doc_k=2, section_k=4, evidence_k=10, flat_k=20, CE on)

Artifacts: `benchmarks/results/evidence_retrieval/hierarchical_dse/`

| Metric | Hierarchical | Flat BM25→CE |
|---|---:|---:|
| Final Hit@5 | 0.900 | **1.000** |
| Mean RR (when found) | **0.986** | 0.958 |

**Failure locus (hierarchical only misses):**

- Document miss: `ops_q08`, `arch_q05`
- Section miss after doc: `api_q03`, `prod_q04`
- Evidence miss after section: none
- Rerank miss after candidate: none

**Takeaway on this corpus:** flat BM25→CE already recovers all 40 golds @5. Hierarchical cascade errors are concentrated in **document** and **section** selection under tight Top-K budgets — not in evidence identity or CE. Locality helps in synthetic adversarial unit tests; it does not improve Hit@5 on this small, well-separated Markdown fixture. Next architectural step (if pursued): relax/adapt document & section budgets or scoring before any production wiring.
