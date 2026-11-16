# Evidence Retrieval Evaluation (Phase 2)

Architecture-validation harness for Continuator's Phase 1 universal evidence layer.

**Not** a LoCoMo replacement, public benchmark, or training set.

---

## Purpose

Measure whether this path works reliably on a controlled Markdown corpus:

```text
Markdown
   ↓
Document structure
   ↓
EvidenceUnit boundaries
   ↓
BM25 candidate retrieval
   ↓
Correct EvidenceUnit retrieved
```

Baseline only: Phase 1 construction + existing BM25. No embeddings, rerankers, query expansion, or LLM calls.

---

## Corpus design

Location: `benchmarks/evidence_retrieval/corpus/`

| Document key | File | Pattern |
|---|---|---|
| `api_auth` | `documents/api_auth.md` | Technical API / OAuth docs |
| `product_guide` | `documents/product_guide.md` | Product/project documentation |
| `ops_runbook` | `documents/ops_runbook.md` | Operational procedures |
| `architecture_handbook` | `documents/architecture_handbook.md` | Longer nested handbook |

Questions: `corpus/questions.json` (40 items).

Self-contained fixtures only — no downloads, no external datasets.

---

## Categories

| Category | Meaning |
|---|---|
| `direct_lookup` | Question shares distinctive terms with the evidence |
| `paraphrase` | Same meaning, substantially different wording |
| `section_specific` | Correctness depends on the right section |
| `nested_section` | Gold lives under nested headings (depth ≥ ~3) |
| `cross_terminology` | Query uses different terms than the evidence wording |

---

## Gold evidence identity

Gold fixtures **do not** hardcode absolute-path `document_id` values.

Each question specifies:

```json
{
  "document_key": "api_auth",
  "heading_path": ["API Authentication Guide", "Token Lifecycle", "Access Tokens"]
}
```

At evaluation time:

1. Ingest each document with stable URI `fixture://evidence-retrieval/{key}.md`
2. Resolve `document_key` + `heading_path` → exactly one `EvidenceUnit`
3. Use that unit's `evidence_id` as exact-match gold

Optional `text_contains` can disambiguate if needed.

This matches Phase 1's stable `section_id` / heading ancestry design.

---

## Evaluation rules

Primary rule: **exact `evidence_id` match**.

No fuzzy semantic judging.

Depth: top-10 BM25 hits via `search_evidence` → `RetrievalUnit` + `BM25Index`.

Metrics (scorable questions only):

- Recall@1 / @3 / @5 / @10 (equivalent to Hit@K for single-gold questions)
- MRR

Reuse: `continuator_memory.evaluation.metrics.reciprocal_rank`, `hit_at_k`, `mean`.

---

## Failure taxonomy

| Label | Meaning |
|---|---|
| `missing_evidence` | Gold unit could not be resolved after ingest (ingestion/indexing identity failure) |
| `retrieval_miss` | Gold exists but not present in top-10 |
| `ranking_cutoff` | Gold present but rank > 5 (primary cutoff) |
| `success` | Gold at rank ≤ 5 |

Reports keep per-K recall so a rank-7 case is R@5 fail / R@10 success.

---

## How to run

```bash
PYTHONPATH=src:. python -m benchmarks.evidence_retrieval.run
```

Artifacts (default):

```text
benchmarks/results/evidence_retrieval/baseline/
  full.md
  full.json
  per_question.csv
  failure_analysis.md
  category_analysis.md
  examples.md
  next_step_recommendation.md
```

Tests:

```bash
PYTHONPATH=src:. python -m pytest benchmarks/evidence_retrieval/tests/test_evidence_eval.py -q
```

---

## Limitations

- Tiny fixture corpus (~40 questions) — architecture signal only
- BM25-only baseline; paraphrase/cross-terminology may fail without semantic retrieval
- Nested-section labels are corpus annotations, not automatic difficulty scores
- Structural failure descriptors are descriptive, not causal
- Does not measure conversation Memory quality or LoCoMo

---

## Flow

```text
fixture Markdown
      ↓
Phase 1 ingest_markdown (fixture:// URI)
      ↓
Document / EvidenceUnits in EvidenceStore
      ↓
existing BM25 search_evidence
      ↓
ranked evidence_id list
      ↓
exact gold identity comparison
      ↓
metrics + diagnostics
```
