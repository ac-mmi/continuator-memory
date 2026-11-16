# Evidence Retrieval Failure Diagnosis

Forensic analysis of the **3 failed questions** from the Phase 2 Markdown EvidenceUnit + BM25 baseline.

**Scope:** diagnosis only. No production ingestion/retrieval changes. No embeddings, rerankers, BM25 tuning, corpus edits, or gold relabeling.

Artifacts:

- `benchmarks/results/evidence_retrieval/baseline/failure_diagnosis.md`
- `benchmarks/results/evidence_retrieval/baseline/failure_diagnosis.json`
- `benchmarks/results/evidence_retrieval/baseline/failure_examples.md`

Command:

```bash
PYTHONPATH=src:. python -m benchmarks.evidence_retrieval.diagnose
```

---

## Inspected baseline schema (MEASURED)

From `full.json` + `scoring.py`:

| Rule | Observed behavior |
|---|---|
| Eval depth | `ranked_ids` length = **10** (`MAX_DEPTH`) |
| Exact-ID gold | `first_evidence_rank(ranked_ids, gold_evidence_id)` |
| `ranking_cutoff` | gold exists ∧ rank ≠ None ∧ rank **> 5** |
| `retrieval_miss` | gold exists ∧ rank is **None** within depth 10 |
| Persisted per question | question, category, gold_spec, resolved_gold, score, ranked_ids, top_hits (≤5 previews), structural |

Baseline already knew *that* three questions failed and their ranks; it did **not** store full gold `retrieval_text`, deep ranks beyond 10, or competitor lexical overlap. Diagnosis re-ingests the same fixtures and reuses `search_evidence` at depth 52 for counterfactuals.

---

## Population

| Metric | Value |
|---|---|
| Questions | 40 |
| Successes | 37 |
| Failures | 3 |
| ranking_cutoff | 2 (`api_q03`, `prod_q09`) |
| retrieval_miss | 1 (`prod_q04`) |
| Documents / units | 4 / 52 |

---

## Failure table

| Question | Category | Outcome | Gold @10 | Gold deep | Mechanism | Confidence |
|---|---|---|---:|---:|---|---|
| `api_q03` | paraphrase | ranking_cutoff | 6 | 6 | `RANKING_LEXICAL_FAILURE` | medium |
| `prod_q04` | nested_section | retrieval_miss | — | 16 | `GOLD_GRANULARITY_FAILURE` | medium |
| `prod_q09` | paraphrase | ranking_cutoff | 7 | 7 | `RANKING_LEXICAL_FAILURE` | medium |

---

## Per-failure summary

### `api_q03` — Case 1 (retrieved below cutoff)

Question uses **invalidate / credential**; gold says **revoke / refresh token**.

- Gold rank **6** (stable at full depth 52 — not a deep miss).
- Sibling **Refresh Tokens** ranks **1** (BM25 5.62 vs gold 2.92).
- Query↔gold stems: `administrator`, `refresh`. Missing: `invalidat`, `credential`, `happen`.
- Synonym gap: `invalidate` ↛ `revoke` under BM25 tokenization/stemming.
- Distinct-stem overlap can favor gold while BM25 score still favors the sibling (repeated `refresh` / section title mass).

**Mechanism:** ranking / lexical competition with a sibling unit — **not** missing evidence, **not** ingest failure.

### `prod_q04` — Case 2 (not in top-10) with granularity caveat

Question: Markdown ingestion preserving nested locations.

- Gold **Structure Preserved** exists; deep rank **16** (11–20 band).
- Rank-1 hit is same-doc **Markdown Documents** (ingestion / markdown / document overlap 3/6).
- That related unit also discusses EvidenceUnits + heading ancestry and shares gold answer stems.
- Exact-ID scoring therefore labels a **retrieval_miss**, but a near-answer related unit was retrieved.

**Mechanism:** `GOLD_GRANULARITY_FAILURE` (exact-ID miss with related useful unit) — **not** a clean true miss, **not** ingest omission.

### `prod_q09` — Case 1 (retrieved below cutoff)

Question uses **sparse / strings / prepared**; gold talks about **BM25 / retrieval text / heading ancestry**.

- Gold rank **7** (stable deep).
- Top competitor is an **ops** “Empty Retrieval” unit matching `document` + `search`.
- Missing gold stems: `sparse`, `string`, `prepar`.

**Mechanism:** paraphrase / cross-terminology lexical under-ranking — gold present but weak term match.

---

## Mechanism counts

| Mechanism | Count |
|---|---|
| `RANKING_LEXICAL_FAILURE` | 2 |
| `GOLD_GRANULARITY_FAILURE` | 1 |

No `CORPUS_INGESTION_FAILURE`. No unit missing from the index.

---

## Decision

**OPTION_F — multiple mechanisms; next experiment must isolate them.**

Reasons (MEASURED):

1. Two failures are mild ranking cutoffs (ranks 6–7) driven by paraphrase/sibling lexical competition.
2. One “miss” is a deep rank-16 with a related unit already answering at rank 1 under exact-ID rules.
3. n=3 on a 40-question / 52-unit fixture cannot justify a single production change.

Suggested isolation experiments (do not implement yet):

- A: paraphrase-only slice / representation ablation (no prod change required to *plan*)
- D: credit related-section soft match vs exact-ID on the same pools
- Keep Memory/LoCoMo pipelines untouched

---

## Epistemic bounds

**MEASURED**

- All three golds exist after ingest; spans and heading paths are correct.
- Cutoff failures are small ranking errors (6–7), not ranks ≫20.
- The miss appears at rank 16 with a related Markdown ingestion unit at rank 1.

**HYPOTHESIS**

- Paraphrase synonyms (`invalidate`/`revoke`, `sparse`/`BM25`) are a recurring BM25 weakness.
- Exact-ID evaluation may understate useful section-level retrieval.

**NOT PROVEN**

- That embeddings/rerankers are required.
- That EvidenceUnit boundaries are wrong.
- Any universal Continuator retrieval quality claim from this fixture.
