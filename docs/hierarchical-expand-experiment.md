# Hierarchical Expand Retrieval Experiment

**Scope:** Isolated candidate-generation ablation over persisted EvidenceUnits.  
**Not in scope:** production search changes, schema changes, new models, LLM agents, query rewriting.

---

## 1. Architecture findings

Persisted units already carry `document_id`, `section_id`, `heading_path`, `ordinal`, and metadata. Production `search_evidence` is flat BM25 over all units.

Experimental pipeline (benchmarks only):

```text
Query
  → Stage 1: search_evidence (BM25, global_top_k)
  → Stage 2: group seeds by document / (document, section)
  → Stage 3: bounded structural expansion
  → Stage 4: BM25 rescoring over candidate pool only
  → ranked EvidenceUnit ids
```

---

## 2. Does SciFact support meaningful hierarchy?

**No.** SciFact adapter assigns:

- `heading_path = (title,)` for every sentence
- unique `section_id = sec:sNNNN` per sentence

`document_has_meaningful_sections()` is False for SciFact docs (no shared `section_id`).

Therefore SciFact evaluation uses **document expansion only**. Section expansion is validated on a structured Markdown-like fixture in unit tests, not claimed as a SciFact result.

---

## 3. Experimental design

| Mode | Behavior |
|---|---|
| `flat` | Global BM25 seeds only; no expansion |
| `document` | Rank seed documents by seed BM25 score sum; expand all units in top docs |
| `section` | Expand shared `section_id` (+ optional parent/child `heading_path`); falls back to document expansion when sections are not meaningful |

---

## 4. Expansion rules

**Document mode**

1. Group seeds by `document_id`
2. Rank docs by Σ seed scores (tie-break `document_id`)
3. Take `max_documents`
4. Add all units in those docs until `max_expanded_units`

**Section mode** (only when a doc shares `section_id` across units)

1. Rank `(document_id, section_id)` groups by Σ seed scores
2. Per doc keep ≤ `max_sections_per_document`
3. Include same-section units
4. Optionally include ancestor / descendant `heading_path` units
5. Optional `sibling_window` within the same section
6. Flat docs fall back to document expansion (no fake hierarchy)

Dedup by `evidence_id`. Deterministic order: `(document_id, ordinal, evidence_id)`.

---

## 5. Candidate budgets (defaults)

| Param | Default |
|---|---|
| `global_top_k` | 50 |
| `max_documents` | 5 |
| `max_sections_per_document` | 3 |
| `max_expanded_units` | 200 |
| `rank_top_k` | 50 |
| `sibling_window` | 0 |

---

## 6. SciFact 50-claim results

Artifacts: `benchmarks/results/evidence_retrieval/scifact/hierarchical_expand_50/`

### Candidate discovery

| Mode | mean seed recall | mean expanded recall | seed hit | expanded hit | gold only via expansion | mean pool |
|---|---:|---:|---:|---:|---:|---:|
| flat | 0.823 | 0.823 | 0.900 | 0.900 | 0 | 50.0 |
| document | 0.823 | **0.886** | 0.900 | **0.920** | **7** | 73.9 |

### Ranking (local BM25 over pool)

| Mode | MRR | Hit@5 | Recall@5 | Hit@10 | Recall@10 |
|---|---:|---:|---:|---:|---:|
| flat | 0.594 | 0.700 | 0.518 | 0.780 | 0.597 |
| document | 0.594 | 0.720 | 0.540 | 0.740 | 0.590 |

**Interpretation:** Document expansion improves **candidate discovery** (7 claims recover gold only after expansion; expanded recall +0.063). Final ranking gains are small / mixed — structural expansion helps recall into the pool more than it helps BM25 top ranks.

---

## 7. What did NOT change

- `search_evidence` / BM25 implementation
- semantic corpus index / RRF / cross-encoder experiments
- ingestion adapters / schema / EvidenceUnit identity rules

---

## ONE recommended next step

**On the 7 SciFact claims where gold appears only after document expansion, measure whether feeding the expanded pool into the existing Union→CE reranker improves Hit@5 versus flat BM25@50 — still experiment-only, no production wiring.**
