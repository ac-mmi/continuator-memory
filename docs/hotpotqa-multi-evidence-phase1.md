# HotpotQA Distractor — Phase 1: Multi-Evidence Materialization

**Status:** experiment-only preparation. **No retrieval experiment has been run.**

## What this phase is

MULTI-EVIDENCE retrieval **preparation** for HotpotQA Distractor Dev:

```text
Question
  → Context Documents (closed per-question pool)
  → Sentence-level EvidenceUnits
  → Gold supporting facts → EvidenceUnit IDs
  → Metric helpers (for later retrieval experiments)
```

## What this phase is NOT

- **Not** ordered Hop-1 → Hop-2 evaluation  
- **Not** a global-corpus retrieval experiment (unlike SciFact)  
- **Not** BM25 / semantic / RRF / CE / query rewriting  
- **Not** production retrieval or schema changes  

HotpotQA Distractor is a **per-question closed pool** (typically 10 paragraphs). Metrics must be interpreted in that setting.

---

## Identity design

Within one example:

```text
source_uri  = hotpotqa://{example_id}/{url_encoded_title}
document_id = document_id_for_uri(source_uri)
ordinal     = ORIGINAL HotpotQA sentence_index (0-based)
evidence_id = evidence_id_for(document_id, ordinal)

(title, sentence_index) ↔ EvidenceUnit   (bijection)
```

Cross-question title collisions cannot merge documents: `example_id` is part of `source_uri`.

---

## Gold resolution policy

- Resolver maps each `supporting_facts` entry `(title, sentence_index)` to one EvidenceUnit.
- **Never silently drops** invalid gold.
- Errors: missing title, out-of-bounds index, negative index, identity gaps.
- Duplicate `(title, sentence_index)` refs are deduped with a warning.
- Examples with any unresolved supporting fact are **`invalid_gold`** and must be excluded from strict evaluation.

### Known dataset defect

Example `5ae61bfd5542992663a4f261` has supporting-fact sentence index `902` with only 5 sentences in context. Counted as `invalid_gold` (`sentence_index_out_of_bounds`).

---

## Metric helpers (K = evidence ranking depth)

| Helper | Meaning |
|---|---|
| Supporting-fact Recall@K | \|top-K ∩ gold\| / \|gold\| |
| At-least-one Hit@K | ≥1 gold EvidenceUnit in top-K |
| All-supporting-facts-found@K | every gold EvidenceUnit in top-K |
| Supporting-document Recall@K | gold docs represented by top-K units / \|gold docs\| |
| All-supporting-documents-found@K | both (all) gold docs represented in top-K |

No ordered-hop metric.

---

## Module layout

```text
benchmarks/evidence_retrieval/hotpotqa/
  corpus.py          # load distractor JSON
  adapter.py         # Document + EvidenceUnit materialization
  gold.py            # supporting-fact resolver
  scoring.py         # multi-evidence metric helpers
  validate_gold.py   # dataset-wide gold integrity audit CLI
  tests/
```

### Gold audit command

```bash
python -m benchmarks.evidence_retrieval.hotpotqa.validate_gold \
  -o benchmarks/results/evidence_retrieval/hotpotqa/gold_resolution_audit
```

---

## Next step (only when asked)

Implement BM25 over each question’s closed distractor EvidenceUnit pool and report multi-evidence metrics A–D — still no ordered-hop claims.
