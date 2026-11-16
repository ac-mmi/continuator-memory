# Evidence-Set Coverage — Dataset Audit

**Audit only.** No retrieval, schema, production, or metric implementation.

## 1. Dataset inventory

| Dataset | Questions/Claims | Gold format | Supports evidence-set coverage? |
|---|---:|---|---|
| `evidence_retrieval_structured_corpus` | 40 | single document_key + heading_path (+ optional text_contains) → exactly one EvidenceUnit | `False` |
| `scifact_claims_dev` | 188 | evidence[doc_id][] = {sentences: [idx...], label: SUPPORT|CONTRADICT}; flattened to N (doc_id, sentence_index) gold EvidenceUnits | `False` |
| `hard_evidence_retrieval_plan` | 0 | Proposed: required_ids_resolve (AND), acceptable_any_of (OR), supporting_resolve — NOT present as data | `False` |
| `locomo10_conversation_memory` | N/A (conversation path) | Questions may have multiple gold message IDs; analyses compute partial vs full gold coverage. Dataset does not ship explicit AND-required vs OR-alternative fields for EvidenceUnits. | `False` |

## 2. Gold semantics

### Structured corpus (`benchmarks/evidence_retrieval/corpus/`)

Each question resolves to exactly one EvidenceUnit. No required_ids / acceptable_any_of / AND-OR metadata.

**Multiple gold IDs exist?** No (0 multi-gold questions).

**All IDs required?** N/A — each question has exactly one gold EvidenceUnit.

### SciFact (`claims_dev.jsonl`)

Annotated rationale sentences for SUPPORT or CONTRADICT. Multiple gold sentence IDs mean multiple annotated evidence sentences — NOT an explicit AND-required answer set, and NOT an explicit OR-alternative set. Official retrieval metrics may score set coverage over annotations, but that is not the same as MULTI_EVIDENCE_REQUIRED for answering.

- Evidence-bearing claims: `188`
- Exactly one gold sentence: `96`
- Multiple gold sentences: `92`
- Multi-document gold: `10`
- Schema fields: `['id', 'claim', 'evidence', 'cited_doc_ids']` / block `['sentences', 'label']`

**Critical distinction:** `n_multi_gold > 0` means multiple annotated rationale sentences exist. It does **not** mean the dataset marks them as AND-required for a complete answer, nor as OR-alternatives.

### Hard-evidence plan

Future schema would explicitly distinguish MULTI_EVIDENCE_REQUIRED (required_ids) from ALTERNATIVE_GOLD (acceptable_any_of). No questions exist yet — do not count as evaluable.

### LoCoMo (note only)

Questions may have multiple gold message IDs; analyses compute partial vs full gold coverage. Dataset does not ship explicit AND-required vs OR-alternative fields for EvidenceUnits.

## 3. Classification counts

### Structured corpus

| Classification | Count |
|---|---:|
| SINGLE_EVIDENCE | 40 |
| ALTERNATIVE_GOLD | 0 |
| MULTI_EVIDENCE_REQUIRED | 0 |
| UNKNOWN | 0 |

### SciFact (evidence-bearing)

| Classification | Count |
|---|---:|
| SINGLE_EVIDENCE | 96 |
| ALTERNATIVE_GOLD | 0 |
| MULTI_EVIDENCE_REQUIRED | 0 |
| UNKNOWN | 92 |

Only classifications supported by actual data are populated. `ALTERNATIVE_GOLD` and `MULTI_EVIDENCE_REQUIRED` remain **0** everywhere because no dataset exposes those semantics.

## 4. Current metric gap

- **Hit@K:** `ANY supporting / any annotated gold found`
- **MRR:** `reciprocal rank of first ANY gold`
- **Structured Recall@K:** `equals Hit@K (single gold)`
- **SciFact sentence_recall@K:** `fraction of annotated gold EvidenceUnits in top-K — set coverage over annotations, NOT labeled complete-required-answer coverage`

**Measures complete required evidence set?** `False`

No dataset marks which multi-gold IDs are jointly required for a complete answer versus interchangeable alternatives.

Therefore Hit@K / MRR answer: *was **any** annotated gold found?* — not *was the **complete required set** found?*

## 5. Recommendation

### **C — Existing datasets do NOT represent required evidence sets**

No evaluable corpus encodes MULTI_EVIDENCE_REQUIRED vs ALTERNATIVE_GOLD. Structured corpus is entirely SINGLE_EVIDENCE. SciFact multi-gold annotations are UNKNOWN (annotated rationales, not AND/OR). Hard-evidence plan defines the needed fields but has zero authored questions. Do not fake Evidence-Set Coverage@K.

#### Smallest missing data structure (future — not authored here)

```json
{
  "fields": [
    "required_ids_resolve (AND \u2014 all needed)",
    "acceptable_any_of (OR \u2014 any-of groups)",
    "optional supporting_resolve"
  ],
  "reference": "docs/hard-evidence-retrieval-benchmark-plan.md",
  "can_implement_metric_now": false,
  "can_implement_after_fixtures": true
}
```

Do **not** implement Evidence-Set Coverage@K until fixtures explicitly encode AND/OR. Do **not** reinterpret SciFact multi-gold as required sets.
