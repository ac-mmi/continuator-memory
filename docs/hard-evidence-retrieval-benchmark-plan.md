# Hard Evidence Retrieval — Benchmark Expansion Plan

**Status:** plan only. Do **not** implement the full corpus or change production retrieval in this task.

Extends Phase 2 (`benchmarks/evidence_retrieval/`, 4 docs / 40 Q / exact-ID BM25 baseline).

---

## Goals

1. Isolate failure modes the 40-Q corpus barely stresses.  
2. Make multi-evidence and granularity measurable **before** prod ranking/agent work.  
3. Keep the suite small enough to author and review: **~80 questions** (acceptable band **60–100**).  
4. Remain offline, deterministic, no downloads.

---

## Corpus strategy

### Keep

- Existing 4 Markdown fixtures and 40 questions as **Level-1/2 regression**.  
- Stable gold resolution via `document_key` + `heading_path` (+ optional `text_contains`).  
- `fixture://evidence-retrieval/{key}.md` ingest URIs.

### Add (proposed new fixture files — not authored yet)

| New doc key | Purpose |
|---|---|
| `token_policy_bundle` | Competing siblings + revocation/expiry distributed facts |
| `crossref_handbook` | Explicit “see section/doc X” links between units |
| `versioned_policy` | 2024 vs 2025 policy paragraphs for temporal/conflict |
| `long_nested_manual` | Deeper heading tree + longer sections for navigation/granularity |

Optionally thicken existing `api_auth` / `product_guide` with 1–2 planted paraphrase traps rather than only new files.

---

## Target mix (~80 questions)

| Category | Scenario ID | Count | Gold shape | Success (primary) |
|---|---|---:|---|---|
| `direct_lookup` | A | 12 | 1 `evidence_id` | exact@5 |
| `paraphrase` | B | 12 | 1 id | exact@5 **and** report exact@10 |
| `cross_terminology` | B | 8 | 1 id | exact@5 |
| `competing_related` | C | 8 | 1 id + `distractor_section_ids` | exact@5; diagnostic: gold rank among sibling set |
| `granularity_parent_child` | D | 8 | `required_id` + `supporting_ids` (parent/sibling) | **supporting@5** primary; exact@5 secondary |
| `multi_evidence` | E | 10 | `required_ids` (2–3) | **coverage@10 == 1.0** |
| `cross_reference` | F | 6 | cite unit optional; **target** `required_ids` | exact or supporting on **target** after ≤2 search rounds (eval harness may simulate round-2 manually later) |
| `multi_hop` | G | 6 | ordered `hop_ids` or unordered `required_ids` | coverage@10; record hops if iterative harness exists |
| `ambiguous` | H | 4 | `acceptable_ids` (any-of groups) | hit any acceptable@5 |
| `absence_or_prohibit` | I | 3 | `must_retrieve_ids` **or** `expect_insufficient=true` | protocol-specific |
| `temporal_or_conflict` | J/K | 3 | current id + optional contradicted id | retrieve current; for conflict retrieve **both** |
| **Total** | | **~80** | | |

Keep ~40 existing items mapped into A/B; add ~40 new hard items. Do not fabricate hundreds.

---

## Gold schema (proposed extension)

Backward compatible with Phase 2:

```json
{
  "question_id": "hard_e01",
  "question": "...",
  "category": "multi_evidence",
  "scenario": "E",
  "gold": {
    "document_key": "token_policy_bundle",
    "heading_path": ["...", "..."],
    "required_ids_resolve": [
      {"document_key": "...", "heading_path": ["..."]},
      {"document_key": "...", "heading_path": ["..."]}
    ],
    "supporting_resolve": [
      {"document_key": "...", "heading_path": ["..."]}
    ],
    "acceptable_any_of": [],
    "distractors": [],
    "expect_insufficient": false,
    "notes": "author rationale"
  }
}
```

At eval time, resolve all `*_resolve` entries to `evidence_id`s exactly as today. **Never** hardcode absolute `doc:` hashes in fixtures.

---

## Evaluation modes (deterministic)

Implement in harness only after fixtures exist; still BM25-first.

| Mode | Formula |
|---|---|
| Exact | `1` if gold `evidence_id` in top-k else `0` |
| Supporting | `1` if any of `required ∪ supporting` in top-k |
| Coverage | `\|top-k ∩ required\| / \|required\|` |
| Partial | `0 < coverage < 1` |
| Sufficiency | `coverage == 1` **or** (if `acceptable_any_of` groups) each group has ≥1 hit |
| Conflict pair | both ids in top-k (for K) |

Report **all** modes; do not replace exact with supporting. Product-oriented headline for hard suite: supporting@5 + coverage@10; architecture headline: exact@5.

**No LLM judge** in this expansion.

---

## Failure taxonomy (reuse + extend)

Keep: `missing_evidence`, `retrieval_miss`, `ranking_cutoff`, `success`.

Add for set tasks:

- `partial_coverage` — some but not all `required_ids`
- `supporting_only` — supporting hit, exact miss
- `wrong_version` — retrieved superseded unit for temporal gold
- `single_side_conflict` — only one of a contradiction pair

---

## Execution plan (when approved)

1. Author new Markdown fixtures + `questions_hard.json` (or version field in `questions.json`).  
2. Extend scoring for set/supporting modes + unit tests.  
3. Run BM25-only baseline → `benchmarks/results/evidence_retrieval/hard_v1/`.  
4. Publish per-category matrix; **then** choose NEXT prod experiment from measured gaps.  
5. Still no embeddings/reranker/agent until category failures demand them.

### Exact command shape (future)

```bash
PYTHONPATH=src:. python -m benchmarks.evidence_retrieval.run --suite hard_v1 -o benchmarks/results/evidence_retrieval/hard_v1
```

(Command not implemented until suite lands.)

---

## What this plan deliberately excludes

- LoCoMo / conversation Memory scoring  
- External datasets  
- Model downloads  
- Agentic loops in the first hard baseline  
- Claiming universal quality from ~80 synthetic questions  

---

## Link to architecture audit

Roadmap **NOW** = this plan’s implementation + BM25 re-baseline.  
See [`docs/hard-evidence-retrieval-architecture-audit.md`](hard-evidence-retrieval-architecture-audit.md).
