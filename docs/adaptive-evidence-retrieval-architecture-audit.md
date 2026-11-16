# Adaptive Evidence Retrieval — Architecture Audit & Gap Analysis

**Mode:** audit / planning only.  
**No production code changes. No adaptive implementation. No full evaluation run.**

Label legend:

- **EXISTS AND USED** — present and wired into a shipping or primary product path
- **EXISTS BUT EXPERIMENT-ONLY** — present in benchmarks/eval modes; not the EvidenceUnit product default
- **PARTIALLY EXISTS** — related pieces exist, but not for EvidenceUnit multi-evidence adaptive control
- **DOES NOT EXIST** — not present as a reusable component

---

## 0. Scope and method

This audit maps **actual repository modules** after SciFact single-evidence experiments, hierarchical DSE experiments, and HotpotQA multi-evidence baselines (BM25, Dense, Union, Union→CE, CE coverage-failure diagnosis).

Two parallel stacks must not be conflated:

| Stack | Identity atom | Primary code |
|---|---|---|
| **Universal evidence** | `EvidenceUnit` / `Document` | `src/continuator_memory/evidence/` |
| **Conversation memory** | Memory / message | `src/continuator_memory/v2/`, `webapi/` |

HotpotQA and SciFact benchmarks operate on **EvidenceUnits**. Most production adaptive-ish modes (multiquery, iterative, LoCoMo CE) operate on **memories/messages**. Reuse is possible at the *tooling* layer (BM25Index, CrossEncoderReranker, RRF helpers), not by assuming memory iterative retrieval already solves EvidenceUnit multi-evidence coverage.

---

## PART 1 — Current architecture map

### 1A. Production — universal EvidenceUnit path (document evidence)

```text
Source file (md/html/docx/pdf/…)
  ↓
continuator_memory.evidence.ingest / adapters
  ↓
Document  (evidence.models.Document)
EvidenceUnit  (evidence.models.EvidenceUnit)
  ↓
EvidenceStore  (evidence.store)
  ↓
search_evidence()  (evidence.search)
  ├── units_to_retrieval_units()
  └── BM25Index.search()  (v2.bm25)
  ↓
list[EvidenceHit]  (unit + document + score + rank)
  ↓
Fixed Top-K truncation (caller-supplied top_k)
```

**What this path does *not* include today:**

- Dense EvidenceUnit indexing/search in production
- Cross-encoder rerank over EvidenceUnits in production
- Hybrid BM25∪Dense over EvidenceUnits in production
- Evidence-set / multi-evidence selection
- Sufficiency assessment over EvidenceUnits
- Adaptive strategy selection

### 1B. Production — conversation / memory workspace path

```text
Question
  ↓
Workspace._retrieve_evidence()  (webapi.workspace)
  ├── mode baseline / lexical / contextual_* …
  ├── retrieve_multiquery_evidence()     (webapi.multiquery)
  ├── retrieve_hybrid_rrf_evidence()     (webapi.hybrid_rrf)
  ├── retrieve_iterative_evidence()      (webapi.iterative)
  │     └── evidence_eval.resolve_evaluation()  (LLM sufficiency + next_queries)
  └── contextual_*_rerank paths          (webapi.contextual_*)
        └── CrossEncoderReranker / v2.rerank.rerank()
  ↓
MemoryRetriever outcomes → SourceEvidence
  ↓
Answer generation (separate layer)
```

Also:

```text
HybridRetriever  (v2.hybrid_retrieval)
  raw conversation chunks ∪ memory sources
  → RRF over messages
```

### 1C. Benchmark — SciFact EvidenceUnit experiment path

```text
SciFact claim
  ↓
adapter.units_from_scifact / corpus loaders
  ↓
BM25Index over EvidenceUnit.retrieval_text
  + LocalSentenceEmbedder / SemanticCorpusIndex  (scifact.semantic_*)
  ↓
optional RRF  (scifact.rrf)
  ↓
union pool → rerank_fixed_pool()  (scifact.union_cross_encoder_rerank)
  ↓
CrossEncoderReranker  (v2.rerank; local_files_only)
  ↓
Top-K EvidenceUnit IDs → scifact.scoring metrics
```

### 1D. Benchmark — HotpotQA closed-pool multi-evidence path

```text
HotpotQA question (distractor context)
  ↓
hotpotqa.adapter.materialize_example
  → question-local Document + sentence EvidenceUnits
  ↓
hotpotqa.gold.resolve_supporting_facts
  ↓
BM25: hotpotqa.bm25_baseline.bm25_rank_closed_pool
Dense: hotpotqa.semantic_baseline.dense_rank_closed_pool
  ↓
dedup_union(BM25@K, Dense@K)  (hotpotqa.union_baseline)
  ↓
rerank_fixed_pool(question, union candidates)  (SciFact CE helper)
  ↓
ranked EvidenceUnit IDs → hotpotqa.scoring
  (supporting_fact_recall / any_hit / all_found / doc recall)
```

**HotpotQA failure locus (measured):**

```text
Union pool COMPLETE (all gold EvidenceUnits present)
        ↓
Independent CE scores each (question, passage)
        ↓
Top-10 truncation
        ↓
≥1 required gold pushed to ranks 11–15  →  CE_COVERAGE_FAILURE
```

This is **not** a candidate-generation miss for those 920 examples.

---

## PART 2 — Capability inventory

| # | Capability | Status | Repository references |
|---|---|---|---|
| 1 | BM25 retrieval | **EXISTS AND USED** (EvidenceUnits via `search_evidence`; also v2/contextual BM25 on memories) | `evidence/search.py`, `v2/bm25.py`, HotpotQA/SciFact baselines |
| 2 | Dense retrieval | **PARTIALLY EXISTS** — production dense is **MemoryRetriever** / embeddings; EvidenceUnit dense is **experiment-only** | `embedding.py`, `v2/memory_retrieval.py`; `scifact/semantic_rank.py`, `hotpotqa/semantic_baseline.py` |
| 3 | Hybrid candidate union | **EXISTS BUT EXPERIMENT-ONLY** for EvidenceUnits; memory-side channel unions exist in webapi | `hotpotqa/union_baseline.py`, `scifact/compare_retrieval_modes.py`, `webapi/contextual_union_rerank.py` |
| 4 | RRF fusion | **EXISTS AND USED** on memory/hybrid paths; **experiment-only** on SciFact EvidenceUnits; **not** used in HotpotQA CE experiment | `v2/hybrid_retrieval.py`, `webapi/multiquery.rrf_fusion`, `webapi/hybrid_rrf.py`, `scifact/rrf.py` |
| 5 | Cross-encoder reranking | **EXISTS AND USED** in some workspace/eval modes; EvidenceUnit CE is experiment-only | `v2/rerank.py`, `webapi/contextual_*`, `scifact/union_cross_encoder_rerank.py`, `hotpotqa/union_ce_rerank.py` |
| 6 | Fixed Top-K truncation | **EXISTS AND USED** everywhere ranked lists are cut | caller `top_k` / `final_k` / metric `@K` |
| 7 | Evidence identity preservation | **EXISTS AND USED** | `EvidenceUnit.evidence_id`, `document_id`, HotpotQA `hotpot_evidence_id`, CE fixed-pool invariants |
| 8 | Multi-evidence representation | **PARTIALLY EXISTS** — gold *sets* + metrics in HotpotQA; no production EvidenceState | `hotpotqa/scoring.py`, `hotpotqa/gold.py`, `evidence_set_coverage_audit.py` |
| 9 | Evidence-set coverage tracking | **EXISTS BUT EXPERIMENT-ONLY** | `hotpotqa/scoring.all_supporting_facts_found_at_k`, union/CE reports, diagnosis |
| 10 | Evidence sufficiency assessment | **PARTIALLY EXISTS** — LLM/heuristic evaluator on **conversation iterative** path; **not** on EvidenceUnit HotpotQA path | `webapi/evidence_eval.py`, `webapi/iterative.py` |
| 11 | Query decomposition | **PARTIALLY EXISTS** — planner multiquery on memories; not EvidenceUnit HotpotQA | `webapi/query_plan.py`, `webapi/multiquery.py` |
| 12 | Query rewriting | **PARTIALLY EXISTS** — plan paraphrases / iterative next_queries (memory path) | `query_plan.py`, `evidence_eval.py` |
| 13 | Multi-query retrieval | **EXISTS AND USED** (memory workspace modes) | `webapi/multiquery.py`, workspace mode `multiquery` |
| 14 | Iterative retrieval | **EXISTS AND USED** (memory workspace `iterative` mode; LLM evaluator) | `webapi/iterative.py` |
| 15 | Retrieval conditioned on selected evidence | **PARTIALLY EXISTS** — iterative follow-ups may use excerpt terms; **no** EvidenceUnit “search for uncovered fact given selected set” | iterative + evidence_eval; absent in HotpotQA EU path |
| 16 | Set-aware reranking | **DOES NOT EXIST** | CE is independent pair scoring only |
| 17 | Diversity/MMR selection | **DOES NOT EXIST** as MMR; **PARTIALLY** channel-quota reduction approximates diversity of *channels*, not evidence-set coverage | `contextual_wide_union_rerank.reduce_by_channel_quota` |
| 18 | Adaptive strategy selection | **DOES NOT EXIST** as a general EvidenceUnit controller; iterative has stop reasons only | `iterative.py` stop constants |
| 19 | Stop/continue decision logic | **PARTIALLY EXISTS** — iterative stop reasons; not EvidenceUnit multi-evidence coverage stop | `STOP_SUFFICIENT`, etc. |
| 20 | Contradiction detection | **PARTIALLY EXISTS** — memory consolidation / claim contradiction fields; not EvidenceUnit ranking | `v2/memory_store`, consolidation |
| 21 | Evidence relationship classification | **DOES NOT EXIST** for EU multi-hop relations | HotpotQA has bridge/comparison labels only as dataset metadata |
| 22 | Multi-hop reasoning support | **DOES NOT EXIST** as ordered-hop retrieval; HotpotQA metrics are unordered multi-evidence coverage | explicit non-goal in HotpotQA baselines |
| 23 | Per-question retrieval state | **PARTIALLY EXISTS** — diagnostics dicts / iterative rounds; no first-class `EvidenceState` for EUs | iterative rounds; HotpotQA per-example JSON rows |

---

## PART 3 — What can be reused

### Reuse directly (stable abstractions)

| Component | Path | Why reusable |
|---|---|---|
| `Document` / `EvidenceUnit` / `EvidenceHit` | `evidence/models.py` | Identity + provenance spine |
| `EvidenceStore` | `evidence/store.py` | Persistence |
| `BM25Index` / `RetrievalUnit` | `v2/bm25.py` | Lexical scoring |
| `units_to_retrieval_units` / `search_evidence` | `evidence/search.py` | EU ↔ BM25 bridge |
| `CrossEncoderReranker` / `rerank()` | `v2/rerank.py` | Local CE, fixed-pool reorder |
| `rerank_fixed_pool` / `PoolCandidate` | `scifact/union_cross_encoder_rerank.py` | EU-oriented CE helper (experiment, but API-clean) |
| `LocalSentenceEmbedder` | `scifact/semantic_rank.py` | Offline dense embeddings |
| HotpotQA materializer + gold resolver | `hotpotqa/adapter.py`, `gold.py` | Closed-pool identity + gold |
| Multi-evidence metrics | `hotpotqa/scoring.py` | All-Found / Any-Hit / SF recall |
| `dedup_union` | `hotpotqa/union_baseline.py` | Candidate set algebra |
| RRF helpers | `multiquery.rrf_fusion`, `scifact/rrf.py`, `v2/hybrid_retrieval` | Rank fusion primitives |

### Needs extension (not drop-in)

| Gap | Why |
|---|---|
| Dense search over persisted EvidenceUnits | Production `search_evidence` is BM25-only |
| First-class candidate pool object for EUs | Today: ad hoc ID lists in benchmarks |
| Coverage-aware selection over a frozen EU pool | Missing; CE only reorders independently |
| Non-LLM sufficiency over multi-evidence gold/coverage proxies | Memory iterative sufficiency is LLM + conversation excerpts |
| Strategy interface spanning EU experiments → product | Memory modes ≠ EU pipeline |

### Experiment-specific — do **not** auto-promote to production

- HotpotQA closed distractor pools (not global corpus retrieval)
- SciFact oracle-document ablations
- Hierarchical DSE / Flat∪Hier (measured **no unique gold**; do not ship as candidate channel)
- HotpotQA heuristic CE-failure taxonomy (diagnosis aid, not product logic)
- MiniLM / ms-marco MiniLM model choices locked in experiment configs
- Any metric that equates All-Found with answer correctness

---

## PART 4 — Minimal missing layers (no rewrite)

Do **not** port the full memory iterative LLM controller onto EvidenceUnits first.

Smallest coherent layers for an **experiment-only** adaptive milestone:

### 4.1 `EvidenceCandidatePool` (thin)

Frozen set of `EvidenceUnit` IDs + optional scores from BM25/Dense/CE.  
Already implicit in HotpotQA artifacts; promote to an explicit dataclass in benchmarks first.

### 4.2 `EvidenceSelectionState` (minimal)

```text
selected_ids: ordered EvidenceUnit ids
remaining_pool_ids: set
scores_by_id: optional
coverage_proxy: e.g. distinct docs / titles / entities touched (heuristic)
```

Not a full “concept graph.” Start with **IDs + ranks + simple novelty**.

### 4.3 `SetAwareSelector` (the missing product-relevant experiment)

Input: frozen pool + query (+ optional CE scores).  
Output: ordered shortlist of size K that optimizes **coverage**, not only independent relevance.

Candidates for first implementation (benchmark-only):

1. **Greedy set selection using CE scores + novelty** (MMR-style / doc-diversity)
2. **Iterative re-score**: pick top CE; down-weight near-duplicates; repeat
3. Later: evidence-conditioned CE (query ⊕ selected texts) — still no LLM required for v1

### 4.4 `CoverageAssessment` (benchmark)

For HotpotQA: reuse gold All-Found as oracle assessment.  
For product later: replace with non-gold proxies only after experiments justify them.

### 4.5 Do **not** build yet

- General `StrategySelector` with decomposition / broaden / multi-hop investigation
- Production wiring of adaptive EU retrieval
- LLM sufficiency on HotpotQA EvidenceUnits (memory iterative already covers conversation; EU problem is set selection under a frozen pool)

**Fit check:** Memory `iterative` proves stop/continue is feasible in-repo, but its sufficiency signal is **answerability of conversation excerpts via LLM**, whereas HotpotQA’s measured bug is **set coverage under independent ranking with gold already in pool**. Different layer.

---

## PART 5 — Retrieval vs ranking vs set selection vs sufficiency vs reasoning

| Layer | Responsibility | Current HotpotQA status |
|---|---|---|
| **Retrieval / candidate generation** | Find candidates | BM25∪Dense succeeds for 5669/7404 (pool-complete) |
| **Independent ranking** | Order by standalone relevance | CE does this well for *some* golds |
| **Evidence-set selection** | Choose a *covering* subset under budget K | **FAILURE POINT** — Top-10 drops complementary golds (920) |
| **Sufficiency** | Decide enough evidence | Not implemented for EU; diagnosis used gold All-Found oracle |
| **Reasoning / answering** | Use evidence to answer | Out of scope for these experiments |

**Where the HotpotQA failure sits:**  
Between **independent ranking** and **evidence-set selection** — after successful retrieval, before any answer generation.

---

## PART 6 — Experimental findings → architecture implications

| Finding | Proves | Architectural implication | Change justified? |
|---|---|---|---|
| 1. SciFact hybrid retrieval | BM25 and dense recover complementary single-evidence candidates | Keep hybrid **candidate generation** for EU experiments | Yes (experiment); production EU dense still optional |
| 2. SciFact CE reranking | Independent CE improves single-evidence ranking on fixed pools | CE is valuable for **ordering**, not automatic set coverage | Yes for single-evidence; insufficient alone for multi-evidence |
| 3. SciFact terminology mismatch | Some misses are vocab gaps / dataset alignment, not ranker bugs | Separate retrieval failure from label/semantics issues | Process discipline; not a new subsystem |
| 4. Hierarchical retrieval failure | Flat∪Hier added no unique gold; hard hierarchy ⊂ flat | Do **not** add hierarchy as production candidate channel | **No** hierarchy shipping |
| 5. HotpotQA BM25 baseline | Closed-pool BM25: high Any-Hit, moderate All-Found (~0.58@10) | Single-channel lexical incomplete for multi-evidence | Motivates multi-channel CG |
| 6. Dense complementarity | Dense finds golds BM25 misses (and vice versa) | Union candidate generation justified | Yes (experiment) |
| 7. Union improvement | All-Found ~0.766 as **set** | Complementarity is real at candidate layer | Yes |
| 8. CE coverage failures (920) | Pool complete but Top-10 incomplete | Independent CE ≠ set selection | **Yes — primary architecture gap** |
| 9. Bridge-heavy CE failures (93.7%) | Multi-hop/bridge questions suffer most | Set-aware / hop-sensitive selection matters more than more BM25 depth alone | Yes |
| 10. Missing golds ranks 11–15 | Near-miss under budget, pool ≤20 | Often fixable by **selection policy**, not more retrieval | Strongly supports set-aware selection milestone |

**Equal-budget caution (from union experiment):** Union(5∪5) did not beat BM25@10 under similar pool size. Architecture must compare **fair budgets**, not only same source depth.

---

## PART 7 — Recommended next milestone (ONE)

### Milestone: Set-aware selection over the frozen HotpotQA BM25@10 ∪ Dense@10 pool

**Goal:** On the **same** frozen candidate pools used by Union→CE, replace independent CE Top-10 truncation with a **coverage-aware selector** that still uses CE scores (and optionally document/title novelty), and measure All-Found@10 / SF-R@10 / Any-Hit@10 vs:

- BM25@10  
- Dense@10  
- Raw Union set (upper bound)  
- Union→independent CE Top-10  

**Constraints:**

- Benchmark-only; no production changes  
- No new embedding model required  
- No LLM required for v1  
- No new retrieval; pool frozen  
- Reuse `rerank_fixed_pool` scores already persisted where possible; re-score only if needed for selector variants  

**Success criteria (measurable):**

1. Reduce CE coverage failures (920) materially while keeping mean selected size ≈ 10.  
2. Improve All-Found@10 vs independent CE without exceeding raw-union All-Found upper bound.  
3. Report gold retention: union → selected Top-10.  
4. Bridge vs comparison and gold-set-size breakdowns.

**Why this milestone (and not query decomposition first):**

- Measured bottleneck is **selection under a complete pool**, not missing candidates.  
- Decomposition would add planner/LLM complexity and new retrieval calls before proving set selection helps.  
- Hierarchy already failed as a candidate channel.

**Explicit non-goals for this milestone:**

- Production adaptive controller  
- Answer generation  
- Ordered multi-hop metrics  
- StrategySelector with many strategies  

---

## Confirmation

- No production behavior changed in this audit task.  
- No adaptive architecture implemented.  
- No full evaluation executed for this audit.  
- Artifacts to maintain: this document + companion JSON.
