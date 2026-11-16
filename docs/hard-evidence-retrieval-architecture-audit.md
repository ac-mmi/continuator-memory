# Hard Evidence Retrieval — Architecture Audit

**Mode:** investigation / planning only.  
**No production ingestion or retrieval changes. No models, embeddings, rerankers, agents, or corpus expansion implemented in this task.**

Label legend:

- **MEASURED** — confirmed in code or Phase 2 baseline/diagnosis artifacts
- **INFERRED** — reasonable from architecture, not yet evaluated on a hard corpus
- **ARCHITECTURAL RECOMMENDATION** — engineering judgment constrained by measured gaps
- **NOT PROVEN** — must not be treated as product truth yet

---

## 1. Current architecture (MEASURED)

### Evidence path (universal Markdown layer)

```text
Markdown file
  → ingest_markdown (fixture:// or file URI)
  → Document (ev_documents)
  → structure-first sections (heading stack)
  → EvidenceUnit (ev_units): text, retrieval_text, heading_path, section_id, char spans
  → search_evidence → RetrievalUnit + BM25Index
  → ranked evidence_id list
  → Phase 2 exact-ID scoring
```

| Component | Exists? | Notes |
|---|---|---|
| Markdown → Document / EvidenceUnit | Yes | `src/continuator_memory/evidence/` |
| Heading ancestry in units + `retrieval_text` | Yes | `"H1 > H2\n\nbody"` |
| Character spans | Yes | when section not size-split |
| Optional oversized split | Yes | extension point; default off |
| BM25 lexical search | Yes | `search_evidence` |
| Dense EvidenceUnit index | **No** | Memory dense path is separate |
| Reranker on EvidenceUnits | **No** | CE exists for conversation experiments only |
| Parent/child expansion at query time | **No** | ancestry stored, not used to expand hits |
| Multi-evidence fusion / set return | **No** | returns ranked list of units |
| Cross-doc link following | **No** | |
| Temporal/version filters | **No** | `content_hash` / `created_at` only |
| Contradiction detection | **No** | |
| Evidence-path iterative / multiquery | **No** | those live on **Memory** workspace path |

### Conversation path (parallel; not the universal evidence spine)

Workspace modes include dense memory retrieval, lexical bonuses, contextual BM25, multiquery, iterative (LLM evaluator follow-ups), experimental graph — all over **Memories/messages**, not EvidenceUnits. **Do not treat conversation-side iterative retrieval as shipping universal evidence capability.**

### Phase 2 baseline (MEASURED)

| Metric | Value |
|---|---|
| Docs / units / questions | 4 / 52 / 40 |
| Success / ranking_cutoff / retrieval_miss | 37 / 2 / 1 |
| MRR / R@5 / R@10 | 0.879 / 0.925 / 0.975 |

Failure mechanisms (diagnosis):

| ID | Outcome | Mechanism |
|---|---|---|
| `api_q03` | ranking_cutoff @6 | paraphrase + sibling lexical competition (`invalidate`≠`revoke`) |
| `prod_q09` | ranking_cutoff @7 | terminology mismatch (`sparse`≠`BM25`) |
| `prod_q04` | miss @ deep 16 | exact-ID granularity; related Markdown unit @1 |

**Biggest proven gap today:** single-pass BM25 + exact-ID scoring cannot cleanly separate paraphrase ranking, related-section competition, and evaluation granularity — and the corpus does not yet stress multi-evidence / multi-hop / temporal / contradiction cases.

---

## 2. Hard scenario taxonomy

Each scenario: user ask → evidence location → why top-k fails → minimum capability → failure layer.

### A. Direct evidence lookup

1. **Ask:** “How long does an access token remain valid?”  
2. **Where:** one EvidenceUnit with matching terms.  
3. **Why top-k fails:** usually does not.  
4. **Minimum capability:** structure-preserving index + lexical (or dense) CG.  
5. **Layer:** none when healthy; otherwise indexing.  
**Status vs Continuator:** Level-1 baseline largely works (MEASURED high direct_lookup hit rate).

### B. Paraphrase / terminology mismatch

1. **Ask:** “What if an admin invalidates a refresh credential?”  
2. **Where:** unit saying “revoke refresh token”.  
3. **Why top-k fails:** stem mismatch; related sections with shared headwords outrank gold.  
4. **Minimum capability:** better representation (dense / synonym-aware) and/or ranking that uses more than raw BM25 TF; optional controlled expansion.  
5. **Layer:** **representation** + **ranking** (MEASURED on `api_q03`, `prod_q09`).  
**Lexical alone:** can succeed when residual shared tokens dominate; **not reliable** when critical verbs differ (`invalidate`/`revoke`, `sparse`/`BM25`).

### C. Evidence hidden by competing related sections

1. **Ask:** revocation question; siblings Access/Refresh Tokens rank higher.  
2. **Where:** gold still in candidate pool (e.g. rank 6).  
3. **Why top-k fails:** cutoff / weak discrimination among near neighbors.  
4. **Minimum capability:** rerank or structure-aware scoring; optionally return section neighborhood as a set.  
5. **Layer:** **ranking** (candidates already exist) — MEASURED `api_q03`.

### D. Evidence granularity mismatch

1. **Ask:** nested Markdown structure question.  
2. **Where:** child “Structure Preserved”; parent/sibling “Markdown Documents” also answers.  
3. **Why top-k fails:** exact-ID eval fails while useful evidence retrieved.  
4. **Minimum capability:** parent/sibling expansion **and/or** multi-mode evaluation (supporting vs exact).  
5. **Layer:** **evidence granularity** + **evaluation** — MEASURED `prod_q04`.

### E. Distributed evidence (multi-unit)

1. **Ask:** “What happens after Token X expires?” needing expiry + consequence units.  
2. **Where:** ≥2 units; neither alone sufficient.  
3. **Why top-k fails:** first hit looks “good enough”; second never searched for.  
4. **Minimum capability:** evidence-**set** objective; coverage-aware CG; stop when set sufficient.  
5. **Layer:** **candidate-generation** + **multi-hop/iterative** (lightweight) + **evaluation**.  
**Status:** `UNSUPPORTED` / `UNKNOWN_NOT_YET_EVALUATED` on evidence path (INFERRED — no multi-gold questions in Phase 2).

### F. Cross-reference investigation

1. **Ask:** question answered only after following “see Authentication Policy”.  
2. **Where:** referent section/doc, not the citing unit.  
3. **Why top-k fails:** cite unit ranks; target never queried.  
4. **Minimum capability:** detect references (regex/links/heading mentions) → second search; explicit link edges later.  
5. **Layer:** **multi-hop / iterative** (can start **deterministic**).  
**Status:** `UNSUPPORTED` for EvidenceUnits (MEASURED: no link follower).

### G. Multi-hop evidence

1. **Ask:** chain A→B→C.  
2. **Where:** answer only at end of chain.  
3. **Why top-k fails:** single query never names C’s terms.  
4. **Minimum capability:** iterative CG with extracted concepts from A/B; budgeted rounds; fusion. LLM optional, not required for all chains.  
5. **Layer:** **iterative investigation** + CG.  
**Status:** `UNSUPPORTED` on evidence path. Conversation `iterative` mode is Memory+LLM-eval and must not be counted as done (MEASURED separation).

### H. Ambiguous question / hypothesis generation

1. **Ask:** “Why does it stop working after renewal?”  
2. **Where:** multiple plausible sections.  
3. **Why top-k fails:** commits to one lexical cluster.  
4. **Minimum capability:** multi-hypothesis query variants → union candidates → rank with diversity; defer commitment.  
5. **Layer:** **query interpretation** + CG.  
**Status:** `UNSUPPORTED` on evidence path (`UNKNOWN` how often needed). Multiquery exists for memories only.

### I. Negative evidence / absence

1. **Ask:** “Is feature X supported?” / “Does policy prohibit Y?”  
2. **Where:** may be nowhere; or explicit forbid/allow elsewhere.  
3. **Why top-k fails:** nearest positive-looking chunk ≠ proof of absence.  
4. **Minimum capability:** retrieve support **and** contradict/forbid candidates; explicit “insufficient evidence” outcome; never treat empty/low score as closed-world “no”.  
5. **Layer:** **evaluation** + policy; retrieval cannot prove absence alone.  
**Status:** `UNSUPPORTED` as a product behavior.

### J. Temporal / versioned evidence

1. **Ask:** “What is the current policy?”  
2. **Where:** multiple dated versions.  
3. **Why top-k fails:** older high-overlap unit wins.  
4. **Minimum capability:** ingest `effective_from`/`supersedes`; filter or prefer current; return version provenance.  
5. **Layer:** **indexing/metadata** + ranking filters.  
**Status:** `UNSUPPORTED` (only `content_hash`/`created_at` on Document).

### K. Contradictory evidence

1. **Ask:** any question where two sources disagree.  
2. **Where:** both units must surface.  
3. **Why top-k fails:** returns one winner.  
4. **Minimum capability:** set retrieval + contradiction flagging (deterministic overlap of conflicting assertions later; LLM judge later).  
5. **Layer:** CG + evaluation + provenance.  
**Status:** `UNSUPPORTED`.

### L. Large-document navigation

1. **Ask:** answer buried deep in a long doc.  
2. **Where:** correct unit among hundreds.  
3. **Why top-k fails:** document-level noise; sibling spam.  
4. **Minimum capability:** doc shortlist → local BM25/dense within doc; hierarchical browse using stored `heading_path`.  
5. **Layer:** **hierarchical CG**.  
**Status:** `PARTIALLY_SUPPORTED` structurally (ancestry stored) but **no** two-stage nav API; corpus too small to measure (UNKNOWN empirically).

---

## 3. Capability matrix

| Scenario | Verdict | Why / why not | Failure layer | Minimal missing capability |
|---|---|---|---|---|
| A Direct lookup | `SUPPORTED_NOW` | Phase 2 direct_lookup strong under BM25 | — | Keep baseline regression |
| B Paraphrase / terminology | `PARTIALLY_SUPPORTED` | Some hits; measured failures when verbs/terms diverge | representation + ranking | Dense and/or better ranking; measure before shipping |
| C Competing related sections | `PARTIALLY_SUPPORTED` | Gold often in pool but below cutoff | ranking | Structure-aware rank / CE / set return |
| D Granularity mismatch | `PARTIALLY_SUPPORTED` | Structure stored; no expansion; exact-ID harsh | granularity + evaluation | Parent/sibling expand + soft-credit metrics |
| E Distributed evidence | `UNSUPPORTED` | Single-list exact-ID; no multi-gold | CG + evaluation | Multi-evidence gold + set metrics |
| F Cross-reference | `UNSUPPORTED` | No link follow on EvidenceUnits | iterative CG | Deterministic ref extract → 2nd search |
| G Multi-hop | `UNSUPPORTED` | No evidence-path iteration | iterative CG | Budgeted deterministic then optional LLM |
| H Ambiguous / hypotheses | `UNSUPPORTED` | No evidence multiquery | query interpretation | Deterministic query variants / union |
| I Absence / negative | `UNSUPPORTED` | No insufficient-evidence protocol | evaluation / product policy | Explicit sufficiency states |
| J Temporal / versioned | `UNSUPPORTED` | No effective-time model | indexing metadata | Version fields + filters |
| K Contradictory | `UNSUPPORTED` | Winner-take-all list | CG + evaluation | Forced multi-source return |
| L Large-doc navigation | `UNKNOWN_NOT_YET_EVALUATED` | Ancestry ready; no staged nav; tiny docs | hierarchical CG | Doc→local search experiment |

---

## 4. Agentic-loop analysis

Investigation sketch:

```text
Question → search → inspect → extract clues → search again → stop when sufficient
```

### Can parts be non-LLM?

| Step | Deterministic option |
|---|---|
| Initial search | BM25 / later dense |
| Inspect | Read `heading_path`, siblings by shared prefix, spans |
| Extract clues | Regex links, heading titles, capitalized terms, `section_id` tokens |
| Search again | Issue queries from extracted strings; union/RRF |
| Stop | Budget rounds; coverage of required gold set in eval; heuristic novelty |

LLM useful later for paraphrase hypotheses and sufficiency judgment — **not** required to start.

### Architecture options

| Option | Pattern | Complexity | Cost | Determinism | Hallucination | Debug | Eval | Usefulness |
|---|---|---|---|---|---|---|---|---|
| **A** Single-pass | query→top-k | Low | Low | High | N/A | Easy | Easy | Level 1; weak on B–G |
| **B** Multi-stage deterministic | CG→rank→parent/child→related expand→**set** | Medium | Low–med | High | Low | Easy | Medium | Levels 3–5; fits stored structure |
| **C** Query expansion / iterative | retrieve→extract→search→fuse | Medium | Med | High if extract deterministic | Low | Medium | Medium | Levels 5–6 without agent |
| **D** LLM-guided investigation | retrieve→LLM hypothesize→loop | High | High | Low | **High** | Hard | Hard | Only after C fails measured hard cases |

### Is an agentic detective loop justified **now**?

**NO.**

Reasons (MEASURED + architecture):

1. Proven failures are paraphrase ranking and exact-ID granularity — solvable with eval redesign + ranking/representation, not an agent.  
2. Multi-hop / cross-ref / distributed evidence are **unevaluated** on the evidence path; building Option D before a hard corpus is speculation.  
3. Conversation already has expensive iterative/LLM paths; duplicating that onto EvidenceUnits would multiply cost and opacity.  
4. Option **B→C** is the incremental path consistent with stored `heading_path` and BM25 `RetrievalUnit` reuse.

**ARCHITECTURAL RECOMMENDATION:** Prefer Option B next (after benchmark expansion); Option C when multi-evidence/cross-ref fixtures fail B; Option D only with budgets and only after C’s ceiling is measured.

---

## 5. Minimum viable universal architecture (incremental)

### Level 1 — Direct evidence

```text
Capability: retrieve the obvious unit
Exists: Markdown EvidenceUnit + BM25 + exact-ID eval
Needed: keep regression suite
Evaluate: Phase 2 direct_lookup R@k / MRR
```

### Level 2 — Paraphrase / terminology

```text
Capability: survive synonym/term drift
Exists: partial (shared tokens sometimes enough)
Needed: measured representation/ranking experiment (dense or CE) on paraphrase slice — still behind a flag; OR accept lexical ceiling first via expanded hard paraphrase gold
Evaluate: paraphrase + cross_terminology R@k; compare BM25 vs +dense/+CE offline
```

### Level 3 — Related competition

```text
Capability: prefer correct sibling among related sections
Exists: gold often in top-10 (MEASURED ranks 6–7)
Needed: ranking signal or neighborhood-aware scoring; do not only deepen k
Evaluate: competing-evidence category; rank of gold among sibling set
```

### Level 4 — Hierarchical / parent-child

```text
Capability: expand or credit structural neighbors
Exists: heading_path / section_id persisted; unused at search time
Needed: deterministic expand(parent|siblings|children) after first-pass hits
Evaluate: granularity + parent/child gold; supporting-evidence credit
```

### Level 5 — Multi-evidence answers

```text
Capability: return a sufficient set
Exists: none on evidence path
Needed: multi-ID gold; set metrics; fusion of multiple queries or expansions
Evaluate: coverage@k, partial vs full set
```

### Level 6 — Cross-ref / multi-hop

```text
Capability: follow refs / chains with budget
Exists: none on evidence path
Needed: Option C deterministic iteration; LLM only if measured necessary
Evaluate: chain fixtures; hops used; evidence-set sufficiency
```

Do **not** jump to Level 6 infrastructure before Levels 2–5 have failing benchmarks that isolate the gap.

---

## 6. Benchmark expansion proposal

See companion: [`docs/hard-evidence-retrieval-benchmark-plan.md`](hard-evidence-retrieval-benchmark-plan.md).

Target **~80 questions** (range 60–100), synthetic fixtures extending Phase 2 docs — **plan only**, do not author the full corpus in this task’s code changes beyond documentation.

Priority order for new items: paraphrase/competition isolation → granularity/supporting credit → multi-evidence → cross-ref → multi-hop → temporal/conflict (few).

---

## 7. Evidence-set evaluation proposal

Exact-ID alone is necessary but insufficient for the product vision (MEASURED `prod_q04`).

| Mode | Definition (deterministic) | When to use |
|---|---|---|
| **Exact evidence** | Gold `evidence_id` ∈ top-k | Regression; boundary fidelity |
| **Supporting evidence** | Any unit in gold `supporting_ids` or same `section_id` prefix / listed alternates ∈ top-k | Granularity-tolerant product metric |
| **Multi-evidence coverage** | \|retrieved ∩ required_ids\| / \|required_ids\| | Distributed answers |
| **Partial evidence** | 0 < coverage < 1 | Diagnostics |
| **Evidence sufficiency** | Deterministic rule: all `required_ids` present **or** explicit `sufficient_if_any` group satisfied | Stop condition for iterative CG |

**Do not** add an LLM judge in the next experiment. Soft credit must be **author-listed** in fixtures (supporting IDs / heading-prefix rules), not model-graded.

Protocol sketch:

```text
For each question:
  resolve required_ids, optional supporting_ids
  retrieve ranked list depth K
  report exact_hit@K, supporting_hit@K, coverage@K, partial_flag
```

Primary product metric for hard scenarios: **supporting_hit@K** and **coverage@K**; keep **exact_hit@K** as architecture fidelity metric.

---

## 8. Recommended roadmap

### NOW

**Expand the synthetic hard-scenario benchmark + dual metrics (exact + supporting/set), then re-run BM25-only baseline.**

| | |
|---|---|
| Benefit | Isolates paraphrase vs granularity vs multi-evidence before any prod change |
| Cost | Low (fixtures + harness extension); no models |
| Eval | New categories; exact vs supporting tables |
| Why now | OPTION_F from diagnosis; stops random mechanism shopping |

### NEXT

**Deterministic multi-stage evidence retrieval (Option B): parent/sibling expansion + evidence-set assembly; offline ranking/representation ablation on the hard paraphrase slice.**

| | |
|---|---|
| Benefit | Attacks measured C/D and prepares E |
| Cost | Medium; still no agent |
| Eval | Hard corpus; supporting + coverage |
| Why next | Uses existing `heading_path`; only after NOW shows failures by category |

### LATER

**Dense EvidenceUnit index, CE rerank, deterministic iterative ref-follow (Option C), then — only if still failing — budgeted LLM investigation (Option D). PDF/HTML/code adapters. Temporal/contradiction product features.**

| | |
|---|---|
| Benefit | Levels 2/6+ and corpus universality |
| Cost | High / opaque if agent-first |
| Eval | Hard + domain corpora |
| Why later | Not justified by current measured failures alone; conversation already has expensive loops |

---

## Decision summary

| Question | Answer |
|---|---|
| Biggest architectural gap | No evidence-**set** / hierarchical expansion / multi-evidence objective — plus no hard corpus for E–K — while production still only does single-pass BM25 + exact-ID |
| Agentic loop now? | **NO** |
| Single next experiment | Hard-scenario benchmark expansion + exact/supporting/set metrics, BM25-only re-baseline |
| Must NOT build yet | LLM agent loop, prod embeddings/reranker wiring, query-rewrite prod path, graph-on-EvidenceUnits, PDF/OCR stack |
