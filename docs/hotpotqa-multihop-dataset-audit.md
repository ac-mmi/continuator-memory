# HotpotQA Distractor — Multi-hop Dataset & Architecture Audit

**Audit only.** No retrieval, embeddings, indexing, reranking, query rewriting, or production changes.

**Source file:** `benchmarks/evidence_retrieval/external/hotpotqa/hotpot_dev_distractor_v1.json`

---

## 1. Dataset structure

| Property | Value |
|---|---|
| Format | JSON array of examples |
| **Total examples** | **7405** |
| Top-level fields | `_id`, `question`, `answer`, `context`, `supporting_facts`, `type`, `level` |

### `context`

List of paragraphs. Each item is:

```text
[ title: str, sentences: list[str] ]
```

- `title` — Wikipedia-style article/paragraph title  
- `sentences` — full paragraph split into sentences (not stubs)  
- Observed length: usually **10** paragraphs per question (7345/7405)

### `supporting_facts`

List of gold evidence pointers:

```text
[ title: str, sentence_index: int ]
```

- `sentence_index` is **0-based** into that title’s `sentences` list in `context`
- Example: `["Scott Derrickson", 0]`

### `answer`

String:

- yes/no (458 examples), or  
- short text span (6947 examples)

### `type` / `level`

| Field | Observed values | Counts |
|---|---|---:|
| `type` | `bridge` | 5918 |
| `type` | `comparison` | 1487 |
| `level` | `hard` | 7405 |

**Context contains full paragraph sentences** for each supplied title (not titles alone).

---

## 2. Supporting-fact mapping

Honest mapping onto Continuator identities:

```text
HotpotQA title
  → Document  (one context paragraph / article title)

HotpotQA sentences[] under that title
  → paragraph body

HotpotQA (title, sentence_index)
  → EvidenceUnit  (one sentence)
```

### Is `(title, sentence_index)` sufficient for stable evidence identity?

**Yes, within a question’s distractor context.**

Evidence from this file:

- **0** duplicate titles inside any example’s `context`
- **0** supporting-fact titles missing from `context`
- **1** out-of-range sentence index (`5ae61bfd5542992663a4f261`, index `902` with only 5 sentences) — treat as gold-resolution error

Recommended experimental identity (no schema change):

```text
source_uri = hotpotqa://{example_id}/{url_encoded_title}
evidence   = sentence at sentence_index
```

---

## 3. Multi-hop structure

### Explicitly provided

| Signal | Present? |
|---|---|
| Multiple supporting facts | Yes (2–8 per question) |
| Multiple supporting documents | Yes — **every** example has **exactly 2** supporting titles |
| Question type `bridge` vs `comparison` | Yes |
| Closed distractor paragraph set | Yes |
| Answer string | Yes |

### Must NOT be inferred

| Inference | Allowed? |
|---|---|
| `supporting_facts` list order = Hop-1 → Hop-2 | **No** |
| `comparison` = ordered multi-hop chain | **No** |
| Multiple sentences ⇒ ordered path | **No** |

**Why:**

- `comparison` questions (1487) are parallel attribute checks (e.g. two nationalities), not sequential hops.
- Even for `bridge`, **1504/5918** examples have the **first two** supporting facts from the **same** document — annotation order is not a hop sequence.
- No field names hop edges, hop numbers, or sub-questions.

**Honest reading:** HotpotQA distractor is **multi-document / multi-evidence** gold with a multi-hop *question construction intent* for `bridge`, but **not** an ordered hop-chain label set.

---

## 4. Retrieval evaluation possibilities

| Metric | Status | Why |
|---|---|---|
| **A. Supporting-fact Recall@K** | **SUPPORTED** | Exact gold set `(title, sentence_index)` |
| **B. All-supporting-facts-found@K** | **SUPPORTED** | Coverage@K == 1.0 over that set |
| **C. At-least-one-supporting-fact-found@K** | **SUPPORTED** | Any-gold Hit@K |
| **D. Document Recall@K** | **SUPPORTED** | Supporting titles exact; always 2 gold docs |
| **E. Ordered Hop-1 → Hop-2 success** | **NOT_SUPPORTED** | No hop-order field; SF order ≠ reasoning order |

---

## 5. Corpus leakage / distractor setting

### What DISTRACTOR contains

For each question, the dataset ships a **closed** `context` list:

- Typically **10** paragraphs = **2 gold + 8 distractors**
- Rarely fewer than 10 (see distribution below)

Retrieval in this setting is:

```text
question-specific context set  (2–10 paragraphs)
```

**Not** a shared global corpus.

### Comparability to SciFact

| | SciFact (our prior work) | HotpotQA distractor |
|---|---|---|
| Search space | Shared abstract corpus | Per-question 10-paragraph pool |
| Gold | claim → sentence IDs | question → (title, sent_idx) |
| Global Hit@K comparable? | — | **No** |

Distractor results must be labeled as **closed-pool / distractor-setting** metrics, not global-corpus retrieval.

---

## 6. Fit with our architecture

**Smallest honest representation (no schema changes):**

```text
HotpotQA title/context
  → Document
  → sentence-level EvidenceUnits
```

- Reuse existing `Document` + `EvidenceUnit`
- Optional `heading_path=(title,)` or empty preamble + ordinal sentences
- Gold resolve via `(title, sentence_index)` after materialization
- Do **not** add hop tables or ordered-edge schema

---

## 7. Dataset statistics (computed)

| Statistic | Value |
|---|---:|
| Total examples | **7405** |
| `bridge` | 5918 |
| `comparison` | 1487 |
| Supporting docs = 2 | **7405 (100%)** |
| Supporting docs = 1 | **0** |
| Multi-sentence support in same doc | 2415 |
| Context docs = 10 | 7345 |
| Duplicate titles in context | 0 |
| SF title missing | 0 |
| SF index OOB | 1 |

### Supporting-fact count distribution

| # supporting facts | Count |
|---:|---:|
| 2 | 4990 |
| 3 | 1774 |
| 4 | 537 |
| 5 | 80 |
| 6 | 14 |
| 7 | 9 |
| 8 | 1 |

### Context-document count distribution

| # context docs | Count |
|---:|---:|
| 2 | 21 |
| 3 | 7 |
| 4 | 9 |
| 5 | 6 |
| 6 | 8 |
| 7 | 2 |
| 8 | 4 |
| 9 | 3 |
| 10 | 7345 |

---

## 8. Representative examples

### A. Clear two-document bridge (+ multi-sentence in one doc)

- **id:** `5a8c7595554299585d9e36b6`
- **type:** `bridge`
- **Q:** What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
- **A:** Chief of Protocol
- **SF:**
  - `Kiss and Tell (1945 film)` [0] — Shirley Temple starred as Corliss Archer
  - `Shirley Temple` [0], [1] — later served as Chief of Protocol

### B. Supporting facts do **not** reveal an ordered hop chain (`comparison`)

- **id:** `5a8b57f25542995d1e6f1371`
- **type:** `comparison`
- **Q:** Were Scott Derrickson and Ed Wood of the same nationality?
- **A:** yes
- **SF:** `Scott Derrickson`[0] (American) **and** `Ed Wood`[0] (American) — parallel facts, not Hop-1→Hop-2

### C. Bridge where SF order is not alternating hops

- **id:** `5a85ea095542994775f606a8`
- **type:** `bridge`
- **SF:** three facts from *The Hork-Bajir Chronicles*, then two from *Animorphs*
- Shows annotation order ≠ a clean Hop-1 doc → Hop-2 doc sequence

---

## 9. Final verdict

### **B — Suitable only for multi-evidence-set retrieval, not ordered multi-hop retrieval**

HotpotQA distractor is a strong next benchmark for:

- multi-document supporting-fact coverage  
- closed-pool / distractor retrieval  
- Document + EvidenceUnit materialization  

It is **not** an honest ordered multi-hop chain benchmark: do not score Hop-1→Hop-2 success from `supporting_facts` order.

---

## 10. ONE next implementation step

**Implement a HotpotQA distractor → `Document` + sentence `EvidenceUnit` materializer and gold resolver, with metrics A–D only** (supporting-fact Recall@K, all-found@K, any-found@K, Document Recall@K) over the per-question context pool.

- No embeddings / CE / production wiring yet  
- No ordered-hop metric  
- Handle the 1 OOB supporting-fact as a resolution error  

---

## Files

| Role | Path |
|---|---|
| Dataset inspected | `benchmarks/evidence_retrieval/external/hotpotqa/hotpot_dev_distractor_v1.json` |
| This audit | `docs/hotpotqa-multihop-dataset-audit.md` |
| Machine-readable | `docs/hotpotqa-multihop-dataset-audit.json` |
