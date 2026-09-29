# Evaluation report: phase2-synthetic-v1

- Run ID: `phase2-synthetic-v1__20260929T144248Z`
- Started: 2026-09-29T14:42:48+00:00  Finished: 2026-09-29T14:46:07+00:00
- Dataset: `benchmarks/synthetic/phase2_benchmark.json` (sha256 `3b7091c565b66f549f85bbfe5024b219cd728ed872a4c225b89c2db34b023d9d`)
- Conversations: 6; questions evaluated: 64 of 64
- Git commit: `None`; package 0.1.0; Python 3.12.7
- Ingest: 156527 ms; total: 198.5 s

## Configuration

| Setting | Value |
|---|---|
| pipeline | `phase3-memory` |
| embedding_model | `sentence-transformers/all-MiniLM-L6-v2` |
| ollama_model | `qwen2.5:1.5b` |
| ollama_host | `http://localhost:11434` |
| window_size | `4` |
| overlap | `2` |
| extraction_model | `qwen2.5:1.5b` |
| generation_top_k | `5` |
| retrieval_ks | `[1, 3, 5, 10]` |
| retrieval_only | `False` |
| limit | `None` |

## Retrieval

Scored on answerable questions only. A chunk is relevant if it belongs to the question's conversation and contains an expected source message. Recall@k is the fraction of expected source messages covered by the top-k chunks; Hit@k is whether any is covered; MRR uses the rank of the first relevant chunk.

| Group | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 57 | 0.667 | 0.947 | 0.982 | 1.000 | 0.719 | 0.982 | 1.000 | 1.000 | 0.832 |
| abstention | 0 | — | — | — | — | — | — | — | — | — |
| basic_fact | 8 | 0.875 | 1.000 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 | 0.938 |
| contradiction | 6 | 0.333 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 0.611 |
| distractor | 7 | 0.429 | 1.000 | 1.000 | 1.000 | 0.429 | 1.000 | 1.000 | 1.000 | 0.667 |
| multi_entity | 7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 8 | 0.375 | 0.750 | 0.875 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 0.875 |
| temporal_update | 8 | 0.625 | 0.875 | 1.000 | 1.000 | 0.625 | 0.875 | 1.000 | 1.000 | 0.740 |
| topic_drift | 7 | 0.857 | 1.000 | 1.000 | 1.000 | 0.857 | 1.000 | 1.000 | 1.000 | 0.905 |
| topic_return | 6 | 0.833 | 1.000 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 1.000 | 0.889 |

## Answers

Deterministic scoring: answerable questions are correct when the normalized answer contains an accepted phrase; unanswerable questions are correct when the answer abstains. Stale mention = the answer mentions a superseded or corrected value (diagnostic only, not part of correctness).

- Accuracy: **0.859** (55/64)
- Answerable accuracy: 0.895 (n=57)
- Abstention accuracy: 0.571 (n=7)
- Answerable accuracy when an expected source was in the top-5 context: 0.895 (n=57); when not: — (n=0)

| Group | n | Correct | Accuracy | False abstention | Stale mention | Errors |
|---|---|---|---|---|---|---|
| **overall** | 64 | 55 | 0.859 | 0.000 | 0.158 | 0 |
| abstention | 7 | 4 | 0.571 | — | — | 0 |
| basic_fact | 8 | 8 | 1.000 | 0.000 | — | 0 |
| contradiction | 6 | 5 | 0.833 | 0.000 | 0.167 | 0 |
| distractor | 7 | 7 | 1.000 | 0.000 | 0.000 | 0 |
| multi_entity | 7 | 7 | 1.000 | 0.000 | — | 0 |
| multi_hop | 8 | 4 | 0.500 | 0.000 | 1.000 | 0 |
| temporal_update | 8 | 7 | 0.875 | 0.000 | 0.250 | 0 |
| topic_drift | 7 | 7 | 1.000 | 0.000 | — | 0 |
| topic_return | 6 | 6 | 1.000 | 0.000 | 0.000 | 0 |

### Diagnostics

Diagnostics only; none of these change correctness. Coverage is the fraction of expected source messages present in the LLM's context.

| Metric | Value | n |
|---|---|---|
| Current-state accuracy (questions with stale values) | 0.842 | 19 |
| Current-state accuracy without stale mention | 0.842 | 19 |
| Multi-source accuracy (≥2 expected sources) | 0.500 | 8 |
| Multi-source context coverage | 0.875 | 8 |
| Context coverage (answerable) | 0.982 | — |
| Memory source coverage (answerable) | 0.588 | 64 |
| Memory source coverage (multi-source) | 0.562 | 8 |
| Questions with relationships in context | 0.094 | 64 |
| Mean memories in context | 5.1 | 64 |
| Mean superseded memories in context | 0.9 | 64 |

## Memory formation

| Metric | Value |
|---|---|
| Extraction model | `qwen2.5:1.5b` |
| Extraction status (messages) | invalid 2, ok 102 |
| Extracted memories / relationships (after validation) | 85 / 0 |
| Memory consolidation actions | CREATE 41, IGNORE 1, SUPERSEDE 8, UPDATE 32 |
| Relationship consolidation actions | CREATE 1, IGNORE 1, SUPERSEDE 1 |
| Entities | 18 |
| Memories stored (current / superseded) | 51 (43 / 8) |
| Relationships stored (current / superseded) | 2 (1 / 1) |
| Extraction latency | 156.1 s |
| Consolidation latency | 45 ms |

## Per question

| ID | Category | Scope | First rank | RR | Recall@5 | Answer correct | Latency ms |
|---|---|---|---|---|---|---|---|
| basic-01 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 928 |
| basic-02 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 689 |
| basic-03 | basic_fact | conversation | 2 | 0.500 | 1.000 | yes | 690 |
| basic-04 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 649 |
| basic-05 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 712 |
| basic-06 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 585 |
| basic-07 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 580 |
| basic-08 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 696 |
| drift-01 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 507 |
| drift-02 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 662 |
| drift-03 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 654 |
| drift-04 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 569 |
| drift-05 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 720 |
| drift-06 | topic_drift | conversation | 3 | 0.333 | 1.000 | yes | 728 |
| drift-07 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 1168 |
| return-01 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 692 |
| return-02 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 487 |
| return-03 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 469 |
| return-04 | topic_return | conversation | 3 | 0.333 | 1.000 | yes | 696 |
| return-05 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 643 |
| return-06 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 440 |
| temporal-01 | temporal_update | conversation | 4 | 0.250 | 1.000 | yes | 698 |
| temporal-02 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 338 |
| temporal-03 | temporal_update | conversation | 3 | 0.333 | 1.000 | yes | 679 |
| temporal-04 | temporal_update | conversation | 3 | 0.333 | 1.000 | no | 728 |
| temporal-05 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 698 |
| temporal-06 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 613 |
| temporal-07 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 701 |
| temporal-08 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 580 |
| contradiction-01 | contradiction | conversation | 2 | 0.500 | 1.000 | yes | 868 |
| contradiction-02 | contradiction | conversation | 2 | 0.500 | 1.000 | yes | 736 |
| contradiction-03 | contradiction | conversation | 3 | 0.333 | 1.000 | no | 787 |
| contradiction-04 | contradiction | conversation | 1 | 1.000 | 1.000 | yes | 778 |
| contradiction-05 | contradiction | conversation | 3 | 0.333 | 1.000 | yes | 735 |
| contradiction-06 | contradiction | conversation | 1 | 1.000 | 1.000 | yes | 797 |
| entity-01 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 545 |
| entity-02 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 539 |
| entity-03 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 561 |
| entity-04 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 661 |
| entity-05 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 631 |
| entity-06 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 583 |
| entity-07 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 590 |
| multihop-01 | multi_hop | conversation | 1 | 1.000 | 1.000 | no | 439 |
| multihop-02 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 435 |
| multihop-03 | multi_hop | conversation | 2 | 0.500 | 1.000 | no | 831 |
| multihop-04 | multi_hop | conversation | 1 | 1.000 | 1.000 | no | 767 |
| multihop-05 | multi_hop | conversation | 2 | 0.500 | 0.500 | no | 699 |
| multihop-06 | multi_hop | conversation | 1 | 1.000 | 0.500 | yes | 516 |
| multihop-07 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 682 |
| multihop-08 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 761 |
| distractor-01 | distractor | global | 3 | 0.333 | 1.000 | yes | 781 |
| distractor-02 | distractor | global | 1 | 1.000 | 1.000 | yes | 663 |
| distractor-03 | distractor | global | 2 | 0.500 | 1.000 | yes | 683 |
| distractor-04 | distractor | global | 1 | 1.000 | 1.000 | yes | 584 |
| distractor-05 | distractor | global | 2 | 0.500 | 1.000 | yes | 739 |
| distractor-06 | distractor | global | 3 | 0.333 | 1.000 | yes | 749 |
| distractor-07 | distractor | global | 1 | 1.000 | 1.000 | yes | 590 |
| abstain-01 | abstention | conversation | n/a | n/a | n/a | yes | 722 |
| abstain-02 | abstention | conversation | n/a | n/a | n/a | yes | 305 |
| abstain-03 | abstention | conversation | n/a | n/a | n/a | yes | 523 |
| abstain-04 | abstention | conversation | n/a | n/a | n/a | yes | 644 |
| abstain-05 | abstention | global | n/a | n/a | n/a | no | 746 |
| abstain-06 | abstention | conversation | n/a | n/a | n/a | no | 647 |
| abstain-07 | abstention | conversation | n/a | n/a | n/a | no | 590 |

## Failures

### temporal-04 (temporal_update)

- Question: Who is the current team lead of Project Atlas?
- Expected answer: Hana Kobayashi
- Generated answer: The current team lead of Project Atlas is Marco Silva [CURRENT].
- Expected sources: m13 (conversation `atlas-project`)
- First expected rank: 3; top-5 chunks: atlas-project:c0000, atlas-project:c0001, atlas-project:c0006, atlas-project:c0008, atlas-project:c0004

### contradiction-03 (contradiction)

- Question: Who will the user stay with in Madrid?
- Expected answer: Their cousin Tom
- Generated answer: The user will stay with Leah's place in Madrid.
- Expected sources: m13 (conversation `spain-corrections`)
- First expected rank: 3; top-5 chunks: spain-corrections:c0000, spain-corrections:c0001, spain-corrections:c0006, spain-corrections:c0005, spain-corrections:c0007

### multihop-01 (multi_hop)

- Question: How much more than the original brake-pad quote did the final car repair cost?
- Expected answer: 70 euros
- Generated answer: The final car repair cost 310 euros, which was 240 euros plus additional labor costs.
- Expected sources: m05, m19 (conversation `weekend-drift`)
- First expected rank: 1; top-5 chunks: weekend-drift:c0009, weekend-drift:c0008, weekend-drift:c0002, weekend-drift:c0001, weekend-drift:c0003

### multihop-03 (multi_hop)

- Question: Who will be leading the Project Atlas team on the current launch date?
- Expected answer: Hana Kobayashi
- Generated answer: The current launch date for Project Atlas is July 1st, and Marco Silva leads the team with a budget of 40,000 euros.
- Expected sources: m13, m19 (conversation `atlas-project`)
- First expected rank: 2; top-5 chunks: atlas-project:c0000, atlas-project:c0006, atlas-project:c0004, atlas-project:c0008, atlas-project:c0001

### multihop-04 (multi_hop)

- Question: By how much did the Project Atlas budget increase?
- Expected answer: 15,000 euros
- Generated answer: The Project Atlas budget increased from 40,000 euros to 55,000 euros.
- Expected sources: m03, m11 (conversation `atlas-project`)
- First expected rank: 1; top-5 chunks: atlas-project:c0000, atlas-project:c0005, atlas-project:c0001, atlas-project:c0008, atlas-project:c0004

### multihop-05 (multi_hop)

- Question: Which city is the person who handles the offsite budget from?
- Expected answer: Taipei
- Generated answer: Mei is the one handling the budget for the offsite.
- Expected sources: m07, m13 (conversation `team-offsite`)
- First expected rank: 2; top-5 chunks: team-offsite:c0000, team-offsite:c0006, team-offsite:c0005, team-offsite:c0004, team-offsite:c0001

### abstain-05 (abstention)

- Question: What are the names of Sam's cats?
- Expected answer: (abstain)
- Generated answer: Jonas' cats are Pixel and Byte.
- Expected sources: — (conversation `sam-profile`)
- First expected rank: None; top-5 chunks: team-offsite:c0004, sam-profile:c0003, alex-profile:c0003, sam-profile:c0000, atlas-project:c0000

### abstain-06 (abstention)

- Question: What is the name of the hotel booked for the last two nights?
- Expected answer: (abstain)
- Generated answer: The hotel booked for the last two nights was named Leah.
- Expected sources: — (conversation `spain-corrections`)
- First expected rank: None; top-5 chunks: spain-corrections:c0002, spain-corrections:c0003, spain-corrections:c0001, spain-corrections:c0007, spain-corrections:c0000

### abstain-07 (abstention)

- Question: What is the name of Alex's manager at Norvik Logistics?
- Expected answer: (abstain)
- Generated answer: Alex works at Norvik Logistics [CURRENT] (messages: m03).
- Expected sources: — (conversation `alex-profile`)
- First expected rank: None; top-5 chunks: alex-profile:c0000, alex-profile:c0001, alex-profile:c0002, alex-profile:c0004, alex-profile:c0005
