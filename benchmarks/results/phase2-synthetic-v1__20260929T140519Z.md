# Evaluation report: phase2-synthetic-v1

- Run ID: `phase2-synthetic-v1__20260929T140519Z`
- Started: 2026-09-29T14:05:19+00:00  Finished: 2026-09-29T14:08:51+00:00
- Dataset: `benchmarks/synthetic/phase2_benchmark.json` (sha256 `3b7091c565b66f549f85bbfe5024b219cd728ed872a4c225b89c2db34b023d9d`)
- Conversations: 6; questions evaluated: 64 of 64
- Git commit: `None`; package 0.1.0; Python 3.12.7
- Ingest: 163027 ms; total: 212.1 s

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
| **overall** | 57 | 0.579 | 0.904 | 0.965 | 1.000 | 0.632 | 0.912 | 0.965 | 1.000 | 0.770 |
| abstention | 0 | — | — | — | — | — | — | — | — | — |
| basic_fact | 8 | 0.625 | 0.750 | 0.875 | 1.000 | 0.625 | 0.750 | 0.875 | 1.000 | 0.740 |
| contradiction | 6 | 0.500 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 0.667 |
| distractor | 7 | 0.429 | 1.000 | 1.000 | 1.000 | 0.429 | 1.000 | 1.000 | 1.000 | 0.690 |
| multi_entity | 7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 8 | 0.375 | 0.812 | 1.000 | 1.000 | 0.750 | 0.875 | 1.000 | 1.000 | 0.844 |
| temporal_update | 8 | 0.625 | 0.875 | 1.000 | 1.000 | 0.625 | 0.875 | 1.000 | 1.000 | 0.760 |
| topic_drift | 7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| topic_return | 6 | 0.000 | 0.833 | 0.833 | 1.000 | 0.000 | 0.833 | 0.833 | 1.000 | 0.382 |

## Answers

Deterministic scoring: answerable questions are correct when the normalized answer contains an accepted phrase; unanswerable questions are correct when the answer abstains. Stale mention = the answer mentions a superseded or corrected value (diagnostic only, not part of correctness).

- Accuracy: **0.797** (51/64)
- Answerable accuracy: 0.842 (n=57)
- Abstention accuracy: 0.429 (n=7)
- Answerable accuracy when an expected source was in the top-5 context: 0.873 (n=55); when not: 0.000 (n=2)

| Group | n | Correct | Accuracy | False abstention | Stale mention | Errors |
|---|---|---|---|---|---|---|
| **overall** | 64 | 51 | 0.797 | 0.035 | 0.211 | 0 |
| abstention | 7 | 3 | 0.429 | — | — | 0 |
| basic_fact | 8 | 7 | 0.875 | 0.125 | — | 0 |
| contradiction | 6 | 4 | 0.667 | 0.000 | 0.333 | 0 |
| distractor | 7 | 7 | 1.000 | 0.000 | 0.000 | 0 |
| multi_entity | 7 | 7 | 1.000 | 0.000 | — | 0 |
| multi_hop | 8 | 5 | 0.625 | 0.000 | 1.000 | 0 |
| temporal_update | 8 | 7 | 0.875 | 0.000 | 0.250 | 0 |
| topic_drift | 7 | 6 | 0.857 | 0.000 | — | 0 |
| topic_return | 6 | 5 | 0.833 | 0.167 | 0.000 | 0 |

### Diagnostics

Diagnostics only; none of these change correctness. Coverage is the fraction of expected source messages present in the LLM's context.

| Metric | Value | n |
|---|---|---|
| Current-state accuracy (questions with stale values) | 0.737 | 19 |
| Current-state accuracy without stale mention | 0.684 | 19 |
| Multi-source accuracy (≥2 expected sources) | 0.625 | 8 |
| Multi-source context coverage | 1.000 | 8 |
| Context coverage (answerable) | 0.965 | — |
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
| Extraction latency | 162.7 s |
| Consolidation latency | 48 ms |

## Per question

| ID | Category | Scope | First rank | RR | Recall@5 | Answer correct | Latency ms |
|---|---|---|---|---|---|---|---|
| basic-01 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 969 |
| basic-02 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 729 |
| basic-03 | basic_fact | conversation | 2 | 0.500 | 1.000 | yes | 673 |
| basic-04 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 782 |
| basic-05 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 855 |
| basic-06 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 656 |
| basic-07 | basic_fact | conversation | 4 | 0.250 | 1.000 | yes | 547 |
| basic-08 | basic_fact | conversation | 6 | 0.167 | 0.000 | no | 870 |
| drift-01 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 574 |
| drift-02 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 808 |
| drift-03 | topic_drift | conversation | 1 | 1.000 | 1.000 | no | 912 |
| drift-04 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 676 |
| drift-05 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 850 |
| drift-06 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 847 |
| drift-07 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 890 |
| return-01 | topic_return | conversation | 2 | 0.500 | 1.000 | yes | 659 |
| return-02 | topic_return | conversation | 2 | 0.500 | 1.000 | yes | 353 |
| return-03 | topic_return | conversation | 3 | 0.333 | 1.000 | yes | 351 |
| return-04 | topic_return | conversation | 8 | 0.125 | 0.000 | no | 849 |
| return-05 | topic_return | conversation | 3 | 0.333 | 1.000 | yes | 1026 |
| return-06 | topic_return | conversation | 2 | 0.500 | 1.000 | yes | 334 |
| temporal-01 | temporal_update | conversation | 4 | 0.250 | 1.000 | yes | 762 |
| temporal-02 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 542 |
| temporal-03 | temporal_update | conversation | 2 | 0.500 | 1.000 | yes | 810 |
| temporal-04 | temporal_update | conversation | 3 | 0.333 | 1.000 | no | 1011 |
| temporal-05 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 859 |
| temporal-06 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 829 |
| temporal-07 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 753 |
| temporal-08 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 657 |
| contradiction-01 | contradiction | conversation | 3 | 0.333 | 1.000 | yes | 1116 |
| contradiction-02 | contradiction | conversation | 3 | 0.333 | 1.000 | yes | 811 |
| contradiction-03 | contradiction | conversation | 3 | 0.333 | 1.000 | no | 947 |
| contradiction-04 | contradiction | conversation | 1 | 1.000 | 1.000 | yes | 1027 |
| contradiction-05 | contradiction | conversation | 1 | 1.000 | 1.000 | no | 847 |
| contradiction-06 | contradiction | conversation | 1 | 1.000 | 1.000 | yes | 865 |
| entity-01 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 659 |
| entity-02 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 670 |
| entity-03 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 742 |
| entity-04 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 923 |
| entity-05 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 1442 |
| entity-06 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 1130 |
| entity-07 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 659 |
| multihop-01 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 540 |
| multihop-02 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 475 |
| multihop-03 | multi_hop | conversation | 4 | 0.250 | 1.000 | no | 789 |
| multihop-04 | multi_hop | conversation | 1 | 1.000 | 1.000 | no | 744 |
| multihop-05 | multi_hop | conversation | 1 | 1.000 | 1.000 | no | 830 |
| multihop-06 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 597 |
| multihop-07 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 860 |
| multihop-08 | multi_hop | conversation | 2 | 0.500 | 1.000 | yes | 857 |
| distractor-01 | distractor | global | 2 | 0.500 | 1.000 | yes | 909 |
| distractor-02 | distractor | global | 1 | 1.000 | 1.000 | yes | 733 |
| distractor-03 | distractor | global | 3 | 0.333 | 1.000 | yes | 703 |
| distractor-04 | distractor | global | 1 | 1.000 | 1.000 | yes | 703 |
| distractor-05 | distractor | global | 2 | 0.500 | 1.000 | yes | 764 |
| distractor-06 | distractor | global | 2 | 0.500 | 1.000 | yes | 958 |
| distractor-07 | distractor | global | 1 | 1.000 | 1.000 | yes | 389 |
| abstain-01 | abstention | conversation | n/a | n/a | n/a | yes | 790 |
| abstain-02 | abstention | conversation | n/a | n/a | n/a | no | 388 |
| abstain-03 | abstention | conversation | n/a | n/a | n/a | yes | 632 |
| abstain-04 | abstention | conversation | n/a | n/a | n/a | yes | 812 |
| abstain-05 | abstention | global | n/a | n/a | n/a | no | 883 |
| abstain-06 | abstention | conversation | n/a | n/a | n/a | no | 707 |
| abstain-07 | abstention | conversation | n/a | n/a | n/a | no | 702 |

## Failures

### basic-08 (basic_fact)

- Question: What is the name of the school where Alex is learning Portuguese?
- Expected answer: Falar Bem
- Generated answer: I don't know; that information is not in the conversation.
- Expected sources: m13 (conversation `alex-profile`)
- First expected rank: 6; top-5 chunks: alex-profile:c0002, alex-profile:c0001, alex-profile:c0000, alex-profile:c0004, alex-profile:c0003

### drift-03 (topic_drift)

- Question: Why does the user need to renew their passport?
- Expected answer: For a trip to Japan in November
- Generated answer: The user needs to renew their passport because they have a trip planned in November, which requires them to apply soon.
- Expected sources: m13 (conversation `weekend-drift`)
- First expected rank: 1; top-5 chunks: weekend-drift:c0006, weekend-drift:c0007, weekend-drift:c0000, weekend-drift:c0008, weekend-drift:c0001

### return-04 (topic_return)

- Question: What is the current date of the Lumen Health interview?
- Expected answer: October 14th
- Generated answer: I don't know; that information is not in the conversation.
- Expected sources: m21 (conversation `weekend-drift`)
- First expected rank: 8; top-5 chunks: weekend-drift:c0008, weekend-drift:c0007, weekend-drift:c0006, weekend-drift:c0000, weekend-drift:c0001

### temporal-04 (temporal_update)

- Question: Who is the current team lead of Project Atlas?
- Expected answer: Hana Kobayashi
- Generated answer: The current team lead of Project Atlas is Marco Silva [CURRENT].
- Expected sources: m13 (conversation `atlas-project`)
- First expected rank: 3; top-5 chunks: atlas-project:c0000, atlas-project:c0001, atlas-project:c0005, atlas-project:c0006, atlas-project:c0004

### contradiction-03 (contradiction)

- Question: Who will the user stay with in Madrid?
- Expected answer: Their cousin Tom
- Generated answer: The user will stay with Leah's place in Madrid.
- Expected sources: m13 (conversation `spain-corrections`)
- First expected rank: 3; top-5 chunks: spain-corrections:c0000, spain-corrections:c0001, spain-corrections:c0005, spain-corrections:c0006, spain-corrections:c0002

### contradiction-05 (contradiction)

- Question: What is the hotel Wi-Fi password?
- Expected answer: harbor-72
- Generated answer: The hotel Wi-Fi password is harbor-27.
- Expected sources: m17 (conversation `spain-corrections`)
- First expected rank: 1; top-5 chunks: spain-corrections:c0007, spain-corrections:c0008, spain-corrections:c0000, spain-corrections:c0003, spain-corrections:c0002

### multihop-03 (multi_hop)

- Question: Who will be leading the Project Atlas team on the current launch date?
- Expected answer: Hana Kobayashi
- Generated answer: Marco Silva is leading the Project Atlas team with a budget of 40,000 euros for the current launch date.
- Expected sources: m13, m19 (conversation `atlas-project`)
- First expected rank: 4; top-5 chunks: atlas-project:c0000, atlas-project:c0001, atlas-project:c0004, atlas-project:c0006, atlas-project:c0008

### multihop-04 (multi_hop)

- Question: By how much did the Project Atlas budget increase?
- Expected answer: 15,000 euros
- Generated answer: The Project Atlas budget increased from 40,000 euros to 55,000 euros.
- Expected sources: m03, m11 (conversation `atlas-project`)
- First expected rank: 1; top-5 chunks: atlas-project:c0000, atlas-project:c0005, atlas-project:c0004, atlas-project:c0003, atlas-project:c0008

### multihop-05 (multi_hop)

- Question: Which city is the person who handles the offsite budget from?
- Expected answer: Taipei
- Generated answer: Mei is the one handling the budget for the offsite.
- Expected sources: m07, m13 (conversation `team-offsite`)
- First expected rank: 1; top-5 chunks: team-offsite:c0005, team-offsite:c0006, team-offsite:c0002, team-offsite:c0003, team-offsite:c0000

### abstain-02 (abstention)

- Question: What is Omar's last name?
- Expected answer: (abstain)
- Generated answer: Omar is the backend developer [CURRENT] (messages: m06).
- Expected sources: — (conversation `team-offsite`)
- First expected rank: None; top-5 chunks: team-offsite:c0001, team-offsite:c0002, team-offsite:c0006, team-offsite:c0004, team-offsite:c0003

### abstain-05 (abstention)

- Question: What are the names of Sam's cats?
- Expected answer: (abstain)
- Generated answer: Jonas has two cats named Pixel and Byte.
- Expected sources: — (conversation `sam-profile`)
- First expected rank: None; top-5 chunks: team-offsite:c0004, team-offsite:c0003, sam-profile:c0003, sam-profile:c0002, sam-profile:c0000

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
- First expected rank: None; top-5 chunks: alex-profile:c0000, alex-profile:c0001, alex-profile:c0002, alex-profile:c0004, alex-profile:c0003
