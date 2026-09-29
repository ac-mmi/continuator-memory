# Evaluation report: phase2-synthetic-v1

- Run ID: `phase2-synthetic-v1__20260929T153430Z`
- Started: 2026-09-29T15:34:30+00:00  Finished: 2026-09-29T15:37:57+00:00
- Dataset: `benchmarks/synthetic/phase2_benchmark.json` (sha256 `3b7091c565b66f549f85bbfe5024b219cd728ed872a4c225b89c2db34b023d9d`)
- Conversations: 6; questions evaluated: 64 of 64
- Git commit: `None`; package 0.1.0; Python 3.12.7
- Ingest: 161178 ms; total: 207.4 s

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

- Accuracy: **0.875** (56/64)
- Answerable accuracy: 0.912 (n=57)
- Abstention accuracy: 0.571 (n=7)
- Answerable accuracy when an expected source was in the top-5 context: 0.912 (n=57); when not: — (n=0)

| Group | n | Correct | Accuracy | False abstention | Stale mention | Errors |
|---|---|---|---|---|---|---|
| **overall** | 64 | 56 | 0.875 | 0.000 | 0.158 | 0 |
| abstention | 7 | 4 | 0.571 | — | — | 0 |
| basic_fact | 8 | 8 | 1.000 | 0.000 | — | 0 |
| contradiction | 6 | 6 | 1.000 | 0.000 | 0.167 | 0 |
| distractor | 7 | 7 | 1.000 | 0.000 | 0.000 | 0 |
| multi_entity | 7 | 7 | 1.000 | 0.000 | — | 0 |
| multi_hop | 8 | 4 | 0.500 | 0.000 | 1.000 | 0 |
| temporal_update | 8 | 8 | 1.000 | 0.000 | 0.000 | 0 |
| topic_drift | 7 | 7 | 1.000 | 0.000 | — | 0 |
| topic_return | 6 | 5 | 0.833 | 0.000 | 0.500 | 0 |

### Diagnostics

Diagnostics only; none of these change correctness. Coverage is the fraction of expected source messages present in the LLM's context.

| Metric | Value | n |
|---|---|---|
| Current-state accuracy (questions with stale values) | 0.895 | 19 |
| Current-state accuracy without stale mention | 0.842 | 19 |
| Multi-source accuracy (≥2 expected sources) | 0.500 | 8 |
| Multi-source context coverage | 0.875 | 8 |
| Context coverage (answerable) | 0.982 | — |
| Memory source coverage (answerable) | 0.430 | 64 |
| Memory source coverage (multi-source) | 0.312 | 8 |
| Questions with relationships in context | 0.000 | 64 |
| Mean memories in context | 0.9 | 64 |
| Mean superseded memories in context | 0.2 | 64 |

## Memory formation

| Metric | Value |
|---|---|
| Extraction model | `qwen2.5:1.5b` |
| Extraction status (messages) | invalid 1, ok 103 |
| Extracted memories / relationships (after validation) | 186 / 0 |
| Memory consolidation actions | CREATE 50, IGNORE 2, KEEP 93, SUPERSEDE 13, UPDATE 27 |
| Relationship consolidation actions | CREATE 1 |
| Entities | 21 |
| Memories stored (current / superseded) | 69 (56 / 13) |
| Relationships stored (current / superseded) | 1 (1 / 0) |
| Extraction latency | 160.8 s |
| Consolidation latency | 70 ms |

## Per question

| ID | Category | Scope | First rank | RR | Recall@5 | Answer correct | Latency ms |
|---|---|---|---|---|---|---|---|
| basic-01 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 1310 |
| basic-02 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 769 |
| basic-03 | basic_fact | conversation | 2 | 0.500 | 1.000 | yes | 821 |
| basic-04 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 626 |
| basic-05 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 749 |
| basic-06 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 791 |
| basic-07 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 652 |
| basic-08 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 1025 |
| drift-01 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 1118 |
| drift-02 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 747 |
| drift-03 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 562 |
| drift-04 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 617 |
| drift-05 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 805 |
| drift-06 | topic_drift | conversation | 3 | 0.333 | 1.000 | yes | 948 |
| drift-07 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 792 |
| return-01 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 669 |
| return-02 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 370 |
| return-03 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 687 |
| return-04 | topic_return | conversation | 3 | 0.333 | 1.000 | no | 594 |
| return-05 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 1109 |
| return-06 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 448 |
| temporal-01 | temporal_update | conversation | 4 | 0.250 | 1.000 | yes | 689 |
| temporal-02 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 372 |
| temporal-03 | temporal_update | conversation | 3 | 0.333 | 1.000 | yes | 810 |
| temporal-04 | temporal_update | conversation | 3 | 0.333 | 1.000 | yes | 579 |
| temporal-05 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 714 |
| temporal-06 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 738 |
| temporal-07 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 731 |
| temporal-08 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 685 |
| contradiction-01 | contradiction | conversation | 2 | 0.500 | 1.000 | yes | 714 |
| contradiction-02 | contradiction | conversation | 2 | 0.500 | 1.000 | yes | 666 |
| contradiction-03 | contradiction | conversation | 3 | 0.333 | 1.000 | yes | 829 |
| contradiction-04 | contradiction | conversation | 1 | 1.000 | 1.000 | yes | 1030 |
| contradiction-05 | contradiction | conversation | 3 | 0.333 | 1.000 | yes | 672 |
| contradiction-06 | contradiction | conversation | 1 | 1.000 | 1.000 | yes | 626 |
| entity-01 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 506 |
| entity-02 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 512 |
| entity-03 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 626 |
| entity-04 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 588 |
| entity-05 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 769 |
| entity-06 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 794 |
| entity-07 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 700 |
| multihop-01 | multi_hop | conversation | 1 | 1.000 | 1.000 | no | 949 |
| multihop-02 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 398 |
| multihop-03 | multi_hop | conversation | 2 | 0.500 | 1.000 | no | 926 |
| multihop-04 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 952 |
| multihop-05 | multi_hop | conversation | 2 | 0.500 | 0.500 | no | 625 |
| multihop-06 | multi_hop | conversation | 1 | 1.000 | 0.500 | no | 550 |
| multihop-07 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 632 |
| multihop-08 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 640 |
| distractor-01 | distractor | global | 3 | 0.333 | 1.000 | yes | 625 |
| distractor-02 | distractor | global | 1 | 1.000 | 1.000 | yes | 644 |
| distractor-03 | distractor | global | 2 | 0.500 | 1.000 | yes | 697 |
| distractor-04 | distractor | global | 1 | 1.000 | 1.000 | yes | 722 |
| distractor-05 | distractor | global | 2 | 0.500 | 1.000 | yes | 1090 |
| distractor-06 | distractor | global | 3 | 0.333 | 1.000 | yes | 680 |
| distractor-07 | distractor | global | 1 | 1.000 | 1.000 | yes | 584 |
| abstain-01 | abstention | conversation | n/a | n/a | n/a | yes | 590 |
| abstain-02 | abstention | conversation | n/a | n/a | n/a | yes | 551 |
| abstain-03 | abstention | conversation | n/a | n/a | n/a | yes | 570 |
| abstain-04 | abstention | conversation | n/a | n/a | n/a | no | 1057 |
| abstain-05 | abstention | global | n/a | n/a | n/a | no | 648 |
| abstain-06 | abstention | conversation | n/a | n/a | n/a | no | 832 |
| abstain-07 | abstention | conversation | n/a | n/a | n/a | yes | 594 |

## Failures

### return-04 (topic_return)

- Question: What is the current date of the Lumen Health interview?
- Expected answer: October 14th
- Generated answer: The current date of the Lumen Health interview is October 12th.
- Expected sources: m21 (conversation `weekend-drift`)
- First expected rank: 3; top-5 chunks: weekend-drift:c0004, weekend-drift:c0003, weekend-drift:c0009, weekend-drift:c0008, weekend-drift:c0005

### multihop-01 (multi_hop)

- Question: How much more than the original brake-pad quote did the final car repair cost?
- Expected answer: 70 euros
- Generated answer: The final car repair cost was 310 euros, which is 240 euros more than the original garage quote for new brake pads.
- Expected sources: m05, m19 (conversation `weekend-drift`)
- First expected rank: 1; top-5 chunks: weekend-drift:c0009, weekend-drift:c0008, weekend-drift:c0002, weekend-drift:c0001, weekend-drift:c0003

### multihop-03 (multi_hop)

- Question: Who will be leading the Project Atlas team on the current launch date?
- Expected answer: Hana Kobayashi
- Generated answer: The current launch date for Project Atlas is July 1st, and Marco Silva leads the team with a budget of 40,000 euros.
- Expected sources: m13, m19 (conversation `atlas-project`)
- First expected rank: 2; top-5 chunks: atlas-project:c0000, atlas-project:c0006, atlas-project:c0004, atlas-project:c0008, atlas-project:c0001

### multihop-05 (multi_hop)

- Question: Which city is the person who handles the offsite budget from?
- Expected answer: Taipei
- Generated answer: Ines(handles budget)
- Expected sources: m07, m13 (conversation `team-offsite`)
- First expected rank: 2; top-5 chunks: team-offsite:c0000, team-offsite:c0006, team-offsite:c0005, team-offsite:c0004, team-offsite:c0001

### multihop-06 (multi_hop)

- Question: Which city is the team member whose birthday falls during the offsite from?
- Expected answer: Cairo
- Generated answer: Ines is from Lisbon.
- Expected sources: m05, m15 (conversation `team-offsite`)
- First expected rank: 1; top-5 chunks: team-offsite:c0006, team-offsite:c0000, team-offsite:c0005, team-offsite:c0003, team-offsite:c0004

### abstain-04 (abstention)

- Question: What salary was offered for the Lumen Health role?
- Expected answer: (abstain)
- Generated answer: The job interview with Lumen Health for a product manager role was scheduled on October 12th at 10am, followed by an update that it moved to October 14th due to one panelist being sick.
- Expected sources: — (conversation `weekend-drift`)
- First expected rank: None; top-5 chunks: weekend-drift:c0004, weekend-drift:c0009, weekend-drift:c0003, weekend-drift:c0008, weekend-drift:c0001

### abstain-05 (abstention)

- Question: What are the names of Sam's cats?
- Expected answer: (abstain)
- Generated answer: Sam has two cats named Pixel and Byte.
- Expected sources: — (conversation `sam-profile`)
- First expected rank: None; top-5 chunks: team-offsite:c0004, sam-profile:c0003, alex-profile:c0003, sam-profile:c0000, atlas-project:c0000

### abstain-06 (abstention)

- Question: What is the name of the hotel booked for the last two nights?
- Expected answer: (abstain)
- Generated answer: The hotel booked for the last two nights has a Wi-Fi password of harbour-27.
- Expected sources: — (conversation `spain-corrections`)
- First expected rank: None; top-5 chunks: spain-corrections:c0002, spain-corrections:c0003, spain-corrections:c0001, spain-corrections:c0006, spain-corrections:c0007
