# Evaluation report: phase2-synthetic-v1

- Run ID: `phase2-synthetic-v1__20260929T124752Z`
- Started: 2026-09-29T12:47:52+00:00  Finished: 2026-09-29T12:48:19+00:00
- Dataset: `benchmarks/synthetic/phase2_benchmark.json` (sha256 `3b7091c565b66f549f85bbfe5024b219cd728ed872a4c225b89c2db34b023d9d`)
- Conversations: 6; questions evaluated: 64 of 64
- Git commit: `None`; package 0.1.0; Python 3.12.7
- Ingest: 211 ms; total: 27.2 s

## Configuration

| Setting | Value |
|---|---|
| embedding_model | `sentence-transformers/all-MiniLM-L6-v2` |
| ollama_model | `qwen2.5:1.5b` |
| ollama_host | `http://localhost:11434` |
| window_size | `4` |
| overlap | `2` |
| generation_top_k | `5` |
| retrieval_ks | `[1, 3, 5, 10]` |
| retrieval_only | `False` |
| limit | `None` |

## Retrieval

Scored on answerable questions only. A chunk is relevant if it belongs to the question's conversation and contains an expected source message. Recall@k is the fraction of expected source messages covered by the top-k chunks; Hit@k is whether any is covered; MRR uses the rank of the first relevant chunk.

| Group | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 57 | 0.667 | 0.947 | 0.947 | 1.000 | 0.719 | 0.982 | 0.982 | 1.000 | 0.830 |
| abstention | 0 | — | — | — | — | — | — | — | — | — |
| basic_fact | 8 | 0.875 | 1.000 | 1.000 | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 | 0.938 |
| contradiction | 6 | 0.333 | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 1.000 | 0.611 |
| distractor | 7 | 0.429 | 1.000 | 1.000 | 1.000 | 0.429 | 1.000 | 1.000 | 1.000 | 0.667 |
| multi_entity | 7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 8 | 0.375 | 0.750 | 0.750 | 1.000 | 0.750 | 1.000 | 1.000 | 1.000 | 0.875 |
| temporal_update | 8 | 0.625 | 0.875 | 0.875 | 1.000 | 0.625 | 0.875 | 0.875 | 1.000 | 0.726 |
| topic_drift | 7 | 0.857 | 1.000 | 1.000 | 1.000 | 0.857 | 1.000 | 1.000 | 1.000 | 0.905 |
| topic_return | 6 | 0.833 | 1.000 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 | 1.000 | 0.889 |

## Answers

Deterministic scoring: answerable questions are correct when the normalized answer contains an accepted phrase; unanswerable questions are correct when the answer abstains. Stale mention = the answer mentions a superseded or corrected value (diagnostic only, not part of correctness).

- Accuracy: **0.812** (52/64)
- Answerable accuracy: 0.825 (n=57)
- Abstention accuracy: 0.714 (n=7)
- Answerable accuracy when an expected source was in the top-5 context: 0.839 (n=56); when not: 0.000 (n=1)

| Group | n | Correct | Accuracy | False abstention | Stale mention | Errors |
|---|---|---|---|---|---|---|
| **overall** | 64 | 52 | 0.812 | 0.000 | 0.158 | 0 |
| abstention | 7 | 5 | 0.714 | — | — | 0 |
| basic_fact | 8 | 8 | 1.000 | 0.000 | — | 0 |
| contradiction | 6 | 4 | 0.667 | 0.000 | 0.167 | 0 |
| distractor | 7 | 6 | 0.857 | 0.000 | 0.000 | 0 |
| multi_entity | 7 | 7 | 1.000 | 0.000 | — | 0 |
| multi_hop | 8 | 4 | 0.500 | 0.000 | 1.000 | 0 |
| temporal_update | 8 | 6 | 0.750 | 0.000 | 0.250 | 0 |
| topic_drift | 7 | 7 | 1.000 | 0.000 | — | 0 |
| topic_return | 6 | 5 | 0.833 | 0.000 | 0.000 | 0 |

## Per question

| ID | Category | Scope | First rank | RR | Recall@5 | Answer correct | Latency ms |
|---|---|---|---|---|---|---|---|
| basic-01 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 1390 |
| basic-02 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 392 |
| basic-03 | basic_fact | conversation | 2 | 0.500 | 1.000 | yes | 577 |
| basic-04 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 441 |
| basic-05 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 410 |
| basic-06 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 411 |
| basic-07 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 328 |
| basic-08 | basic_fact | conversation | 1 | 1.000 | 1.000 | yes | 351 |
| drift-01 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 388 |
| drift-02 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 378 |
| drift-03 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 460 |
| drift-04 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 460 |
| drift-05 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 468 |
| drift-06 | topic_drift | conversation | 3 | 0.333 | 1.000 | yes | 483 |
| drift-07 | topic_drift | conversation | 1 | 1.000 | 1.000 | yes | 405 |
| return-01 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 411 |
| return-02 | topic_return | conversation | 1 | 1.000 | 1.000 | no | 331 |
| return-03 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 310 |
| return-04 | topic_return | conversation | 3 | 0.333 | 1.000 | yes | 316 |
| return-05 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 569 |
| return-06 | topic_return | conversation | 1 | 1.000 | 1.000 | yes | 302 |
| temporal-01 | temporal_update | conversation | 7 | 0.143 | 0.000 | no | 471 |
| temporal-02 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 308 |
| temporal-03 | temporal_update | conversation | 3 | 0.333 | 1.000 | yes | 463 |
| temporal-04 | temporal_update | conversation | 3 | 0.333 | 1.000 | yes | 399 |
| temporal-05 | temporal_update | conversation | 1 | 1.000 | 1.000 | no | 398 |
| temporal-06 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 364 |
| temporal-07 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 551 |
| temporal-08 | temporal_update | conversation | 1 | 1.000 | 1.000 | yes | 379 |
| contradiction-01 | contradiction | conversation | 2 | 0.500 | 1.000 | yes | 464 |
| contradiction-02 | contradiction | conversation | 2 | 0.500 | 1.000 | yes | 417 |
| contradiction-03 | contradiction | conversation | 3 | 0.333 | 1.000 | yes | 425 |
| contradiction-04 | contradiction | conversation | 1 | 1.000 | 1.000 | no | 466 |
| contradiction-05 | contradiction | conversation | 3 | 0.333 | 1.000 | no | 466 |
| contradiction-06 | contradiction | conversation | 1 | 1.000 | 1.000 | yes | 449 |
| entity-01 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 375 |
| entity-02 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 372 |
| entity-03 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 341 |
| entity-04 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 386 |
| entity-05 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 369 |
| entity-06 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 396 |
| entity-07 | multi_entity | conversation | 1 | 1.000 | 1.000 | yes | 314 |
| multihop-01 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 322 |
| multihop-02 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 276 |
| multihop-03 | multi_hop | conversation | 2 | 0.500 | 0.500 | no | 388 |
| multihop-04 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 453 |
| multihop-05 | multi_hop | conversation | 2 | 0.500 | 0.500 | no | 416 |
| multihop-06 | multi_hop | conversation | 1 | 1.000 | 0.500 | no | 341 |
| multihop-07 | multi_hop | conversation | 1 | 1.000 | 1.000 | yes | 352 |
| multihop-08 | multi_hop | conversation | 1 | 1.000 | 0.500 | no | 537 |
| distractor-01 | distractor | global | 3 | 0.333 | 1.000 | yes | 390 |
| distractor-02 | distractor | global | 1 | 1.000 | 1.000 | yes | 333 |
| distractor-03 | distractor | global | 2 | 0.500 | 1.000 | yes | 411 |
| distractor-04 | distractor | global | 1 | 1.000 | 1.000 | yes | 353 |
| distractor-05 | distractor | global | 2 | 0.500 | 1.000 | yes | 586 |
| distractor-06 | distractor | global | 3 | 0.333 | 1.000 | no | 493 |
| distractor-07 | distractor | global | 1 | 1.000 | 1.000 | yes | 333 |
| abstain-01 | abstention | conversation | n/a | n/a | n/a | yes | 441 |
| abstain-02 | abstention | conversation | n/a | n/a | n/a | yes | 364 |
| abstain-03 | abstention | conversation | n/a | n/a | n/a | yes | 359 |
| abstain-04 | abstention | conversation | n/a | n/a | n/a | yes | 364 |
| abstain-05 | abstention | global | n/a | n/a | n/a | no | 375 |
| abstain-06 | abstention | conversation | n/a | n/a | n/a | no | 400 |
| abstain-07 | abstention | conversation | n/a | n/a | n/a | yes | 451 |

## Failures

### return-02 (topic_return)

- Question: When will the user plant the tomatoes?
- Expected answer: April 20th
- Generated answer: Mid-to-late April is sensible for your area.
- Expected sources: m17 (conversation `weekend-drift`)
- First expected rank: 1; top-5 chunks: weekend-drift:c0007, weekend-drift:c0000, weekend-drift:c0001, weekend-drift:c0008, weekend-drift:c0006

### temporal-01 (temporal_update)

- Question: What is the current launch date for Project Atlas?
- Expected answer: July 1st
- Generated answer: The current launch date for Project Atlas is June 17th.
- Expected sources: m19 (conversation `atlas-project`)
- First expected rank: 7; top-5 chunks: atlas-project:c0000, atlas-project:c0004, atlas-project:c0003, atlas-project:c0006, atlas-project:c0002

### temporal-05 (temporal_update)

- Question: On which day of the week is the weekly sync held now?
- Expected answer: Thursday
- Generated answer: [conv:c0007]
- Expected sources: m15 (conversation `atlas-project`)
- First expected rank: 1; top-5 chunks: atlas-project:c0007, atlas-project:c0002, atlas-project:c0006, atlas-project:c0001, atlas-project:c0004

### contradiction-04 (contradiction)

- Question: On which day of the week is the Museo del Faro closed?
- Expected answer: Wednesday
- Generated answer: The Museo del Faro is closed on Mondays.
- Expected sources: m15 (conversation `spain-corrections`)
- First expected rank: 1; top-5 chunks: spain-corrections:c0007, spain-corrections:c0001, spain-corrections:c0006, spain-corrections:c0002, spain-corrections:c0000

### contradiction-05 (contradiction)

- Question: What is the hotel Wi-Fi password?
- Expected answer: harbor-72
- Generated answer: The hotel Wi-Fi password is harbour-72.
- Expected sources: m17 (conversation `spain-corrections`)
- First expected rank: 3; top-5 chunks: spain-corrections:c0003, spain-corrections:c0002, spain-corrections:c0007, spain-corrections:c0008, spain-corrections:c0000

### multihop-03 (multi_hop)

- Question: Who will be leading the Project Atlas team on the current launch date?
- Expected answer: Hana Kobayashi
- Generated answer: Marco Silva is currently leading the Project Atlas team.
- Expected sources: m13, m19 (conversation `atlas-project`)
- First expected rank: 2; top-5 chunks: atlas-project:c0000, atlas-project:c0006, atlas-project:c0004, atlas-project:c0001, atlas-project:c0005

### multihop-05 (multi_hop)

- Question: Which city is the person who handles the offsite budget from?
- Expected answer: Taipei
- Generated answer: Mei is the one handling the offsite budget.
- Expected sources: m07, m13 (conversation `team-offsite`)
- First expected rank: 2; top-5 chunks: team-offsite:c0000, team-offsite:c0006, team-offsite:c0005, team-offsite:c0004, team-offsite:c0001

### multihop-06 (multi_hop)

- Question: Which city is the team member whose birthday falls during the offsite from?
- Expected answer: Cairo
- Generated answer: Lisbon
- Expected sources: m05, m15 (conversation `team-offsite`)
- First expected rank: 1; top-5 chunks: team-offsite:c0006, team-offsite:c0000, team-offsite:c0005, team-offsite:c0003, team-offsite:c0004

### multihop-08 (multi_hop)

- Question: Which team member cannot go on boat trips, and what is their role?
- Expected answer: Omar, the backend developer
- Generated answer: Omar cannot go on boat trips because he gets seasick. Omar's role is QA (Quality Assurance).
- Expected sources: m05, m11 (conversation `team-offsite`)
- First expected rank: 1; top-5 chunks: team-offsite:c0005, team-offsite:c0004, team-offsite:c0000, team-offsite:c0006, team-offsite:c0003

### distractor-06 (distractor)

- Question: Who is flying in for Sam's birthday, and from where?
- Expected answer: Sam's brother, flying in from Dublin
- Generated answer: Sam's birthday is on September 8th. The person flying in from Dublin is Mei.
- Expected sources: m11 (conversation `sam-profile`)
- First expected rank: 3; top-5 chunks: alex-profile:c0005, team-offsite:c0006, sam-profile:c0004, spain-corrections:c0005, spain-corrections:c0000

### abstain-05 (abstention)

- Question: What are the names of Sam's cats?
- Expected answer: (abstain)
- Generated answer: Sam's cats are Pixel and Byte.
- Expected sources: — (conversation `sam-profile`)
- First expected rank: None; top-5 chunks: team-offsite:c0004, sam-profile:c0003, alex-profile:c0003, team-offsite:c0001, team-offsite:c0003

### abstain-06 (abstention)

- Question: What is the name of the hotel booked for the last two nights?
- Expected answer: (abstain)
- Generated answer: Leah's.
- Expected sources: — (conversation `spain-corrections`)
- First expected rank: None; top-5 chunks: spain-corrections:c0002, spain-corrections:c0003, spain-corrections:c0001, spain-corrections:c0007, spain-corrections:c0000
