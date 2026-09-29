# Evaluation report: phase2-synthetic-v1

- Run ID: `phase2-synthetic-v1__20260929T123127Z`
- Started: 2026-09-29T12:31:27+00:00  Finished: 2026-09-29T12:31:28+00:00
- Dataset: `benchmarks/synthetic/phase2_benchmark.json` (sha256 `3b7091c565b66f549f85bbfe5024b219cd728ed872a4c225b89c2db34b023d9d`)
- Conversations: 6; questions evaluated: 64 of 64
- Git commit: `None`; package 0.1.0; Python 3.12.7
- Ingest: 217 ms; total: 0.5 s

## Configuration

| Setting | Value |
|---|---|
| embedding_model | `sentence-transformers/all-MiniLM-L6-v2` |
| ollama_model | `None` |
| ollama_host | `None` |
| window_size | `4` |
| overlap | `2` |
| generation_top_k | `5` |
| retrieval_ks | `[1, 3, 5, 10]` |
| retrieval_only | `True` |
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

Generation skipped (`--retrieval-only`).

## Per question

| ID | Category | Scope | First rank | RR | Recall@5 | Answer correct | Latency ms |
|---|---|---|---|---|---|---|---|
| basic-01 | basic_fact | conversation | 1 | 1.000 | 1.000 | — | 3 |
| basic-02 | basic_fact | conversation | 1 | 1.000 | 1.000 | — | 3 |
| basic-03 | basic_fact | conversation | 2 | 0.500 | 1.000 | — | 3 |
| basic-04 | basic_fact | conversation | 1 | 1.000 | 1.000 | — | 3 |
| basic-05 | basic_fact | conversation | 1 | 1.000 | 1.000 | — | 3 |
| basic-06 | basic_fact | conversation | 1 | 1.000 | 1.000 | — | 3 |
| basic-07 | basic_fact | conversation | 1 | 1.000 | 1.000 | — | 3 |
| basic-08 | basic_fact | conversation | 1 | 1.000 | 1.000 | — | 3 |
| drift-01 | topic_drift | conversation | 1 | 1.000 | 1.000 | — | 3 |
| drift-02 | topic_drift | conversation | 1 | 1.000 | 1.000 | — | 3 |
| drift-03 | topic_drift | conversation | 1 | 1.000 | 1.000 | — | 3 |
| drift-04 | topic_drift | conversation | 1 | 1.000 | 1.000 | — | 3 |
| drift-05 | topic_drift | conversation | 1 | 1.000 | 1.000 | — | 3 |
| drift-06 | topic_drift | conversation | 3 | 0.333 | 1.000 | — | 3 |
| drift-07 | topic_drift | conversation | 1 | 1.000 | 1.000 | — | 3 |
| return-01 | topic_return | conversation | 1 | 1.000 | 1.000 | — | 3 |
| return-02 | topic_return | conversation | 1 | 1.000 | 1.000 | — | 3 |
| return-03 | topic_return | conversation | 1 | 1.000 | 1.000 | — | 3 |
| return-04 | topic_return | conversation | 3 | 0.333 | 1.000 | — | 3 |
| return-05 | topic_return | conversation | 1 | 1.000 | 1.000 | — | 3 |
| return-06 | topic_return | conversation | 1 | 1.000 | 1.000 | — | 3 |
| temporal-01 | temporal_update | conversation | 7 | 0.143 | 0.000 | — | 3 |
| temporal-02 | temporal_update | conversation | 1 | 1.000 | 1.000 | — | 3 |
| temporal-03 | temporal_update | conversation | 3 | 0.333 | 1.000 | — | 3 |
| temporal-04 | temporal_update | conversation | 3 | 0.333 | 1.000 | — | 3 |
| temporal-05 | temporal_update | conversation | 1 | 1.000 | 1.000 | — | 3 |
| temporal-06 | temporal_update | conversation | 1 | 1.000 | 1.000 | — | 3 |
| temporal-07 | temporal_update | conversation | 1 | 1.000 | 1.000 | — | 3 |
| temporal-08 | temporal_update | conversation | 1 | 1.000 | 1.000 | — | 3 |
| contradiction-01 | contradiction | conversation | 2 | 0.500 | 1.000 | — | 3 |
| contradiction-02 | contradiction | conversation | 2 | 0.500 | 1.000 | — | 3 |
| contradiction-03 | contradiction | conversation | 3 | 0.333 | 1.000 | — | 3 |
| contradiction-04 | contradiction | conversation | 1 | 1.000 | 1.000 | — | 3 |
| contradiction-05 | contradiction | conversation | 3 | 0.333 | 1.000 | — | 3 |
| contradiction-06 | contradiction | conversation | 1 | 1.000 | 1.000 | — | 3 |
| entity-01 | multi_entity | conversation | 1 | 1.000 | 1.000 | — | 3 |
| entity-02 | multi_entity | conversation | 1 | 1.000 | 1.000 | — | 3 |
| entity-03 | multi_entity | conversation | 1 | 1.000 | 1.000 | — | 3 |
| entity-04 | multi_entity | conversation | 1 | 1.000 | 1.000 | — | 3 |
| entity-05 | multi_entity | conversation | 1 | 1.000 | 1.000 | — | 3 |
| entity-06 | multi_entity | conversation | 1 | 1.000 | 1.000 | — | 3 |
| entity-07 | multi_entity | conversation | 1 | 1.000 | 1.000 | — | 3 |
| multihop-01 | multi_hop | conversation | 1 | 1.000 | 1.000 | — | 5 |
| multihop-02 | multi_hop | conversation | 1 | 1.000 | 1.000 | — | 3 |
| multihop-03 | multi_hop | conversation | 2 | 0.500 | 0.500 | — | 3 |
| multihop-04 | multi_hop | conversation | 1 | 1.000 | 1.000 | — | 3 |
| multihop-05 | multi_hop | conversation | 2 | 0.500 | 0.500 | — | 3 |
| multihop-06 | multi_hop | conversation | 1 | 1.000 | 0.500 | — | 3 |
| multihop-07 | multi_hop | conversation | 1 | 1.000 | 1.000 | — | 3 |
| multihop-08 | multi_hop | conversation | 1 | 1.000 | 0.500 | — | 3 |
| distractor-01 | distractor | global | 3 | 0.333 | 1.000 | — | 3 |
| distractor-02 | distractor | global | 1 | 1.000 | 1.000 | — | 3 |
| distractor-03 | distractor | global | 2 | 0.500 | 1.000 | — | 3 |
| distractor-04 | distractor | global | 1 | 1.000 | 1.000 | — | 3 |
| distractor-05 | distractor | global | 2 | 0.500 | 1.000 | — | 3 |
| distractor-06 | distractor | global | 3 | 0.333 | 1.000 | — | 3 |
| distractor-07 | distractor | global | 1 | 1.000 | 1.000 | — | 3 |
| abstain-01 | abstention | conversation | n/a | n/a | n/a | — | 3 |
| abstain-02 | abstention | conversation | n/a | n/a | n/a | — | 3 |
| abstain-03 | abstention | conversation | n/a | n/a | n/a | — | 3 |
| abstain-04 | abstention | conversation | n/a | n/a | n/a | — | 3 |
| abstain-05 | abstention | global | n/a | n/a | n/a | — | 3 |
| abstain-06 | abstention | conversation | n/a | n/a | n/a | — | 3 |
| abstain-07 | abstention | conversation | n/a | n/a | n/a | — | 3 |

## Failures

### temporal-01 (temporal_update)

- Question: What is the current launch date for Project Atlas?
- Expected answer: July 1st
- Generated answer: —
- Expected sources: m19 (conversation `atlas-project`)
- First expected rank: 7; top-5 chunks: atlas-project:c0000, atlas-project:c0004, atlas-project:c0003, atlas-project:c0006, atlas-project:c0002
