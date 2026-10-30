# continuator-memory

Research project comparing conversational memory architectures. Phase 1 is a
minimal, deterministic conversational RAG baseline that later phases are
benchmarked against.

## Phase 1 architecture

```
conversation JSON
  → parse + validate            (models.py)
  → sliding-window chunking     (chunking.py)
  → local embeddings            (embedding.py, Sentence Transformers on CPU)
  → SQLite persistence          (store.py)
  → cosine similarity, top-k    (retrieval.py, brute force in NumPy)
  → prompt = top-k chunks + question → Ollama   (generation.py)
  → answer
```

`pipeline.py` wires these together and `cli.py` exposes them.

That is the entire system. Phase 1 has no long-term memory, memory extraction,
summarization, topic detection, agents, or external vector database.

### Components

| Module | Responsibility |
| --- | --- |
| `models.py` | `Message`, `Conversation`, `Chunk`, `RetrievedChunk`; JSON parsing/validation |
| `chunking.py` | Deterministic window over whole messages (default 4 messages, overlap 2) |
| `embedding.py` | `SentenceTransformerEmbedder`, CPU, L2-normalized float32 vectors |
| `store.py` | SQLite schema and read/write |
| `retrieval.py` | `cosine_similarity`, `top_k`, `Retriever` |
| `generation.py` | Prompt construction, Ollama `/api/chat` client (stdlib HTTP) |
| `pipeline.py` | `ingest`, `retrieve`, `ask` |
| `cli.py` | `ingest`, `retrieve`, `ask` commands |

### Chunking

Messages are grouped into windows of `window_size` consecutive messages with
`overlap` shared messages between neighbouring windows (stride =
`window_size - overlap`). Windows are emitted until one contains the final
message, so every message is covered and the last window may be shorter.
Chunk text is `"{role}: {content}"` per message, joined by newlines. Chunk IDs
are `{conversation_id}:c{position:04d}`.

### Storage (SQLite)

| Table | Contents |
| --- | --- |
| `conversations` | `conversation_id`, `title` |
| `messages` | `(conversation_id, message_id)` PK, `position`, `role`, `content`, `timestamp` |
| `chunks` | `chunk_id`, `conversation_id`, `position`, `text`, `metadata_json` (chunker name, window, overlap, start/end position) |
| `chunk_messages` | provenance: `chunk_id` → ordered `(conversation_id, message_id)` |
| `chunk_embeddings` | `chunk_id`, `model_name`, `dim`, float32 `vector` BLOB |

Re-ingesting a conversation ID replaces it atomically. Embeddings are keyed by
model name, so retrieval only compares vectors from the configured model.

### Retrieval and provenance

Every stored embedding for the configured model (optionally limited to one
conversation) is loaded into memory and scored by cosine similarity against the
query embedding. The top `k` are returned, sorted by score descending with
ties broken by `chunk_id`. Each result contains:

- `chunk_id`
- `conversation_id`
- `source_message_ids` (ordered; resolvable via `Store.get_messages`)
- `score` (cosine similarity)
- `text`

### Generation

The prompt lists each retrieved chunk as `[chunk_id] (messages: ...)` followed
by its text, then the question. The system prompt tells the model to answer
only from the excerpts and cite chunk IDs. Ollama runs with `temperature=0`
and `seed=0`.

### Determinism

- Chunking depends only on the input and the window/overlap parameters.
- The embedding model name is fixed and stored alongside every vector.
- `k` is explicit, and ties are broken by `chunk_id`.
- Generation uses temperature 0 and a fixed seed.
- The core code generates no random or synthetic data.

## Running locally

Requirements: Python ≥ 3.10 and [Ollama](https://ollama.com).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

ollama pull qwen2.5:1.5b   # default generation model
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) into
the Hugging Face cache.

### Ingest the example conversation

```bash
continuator-memory ingest benchmarks/synthetic/trip_planning.json
```

### Inspect retrieved chunks

```bash
continuator-memory retrieve "What is Priya allergic to?" -c trip-planning-001 -k 3
continuator-memory retrieve "What is Priya allergic to?" -c trip-planning-001 -k 3 --json
```

### Ask a question

```bash
continuator-memory ask "What is Priya allergic to?" -c trip-planning-001 -k 3
continuator-memory ask "When is the Sintra trip?" -k 3 --show-chunks
continuator-memory ask "When is the Sintra trip?" -k 3 --json
```

Leaving out `-c/--conversation` searches every ingested conversation.
`python -m continuator_memory ...` works the same way.

### Memory extraction (opt-in)

Imports default to `extract="messages"`, one stored record per message with no
model call. `extract="memories"` is opt-in, available through `POST /import` and
`ingest_imported`. It is English-only.

Enforced by code:

- Each memory cites its own source message, speaker label, role and timestamp.
- A claim whose content stems are not in its source is rejected, and so is a
  claim that cannot be compared (no source tokens, or text outside Latin script).
- Memories are superseded only by an explicit relation or an identity-safe slot
  match, and old versions are kept.
- Re-import over stored memories is failure-safe. Nothing is written until
  extraction finishes, and the replacement is one transaction. A run with a
  failed window, or with no memories because output was rejected or missing,
  keeps the stored conversation and memories.

Requested of the model in the prompt, and not enforced by code: durability
selection, not turning assistant suggestions into user facts, and keeping
hypotheticals and negation intact.

Known limits:

- Durability is not enforced, so temporary events and small talk may be stored
  as current memories.
- Source validation compares word stems, not meaning. It cannot guarantee
  entailment: it rejects many faithful paraphrases and does not detect a flipped
  negation or a changed speaker.
- Unsupported inferred details can pass validation, including gendered pronouns
  the speaker never used, because pronouns are stopwords.
- Temporal status comes from keywords and can be wrong.
- Subject names are not normalized ("Sam" and "the user" stay separate), and a
  correction can only link to one of the last 20 claims. A correction split into
  several memories may link only one of them to the old fact.
- Validator rejections from a run that still produces memories are not failures,
  so a re-import can store fewer memories. A run where the model reports nothing
  durable in every message counts as successful and replaces stored memories.

- Headerless transcript parsing is best-effort and can misattribute speakers.
  For example, with only single blank lines between paragraphs, an uppercase user
  paragraph can merge into the assistant turn, so its facts carry the assistant
  role. Use explicit speaker headers (`You:` and `ChatGPT:`) or JSON when speaker
  attribution matters.

The heuristic evidence evaluator in `webapi/evidence_eval.py` contains
benchmark-specific logic for the names Caroline and Melanie, the speakers of
LoCoMo conv-26, so its results can change with the name. It is diagnostic-only:
`iterative_diag` and `ranking_diag` use it, and normal API answer generation does
not.

See `reports/memory_extraction_v1.md` (§9 smoke test, §10 limitations) and
`reports/memory_extraction_heuristic_audit.md`.

### Configuration

| Environment variable | Default |
| --- | --- |
| `CONTINUATOR_DB_PATH` (or `--db`) | `data/continuator.db` |
| `CONTINUATOR_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `CONTINUATOR_OLLAMA_MODEL` | `qwen2.5:1.5b` |
| `OLLAMA_HOST` | `http://localhost:11434` |
| `CONTINUATOR_CHUNK_WINDOW` | `4` |
| `CONTINUATOR_CHUNK_OVERLAP` | `2` |
| `CONTINUATOR_TOP_K` | `5` |

Changing the chunking parameters or embedding model requires re-ingesting.

### Input format

```json
{
  "conversation_id": "trip-planning-001",
  "title": "optional",
  "messages": [
    {"id": "m001", "role": "user", "content": "...", "timestamp": "optional"},
    {"id": "m002", "role": "assistant", "content": "..."}
  ]
}
```

`role` must be one of `system`, `user`, `assistant`. If `id` is missing, it
becomes `{conversation_id}:m{position:04d}`. Message IDs must be unique within
a conversation.

### Tests

```bash
pytest                      # all tests
pytest tests/unit           # fast, no model required (hashing test embedder)
pytest -m integration       # real MiniLM retrieval on the example dataset
```

## Example dataset

`benchmarks/synthetic/trip_planning.json` is a hand-written 18-message
conversation. `trip_planning_questions.json` lists questions with expected
answers and the message IDs that contain them, for later retrieval and answer
evaluation.

## Phase 1 limitations

- Retrieval is brute force over every stored vector, which is fine at this scale.
- Chunks are whole-message windows. A very long message is not split, and text
  beyond the embedding model's 256-token limit is truncated.
- The embedding model revision is not pinned. The model name is recorded, but
  a Hugging Face update to the same name could change vectors.
- Ollama output with temperature 0 and a fixed seed is reproducible on the
  same machine, model, and Ollama version, but not guaranteed across them.
- The small default LLM does not always cite chunk IDs as instructed.

## Phase 2: evaluation harness

Phase 2 measures the frozen Phase 1 baseline as a control group. It adds no
memory features and does not change retrieval or generation.

### What a run does

1. Loads a benchmark dataset: conversations, plus questions with expected
   answers and expected source message IDs.
2. Ingests every conversation into a fresh in-memory SQLite index. Your
   `data/continuator.db` is never touched.
3. For each question:
   - Calls `Pipeline.retrieve(question, 10, scope)` to score retrieval.
   - Calls `Pipeline.ask(question, k, scope)` to generate the answer, using
     the unchanged Phase 1 path and `k = --top-k` (default 5).
4. Writes a JSON results file and a Markdown report to `benchmarks/results/`.

Code lives in `src/continuator_memory/evaluation/`: `dataset.py`,
`metrics.py`, `harness.py`, and `report.py`.

### Metrics

Retrieval and answer correctness are scored separately.

**Retrieval** is scored on answerable questions only. A retrieved chunk is
*relevant* if it belongs to the question's conversation and contains at least
one expected source message.

- **First expected rank**: the rank of the first relevant chunk, or none if no
  relevant chunk is retrieved.
- **Reciprocal rank**: 1/rank, or 0 if nothing relevant was retrieved.
  **MRR** is its mean.
- **Recall@k** (k = 1, 3, 5, 10): the fraction of expected source messages
  covered by the top-k chunks. This is fractional for multi-hop questions that
  need two messages.
- **Hit@k**: whether any expected source message is in the top k.

**Answers** are scored deterministically, with no LLM judge. Text is
normalized by casefolding, stripping accents, turning punctuation into spaces,
and collapsing whitespace. Phrases then match on word boundaries.

- *Answerable questions* are correct when the answer contains one of the
  `match_any` phrases (default: the expected answer) and all of the
  `match_all` phrases.
- *Abstention questions* are correct when the answer contains an abstention
  phrase such as "don't know", "not mentioned", or "no information".
- **Diagnostics**, which do not count toward correctness:
  - *false abstention*: the model abstained on an answerable question.
  - *stale mention*: the answer mentions a superseded or corrected value
    listed in `stale_answers`.
  - Answerable accuracy is also split by whether an expected source was in the
    top-k context given to the LLM, so you can tell retrieval failures apart
    from generation failures.

Per question, the results record:
- question ID, category, and scope
- the top-10 retrieved chunks, expected source IDs, first expected rank,
  reciprocal rank, and Recall/Hit@k
- the generated answer, expected answer, match result, and diagnostic flags
- the chunk IDs used as context
- latency for retrieval, generation, and in total

Aggregates are reported overall and per category. The run metadata records the
dataset SHA-256, configuration, git commit, and Python version.

### Dataset format

```json
{
  "dataset_id": "phase2-synthetic-v1",
  "conversations": [ <Phase 1 conversation objects> ],
  "questions": [
    {"id": "temporal-01", "category": "temporal_update",
     "conversation_id": "atlas-project",
     "scope": "conversation",
     "question": "What is the current launch date for Project Atlas?",
     "expected_answer": "July 1st",
     "match_any": ["July 1st", "July 1", "1st of July"],
     "stale_answers": ["June 3rd", "June 17th"],
     "expected_source_message_ids": ["m19"]},
    {"id": "abstain-01", "category": "abstention", "conversation_id": "alex-profile",
     "answerable": false, "question": "What is Alex's favorite movie?",
     "expected_answer": null, "expected_source_message_ids": []}
  ]
}
```

`scope` is either `conversation`, which restricts retrieval to the question's
conversation (the default), or `global`, which searches every conversation in
the dataset.

### Benchmark: `benchmarks/synthetic/phase2_benchmark.json`

The benchmark has 64 hand-designed questions over 6 conversations:

| Category | n | Failure mode targeted |
| --- | --- | --- |
| `basic_fact` | 8 | single-message fact lookup |
| `topic_drift` | 7 | facts introduced in off-topic digressions |
| `topic_return` | 6 | facts from later messages that return to an earlier topic |
| `temporal_update` | 8 | values that change over time (current vs original) |
| `contradiction` | 6 | user corrections of earlier statements |
| `multi_entity` | 7 | attributes of one person among four similar people |
| `multi_hop` | 8 | two source messages that never share a chunk |
| `distractor` | 7 | global search across near-duplicate profiles (Alex/Sam) |
| `abstention` | 7 | information not present, including near-miss distractors |

### Running an evaluation

```bash
source .venv/bin/activate
ollama serve   # if not already running

# full run: retrieval and generation
continuator-memory evaluate benchmarks/synthetic/phase2_benchmark.json

# retrieval metrics only (no Ollama calls)
continuator-memory evaluate benchmarks/synthetic/phase2_benchmark.json --retrieval-only

# quick check on the first 3 questions
continuator-memory evaluate benchmarks/synthetic/phase2_benchmark.json --limit 3
```

Each run writes `benchmarks/results/<dataset_id>__<UTC timestamp>.json` and a
`.md` file with the same name. Environment variables from Phase 1
(`CONTINUATOR_OLLAMA_MODEL`, `CONTINUATOR_EMBEDDING_MODEL`, chunk settings)
apply and are recorded in the results.

## Phase 3: memory formation

Phase 3 adds a structured memory layer, turned on with `--memory`. It sits
next to the Phase 1 chunk index and does not replace it. Without `--memory`,
every command behaves exactly as in Phases 1 and 2.

### How it works

1. **Extraction** (`memory/extraction.py`): each message is sent to Ollama
   (`CONTINUATOR_EXTRACTION_MODEL`, which defaults to the generation model)
   together with the last 3 messages and the currently known facts, both
   marked "do not extract". The model must return JSON that matches a schema
   (`{"facts": [{subject, predicate, value, kind, previous_value}]}`). Every
   item is validated. Its value must be grounded in the message text, and its
   subject in the message or its context. Invalid output is logged in
   `extraction_log` and skipped; it never raises.
2. **Consolidation** (`memory/consolidation.py`, `memory/formation.py`):
   each fact is routed to a relationship if its value names a known entity,
   and to an attribute memory otherwise. It is then compared with the
   existing records in its slot (entity + predicate) and gets one of these
   actions: CREATE, UPDATE (a new source for an existing value), SUPERSEDE
   (closes the old record with `valid_until`, keeps it as history, and links
   it with `supersedes_*_id`), KEEP, or IGNORE. A `reference` to a past value
   is always ignored. Reverting to a superseded value requires the replaced
   value to be stated in the message. Every decision is logged in
   `consolidation_log`.
3. **Retrieval** (`memory/retrieval.py`): records are scored by IDF-weighted
   lexical overlap with the question, plus a bonus when the question names
   the entity. That score is multiplied by semantic support, which is how
   highly the record's source messages rank in the Phase 1 semantic search.
   Superseded records are down-weighted unless the question asks about the
   past (for example "originally" or "previously"). Matched entities are
   expanded one hop along relationships. Supporting chunks are chosen by
   fusing the semantic ranking with the chunks that contain the memories'
   source messages.
4. **Context** (`memory/context.py`): the prompt contains a `MEMORY:` block
   with each fact marked `[CURRENT]` or `[SUPERSEDED by X]`, plus the source
   message IDs, followed by the conversation excerpts. When nothing relevant
   is stored or retrieved, the model is told to answer with a fixed "I don't
   know" phrase.

Everything is stored in the same SQLite database, in the tables `entities`,
`memories`, `memory_sources`, `relationships`, `relationship_sources`,
`extraction_log`, and `consolidation_log`. All of them cascade from
`conversations`, so re-ingesting a conversation rebuilds its memory.

### Commands

```bash
continuator-memory ingest --memory benchmarks/synthetic/trip_planning.json
continuator-memory memories -c <conversation_id> [--current-only]
continuator-memory retrieve --memory "Where are we staying now?"
continuator-memory ask --memory "Where are we staying now?"

# Phase 3 benchmark (same dataset, questions, and scorer as Phase 2)
continuator-memory evaluate benchmarks/synthetic/phase2_benchmark.json --memory
```

Memory ingest makes one LLM call per message, roughly 1 to 1.5 s each with
`qwen2.5:1.5b`. Results record `"pipeline": "phase3-memory"`, the extraction
model, and a "Memory formation" section with extraction and consolidation
statistics. The aggregate metrics also include diagnostics for current-state
accuracy, multi-source context coverage, and memory source coverage.

### Phase 3 limitations

- Extraction quality depends on the local model. Small models sometimes
  mislabel a reference as an update, or attach a fact to the wrong subject.
  Grounding and the consolidation rules catch some of these errors, but not
  all.
- Memory relevance is lexical (there are no memory embeddings), so a
  paraphrased question relies on semantic support from the source chunks.
- `valid_from` is the message timestamp; dates stated inside messages are
  not parsed.
- Detecting historical questions uses a small English keyword list
  (`memory/query.py`).
