# V1 conversational memory extraction — audit, change, evaluation

Scope: extraction and memory maintenance only. Retrieval (graph-hybrid) unchanged. Nothing committed.

## 1. Audit of the existing pipeline

| Stage | Code | Finding |
|---|---|---|
| Import entry | `webapi/extract.py` `ingest_imported`, `PassThroughAtomizer` | Default `extract="messages"` stores **one claim per raw message** — chunks are the memories. Every prior LoCoMo/LME evaluation used this. |
| Pipeline | `v2/pipeline.py` `V2IngestionPipeline.ingest` | Windowed LLM extraction; deletes the conversation's semantic rows then re-inserts → idempotent re-ingest. No token accounting. |
| Windows | `v2/windowing.py` `build_windows` | 4 target + 4 context messages; prior claims passed as identity context only. |
| Prompt | `v2/prompt.py` `SYSTEM_PROMPT`, `output_schema` | "Extract every proposition… do not decide what is important"; `claim_type` left empty. No durability selection, no memory type. |
| Model call | `v2/atomizer.py` `SemanticAtomizer`, `OllamaChatClient` | JSON-schema constrained, qwen2.5:14b, temp 0, seed 0, num_ctx 8192, num_predict 2048. |
| Validation | `v2/validation.py` `validate_atomization`, `classify_support` | Strict lexical support check (SOURCE / IDENTITY_CONTEXT, else `UNSUPPORTED_CLAIM` rejected); ID, relation and temporal checks. Good anti-hallucination guard. |
| Storage | `v2/store.py` `SemanticStore` | `v2_claims` (with `metadata_json`), `v2_relations`, `v2_atomization_log` (raw output + issues). Raw messages kept unchanged as evidence. |
| Maintenance | `v2/consolidation.py` `consolidate`, `v2/identity.py` `claim_attribute` | Restatement dedup keeps all sources; supersession via explicit UPDATES/CONTRADICTS or identity-safe slot (subject + attribute, attribute = `metadata["attribute"]` else **`claim_type`**); statuses current/superseded/historical/contested/future/conditional; never deletes. |
| Load | `v2/memory.py` `MemoryRetriever` | Builds memories at load time from claims + relations. |
| Snapshot store | `v2/memory_store.py` | Not on the retrieval path. |
| Workspace | `webapi/workspace.py` `import_text` | Accepted `messages`/`v2` only. |

Prior evidence: `benchmarks/experiments/extraction_audit` (gold set, 113 assertions); phase-7 calibration on LongMemEval with 14b: 81% valid JSON, ~202 s/window, 21% rejected, output-length cutoffs.

**No fundamental blocker.** The V2 path already has provenance, validation, versioned conservative updates and idempotence. Gaps: (a) raw-message default, (b) no durability selection, (c) no memory type — and `claim_type` cannot carry one safely, because it is the slot attribute: two unrelated preferences would supersede each other, (d) no contract-level record view, (e) token cost invisible.

## 2. External designs (read from source at pinned commits)

| | mem0 OSS (26cdefb) | supermemory (0f2b24b) | Letta (archive 56ba9c2) | Ours before | V1 change |
|---|---|---|---|---|---|
| Extraction | 1 LLM call per `add()`, atomic 15–80-word facts from both roles, `attributed_to` | Engine not in repo; client sends whole conversation as one document | None; agent edits memory blocks via tool calls | Every proposition, or raw messages | Durable-only memories per message; `no_durable_assertion` allowed |
| Provenance | No source message IDs | Schema only (hosted) | None | Message + claim IDs | Kept; exposed as `sources[]` |
| Dedup | md5 exact + LLM "skip" instruction | Claimed (hosted) | None | Restatement dedup, sources merged | Unchanged |
| Updates | ADD-only now (legacy ADD/UPDATE/DELETE prompt not called) | Schema has version/isLatest/parent/updates/extends (logic hosted, unverified) | Overwrite block; history off by default | Explicit relations + slot, never delete | Unchanged; memory type kept out of slot |
| Speaker safety | Attribution field | `isInference` flag (schema) | n/a | Lexical validator | Prompt rule: assistant suggestions never user facts |
| Malformed output | Parse fallback → empty list, no retry | — | — | Validated, logged | Unchanged |
| Time | Observation date defaults to today | `forgetAfter` (schema) | — | Message timestamps + temporal fields | Unchanged |

Takeaway: nothing to adopt wholesale. The useful idea from mem0 (select durable atomic facts, attribute speaker) is a prompt change on top of our stricter validation/provenance/versioning, which none of the three implement in open source.

## 3. V1 memory contract (`v2/memory_records.py` `MemoryRecord`)

| Field | Source |
|---|---|
| `memory_id` | consolidation ID `mem:<message_id>:c<n>` — stable for the same extraction output |
| `content` | atomic memory text (validated against source) |
| `memory_type` | `preference, personal_fact, goal, project, event, decision, relationship, procedural` — stored in `Claim.metadata["memory_type"]`, not `claim_type` |
| `subject`, `entities` | consolidation |
| `sources[]` | `message_id, claim_id, speaker, role, timestamp` for every message that stated it |
| `assertion` | `explicit` or `identity_resolved` (pronoun resolved from context). V1 produces no inferred memories. |
| `temporal`, `event_time`, `valid_from`, `valid_until` | temporal fields |
| `polarity` | positive / negative |
| `status`, `superseded_by`, `links` | lifecycle from consolidation |

Enforced by code: a lexical source check (content stems must be in the source; see §8 for its limits), source and speaker recorded on each memory, zero memories per message allowed, updates only via explicit UPDATES/CONTRADICTS or identity-safe slot match, no memory version deleted by consolidation.

Prompt instruction only, not enforced by code: no assistant suggestions as user facts, durability selection, keeping hypotheticals and negation intact.

## 4. Implementation (uncommitted)

- `v2/memory_prompt.py` (new): `MEMORY_SYSTEM_PROMPT` (11 rules), `build_memory_prompt`, `memory_output_schema` (requires `memory_type` enum), two few-shot examples that pass the validator cleanly. (Correction: one was not invented. It was derived from LoCoMo conv-30 and has since been replaced; see §8.)
- `v2/atomizer.py`: `ExtractionProtocol` (`ATOMIZATION_PROTOCOL` default unchanged, `MEMORY_PROTOCOL`); `memory_atomizer()`; `OllamaChatClient.last_usage` (prompt/eval tokens, done_reason); `AtomizationResult.usage`.
- `v2/models.py`: `MEMORY_TYPES`.
- `v2/validation.py`: validates `memory_type` into metadata; unknown value → `INVALID_MEMORY_TYPE` flagged and dropped, claim kept.
- `v2/pipeline.py`: `V2IngestReport.model_calls / prompt_tokens / output_tokens`.
- `webapi/extract.py`: `extract="memories"`; unknown modes raise `ValueError`.
- `webapi/workspace.py`: `import_text` accepts `memories` (label `memory_extraction`).
- `v2/memory_records.py` (new): read-only contract projection.
- Storage schema, consolidation, retrieval: **unchanged**. Default import mode: **unchanged** (`messages`).

## 5. Tests

`tests/unit/test_memory_extraction_v1.py` — 11 tests, all pass: multiple typed memories per message; none per message; two preferences don't supersede; repeated fact → one memory, two sources; explicit UPDATES supersedes while an unlinked change keeps both current; "used to" → historical; speaker attribution and provenance (assistant suggestion not stored); malformed / invalid-type / unsupported output handled; reprocessing gives identical IDs and no duplicates; unknown mode refused; graph-hybrid `evaluate_question` works with original turns as evidence.

`tests/unit/test_v2_pipeline.py` expectation updated for the three new report fields.

Full suite: **1,132 passed, 1 failed** — `test_webapi.py::test_db_sample_uses_import_db_when_served_sqlite_has_no_claims`, which fails without these changes too. (Correction: that last statement could not be verified later, because this code has no committed baseline. See §9 for current test results.)

## 6. Before/after evaluation

> **Old prompt.** Every result in this section was produced with the earlier prompt, which contained a LoCoMo-derived example. None was rerun with the current prompt. The figures are kept as a historical record. The only result from the current prompt is the smoke test in §9.

Fixed examples: 6 synthetic conversations (104 messages, 113 gold assertions, 13 update pairs) + LoCoMo conv-30 session 1 (28 turns, no gold — inspected manually). Same windows, validator, consolidation, model and settings for both LLM arms; only the instruction differs. One call per window, 0 failed windows. Reproduce: `python -m benchmarks.experiments.memory_v1.run` then `python -m benchmarks.experiments.memory_v1.score` (outputs in `runs/memory_v1/`).

| Synthetic (6 convs) | messages (default) | v2 (current atomizer) | **memories (V1)** |
|---|---|---|---|
| Memories stored | 104 | 102 | 82 |
| Gold recall | 100% (trivial — raw text) | **90.3%** (102/113) | 85.0% (96/113) |
| Share of memories matching gold (precision proxy) | 78.8% | 91.2% | **95.1%** |
| Extra matches per gold (duplicate proxy) | 45 | 14 | **12** |
| Near-duplicate pairs (Jaccard ≥ 0.8) | 0 | 0 | 1 |
| Provenance points at originating message | 100% | 100% | 100% |
| Memories sourced only from assistant turns | 52 | 9 | **1** |
| Update pairs handled (of 13) | 1 | 8 | **9** |
| … stale old value still current | — | 4 | **1** |
| … new value missing / not current | — | 0 / 1 | 1 / 2 |
| Prompt / output tokens | 0 | 60,728 / 12,541 | 56,209 / 11,540 |
| Latency (local 14b) | ~0 | 1,607 s | 1,605 s |
| Rejected `UNSUPPORTED_CLAIM` | — | 9 | 16 |

Memory types assigned: personal_fact 42, event 21, goal 6, project 6, preference 4, procedural/relationship/decision 1 each.

| LoCoMo conv-30 S1 (28 turns) | v2 | memories (V1) |
|---|---|---|
| Memories | 24 | 15 |
| Tokens (prompt / output) | 17,867 / 4,298 | 16,526 / 3,204 |
| Latency | 591 s | 551 s |
| Memories naming the wrong speaker | 0 | 0 |

### Representative examples

Kept as memories (LoCoMo, V1; paraphrased, dataset text removed): a job loss (event), a business being started (project), a past hobby (personal_fact, historical), a stated favourite (preference).

Dropped by V1 but stored by v2 (LoCoMo; paraphrased): general opinions about an activity, compliments about a photo, and one speaker's reaction to another's plans.

Synthetic, V1: `Sam has a dog` + `The dog is a three-year-old border collie called Pepper` (two atomic memories from one message, m07); assistant advice such as `Squealing often means worn pads` (stored by v2) is not stored.

Remaining V1 noise (LoCoMo; paraphrased): a speculative remark about how an upcoming performance would go (kept, marked `future`).

### Why recall dropped (diagnosed from `v2_atomization_log`, no extra calls)

Of the 10 gold facts v2 captured and V1 missed, one (W06) is assistant advice, correctly excluded. One (W21) was never emitted. **The rest were emitted by the model and rejected by the strict lexical validator** because of light paraphrase, e.g.:

- `The user drinks cortado with oat milk regularly.` → unsupported tokens `drink`, `regularly`
- `Marco Silva is the team lead for Project Atlas.` → `project`, `atla` (not in that message)
- `The cake for Omar's birthday will be nut-free.` → `fre`

So the loss is a prompt-adherence / validator interaction, not missing extraction.

### Conclusion

Measured: V1 stores fewer, more precise, typed memories (95.1% vs 91.2% gold-matched; 1 vs 9 assistant-sourced), with slightly better update handling (9 vs 8 handled; 1 vs 4 stale-current) and ~7–8% fewer tokens at the same latency. It **loses 5.3 points of gold recall** (85.0% vs 90.3%). It is not a strict improvement over the current atomizer; it is a clear improvement over the default raw-message mode on precision, assistant contamination and updates. Sample is small (113 gold, one LoCoMo session); differences of 1–2 items are within noise.

## 7. Remaining V1 blockers

> **Superseded list.** This was written before the heuristic audit. The opt-in release blockers are the three in §8. Unresolved items are tracked in §10.

1. **Validator vs paraphrase** — 16 rejections, the main recall loss. Needs either a vetted validator relaxation (e.g. stem/synonym tolerance for non-entity words) or a stricter prompt, measured on the same gold.
2. **Default import mode is still `messages`.** Switching it changes every downstream evaluation; not done.
3. **Mixed subject naming** ("Sam" vs "the user") weakens exact restatement dedup and slot matching.
4. **Unlinked changes stay double-current**; supersession relies on the model emitting UPDATES.
5. **Temporal rule marks past-tense moves historical** (`Sam relocated to Braga last month` → historical although still true).
6. **Latency**: ~50–60 s per window on local 14b; full LoCoMo would take hours per conversation.
7. **No extraction cache**: reprocessing is idempotent in output but re-calls the model.
8. **Workspace import errors when zero memories are extracted** from a conversation.
9. **`observation_date` accepts ISO only.**
10. Speculative future statements (`will impress…`) still pass as memories (status `future`).

## 8. Corrections after the heuristic audit

Blocker fixes from `reports/memory_extraction_heuristic_audit.md`. **The results in §6 were produced with the earlier prompt and have not been rerun**, so they describe a prompt that no longer exists. They also never covered held-out data: the earlier prompt contained an example derived from LoCoMo conv-30, the same conversation as the LoCoMo sample in §6.

- **Prompt:** the conv-30 example and a conv-26 speaker name were replaced with fictional content. The extraction rules are unchanged.
- **Re-import:** when memories are already stored, extraction and validation finish before anything is written, and raw and semantic rows are then replaced in one transaction. An exception or interruption changes nothing. A run with a failed window, or with zero claims because output was rejected or missing, keeps previously stored rows (`V2IngestReport.kept_existing`), and `Workspace.import_text` reports it as an error. A first import has nothing to protect and still writes windows as they complete, so partial results survive a failure.
- **Validation:** a claim that cannot be compared is rejected as `UNVERIFIABLE_CLAIM` rather than accepted. This covers a source with no comparable tokens, a claim with none beyond frame words, and any claim word lost by tokenization such as non-Latin text. Entities, qualifiers and temporal expressions with the same problem are dropped.
- **Scope:** V1 extraction is English-only and `extract="memories"` remains opt-in.

Known limits that remain: the source check is lexical, not semantic. It rejects faithful paraphrases and does not detect changed negation, speaker or certainty. Claims the validator rejects from a run that still produces claims do not block replacement, so a re-import can store fewer memories. A run where the model reports nothing durable in every message counts as successful and replaces stored rows, so a model wrongly returning that for everything would remove them. Temporal status logic and correction linking are unchanged. Item 1 of §7 should not be read as a recommendation: the audit found that stem or synonym tolerance would not address most rejections.

## 9. Release candidate status and smoke test

### Implementation status

- The three opt-in blockers are fixed: prompt examples, re-import safety and unverifiable claims (§8).
- Prompt examples are fictional.
- Replacing stored memories on re-import is one transaction.
- Unverifiable claims are rejected.
- The default extraction mode remains `messages`, and `extract="memories"` is opt-in.
- Graph-hybrid retrieval and ranking are unchanged.

Test results at the time of the smoke test. Later corrections changed source and tests, so current results are in `memory_extraction_v1_release_manifest.md` §8:

| Suite | Result |
|---|---|
| `tests/unit/test_memory_v1_blockers.py` | 35 passed |
| Full `tests/` | 1,169 passed, 1 failed |
| Phase-7 pilot tests | 38 passed, 4 failed |

The full-suite failure is `test_webapi.py::test_db_sample_uses_import_db_when_served_sqlite_has_no_claims`. It failed before the blocker fixes and comes from how `open_workspace` treats a dedicated database, not from memory extraction. The phase-7 failures check frozen hashes and recorded runs, and they are not part of the normal suite.

### Smoke test: current prompt, real model

**This is a small smoke test, not an accuracy benchmark or evidence of generalization.** It is one fictional six-message conversation, one run, with no retry or tuning.

| Item | Value |
|---|---|
| Model | `qwen2.5:14b` through local Ollama (digest not recorded) |
| Configuration | temperature 0, seed 0, `num_ctx` 8192, `num_predict` 2048, 4 targets and 4 context messages per window, 20 prior claims |
| Scope | occupation and preference, small talk, unconfirmed assistant suggestion, correction, a message with nothing durable |
| Model calls | 2 (0 failed windows) |
| Tokens | 4,242 input, 729 output |
| Runtime | 69.3 s (43.0 s and 26.3 s) |

Conversation:

| ID | Role | Text |
|---|---|---|
| s1 | user | Hi, I'm Marisol. I work as a marine surveyor in Tromso, and I strongly prefer written updates over phone calls. |
| s2 | assistant | Nice to meet you, Marisol. I'll keep that in mind. |
| s3 | user | Right now I'm just waiting for my kettle to boil, it's taking ages this morning. |
| s4 | assistant | You might enjoy listening to podcasts while you work on surveys. |
| s5 | user | Actually, I should correct something: I don't work in Tromso anymore, I work as a marine surveyor in Bergen now. |
| s6 | user | Okay, thanks for that. |

Stored memories:

| Memory | Status | Source |
|---|---|---|
| The user is Marisol. | current | s1 |
| Marisol works as a marine surveyor in Tromso. | superseded | s1 |
| Marisol strongly prefers written updates over phone calls. | current | s1 |
| Marisol is waiting for her kettle to boil. | current | s3 |
| Marisol does not work in Tromso anymore. | current | s5 |
| Marisol works as a marine surveyor in Bergen. | current | s5 |

**Result: 8 of 9 checks passed.**

Failed check:

1. **Temporary small talk was stored as a current memory:** `Marisol is waiting for her kettle to boil.`

Other problems found in the run:

2. **Unsupported pronoun:** the model added "her", although the speaker's gender was never stated. The validator does not catch this because pronouns are stopwords.
3. **Bergen is not linked to Tromso:** the Tromso memory was superseded by the negation memory (`Marisol does not work in Tromso anymore.`). The model's second `UPDATES` relation, from the Bergen memory, created no link because its target was already superseded. Bergen is current, but its record has no link to the memory it replaces.

Passed: structured extraction, several memories from one message, no storage of the assistant suggestion, correction handling under existing update rules, source and speaker attribution, accurate reporting of empty extraction (`no_durable_assertion: 3`), usage accounting, and database integrity (`integrity_check` ok, no foreign key violations).

The temporary database and script were deleted after the run, so this section is the record of it.

## 10. Known limitations of the release candidate

None of these is solved in V1.

- **Durability** is not enforced by code; temporary events and small talk may be stored as current memories (seen in §9).
- **Assistant-suggestion filtering** is prompt-driven; validation does not check it.
- **Source validation** is word-stem based and cannot guarantee semantic entailment. It rejects faithful paraphrases and accepts flipped negation, changed speakers and stripped hypotheticals.
- **Unsupported inferred details**, including gendered pronouns, may pass validation (seen in §9).
- **Temporal status** comes from keywords and can be wrong.
- **Subject normalization** is unresolved: "Sam" and "the user" remain separate memories.
- **Older correction linking** is unresolved: a correction can only link to one of the last 20 claims, and a correction split into several memories may link only one of them (seen in §9).
- **Partial extraction rejection** can leave fewer stored memories after a re-import. A clean run where the model reports nothing durable in every message also replaces stored memories.
- **Scope** is English-only.
- **Headerless transcript parsing** is best-effort and can misattribute speakers. With only single blank lines between paragraphs, an uppercase user paragraph can merge into the assistant turn, so its facts carry the assistant role. Use explicit speaker headers or JSON when speaker attribution matters.
- **Name-specific evaluator logic:** the heuristic evaluator in `webapi/evidence_eval.py` is benchmark-specific, with the LoCoMo conv-26 speakers Caroline and Melanie hardcoded, so its results can change with the name. It is diagnostic-only and not part of normal API answer generation.
- **The mode remains opt-in.** The web UI and CLI do not offer it; it is reached through `POST /import` or `ingest_imported`.
