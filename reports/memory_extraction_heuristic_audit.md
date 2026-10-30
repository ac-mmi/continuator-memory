# Memory extraction heuristic audit (`extract="memories"`)

Date: 2026-10-30. Audit only: no production code, prompts, tests, configuration or evaluation scripts were changed, and nothing was committed.

> **Point-in-time record.** This describes the code before the blocker fixes, so F1, F5 and the fail-open part of F3 are now fixed, and some line numbers have moved. See `reports/memory_extraction_v1.md` §8-§10 for current status.

## 1. Executive verdict

**Mostly general-purpose, with material risks.**

- **The mechanism is not fitted to LoCoMo.** The production extraction path has no dataset branch, conversation ID, question category or expected answer. Windows, provenance, typed metadata and conservative versioning are general designs.
- **One piece of benchmark content is in production.** A few-shot example in the memory prompt is built from LoCoMo conv-30, the same conversation used as its only real-world evaluation. The module's docstring says the opposite.
- **The semantic guarantees are weaker than reported.** The source check is a bag of English word stems. It rejects many faithful memories and accepts negation flips, turned-around suggestions, stripped hypotheticals and any non-Latin text. Speaker, hypothetical and durability rules are enforced only by the prompt.
- **Current or past status comes from keyword matching**, which misclassifies common cases.

**Decision: keep the mode opt-in.** Three small blockers apply even to an opt-in release (§8). No generalization claim is supported by the evidence so far.

### How this audit gathered evidence

| Method | What it can show |
|---|---|
| Code reading along the real execution path | What the code does |
| Deterministic probes: real validator, temporal and consolidation code with hand-written model output, on temporary databases | What the rules **would** do with that output, not how often the model produces it |
| Offline reanalysis of saved `runs/memory_v1` logs and databases, with no model calls | What happened in the existing run |
| Existing tests, re-run | `test_memory_extraction_v1.py`: 11 passed. `tests/unit`: 1,132 passed, 1 failed (`test_webapi.py::test_db_sample_uses_import_db_when_served_sqlite_has_no_claims`). I did not check the v1 report's statement that this failure predates the change. |

## 2. Scope and files inspected

| Area | Files (under `src/continuator_memory/` unless a path is given) |
|---|---|
| Extraction | `v2/memory_prompt.py`, `v2/prompt.py`, `v2/atomizer.py`, `v2/windowing.py`, `v2/pipeline.py`, `webapi/extract.py` |
| Validation | `v2/validation.py`, `memory/text.py`, `memory/models.py` (`normalize_value`) |
| Storage and record view | `v2/store.py`, `v2/models.py`, `v2/memory.py`, `v2/memory_records.py` |
| Maintenance | `v2/consolidation.py`, `v2/identity.py`, `v2/temporal.py` |
| Retrieval | `v2/memory_retrieval.py` and `v2/graph_index.py` (relevant parts), `v2/turn_context.py` (docstring), `webapi/locomo_report.py` (retriever and graph setup) |
| Import surfaces | `webapi/workspace.py`, `webapi/server.py`, `webapi/transcript.py`, `webapi/locomo.py`, `cli.py`, `web/src/components/source/ImportConversation.tsx`, `web/src/data/service.ts` |
| Tests | `tests/unit/test_memory_extraction_v1.py`, test names and selected cases in `test_v2_validation.py`, a keyword search of `test_v2_temporal.py` |
| Evaluation | `benchmarks/experiments/memory_v1/{run,score}.py`, `extraction_audit/audit.py` (`_value_ok`), `runs/memory_v1/*`, `reports/memory_extraction_v1.md`, `locomo10.json`, `benchmarks/synthetic/phase2_benchmark.json` |

The manual labels in §3 (F2) and §6 are from one reviewer, me.

## 3. Findings by severity

**No critical finding is confirmed.** I found no real-model run showing stored fabricated data. That is an absence of evidence from a small sample, not proof of safety.

### High

**F1. A production few-shot example is LoCoMo conv-30 content (confirmed).**
- `v2/memory_prompt.py:133-147` used that conversation's two speakers, a named employer and two of their businesses. All appear in conv-30 of `locomo10.json`, and two of the terms appear in session 1 (D1:3, D1:4). (Specifics removed from this record.)
- Line 35 used the conv-26 speaker as the name-format example.
- The docstring at line 7 says "The examples use invented text, not benchmark data". The wording is rearranged, but that statement is wrong.
- Effect: the 28-turn conv-30 result is **not held out** and cannot be cited as generalization evidence. How much this changes model behaviour outside LoCoMo is unmeasured.
- The first example (Priya, Lyon, Saint-Luc) has no matches in either dataset.
- Related, lower severity: the v2 comparison prompt uses "Leah" (`v2/prompt.py:31`), who also appears in the synthetic `spain-corrections` conversation.

**F2. The source check rejects faithful memories and causes most of the measured recall loss (confirmed).**
- `v2/validation.py:167` requires every content stem in the claim to appear in the source, apart from `FRAME_TOKENS` (lines 64-69).
- Of 17 gold facts the memories arm missed, **13 were emitted by the model and rejected**.
- Rejection patterns from conv-30 session 1, reworded with the dataset text removed (the informal verbs are the point):

| Claim | Source word | Unsupported stem |
|---|---|---|
| `<Speaker> wants to start <business>…` | "I wanna start <business>" | `want` |
| `<Speaker> is going to start a business.` | "I'm gonna take a shot" | `going` |
| `<Speaker> wants to share a passion for <hobby>…` | "passionate about <hobby>" | `passion`, `want` |

- That session had 12 `UNSUPPORTED_CLAIM` rejections against 15 stored memories. An informal register is therefore a realistic failure trigger.
- The 16 synthetic rejections, labelled manually:

| Label | Count | Examples |
|---|---|---|
| Faithful, source wording only | 4 | "I'd like to visit" → `wants to visit`; "no nuts" → `nut-free` |
| Faithful, enriched from earlier context so the memory stands alone | 9 | `job interview`, `vegetable garden`, `car repairs`, `Project Atlas` |
| Added a mild inference | 2 | `Alex's home`; `to Porto` |
| Dropping it was desirable | 1 | Assistant suggestion, rejected only because of incidental wording |

- None of the 16 was a clearly invented fact, so this sample does not show the guard's benefit. It does not show the guard is useless either.
- **The v1 report's diagnosis is incomplete.** It calls these "light paraphrase" and suggests stem or synonym tolerance. In this sample, 9 of 16 are context enrichment, which synonym lists would not fix. Rule 3 of the prompt ("self-contained sentence") pulls against the validator's evidence model.
- Extending the word lists would add another heuristic and leave F3 unsolved.

**F3. The source check accepts unfaithful claims, and fails open on non-Latin text (confirmed by probe; real-world frequency unknown).** `claim` is checked as an unordered set of stems, with negations and "user" exempt:

| Source | Hand-written claim | Validator result |
|---|---|---|
| "I like coffee." | `The user does not like coffee.` | SOURCE |
| "I don't like coffee." | `The user likes coffee.` | SOURCE |
| Assistant: "You should try yoga." | `The user tries yoga.` | SOURCE |
| Assistant: "Do you live in Paris?" | `The user lives in Paris.` | SOURCE |
| Assistant: "I am allergic to penicillin." | `The user is allergic to penicillin.` | SOURCE |
| "I might have diabetes." | `The user has diabetes.` | SOURCE |
| "If I get the job, I will move to Berlin." | `The user will move to Berlin.` | SOURCE |
| "Anna: Bob hired me." | `Anna hired Bob.` | SOURCE |
| "I paint landscapes." (another named speaker in context) | `<Name> paints landscapes.` | IDENTITY_CONTEXT |
| Chinese "I like tea" | Chinese "The user lives in Paris" | SOURCE |
| Hindi "I like tea" | Hindi "The user lives in Paris" | SOURCE |

- **Non-Latin text passes unconditionally.** `normalize_value` (`memory/models.py:81`) strips everything outside `[0-9a-z]`, so no tokens remain and `missing` is empty at `validation.py:167-169`.
- **"I" counts as a referring expression** (`PRONOUNS`, lines 70-72, used at lines 154 and 186). That makes the identity gate almost always open for first-person messages, which enables the wrong-speaker case above.
- `polarity` is not compared with the text, and validation never looks at the message role.
- `reports/memory_extraction_v1.md:53` lists "no assistant suggestions as user facts" under "Rules enforced". In fact, only prompt rule 6 addresses it. `memory_records.py:15` says "every word comes from the source message" for `explicit`, which is also overstated.

**F4. Current or past status is inferred from single words (confirmed by probe and by the real run).** `v2/temporal.py:54-71,163-184`.
- **The same session gives inconsistent results.** one speaker's job loss dated "yesterday" → historical, while another's dated "this month" → current (LoCoMo; paraphrased). The facts have the same structure and the same present consequence.
- **Wrong in the synthetic run:** `Sam relocated to Braga last month` → historical, because `relocated` is not on the `TRANSITION` list. `moved` would be classed as a transition.
- **Probe results:**

| Claim | Class |
|---|---|
| `<Name> opened an online shop last month.` | past |
| `The user married Tom three years ago.` | past |
| `Priya had a baby.` | past |
| `The user is the former mayor's aide.` | past |
| `Will lives in Boston.` | future |
| `May works with the user.` | conditional |
| `The user would like a window seat.` | conditional |
| `The user could not attend.` | conditional |
| `The user will never eat meat.` | future |

- **Retrieval effect:** non-current statuses take a 0.05 penalty on questions without a matching temporal cue (`memory_retrieval.py:64,356`). The code describes this as reordering only near-ties, and the effect on answer quality is unmeasured.
- Plain English only, by construction.

**F5. Re-importing during a model failure wipes the stored memories (confirmed by probe).** `v2/pipeline.py:80-82` deletes semantic rows before any model call.
- In the probe, a first import stored one memory. A re-import with the model down left **zero memories** and `failed_windows=1`.
- `workspace.py:221-224` then raises an error saying "The previous workspace was left unchanged". That is true of in-memory state, but the SQLite rows are gone, so a restart would lose them.
- A partly failed re-import similarly replaces good memories with fewer. There is no retry.
- Raw messages survive, so this is recoverable by re-extraction.

### Medium

**F6. Updates depend on the model naming a recent claim (confirmed).**
- This mode has no slot rule in practice: its schema has no `claim_type`, and 0 of 97 stored claims in the run carried one. Supersession needs an explicit `UPDATES` relation.
- The target must be among the last 20 claims (`atomizer.py:31`, `pipeline.py:86`). In a probe, a correction 24 claims later was rejected with `UNAVAILABLE_SOURCE`, and Porto and Lisbon both stayed current. The real run had the same issue code once in `spain-corrections`.
- A correction containing `their` is refused as an unresolved reference (`consolidation.py:122,136-137,503-504`). In a probe, `The user quit their office` with a valid `UPDATES` relation left both memories current.
- The conservative result, keeping both, is justified and is not data loss. It does mean stale facts remain current.

**F7. Deduplication depends on exact wording and subject naming (confirmed).** `restatement_key` (`consolidation.py:140-156`) compares normalized text.
- `Sam has a dog.` and `The user has a dog.` stay separate, with different subjects.
- The prompt permits both forms. In the synthetic run, 28 of 82 memories open with "the user". The other 54 open with a name or some other subject, so that number is an upper bound on name use.
- Any claim containing `it`, `their` and similar words never merges, by design.
- Duplicates are the safer failure, but they accumulate.

**F8. Speaker semantics assume two roles (confirmed from code).**
- Prompt rule 6 discusses "the assistant", while the second few-shot stores personal facts from a named speaker with role `assistant`. The LoCoMo adapter maps the second human to that role (`webapi/locomo.py:165-168`).
- A transcript with two unlabelled humans has no way to distinguish them. In a probe, two speakers both labelled `user` produced `The user is vegan` and `The user is not vegan`, both current.
- `split_speaker` (`graph_index.py:45,61-66`) reports "Note", "Update" and "TODO" as speakers, and falls back to the role for `anna:` and Cyrillic names. This value is what `SourceRef.speaker` records.

**F9. Key tests encode scripted output rather than behaviour (confirmed).** The model is replaced by a script in every test in `test_memory_extraction_v1.py`.
- `:166` asserts no "watercolour" memory exists, but the script emitted none, so it cannot fail for the behaviour it appears to test.
- `:93-97` has the same limitation for "no durable information".
- No test asserts acceptance of a faithful paraphrase, while `test_v2_validation.py` has several rejection cases.
- `:120-140` records keeping both versions on unlinked changes as expected behaviour, which is a legitimate design record.

**F10. Memory does not span imports (confirmed from code).** The prompt defines a memory as useful "in a later conversation". However:
- Consolidation is per conversation (`consolidation.py:345-346`).
- Every transcript import uses the fixed ID `imported` (`transcript.py:11,72`), so each import replaces the last.
- Retrieval can search several conversation IDs, but nothing here merges or updates facts across them.

Whether V1 needs cross-conversation memory is a scope decision that I cannot determine from the code.

**F11. The transcript parser contains phrases from specific transcripts (confirmed).** `transcript.py:33-41` assigned the assistant role using a list that included health and daily-routine phrases from a specific conversation, used at line 166. Those phrases have since been removed. This is not LoCoMo, but it is overfitting upstream of speaker attribution. It only affects text that lacks explicit role headers, and I did not exhaustively trace that path.

**F12. Cost is only partly observable, and the mode is hard to reach (confirmed).** See §7.

### Low

| ID | Finding | Location |
|---|---|---|
| F13 | Validator accepts `claim_type` if the model returns it. In a probe it turned two preferences into a supersession. 0 of 97 claims in the run had one, and the schema omits it. | `validation.py:420-425` |
| F14 | Speculation is stored: a remark about how a future performance would go (D1:26), status future, type `personal_fact`. Durability and type accuracy are not scored. | `score.py`, `runs/memory_v1` |
| F15 | Same-session context relies on equal timestamps, an assumption stated to match the LoCoMo adapter. It only applies to the optional context-weight feature. | `turn_context.py:8-9` |
| F16 | `CHANGE_CUE` is a word list, but only flags and never blocks. | `validation.py:81-85,556-558` |

## 4. Rule classification

Categories: **A** general and justified, **B** potentially brittle, **C** benchmark-specific or overfit, **D** uncertain. Action: retain, investigate or redesign (eventually).

| Rule | Location | What it does | Cat. | Evidence | Realistic failure | Sev. | Action |
|---|---|---|---|---|---|---|---|
| Few-shot 2 | `memory_prompt.py:133-147` | Teaches format using conv-30 entities | C | Terms found in `locomo10.json` | Evaluation not held out | High | Replace |
| conv-26 speaker name as example | `memory_prompt.py:35` | Name-format illustration | C | conv-26 speaker | Minimal by itself | Low | Replace |
| Few-shot 1 | `memory_prompt.py:112-131` | Teaches update and suggestion handling | A | No dataset matches | Few examples for wide inputs | Low | Retain |
| Stem subset support | `validation.py:167` | All stems must be in source | B | 13 of 17 misses; F3 table | "I wanna…" rejected; negation flip accepted | High | Redesign |
| `FRAME_TOKENS` | `validation.py:64-69` | Exempts user, negations, causal words | B | Flip probes | `not` added or removed | High | Redesign |
| Non-Latin pass-through | `memory/models.py:81`, `validation.py:167` | Empty token set means supported | B | Probes | Any Chinese or Hindi claim stored | High | Fail closed |
| `PRONOUNS` includes I/me | `validation.py:70-72,154,186` | Opens reference resolution | B | Wrong-speaker probe | Wrong speaker accepted | High | Investigate |
| `MAX_REFERENT_TOKENS = 2` | `validation.py:59,186` | Limits context resolution | B | No stated basis; `Project Atlas` was rejected at 2 tokens for lack of a pronoun | Long referents refused | Med | Investigate |
| English suffix stemmer | `validation.py:129-134` | Crude roots (`atla`, `fre`) | B | `bought`/`buy` and `kids`/`children` rejected | Irregular forms | Med | Redesign |
| Source message ID checks | `validation.py:329-337,378-386` | Claims must cite their own target | A | Tests exist | None observed | n/a | Retain |
| Exact-substring relation evidence | `validation.py:552` | Evidence must be verbatim | A | Tested | Model reformats a quote | Low | Retain |
| Causal connective check | `validation.py:75-79,214-238` | Requires a cue joining clauses | B | Fixed cue list | Causality without a listed cue | Low | Retain |
| `CHANGE_CUE` | `validation.py:81-85` | Flags only | A | Non-blocking | Noise only | Low | Retain |
| `PAST_ANCHOR`, `PAST_STATIVE` → historical | `temporal.py:61-64,180-181` | One word gives historical | B | Two job losses; Braga | Lasting event marked past | High | Redesign |
| `TRANSITION` list | `temporal.py:66-68` | Fixed change verbs | B | `moved` vs `relocated` | Unlisted verb | High | Redesign |
| `CONDITIONAL`, `FUTURE` lists | `temporal.py:54-55` | Modal word gives status | B | `would like`, `could not` | Lasting preference marked conditional | Med | Redesign |
| May/Will capital handling | `temporal.py:96` | Removes mid-sentence capitals only | B | `Will lives in Boston` → future | Name at sentence start | Med | Investigate |
| Observation time ≠ event time | `temporal.py:3-7` | Keeps timestamps apart from event dates | A | Sound principle | None observed | n/a | Retain |
| Delete before extract | `pipeline.py:82` | Clears rows first | B | Probe E | Model outage on re-import | High | Fix |
| `DEFAULT_PRIOR_LIMIT = 20` | `atomizer.py:31` | Bounds prompt context | B (real constraint, arbitrary value) | Probe F | Correction of an older fact | Med | Investigate |
| Windows 4 targets, 4 context | `windowing.py:14-15` | Bounds prompt size | A (limits context reach) | Rejections citing earlier topics | Topic beyond window | Med | Retain, measure |
| `num_ctx 8192`, `num_predict 2048` | `atomizer.py:29-30` | Inherited from earlier experiments | A | Comment states origin | Long messages truncate | Med | Investigate |
| `AMBIGUOUS_PRONOUNS` blocks merges and updates | `consolidation.py:122,136` | Avoids unsafe merges | A, conservative | Probe G | Possessive `their` blocks an update | Med | Retain, investigate |
| Exact restatement key | `consolidation.py:140-156` | Merges only equal wording | A, conservative | Probe H | Sam vs the user | Med | Retain now |
| Explicit-relation updates only | `consolidation.py:341-356` | No guessed supersession | A | Test | Unlinked change stays current | Med | Retain |
| Never delete old memories | `consolidation.py:616-622` | Keeps history | A | Code | None observed | n/a | Retain |
| Memory type as metadata | `v2/models.py:27-31`, `validation.py:426-433` | Stops type becoming a slot | A | Test and code | See F13 | Low | Retain |
| Eight memory types | `v2/models.py:30-31` | Fixed enumeration | D | Accuracy unscored | Miscategorized facts | Low | Investigate |
| Prompt durability list | `memory_prompt.py:37-39` | Skip small talk, reactions and so on | D | Removed noise, also left speculation | Temporary opinion kept | Med | Investigate |
| Assistant rule 6 | `memory_prompt.py:41-43` | Prompt only | A in principle, B in enforcement | Probes; 1 of 82 assistant-sourced in real run | Model turns advice into a fact | High | Add check |
| "the user" naming | `memory_prompt.py:33-36` | Name if label exists, else role | B | F7, F8 | Two unlabelled humans | Med | Investigate |
| Speaker regex | `graph_index.py:45` | Capitalized prefix means speaker | B | `Note:`, `TODO:` | Wrong `SourceRef.speaker` | Low | Investigate |
| Transcript phrase list | `transcript.py:33-41` | Text prefix implies assistant role | C (one transcript's content) | health and routine phrases (removed) | Wrong role on headerless text | Med | Investigate |
| Equal timestamp means same session | `turn_context.py:8-9` | Adapter assumption | C/B | Docstring | Per-message timestamps give no context | Low | Investigate |

## 5. Core semantic coverage and missing tests

"Real run" means `runs/memory_v1`. "Scripted" means the model output was written by hand, so that test exercises validation and consolidation only.

| # | Case | Test coverage | Evidence and result | Gap |
|---|---|---|---|---|
| 1 | Multiple facts in a message | Scripted `:79` | Real run has multi-fact messages (`Sam has a dog` plus the Pepper fact) | Model behaviour only sampled |
| 2 | No useful memory | Scripted `:93` | Status path works | Model decision untested |
| 3 | Temporary opinion, small talk | **None** | v1 report shows opinions dropped; speculation also stored (F14) | **Demonstrated gap** |
| 4 | Unconfirmed assistant suggestion | Scripted `:153`, cannot fail | 1 of 82 assistant-sourced in real run; validator accepts in probes | **No enforcement test** |
| 5 | Natural paraphrase | **Rejection only** | F2 | **Demonstrated gap** |
| 6 | Correction or update | Scripted `:120` | 9 of 13 pairs handled in real run; F6 | **Gap past 20 claims, with pronouns, and when unlinked** |
| 7 | Unrelated facts about one person | Scripted `:100` | Holds because no slot exists | F13 untested |
| 8 | Name, pronoun, role reference | Partial (`test_v2_validation.py:169-206`) | F7, F8; wrong-speaker probe | **Roles and mixed naming untested** |
| 9 | Past event, present consequence | **None**; only "used to" at `:143` | Wrong on a LoCoMo speaker and Sam in real run | **Demonstrated gap** |
| 10 | Ambiguous, hypothetical, uncertain, negated | `test_negation_preserved`, `test_v2_temporal.py`; none on polarity cross-check | Probes: `The user gets the job` stored as current from "If I get the job" | **Demonstrated gap at the rule level** |
| 11 | Reprocessing makes no duplicates | Scripted `:188`, identical output only | Holds for duplicates; **fails on model failure** (F5) | **Failure and changed-output cases untested** |
| 12 | Traceable to source and speaker | Scripted `:153` | 100% originating source in the run for matched gold; speaker label can be wrong (F8) | Label false positives untested |

Minimum missing tests, in priority order:
1. Re-import with a failing model preserves existing memories.
2. A non-Latin or empty-token claim is not auto-accepted.
3. A negation flip is caught, or is an explicitly documented limit.
4. Advice from an `assistant` message cannot become a `user` fact.
5. Status for a past event with a lasting consequence.
6. A correction beyond the prior window.
7. A labelled set of faithful paraphrases is accepted.

## 6. Limitations of the existing evaluation

**What it genuinely establishes (on 6 synthetic conversations, one model, one seed):**
- The pipeline runs end to end with no failed windows.
- Provenance is intact for matched gold facts.
- Token and latency totals match the v1 report. I re-summed them from `runs/memory_v1`.
- The memories arm stored fewer records with fewer assistant-sourced items (1 vs 9) and fewer stale values (1 vs 4).

**Recall figures.**
- Recall is 96 of 113, with an approximate 95% Wilson interval of 77-90%.
- Against v2, 10 facts were captured only by v2 and 4 only by memories. An exact sign test gives p≈0.18.
- That test treats facts as independent, and they are not (W03 and W04 came from one claim). The 5.3-point difference is not established as real, in either direction.

**Why recall was lost (17 missed by memories):**

| Cause | Count | Facts | Notes |
|---|---|---|---|
| Emitted, rejected by source validation | 13 | A08, A12, S08, W03, W04, W18, W19, W22, W23, W24, P03, C07, T21 | 8 captured by v2; v2's validator also rejected 4 of the other 5 (A08, W03, W04, C07) |
| Selected out by design | 1 | W06 | Assistant advice |
| Sub-fact dropped by the model | 2 | W21, P17 | Causal detail; P17 also missed by v2 |
| Not emitted, cause unclear | 1 | T01 | "There are four people"; also missed by v2 |

- **Multiple causes, with validation dominant.** The v1 report's breakdown is accurate for its 10 v2-only facts, but those understate the validator's share by comparing only against v2.
- **Restoring all 13 would give 109 of 113 (96.5%). This is an upper bound and not a forecast**, since the same change would also admit unfaithful claims.

**Limits on what can be concluded:**
- Gold was built for a different extraction audit and lists assertions, not durable facts. "Recall" therefore does not measure durable recall, and it penalizes intended drops such as W06.
- Precision is a proxy. The 95.1% figure means "matches a gold entry", so true memories outside gold count against it, and there is no faithfulness label.
- Matching is lexical, on any one value plus a source message. It says nothing about whether a sentence is correct as a whole, and memory type, status correctness and subject are not scored as such.
- Annotation is hand-built, with no inter-annotator data that I found.
- The data is synthetic and English, with clean grammar and unusually explicit corrections. Informal conv-30 had a visibly higher rejection load.
- The 28-turn LoCoMo sample has no gold, is contaminated by F1, and its wrong-speaker check only tests for two names.
- One model, seed and machine. Repeatability under a changed model or seed is unmeasured.
- The evaluation was only run for extraction. No answer-quality comparison for this mode exists in the evidence reviewed.

**What generalizes.** As concepts: bag-of-stems cannot verify meaning, keyword status cannot separate event from state, and explicit-only updates trade stale values for safety. These follow from the logic and hold beyond these datasets. Every percentage above remains unverified outside them.

## 7. Architecture and release risks

### Confirmed from code or probes

| Question | Result |
|---|---|
| Existing modes compatible? | Yes. `extract="messages"` remains the default (`extract.py:45`, `workspace.py:196`, `server.py:126`). `ATOMIZATION_PROTOCOL` is unchanged and `SemanticAtomizer` defaults to it (`atomizer.py:45,135`). Unit suite result is in §1. |
| Memory types are metadata, not keys? | Yes (`validation.py:426-433`; `:88` test), except the F13 route if the model emits `claim_type`. |
| Provenance survives? | Yes. Provenance comes from the message at validation (`validation.py:436`), rows reference raw messages (`store.py:29-57`), and `SourceRef` carries ID, speaker, role, timestamp and claim. The weak part is the speaker label (F8). |
| Conservative updates safe? | Yes as a false-merge guard; stale values remain (F6). |
| Reprocessing safe? | No duplicates for identical output. **Not safe on model failure** (F5). Memory IDs are stable only for the same output: they derive from message ID and accepted-claim order (`validation.py:434`), so a different model output renumbers them. |
| Graph-hybrid consumes the records? | Yes at the test level (`:208-234`), with no dataset-specific condition. However, `graph_hybrid` lives in `webapi/locomo_report.py`; a search of `workspace.py` and `server.py` found no use. The test also sets private attributes. The graph itself is built from raw messages (`locomo_report.py:249`), so messages with no memory stay reachable through it. |
| `MemoryRecord` used in production? | No. Only `benchmarks/experiments/memory_v1/run.py` and tests call `memory_records`. |
| Raw-message default a product gap? | **Both.** It is an intentional compatibility choice, since all prior evaluations depend on it (`reports/memory_extraction_v1.md` §1, §7). It is also a real gap: default imports get no durable extraction. |
| Reachable by users? | Only through a direct `POST /import` or Python. The web UI sends only `v2` or `messages` (`ImportConversation.tsx:43`, `service.ts:41`) and the CLI has no such option. |

### Cost, token and latency observability

| Signal | State |
|---|---|
| Latency per window | Persisted (`store.py:91`) |
| Tokens and model calls | In `V2IngestReport` in memory only. **Not persisted**, since the log schema has no such column (`store.py:79-93`) |
| `done_reason` (output cutoffs) | Captured per call, then **dropped** from the report (`atomizer.py:120`, `pipeline.py:100-103`) |
| Issue counts | In the report, not in the API response. Per-window issues are in the log |
| HTTP import response | Only windows, failed windows and counts (`workspace.py:252-263`); **no tokens, latency or rejections** |
| Extraction mode after restart | **Lost.** The log stores the model name but not the protocol, and `from_sqlite` reports `message_claims` for any imported conversation (`workspace.py:186`) |

**Enough for a V1 opt-in if documented; not enough to run at default scale.**

Measured latency was 59 s per window on synthetic data and 79 s on conv-30, on one local setup. Extrapolating linearly, at 4 targets per window:

| Input | Windows | Estimated time |
|---|---|---|
| conv-30 (369 turns) | 93 | 1.5-2 hours |
| Maximum import (2,000 messages) | 500 | 8-11 hours |
| All LoCoMo (5,882 turns) | 1,471 | 24-32 hours |

### Assumptions, not verified

- Ollama's JSON-schema enforcement blocks fields outside the schema. The absence of `claim_type` in the run is consistent with this, but the code does not forbid them.
- Model output is repeatable across versions at temperature 0 and seed 0.
- Extending time estimates linearly holds for longer conversations; the prior context grows only to its limit, which supports it but does not prove it.

## 8. Should the mode stay opt-in?

**Stay opt-in. It is not ready to be the default.**

- The current default has perfect trivial recall and no model cost. Replacing it introduces data loss from validator rejection, stale status and multi-hour imports, without any answer-quality evidence in its favour.
- The evidence does not show it beats v2: it trades recall for precision, on a small and partly contaminated sample.
- Its strongest result, fewer assistant-sourced memories, comes from the prompt and has no deterministic backstop.
- The evaluation's only real conversation is not held out.

### Release blockers

**For an opt-in V1 (all small; confirmed):**
1. **F1:** Replace the conv-30 example and the conv-26 name with actually invented text, and correct the docstring. Do not cite conv-30 session 1 as held-out evidence.
2. **F5:** Extract first, then swap results only if no window failed. Otherwise keep the old rows and report the failure.
3. **F3 (partial):** Make the empty-token case fail closed, and correct the "Rules enforced" claims in the v1 report and the `explicit` wording in `memory_records.py` so the docs state what is prompt-only. Document V1 as English-only unless evidence appears otherwise.

**Additionally, before any change of default:**
4. Build an independently labelled held-out set: real conversations, informal register, and labels for faithfulness, durability, status and type. Fix this set **before** changing the rules. Include answer quality in the comparison.
5. Redesign the support check as a separate decision from word overlap, covering negation, speaker direction, hypotheticals and context-resolved memories. Measure both false accepts and false rejects. Do not extend the synonym or stem lists to chase the 13 facts.
6. Stop setting `historical` from a single word. Separate "event happened in the past" from "state ended". Only explicit past-state wording, or an update, should end a state.
7. Decide how corrections beyond the last 20 claims find their target.
8. Persist tokens, `done_reason` and protocol version per window; add the same to the import response.

### Can wait until after V1

Subject normalization (F7), multi-speaker identity (F8), speaker regex precision, transcript phrase list (F11), cross-conversation memory if out of V1 scope (F10), extraction cache, UI and CLI exposure, retries.

## 9. Uncertainties and what would resolve them

| Uncertainty | Why it is open | What would resolve it |
|---|---|---|
| How often the model produces F3-style unfaithful claims | Probes show the rule permits them, not the frequency | Manual faithfulness labels on stored memories across held-out real conversations |
| How much the contaminated example affects behaviour | No run without it exists | A rerun of the existing arm after the example swap, compared per record. It should be read as a contamination check, not as tuning |
| True durable recall and precision | Gold is not durability-labelled | Blocker 4's labelled set |
| Whether the 5.3-point v2 difference is real | Small, dependent sample | Larger held-out set and multiple seeds or models |
| Effect of status errors on answers | Only the ranking penalty is known | Answer-level comparison on status-sensitive questions |
| Whether strict validation prevents real fabrications | 0 of 16 synthetic rejections were clearly invented; conv-30's 12 were not fully labelled; other runs were out of scope | Label rejected claims across more data, including the LongMemEval calibration logs |
| Performance on non-English or very long conversations | Not evaluated | A small targeted sample after blocker 3 |
| Whether V1 must span conversations or support non-English | Product scope, not stated in code | Your decision |
| Age of the failing webapi test | I confirmed the failure, not its origin | Run it against the stashed or committed baseline |
| Whether speaker-role mis-assignment from `_ASSISTANT_START` affects real imports | The path was read in part only | Trace on headerless sample transcripts |
