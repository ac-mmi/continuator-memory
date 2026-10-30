# Memory extraction V1 release manifest

Audience: the engineer staging the opt-in `extract="memories"` release candidate, and the person approving it.

**Nothing has been staged, committed, pushed or tagged.** Decisions recorded so far are in §14.

## 1. Current HEAD and existing tags

| Item | Value |
|---|---|
| HEAD | `8ec19eb` V1 reproducibility hardening (branch `main`) |
| Tags | `v1.0.0` at `d2a665e` (V1 Phase 3.6 frozen baseline), `v1.0.1` at `8ec19eb` (V1 frozen reproducibility baseline) |
| Package version | `0.1.0` in `pyproject.toml` and `__init__.py` |
| Uncommitted paths, including this manifest | 1284 (5 modified, 1279 untracked) |

## 2. Release scope

**Included:**
- `ingest_imported(extract="memories")`
- `memory_records` and `MemoryRecord`
- `POST /import` and `Workspace.import_text`
- The existing retrieval read path, including `graph_hybrid`, because a release test requires it
- The default mode stays `messages`

**Not part of this release:**
- **Web UI:** it offers only `v2` and `messages` (`ImportConversation.tsx:43`, `service.ts:41`), so it is separate product work.
- **A CLI option for memory extraction:** none exists. The included `cli.py` change only adds `serve`, which starts the API.

## 3. Exact required file list

Totals:

| Category | Count |
|---|---|
| A. Production source | 42 |
| B. Tests and fixtures | 27 |
| C. Documentation, whole files | 3 |
| C. Documentation, partial: `README.md` | 1 |
| D. Runtime configuration and packaging | 2 |
| **Total** | **75** |

Of these, 5 are modified tracked files and 70 are new.

### A. Required production source

**Core extraction path: `ingest_imported(extract="memories")`**

| File | Why it is needed |
|---|---|
| `src/continuator_memory/webapi/__init__.py` | Package marker for the entry point |
| `src/continuator_memory/webapi/extract.py` | `ingest_imported`, `EXTRACT_MODES`, default mode |
| `src/continuator_memory/v2/__init__.py` | Package marker |
| `src/continuator_memory/v2/pipeline.py` | Windows, staging, transaction-safe replacement, usage report |
| `src/continuator_memory/v2/atomizer.py` | `MEMORY_PROTOCOL`, Ollama client with JSON schema, usage capture |
| `src/continuator_memory/v2/memory_prompt.py` | Production prompt, fictional examples, output schema |
| `src/continuator_memory/v2/prompt.py` | Atomizer protocol, plus `_message_line` and `_prior_line` imported by the memory prompt |
| `src/continuator_memory/v2/windowing.py` | `ContextWindow`, `build_windows` |
| `src/continuator_memory/v2/validation.py` | Structured output parsing, source check, `UNVERIFIABLE_CLAIM` |
| `src/continuator_memory/v2/models.py` | `Claim`, `Relation`, `MEMORY_TYPES`, statuses |
| `src/continuator_memory/v2/store.py` | Semantic schema (`CREATE TABLE IF NOT EXISTS`; no migration files) and non-committing writers |
| `src/continuator_memory/store.py` | Modified: `write_conversation` allows a single transaction |

**Provenance and `MemoryRecord`**

| File | Why it is needed |
|---|---|
| `src/continuator_memory/v2/memory_records.py` | `MemoryRecord`, `memory_records` |
| `src/continuator_memory/v2/consolidation.py` | Versioning, dedup, update handling |
| `src/continuator_memory/v2/identity.py` | Subject and slot identity used by consolidation |
| `src/continuator_memory/v2/memory.py` | `Memory`, sources, links, status constants |
| `src/continuator_memory/v2/temporal.py` | Temporal classes used by consolidation |
| `src/continuator_memory/v2/graph_index.py` | `split_speaker` used by `memory_records`; graph retrieval |

**API integration: `POST /import` and `Workspace.import_text`**

| File | Why it is needed |
|---|---|
| `src/continuator_memory/webapi/server.py` | HTTP API, `POST /import` |
| `src/continuator_memory/webapi/__main__.py` | `python -m continuator_memory.webapi` |
| `src/continuator_memory/webapi/workspace.py` | `import_text`, `kept_existing` error path, read path |
| `src/continuator_memory/webapi/transcript.py` | `parse_transcript` for imports |
| `src/continuator_memory/webapi/sample.py` | Imported at load by `workspace.py` |
| `src/continuator_memory/webapi/answering.py` | Imported at load by `workspace.py`; answer path |
| `src/continuator_memory/webapi/evidence_eval.py` | Imported at load by `workspace.py` |
| `src/continuator_memory/webapi/iterative.py` | Imported at load by `workspace.py` |
| `src/continuator_memory/webapi/multiquery.py` | Imported at load by `iterative.py` |
| `src/continuator_memory/webapi/prompt_budget.py` | Imported at load by `answering.py` |
| `src/continuator_memory/webapi/query_plan.py` | Imported at load by `answering.py` |
| `src/continuator_memory/envfile.py` | `load_root_env`, used by server, workspace and CLI |
| `src/continuator_memory/generation.py` | Modified: `answering.py` calls `generate(temperature=, num_predict=)` and reads `last_usage` |
| `src/continuator_memory/cli.py` | Modified: `serve` command, required by `test_serve_command_does_not_build_pipeline` |
| `src/continuator_memory/v2/source_retrieval.py` | Moved from `benchmarks/`: memory-to-source retrieval, imported by `workspace.py`, `multiquery.py`, `iterative.py`, `locomo_report.py` |
| `src/continuator_memory/v2/bprime_links.py` | Moved from `benchmarks/`: `BPrimeConfig` and link expansion, imported by `source_retrieval.py`, `workspace.py`, `locomo_report.py` |

**Existing retrieval compatibility**

| File | Why it is needed |
|---|---|
| `src/continuator_memory/v2/memory_retrieval.py` | Ranks consolidated memories |
| `src/continuator_memory/v2/hybrid_retrieval.py` | Imported by `workspace.py` |
| `src/continuator_memory/v2/lexical.py` | Lexical weight for ranking |
| `src/continuator_memory/v2/turn_context.py` | Context weight for ranking |
| `src/continuator_memory/webapi/locomo_report.py` | Holds `graph_hybrid`; needed by `test_extracted_memories_work_with_graph_hybrid_retrieval`. No dataset is read at import |
| `src/continuator_memory/webapi/iterative_diag.py` | Imported at load by `locomo_report.py` |
| `src/continuator_memory/evaluation/answer_scoring.py` | Imported at load by `locomo_report.py` |
| `src/continuator_memory/v2/memory_store.py` | Not on the production path; imported by the consolidation, temporal, identity and retrieval tests |

### B. Required tests and fixtures

These run alongside the tracked tests in `tests/`, plus `tests/helpers.py` and `tests/conftest.py`, which are already committed.

| File | Why it is needed |
|---|---|
| `tests/unit/test_memory_v1_blockers.py` | Prompt, failure-safe re-import and fail-safe validation |
| `tests/unit/test_memory_extraction_v1.py` | Memory contract, provenance, graph-hybrid compatibility |
| `tests/unit/test_v2_pipeline.py` | Pipeline and replacement |
| `tests/unit/test_v2_validation.py` | Validation |
| `tests/unit/test_v2_windowing.py` | Windows |
| `tests/unit/test_v2_consolidation.py` | Consolidation |
| `tests/unit/test_v2_identity.py` | Identity |
| `tests/unit/test_v2_temporal.py` | Temporal semantics |
| `tests/unit/test_v2_temporal_relative.py` | Relative dates (32 cases); no longer needs a LoCoMo script |
| `tests/unit/test_v2_memory_store.py` | `memory_store.py` |
| `tests/unit/test_v2_memory_retrieval.py` | Memory retrieval |
| `tests/unit/test_v2_hybrid_retrieval.py` | Hybrid retrieval |
| `tests/unit/test_graph_index.py` | Graph retrieval and speaker parsing |
| `tests/unit/test_lexical.py` | Lexical ranking |
| `tests/unit/test_turn_context.py` | Turn context |
| `tests/unit/test_webapi.py` | API, import and workspace (holds the known failure) |
| `tests/unit/test_transcript.py` | Transcript parsing |
| `tests/unit/test_envfile.py` | Env file loading |
| `tests/unit/test_answering.py` | Answer path |
| `tests/unit/test_answer_attribution.py` | Answer attribution |
| `tests/unit/test_prompt_budget_and_dates.py` | Prompt budget |
| `tests/unit/test_trim_and_artifacts.py` | Answer trimming |
| `tests/unit/_lme_fixtures.py` | Fixture imported by `test_trim_and_artifacts.py` |
| `tests/unit/test_query_plan.py` | Query plan |
| `tests/unit/test_evidence_eval.py` | Evidence evaluation |
| `tests/unit/test_multiquery.py` | Multi-query |
| `tests/unit/test_iterative.py` | Iterative retrieval |

### C. Required documentation

| File | Why it is needed |
|---|---|
| `reports/memory_extraction_v1.md` | Implementation record, smoke test, limitations |
| `reports/memory_extraction_heuristic_audit.md` | Audit behind the blockers; cited by the README |
| `reports/memory_extraction_v1_release_manifest.md` | This manifest |

`README.md` is partial: stage only the `### Memory extraction (opt-in)` section (§6).

### D. Required runtime configuration and packaging

| File | Why it is needed |
|---|---|
| `.gitignore` | Modified: one line, ignores `.env`, which `envfile.py` reads |
| `.env.example` | Documents the env file that `envfile.py` loads; no real values |

`pyproject.toml` needs no change. The release adds no third-party dependency: only `numpy` and `sentence-transformers` are imported, and both are already declared. `reproducibility/` is tracked and unchanged. There are no migration files; the schema is created on open.

### Staging list

Another engineer can stage from this block. It excludes `README.md`, which is handled in §6.

<!-- staging-list-start -->
```text
.env.example
.gitignore
reports/memory_extraction_heuristic_audit.md
reports/memory_extraction_v1.md
reports/memory_extraction_v1_release_manifest.md
src/continuator_memory/cli.py
src/continuator_memory/envfile.py
src/continuator_memory/evaluation/answer_scoring.py
src/continuator_memory/generation.py
src/continuator_memory/store.py
src/continuator_memory/v2/__init__.py
src/continuator_memory/v2/atomizer.py
src/continuator_memory/v2/bprime_links.py
src/continuator_memory/v2/consolidation.py
src/continuator_memory/v2/graph_index.py
src/continuator_memory/v2/hybrid_retrieval.py
src/continuator_memory/v2/identity.py
src/continuator_memory/v2/lexical.py
src/continuator_memory/v2/memory.py
src/continuator_memory/v2/memory_prompt.py
src/continuator_memory/v2/memory_records.py
src/continuator_memory/v2/memory_retrieval.py
src/continuator_memory/v2/memory_store.py
src/continuator_memory/v2/models.py
src/continuator_memory/v2/pipeline.py
src/continuator_memory/v2/prompt.py
src/continuator_memory/v2/source_retrieval.py
src/continuator_memory/v2/store.py
src/continuator_memory/v2/temporal.py
src/continuator_memory/v2/turn_context.py
src/continuator_memory/v2/validation.py
src/continuator_memory/v2/windowing.py
src/continuator_memory/webapi/__init__.py
src/continuator_memory/webapi/__main__.py
src/continuator_memory/webapi/answering.py
src/continuator_memory/webapi/evidence_eval.py
src/continuator_memory/webapi/extract.py
src/continuator_memory/webapi/iterative.py
src/continuator_memory/webapi/iterative_diag.py
src/continuator_memory/webapi/locomo_report.py
src/continuator_memory/webapi/multiquery.py
src/continuator_memory/webapi/prompt_budget.py
src/continuator_memory/webapi/query_plan.py
src/continuator_memory/webapi/sample.py
src/continuator_memory/webapi/server.py
src/continuator_memory/webapi/transcript.py
src/continuator_memory/webapi/workspace.py
tests/unit/_lme_fixtures.py
tests/unit/test_answer_attribution.py
tests/unit/test_answering.py
tests/unit/test_envfile.py
tests/unit/test_evidence_eval.py
tests/unit/test_graph_index.py
tests/unit/test_iterative.py
tests/unit/test_lexical.py
tests/unit/test_memory_extraction_v1.py
tests/unit/test_memory_v1_blockers.py
tests/unit/test_multiquery.py
tests/unit/test_prompt_budget_and_dates.py
tests/unit/test_query_plan.py
tests/unit/test_transcript.py
tests/unit/test_trim_and_artifacts.py
tests/unit/test_turn_context.py
tests/unit/test_v2_consolidation.py
tests/unit/test_v2_hybrid_retrieval.py
tests/unit/test_v2_identity.py
tests/unit/test_v2_memory_retrieval.py
tests/unit/test_v2_memory_store.py
tests/unit/test_v2_pipeline.py
tests/unit/test_v2_temporal.py
tests/unit/test_v2_temporal_relative.py
tests/unit/test_v2_validation.py
tests/unit/test_v2_windowing.py
tests/unit/test_webapi.py
```
<!-- staging-list-end -->

```bash
# 1. Stage the whole files, from the repository root.
awk '/^<!-- staging-list-start -->$/{f=1;next} /^<!-- staging-list-end -->$/{f=0} f && !/^```/' \
  reports/memory_extraction_v1_release_manifest.md > /tmp/v1_release_files.txt
git add --pathspec-from-file=/tmp/v1_release_files.txt

# 2. Stage only the README section (see section 6).

# 3. Review before committing.
git diff --cached --stat
git diff --cached --name-only | grep -E '^(runs/|web/|scripts/|locomo|data/|benchmarks/external/)|\.db$|^\.env$'  # must print nothing
```

## 4. Excluded files and rules

Everything else stays uncommitted: **1209 paths**, plus the rest of `README.md`.

### E. Optional evaluation code and reports

Not required at runtime or by release tests. Report §6 of `memory_extraction_v1.md` cites them, so its reproduction commands will not work from the release tree unless a later commit adds them.

- `benchmarks/experiments/memory_v1/`
- `benchmarks/experiments/extraction_audit/`
- `runs/memory_v1/` (except its databases; see G)
- `reports/*` except the three listed in §3C

### F. Unrelated or experimental work

| Rule | Paths | Note |
|---|---|---|
| `benchmarks/experiments/` | 909 | Includes all of `phase7_memory_source_retrieval/`, whose `retrieval.py` and `bprime_links.py` are now thin wrappers over the production modules |
| `benchmarks/synthetic/phase3c_retrieval/`, `benchmarks/synthetic/phase4_e2e/` | 18 | Experiment inputs |
| `benchmarks/annotations/` | 1 | Evaluation annotations |
| `reports/` except the 3 listed files | 69 | Generated reports |
| `web/` | 58 | Separate product work |
| `scripts/` | 2 | LoCoMo scoring and judging |
| `ARCHITECTURE_CRIME_SCENE.md` | 1 | Purpose not established; left out |

Source files left out:

- `src/continuator_memory/v2/context.py`
- `src/continuator_memory/v2/context_store.py`
- `src/continuator_memory/v2/entities.py`
- `src/continuator_memory/v2/entity_store.py`
- `src/continuator_memory/v2/evidence_diagnostic.py`
- `src/continuator_memory/v2/expansion.py`
- `src/continuator_memory/v2/expansion_store.py`
- `src/continuator_memory/v2/relationship_diagnostic.py`
- `src/continuator_memory/v2/relationship_store.py`
- `src/continuator_memory/v2/relationships.py`
- `src/continuator_memory/webapi/hybrid_diag.py`
- `src/continuator_memory/webapi/lexical_diag.py`
- `src/continuator_memory/webapi/locomo.py`
- `src/continuator_memory/webapi/ranking_diag.py`
- `src/continuator_memory/webapi/rewrite_eval.py`

Test files left out, each because it needs excluded code or data:

- `tests/unit/test_eval_instrumentation.py`
- `tests/unit/test_hybrid_diag.py`
- `tests/unit/test_judge_locomo_answers.py`
- `tests/unit/test_locomo_adapter.py`
- `tests/unit/test_locomo_report.py`
- `tests/unit/test_neutral_shuffle.py`
- `tests/unit/test_score_locomo_answers.py`
- `tests/unit/test_v2_context.py`
- `tests/unit/test_v2_entities.py`
- `tests/unit/test_v2_evidence_diagnostic.py`
- `tests/unit/test_v2_expansion.py`
- `tests/unit/test_v2_relationship_diagnostic.py`
- `tests/unit/test_v2_relationships.py`

### G. Sensitive or unsafe artifacts, never commit

| Rule | Paths | Note |
|---|---|---|
| `locomo10.json`, `locomo_conv26.json`, `locomo_conv30.json`, `locomo_conv41.json` to `locomo_conv44.json`, `locomo_conv47.json` to `locomo_conv50.json` | 11 | Not used at runtime. **Redistribution rights not established** |
| `benchmarks/external/` | 20 | 265 MB, including `longmemeval/longmemeval_s_cleaned.json`. Not used at runtime. **Redistribution rights not established** |
| `runs/` | 92 | 115 MB of generated output with dataset text. Not used at runtime |
| `.env`, `data/`, `*.db` | ignored by git | Local configuration and databases. Must stay ignored |

The staging commands name files explicitly, so none of these can be picked up by them. Never use `git add .` or `git add -A` here.

## 5. Dependency reasoning

- **Method:** I parsed imports from each entry point, including function-level imports, then proved the result by running the tree (§8).
- **`ingest_imported` and `memory_records`** need 26 modules, and these import without `benchmarks/`.
- **The API path needs more than its name suggests.** `workspace.py` pulls in the answer and retrieval modules at load, so `POST /import` needs nearly all of the source above.
- **Production no longer imports `benchmarks`.** An AST scan of `src/` finds no such import, and the `sys.path` change in `workspace.py` is gone. With only `src/` present, `workspace`, `server`, `iterative`, `multiquery`, `locomo_report`, `extract`, `source_retrieval`, `bprime_links` and `memory_records` all import, and `benchmarks` is not loaded.
- **What moved:** the whole of `retrieval.py` (source resolution, budget, evidence) and `bprime_links.py` (config, classification loading, link expansion). No runner, experiment or evaluation logic moved. The frozen default classifications path stayed in the benchmark wrapper.
- **One behaviour change:** `BPrimeConfig(enabled=True)` with no path now raises `ValueError` in production instead of falling back to a benchmark file. `workspace.py` already checks for a path first. `locomo_report.py` in `bprime` mode with no classifications file now raises. The benchmark wrapper keeps the old default.
- **`cli.py` is not needed for extraction.** It's included because `test_webapi.py` covers `serve`. Its dependencies are all in this list.
- **`generation.py` is required** by the answer path, whose calls fail against the committed signature.
- **Datasets:** no file in this list reads LoCoMo, LongMemEval or `runs/` at import or during release tests. `locomo_report.py` reads a dataset only when it's run as an evaluation command.

## 6. Unrelated changes requiring separate commits

| File | Finding | Action |
|---|---|---|
| `README.md` | **Mixed.** One hunk holds Web UI, Memory extraction and LoCoMo report sections; another adds `CONTINUATOR_ANSWER_TOP_K` | Stage only the memory section |
| `src/continuator_memory/cli.py` | One concern: `serve` and `.env` loading. Not extraction, but required by a release test | Stage whole |
| `src/continuator_memory/generation.py` | One concern: options and usage. Required by the answer path | Stage whole |
| `.gitignore` | One line, `.env` | Stage whole |
| `src/continuator_memory/store.py` | V1 transaction change only | Stage whole |

`cli.py` loading `.env` also changes the existing CLI: it now reads a root `.env` if present, without overriding variables already set.

**To stage only the README section**, run this from the repository root. It changes the index only, and the rest of your README edits stay in the working tree.

```bash
python3 - <<'EOF'
import subprocess
head = subprocess.check_output(["git", "show", "HEAD:README.md"], text=True)
wt = open("README.md").read()
sec = wt[wt.index("### Memory extraction (opt-in)"):wt.index("### LoCoMo RAG report")]
anchor = "`python -m continuator_memory ...` works the same way.\n\n"
assert head.count(anchor) == 1
open("/tmp/README.v1", "w").write(head.replace(anchor, anchor + sec))
EOF
git update-index --cacheinfo 100644,"$(git hash-object -w /tmp/README.v1)",README.md
git diff --cached -- README.md   # must show only the memory extraction section
```

After committing, the Web UI and LoCoMo README sections remain as uncommitted changes for a separate commit, with `web/` and the evaluation work.

## 7. Security review

**Inspected:** the 73 whole files listed in §3 other than this manifest, which was written afterwards, plus `README.md`. About 1 MB of text. No binaries, databases or JSON data files are in the list.

| Check | Result |
|---|---|
| Token and key patterns (OpenAI, Anthropic, AWS, GitHub, GitLab, Slack, Hugging Face, JWT, private keys, `password=`-style assignments) | No matches |
| Local absolute paths (`/Users/`, `/home/`, `/private/tmp`, `/var/folders`, `C:\`) | No matches |
| Email addresses and non-loopback IP addresses | No matches |
| Git user name and email, local account name | No matches |
| `.env.example` | One non-secret value, a model name |
| Environment-specific defaults | `127.0.0.1`, `localhost` CORS, `localhost:11434`, relative `data/*.db`. All overridable and safe |

**Privacy corrections made:**

| Content | Finding | Action |
|---|---|---|
| `transcript.py` assistant-phrase list | Health and daily-routine phrases. No source conversation is in the repository, so the origin can't be confirmed; treated as private | Removed from the regex, not replaced: invented phrases have no place in a production parser |
| Work-shift wording in `test_webapi.py`, `test_transcript.py` | Matches the removed phrases, so treated as the same conversation | Replaced with a fictional ferry timetable |
| `_lme_fixtures.py` and `test_graph_index.py` | Verbatim LongMemEval text, including therapy and anxiety content (dataset text, not known to be personal) | Replaced with fictional text; same length and `$20` offset, so the assertions are unchanged |
| LoCoMo speakers and excerpts in 8 test files | Sensitive topics and verbatim turns; not needed by the logic | Replaced with fictional names and topics |
| LoCoMo quotes in both reports | Not needed once paraphrased | Paraphrased, or replaced with placeholders such as `<Speaker>`; all figures kept and labelled as evaluation records |

**Rescanned after the corrections:** the removed phrases, shift wording, secrets and local paths have no matches in the release files, except this manifest's own description of the scan patterns.

**Dataset references that remain:**

- `src/continuator_memory/webapi/evidence_eval.py:225` hardcodes `caroline` and `melanie` in the heuristic evaluator. It changes results by name, but the API can't reach it (§13).
- `iterative_diag.py` holds a LoCoMo question and default paths; `locomo_report.py` is LoCoMo evaluation code; `graph_index.py:44` names a speaker in a comment.
- `test_memory_v1_blockers.py` lists LoCoMo speakers to assert they are absent from the prompt, so the list is needed.
- `D1:9`-style message IDs and identifiers such as `MSG_731E37D7` remain as test IDs. The reports still cite conversation IDs and counts as evaluation records.

**Limits of this review:** the scan is pattern-based, so it can miss unusual secret formats and encoded values, and it cannot judge whether text is private. I read the release code selectively, not every line. **Files outside this list were not reviewed, and this is no statement that the repository as a whole is clean.** Git history was not scanned.

## 8. Known test failures and release limitations

**Isolation test.** I cloned the repository to a temporary directory and added decoy dataset and `runs/` files. I then staged using the commands above in that clone, exported only the index with `git checkout-index` and ran tests against the export. The real repository's index was not touched.

| Run on the exported tree | Result |
|---|---|
| `tests/unit` | **878 passed, 1 failed** (the known failure below) |
| Blockers, V1 contract, relative dates, transcript and trimming tests | 103 passed |
| Staged files | 75 (70 added, 5 modified) |
| Forbidden paths staged, including decoys | 0 |
| Imports with only `src/` present | All pass; `benchmarks` not loaded |

In the full working tree, `tests/unit` is 1183 passed, 1 failed (the same failure). With `benchmarks/` removed from the export, the result is 856 passed, 13 failed, 10 errors. An earlier run classified these as the known failure plus 22 missing tracked data files under `benchmarks/`, which this release doesn't change; the counts are the same now. `benchmarks/experiments/phase7_memory_source_retrieval/tests` is 82 passed, 1 failed, unchanged from before the move; the failure is `test_existing_v1_v2_code_matches_the_frozen_dependency_record`, a hash check that was already failing.

`tests/integration` needs the real embedding model and was not run. No model was called.

**Regression fixtures are now fictional.** Answer-path tests that used saved model output keep the same structure, but they no longer replay the literal text from those runs.

**Known failure:** `tests/unit/test_webapi.py::test_db_sample_uses_import_db_when_served_sqlite_has_no_claims`. It comes from how `open_workspace` treats a dedicated database, and an equivalent flow passes when `CONTINUATOR_DB_PATH` points at the served database. It fails in the full working tree too. Whether it predates the memories work cannot be proven, because no committed baseline exists.

**Temporal tests:** `test_v2_temporal_relative.py` is now in the release. Only `test_judge_uses_the_shared_resolver` needed the LoCoMo script, so it moved to `test_judge_locomo_answers.py`, which stays excluded. **Correction:** the earlier manifest said 13 tests; the file has 32 cases once parametrized.

**Release limitations:**
- **Packaging:** the API path no longer needs a source checkout to import. No wheel was built or installed, so that remains untested.
- **Evaluation code in a production package:** `locomo_report.py` ships only because a release test needs it.
- **Model not documented:** extraction defaults to `qwen2.5:14b` (`CONTINUATOR_V2_ATOMIZER_MODEL`), which neither the README nor `.env.example` mentions.
- **Headerless transcripts:** see §13. With only single blank lines, uppercase user paragraphs are merged into the assistant turn. Use explicit headers or JSON when roles matter.
- **Stale docstrings:** `v2/__init__.py` describes modules excluded here, and the `pyproject.toml` description still says Phase 1.
- **Semantic limits:** see `memory_extraction_v1.md` §9 (smoke test, 8 of 9 checks) and §10.

## 9. Recommended release-candidate tag

**`v1.1.0-rc.1`**

- `v1.0.0` and `v1.0.1` are frozen baselines, and this adds a new opt-in feature on top, so a minor version fits better than a patch.
- `-rc.1` marks it as a candidate.
- **Version mismatch:** see §11. Nothing was changed.

## 10. Final human approval checklist

- [x] Accept headerless transcript parsing as best-effort, with its speaker misattribution risk (§13, §14).
- [x] Accept the name-specific evaluator as diagnostic-only and benchmark-specific (§13, §14).
- [x] Record the extraction limitations in the README and `memory_extraction_v1.md` §10 (§14).
- [ ] Decide on the license (§11).
- [ ] Decide on the package version (§11).
- [ ] Confirm the LoCoMo and LongMemEval references left in shipped files are acceptable, given that redistribution rights are not established (§7).
- [ ] Accept the known failing test (§8).
- [ ] Confirm `cli.py`'s `.env` loading is acceptable for existing CLI users (§6).
- [ ] Approve the tag name `v1.1.0-rc.1` (§9).
- [ ] Stage with §3 and §6, then confirm the `git diff --cached` checks.
- [ ] Rerun `tests/unit` from a clean checkout of the resulting commit, then tag.

## 11. Release decisions for the project owner

Neither has been changed.

| Decision | Current state | Why it needs an owner |
|---|---|---|
| **License** | MIT license added in `LICENSE`; package version remains unchanged | MIT terms apply to the released code |
| **Package version** | `0.1.0` in `pyproject.toml` and `__init__.py`; tags are `v1.0.0`, `v1.0.1`, proposed `v1.1.0-rc.1` | The mismatch predates this work. Either align the package with the tags or keep the two schemes separate on purpose |

## 12. Pre-staging corrections: exact files changed

**Added to the release (3):** `src/continuator_memory/v2/source_retrieval.py`, `src/continuator_memory/v2/bprime_links.py`, `tests/unit/test_v2_temporal_relative.py`.

**Removed from the release (3):** `benchmarks/experiments/phase7_memory_source_retrieval/__init__.py`, `bprime_links.py`, `retrieval.py`.

**Changed, already in the release:**

- Source: `webapi/workspace.py`, `iterative.py`, `multiquery.py`, `locomo_report.py` (imports), `webapi/transcript.py` (phrase list). All under `src/continuator_memory/`.
- Tests: `test_answering.py`, `test_answer_attribution.py`, `test_evidence_eval.py`, `test_query_plan.py`, `test_turn_context.py`, `test_v2_memory_retrieval.py`, `test_graph_index.py`, `test_memory_extraction_v1.py`, `test_trim_and_artifacts.py`, `_lme_fixtures.py`, `test_webapi.py`, `test_transcript.py`. All under `tests/unit/`.
- Docs: `reports/memory_extraction_v1.md`, `reports/memory_extraction_heuristic_audit.md`, this manifest.

**Changed, outside the release, left uncommitted:** `benchmarks/experiments/phase7_memory_source_retrieval/retrieval.py` and `bprime_links.py` (now wrappers), `tests/unit/test_judge_locomo_answers.py` (received the moved test).

**Not changed:** extraction prompts, memory semantics, the default import mode, graph-hybrid behaviour, `pyproject.toml`. No datasets or generated artifacts were added.

## 13. Final review: transcript semantics, name-specific logic, B′

**Summary**

| Question | Answer |
|---|---|
| Was the parser change a regression? | **Partly.** A narrow regression came from the privacy cleanup, and it is accepted. Uppercase misclassification is an older defect, now fixed where the text has structure |
| Was the name-specific logic reachable from production? | **No.** It is diagnostics-only, but it ships in a production module and its results depend on the name |
| Is the 75-file manifest still valid? | **Yes.** No file was added or removed; three listed files changed |

### Transcript parser

`transcript.py` was never committed, so the original was reconstructed from this work's session record. I ran that version and the cleaned-up one over 18 sample paragraphs in a scratch script; the removed phrases were not written to the repository.

**Original and cleaned-up parser.** After the first paragraph, a paragraph counts as a user turn only if it starts lowercase or contains `i don't`, `i dont`, `you know` or `also i don't`/`also i dont`. **These conditions were not added by the cleanup; only the phrase list changed.** They are generic, but weak.

| Sample paragraph | Original | After cleanup |
|---|---|---|
| Uppercase user statement, question or correction | Not user | Not user (older defect) |
| Lowercase user message | User | User |
| Ordinary assistant paragraph, short title or numbered step | Not user | Not user |
| `As you know, ...` or `Sorry, I don't ...` from the assistant | User (wrong) | User (wrong) |
| Assistant paragraph starting with a removed phrase, and also starting lowercase or containing `you know` / `i don't` | Not user | **User (regression)** |

**Regression: 4 of the 18 samples, all built to target it, so this shows the mechanism and not how often it occurs.** Restoring the list would reintroduce the private phrases, and it only ever covered one conversation, so it stays removed.

**Fix.** When two or more blank lines appear, they are now treated as turn breaks, and roles alternate from the user. This is structural, not another phrase list. It applies only if all of the following hold, and otherwise the earlier logic runs:
- No block in an assistant position looks like a user turn.
- No later block in a user position starts with an assistant phrase or a heading.

**What this does not fix**
- **Single blank lines only:** uppercase user paragraphs still merge into the assistant turn. Text alone gives no reliable signal, and a first-person rule would be another brittle list. A test pins this. A user fact merged this way carries the assistant role.
- **The `you know` and `i don't` rule** still marks some assistant paragraphs as user. Older behaviour, left alone.
- **Uppercase assistant text with an internal double blank line** can pass the checks and flip later roles, because nothing in the text exposes it. Real pasted transcripts were not available to measure how often this happens.
- **Remaining phrase list:** it has no private content, but it was still fitted to informal chat.

Explicit headers and JSON take a separate path and are unaffected.

### `evidence_eval.py:225`

- **Why the names are there:** `caroline` and `melanie` are the two speakers of LoCoMo conv-26 (`locomo_conv26.json`, `speaker_a` and `speaker_b`), the conversation `iterative_diag.py` evaluates by default. So this is benchmark-specific.
- **Results depend on the name.** My earlier four probes didn't exercise the affected branches, so they proved nothing. These two do:

| Question and excerpt | `Caroline`, `Melanie` | `Priya`, `Sam`, `Oluwaseun` |
|---|---|---|
| `When did <name> arrive?` / `Lena: I saw <name> last May.` | insufficient | sufficient |
| `What did <name> buy?` / `Lena: I saw <name> near the harbour market.` | insufficient | sufficient |

- **Not reachable from the API:**
  - `heuristic_evaluation` is referenced only in `evidence_eval.py`, `iterative_diag.py` and `ranking_diag.py`.
  - It reaches `ask` only through `Workspace.set_evaluator`, which only those two diagnostics call.
  - `_resolve_evaluator` otherwise returns the configured generator or `None`, and `evaluation_from_generator` has no heuristic fallback.
  - Importing `webapi.server` loads neither diagnostic.
- **Decision:** documented, not changed, as the brief directs for evaluation-only code. `iterative_diag.py` ships because `locomo_report.py` imports its defaults, so any results from its heuristic mode are name-dependent. Removing the names would be a sensible separate cleanup.

### B′

| Check | Result |
|---|---|
| Missing configuration, API path | `ValueError` naming `CONTINUATOR_BPRIME_CLASSIFICATIONS`; `server.py` returns HTTP 400 for `ValueError` (read from the code, not sent over HTTP) |
| Missing configuration, `BPrimeConfig(enabled=True)` | `ValueError` naming `classifications_path` |
| Explicit path | Used as given; link expansion still returns the expected memory |
| API imports | No `benchmarks` module loaded, checked in a subprocess |
| Ranking and algorithms | Not changed |

### Files and tests

**Changed:** `src/continuator_memory/webapi/transcript.py`, `tests/unit/test_transcript.py`, `tests/unit/test_webapi.py`, this manifest. All were already listed.

**Added 16 test cases:**
- `test_transcript.py`, 12 cases: uppercase and lowercase paragraphs, ordinary paragraphs, heading and lowercase fallbacks, the pinned limitation, 5 header formats, JSON and single block.
- `test_webapi.py`, 4 cases: API imports, the iterative mode not using the heuristic, and missing and explicit B′ configuration.

The iterative-mode test patches the module attribute, so on its own it wouldn't catch a by-name import. The reference search above covers that case.

## 14. Documentation approval decisions

**Decided by the project owner:**

| Decision | Recorded in |
|---|---|
| Headerless transcript parsing is best-effort and can misattribute speakers. Explicit speaker headers or JSON are recommended when attribution matters | README, `memory_extraction_v1.md` §10 |
| The Caroline and Melanie logic in the heuristic evaluator is diagnostic-only and benchmark-specific, not part of normal API answer generation. The code stays as it is | README, `memory_extraction_v1.md` §10 |
| Extraction limitations stay documented, including temporary small-talk leakage, unsupported inferred details and English-only scope | README, `memory_extraction_v1.md` §10 |

**Outstanding owner decisions:**

| Decision | State | Must be settled before |
|---|---|---|
| License | MIT license added in `LICENSE` | Confirmed by project owner |
| Package version | `0.1.0` while tags use `v1.x`; unchanged (§11) | Tagging, if the two are meant to match |

The other unticked items in §10 have not been decided either. None prevents staging, because staging is reversible. They need settling before the commit or tag.

**Release file list:** 76 files (71 added, 5 modified). This step changed only `README.md`, `reports/memory_extraction_v1.md` and this manifest. Production code, tests, prompts and retrieval were not touched.
