"""Neutral-shuffle control for the graph-hybrid answer evaluation.

Question: when only the ORDER of the baseline's own top-25 evidence changes, how often does the
answer label change? Compared with the baseline -> graph-hybrid answer changes on the same questions
(12 better / 6 worse in runs/graph_rag/answer_eval/answer_eval_report.md), this estimates how much
of that change is context-order sensitivity rather than graph retrieval.

Inputs (read only):  runs/graph_rag/answer_eval/{records.jsonl, scores.json, sample_manifest.json}
                     data/locomo-convXX.db (copied; the original is never opened)
Outputs:             runs/graph_rag/neutral_shuffle/

Eligible questions: baseline and graph_hybrid have the same NUMBER of gold turns in the prompt
(``--definition count``, the 89-question group of the answer-eval report; default) or the same SET
(``--definition set``, 88: conv-49 Q-012 has equal counts but different turns). Every record carries
``sameGoldSet`` so either analysis is possible from one run.

Per question:
  1. Rebuild the saved baseline prompt with the production ``build_answer_context`` from the saved
     top-25 memories and excerpt order; it must equal the saved prompt byte for byte.
  2. Shuffle the excerpt order with ``random.Random(f"{SHUFFLE_SEED}|{sampleId}|{questionId}")``.
     If the permutation equals the original, it is rotated by one (documented, deterministic).
  3. Rebuild the prompt in the shuffled order; it must contain exactly the same excerpt blocks, the
     same header and the same question. Nothing else changes: evidence content, answer prompt,
     system prompt, model, generation settings and token budget are the saved ones.
  4. ``run`` generates one answer per question with the production ``generate_answer``.

Subcommands:
  dry-run           validate the sample, rebuild and shuffle every prompt, print the plan; no model calls
  run               generate missing answers (resumable; completed questions are never called again)
  labels-template   write labels_template.json: answers identical to the saved baseline answer inherit
                    its label; the rest must be labelled (C / P / W, optional ":abs" ":fp" ":temp")
  compare           compare labels.json against the saved baseline labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ANSWER_EVAL = Path("runs/graph_rag/answer_eval")
OUT = Path("runs/graph_rag/neutral_shuffle")
SHUFFLE_SEED = 20261101
EXPECTED_MODEL = "qwen2.5:7b"
_BLOCK = re.compile(r"^\[([^\]]+)\] .*?(?=\n\n\[|\n\nQuestion:)", re.S | re.M)


# ---- pure helpers (unit-tested) -------------------------------------------------------------------

def shuffle_order(order: list[str], sample_id: str, question_id: str, seed: int = SHUFFLE_SEED) -> list[str]:
    """Deterministic permutation of ``order``; never the identity when there are at least two items."""
    out = list(order)
    random.Random(f"{seed}|{sample_id}|{question_id}").shuffle(out)
    if len(out) > 1 and out == list(order):
        out = out[1:] + out[:1]
    return out


def prompt_blocks(prompt: str) -> list[str]:
    """Excerpt blocks of a rendered answer prompt, in prompt order."""
    return [m.group(0) for m in _BLOCK.finditer(prompt)]


def prompt_frame(prompt: str) -> tuple[str, str]:
    """(header before the first excerpt, tail from the question on): everything but the excerpts."""
    blocks = list(_BLOCK.finditer(prompt))
    if not blocks:
        return prompt, ""
    return prompt[: blocks[0].start()], prompt[blocks[-1].end():]


def same_content_different_order(original: str, shuffled: str) -> bool:
    return (sorted(prompt_blocks(original)) == sorted(prompt_blocks(shuffled))
            and prompt_frame(original) == prompt_frame(shuffled))


def gold_in_prompt(record: dict) -> list[str]:
    return [g for g in record["evaluationOnly"]["goldEvidence"] if g in record["promptMessageIds"]]


def eligible(records: dict, definition: str = "count") -> list[tuple[str, str]]:
    keys = sorted({(s, q) for s, q, _ in records})
    out = []
    for key in keys:
        base, graph = gold_in_prompt(records[(*key, "hybrid")]), gold_in_prompt(records[(*key, "graph_hybrid")])
        if (definition == "count" and len(base) == len(graph)) or (definition == "set" and set(base) == set(graph)):
            out.append(key)
    return out


def completed(answers_path: Path) -> tuple[set[tuple[str, str]], dict[tuple[str, str], int]]:
    """Questions with an ``ok`` answer, and the number of attempts per question."""
    done, attempts = set(), {}
    if answers_path.exists():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["sampleId"], r["questionId"])
            attempts[key] = attempts.get(key, 0) + 1
            if r["status"] == "ok":
                done.add(key)
    return done, attempts


# ---- loading ---------------------------------------------------------------------------------------

def load_records() -> dict:
    recs = {}
    for line in (ANSWER_EVAL / "records.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        recs[(r["sampleId"], r["questionId"], r["condition_mode"])] = r
    return recs


def saved_labels() -> dict[tuple[str, str], dict]:
    rows = json.loads((ANSWER_EVAL / "scores.json").read_text())["rows"]
    return {(r["sampleId"], r["questionId"]): {"hybrid": r["hybrid"], "graph_hybrid": r["graph_hybrid"]} for r in rows}


def _conversation(sample_id: str):
    from continuator_memory.webapi.workspace import Workspace

    src = Path(f"data/locomo-{sample_id.replace('-', '')}.db")
    db = OUT / "db" / src.name
    db.parent.mkdir(parents=True, exist_ok=True)
    if not db.exists():
        shutil.copy(src, db)
    ws = Workspace.from_sqlite(db, k=25)  # no retrieval index and no embeddings are built
    return ws


def _settings():
    from continuator_memory.webapi.answering import load_generation_settings, parse_generation_overrides

    return parse_generation_overrides(None, load_generation_settings())


def rebuild(record: dict, conversation, order: list[str], settings):
    """The saved baseline evidence rendered by the production context builder, in ``order``."""
    from continuator_memory.webapi.answering import build_answer_context

    top = [e for e in record["retrieved"] if e["rank"] is not None and e["rank"] <= 25]
    matches = [SimpleNamespace(memory_id=e["memoryId"], source_message_ids=tuple(e["sourceMessageIds"]),
                               statement="", rank=e["rank"], score=e["score"], similarity=e["similarity"])
               for e in sorted(top, key=lambda e: e["rank"])]
    evidence = SimpleNamespace(selection=SimpleNamespace(selected=tuple(order), dropped=()), matches=matches)
    return build_answer_context(record["question"], evidence, conversation, seed_ids=[], expanded_ids=[],
                                max_context_chars=settings.max_context_chars,
                                context_tokens=settings.num_ctx, output_tokens=settings.num_predict)


def plan(definition: str) -> tuple[list[dict], list[str]]:
    """Validate everything without model calls. Returns (items, problems)."""
    records = load_records()
    labels = saved_labels()
    settings = _settings()
    problems = []
    if settings.model != EXPECTED_MODEL:
        problems.append(f"generation model is {settings.model!r}; set CONTINUATOR_OLLAMA_MODEL={EXPECTED_MODEL}")
    keys = eligible(records, definition)
    conversations = {}
    items = []
    for sid, qid in keys:
        rec = records[(sid, qid, "hybrid")]
        usage = rec["generation"]["usage"] or {}
        if rec["generation"]["model"] != settings.model or usage.get("num_ctx") != settings.num_ctx \
                or usage.get("num_predict") != settings.num_predict:
            problems.append(f"{sid}/{qid}: saved generation settings differ from the current ones")
        if sid not in conversations:
            conversations[sid] = _conversation(sid).conversation
        original = list(rec["promptMessageIds"])
        ctx = rebuild(rec, conversations[sid], original, settings)
        if ctx.prompt != rec["prompt"]:
            problems.append(f"{sid}/{qid}: rebuilt prompt differs from the saved baseline prompt")
        if ctx.truncated:
            problems.append(f"{sid}/{qid}: original prompt would be trimmed")
        order = shuffle_order(original, sid, qid)
        sctx = rebuild(rec, conversations[sid], order, settings)
        if [e.message_id for e in sctx.excerpts] != order:
            problems.append(f"{sid}/{qid}: shuffled prompt does not follow the shuffled order")
        if not same_content_different_order(rec["prompt"], sctx.prompt):
            problems.append(f"{sid}/{qid}: shuffled prompt changed content, header or question")
        if sctx.truncated:
            problems.append(f"{sid}/{qid}: shuffled prompt would be trimmed")
        base_gold, graph_gold = gold_in_prompt(rec), gold_in_prompt(records[(sid, qid, "graph_hybrid")])
        items.append({"sampleId": sid, "questionId": qid, "stratum": rec["stratum"],
                      "sameGoldSet": set(base_gold) == set(graph_gold),
                      "originalOrder": original, "shuffledOrder": order,
                      "goldPositionsOriginal": [original.index(g) + 1 for g in base_gold],
                      "goldPositionsShuffled": [order.index(g) + 1 for g in base_gold],
                      "promptSha256": hashlib.sha256(sctx.prompt.encode()).hexdigest(),
                      "savedBaselineLabel": labels[(sid, qid)]["hybrid"]["label"],
                      "savedGraphHybridLabel": labels[(sid, qid)]["graph_hybrid"]["label"]})
    return items, problems


# ---- commands --------------------------------------------------------------------------------------

def _config(settings) -> dict:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/version", timeout=5) as resp:
            ollama = json.loads(resp.read().decode())["version"]
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal for the dry run
        ollama = f"unreachable: {exc}"
    return {"shuffleSeed": SHUFFLE_SEED, "generation": {**settings.public_dict(), "numCtx": settings.num_ctx,
                                                         "seed": 0, "temperature": settings.temperature},
            "ollamaVersion": ollama, "source": str(ANSWER_EVAL)}


def cmd_dry_run(args) -> int:
    items, problems = plan(args.definition)
    settings = _settings()
    done, attempts = completed(OUT / "answers.jsonl")
    todo = [i for i in items if (i["sampleId"], i["questionId"]) not in done]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan.json").write_text(json.dumps({"definition": args.definition, "config": _config(settings),
                                               "n": len(items), "items": items}, indent=1))
    by_stratum = {}
    for i in items:
        by_stratum[i["stratum"]] = by_stratum.get(i["stratum"], 0) + 1
    print(f"definition={args.definition} eligible={len(items)} by stratum={by_stratum}")
    differing = [i["sampleId"] + "/" + i["questionId"] for i in items if not i["sameGoldSet"]]
    print(f"sameGoldSet false: {differing}")
    print(f"saved baseline labels: { {l: sum(1 for i in items if i['savedBaselineLabel'] == l) for l in 'CPW'} }")
    print(f"already completed: {len(done)}  planned model calls now: {len(todo)}")
    print(f"generation: model={settings.model} temperature={settings.temperature} num_predict={settings.num_predict} "
          f"num_ctx={settings.num_ctx}  shuffle seed={SHUFFLE_SEED}")
    print(f"plan written to {OUT / 'plan.json'}")
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems:
            print("  -", p)
        return 1
    print("VALIDATION OK: every rebuilt baseline prompt equals the saved one; every shuffled prompt has the "
          "same excerpts, header and question in a different order")
    return 0


def cmd_run(args) -> int:
    from continuator_memory.webapi.answering import generate_answer

    items, problems = plan(args.definition)
    if problems:
        print("refusing to run; dry-run problems:", *problems, sep="\n  - ")
        return 1
    settings = _settings()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_config.json").write_text(json.dumps(_config(settings), indent=1))
    answers_path = OUT / "answers.jsonl"
    done, attempts = completed(answers_path)
    records = load_records()
    workspaces = {}
    calls = 0
    for item in items:
        key = (item["sampleId"], item["questionId"])
        if key in done:
            continue
        if attempts.get(key, 0) >= args.max_attempts:
            print(f"skip {key}: {attempts[key]} failed attempts (raise --max-attempts to retry)")
            continue
        sid, qid = key
        if sid not in workspaces:
            workspaces[sid] = _conversation(sid)
        ws = workspaces[sid]
        rec = records[(sid, qid, "hybrid")]
        ctx = rebuild(rec, ws.conversation, item["shuffledOrder"], settings)
        generator = ws._resolve_generator(settings)
        if generator is not None and hasattr(generator, "last_usage"):
            generator.last_usage = None
        started = time.perf_counter()
        try:
            answer, meta = generate_answer(ctx, settings, generator)
            error = meta.get("error")
            status = meta.get("status")
        except Exception as exc:  # noqa: BLE001 - recorded and retried on the next run
            answer, meta, error, status = "", {}, f"{type(exc).__name__}: {exc}", "exception"
        calls += 1
        row = {"sampleId": sid, "questionId": qid, "stratum": item["stratum"], "sameGoldSet": item["sameGoldSet"],
               "status": status, "error": error, "rawAnswer": meta.get("rawAnswer"), "finalAnswer": answer,
               "usage": getattr(generator, "last_usage", None), "generationMs": round((time.perf_counter() - started) * 1000, 1),
               "model": meta.get("model", settings.model), "shuffleSeed": SHUFFLE_SEED,
               "originalOrder": item["originalOrder"], "shuffledOrder": item["shuffledOrder"],
               "promptSha256": hashlib.sha256(ctx.prompt.encode()).hexdigest(), "prompt": ctx.prompt,
               "savedBaselineFinalAnswer": rec["finalAnswer"], "savedBaselineLabel": item["savedBaselineLabel"],
               "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with answers_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{len(done) + calls}/{len(items)}] {sid} {qid} {status} {(answer or '')[:70]!r}", flush=True)
    done, _ = completed(answers_path)
    missing = [f"{i['sampleId']}/{i['questionId']}" for i in items if (i["sampleId"], i["questionId"]) not in done]
    print(f"model calls this run: {calls}; completed {len(done)}/{len(items)}; missing: {missing or 'none'}")
    if not missing:
        (OUT / "COMPLETE").write_text(f"{len(done)} answers, definition={args.definition}\n")
        print(f"COMPLETE: all {len(items)} answers saved to {answers_path}")
        return 0
    return 2


def cmd_labels_template(args) -> int:
    items, _ = plan(args.definition)
    answers = {}
    for line in (OUT / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] == "ok":
            answers[(r["sampleId"], r["questionId"])] = r
    template = {}
    for item in items:
        key = (item["sampleId"], item["questionId"])
        r = answers.get(key)
        if r is None:
            continue
        identical = (r["finalAnswer"] or "").strip() == (r["savedBaselineFinalAnswer"] or "").strip()
        template[f"{key[0]}/{key[1]}"] = {"label": item["savedBaselineLabel"] if identical else None,
                                          "inherited": identical, "answer": r["finalAnswer"]}
    (OUT / "labels_template.json").write_text(json.dumps(template, indent=1, ensure_ascii=False))
    print(f"{len(template)} answers; {sum(v['inherited'] for v in template.values())} identical to the saved baseline "
          f"answer (label inherited); {sum(not v['inherited'] for v in template.values())} need a label")
    return 0


def _points(label: str) -> int:
    return {"C": 2, "P": 1, "W": 0}[label.split(":")[0]]


def cmd_compare(args) -> int:
    items, _ = plan(args.definition)
    labels = json.loads((OUT / "labels.json").read_text())
    saved = saved_labels()
    rows = []
    for item in items:
        k = f"{item['sampleId']}/{item['questionId']}"
        if k not in labels or not labels[k].get("label"):
            print("unlabelled:", k)
            return 1
        base = saved[(item["sampleId"], item["questionId"])]
        rows.append({"q": k, "sameGoldSet": item["sameGoldSet"], "shuffle": _points(labels[k]["label"]),
                     "base": base["hybrid"]["points"], "graph": base["graph_hybrid"]["points"]})
    out = {}
    for name, sub in (("all", rows), ("sameGoldSet", [r for r in rows if r["sameGoldSet"]])):
        def flips(a, b):
            better = sum(1 for r in sub if r[a] > r[b])
            worse = sum(1 for r in sub if r[a] < r[b])
            n = better + worse
            p = None if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(better, worse) + 1)) / 2 ** n)
            return {"better": better, "worse": worse, "same": len(sub) - n, "signP": p}
        out[name] = {"n": len(sub), "shuffleVsBaseline": flips("shuffle", "base"),
                     "graphVsBaseline": flips("graph", "base")}
    (OUT / "comparison.json").write_text(json.dumps({"rows": rows, **out}, indent=1))
    print(json.dumps(out, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.experiments.graph_rag.neutral_shuffle")
    parser.add_argument("command", choices=("dry-run", "run", "labels-template", "compare"))
    parser.add_argument("--definition", choices=("count", "set"), default="count")
    parser.add_argument("--max-attempts", type=int, default=2, help="attempts per question before it is skipped")
    args = parser.parse_args(argv)
    return {"dry-run": cmd_dry_run, "run": cmd_run, "labels-template": cmd_labels_template,
            "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
