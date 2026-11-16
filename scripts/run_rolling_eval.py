#!/usr/bin/env python3
"""Run the rolling-summary extraction experiment. Always writes to a new, uniquely named directory.

  --smoke     windows 0 and 1 only (the first 8 messages)
  --dry-run   a scripted client, with no Ollama call; checks that the runner and its files work

Exit status: 0 complete and clean; 3 complete, but with failed or truncated windows, unaccounted targets or
summary problems; 1 could not start; 130 interrupted (partial results are kept).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

from continuator_memory.envfile import load_root_env
from continuator_memory.models import Conversation, parse_conversation
from continuator_memory.store import Store
from continuator_memory.v2 import structured_rolling, structured_store, structured_targets
from continuator_memory.v2.atomizer import AtomizationRequestError, OllamaChatClient, V2Settings
from continuator_memory.v2.structured_rolling import (
    DEFAULT_MAX_SUMMARY_TOKENS,
    ROLLING_EXAMPLES,
    ROLLING_PROTOCOL_VERSION,
    ROLLING_SYSTEM_PROMPT,
    RollingExtractor,
    estimate_tokens,
    ingest_rolling,
    rolling_extractor,
)

DEFAULT_INPUT = Path.home() / "Downloads/continuator_memory_test_conversation.json"
EXPECTED_INPUT_SHA256 = "23d165fd2e00c22349735dbfc37328a0f7b049a749cc9c149ff6c59ea8c30c5f"
EXPECTED_MODEL = "qwen2.5:14b"
DEFAULT_ROOT = Path.home() / "continuator_eval_artifacts/rolling_eval"
SMOKE_MESSAGES = 8


class MeteredClient(OllamaChatClient):
    """The same request as ``OllamaChatClient``, also keeping Ollama's timing fields for the runner."""

    last_row: dict | None = None

    def complete_json(self, system, prompt, schema, examples=()):
        payload = json.dumps(self.payload(system, prompt, schema, examples)).encode("utf-8")
        request = urllib.request.Request(f"{self.host}/api/chat", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        started = time.time()
        self.last_row = {}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            self.last_usage = {key: body.get(key) for key in ("prompt_eval_count", "eval_count", "done_reason")}
            self.last_row = {key: body.get(key) for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration", "prompt_eval_count", "eval_count", "done_reason", "model")}
            return body["message"]["content"]
        except urllib.error.HTTPError as exc:
            self.last_row = {"error": f"HTTP {exc.code}: {exc.read()[:500]!r}"}
            raise AtomizationRequestError(self.last_row["error"]) from exc
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.last_row = {"error": repr(exc)}
            raise AtomizationRequestError(f"atomization request failed: {exc}") from exc
        finally:
            self.last_row["wall_s"] = round(time.time() - started, 3)


class DryRunClient:
    """No network. Every target is nothing durable, with a numbered summary."""

    model_name = "dry-run"
    last_usage = None
    last_row = None
    calls = 0

    def complete_json(self, system, prompt, schema, examples=()):
        slots = schema["properties"]["targets"]["required"]
        self.calls += 1
        self.last_usage = {"prompt_eval_count": 0, "eval_count": 0, "done_reason": "stop"}
        self.last_row = {"wall_s": 0.0, "done_reason": "stop", "prompt_eval_count": 0, "eval_count": 0, "model": "dry-run"}
        return json.dumps({"entities": [], "targets": {k: {"outcome": "nothing_durable"} for k in slots}, "rolling_summary": f"Dry-run summary after call {self.calls}."})


def fail(message: str) -> int:
    print(f"CANNOT START: {message}", file=sys.stderr, flush=True)
    return 1


def seconds(row: dict, key: str) -> float | None:
    value = row.get(key)
    return round(value / 1e9, 3) if value is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)

    load_root_env()
    if not args.input.is_file():
        return fail(f"input not found: {args.input}")
    data = args.input.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    conversation = parse_conversation(json.loads(data))
    if not args.dry_run:
        if digest != EXPECTED_INPUT_SHA256:
            return fail(f"input does not match the evaluated conversation (sha256 {digest})")
        if len(conversation.messages) != 62:
            return fail(f"expected 62 messages, found {len(conversation.messages)}")
    if args.smoke:
        conversation = Conversation(conversation.conversation_id, conversation.messages[:SMOKE_MESSAGES], conversation.title)

    settings = V2Settings.from_env()
    if args.dry_run:
        client = DryRunClient()
        extractor = RollingExtractor(client)
    else:
        if settings.atomizer_model != EXPECTED_MODEL:
            return fail(f"model is {settings.atomizer_model}, expected {EXPECTED_MODEL}")
        base = rolling_extractor(settings).client
        client = MeteredClient(base.model_name, base.host, base.num_ctx, base.num_predict)
        extractor = RollingExtractor(client)
        try:
            with urllib.request.urlopen(f"{settings.ollama_host.rstrip('/')}/api/tags", timeout=5) as response:
                models = [m["name"] for m in json.loads(response.read())["models"]]
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            return fail(f"Ollama is not reachable at {settings.ollama_host}: {exc}")
        if EXPECTED_MODEL not in models:
            return fail(f"{EXPECTED_MODEL} is not installed in Ollama")

    kind = "dryrun" if args.dry_run else "smoke" if args.smoke else "full"
    out = args.output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{kind}"
    out.mkdir(parents=True, exist_ok=False)  # A second run in the same second fails rather than overwrites.
    code = out / "code"
    code.mkdir()
    for module in (structured_rolling, structured_targets, structured_store):
        shutil.copy2(module.__file__, code)
    shutil.copy2(__file__, code)
    (out / "system_prompt.txt").write_text(ROLLING_SYSTEM_PROMPT, encoding="utf-8")
    (out / "examples.json").write_text(json.dumps(ROLLING_EXAMPLES, indent=1), encoding="utf-8")
    config = {"protocol": ROLLING_PROTOCOL_VERSION, "kind": kind, "started": datetime.now().isoformat(), "input": str(args.input),
              "input_sha256": digest, "messages": len(conversation.messages), "model": client.model_name,
              "settings": dataclasses.asdict(settings), "num_ctx": getattr(client, "num_ctx", None), "num_predict": getattr(client, "num_predict", None),
              "max_summary_tokens": DEFAULT_MAX_SUMMARY_TOKENS, "summary_size_is_estimate": "characters / 4", "concurrency": 1}
    (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    print(f"ARTIFACTS: {out}", flush=True)

    metrics = (out / "metrics.jsonl").open("w", encoding="utf-8")

    def on_window(window, result) -> None:
        i, validated, row = window.index, result.validated, dict(getattr(client, "last_row", None) or {})
        (out / f"prompt_w{i:02d}.txt").write_text(result.prompt, encoding="utf-8")
        (out / f"schema_w{i:02d}.json").write_text(json.dumps(structured_rolling.rolling_output_schema(window), indent=1), encoding="utf-8")
        (out / f"raw_w{i:02d}.txt").write_text(validated.raw or "", encoding="utf-8")
        (out / f"normalized_w{i:02d}.json").write_text(json.dumps(validated.normalized, indent=1, ensure_ascii=False), encoding="utf-8")
        (out / f"summary_w{i:02d}.txt").write_text(result.summary_out, encoding="utf-8")
        eval_s = seconds(row, "eval_duration")
        rejected = [x.to_dict() for x in validated.issues if x.action == "rejected"]
        line = {"window": i, "targets": list(window.target_ids), "context": list(window.context_ids),
                "input_tokens": row.get("prompt_eval_count"), "output_tokens": row.get("eval_count"),
                "summary_in_tokens_est": estimate_tokens(result.summary_in), "summary_in_chars": len(result.summary_in),
                "summary_out_tokens_est": estimate_tokens(result.summary_out), "summary_out_chars": len(result.summary_out),
                "summary_status": result.summary_status, "load_s": seconds(row, "load_duration"),
                "prompt_eval_s": seconds(row, "prompt_eval_duration"), "generation_s": eval_s, "total_ollama_s": seconds(row, "total_duration"),
                "wall_s": row.get("wall_s"), "tokens_per_s": round(row["eval_count"] / eval_s, 2) if eval_s and row.get("eval_count") else None,
                "done_reason": row.get("done_reason"), "memories": len(validated.memories), "entities": len(validated.entities),
                "relationships": len(validated.relationships), "outcomes": dict(Counter(o.outcome for o in validated.outcomes)),
                "needs_review": sum(r.needs_review for r in (*validated.memories, *validated.relationships)),
                "rejected": rejected, "issues": [x.to_dict() for x in validated.issues], "status": validated.status, "error": validated.error or row.get("error")}
        metrics.write(json.dumps(line, ensure_ascii=False) + "\n")
        metrics.flush()
        print(f"window {i:02d} | {validated.status:7} | finish {str(row.get('done_reason')):6} | out {row.get('eval_count')} | "
              f"{row.get('wall_s')} s | memories {len(validated.memories)} rel {len(validated.relationships)} | "
              f"summary {result.summary_status} ~{estimate_tokens(result.summary_out)} tok", flush=True)

    store = Store(str(out / "eval.db"))
    started = time.time()
    try:
        report = ingest_rolling(conversation, store, extractor, settings.targets_per_window, settings.context_before, settings.prior_limit, on_window=on_window)
    except KeyboardInterrupt:
        print(f"STATUS: INTERRUPTED, partial results kept\nARTIFACTS: {out}", flush=True)
        return 130
    finally:
        metrics.close()
        store.close()
    wall = round(time.time() - started, 1)
    problems = []
    if report.failed_windows:
        problems.append(f"{report.failed_windows} failed windows")
    if report.truncated_windows:
        problems.append(f"{report.truncated_windows} truncated windows")
    if report.unaccounted_targets:
        problems.append(f"{len(report.unaccounted_targets)} unaccounted targets")
    summary_problems = [s for s in report.summary_statuses if s != "updated"]
    if summary_problems:
        problems.append(f"summary not updated cleanly in {len(summary_problems)} windows")
    (out / "run.json").write_text(json.dumps({"wall_s": wall, "report": report.to_dict(), "problems": problems}, indent=1), encoding="utf-8")
    print(f"windows {report.windows} | calls {report.model_calls} | memories {report.memories} | relationships {report.relationships} | "
          f"entities {report.entities} | input {report.prompt_tokens} | output {report.output_tokens} | wall {wall} s", flush=True)
    if args.smoke:
        first = (out / "summary_w00.txt").read_text(encoding="utf-8")
        second = (out / "summary_w01.txt").read_text(encoding="utf-8")
        checks = {"summary 0 exists": bool(first), "window 0 prompt says no summary yet": "(none yet" in (out / "prompt_w00.txt").read_text(encoding="utf-8"),
                  "window 1 prompt contains summary 0": bool(first) and first in (out / "prompt_w01.txt").read_text(encoding="utf-8"),
                  "summary 1 differs from summary 0": bool(second) and second != first, "exactly 2 model calls": report.model_calls == 2}
        for name, passed in checks.items():
            print(f"CHECK {'PASS' if passed else 'FAIL'}: {name}", flush=True)
            if not passed:
                problems.append(name)
    print(f"STATUS: {'OK' if not problems else 'COMPLETE WITH PROBLEMS: ' + '; '.join(problems)}", flush=True)
    print(f"ARTIFACTS: {out}", flush=True)
    return 0 if not problems else 3


if __name__ == "__main__":
    raise SystemExit(main())
