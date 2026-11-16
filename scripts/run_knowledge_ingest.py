#!/usr/bin/env python3
"""Generate Markdown knowledge documents into a new scratch database. Writes only to a new, unique directory.

One of these is required, so nothing reaches a model by accident:
  --dry-run   a scripted client and embedder: no Ollama, no model download
  --live      the local Ollama model, one request at a time

  --smoke     chunks 0 and 1 only (the first 8 messages)

Exit status: 0 complete and clean; 3 complete with failed or truncated documents; 1 could not start; 130 interrupted
(earlier documents are kept).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

from continuator_memory.config import Settings
from continuator_memory.embedding import SentenceTransformerEmbedder
from continuator_memory.envfile import load_root_env
from continuator_memory.generation import OllamaGenerator
from continuator_memory.knowledge import documents as documents_module, ingest as ingest_module, store as store_module
from continuator_memory.knowledge.documents import FORMAT_VERSION, PROMPT_VERSION, SYSTEM_PROMPT
from continuator_memory.knowledge.ingest import CHUNK_SIZE, CONTEXT_MESSAGES, NUM_CTX, NUM_PREDICT, ingest_documents
from continuator_memory.models import Conversation, parse_conversation
from continuator_memory.store import Store

DEFAULT_INPUT = Path.home() / "Downloads/continuator_memory_test_conversation.json"
EXPECTED_INPUT_SHA256 = "23d165fd2e00c22349735dbfc37328a0f7b049a749cc9c149ff6c59ea8c30c5f"
MODEL = "qwen2.5:14b"
REQUEST_TIMEOUT_S = 600.0
SMOKE_MESSAGES = 8
DEFAULT_ROOT = Path.home() / "continuator_eval_artifacts/knowledge_eval"


class DryRunClient:
    """No network. Cites each target message once."""

    model_name = "dry-run"
    last_usage: dict | None = None

    def generate(self, system, prompt, *, temperature=None, num_predict=None):
        self.last_usage = {"prompt_eval_count": 0, "eval_count": 0, "done_reason": "stop"}
        cited = [line[1:].split("]")[0] for line in prompt.splitlines() if line.startswith("[M")]
        return "## Facts and preferences\n" + "\n".join(f"- A scripted statement about {c}. [{c}]" for c in cited)


class DryRunEmbedder:
    model_name = "dry-run-embedder"

    def embed(self, texts):
        rows = [np.frombuffer(hashlib.sha256(t.encode("utf-8")).digest(), dtype=np.uint8) for t in texts]
        return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 32), np.float32)


def fail(message: str) -> int:
    print(f"CANNOT START: {message}", file=sys.stderr, flush=True)
    return 1


def seconds(usage: dict, key: str) -> float | None:
    value = usage.get(key)
    return round(value / 1e9, 3) if value is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)

    if not args.input.is_file():
        return fail(f"input not found: {args.input}")
    data = args.input.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    conversation = parse_conversation(json.loads(data))
    if args.live:
        load_root_env()
        if digest != EXPECTED_INPUT_SHA256:
            return fail(f"input does not match the evaluated conversation (sha256 {digest})")
        if len(conversation.messages) != 62:
            return fail(f"expected 62 messages, found {len(conversation.messages)}")
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        try:
            with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as response:
                models = [m["name"] for m in json.loads(response.read())["models"]]
        except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
            return fail(f"Ollama is not reachable at {host}: {exc}")
        if MODEL not in models:
            return fail(f"{MODEL} is not installed in Ollama")
        client = OllamaGenerator(MODEL, host, timeout=REQUEST_TIMEOUT_S, num_ctx=NUM_CTX)
        embedder = SentenceTransformerEmbedder(Settings.from_env().embedding_model)
    else:
        client, embedder = DryRunClient(), DryRunEmbedder()
    if args.smoke:
        conversation = Conversation(conversation.conversation_id, conversation.messages[:SMOKE_MESSAGES], conversation.title)

    kind = ("dryrun" if args.dry_run else "live") + ("_smoke" if args.smoke else "_full")
    out = args.output_root / f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{kind}"
    out.mkdir(parents=True, exist_ok=False)  # A second run in the same second fails rather than overwrites.
    code = out / "code"
    code.mkdir()
    for module in (documents_module, ingest_module, store_module):
        shutil.copy2(module.__file__, code)
    shutil.copy2(__file__, code)
    (out / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (out / "config.json").write_text(json.dumps({
        "kind": kind, "started": datetime.now().isoformat(), "input": str(args.input), "input_sha256": digest,
        "messages": len(conversation.messages), "model": client.model_name, "embedding_model": embedder.model_name,
        "prompt_version": PROMPT_VERSION, "format_version": FORMAT_VERSION, "chunk_size": CHUNK_SIZE,
        "context_messages": CONTEXT_MESSAGES, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "concurrency": 1}, indent=1), encoding="utf-8")
    print(f"ARTIFACTS: {out}", flush=True)

    metrics = (out / "metrics.jsonl").open("w", encoding="utf-8")

    def on_document(chunk, prompt, row, passages) -> None:
        i = chunk.position
        usage = row.usage or {}
        (out / f"prompt_c{i:02d}.txt").write_text(prompt, encoding="utf-8")
        (out / f"document_c{i:02d}.md").write_text(row.content_markdown or "", encoding="utf-8")
        (out / f"claims_c{i:02d}.json").write_text(json.dumps(list(row.claims), indent=1, ensure_ascii=False), encoding="utf-8")
        review = sum(bool(c["review_reasons"]) for c in row.claims)
        metrics.write(json.dumps({
            "chunk": i, "chunk_id": chunk.chunk_id, "status": row.status, "error": row.error, "done_reason": usage.get("done_reason"),
            "input_tokens": usage.get("prompt_eval_count"), "output_tokens": usage.get("eval_count"),
            "load_s": seconds(usage, "load_duration"), "prompt_eval_s": seconds(usage, "prompt_eval_duration"),
            "generation_s": seconds(usage, "eval_duration"), "total_ollama_s": seconds(usage, "total_duration"),
            "wall_s": round(row.latency_ms / 1000, 3), "claims": len(row.claims), "claims_needing_review": review,
            "passages": len(passages), "issues": list(row.issues), "uncited_targets": list(row.uncited_target_ids)}, ensure_ascii=False) + "\n")
        metrics.flush()
        print(f"chunk {i:02d} | {row.status:15} | finish {str(usage.get('done_reason')):6} | out {usage.get('eval_count')} | "
              f"{round(row.latency_ms / 1000, 1)} s | claims {len(row.claims)} review {review} | passages {len(passages)}", flush=True)

    store = Store(str(out / "eval.db"))
    started = time.time()
    try:
        report = ingest_documents(conversation, store, client, embedder, on_document=on_document)
    except KeyboardInterrupt:
        print(f"STATUS: INTERRUPTED, earlier documents kept\nARTIFACTS: {out}", flush=True)
        return 130
    finally:
        metrics.close()
        store.close()
    wall = round(time.time() - started, 1)
    (out / "run.json").write_text(json.dumps({"wall_s": wall, "report": report.to_dict()}, indent=1), encoding="utf-8")
    print(f"documents {report.documents} | calls {report.model_calls} | ok {report.ok} | nothing durable {report.nothing_durable} | "
          f"truncated {report.truncated} | failed {report.failed} | claims {report.claims} | review {report.claims_needing_review} | "
          f"passages {report.passages} | input {report.prompt_tokens} | output {report.output_tokens} | wall {wall} s", flush=True)
    problems = report.failed + report.truncated
    print(f"STATUS: {'OK' if not problems else f'COMPLETE WITH PROBLEMS: {problems} documents failed or truncated'}", flush=True)
    print(f"ARTIFACTS: {out}", flush=True)
    return 0 if not problems else 3


if __name__ == "__main__":
    raise SystemExit(main())
