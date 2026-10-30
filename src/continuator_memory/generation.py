"""Answer generation with a local Ollama model."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from continuator_memory.models import RetrievedChunk

SYSTEM_PROMPT = (
    "You answer questions about a past conversation using only the provided excerpts. "
    "If the excerpts do not contain the answer, say that you don't know. "
    "Cite the chunk IDs you used in square brackets, e.g. [conv:c0001]."
)


class GenerationError(RuntimeError):
    pass


class Generator(Protocol):
    model_name: str

    def generate(self, system: str, prompt: str) -> str: ...


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if chunks:
        blocks = [
            f"[{c.chunk_id}] (messages: {', '.join(c.source_message_ids)})\n{c.text}"
            for c in chunks
        ]
        context = "\n\n".join(blocks)
    else:
        context = "(no excerpts retrieved)"
    return f"Conversation excerpts:\n\n{context}\n\nQuestion: {question}\nAnswer:"


# Ollama truncates prompts longer than the context window without raising an error.
# Measured Phase 3.6 answer prompts: 676-850 tokens, completions at most 48 tokens.
ANSWER_NUM_CTX = 4096
_USAGE_KEYS = ("prompt_eval_count", "eval_count", "done_reason", "total_duration",
               "load_duration", "prompt_eval_duration", "eval_duration")


class OllamaGenerator:
    """Calls Ollama's /api/chat with temperature 0, a fixed seed and an explicit context window."""

    def __init__(
        self,
        model_name: str,
        host: str,
        timeout: float = 300.0,
        seed: int = 0,
        num_ctx: int = ANSWER_NUM_CTX,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.seed = seed
        self.num_ctx = num_ctx
        # Token counts Ollama reported for the most recent successful call (diagnostics only).
        self.last_usage: dict | None = None

    def generate(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        options: dict = {
            "temperature": 0.0 if temperature is None else temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
        }
        if num_predict is not None:
            options["num_predict"] = num_predict
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": options,
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GenerationError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise GenerationError(
                f"could not reach Ollama at {self.host} ({exc.reason}); is `ollama serve` running?"
            ) from exc
        try:
            text = body["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise GenerationError(f"unexpected Ollama response: {body!r}") from exc
        self.last_usage = {
            "num_ctx": self.num_ctx,
            "num_predict": num_predict,
            **{key: body[key] for key in _USAGE_KEYS if key in body},
        }
        return text
