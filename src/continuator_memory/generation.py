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


class OllamaGenerator:
    """Calls Ollama's /api/chat with temperature 0 and a fixed seed."""

    def __init__(self, model_name: str, host: str, timeout: float = 300.0, seed: int = 0):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.seed = seed

    def generate(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "seed": self.seed},
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
            return body["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise GenerationError(f"unexpected Ollama response: {body!r}") from exc
