"""Semantic atomizer: one LLM call per context window, followed by deterministic validation."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from continuator_memory.v2.models import Claim
from continuator_memory.v2.memory_prompt import (
    MEMORY_EXAMPLES,
    MEMORY_PROTOCOL_VERSION,
    MEMORY_SYSTEM_PROMPT,
    build_memory_prompt,
    memory_output_schema,
)
from continuator_memory.v2.prompt import EXAMPLES, SYSTEM_PROMPT, build_prompt, output_schema
from continuator_memory.v2.validation import ValidatedWindow, failed_window, validate_atomization
from continuator_memory.v2.windowing import DEFAULT_CONTEXT_BEFORE, DEFAULT_TARGETS_PER_WINDOW, ContextWindow

DEFAULT_ATOMIZER_MODEL = "qwen2.5:14b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
# Values used by the V2 extraction experiments (runs 15-21) with qwen2.5:14b.
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 2048
DEFAULT_PRIOR_LIMIT = 20


@dataclass(frozen=True)
class ExtractionProtocol:
    """Prompt, schema and examples for one extraction instruction; windows and validation are shared."""
    name: str
    system: str
    build_prompt: Callable[..., str]
    output_schema: Callable[[ContextWindow], dict[str, Any]]
    examples: tuple[tuple[str, str], ...] = ()


# The semantic atomizer: every proposition of every target message (unchanged default).
ATOMIZATION_PROTOCOL = ExtractionProtocol("atomization-v2", SYSTEM_PROMPT, build_prompt, output_schema, EXAMPLES)
# V1 memory extraction: durable, independently useful memories with a memory type.
MEMORY_PROTOCOL = ExtractionProtocol(MEMORY_PROTOCOL_VERSION, MEMORY_SYSTEM_PROMPT, build_memory_prompt,
                                     memory_output_schema, MEMORY_EXAMPLES)


class AtomizationRequestError(RuntimeError):
    """The atomizer model could not be reached or returned no content."""


class JSONCompletionClient(Protocol):
    model_name: str

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any],
                      examples: tuple[tuple[str, str], ...] = ()) -> str: ...


@dataclass(frozen=True)
class V2Settings:
    atomizer_model: str = DEFAULT_ATOMIZER_MODEL
    ollama_host: str = DEFAULT_OLLAMA_HOST
    num_ctx: int = DEFAULT_NUM_CTX
    num_predict: int = DEFAULT_NUM_PREDICT
    targets_per_window: int = DEFAULT_TARGETS_PER_WINDOW
    context_before: int = DEFAULT_CONTEXT_BEFORE
    prior_limit: int = DEFAULT_PRIOR_LIMIT

    @classmethod
    def from_env(cls) -> V2Settings:
        env = os.environ
        return cls(
            atomizer_model=env.get("CONTINUATOR_V2_ATOMIZER_MODEL", DEFAULT_ATOMIZER_MODEL),
            ollama_host=env.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            num_ctx=int(env.get("CONTINUATOR_V2_NUM_CTX", DEFAULT_NUM_CTX)),
            num_predict=int(env.get("CONTINUATOR_V2_NUM_PREDICT", DEFAULT_NUM_PREDICT)),
            targets_per_window=int(env.get("CONTINUATOR_V2_TARGETS_PER_WINDOW", DEFAULT_TARGETS_PER_WINDOW)),
            context_before=int(env.get("CONTINUATOR_V2_CONTEXT_BEFORE", DEFAULT_CONTEXT_BEFORE)),
            prior_limit=int(env.get("CONTINUATOR_V2_PRIOR_LIMIT", DEFAULT_PRIOR_LIMIT)),
        )


class OllamaChatClient:
    """Ollama /api/chat with a JSON schema, temperature 0, a fixed seed and configurable limits."""

    def __init__(self, model_name: str, host: str, num_ctx: int = DEFAULT_NUM_CTX,
                 num_predict: int = DEFAULT_NUM_PREDICT, seed: int = 0, timeout: float = 600.0):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.seed = seed
        self.timeout = timeout
        self.last_usage: dict[str, Any] | None = None  # Ollama token counts of the last successful call

    def payload(self, system: str, prompt: str, schema: dict[str, Any],
                examples: tuple[tuple[str, str], ...] = ()) -> dict[str, Any]:
        messages = [{"role": "system", "content": system}]
        for example_prompt, example_output in examples:
            messages += [{"role": "user", "content": example_prompt},
                         {"role": "assistant", "content": example_output}]
        messages.append({"role": "user", "content": prompt})
        return {"model": self.model_name, "messages": messages, "stream": False, "format": schema,
                "options": {"temperature": 0, "seed": self.seed, "num_predict": self.num_predict,
                            "num_ctx": self.num_ctx}}

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any],
                      examples: tuple[tuple[str, str], ...] = ()) -> str:
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(self.payload(system, prompt, schema, examples)).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["message"]["content"]
            self.last_usage = {key: body.get(key) for key in ("prompt_eval_count", "eval_count", "done_reason")}
            return content
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise AtomizationRequestError(f"atomization request failed: {exc}") from exc


@dataclass
class AtomizationResult:
    validated: ValidatedWindow
    prompt: str
    latency_ms: float
    usage: dict[str, Any] | None = None  # model token counts, when the client reports them


class SemanticAtomizer:
    def __init__(self, client: JSONCompletionClient, protocol: ExtractionProtocol = ATOMIZATION_PROTOCOL):
        self.client = client
        self.protocol = protocol
        self.model_name = client.model_name

    def atomize(self, window: ContextWindow, prior: Sequence[Claim] = ()) -> AtomizationResult:
        p = self.protocol
        prompt = p.build_prompt(window, prior)
        if hasattr(self.client, "last_usage"):
            self.client.last_usage = None
        t0 = time.perf_counter()
        try:
            raw = self.client.complete_json(p.system, prompt, p.output_schema(window), p.examples)
        except AtomizationRequestError as exc:
            validated = failed_window(window, None, str(exc))
        else:
            validated = validate_atomization(raw, window, prior)
        return AtomizationResult(validated, prompt, round((time.perf_counter() - t0) * 1000, 3),
                                 getattr(self.client, "last_usage", None))


def default_atomizer(settings: V2Settings | None = None) -> SemanticAtomizer:
    s = settings or V2Settings.from_env()
    return SemanticAtomizer(OllamaChatClient(s.atomizer_model, s.ollama_host, s.num_ctx, s.num_predict))


def memory_atomizer(settings: V2Settings | None = None) -> SemanticAtomizer:
    """V1 memory extraction with the same model client, windows, validation and consolidation."""
    s = settings or V2Settings.from_env()
    return SemanticAtomizer(OllamaChatClient(s.atomizer_model, s.ollama_host, s.num_ctx, s.num_predict),
                            MEMORY_PROTOCOL)
