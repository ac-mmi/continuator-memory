from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONVERSATION = ROOT / "benchmarks" / "synthetic" / "trip_planning.json"
EXAMPLE_QUESTIONS = ROOT / "benchmarks" / "synthetic" / "trip_planning_questions.json"
PHASE2_BENCHMARK = ROOT / "benchmarks" / "synthetic" / "phase2_benchmark.json"


class ScriptedJSONClient:
    """Stands in for Ollama: returns a fixed JSON document per message content."""

    model_name = "scripted-extractor"

    def __init__(self, by_content: dict[str, object], default: object | None = None):
        self.by_content = by_content
        self.default = {"memories": [], "relationships": []} if default is None else default
        self.prompts: list[str] = []

    def complete_json(self, system: str, prompt: str, schema: dict, examples=()) -> str:
        import json

        self.prompts.append(prompt)
        message = prompt.split("\nMESSAGE (", 1)[1].split("):\n", 1)[1].rsplit("\n\nJSON:", 1)[0]
        doc = self.by_content.get(message, self.default)
        return doc if isinstance(doc, str) else json.dumps(doc)


def mem(subject, predicate, value, kind="state", previous_value="", valid_from=""):
    return {"subject": subject, "predicate": predicate, "value": value, "kind": kind,
            "previous_value": previous_value, "valid_from": valid_from}


def rel(subject, predicate, obj, kind="state", previous="", valid_from=""):
    return {"subject": subject, "predicate": predicate, "object": obj, "kind": kind,
            "previous": previous, "valid_from": valid_from}


def doc(memories=(), relationships=()):
    return {"memories": list(memories), "relationships": list(relationships)}


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests (no model download)."""

    def __init__(self, dim: int = 256, model_name: str = "test-hashing-embedder"):
        self.dim = dim
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                out[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out
