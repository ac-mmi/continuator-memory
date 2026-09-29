import io
import json
import urllib.error

import pytest

from continuator_memory.generation import (
    SYSTEM_PROMPT,
    GenerationError,
    OllamaGenerator,
    build_prompt,
)
from continuator_memory.models import RetrievedChunk, load_conversation
from continuator_memory.pipeline import Pipeline
from tests.helpers import EXAMPLE_CONVERSATION


def _chunk(cid="conv:c0001", mids=("m1", "m2"), text="user: hi"):
    return RetrievedChunk(1, cid, "conv", mids, 0.5, text)


def test_build_prompt_includes_provenance_and_question():
    prompt = build_prompt("What?", [_chunk()])
    assert "[conv:c0001] (messages: m1, m2)\nuser: hi" in prompt
    assert prompt.endswith("Question: What?\nAnswer:")


def test_build_prompt_without_chunks():
    assert "(no excerpts retrieved)" in build_prompt("What?", [])


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_request_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return _FakeResponse(json.dumps({"message": {"content": " Shellfish. "}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    gen = OllamaGenerator("some-model", "http://localhost:11434/")
    assert gen.generate("sys", "prompt") == "Shellfish."
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"] == {
        "model": "some-model",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
        ],
        "stream": False,
        "options": {"temperature": 0, "seed": 0},
    }


def test_ollama_unreachable(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(GenerationError, match="ollama serve"):
        OllamaGenerator("m", "http://localhost:1").generate("s", "p")


class _RecordingGenerator:
    model_name = "recording"

    def __init__(self):
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return "answer"


def test_pipeline_ask_passes_retrieved_chunks_to_generator(store, embedder):
    gen = _RecordingGenerator()
    pipeline = Pipeline(store, embedder, gen)
    pipeline.ingest(load_conversation(EXAMPLE_CONVERSATION))
    answer = pipeline.ask("What is Priya allergic to?", k=2)
    assert answer.answer == "answer"
    assert answer.model_name == "recording"
    assert len(answer.retrieved) == 2
    assert gen.calls == [(SYSTEM_PROMPT, answer.prompt)]
    for chunk in answer.retrieved:
        assert f"[{chunk.chunk_id}]" in answer.prompt
