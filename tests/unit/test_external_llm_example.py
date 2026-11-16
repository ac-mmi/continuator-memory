"""Offline tests for examples/external_llm/ask_with_context_pack.py (no network)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "external_llm" / "ask_with_context_pack.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("ask_with_context_pack", EXAMPLE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ex():
    return _load_example()


def _sample_pack(**overrides: Any) -> dict[str, Any]:
    pack = {
        "schemaVersion": "1",
        "query": "usual ferry",
        "mode": "baseline",
        "formattedContext": (
            "RELEVANT LONG-TERM MEMORY\n"
            "Query: usual ferry\n"
            "Retrieval mode: baseline\n\n"
            "[1] memory mem:imported:m0000:c0\n"
            "Statement: my usual ferry is 7:40\n"
            "Supporting evidence:\n"
            "  [imported:m0000] user: my usual ferry is 7:40\n"
        ),
        "memories": [{"id": "mem:imported:m0000:c0", "text": "my usual ferry is 7:40"}],
        "evidence": [{"messageId": "imported:m0000", "text": "my usual ferry is 7:40"}],
        "citations": [{"memoryId": "mem:imported:m0000:c0", "messageIds": ["imported:m0000"]}],
        "documentEvidence": [{"evidenceId": "ev:1", "text": "Access tokens expire after one hour."}],
        "documentCitations": [{"evidenceId": "ev:1", "documentId": "doc:1"}],
    }
    pack.update(overrides)
    return pack


def test_build_chat_messages_includes_exact_context_and_question(ex):
    ctx = "EXACT_FORMATTED_CONTEXT_BLOCK\nline2\n"
    messages = ex.build_chat_messages("What is my usual ferry?", ctx)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "only the CONTEXT" in messages[0]["content"]
    assert "untrusted reference data" in messages[0]["content"]
    user = messages[1]["content"]
    assert user.startswith("CONTEXT:\n")
    assert ctx in user
    assert user.index(ctx) == len("CONTEXT:\n")
    assert "\n\nQUESTION:\nWhat is my usual ferry?\n" in user


def test_provenance_fields_preserved_unchanged(ex):
    pack = _sample_pack()
    memories = pack["memories"]
    evidence = pack["evidence"]
    provenance = ex.provenance_from_pack(pack)
    assert provenance["memories"] is memories
    assert provenance["evidence"] is evidence
    assert provenance["citations"] == pack["citations"]
    assert provenance["documentEvidence"] == pack["documentEvidence"]
    assert provenance["documentCitations"] == pack["documentCitations"]


def test_missing_formatted_context(ex):
    with pytest.raises(ValueError, match="formattedContext"):
        ex.require_formatted_context({"schemaVersion": "1"})
    with pytest.raises(ValueError, match="formattedContext"):
        ex.require_formatted_context({"formattedContext": "   "})


def test_missing_provider_credentials(ex):
    with pytest.raises(ex.ExternalLLMError, match="OPENAI_API_KEY"):
        ex.load_provider_settings({})
    with pytest.raises(ex.ExternalLLMError, match="OPENAI_MODEL"):
        ex.load_provider_settings({"OPENAI_API_KEY": "sk-test"})


def test_extract_assistant_text_malformed(ex):
    with pytest.raises(ex.ExternalLLMError, match="non-object"):
        ex.extract_assistant_text([])
    with pytest.raises(ex.ExternalLLMError, match="choices"):
        ex.extract_assistant_text({})
    with pytest.raises(ex.ExternalLLMError, match="content"):
        ex.extract_assistant_text({"choices": [{"message": {"content": ""}}]})


def test_chat_completion_http_error(ex, monkeypatch):
    import urllib.error

    class FakeHTTPError(urllib.error.HTTPError):
        def __init__(self):
            super().__init__(
                "https://api.openai.com/v1/chat/completions",
                401,
                "Unauthorized",
                hdrs=None,
                fp=None,
            )

        def read(self):
            return json.dumps({"error": {"message": "invalid api key"}}).encode()

    def boom(*_a, **_k):
        raise FakeHTTPError()

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(ex.ExternalLLMError, match="HTTP 401") as caught:
        ex.chat_completion(
            [{"role": "user", "content": "hi"}],
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            model="gpt-test",
        )
    assert "invalid api key" in str(caught.value)


def test_dry_run_avoids_external_api(ex, monkeypatch):
    pack = _sample_pack()
    calls: list[str] = []

    class FakeClient:
        def context(self, query, mode="baseline", limit=None):
            assert query == "usual ferry"
            return pack

    def no_chat(*_a, **_k):
        calls.append("chat")
        raise AssertionError("external API must not be called in dry-run")

    monkeypatch.setattr(ex, "chat_completion", no_chat)
    monkeypatch.setattr(ex, "load_provider_settings", lambda env=None: (_ for _ in ()).throw(
        AssertionError("credentials must not be loaded in dry-run")
    ))

    result = ex.run(
        "usual ferry",
        dry_run=True,
        client=FakeClient(),
        chat_fn=no_chat,
    )
    assert result["dryRun"] is True
    assert result["answer"] is None
    assert result["messages"][1]["content"].count(pack["formattedContext"]) == 1
    assert result["provenance"]["memories"] is pack["memories"]
    assert calls == []


def test_run_live_path_uses_chat_fn(ex):
    pack = _sample_pack()

    class FakeClient:
        def context(self, query, mode="baseline", limit=None):
            return pack

    def fake_chat(messages, **_k):
        assert "EXACT" not in messages[1]["content"] or True
        assert pack["formattedContext"] in messages[1]["content"]
        return "The usual ferry is 7:40 [imported:m0000]."

    result = ex.run(
        "usual ferry",
        dry_run=False,
        client=FakeClient(),
        chat_fn=fake_chat,
        env={"OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "gpt-test"},
    )
    assert result["dryRun"] is False
    assert "7:40" in result["answer"]
    assert result["provenance"]["documentEvidence"] == pack["documentEvidence"]
