"""Stdlib .env loading. No Ollama."""

from __future__ import annotations

from pathlib import Path

from continuator_memory.envfile import apply_env_file, load_root_env, parse_env_line
from continuator_memory.webapi.answering import NOT_CONFIGURED_ANSWER, load_generation_settings
from continuator_memory.webapi.workspace import Workspace, open_workspace
from tests.helpers import HashingEmbedder

ENV_BODY = """
# comment
CONTINUATOR_OLLAMA_MODEL=qwen2.5:1.5b
export OLLAMA_HOST="http://localhost:11434"
CONTINUATOR_ANSWER_TEMPERATURE=0
ignored
=nokey
"""


def test_parse_env_line_skips_comments_and_unwraps_quotes():
    assert parse_env_line("# CONTINUATOR_OLLAMA_MODEL=x") is None
    assert parse_env_line("") is None
    assert parse_env_line("export OLLAMA_HOST='http://localhost:11434'") == (
        "OLLAMA_HOST",
        "http://localhost:11434",
    )
    assert parse_env_line("CONTINUATOR_OLLAMA_MODEL=qwen2.5:1.5b") == (
        "CONTINUATOR_OLLAMA_MODEL",
        "qwen2.5:1.5b",
    )


def test_apply_env_file_sets_only_missing_keys(tmp_path):
    path = tmp_path / ".env"
    path.write_text(ENV_BODY, encoding="utf-8")
    environ = {"CONTINUATOR_OLLAMA_MODEL": "already-set", "KEEP": "1"}
    applied = apply_env_file(path, environ)
    assert environ["CONTINUATOR_OLLAMA_MODEL"] == "already-set"
    assert "CONTINUATOR_OLLAMA_MODEL" not in applied
    assert environ["OLLAMA_HOST"] == "http://localhost:11434"
    assert environ["CONTINUATOR_ANSWER_TEMPERATURE"] == "0"
    assert environ["KEEP"] == "1"


def test_missing_env_file_is_a_noop(tmp_path):
    assert apply_env_file(tmp_path / "missing.env", {}) == ()


def test_load_generation_settings_reads_file_then_preserves_shell(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("CONTINUATOR_OLLAMA_MODEL=qwen2.5:1.5b\n", encoding="utf-8")
    monkeypatch.delenv("CONTINUATOR_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("CONTINUATOR_ANSWER_MODEL", raising=False)
    load_root_env(paths=(path,))
    settings = load_generation_settings()
    assert settings.configured is True
    assert settings.model == "qwen2.5:1.5b"

    monkeypatch.setenv("CONTINUATOR_OLLAMA_MODEL", "from-shell")
    load_root_env(paths=(path,))
    assert load_generation_settings().model == "from-shell"


def test_missing_model_still_explains_configuration(monkeypatch):
    monkeypatch.delenv("CONTINUATOR_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("CONTINUATOR_ANSWER_MODEL", raising=False)
    monkeypatch.setenv("CONTINUATOR_ANSWER_ENABLED", "1")
    settings = load_generation_settings()
    assert settings.configured is False
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    workspace._generation = settings
    result = workspace.ask("weekday bicycle", retrieval_mode="baseline", generate=True)
    assert result["answer"]["content"] == NOT_CONFIGURED_ANSWER
    assert result["answer"]["evidence"]["generation"]["status"] == "not_configured"


def test_open_workspace_loads_env_before_generation_settings(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("CONTINUATOR_OLLAMA_MODEL=qwen2.5:1.5b\n", encoding="utf-8")
    monkeypatch.delenv("CONTINUATOR_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("CONTINUATOR_ANSWER_MODEL", raising=False)
    monkeypatch.setattr(
        "continuator_memory.envfile.default_env_paths",
        lambda: (path,),
    )
    workspace = open_workspace(sample="fixture", embedder=HashingEmbedder(), k=1)
    assert workspace._generation.configured is True
    assert workspace._generation.model == "qwen2.5:1.5b"


def test_example_env_documents_the_default_model():
    text = Path(".env.example").read_text(encoding="utf-8")
    assert "CONTINUATOR_OLLAMA_MODEL=qwen2.5:1.5b" in text
    assert "CONTINUATOR_ANSWER_MODEL" in text
    assert "OLLAMA_HOST" in text
