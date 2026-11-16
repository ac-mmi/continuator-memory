"""M4.1.1 developer-experience remediation: CLI help, fixture honesty, docs."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from continuator_memory.cli import _CLI_EPILOG, main
from continuator_memory.webapi.server import format_serve_startup_line
from continuator_memory.webapi.workspace import Workspace, open_workspace
from tests.helpers import HashingEmbedder

ROOT = Path(__file__).resolve().parents[2]


def test_cli_help_lists_product_before_research_and_labels_both():
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exited:
            main(["--help"])
    assert exited.value.code == 0
    text = buf.getvalue()
    assert "serve" in text
    assert "ingest-markdown" in text
    assert "product HTTP API" in text or "product:" in text.lower() or "product HTTP" in text
    # Product commands appear before Phase-1 research verbs in the listing.
    assert text.index("serve") < text.index("  ingest ")
    assert text.index("ingest-markdown") < text.index("  ingest ")
    assert "research/legacy" in text
    assert "not product POST /ask" in text or "not the Context Pack" in text
    assert "POST /import" in text or "import_conversation" in _CLI_EPILOG
    assert "Phase 1" in text or "research" in text.lower()


def test_cli_epilog_distinguishes_product_and_research():
    assert "product HTTP API" in _CLI_EPILOG
    assert "ingest | retrieve | ask" in _CLI_EPILOG.replace("\n", " ") or (
        "ingest" in _CLI_EPILOG and "retrieve" in _CLI_EPILOG and "ask" in _CLI_EPILOG
    )
    assert "research" in _CLI_EPILOG.lower()
    assert "POST /documents/ingest" in _CLI_EPILOG


def test_serve_help_identifies_product_http_api():
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit):
            main(["serve", "--help"])
    text = buf.getvalue()
    assert "Context Pack" in text or "/context" in text
    assert "import" in text.lower()
    assert "fixture" in text.lower()
    # Parent help also lists the product serve one-liner.
    parent = io.StringIO()
    with redirect_stdout(parent):
        with pytest.raises(SystemExit):
            main(["--help"])
    assert "product HTTP API" in parent.getvalue()


def test_fixture_health_origin_fields():
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    health = ws.health()
    assert health["ok"] is True
    assert health["origin"] == "fixture"
    assert health["sampleId"] == "fixture"
    assert health["conversationId"] == "web-sample"
    assert health["importedFilename"] is None
    # Fixture Context Pack still works (demo data — not an application corpus).
    pack = ws.context("ferry").to_dict()
    assert pack["schemaVersion"] == "1"
    assert pack["conversationId"] == "web-sample"
    assert isinstance(pack["formattedContext"], str)
    assert pack["diagnostics"]["answerGeneration"] == "not_run"


def test_serve_startup_line_identifies_fixture():
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    line = format_serve_startup_line(ws, host="127.0.0.1", port=8765)
    assert "fixture demo corpus" in line
    assert "not your data" in line
    assert "origin=fixture" in line
    assert "conversationId=web-sample" in line
    assert "http://127.0.0.1:8765" in line


def test_serve_startup_line_identifies_imported(tmp_path):
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    sample = (ROOT / "examples" / "quickstart" / "conversation.txt").read_text(encoding="utf-8")
    result = ws.import_text(
        sample,
        filename="conversation.txt",
        extract="messages",
        db_path=tmp_path / "dx.db",
    )
    assert result["ok"] is True
    health = ws.health()
    assert health["origin"] == "imported"
    assert health["sampleId"] == "imported"
    assert health["conversationId"] == "imported"
    assert health["importedFilename"] == "conversation.txt"

    line = format_serve_startup_line(ws, host="127.0.0.1", port=9000)
    assert "imported product corpus" in line
    assert "conversation.txt" in line
    assert "origin=imported" in line
    assert "not your data" not in line


def test_open_workspace_fixture_default_unchanged():
    ws = open_workspace(sample="fixture", embedder=HashingEmbedder(), k=1)
    assert ws.health()["origin"] == "fixture"


def test_readme_and_quickstart_use_ollama_host_not_continuator_ollama_host():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = (ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
    assert "CONTINUATOR_OLLAMA_HOST" not in readme
    assert "| `OLLAMA_HOST`" in readme or "`OLLAMA_HOST`" in readme
    assert "origin" in quickstart and "imported" in quickstart
    assert "fixture" in quickstart.lower()
    assert "documents/ingest" in quickstart or "ingest_markdown" in quickstart
    assert ".env" in readme


def test_architecture_mentions_sdk_and_document_ingest():
    text = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "continuator_memory.sdk" in text
    assert "/documents/ingest" in text
    assert "schema v1" in text or "schemaVersion" in text
    assert "external_llm" in text
