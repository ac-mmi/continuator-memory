"""M3.1.1: canonical DB path, import/ingest alignment, embedding load errors."""

from __future__ import annotations

from pathlib import Path
from threading import Thread

import pytest

from continuator_memory.config import (
    DEFAULT_DB_PATH,
    Settings,
    legacy_import_db_path,
    resolve_product_db_path,
)
from continuator_memory.embedding import (
    EmbeddingLoadError,
    SentenceTransformerEmbedder,
    classify_embedding_load_failure,
)
from continuator_memory.evidence.ingest import ingest_markdown
from continuator_memory.evidence.store import EvidenceStore
from continuator_memory.store import Store
from continuator_memory.webapi.server import make_server
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder
from tests.unit.test_webapi import FERRY_TRANSCRIPT, _http


# --- Canonical DB path ------------------------------------------------------


def test_resolve_product_db_path_default():
    assert resolve_product_db_path(env={}) == Path(DEFAULT_DB_PATH)


def test_resolve_product_db_path_canonical_wins_over_legacy(tmp_path):
    env = {
        "CONTINUATOR_DB_PATH": str(tmp_path / "canonical.db"),
        "CONTINUATOR_IMPORT_DB": str(tmp_path / "legacy-import.db"),
    }
    assert resolve_product_db_path(env=env) == tmp_path / "canonical.db"


def test_resolve_product_db_path_legacy_alias_when_canonical_unset(tmp_path):
    env = {"CONTINUATOR_IMPORT_DB": str(tmp_path / "legacy-import.db")}
    assert resolve_product_db_path(env=env) == tmp_path / "legacy-import.db"


def test_resolve_product_db_path_explicit_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTINUATOR_DB_PATH", str(tmp_path / "env.db"))
    monkeypatch.setenv("CONTINUATOR_IMPORT_DB", str(tmp_path / "import.db"))
    assert resolve_product_db_path(explicit=tmp_path / "explicit.db") == tmp_path / "explicit.db"


def test_settings_from_env_uses_resolver(tmp_path, monkeypatch):
    monkeypatch.delenv("CONTINUATOR_DB_PATH", raising=False)
    monkeypatch.setenv("CONTINUATOR_IMPORT_DB", str(tmp_path / "only-import.db"))
    assert Settings.from_env().db_path == tmp_path / "only-import.db"


def test_legacy_import_db_path_helper(tmp_path):
    assert legacy_import_db_path(env={}) is None
    assert legacy_import_db_path(env={"CONTINUATOR_IMPORT_DB": str(tmp_path / "x.db")}) == tmp_path / "x.db"


def test_import_and_document_ingest_share_default_db(tmp_path, monkeypatch):
    db = tmp_path / "shared.db"
    monkeypatch.setenv("CONTINUATOR_DB_PATH", str(db))
    monkeypatch.delenv("CONTINUATOR_IMPORT_DB", raising=False)

    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    ws.import_text(FERRY_TRANSCRIPT, filename="conv.txt", extract="messages")
    store = getattr(ws, "_store")
    assert Path(store.path) == db

    policy = tmp_path / "policy.md"
    policy.write_text(
        "# Auth\n\n### Token Refresh\n\nAccess tokens expire after one hour.\n",
        encoding="utf-8",
    )
    # CLI-equivalent: same Settings.db_path / CONTINUATOR_DB_PATH
    with Store(str(Settings.from_env().db_path)) as product:
        result = ingest_markdown(policy, EvidenceStore(product.connection))
    assert result.units
    assert EvidenceStore(store.connection).get_document(result.document.document_id) is not None
    store.close()


def test_import_uses_legacy_import_db_when_only_legacy_set(tmp_path, monkeypatch):
    db = tmp_path / "legacy-only.db"
    monkeypatch.delenv("CONTINUATOR_DB_PATH", raising=False)
    monkeypatch.setenv("CONTINUATOR_IMPORT_DB", str(db))
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(FERRY_TRANSCRIPT, filename="conv.txt", extract="messages")
    assert Path(getattr(ws, "_store").path) == db
    getattr(ws, "_store").close()


# --- Embedding errors -------------------------------------------------------


def test_classify_embedding_network_and_dependency():
    net = classify_embedding_load_failure("m", RuntimeError("403 Forbidden from Hub"))
    assert isinstance(net, EmbeddingLoadError)
    assert net.reason == "network"
    assert "hf download" in str(net)

    dep = classify_embedding_load_failure("m", ModuleNotFoundError("sentence_transformers"))
    assert dep.reason == "dependency"
    assert "sentence-transformers" in str(dep)


def test_sentence_transformer_embedder_wraps_failures(monkeypatch):
    import sentence_transformers

    class Boom:
        def __init__(self, *a, **k):
            raise OSError("403 Forbidden resolving revision")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Boom)
    with pytest.raises(EmbeddingLoadError, match="not available") as caught:
        SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    assert caught.value.reason == "network"


def test_workspace_context_surfaces_embedding_load_error(monkeypatch):
    ws = Workspace.from_sample(embedder=None, k=1)

    def fail_model(model_name, device="cpu", batch_size=32):
        raise EmbeddingLoadError(
            f"embedding model {model_name!r} is not available (network)",
            reason="network",
        )

    monkeypatch.setattr(
        "continuator_memory.embedding.SentenceTransformerEmbedder",
        fail_model,
    )
    with pytest.raises(EmbeddingLoadError, match="not available"):
        ws.context("weekday bicycle", retrieval_mode="baseline")


def test_http_context_returns_503_for_embedding_failure(monkeypatch):
    ws = Workspace.from_sample(embedder=None, k=1)

    def fail_model(model_name, device="cpu", batch_size=32):
        raise EmbeddingLoadError(
            "embedding model 'x' is not available (network or Hub access failed: 403). "
            "Ensure network access for the first download, or pre-cache the model "
            "(for example: hf download x).",
            reason="network",
        )

    monkeypatch.setattr(
        "continuator_memory.embedding.SentenceTransformerEmbedder",
        fail_model,
    )
    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = _http(httpd, "POST", "/context", {"query": "ferry"})
        assert status == 503
        assert "embedding model" in body["error"]
        assert "Traceback" not in body["error"]
        assert "hf download" in body["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_retrieval_still_works_with_provided_embedder():
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    pack = ws.context("weekday bicycle", retrieval_mode="baseline")
    assert pack.evidence.matches
    assert pack.diagnostics["answerGeneration"] == "not_run"
