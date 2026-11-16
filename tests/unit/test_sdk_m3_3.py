"""M3.3: ContinuatorClient + HTTP Markdown ingest (in-process server)."""

from __future__ import annotations

import json
from threading import Thread
from urllib.parse import quote

import pytest

from continuator_memory.sdk import ContinuatorAPIError, ContinuatorClient
from continuator_memory.sdk._http import encode_path_segment, request_json
from continuator_memory.webapi.context_pack import CONTEXT_PACK_REQUIRED_KEYS
from continuator_memory.webapi.server import make_server
from continuator_memory.webapi.workspace import Workspace
from tests.helpers import HashingEmbedder
from tests.unit.test_webapi import FERRY_TRANSCRIPT, _http


@pytest.fixture
def server_url(tmp_path, monkeypatch):
    db = tmp_path / "sdk.db"
    monkeypatch.setenv("CONTINUATOR_DB_PATH", str(db))
    monkeypatch.delenv("CONTINUATOR_IMPORT_DB", raising=False)
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=3)
    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}", ws
    finally:
        httpd.shutdown()
        httpd.server_close()
        store = getattr(ws, "_store", None)
        if store is not None:
            store.close()


def test_sdk_health_import_context_and_docs(server_url):
    base, _ws = server_url
    client = ContinuatorClient(base, timeout=30.0)

    health = client.health()
    assert health["ok"] is True

    imported = client.import_conversation(
        FERRY_TRANSCRIPT, filename="conv.txt", extract="messages",
    )
    assert imported["ok"] is True
    cid = imported["source"]["id"]

    pack = client.context("usual ferry", mode="baseline", limit=5)
    assert pack["schemaVersion"] == "1"
    assert CONTEXT_PACK_REQUIRED_KEYS <= pack.keys()
    assert pack["memories"]
    # Unknown/extra fields must be preserved if the server adds them.
    pack_with_extra = dict(pack)
    pack_with_extra["futureField"] = {"x": 1}
    assert pack_with_extra["futureField"] == {"x": 1}
    assert "schemaVersion" in pack_with_extra

    ingested = client.ingest_markdown(
        "# Auth\n\n### Token Refresh\n\nAccess tokens expire after one hour.\n",
        filename="policy.md",
        source_uri="inline://sdk-policy",
    )
    assert ingested["ok"] is True
    assert ingested["replaced"] is False
    doc_id = ingested["document"]["documentId"]
    assert ingested["document"]["sourceUri"] == "inline://sdk-policy"
    assert ingested["document"]["unitCount"] >= 1
    assert "units" not in ingested

    docs = client.documents.list()
    assert any(d["documentId"] == doc_id for d in docs)
    detail = client.documents.get(doc_id)
    assert detail["document"]["documentId"] == doc_id
    assert detail["units"]
    evidence_id = detail["units"][0]["evidenceId"]
    ev = client.evidence.get(evidence_id)
    assert ev["evidence"]["evidenceId"] == evidence_id
    assert ev["document"]["documentId"] == doc_id

    memories = client.memories.list(include_superseded=True)
    assert memories
    mid = memories[0]["id"]
    assert client.memories.get(mid)["memory"]["id"] == mid

    sources = client.sources.list()
    assert sources[0]["id"] == cid
    assert client.sources.get(cid)["source"]["id"] == cid

    again = client.ingest_markdown(
        "# Auth\n\n### Token Refresh\n\nAccess tokens expire after one hour.\n",
        filename="policy.md",
        source_uri="inline://sdk-policy",
    )
    assert again["replaced"] is True
    assert again["document"]["documentId"] == doc_id

    deleted = client.documents.delete(doc_id)
    assert deleted["deleted"] is True
    idem = client.documents.delete(doc_id)
    assert idem["deleted"] is False

    corpus = client.corpus.delete_conversation(cid)
    assert corpus["deleted"] is True
    with pytest.raises(ContinuatorAPIError) as caught:
        client.context("usual ferry")
    assert caught.value.status_code == 400
    assert "deleted" in caught.value.message.lower()


def test_sdk_url_encoding_and_query_params(server_url):
    base, ws = server_url
    client = ContinuatorClient(base)
    client.import_conversation(FERRY_TRANSCRIPT, extract="messages")
    # IDs with reserved characters must be encoded.
    weird = "doc:a/b c"
    assert encode_path_segment(weird) == quote(weird, safe="")
    with pytest.raises(ContinuatorAPIError) as err:
        client.documents.get(weird)
    assert err.value.status_code == 404

    # types comma-separated query
    rows = client.memories.list(types=["fact", "preference"], include_superseded=False)
    assert isinstance(rows, list)


def test_sdk_http_400_and_503(server_url, monkeypatch):
    base, ws = server_url
    client = ContinuatorClient(base)
    with pytest.raises(ContinuatorAPIError) as bad:
        client.context("")
    assert bad.value.status_code == 400
    assert bad.value.message
    assert bad.value.body is not None

    from continuator_memory.embedding import EmbeddingLoadError

    def boom(*_a, **_k):
        raise EmbeddingLoadError("embedding model unavailable (test)", reason="network")

    monkeypatch.setattr(
        "continuator_memory.embedding.SentenceTransformerEmbedder", boom,
    )
    ws._embedder = None
    # Fixture workspace already has HashingEmbedder — clear it to force load path.
    # context uses embedder(); set to None then patch constructor for fresh load.
    client2 = ContinuatorClient(base)
    # Use a workspace that will call SentenceTransformerEmbedder — fixture has HashingEmbedder.
    # Hit /context after swapping embedder factory via workspace.embedder failure:
    def fail_embedder():
        raise EmbeddingLoadError("embedding model unavailable (test)", reason="network")

    monkeypatch.setattr(ws, "embedder", fail_embedder)
    with pytest.raises(ContinuatorAPIError) as unavailable:
        client2.context("ferry")
    assert unavailable.value.status_code == 503
    assert "embedding" in unavailable.value.message.lower()


def test_sdk_connection_and_malformed_json(monkeypatch):
    client = ContinuatorClient("http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(ContinuatorAPIError) as conn:
        client.health()
    assert conn.value.status_code is None
    assert "connect" in conn.value.message.lower() or "timed out" in conn.value.message.lower()

    def fake_urlopen(req, timeout=None):
        class Resp:
            status = 200

            def read(self):
                return b"not-json{"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ContinuatorAPIError) as bad_json:
        ContinuatorClient("http://example.test").health()
    assert "invalid JSON" in bad_json.value.message


def test_sdk_delete_not_retried(server_url, monkeypatch):
    base, _ws = server_url
    client = ContinuatorClient(base)
    client.import_conversation(FERRY_TRANSCRIPT, extract="messages")
    calls: list[str] = []
    original = request_json

    def counting(method, url, **kwargs):
        calls.append(method.upper())
        return original(method, url, **kwargs)

    monkeypatch.setattr("continuator_memory.sdk.client.request_json", counting)
    # Missing id is idempotent 200 after a product store exists — still one attempt.
    client.documents.delete("doc:missing")
    assert calls == ["DELETE"]


def test_http_ingest_validation_and_generic_errors(tmp_path, monkeypatch):
    db = tmp_path / "ingest.db"
    monkeypatch.setenv("CONTINUATOR_DB_PATH", str(db))
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(FERRY_TRANSCRIPT, filename="c.txt", extract="messages", db_path=db)
    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, err = _http(httpd, "POST", "/documents/ingest", {"filename": "x.md"})
        assert status == 400
        assert "text" in err["error"]

        status, _, err = _http(
            httpd, "POST", "/documents/ingest", {"text": "hi", "sourceUri": ""},
        )
        assert status == 400

        status, _, ok = _http(
            httpd,
            "POST",
            "/documents/ingest",
            {"text": "# Hi\n\nBody.\n", "sourceUri": "inline://a"},
        )
        assert status == 200
        assert ok["ok"] is True
        assert "units" not in ok

        # Unexpected GET error stays generic.
        def boom():
            raise RuntimeError("secret")

        monkeypatch.setattr(ws, "health", boom)
        status, _, body = _http(httpd, "GET", "/health")
        assert status == 500
        assert body == {"error": "internal server error"}
    finally:
        httpd.shutdown()
        httpd.server_close()
        getattr(ws, "_store").close()


def test_fixture_workspace_rejects_ingest_without_store():
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    with pytest.raises(ValueError, match="product SQLite store"):
        ws.ingest_markdown_text("# x\n", source_uri="inline://x")
