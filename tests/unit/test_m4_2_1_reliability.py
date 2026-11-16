"""M4.2.1 API/SDK reliability remediation tests."""

from __future__ import annotations

import json
import logging
from http.client import HTTPConnection
from pathlib import Path
from threading import Event, Thread

import pytest

from continuator_memory.embedding import EmbeddingLoadError
from continuator_memory.sdk import ContinuatorAPIError, ContinuatorClient
from continuator_memory.sdk._http import request_json
from continuator_memory.store import Store
from continuator_memory.webapi.server import (
    IMPORT_BODY_LIMIT,
    JSON_BODY_LIMIT,
    allow_header_for_path,
    make_server,
    product_db_corpus_status,
    resolve_serve_sample_mode,
    serve_from_env,
)
from continuator_memory.webapi.workspace import Workspace, open_workspace
from tests.helpers import HashingEmbedder
from tests.unit.test_webapi import FERRY_TRANSCRIPT, _http


def test_json_body_limits_constants():
    assert JSON_BODY_LIMIT == 512_000
    assert IMPORT_BODY_LIMIT == 8_000_000


def test_product_db_corpus_status_missing_empty_usable(tmp_path):
    missing = tmp_path / "nope.db"
    assert product_db_corpus_status(missing) == "missing"

    empty = tmp_path / "empty.db"
    Store(str(empty)).close()
    assert product_db_corpus_status(empty) == "empty"

    usable = tmp_path / "usable.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(FERRY_TRANSCRIPT, extract="messages", db_path=usable)
    getattr(ws, "_store").close()
    assert product_db_corpus_status(usable) == "usable"


def test_resolve_serve_sample_auto_and_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("CONTINUATOR_WEB_SAMPLE", raising=False)
    monkeypatch.delenv("CONTINUATOR_DB_PATH", raising=False)
    monkeypatch.delenv("CONTINUATOR_IMPORT_DB", raising=False)

    missing = tmp_path / "a.db"
    mode, path = resolve_serve_sample_mode(sample=None, db=missing, env={})
    assert mode == "fixture"
    assert path == missing

    Store(str(missing)).close()
    mode, _ = resolve_serve_sample_mode(sample=None, db=missing, env={})
    assert mode == "fixture"  # empty schema-only DB

    db = tmp_path / "prod.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(FERRY_TRANSCRIPT, extract="messages", db_path=db)
    getattr(ws, "_store").close()

    mode, path = resolve_serve_sample_mode(sample=None, db=db, env={})
    assert mode == "db"
    assert path == db

    mode, _ = resolve_serve_sample_mode(sample="fixture", db=db, env={})
    assert mode == "fixture"

    mode, _ = resolve_serve_sample_mode(
        sample=None, db=db, env={"CONTINUATOR_WEB_SAMPLE": "fixture"},
    )
    assert mode == "fixture"

    mode, _ = resolve_serve_sample_mode(sample="db", db=db, env={})
    assert mode == "db"


def test_serve_from_env_reopens_imported_db(tmp_path, monkeypatch):
    db = tmp_path / "restart.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(FERRY_TRANSCRIPT, filename="c.txt", extract="messages", db_path=db)
    getattr(ws, "_store").close()

    monkeypatch.delenv("CONTINUATOR_WEB_SAMPLE", raising=False)
    monkeypatch.setenv("CONTINUATOR_DB_PATH", str(db))

    opened: list[Workspace] = []

    def fake_forever(self):
        return None

    monkeypatch.setattr(
        "continuator_memory.webapi.server.ThreadingHTTPServer.serve_forever",
        fake_forever,
    )
    monkeypatch.setattr(
        "continuator_memory.webapi.server.make_server",
        lambda workspace, **k: opened.append(workspace) or type(
            "S", (), {"serve_forever": lambda self: None}
        )(),
    )

    serve_from_env(host="127.0.0.1", port=0, sample=None, db=str(db))
    assert len(opened) == 1
    health = opened[0].health()
    assert health["origin"] == "imported"
    assert health["conversationId"] == "imported"
    getattr(opened[0], "_store").close()


def test_serve_from_env_explicit_fixture_ignores_db(tmp_path, monkeypatch):
    db = tmp_path / "has.db"
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    ws.import_text(FERRY_TRANSCRIPT, extract="messages", db_path=db)
    getattr(ws, "_store").close()

    opened: list[Workspace] = []
    monkeypatch.setattr(
        "continuator_memory.webapi.server.make_server",
        lambda workspace, **k: opened.append(workspace) or type(
            "S", (), {"serve_forever": lambda self: None}
        )(),
    )
    serve_from_env(host="127.0.0.1", port=0, sample="fixture", db=str(db))
    assert opened[0].health()["origin"] == "fixture"


def test_serve_from_env_invalid_db_exits(tmp_path, monkeypatch):
    bad = tmp_path / "bad.db"
    bad.write_text("not a sqlite database", encoding="utf-8")
    monkeypatch.delenv("CONTINUATOR_WEB_SAMPLE", raising=False)
    with pytest.raises(SystemExit) as exited:
        serve_from_env(host="127.0.0.1", port=0, sample="db", db=str(bad))
    assert exited.value.code == 2


def test_first_startup_without_db_uses_fixture(tmp_path, monkeypatch):
    monkeypatch.delenv("CONTINUATOR_WEB_SAMPLE", raising=False)
    mode, _ = resolve_serve_sample_mode(sample=None, db=tmp_path / "fresh.db", env={})
    assert mode == "fixture"
    ws = open_workspace(sample="fixture", embedder=HashingEmbedder(), k=1)
    assert ws.health()["origin"] == "fixture"


def test_ensure_embedder_returns_cached_instance():
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    assert ws.ensure_embedder() is ws._embedder


def test_concurrent_embedder_init_once(monkeypatch):
    ws = Workspace.from_sample(embedder=None, k=1)
    ws._embedder = None
    started = Event()
    proceed = Event()
    inits = {"n": 0}

    class Slow(HashingEmbedder):
        def __init__(self, *_a, **_k):
            inits["n"] += 1
            started.set()
            assert proceed.wait(timeout=5)
            super().__init__()

    import continuator_memory.embedding as emb

    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", Slow)

    errors: list[BaseException] = []
    outs: list[object] = []

    def worker():
        try:
            outs.append(ws.ensure_embedder())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = Thread(target=worker)
    t2 = Thread(target=worker)
    t1.start()
    assert started.wait(timeout=5)
    t2.start()
    # Second thread blocks on embedder_init_lock until first finishes.
    proceed.set()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not errors, errors
    assert inits["n"] == 1
    assert outs[0] is outs[1]


def test_embedder_failure_still_503_path(monkeypatch):
    ws = Workspace.from_sample(embedder=None, k=1)
    ws._embedder = None

    def boom(*_a, **_k):
        raise EmbeddingLoadError("embedding model unavailable (test)", reason="network")

    import continuator_memory.embedding as emb

    monkeypatch.setattr(emb, "SentenceTransformerEmbedder", boom)
    with pytest.raises(EmbeddingLoadError):
        ws.ensure_embedder()


def test_http_405_and_allow_header():
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, headers, data = _http(httpd, "PUT", "/health")
        assert status == 405
        assert data == {"error": "method not allowed"}
        allow = {k.lower(): v for k, v in headers.items()}.get("allow", "")
        assert "GET" in allow
        assert "OPTIONS" in allow

        status, headers, data = _http(httpd, "PATCH", "/context")
        assert status == 405
        assert data["error"] == "method not allowed"
        allow = {k.lower(): v for k, v in headers.items()}.get("allow", "")
        assert "POST" in allow

        status, _, data = _http(httpd, "GET", "/context")
        assert status == 405
        assert data["error"] == "method not allowed"

        status, _, data = _http(httpd, "PUT", "/no-such-route")
        assert status == 404
        assert "not a registered route" in data["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_invalid_json_and_oversized_and_incomplete():
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request(
            "POST",
            "/context",
            body=b"{not-json",
            headers={"Content-Type": "application/json", "Content-Length": "9"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        assert resp.status == 400
        assert "invalid JSON" in json.loads(raw)["error"]
        conn.close()

        conn = HTTPConnection(host, port, timeout=5)
        conn.request(
            "POST",
            "/context",
            body=b"{}",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(JSON_BODY_LIMIT + 1),
            },
        )
        resp = conn.getresponse()
        assert resp.status == 400
        assert "too large" in json.loads(resp.read())["error"]
        conn.close()

        # Missing Content-Length → empty object → validation error (safe)
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("POST", "/context", body=None, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        err = json.loads(resp.read())["error"]
        assert "query" in err.lower() or "string" in err.lower()
        conn.close()

        # Incomplete body: declare 20 bytes, send 2, then half-close so read returns early.
        import socket

        sock = socket.create_connection((host, port), timeout=5)
        sock.sendall(
            b"POST /context HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 20\r\n"
            b"\r\n"
            b"{}"
        )
        sock.shutdown(socket.SHUT_WR)
        raw = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            raw += chunk
        sock.close()
        assert b"400" in raw.split(b"\r\n", 1)[0]
        assert b"incomplete" in raw
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_ask_logs_unexpected_errors(caplog, monkeypatch):
    ws = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    httpd = make_server(ws, host="127.0.0.1", port=0)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        def boom(*_a, **_k):
            raise RuntimeError("injected ask failure")

        monkeypatch.setattr(ws, "ask", boom)
        with caplog.at_level(logging.ERROR):
            status, _, data = _http(
                httpd, "POST", "/ask", {"question": "usual ferry", "generate": False},
            )
        assert status == 500
        assert data == {"error": "internal server error"}
        assert any("unexpected /ask failure" in r.message for r in caplog.records)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_sdk_empty_response_raises(monkeypatch):
    class Resp:
        status = 200

        def read(self):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp())
    with pytest.raises(ContinuatorAPIError, match="empty response"):
        ContinuatorClient("http://example.test").health()


def test_allow_header_helpers():
    assert allow_header_for_path("/health") == "GET, OPTIONS"
    assert allow_header_for_path("/context") == "POST, OPTIONS"
    assert "DELETE" in (allow_header_for_path("/documents/x") or "")
    assert allow_header_for_path("/unknown") is None
