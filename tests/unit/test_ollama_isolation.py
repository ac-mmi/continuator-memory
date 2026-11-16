"""The guard in tests/conftest.py: ordinary tests cannot reach Ollama or leak model settings."""

from __future__ import annotations

import os
import socket
import urllib.request
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from continuator_memory import envfile
from continuator_memory.generation import GenerationError, OllamaGenerator
from continuator_memory.memory.extraction import OllamaJSONClient
from continuator_memory.v2.atomizer import OllamaChatClient
from continuator_memory.webapi.workspace import Workspace
from tests.conftest import MODEL_ENV_KEYS
from tests.helpers import HashingEmbedder

BLOCKED = pytest.fail.Exception


@pytest.mark.parametrize("call", [
    lambda: OllamaGenerator("m", "http://localhost:11434").generate("s", "p"),
    lambda: OllamaChatClient("m", "http://localhost:11434").complete_json("s", "p", {}),
    lambda: OllamaJSONClient("m", "http://localhost:11434").complete_json("s", "p", {}),
    lambda: urllib.request.urlopen("http://example.invalid/api/chat"),
    lambda: urllib.request.urlopen("https://ollama.example.com/api/tags"),
])
def test_every_client_is_blocked(call):
    with pytest.raises(BLOCKED):
        call()


def test_direct_socket_to_ollama_is_blocked():
    with socket.socket() as sock, pytest.raises(BLOCKED):
        sock.connect(("127.0.0.1", 11434))


def test_http_client_to_ollama_is_blocked():
    connection = HTTPConnection("localhost", 11434, timeout=1)
    with pytest.raises(BLOCKED):
        connection.request("POST", "/api/chat", body="{}")


def test_custom_ollama_host_port_is_blocked(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:12345")
    with socket.socket() as sock, pytest.raises(BLOCKED):
        sock.connect(("127.0.0.1", 12345))


def test_a_request_error_fallback_cannot_hide_the_attempt():
    """The original failure: planning with a configured model, generate=False, no fake planner."""
    os.environ["CONTINUATOR_OLLAMA_MODEL"] = "any-model"
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    with pytest.raises(BLOCKED):
        workspace.ask("weekday bicycle", retrieval_mode="multiquery", generate=False)


def test_iterative_mode_with_no_fake_is_blocked_too():
    os.environ["CONTINUATOR_OLLAMA_MODEL"] = "any-model"
    workspace = Workspace.from_sample(embedder=HashingEmbedder(), k=1)
    with pytest.raises(BLOCKED):
        workspace.ask("weekday bicycle", retrieval_mode="iterative", generate=False)


def test_local_servers_on_other_ports_still_work():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET", "/api/anything")
        assert connection.getresponse().status == 204
    finally:
        server.shutdown()
        server.server_close()


def test_a_tests_own_fake_replaces_the_guard(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: "faked")
    assert urllib.request.urlopen("http://localhost:11434/api/chat") == "faked"


def test_model_settings_start_cleared():
    assert not [key for key in MODEL_ENV_KEYS if key in os.environ]


def test_real_env_files_are_not_loaded():
    assert envfile.default_env_paths() == ()


def test_generation_error_handling_is_still_testable(monkeypatch):
    def refused(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    with pytest.raises((GenerationError, OSError)):
        OllamaGenerator("m", "http://localhost:11434").generate("s", "p")


class TestNothingLeaksBetweenTests:
    """Order matters: the first writes the environment directly, the second checks it is gone."""

    def test_a_write_directly_and_through_monkeypatch(self, monkeypatch):
        os.environ["CONTINUATOR_QUERY_PLAN_MODEL"] = "leak"
        os.environ["CONTINUATOR_ANSWER_MODEL"] = "leak"
        monkeypatch.setenv("CONTINUATOR_OLLAMA_MODEL", "leak")
        os.environ["UNRELATED_ISOLATION_MARKER"] = "kept"

    def test_b_model_keys_are_gone_and_other_keys_are_not_touched(self):
        assert not [key for key in MODEL_ENV_KEYS if key in os.environ]
        assert os.environ.pop("UNRELATED_ISOLATION_MARKER") == "kept"


@pytest.mark.live_ollama
def test_the_marker_removes_the_guard_without_connecting():
    """Only inspects what is installed: this makes no connection."""
    assert urllib.request.urlopen.__name__ == "urlopen"
    assert socket.socket.connect.__name__ == "connect"
    assert envfile.default_env_paths.__name__ == "default_env_paths"
