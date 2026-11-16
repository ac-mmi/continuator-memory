import os
import socket
import urllib.parse
import urllib.request

import pytest

from continuator_memory import envfile
from continuator_memory.store import Store
from tests.helpers import HashingEmbedder

DEFAULT_OLLAMA_PORT = 11434
# Set by tests or by loading a .env file, and able to switch on a real model call in a later test.
MODEL_ENV_KEYS = ("CONTINUATOR_OLLAMA_MODEL", "CONTINUATOR_ANSWER_MODEL", "CONTINUATOR_QUERY_PLAN_MODEL")


def _ollama_ports() -> set[int]:
    ports = {DEFAULT_OLLAMA_PORT}
    host = os.environ.get("OLLAMA_HOST", "")
    if host:
        port = urllib.parse.urlparse(host if "://" in host else f"http://{host}").port
        if port:
            ports.add(port)
    return ports


def _install_ollama_guard(patcher: pytest.MonkeyPatch) -> None:
    """Fail any test that reaches Ollama. ``pytest.fail`` is a BaseException, so the application's
    ``except Exception`` fallbacks cannot turn the attempt into a silent pass."""
    real_urlopen = urllib.request.urlopen
    real_connect = socket.socket.connect

    def guarded_urlopen(url, *args, **kwargs):
        target = getattr(url, "full_url", url)
        if urllib.parse.urlparse(target).path.startswith("/api/"):
            pytest.fail(f"test tried to call Ollama at {target}; mock it or mark the test live_ollama")
        return real_urlopen(url, *args, **kwargs)

    def guarded_connect(self, address):
        if isinstance(address, tuple) and len(address) >= 2 and address[1] in _ollama_ports():
            pytest.fail(f"test tried to connect to Ollama at {address}; mock it or mark the test live_ollama")
        return real_connect(self, address)

    patcher.setattr(urllib.request, "urlopen", guarded_urlopen)
    patcher.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(autouse=True)
def _ollama_isolation(request):
    """Ordinary tests cannot reach Ollama, and model settings cannot leak between tests.

    This deliberately does not request ``monkeypatch``. Autouse fixtures set up first, so this one finishes
    after a test's own monkeypatch has undone its changes, and its restore is what remains. Only the
    Ollama-related keys are restored; the rest of the environment is left alone. A test that wants ``.env``
    behaviour supplies its own file or patches ``default_env_paths`` itself.
    """
    if request.node.get_closest_marker("live_ollama"):
        yield
        return
    keys = (*MODEL_ENV_KEYS, "OLLAMA_HOST")
    saved = {key: os.environ.get(key) for key in keys}
    for key in MODEL_ENV_KEYS:
        os.environ.pop(key, None)
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(envfile, "default_env_paths", lambda: ())
        _install_ollama_guard(patcher)
        yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()
