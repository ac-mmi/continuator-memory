import io
import json
import urllib.error

from continuator_memory import manifest
from continuator_memory.generation import ANSWER_NUM_CTX, OllamaGenerator
from continuator_memory.memory.extraction import EXTRACTION_NUM_CTX, OllamaJSONClient


def test_pinned_v1_manifest_values():
    assert manifest.PINNED["embedding_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert manifest.PINNED["ollama_version"] == "0.30.6"
    assert manifest.PINNED["ollama_model"] == "qwen2.5:1.5b"
    assert manifest.PINNED["ollama_model_digest"] == "65ec06548149"


def test_package_versions_reports_every_tracked_package():
    versions = manifest.package_versions()
    assert set(versions) == set(manifest.TRACKED_PACKAGES)
    assert versions["numpy"]


def test_embedding_revision_reads_hf_cache(tmp_path, monkeypatch):
    ref = tmp_path / "models--org--model" / "refs"
    ref.mkdir(parents=True)
    (ref / "main").write_text("abc123\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert manifest.embedding_revision("org/model") == "abc123"
    assert manifest.embedding_revision("org/missing") is None


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_info_reads_version_and_digest(monkeypatch):
    bodies = {
        "http://h/api/version": {"version": "0.30.6"},
        "http://h/api/tags": {"models": [{"name": "m", "digest": "65ec06548149ff"}]},
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url, timeout: _FakeResponse(json.dumps(bodies[url]).encode()),
    )
    assert manifest.ollama_info("http://h/", "m") == {
        "ollama_version": "0.30.6", "ollama_model_digest": "65ec06548149ff",
    }


def test_ollama_info_unreachable_is_none(monkeypatch):
    def refuse(url, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    assert manifest.ollama_info("http://h", "m") == {
        "ollama_version": None, "ollama_model_digest": None,
    }


def _observed(**overrides):
    obs = {k: v for k, v in manifest.PINNED.items() if k != "packages"}
    obs["packages"] = dict(manifest.PINNED["packages"])
    obs["ollama_model_digest"] = "65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b"
    obs.update(overrides)
    return obs


def test_matches_pinned():
    assert manifest.matches_pinned(_observed())
    assert not manifest.matches_pinned(_observed(ollama_model_digest="deadbeef"))
    assert not manifest.matches_pinned(_observed(embedding_revision="other"))
    assert not manifest.matches_pinned(_observed(packages={"numpy": "0"}))


def test_explicit_context_windows_are_defaults():
    assert ANSWER_NUM_CTX == 4096 and EXTRACTION_NUM_CTX == 4096
    assert OllamaGenerator("m", "http://h").num_ctx == ANSWER_NUM_CTX
    assert OllamaJSONClient("m", "http://h").num_ctx == EXTRACTION_NUM_CTX
