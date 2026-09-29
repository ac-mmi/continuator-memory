"""V1 reproducibility manifest: the pinned environment and what is actually installed.

Metadata only; nothing here is read by ingestion, retrieval, memory or generation.
Run ``python -m continuator_memory.manifest`` to print the observed manifest as JSON.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Any

from continuator_memory.config import Settings

PINNED: dict[str, Any] = {
    "python": "3.12.7",
    "platform": "macOS-26.3.1-arm64-arm-64bit",
    "packages": {
        "numpy": "2.5.3",
        "sentence-transformers": "6.1.0",
        "torch": "2.14.0",
        "transformers": "5.17.0",
    },
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "ollama_version": "0.30.6",
    "ollama_model": "qwen2.5:1.5b",
    "ollama_model_digest": "65ec06548149",
}

TRACKED_PACKAGES = ("continuator-memory", "numpy", "sentence-transformers", "torch", "transformers")


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def embedding_revision(model_name: str) -> str | None:
    """Snapshot hash the Hugging Face cache resolves ``model_name`` to, if cached."""
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(Path.home(), ".cache", "huggingface")), "hub"
    )
    ref = Path(cache) / f"models--{model_name.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text().strip() or None
    except OSError:
        return None


def _get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_info(host: str, model: str, timeout: float = 5.0) -> dict[str, str | None]:
    """Server version and local digest of ``model``; ``None`` fields if unreachable."""
    host = host.rstrip("/")
    info: dict[str, str | None] = {"ollama_version": None, "ollama_model_digest": None}
    try:
        info["ollama_version"] = _get_json(f"{host}/api/version", timeout).get("version")
        for entry in _get_json(f"{host}/api/tags", timeout).get("models", []):
            if entry.get("name") == model:
                info["ollama_model_digest"] = entry.get("digest")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, AttributeError):
        pass
    return info


def observed(settings: Settings) -> dict[str, Any]:
    ollama = ollama_info(settings.ollama_host, settings.ollama_model)
    extraction_digest = (
        ollama["ollama_model_digest"]
        if settings.extraction_model == settings.ollama_model
        else ollama_info(settings.ollama_host, settings.extraction_model)["ollama_model_digest"]
    )
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "embedding_model": settings.embedding_model,
        "embedding_revision": embedding_revision(settings.embedding_model),
        "ollama_version": ollama["ollama_version"],
        "ollama_model": settings.ollama_model,
        "ollama_model_digest": ollama["ollama_model_digest"],
        "extraction_model": settings.extraction_model,
        "extraction_model_digest": extraction_digest,
    }


def matches_pinned(obs: dict[str, Any]) -> bool:
    digest = obs.get("ollama_model_digest") or ""
    return (
        obs.get("python") == PINNED["python"]
        and all(obs["packages"].get(k) == v for k, v in PINNED["packages"].items())
        and obs.get("embedding_model") == PINNED["embedding_model"]
        and obs.get("embedding_revision") == PINNED["embedding_revision"]
        and obs.get("ollama_version") == PINNED["ollama_version"]
        and obs.get("ollama_model") == PINNED["ollama_model"]
        and digest.startswith(PINNED["ollama_model_digest"])
    )


def main() -> int:
    obs = observed(Settings.from_env())
    print(json.dumps({"pinned": PINNED, "observed": obs, "matches_pinned": matches_pinned(obs)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
