"""Runtime settings, read from environment variables with fixed defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_DB_PATH = "data/continuator.db"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_WINDOW_SIZE = 4
DEFAULT_OVERLAP = 2
DEFAULT_TOP_K = 5

# Legacy alias used by older docs/scripts. Prefer CONTINUATOR_DB_PATH.
_LEGACY_IMPORT_DB_ENV = "CONTINUATOR_IMPORT_DB"
_CANONICAL_DB_ENV = "CONTINUATOR_DB_PATH"


def resolve_product_db_path(
    *,
    env: Mapping[str, str] | None = None,
    explicit: str | Path | None = None,
) -> Path:
    """Return the canonical product SQLite path.

    Precedence:
    1. ``explicit`` (CLI ``--db``, ``import_text(db_path=…)``, etc.)
    2. ``CONTINUATOR_DB_PATH`` (canonical)
    3. ``CONTINUATOR_IMPORT_DB`` (legacy alias, for backward compatibility)
    4. ``data/continuator.db``

    Import, CLI document ingest, and ``Settings.db_path`` all use this resolver so a
    single variable is enough. When both legacy variables are set to different
    paths, ``CONTINUATOR_DB_PATH`` wins for all product writes.
    """
    if explicit is not None:
        return Path(explicit)
    mapping = os.environ if env is None else env
    canonical = mapping.get(_CANONICAL_DB_ENV)
    if canonical:
        return Path(canonical)
    legacy = mapping.get(_LEGACY_IMPORT_DB_ENV)
    if legacy:
        return Path(legacy)
    return Path(DEFAULT_DB_PATH)


def legacy_import_db_path(env: Mapping[str, str] | None = None) -> Path | None:
    """Return ``CONTINUATOR_IMPORT_DB`` when set, else ``None``.

    Used only for the historical ``open_workspace`` fallback when serving an empty
    default DB. Normal import/ingest paths must use :func:`resolve_product_db_path`.
    """
    mapping = os.environ if env is None else env
    raw = mapping.get(_LEGACY_IMPORT_DB_ENV)
    return Path(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    db_path: Path
    embedding_model: str
    ollama_model: str
    ollama_host: str
    extraction_model: str
    window_size: int
    overlap: int
    top_k: int

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        ollama_model = env.get("CONTINUATOR_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        return cls(
            db_path=resolve_product_db_path(env=env),
            embedding_model=env.get("CONTINUATOR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            ollama_model=ollama_model,
            ollama_host=env.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            extraction_model=env.get("CONTINUATOR_EXTRACTION_MODEL", ollama_model),
            window_size=int(env.get("CONTINUATOR_CHUNK_WINDOW", DEFAULT_WINDOW_SIZE)),
            overlap=int(env.get("CONTINUATOR_CHUNK_OVERLAP", DEFAULT_OVERLAP)),
            top_k=int(env.get("CONTINUATOR_TOP_K", DEFAULT_TOP_K)),
        )
