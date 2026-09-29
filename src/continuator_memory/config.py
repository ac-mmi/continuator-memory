"""Runtime settings, read from environment variables with fixed defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_PATH = "data/continuator.db"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_WINDOW_SIZE = 4
DEFAULT_OVERLAP = 2
DEFAULT_TOP_K = 5


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
            db_path=Path(env.get("CONTINUATOR_DB_PATH", DEFAULT_DB_PATH)),
            embedding_model=env.get("CONTINUATOR_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            ollama_model=ollama_model,
            ollama_host=env.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            extraction_model=env.get("CONTINUATOR_EXTRACTION_MODEL", ollama_model),
            window_size=int(env.get("CONTINUATOR_CHUNK_WINDOW", DEFAULT_WINDOW_SIZE)),
            overlap=int(env.get("CONTINUATOR_CHUNK_OVERLAP", DEFAULT_OVERLAP)),
            top_k=int(env.get("CONTINUATOR_TOP_K", DEFAULT_TOP_K)),
        )
