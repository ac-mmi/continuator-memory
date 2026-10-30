"""Load a repo-root ``.env`` into the process environment.

Existing shell variables are never overwritten. Missing files are ignored.
"""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def default_env_paths() -> tuple[Path, ...]:
    cwd = Path.cwd() / ".env"
    repo = _REPO_ROOT / ".env"
    try:
        if cwd.resolve() == repo.resolve():
            return (cwd,)
    except OSError:
        return (cwd, repo)
    return (cwd, repo)


def parse_env_line(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[7:].strip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    if not _ENV_KEY.fullmatch(key):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def apply_env_file(path: Path, environ: MutableMapping[str, str]) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    applied: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key in environ:
            continue
        environ[key] = value
        applied.append(key)
    return tuple(applied)


def load_root_env(
    environ: MutableMapping[str, str] | None = None,
    paths: tuple[Path, ...] | None = None,
) -> tuple[str, ...]:
    """Load ``.env`` files without replacing variables already in ``environ``."""
    target = environ if environ is not None else os.environ
    applied: list[str] = []
    seen: set[Path] = set()
    for path in paths if paths is not None else default_env_paths():
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        applied.extend(apply_env_file(path, target))
    return tuple(applied)
