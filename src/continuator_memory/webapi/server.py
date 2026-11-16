"""Stdlib HTTP API matching `web/src/data/http/httpService.ts`. No extra dependencies."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from continuator_memory.config import resolve_product_db_path
from continuator_memory.embedding import EmbeddingLoadError
from continuator_memory.models import ParseError
from continuator_memory.envfile import load_root_env
from continuator_memory.store import Store
from continuator_memory.v2.store import SemanticStore
from continuator_memory.webapi.workspace import Workspace, open_workspace

# Request body limits (Content-Length). Documented in README / quickstart.
JSON_BODY_LIMIT = 512_000
IMPORT_BODY_LIMIT = 8_000_000

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_CORS = "http://127.0.0.1:5173,http://localhost:5173"

_LOG = logging.getLogger(__name__)


def product_db_corpus_status(db_path: Path) -> str:
    """Classify a product SQLite path for serve auto-reopen.

    Returns:
        ``missing`` — file does not exist
        ``empty`` — file exists but has no conversation with claims (usable corpus)
        ``usable`` — at least one conversation has V2 claims

    Raises on a present but unreadable/corrupt database (do not fall back to fixture).
    """
    path = Path(db_path)
    if not path.is_file():
        return "missing"
    try:
        store = Store(str(path))
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError(f"failed to open product database {path}: {exc}") from exc
    try:
        listed = store.list_conversations()
        if not listed:
            return "empty"
        semantic = SemanticStore(store.connection)
        for cid, _title in listed:
            if semantic.get_claims(cid):
                return "usable"
        return "empty"
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError(f"failed to read product database {path}: {exc}") from exc
    finally:
        try:
            store.close()
        except Exception:
            pass


def resolve_serve_sample_mode(
    *,
    sample: str | None = None,
    db: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, Path]:
    """Resolve ``(sample_mode, db_path)`` for ``serve_from_env``.

    Precedence for ``sample_mode``:
    1. Explicit ``sample`` argument (CLI ``--sample``)
    2. Explicit ``CONTINUATOR_WEB_SAMPLE`` environment variable
    3. Auto: ``db`` when the product SQLite file has a usable corpus; else ``fixture``

    Never silently falls back to fixture when an explicit ``db`` mode fails later;
    auto mode only selects ``db`` after a successful corpus probe.
    """
    mapping = os.environ if env is None else env
    db_path = resolve_product_db_path(explicit=db, env=mapping)

    explicit = sample
    if explicit is None:
        raw = (mapping.get("CONTINUATOR_WEB_SAMPLE") or "").strip()
        if raw:
            explicit = raw
    if explicit is not None:
        if explicit not in {"fixture", "db"}:
            raise ValueError("sample must be 'fixture' or 'db'")
        return explicit, db_path

    status = product_db_corpus_status(db_path)
    if status == "usable":
        return "db", db_path
    return "fixture", db_path


def allow_header_for_path(path: str) -> str | None:
    """Return RFC Allow value for a known API path, or None if the path is unknown."""
    if path == "/health":
        return "GET, OPTIONS"
    if path in {"/import", "/context", "/ask", "/documents/ingest"}:
        return "POST, OPTIONS"
    if path in {"/memories", "/documents", "/sources", "/graph", "/conversations"}:
        return "GET, OPTIONS"
    if path.startswith("/memories/") or path.startswith("/evidence/"):
        return "GET, OPTIONS"
    if path.startswith("/sources/") or path.startswith("/conversations/"):
        return "GET, OPTIONS"
    if path.startswith("/documents/"):
        return "GET, DELETE, OPTIONS"
    if path.startswith("/corpus/conversations/"):
        return "DELETE, OPTIONS"
    return None


class APIHandler(BaseHTTPRequestHandler):
    workspace: Workspace
    cors_origins: frozenset[str]

    def log_message(self, format: str, *args: object) -> None:
        sys_stderr = __import__("sys").stderr
        sys_stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_PUT(self) -> None:  # noqa: N802
        return self._unsupported_method("PUT")

    def do_PATCH(self) -> None:  # noqa: N802
        return self._unsupported_method("PATCH")

    def do_HEAD(self) -> None:  # noqa: N802
        return self._unsupported_method("HEAD")

    def _unsupported_method(self, method: str) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        allow = allow_header_for_path(path)
        if allow is None:
            return self._error(404, f"{method} {path} is not a registered route")
        return self._error(405, "method not allowed", allow=allow)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if path == "/health":
                return self._json(200, self.workspace.health())
            if path == "/memories":
                types = [t for t in query.get("types", [""])[0].split(",") if t]
                include = query.get("include_superseded", ["true"])[0].lower() != "false"
                return self._json(
                    200,
                    self.workspace.list_memories(
                        text=query.get("q", [""])[0],
                        types=types or None,
                        include_superseded=include,
                    ),
                )
            if path.startswith("/memories/"):
                detail = self.workspace.get_memory(_id_from(path, "/memories/"))
                return self._json(200, detail) if detail else self._error(404, "memory not found")
            if path == "/documents":
                return self._json(200, self.workspace.list_documents())
            if path.startswith("/documents/"):
                detail = self.workspace.get_document(_id_from(path, "/documents/"))
                return self._json(200, detail) if detail else self._error(404, "document not found")
            if path.startswith("/evidence/"):
                detail = self.workspace.get_evidence_unit(_id_from(path, "/evidence/"))
                return self._json(200, detail) if detail else self._error(404, "evidence not found")
            if path == "/sources":
                return self._json(200, self.workspace.list_sources())
            if path.startswith("/sources/"):
                detail = self.workspace.get_source(_id_from(path, "/sources/"))
                return self._json(200, detail) if detail else self._error(404, "source not found")
            if path == "/graph":
                return self._json(200, self.workspace.get_graph())
            if path == "/conversations":
                return self._json(200, self.workspace.list_conversations())
            if path.startswith("/conversations/"):
                detail = self.workspace.get_conversation(_id_from(path, "/conversations/"))
                return self._json(200, detail) if detail else self._error(404, "conversation not found")
            return self._method_or_path_error("GET", path)
        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception:  # pragma: no cover - unexpected
            _LOG.exception("unexpected GET failure for %s", path)
            return self._error(500, "internal server error")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/import":
            return self._import()
        if path == "/context":
            return self._context()
        if path == "/ask":
            return self._ask()
        if path == "/documents/ingest":
            return self._ingest_document()
        return self._method_or_path_error("POST", path)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path.startswith("/documents/"):
                result = self.workspace.delete_document(_id_from(path, "/documents/"))
                return self._json(200, result)
            if path.startswith("/corpus/conversations/"):
                try:
                    result = self.workspace.delete_corpus_conversation(
                        _id_from(path, "/corpus/conversations/")
                    )
                except ValueError:
                    raise
                except Exception:
                    # Logged inside Workspace; do not leak internal exception text.
                    return self._error(500, "corpus conversation deletion failed")
                return self._json(200, result)
            return self._method_or_path_error("DELETE", path)
        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception:  # pragma: no cover
            _LOG.exception("unexpected DELETE failure for %s", path)
            return self._error(500, "internal server error")

    def _method_or_path_error(self, method: str, path: str) -> None:
        allow = allow_header_for_path(path)
        if allow is not None and method not in {part.strip() for part in allow.split(",")}:
            return self._error(405, "method not allowed", allow=allow)
        return self._error(404, f"{method} {path} is not a registered route")

    def _ask(self) -> None:
        try:
            body = self._read_json()
            if not isinstance(body, dict):
                return self._error(400, "JSON object required")
            question = body.get("question")
            if not isinstance(question, str):
                return self._error(400, "question must be a string")
            conversation_id = body.get("conversation_id")
            if conversation_id is not None and not isinstance(conversation_id, str):
                return self._error(400, "conversation_id must be a string")
            mode = body.get("retrieval_mode", "baseline")
            if not isinstance(mode, str):
                return self._error(400, "retrieval_mode must be a string")
            generate = body.get("generate")
            if generate is not None and not isinstance(generate, bool):
                return self._error(400, "generate must be a boolean")
            generation = body.get("generation")
            result = self.workspace.ask(
                question,
                conversation_id=conversation_id,
                retrieval_mode=mode,
                generate=generate,
                generation=generation,
            )
            return self._json(200, result)
        except ValueError as exc:
            return self._error(400, str(exc))
        except FileNotFoundError as exc:
            return self._error(400, str(exc))
        except EmbeddingLoadError as exc:
            return self._error(503, str(exc))
        except Exception:  # pragma: no cover
            _LOG.exception("unexpected /ask failure")
            return self._error(500, "internal server error")

    def _context(self) -> None:
        """Retrieve a Context Pack (memories + evidence + formatted context; no answering LLM)."""
        try:
            body = self._read_json()
            if not isinstance(body, dict):
                return self._error(400, "JSON object required")
            # Accept query (Context Pack API) or question (same as /ask).
            query = body.get("query", body.get("question"))
            if not isinstance(query, str):
                return self._error(400, "query (or question) must be a string")
            mode = body.get("mode", body.get("retrieval_mode", "baseline"))
            if not isinstance(mode, str):
                return self._error(400, "mode (or retrieval_mode) must be a string")
            limit = body.get("limit", body.get("k"))
            if limit is not None and (
                not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
            ):
                return self._error(400, "limit (or k) must be a positive integer")
            generation = body.get("generation")
            pack = self.workspace.context(
                query,
                retrieval_mode=mode,
                generation=generation,
                k=limit,
            )
            return self._json(200, pack.to_dict())
        except ValueError as exc:
            return self._error(400, str(exc))
        except FileNotFoundError as exc:
            return self._error(400, str(exc))
        except EmbeddingLoadError as exc:
            return self._error(503, str(exc))
        except Exception as exc:  # pragma: no cover
            return self._error(500, "internal server error")

    def _import(self) -> None:
        try:
            body = self._read_json(max_bytes=IMPORT_BODY_LIMIT)
            if not isinstance(body, dict):
                return self._error(400, "JSON object required")
            text = body.get("text")
            if not isinstance(text, str):
                return self._error(400, "text must be a string")
            filename = body.get("filename", "conversation.txt")
            if not isinstance(filename, str) or not filename.strip():
                return self._error(400, "filename must be a non-empty string")
            extract = body.get("extract", "messages")
            if not isinstance(extract, str):
                return self._error(400, "extract must be a string")
            result = self.workspace.import_text(text, filename=filename.strip(), extract=extract)
            return self._json(200, result)
        except (ValueError, ParseError) as exc:
            return self._error(400, str(exc))
        except FileNotFoundError as exc:
            return self._error(400, str(exc))
        except Exception:  # pragma: no cover
            _LOG.exception("unexpected import failure")
            return self._error(500, "internal server error")

    def _ingest_document(self) -> None:
        """Ingest Markdown text into the product EvidenceStore (JSON body)."""
        try:
            body = self._read_json()
            if not isinstance(body, dict):
                return self._error(400, "JSON object required")
            text = body.get("text")
            if not isinstance(text, str):
                return self._error(400, "text must be a string")
            filename = body.get("filename", "document.md")
            if filename is not None and not isinstance(filename, str):
                return self._error(400, "filename must be a string")
            source_uri = body.get("sourceUri")
            if source_uri is not None and not isinstance(source_uri, str):
                return self._error(400, "sourceUri must be a string")
            result = self.workspace.ingest_markdown_text(
                text,
                filename=(filename or "document.md"),
                source_uri=source_uri,
            )
            return self._json(200, result)
        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception:  # pragma: no cover
            _LOG.exception("unexpected document ingest failure")
            return self._error(500, "internal server error")

    def _read_json(self, max_bytes: int = JSON_BODY_LIMIT) -> Any:
        raw_len = self.headers.get("Content-Length") or "0"
        try:
            length = int(raw_len)
        except ValueError as exc:
            raise ValueError("Content-Length must be an integer") from exc
        if length < 0:
            raise ValueError("Content-Length must be non-negative")
        if length > max_bytes:
            raise ValueError("request body too large")
        if length == 0:
            # No body (or missing Content-Length): treat as empty JSON object.
            return {}
        raw = self.rfile.read(length)
        if len(raw) < length:
            raise ValueError("incomplete request body")
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc

    def _json(self, status: int, payload: Any, *, allow: str | None = None) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors()
        if allow:
            self.send_header("Allow", allow)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, *, allow: str | None = None) -> None:
        self._json(status, {"error": message}, allow=allow)

    def _cors(self) -> None:
        origin = self.headers.get("Origin", "")
        if "*" in self.cors_origins:
            self.send_header("Access-Control-Allow-Origin", origin or "*")
        elif origin in self.cors_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")


def _id_from(path: str, prefix: str) -> str:
    from urllib.parse import unquote

    return unquote(path[len(prefix):])


def make_server(
    workspace: Workspace,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    cors: str | None = None,
) -> ThreadingHTTPServer:
    origins = frozenset(
        o.strip() for o in (cors or os.environ.get("CONTINUATOR_WEB_CORS", DEFAULT_CORS)).split(",") if o.strip()
    )

    class Bound(APIHandler):
        pass

    Bound.workspace = workspace
    Bound.cors_origins = origins
    return ThreadingHTTPServer((host, port), Bound)


def format_serve_startup_line(
    workspace: Workspace,
    *,
    host: str,
    port: int,
) -> str:
    """Human-readable startup line distinguishing fixture demo vs product corpus.

    Health fields (GET /health) used by docs:
    - ``origin`` / ``sampleId``: ``fixture`` (built-in demo), ``imported`` (POST /import),
      or ``sqlite`` (loaded from DB).
    - ``importedFilename``: set when origin is ``imported``.
    - ``conversationId``: active corpus id (``web-sample`` for fixture; ``imported`` after import).
    """
    health = workspace.health()
    origin = health.get("origin") or health.get("sampleId") or "unknown"
    n_mem = health.get("nMemories")
    cid = health.get("conversationId")
    base = f"continuator-webapi on http://{host}:{port}"
    if origin == "fixture":
        return (
            f"{base} [fixture demo corpus — not your data; "
            f"POST /import or ContinuatorClient.import_conversation before trusting /context] "
            f"(origin={origin}, conversationId={cid}, memories={n_mem}, k={workspace.k})"
        )
    if origin == "imported":
        fname = health.get("importedFilename") or "?"
        return (
            f"{base} [imported product corpus: {fname}] "
            f"(origin={origin}, conversationId={cid}, memories={n_mem}, k={workspace.k})"
        )
    return (
        f"{base} [product corpus] "
        f"(origin={origin}, conversationId={cid}, memories={n_mem}, k={workspace.k})"
    )


def serve_from_env(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    sample: str | None = None,
    db: str | None = None,
    classifications: str | None = None,
    conversation_id: str | None = None,
    k: int | None = None,
) -> None:
    load_root_env()
    try:
        sample_mode, db_path = resolve_serve_sample_mode(sample=sample, db=db)
    except (ValueError, RuntimeError) as exc:
        print(f"continuator-webapi startup error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc
    class_path = classifications or os.environ.get("CONTINUATOR_BPRIME_CLASSIFICATIONS")
    conv = conversation_id or os.environ.get("CONTINUATOR_CONVERSATION_ID")
    k_raw = os.environ.get("CONTINUATOR_TOP_K")
    resolved_k = k if k is not None else (int(k_raw) if k_raw else None)
    budget_raw = os.environ.get("CONTINUATOR_BUDGET_CHARS")
    budget = int(budget_raw) if budget_raw else None
    try:
        workspace = open_workspace(
            sample=sample_mode,
            db_path=db_path if sample_mode == "db" else None,
            conversation_id=conv,
            classifications_path=Path(class_path) if class_path else None,
            k=resolved_k,
            budget_chars=budget,
        )
    except Exception as exc:
        # Explicit or auto-selected db mode must not fall back to fixture.
        print(
            f"continuator-webapi failed to open workspace "
            f"(sample={sample_mode}, db={db_path}): {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2) from exc
    httpd = make_server(workspace, host=host, port=port)
    print(format_serve_startup_line(workspace, host=host, port=port), flush=True)
    httpd.serve_forever()
