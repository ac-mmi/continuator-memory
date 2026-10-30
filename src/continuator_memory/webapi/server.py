"""Stdlib HTTP API matching `web/src/data/http/httpService.ts`. No extra dependencies."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from continuator_memory.models import ParseError
from continuator_memory.envfile import load_root_env
from continuator_memory.webapi.workspace import Workspace, open_workspace

IMPORT_BODY_LIMIT = 8_000_000

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_CORS = "http://127.0.0.1:5173,http://localhost:5173"


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
            return self._error(404, f"GET {path} is not a registered route")
        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # pragma: no cover - unexpected
            return self._error(500, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/import":
            return self._import()
        if path != "/ask":
            return self._error(404, f"POST {path} is not a registered route")
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
        except Exception as exc:  # pragma: no cover
            return self._error(500, str(exc))

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
        except Exception as exc:  # pragma: no cover
            return self._error(500, str(exc))

    def _read_json(self, max_bytes: int = 512_000) -> Any:
        length = int(self.headers.get("Content-Length") or "0")
        if length > max_bytes:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _cors(self) -> None:
        origin = self.headers.get("Origin", "")
        if "*" in self.cors_origins:
            self.send_header("Access-Control-Allow-Origin", origin or "*")
        elif origin in self.cors_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
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
    sample_mode = sample or os.environ.get("CONTINUATOR_WEB_SAMPLE", "fixture")
    db_path = Path(db or os.environ.get("CONTINUATOR_DB_PATH", "data/continuator.db"))
    class_path = classifications or os.environ.get("CONTINUATOR_BPRIME_CLASSIFICATIONS")
    conv = conversation_id or os.environ.get("CONTINUATOR_CONVERSATION_ID")
    k_raw = os.environ.get("CONTINUATOR_TOP_K")
    resolved_k = k if k is not None else (int(k_raw) if k_raw else None)
    budget_raw = os.environ.get("CONTINUATOR_BUDGET_CHARS")
    budget = int(budget_raw) if budget_raw else None
    workspace = open_workspace(
        sample=sample_mode,
        db_path=db_path if sample_mode == "db" else None,
        conversation_id=conv,
        classifications_path=Path(class_path) if class_path else None,
        k=resolved_k,
        budget_chars=budget,
    )
    httpd = make_server(workspace, host=host, port=port)
    print(
        f"continuator-webapi {sample_mode} on http://{host}:{port} "
        f"(memories={workspace.health()['nMemories']}, k={workspace.k})",
        flush=True,
    )
    httpd.serve_forever()
