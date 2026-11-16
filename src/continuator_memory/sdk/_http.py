"""Minimal stdlib HTTP helpers for ContinuatorClient."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class ContinuatorAPIError(Exception):
    """Raised when the Continuator HTTP API returns an error or the request fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body


def join_url(base_url: str, path: str, query: dict[str, str] | None = None) -> str:
    base = base_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    url = f"{base}{path}"
    if query:
        return f"{url}?{urllib.parse.urlencode(query)}"
    return url


def encode_path_segment(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def request_json(
    method: str,
    url: str,
    *,
    body: dict[str, Any] | list[Any] | None = None,
    timeout: float = 60.0,
) -> Any:
    """Perform one JSON HTTP request. Never retries."""
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = getattr(resp, "status", None)
            if not raw:
                raise ContinuatorAPIError(
                    "Continuator API returned an empty response body "
                    "(expected JSON)",
                    status_code=status,
                    body=None,
                )
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ContinuatorAPIError(
                    f"invalid JSON response from Continuator API: {exc}",
                    status_code=status,
                    body=None,
                ) from None
    except ContinuatorAPIError:
        raise
    except urllib.error.HTTPError as exc:
        raw = exc.read() if hasattr(exc, "read") else b""
        parsed: dict[str, Any] | None
        message = f"Continuator API HTTP {exc.code}"
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else None
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
            message = parsed["error"]
        raise ContinuatorAPIError(message, status_code=exc.code, body=parsed if isinstance(parsed, dict) else None) from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        detail = str(reason) if reason is not None else str(exc)
        raise ContinuatorAPIError(
            f"failed to connect to Continuator API: {detail}",
            status_code=None,
            body=None,
        ) from None
    except TimeoutError:
        raise ContinuatorAPIError(
            "Continuator API request timed out",
            status_code=None,
            body=None,
        ) from None
    except socket.timeout:
        raise ContinuatorAPIError(
            "Continuator API request timed out",
            status_code=None,
            body=None,
        ) from None
