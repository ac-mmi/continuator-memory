"""Synchronous Continuator HTTP client (stdlib urllib)."""

from __future__ import annotations

from typing import Any

from continuator_memory.sdk._http import (
    ContinuatorAPIError,
    encode_path_segment,
    join_url,
    request_json,
)


class _DocumentsAPI:
    def __init__(self, client: ContinuatorClient):
        self._client = client

    def list(self) -> list[dict[str, Any]]:
        return self._client._request("GET", "/documents")

    def get(self, document_id: str) -> dict[str, Any]:
        return self._client._request(
            "GET", f"/documents/{encode_path_segment(document_id)}",
        )

    def delete(self, document_id: str) -> dict[str, Any]:
        return self._client._request(
            "DELETE", f"/documents/{encode_path_segment(document_id)}",
        )

    def ingest_markdown(
        self,
        text: str,
        *,
        filename: str = "document.md",
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text, "filename": filename}
        if source_uri is not None:
            body["sourceUri"] = source_uri
        return self._client._request("POST", "/documents/ingest", body=body)


class _EvidenceAPI:
    def __init__(self, client: ContinuatorClient):
        self._client = client

    def get(self, evidence_id: str) -> dict[str, Any]:
        return self._client._request(
            "GET", f"/evidence/{encode_path_segment(evidence_id)}",
        )


class _MemoriesAPI:
    def __init__(self, client: ContinuatorClient):
        self._client = client

    def list(
        self,
        *,
        q: str | None = None,
        types: list[str] | None = None,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        query: dict[str, str] = {
            "include_superseded": "true" if include_superseded else "false",
        }
        if q:
            query["q"] = q
        if types:
            query["types"] = ",".join(types)
        return self._client._request("GET", "/memories", query=query)

    def get(self, memory_id: str) -> dict[str, Any]:
        return self._client._request(
            "GET", f"/memories/{encode_path_segment(memory_id)}",
        )


class _SourcesAPI:
    def __init__(self, client: ContinuatorClient):
        self._client = client

    def list(self) -> list[dict[str, Any]]:
        return self._client._request("GET", "/sources")

    def get(self, conversation_id: str) -> dict[str, Any]:
        return self._client._request(
            "GET", f"/sources/{encode_path_segment(conversation_id)}",
        )


class _CorpusAPI:
    def __init__(self, client: ContinuatorClient):
        self._client = client

    def delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        return self._client._request(
            "DELETE",
            f"/corpus/conversations/{encode_path_segment(conversation_id)}",
        )


class ContinuatorClient:
    """Minimal sync client for a running Continuator HTTP server.

    Returns raw JSON dicts/lists from the API without renaming fields.
    Context Pack responses currently use ``schemaVersion`` ``\"1\"``; unknown
    fields are preserved. Future schema versions are not rejected here.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8765", *, timeout: float = 60.0):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = float(timeout)
        self.documents = _DocumentsAPI(self)
        self.evidence = _EvidenceAPI(self)
        self.memories = _MemoriesAPI(self)
        self.sources = _SourcesAPI(self)
        self.corpus = _CorpusAPI(self)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> Any:
        url = join_url(self.base_url, path, query)
        return request_json(method, url, body=body, timeout=self.timeout)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def import_conversation(
        self,
        text: str,
        *,
        filename: str = "conversation.txt",
        extract: str = "messages",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/import",
            body={"text": text, "filename": filename, "extract": extract},
        )

    def context(
        self,
        query: str,
        *,
        mode: str = "baseline",
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve a Context Pack (schema v1 today). Returns the complete raw dict."""
        body: dict[str, Any] = {"query": query, "mode": mode}
        if limit is not None:
            body["limit"] = limit
        return self._request("POST", "/context", body=body)

    def ingest_markdown(
        self,
        text: str,
        *,
        filename: str = "document.md",
        source_uri: str | None = None,
    ) -> dict[str, Any]:
        """Ingest Markdown via ``POST /documents/ingest`` (same DB as document APIs)."""
        return self.documents.ingest_markdown(
            text, filename=filename, source_uri=source_uri,
        )


__all__ = ["ContinuatorAPIError", "ContinuatorClient"]
