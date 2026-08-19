"""Dependency-free HTTP client for the TermyteDB service."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


class TermyteDBError(RuntimeError):
    def __init__(self, status: int, detail: object, request_id: str | None):
        super().__init__(f"TermyteDB request failed ({status}): {detail}")
        self.status, self.detail, self.request_id = status, detail, request_id


class TermyteDBClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0, retries: int = 2, auth_token: str | None = None):
        self.base_url, self.timeout, self.retries, self.auth_token = base_url.rstrip("/"), timeout, max(0, retries), auth_token

    def request(self, method: str, path: str, *, body: Mapping[str, Any] | None = None, query: Mapping[str, object] | None = None) -> Any:
        from urllib.parse import urlencode
        url = f"{self.base_url}/{path.lstrip('/')}" + (("?" + urlencode(query)) if query else "")
        data = None if body is None else json.dumps(body).encode()
        request_id = str(uuid4())
        headers = {"accept": "application/json", "x-request-id": request_id}
        if self.auth_token:
            headers["authorization"] = f"Bearer {self.auth_token}"
        if data:
            headers["content-type"] = "application/json"
        request = Request(url, data=data, method=method.upper(), headers=headers)
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                    return json.loads(raw) if raw else None
            except HTTPError as exc:
                raw = exc.read()
                try:
                    detail: object = json.loads(raw)
                except json.JSONDecodeError:
                    detail = raw.decode(errors="replace")
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt >= self.retries:
                    raise TermyteDBError(exc.code, detail, exc.headers.get("x-request-id")) from exc
            except URLError as exc:
                if attempt >= self.retries:
                    raise TermyteDBError(0, str(exc.reason), request_id) from exc
            time.sleep(min(0.25 * 2**attempt, 2.0))
        raise AssertionError("unreachable")

    def ingest(self, event: Mapping[str, Any]) -> Any:
        return self.request("POST", "/v1/events", body=event)

    def ingest_batch(self, events: list[Mapping[str, Any]]) -> Any:
        return self.request("POST", "/v1/events:batch", body={"events": events})

    def process(self, namespace_id: str, **options: object) -> Any:
        return self.request("POST", "/v1/process", body={"namespace_id": namespace_id, **options})

    def search(self, namespace_id: str, query: str, *, limit: int = 10, historical: bool = False) -> Any:
        return self.request("POST", "/v1/search", body={"namespace_id": namespace_id, "query": query, "limit": limit, "historical": historical})

    def context(self, namespace_id: str, query: str, *, token_budget: int = 500, limit: int = 10, historical: bool = False) -> Any:
        body = {"namespace_id": namespace_id, "query": query, "token_budget": token_budget, "limit": limit, "historical": historical}
        return self.request("POST", "/v1/context", body=body)

    def health(self) -> Any:
        return self.request("GET", "/health")

    def events(self, namespace_id: str, *, limit: int = 100, offset: int = 0) -> Any:
        return self.request("GET", "/v1/events", query={"namespace_id": namespace_id, "limit": limit, "offset": offset})

    def evidence(self, namespace_id: str, *, limit: int = 100, offset: int = 0) -> Any:
        return self.request("GET", "/v1/evidence", query={"namespace_id": namespace_id, "limit": limit, "offset": offset})

    def memories(self, namespace_id: str, *, limit: int = 100, offset: int = 0) -> Any:
        return self.request("GET", "/v1/memories", query={"namespace_id": namespace_id, "limit": limit, "offset": offset})

    def jobs(self, namespace_id: str, *, limit: int = 100, offset: int = 0) -> Any:
        return self.request("GET", "/v1/jobs", query={"namespace_id": namespace_id, "limit": limit, "offset": offset})

    def memory(self, namespace_id: str, memory_id: str) -> Any:
        from urllib.parse import quote
        return self.request("GET", f"/v1/memories/{quote(memory_id, safe='')}", query={"namespace_id": namespace_id})

    def history(self, namespace_id: str, memory_id: str) -> Any:
        from urllib.parse import quote
        return self.request("GET", f"/v1/memories/{quote(memory_id, safe='')}/history", query={"namespace_id": namespace_id})

    def invalidate(self, namespace_id: str, memory_id: str, reason: str) -> Any:
        from urllib.parse import quote
        return self.request("POST", f"/v1/memories/{quote(memory_id, safe='')}/invalidate", query={"namespace_id": namespace_id, "reason": reason})

    def forget(self, namespace_id: str, memory_id: str, reason: str) -> Any:
        from urllib.parse import quote
        return self.request("POST", f"/v1/memories/{quote(memory_id, safe='')}/forget", query={"namespace_id": namespace_id, "reason": reason})

    def restore(self, namespace_id: str, memory_id: str) -> Any:
        from urllib.parse import quote
        return self.request("POST", f"/v1/memories/{quote(memory_id, safe='')}/restore", query={"namespace_id": namespace_id})

    def feedback(self, feedback: Mapping[str, Any]) -> Any:
        return self.request("POST", "/v1/feedback", body=feedback)

    def export_namespace(self, namespace_id: str) -> Any:
        return self.request("GET", "/v1/export", query={"namespace_id": namespace_id})

    def delete_namespace(self, namespace_id: str) -> Any:
        from urllib.parse import quote
        return self.request("DELETE", f"/v1/namespaces/{quote(namespace_id, safe='')}")

    def metrics(self, namespace_id: str) -> Any:
        return self.request("GET", "/v1/metrics", query={"namespace_id": namespace_id})

    def integrity(self) -> Any:
        return self.request("GET", "/v1/integrity")

    def ready(self) -> Any:
        return self.request("GET", "/ready")
