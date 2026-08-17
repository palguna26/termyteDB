"""Small dependency-free HTTP client for the TermyteDB service."""

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
        self.status = status
        self.detail = detail
        self.request_id = request_id


class TermyteDBClient:
    def __init__(self, base_url: str, *, timeout: float = 30.0, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = max(0, retries)

    def request(self, method: str, path: str, *, body: Mapping[str, Any] | None = None, query: Mapping[str, object] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if query:
            from urllib.parse import urlencode
            url += "?" + urlencode(query)
        data = None if body is None else json.dumps(body).encode()
        request_id = str(uuid4())
        request = Request(url, data=data, method=method.upper(), headers={"accept": "application/json", "x-request-id": request_id, **({"content-type": "application/json"} if data else {})})
        for attempt in range(self.retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    content = response.read()
                    return json.loads(content) if content else None
            except HTTPError as exc:
                raw = exc.read()
                detail: object = raw.decode(errors="replace")
                try:
                    detail = json.loads(raw)
                except json.JSONDecodeError:
                    pass
                if exc.code not in {408, 429, 500, 502, 503, 504} or attempt >= self.retries:
                    raise TermyteDBError(exc.code, detail, exc.headers.get("x-request-id")) from exc
            except URLError as exc:
                if attempt >= self.retries:
                    raise TermyteDBError(0, str(exc.reason), request_id) from exc
            time.sleep(min(0.25 * 2**attempt, 2.0))
        raise AssertionError("unreachable")

    def ingest(self, event: Mapping[str, Any]) -> Any:
        return self.request("POST", "/v1/events", body=event)

    def process(self, namespace_id: str, **options: object) -> Any:
        return self.request("POST", "/v1/process", body={"namespace_id": namespace_id, **options})

    def search(self, namespace_id: str, query: str, *, limit: int = 10, historical: bool = False) -> Any:
        return self.request("POST", "/v1/search", body={"namespace_id": namespace_id, "query": query, "limit": limit, "historical": historical})

    def context(self, namespace_id: str, query: str, *, token_budget: int = 500, limit: int = 10, historical: bool = False) -> Any:
        return self.request("POST", "/v1/context", body={"namespace_id": namespace_id, "query": query, "token_budget": token_budget, "limit": limit, "historical": historical})

    def health(self) -> Any:
        return self.request("GET", "/health")
