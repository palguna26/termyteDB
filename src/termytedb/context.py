from __future__ import annotations

from .repository import Repository
from .schemas import ContextResponse, SearchResult


def token_count(text: str) -> int:
    return len(text.split())


def build_context(repository: Repository, namespace_id: str, query: str, limit: int, token_budget: int) -> ContextResponse:
    results = repository.search(namespace_id, query, limit)
    selected: list[SearchResult] = []
    chunks: list[str] = []
    used = 0
    for result in results:
        citation = result.citations[0] if result.citations else None
        line = f"[{result.kind}] {result.statement}"
        if citation:
            line += f" (evidence:{citation.event_id})"
        cost = token_count(line)
        if used + cost > token_budget:
            continue
        selected.append(result)
        chunks.append(line)
        used += cost
    return ContextResponse(
        namespace_id=namespace_id,
        query=query,
        abstained=not selected,
        token_count=used,
        text="\n".join(chunks),
        results=selected,
    )
