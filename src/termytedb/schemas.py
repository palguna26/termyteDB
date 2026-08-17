from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class EventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    type: str = Field(min_length=1, max_length=100)
    payload: dict[str, Any]
    occurred_at: datetime | None = None
    stream_id: str | None = None


class EventReceipt(BaseModel):
    event_id: UUID
    namespace_id: str
    content_hash: str
    duplicate: bool
    job_id: UUID


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_id: str = Field(min_length=1)
    limit: int = Field(default=100, ge=1, le=1000)
    lease_seconds: int = Field(default=30, ge=1, le=3600)


class ProcessResponse(BaseModel):
    processed: int
    failed: int
    dead_lettered: int


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)


class EvidenceCitation(BaseModel):
    event_id: UUID
    start_offset: int
    end_offset: int
    excerpt: str


class SearchResult(BaseModel):
    memory_id: UUID
    memory_version_id: UUID
    statement: str
    kind: str
    score: float
    status: str
    citations: list[EvidenceCitation]


class ContextRequest(SearchRequest):
    token_budget: int = Field(default=500, ge=1, le=10000)


class ContextResponse(BaseModel):
    namespace_id: str
    query: str
    abstained: bool
    token_count: int
    text: str
    results: list[SearchResult]


class MemoryResponse(BaseModel):
    memory_id: UUID
    namespace_id: str
    kind: str
    subject_key: str
    status: str
    confidence: float
    current_version_id: UUID
    version: int
    statement: str
    citations: list[EvidenceCitation]


class StoredJob(BaseModel):
    id: UUID
    namespace_id: str
    event_id: UUID
    status: Literal["pending", "processing", "completed", "failed", "dead"]
    attempts: int
