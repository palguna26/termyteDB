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
    protocol_version: str = Field(default="event-v1", pattern=r"^event-v1$")
    idempotency_key: str = Field(min_length=1)
    type: str = Field(min_length=1, max_length=100)
    payload: dict[str, Any]
    occurred_at: datetime | None = None
    stream_id: str | None = None
    actor_id: str | None = Field(default=None, max_length=200)
    agent_id: str | None = Field(default=None, max_length=200)
    session_id: str | None = Field(default=None, max_length=200)
    source_id: str | None = Field(default=None, max_length=200)
    artifacts: list[ArtifactInput] = Field(default_factory=list, max_length=20)


class ArtifactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    media_type: str = Field(min_length=1, max_length=200)
    size_bytes: int = Field(ge=0, le=100_000_000)
    uri: str | None = Field(default=None, max_length=2000)
    metadata: dict[str, str] = Field(default_factory=dict, max_length=20)


MemoryKind = Literal[
    "fact",
    "decision",
    "attempt",
    "failure",
    "outcome",
    "constraint",
    "procedure",
    "task_state",
    "correction",
    "question",
]
ReconciliationIntent = Literal["insert", "reinforce", "update", "supersede", "dispute", "ignore"]
Durability = Literal["permanent", "session", "task"]


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: UUID
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    excerpt: str = Field(min_length=1, max_length=2000)


class ExtractionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: MemoryKind
    subject: str = Field(min_length=1, max_length=200)
    statement: str = Field(min_length=1, max_length=2000)
    evidence: list[EvidenceSpan] = Field(min_length=1, max_length=8)
    confidence: float = Field(ge=0, le=1)
    importance: float = Field(default=0.5, ge=0, le=1)
    durability: Durability
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    intent: ReconciliationIntent = "insert"
    existing_memory_id: UUID | None = None
    existing_memory_ref: str | None = Field(default=None, pattern=r"^m[0-9]+$")
    timestamp: datetime | None = None
    source_role: Literal["user", "assistant"] = "user"


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^extraction-v1$")
    prompt_version: str = Field(min_length=1, max_length=100)
    candidates: list[ExtractionCandidate] = Field(max_length=50)


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_id: str = Field(min_length=1)
    events: list[UUID] = Field(min_length=1, max_length=20)
    evidence_text: dict[UUID, str]
    existing_memories: list[dict[str, Any]] = Field(default_factory=list, max_length=20)


class ExtractionResult(BaseModel):
    run_id: UUID
    accepted: int
    rejected: int
    actions: dict[str, int]


class EventReceipt(BaseModel):
    event_id: UUID
    namespace_id: str
    content_hash: str
    duplicate: bool
    job_id: UUID


class BatchEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[EventInput] = Field(min_length=1, max_length=1000)


class BatchEventResponse(BaseModel):
    receipts: list[EventReceipt]


class MemoryHistoryResponse(BaseModel):
    memory_id: UUID
    versions: list[dict[str, Any]]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_id: str = Field(min_length=1)
    memory_id: UUID
    label: Literal["useful", "not_useful", "wrong", "stale"]
    note: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    id: UUID
    namespace_id: str
    memory_id: UUID
    label: str


class EpisodeStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_id: str = Field(min_length=1)
    status: Literal["active", "completed", "failed", "abandoned", "interrupted"]
    summary: str | None = Field(default=None, max_length=2000)


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_id: str = Field(min_length=1)
    limit: int = Field(default=100, ge=1, le=1000)
    lease_seconds: int = Field(default=30, ge=1, le=3600)
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)


class ProcessResponse(BaseModel):
    processed: int
    failed: int
    dead_lettered: int
    accepted: int = 0
    rejected: int = 0


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    historical: bool = False


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
    lexical_score: float = 0.0
    vector_score: float = 0.0
    component_scores: dict[str, float] = Field(default_factory=dict)
    status: str
    citations: list[EvidenceCitation]


class ContextRequest(SearchRequest):
    token_budget: int = Field(default=500, ge=1, le=10000)


class ContextResponse(BaseModel):
    request_id: UUID | None = None
    namespace_id: str
    query: str
    abstained: bool
    token_count: int
    text: str
    results: list[SearchResult]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    memory_id: UUID
    namespace_id: str
    kind: str
    subject_key: str
    status: str
    confidence: float
    importance: float
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
