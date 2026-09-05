"""Public data models used by the memory engine."""

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
ReconciliationIntent = Literal["insert", "reinforce", "update", "supersede", "dispute", "contradiction", "ignore"]
# Canonical superset used for LLM reconciliation (contradiction is alias for dispute for backwards compat)
ReconciliationAction = Literal["insert", "reinforce", "update", "supersede", "dispute", "contradiction", "ignore"]
Durability = Literal["permanent", "session", "task"]

MemoryTypeV3 = Literal["profile", "preference", "event", "assistant_knowledge", "decision", "task", "correction", "fact"]
LifecycleV3 = Literal["stable", "current", "historical", "instruction", "task"]

ExtractionStage = Literal[
    "facts",
    "preferences",
    "events",
    "decisions",
    "relationships",
    "reconciliation",
]

EXTRACTION_STAGES: tuple[ExtractionStage, ...] = (
    "facts",
    "preferences",
    "events",
    "decisions",
    "relationships",
)

RECONCILIATION_STAGE: ExtractionStage = "reconciliation"


class TemporalBlock(BaseModel):
    """Every memory carries a temporal block used at retrieval.

    `valid_from`/`valid_until` define the world-time interval for which the
    statement is true. `recorded_at` is the ingestion time. The retrieval
    pipeline weights `valid_from` recency (see Repository.search) and filters
    expired blocks for non-historical queries.
    """

    model_config = ConfigDict(extra="forbid")

    valid_from: datetime | None = None
    valid_until: datetime | None = None
    recorded_at: datetime | None = None

    def is_active(self, at: datetime | None = None) -> bool:
        now = at or utc_now()
        if self.valid_from and self.valid_from > now:
            return False
        if self.valid_until and self.valid_until <= now:
            return False
        return True

    def overlaps_year(self, year: int) -> bool:
        needle = str(year)
        return needle in (self.valid_from.isoformat() if self.valid_from else "") or needle in (
            self.valid_until.isoformat() if self.valid_until else ""
        )


def temporal_recency_score(valid_from: datetime | None, now: datetime | None = None) -> float:
    """Small 0..0.02 recency bonus; newer `valid_from` wins when scores tie."""
    if not valid_from:
        return 0.0
    current = now or utc_now()
    try:
        age_days = max(0.0, (current - valid_from).total_seconds() / 86400)
    except Exception:
        return 0.0
    # Decay over 90 days, clamped to 0..0.02 - matches former placeholder but continuous.
    return max(0.0, min(0.02, 0.02 * (1.0 - min(age_days, 90.0) / 90.0)))


TemporalIntent = Literal["latest", "historical", "earliest", "before", "after", "around", "none"]


class TemporalQuery(BaseModel):
    """Explicit temporal representation for date-aware retrieval.

    ``reference_date`` is the benchmark ``question_date`` (never the machine
    clock) so "current" means valid at question time.  Dates influence ranking,
    not hard-filtering, unless the query clearly requires it.
    """

    model_config = ConfigDict(extra="forbid")

    reference_date: datetime | None = None
    intent: TemporalIntent = "none"
    target_date: datetime | None = None
    date_range_start: datetime | None = None
    date_range_end: datetime | None = None


def temporal_valid_at_score(
    valid_from: datetime | None,
    valid_until: datetime | None,
    recorded_at: datetime | None,
    query: TemporalQuery,
) -> float:
    """Score how well a memory's temporal block matches a temporal query."""
    if query.intent == "none":
        return 0.0
    ref = query.reference_date or utc_now()
    if query.intent == "latest":
        # Prefer facts valid at the question date.
        try:
            if valid_from and valid_from > ref:
                return -0.03
            if valid_until and valid_until <= ref:
                return -0.02
            if valid_from:
                age_days = max(0.0, (ref - valid_from).total_seconds() / 86400)
                return max(0.0, 0.05 * (1.0 - min(age_days, 365.0) / 365.0))
            return 0.02
        except Exception:
            return 0.0
    if query.intent == "historical":
        # Include expired historical versions.
        if valid_until is not None:
            return 0.03
        return 0.01
    if query.intent == "earliest":
        return 0.0
    if query.intent in {"around", "before", "after"}:
        anchor = valid_from or recorded_at
        if anchor is None:
            return -0.01 if query.intent == "around" else 0.0
        try:
            anchor_naive = anchor.replace(tzinfo=None) if anchor.tzinfo else anchor
        except Exception:
            return 0.0
        if query.intent == "around" and query.date_range_start and query.date_range_end:
            try:
                start = query.date_range_start.replace(tzinfo=None) if query.date_range_start.tzinfo else query.date_range_start
                end = query.date_range_end.replace(tzinfo=None) if query.date_range_end.tzinfo else query.date_range_end
                if start <= anchor_naive < end:
                    return 0.08
                gap = min(abs((anchor_naive - start).days), abs((anchor_naive - end).days))
                if gap <= 90:
                    return 0.04 * (1.0 - gap / 90.0)
            except Exception:
                return 0.0
            return 0.0
        if query.intent == "before" and query.target_date:
            try:
                target = query.target_date.replace(tzinfo=None) if query.target_date.tzinfo else query.target_date
                return 0.05 if anchor_naive < target else -0.02
            except Exception:
                return 0.0
        if query.intent == "after" and query.target_date:
            try:
                target = query.target_date.replace(tzinfo=None) if query.target_date.tzinfo else query.target_date
                return 0.05 if anchor_naive >= target else -0.02
            except Exception:
                return 0.0
    return 0.0


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
    evidence: list[EvidenceSpan] = Field(default_factory=list, max_length=8)
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
    source_stage: ExtractionStage | None = None
    source_chunk_ids: list[str] = Field(default_factory=list, max_length=20)
    event_dates: list[datetime] = Field(default_factory=list, max_length=20)
    entities: list[str] = Field(default_factory=list, max_length=20)
    relation: Literal["insert", "reinforce", "update", "supersede", "dispute", "extends", "derives", "ignore"] | None = None
    # v3 migration fields (optional, for typed extraction)
    v3_type: MemoryTypeV3 | None = None
    v3_lifecycle: LifecycleV3 | None = None
    v3_state_key: str | None = None
    v3_source_labels: list[str] = Field(default_factory=list, max_length=8)
    v3_importance_int: int | None = Field(default=None, ge=1, le=5)


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(pattern=r"^extraction-v1$")
    prompt_version: str = Field(min_length=1, max_length=100)
    candidates: list[ExtractionCandidate] = Field(max_length=50)


class SimpleExtractionResponse(BaseModel):
    """Small, Mem0-style LLM contract used for production extraction.

    Provenance and storage metadata are created by TermyteDB after the model
    returns these memory statements.  Keeping the LLM response this small makes
    it work with a much wider range of models.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["extraction-v2"] | None = None
    memory: list[str] = Field(default_factory=list, max_length=50)
    memories: list[dict[str, Any]] = Field(default_factory=list, max_length=50)


class ExtractionMemoryV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=2000)
    source_events: list[str] = Field(min_length=1, max_length=8)
    type: MemoryTypeV3
    importance: int = Field(ge=1, le=5)
    lifecycle: LifecycleV3
    state_key: str | None = Field(default=None, max_length=200, pattern=r"^[a-z0-9_.]+\.[a-z0-9_.]+$")


class ExtractionResponseV3(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["extraction-v3"] = "extraction-v3"
    # The processor keeps at most 8-12 records per session.  Matching that
    # bound in the provider schema prevents smaller models from spending their
    # whole completion on a long, invalid list and losing the entire session.
    memories: list[ExtractionMemoryV3] = Field(default_factory=list, max_length=12)


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_id: str = Field(min_length=1)
    events: list[UUID] = Field(min_length=1, max_length=1000)
    evidence_text: dict[UUID, str]
    # Prompt-local labels keep provider output small and prevent it from
    # fabricating database UUIDs.  The processor resolves them before storage.
    event_labels: dict[str, UUID] = Field(default_factory=dict)
    chunk_labels: dict[str, str] = Field(default_factory=dict)
    # The provider never needs to choose a chunk.  Once it cites an event, the
    # engine attaches the event's trusted source chunk itself.
    event_chunk_labels: dict[UUID, str] = Field(default_factory=dict)
    event_roles: dict[UUID, Literal["user", "assistant"]] = Field(default_factory=dict)
    existing_memories: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    stage: ExtractionStage = "facts"
    # v3: explicit split between context and extractable events
    extractable_event_ids: list[UUID] = Field(default_factory=list)
    context_event_ids: list[UUID] = Field(default_factory=list)
    event_timestamps: dict[UUID, str] = Field(default_factory=dict)
    event_session_ids: dict[UUID, str] = Field(default_factory=dict)
    extraction_schema: Literal["v2", "v3"] = "v2"


class ReconciliationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_index: int = Field(ge=0)
    action: ReconciliationAction
    existing_memory_ref: str | None = Field(default=None, pattern=r"^m[0-9]+$")
    confidence: float = Field(default=0.99, ge=0, le=1)
    reason: str = Field(default="", max_length=2000)


class ReconciliationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace_id: str = Field(min_length=1)
    existing_memories: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    new_candidates: list[ExtractionCandidate] = Field(default_factory=list, max_length=50)
    stage: ExtractionStage = "reconciliation"


class ReconciliationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="reconciliation-v1", pattern=r"^reconciliation-v1$")
    prompt_version: str = Field(min_length=1, max_length=100)
    decisions: list[ReconciliationDecision] = Field(default_factory=list, max_length=50)


class CandidateReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    decision: Literal["accepted", "rejected"]
    reason: str = Field(default="", max_length=2000)
    stage: ExtractionStage | None = None


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
    job_id: UUID | None = None
    accepted: int = 0
    rejected: int = 0


class BatchEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[EventInput] = Field(min_length=1, max_length=1000)


class BatchEventResponse(BaseModel):
    receipts: list[EventReceipt]
    accepted: int = 0
    rejected: int = 0


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
    lease_seconds: int = Field(default=180, ge=1, le=3600)
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
    citations: list[EvidenceCitation] = Field(default_factory=list)
    # Simplified provenance — optional per Phase 1
    source_event_ids: list[UUID] = Field(default_factory=list)
    evidence_excerpt: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    document_date: str | None = None
    event_dates: list[str] = Field(default_factory=list)


class SessionSearchResult(BaseModel):
    """A raw conversation session returned by the fallback retrieval layer."""

    session_id: str
    event_ids: list[UUID]
    text: str
    occurred_at: str | None = None
    score: float


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
    citations: list[EvidenceCitation] = Field(default_factory=list)
    temporal: TemporalBlock | None = None
    # Simplified provenance and timestamps (Phase 1 / Phase 3)
    source_event_ids: list[UUID] = Field(default_factory=list)
    evidence_excerpt: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def id(self) -> UUID:
        return self.memory_id

    @property
    def subject(self) -> str:
        return self.subject_key


class StoredJob(BaseModel):
    id: UUID
    namespace_id: str
    event_id: UUID
    status: Literal["pending", "processing", "completed", "failed", "dead"]
    attempts: int
