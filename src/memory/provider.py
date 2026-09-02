from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config.prompts import (
    build_extraction_prompt as _build_extraction_prompt,
)
from ..config.prompts import (
    build_extraction_v3_prompt as _build_extraction_v3_prompt,
)
from ..config.prompts import (
    build_reconciliation_prompt as _build_reconciliation_prompt,
)
from ..config.prompts import (
    build_session_summary_prompt as _build_session_summary_prompt,
)
from ..config.prompts import (
    clean_json_response as _clean_json_response,
)
from ..config.prompts import (
    extraction_response_format as _extraction_response_format,
)
from ..config.prompts import (
    extraction_response_format_v3 as _extraction_response_format_v3,
)
from ..config.prompts import (
    get_extraction_schema as _get_extraction_schema,
)
from ..config.prompts import (
    reconciliation_response_format as _reconciliation_response_format,
)

# Re-export for backward compatibility: existing imports `from memory.provider
# import build_extraction_prompt` continue to work while config is source of truth.
from ..models import (
    ExtractionCandidate,
    ExtractionMemoryV3,
    ExtractionRequest,
    ExtractionResponse,
    ExtractionResponseV3,
    ReconciliationRequest,
    ReconciliationResponse,
    SimpleExtractionResponse,
)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    value = str(value).strip()
    try:
        # Seconds as integer/float
        return max(0.0, float(value))
    except ValueError:
        pass
    # HTTP date - crude fallback: treat as 5s
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(value)
        if dt is not None:
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            delta = (dt - now).total_seconds()
            return max(0.0, delta)
    except Exception:
        return None
    return None


def _openrouter_chat(base_url: str, api_key: str | None, body: dict[str, object], *, title: str, timeout: float) -> tuple[dict[str, Any], bytes]:
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    raw = urlopen(
        Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
                "http-referer": "https://termyte.dev",
                "X-OpenRouter-Title": title,
            },
            method="POST",
        ),
        timeout=timeout,
    ).read()
    return json.loads(raw.decode("utf-8")), raw


def _retry_sleep(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(60.0, max(0.5, retry_after))
    # Exponential backoff: 1, 2, 4, 8 ... capped at 30s, with small jitter
    base = min(30.0, 1.0 * (2**attempt))
    # Add tiny jitter via hash of time to avoid thundering herd without random import
    jitter = (hash(str(time.perf_counter())) % 100) / 1000.0
    return base + jitter


def _get_retry_budget() -> int:
    raw = os.environ.get("TERMYTEDB_EXTRACTION_RETRIES")
    if raw is None:
        # One retry keeps ingestion resilient without multiplying benchmark cost.
        return 1
    try:
        return max(0, min(6, int(raw.strip())))
    except ValueError:
        return 3


def _cancellable_sleep(seconds: float, cancellation: Callable[[], bool] | None, started: float, timeout_seconds: float) -> None:
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if cancellation and cancellation():
            raise ProviderError("extraction cancelled", retryable=True, error_class="cancelled")
        if time.perf_counter() - started >= timeout_seconds:
            raise ProviderError("extraction timeout", retryable=True, error_class="timeout")
        # sleep in small increments to stay cancellation-aware
        time.sleep(min(0.05, end - time.perf_counter()))


def _message_text(payload: dict[str, Any], *, text_parts_only: bool = False) -> str:
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict) and (not text_parts_only or part.get("type", "text") in {"text", "output_text"})
        )
    if isinstance(content, dict):
        return json.dumps(content)
    return str(content or "")


def clean_json_response(value: str) -> str:
    """Backward-compat wrapper - canonical implementation lives in `config.prompts`."""
    return _clean_json_response(value)


def extraction_response_format() -> dict[str, object]:
    """Backward-compat wrapper - canonical implementation lives in `config.prompts`."""
    # Dispatch based on env schema
    if _get_extraction_schema() == "v3":
        return _extraction_response_format_v3()
    return _extraction_response_format()


def build_extraction_prompt(request: ExtractionRequest) -> str:
    """Backward-compat wrapper - canonical implementation lives in `config.prompts`."""
    return _build_extraction_prompt(request)


def build_extraction_v3_prompt(request: ExtractionRequest) -> str:
    return _build_extraction_v3_prompt(request)


def _simple_subject(statement: str) -> str:
    words = [word.strip(".,:;!?()[]{}\"'").casefold() for word in statement.split()]
    words = [word for word in words if word]
    return " ".join(words[:6]) or "memory"


def _v3_type_to_kind(t: str) -> str:
    mapping = {
        "profile": "fact",
        "preference": "fact",
        "event": "fact",
        "assistant_knowledge": "fact",
        "decision": "decision",
        "task": "task_state",
        "correction": "correction",
        "fact": "fact",
    }
    return mapping.get(t, "fact")


def _v3_lifecycle_to_durability(l: str) -> str:
    if l == "task":
        return "task"
    return "permanent"


def _v3_importance_to_float(v: int) -> float:
    return max(0.0, min(1.0, v / 5.0))


def _normalize_state_key(value: str | None) -> str | None:
    if not value:
        return None
    normalized = " ".join(value.strip().split()).casefold()
    # must contain dot per pattern; ensure normalized matches pattern
    if "." not in normalized:
        return None
    return normalized


def _v3_response_to_extraction(value: Any, request: ExtractionRequest | None = None) -> ExtractionResponse:
    """Convert extraction-v3 JSON into storage candidates with deterministic mapping."""
    try:
        parsed = ExtractionResponseV3.model_validate(value)
    except Exception:
        # Fallback to simple if validation fails (non-strict tolerance)
        # Try tolerant parsing: ignore unknown fields
        try:
            # Manual tolerant parse
            raw_mems = value.get("memories", []) if isinstance(value, dict) else []
            candidates_tolerant: list[ExtractionCandidate] = []
            for fields in raw_mems:
                if not isinstance(fields, dict):
                    continue
                statement = " ".join(str(fields.get("statement", "")).split())
                source_events = fields.get("source_events") or []
                if not statement or not source_events:
                    continue
                t = str(fields.get("type", "fact"))
                imp = int(fields.get("importance", 3))
                life = str(fields.get("lifecycle", "stable"))
                sk = _normalize_state_key(fields.get("state_key"))
                kind = _v3_type_to_kind(t)
                subject = sk if sk and life == "current" else _simple_subject(statement)
                if sk and life == "current":
                    subject = sk
                try:
                    candidates_tolerant.append(
                        ExtractionCandidate(
                            kind=kind,  # type: ignore[arg-type]
                            subject=subject,
                            statement=statement,
                            evidence=[],
                            confidence=0.9,
                            importance=_v3_importance_to_float(max(1, min(5, imp))),
                            durability=_v3_lifecycle_to_durability(life),  # type: ignore[arg-type]
                            v3_type=t,  # type: ignore[arg-type]
                            v3_lifecycle=life,  # type: ignore[arg-type]
                            v3_state_key=sk,
                            v3_source_labels=[str(x) for x in source_events],
                            v3_importance_int=max(1, min(5, imp)),
                        )
                    )
                except Exception:
                    continue
            return ExtractionResponse(schema_version="extraction-v1", prompt_version="extraction-v3-tolerant", candidates=candidates_tolerant)
        except Exception:
            raise
    candidates: list[ExtractionCandidate] = []
    for mem in parsed.memories:
        kind = _v3_type_to_kind(mem.type)
        subject = _normalize_state_key(mem.state_key) if mem.state_key and mem.lifecycle == "current" else _simple_subject(mem.statement)
        if mem.state_key and mem.lifecycle == "current":
            subject = _normalize_state_key(mem.state_key) or _simple_subject(mem.statement)
        try:
            candidates.append(
                ExtractionCandidate(
                    kind=kind,  # type: ignore[arg-type]
                    subject=subject,
                    statement=mem.statement,
                    evidence=[],
                    confidence=0.9,
                    importance=_v3_importance_to_float(mem.importance),
                    durability=_v3_lifecycle_to_durability(mem.lifecycle),  # type: ignore[arg-type]
                    v3_type=mem.type,
                    v3_lifecycle=mem.lifecycle,
                    v3_state_key=_normalize_state_key(mem.state_key),
                    v3_source_labels=list(mem.source_events),
                    v3_importance_int=mem.importance,
                )
            )
        except Exception:
            continue
    return ExtractionResponse(schema_version="extraction-v1", prompt_version="extraction-v3", candidates=candidates)


def _simple_response_to_extraction(value: Any, request: ExtractionRequest | None = None) -> ExtractionResponse:
    """Convert the tiny LLM response into the existing storage contract.

    Source event links are assigned by the processor.  We intentionally do not
    ask the LLM to fabricate IDs, offsets, excerpts, or database actions.
    """
    if isinstance(value, dict) and value.get("schema_version") == "extraction-v3":
        return _v3_response_to_extraction(value, request)
    if isinstance(value, dict) and value.get("schema_version") == "extraction-v1":
        return ExtractionResponse.model_validate(value)
    if isinstance(value, dict) and value.get("schema_version") == "extraction-v2":
        candidates: list[ExtractionCandidate] = []
        event_labels = request.event_labels if request else {}
        chunk_labels = request.chunk_labels if request else {}
        event_chunk_labels = request.event_chunk_labels if request else {}
        roles = request.event_roles if request else {}
        for fields in value.get("memories", []):
            if not isinstance(fields, dict):
                continue
            # The model only supplies the memory and compact event label.  It
            # does not need to reproduce fragile quotes, roles, or chunk ids.
            statement = " ".join(str(fields.get("memory") or fields.get("statement", "")).split())
            event_id = event_labels.get(str(fields.get("source_event") or fields.get("source_event_label", "")))
            if not statement or event_id is None or request is None:
                continue
            source = request.evidence_text.get(event_id, "")
            if not source:
                continue
            quote = str(fields.get("evidence_quote", "")).strip()
            if quote:
                start = source.find(quote)
                if start < 0:
                    continue
            else:
                # Trusted raw event text is evidence.  Never substitute the
                # generated statement as evidence.
                start = 0
                quote = source[:2000]
            chunk_label = str(fields.get("source_chunk_label", "")) or event_chunk_labels.get(event_id, "")
            chunk_id = chunk_labels.get(chunk_label)
            role = roles.get(event_id, "user")
            entity = " ".join(str(fields.get("entity_key", "")).casefold().split())
            predicate = " ".join(str(fields.get("predicate_key", "")).casefold().split())
            subject = f"{entity}::{predicate}" if entity and predicate else _simple_subject(statement)
            kind = str(fields.get("kind", "fact"))
            if kind not in {"fact", "decision", "attempt", "failure", "outcome", "constraint", "procedure", "task_state", "correction", "question"}:
                kind = "fact"
            action = str(fields.get("action", "insert"))
            if action not in {"insert", "reinforce", "update", "supersede", "ignore"}:
                action = "insert"
            update: dict[str, Any] = {
                "kind": kind,
                "subject": subject,
                "statement": statement,
                "evidence": [{"event_id": event_id, "start_offset": start, "end_offset": start + len(quote), "excerpt": quote}],
                "confidence": float(fields.get("confidence", 0.9)),
                "importance": float(fields.get("importance", 0.5)),
                "durability": str(fields.get("durability", "permanent")),
                "intent": action,
                "source_role": role,
                "source_chunk_ids": [chunk_id] if chunk_id else [],
                "entities": [entity] if entity else [],
            }
            if fields.get("event_time"):
                update["valid_from"] = fields["event_time"]
            try:
                candidates.append(ExtractionCandidate.model_validate(update))
            except (TypeError, ValueError):
                continue
        return ExtractionResponse(schema_version="extraction-v1", prompt_version="grounded-v2-simple", candidates=candidates)
    simple = SimpleExtractionResponse.model_validate(value)
    candidates = []
    seen: set[str] = set()
    structured = simple.memories
    values = structured or [{"statement": raw} for raw in simple.memory]
    for item in values:
        raw = item.get("statement", "") if isinstance(item, dict) else item
        statement = " ".join(str(raw).strip().split())
        key = statement.casefold()
        if len(statement) < 3 or key in seen:
            continue
        seen.add(key)
        fields = item if isinstance(item, dict) else {}
        candidates.append(
            ExtractionCandidate(
                kind="fact",
                subject=str(fields.get("subject") or _simple_subject(statement)),
                statement=statement,
                evidence=[],
                confidence=0.9,
                importance=0.5,
                durability="permanent",
                intent="insert",
                source_role="user",
                source_chunk_ids=[str(value) for value in fields.get("source_chunk_ids", [])],
                entities=[str(value) for value in fields.get("entities", [])],
                relation=fields.get("relation"),
            )
        )
    return ExtractionResponse(schema_version="extraction-v1", prompt_version="simple-v1", candidates=candidates)


def _empty_extraction_response() -> ExtractionResponse:
    return ExtractionResponse(schema_version="extraction-v1", prompt_version="simple-v1", candidates=[])


def configured_extraction_provider() -> ExtractionProvider | None:
    endpoint = os.environ.get("TERMYTEDB_EXTRACTION_URL")
    model = os.environ.get("TERMYTEDB_EXTRACTION_MODEL")
    api_key = os.environ.get("TERMYTEDB_EXTRACTION_API_KEY")
    base_url = os.environ.get("TERMYTEDB_EXTRACTION_BASE_URL")
    provider_name = os.environ.get("TERMYTEDB_EXTRACTION_PROVIDER")

    if provider_name == "http" or endpoint:
        return HttpExtractionProvider(endpoint, model, api_key)
    if provider_name == "openrouter" or api_key or os.environ.get("OPENROUTER_API_KEY"):
        return OpenRouterExtractionProvider(model, api_key=api_key, base_url=base_url)
    return None


def default_extraction_provider() -> ExtractionProvider:
    provider = configured_extraction_provider()
    if provider is not None:
        return provider
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("TERMYTEDB_ALLOW_FAKE_EXTRACTION") == "1":
        return FakeExtractionProvider()
    raise ValueError("no extraction provider configured; set TERMYTEDB_EXTRACTION_URL or OPENROUTER_API_KEY, or pass extraction_provider explicitly")


@dataclass(frozen=True)
class ProviderResult:
    response: ExtractionResponse
    provider_name: str
    model_name: str
    prompt_version: str
    raw_response_hash: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    stage: str | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    response: ReconciliationResponse
    provider_name: str
    model_name: str
    prompt_version: str
    raw_response_hash: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    stage: str = "reconciliation"


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, error_class: str, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.error_class = error_class
        self.retry_after = retry_after


class ExtractionProvider(Protocol):
    name: str
    model: str

    def extract(
        self,
        request: ExtractionRequest,
        timeout_seconds: float = 30.0,
        cancellation: Callable[[], bool] | None = None,
    ) -> ProviderResult: ...

    def reconcile(
        self,
        request: ReconciliationRequest,
        timeout_seconds: float = 30.0,
        cancellation: Callable[[], bool] | None = None,
    ) -> ReconciliationResult: ...


class SessionSummaryProvider(Protocol):
    name: str

    def summarize(self, text: str, *, namespace_id: str, episode_id: str) -> str: ...


class FakeExtractionProvider:
    """Deterministic, offline extraction provider."""

    name = "fake"
    model = "fake-v1"

    def __init__(self, response: ExtractionResponse | None = None, reconciliation_response: ReconciliationResponse | None = None, v3_response: ExtractionResponseV3 | None = None):
        self.response = response
        self.v3_response = v3_response
        self.reconciliation_response = reconciliation_response
        # For multi-pass testing: allow per-stage canned responses
        self.stage_responses: dict[str, ExtractionResponse] = {}

    def set_stage_response(self, stage: str, response: ExtractionResponse) -> None:
        self.stage_responses[stage] = response

    def extract(
        self,
        request: ExtractionRequest,
        timeout_seconds: float = 30.0,
        cancellation: Callable[[], bool] | None = None,
    ) -> ProviderResult:
        del timeout_seconds
        if cancellation and cancellation():
            raise ProviderError("extraction cancelled", retryable=True, error_class="cancelled")
        started = time.perf_counter()
        stage = getattr(request, "stage", "facts") or "facts"
        is_v3 = getattr(request, "extraction_schema", "v2") == "v3"
        response = self.response
        # v3 injected response handling
        if is_v3 and self.v3_response is not None and response is None:
            # Convert v3 response via helper to ExtractionResponse with v3 metadata
            response = _v3_response_to_extraction(self.v3_response.model_dump(mode="json"), request)
            prompt_version = "fake-v3"
            raw = json.dumps(self.v3_response.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            return ProviderResult(
                response=response,
                provider_name=self.name,
                model_name=self.model,
                prompt_version=prompt_version,
                raw_response_hash=hashlib.sha256(raw.encode()).hexdigest(),
                input_tokens=len(json.dumps(request.model_dump(mode="json"), sort_keys=True).split()),
                output_tokens=len(raw.split()),
                latency_ms=int((time.perf_counter() - started) * 1000),
                stage=stage,
            )
        if response is None and stage in self.stage_responses:
            response = self.stage_responses[stage]
        if response is None:
            from .extraction import rule_candidate_to_contract
            from .extractor import extract as rule_extract

            candidates = []
            # Build reverse label map for v3 source_events
            reverse_labels = {str(v): k for k, v in (request.event_labels or {}).items()}
            for event_id, text in request.evidence_text.items():
                # For v3, only process extractable events; skip context-only if schema v3
                if is_v3 and request.extractable_event_ids and event_id not in request.extractable_event_ids:
                    continue
                for item in rule_extract({"text": text}):
                    c = rule_candidate_to_contract(item, event_id, text)
                    # Tag with source stage for tracing
                    try:
                        c = c.model_copy(update={"source_stage": stage})  # type: ignore[arg-type]
                    except Exception:
                        pass
                    if is_v3:
                        # Enrich with v3 metadata preserving labels
                        label = reverse_labels.get(str(event_id), "")
                        v3_type = "preference" if "prefer" in c.statement.casefold() else ("decision" if c.kind == "decision" else "fact")
                        v3_lifecycle = "current" if "prefer" in c.statement.casefold() or "currently" in c.statement.casefold() else "stable"
                        state_key = "user.preference.general" if v3_type == "preference" and v3_lifecycle == "current" else None
                        try:
                            c = c.model_copy(update={
                                "v3_type": v3_type,  # type: ignore[arg-type]
                                "v3_lifecycle": v3_lifecycle,  # type: ignore[arg-type]
                                "v3_state_key": state_key,
                                "v3_source_labels": [label] if label else [],
                                "v3_importance_int": 5 if v3_type == "preference" else 4,
                                "importance": _v3_importance_to_float(5 if v3_type == "preference" else 4),
                                "durability": _v3_lifecycle_to_durability(v3_lifecycle),  # type: ignore[arg-type]
                            })
                        except Exception:
                            pass
                    candidates.append(c)
            if request.existing_memories:
                candidates = self._reconcile_candidates(candidates, request.existing_memories)
            # Prompt version encodes stage for tracing
            if is_v3:
                prompt_version = "fake-v3"
            else:
                prompt_version = f"fake-v1-{stage}" if stage != "facts" else "fake-v1"
            response = ExtractionResponse(schema_version="extraction-v1", prompt_version=prompt_version, candidates=candidates)
        else:
            # Ensure candidates carry source_stage if not set
            updated = []
            for c in response.candidates:
                if getattr(c, "source_stage", None) is None:
                    try:
                        updated.append(c.model_copy(update={"source_stage": stage}))  # type: ignore[arg-type]
                    except Exception:
                        updated.append(c)
                else:
                    updated.append(c)
            if updated != list(response.candidates):
                response = response.model_copy(update={"candidates": updated})
        raw = json.dumps(response.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return ProviderResult(
            response=response,
            provider_name=self.name,
            model_name=self.model,
            prompt_version=response.prompt_version,
            raw_response_hash=hashlib.sha256(raw.encode()).hexdigest(),
            input_tokens=len(json.dumps(request.model_dump(mode="json"), sort_keys=True).split()),
            output_tokens=len(raw.split()),
            latency_ms=int((time.perf_counter() - started) * 1000),
            stage=stage,
        )

    def reconcile(
        self,
        request: ReconciliationRequest,
        timeout_seconds: float = 30.0,
        cancellation: Callable[[], bool] | None = None,
    ) -> ReconciliationResult:
        del timeout_seconds
        if cancellation and cancellation():
            raise ProviderError("reconciliation cancelled", retryable=True, error_class="cancelled")
        started = time.perf_counter()
        response = self.reconciliation_response
        if response is None:
            # Deterministic fallback: map candidates that match existing ref via subject/kind to supersede
            from ..models import ReconciliationDecision, ReconciliationResponse

            decisions: list[ReconciliationDecision] = []
            for idx, cand in enumerate(request.new_candidates):
                matched_ref = None
                for mem in request.existing_memories:
                    if str(mem.get("kind", "")).casefold() == str(cand.kind).casefold() and str(mem.get("subject_key", "")).casefold() == str(cand.subject).casefold():
                        matched_ref = str(mem.get("ref", ""))
                        break
                if matched_ref:
                    decisions.append(
                        ReconciliationDecision(
                            candidate_index=idx,
                            action="supersede" if cand.statement != str(next((m.get("statement") for m in request.existing_memories if m.get("ref") == matched_ref), "")) else "reinforce",
                            existing_memory_ref=matched_ref,
                            confidence=0.99,
                            reason="Deterministic fake reconciliation.",
                        )
                    )
                else:
                    decisions.append(
                        ReconciliationDecision(candidate_index=idx, action="insert", existing_memory_ref=None, confidence=0.99, reason="No matching existing memory.")
                    )
            response = ReconciliationResponse(schema_version="reconciliation-v1", prompt_version="fake-reconciliation-v1", decisions=decisions)
        raw = json.dumps(response.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return ReconciliationResult(
            response=response,
            provider_name=self.name,
            model_name=self.model,
            prompt_version=response.prompt_version,
            raw_response_hash=hashlib.sha256(raw.encode()).hexdigest(),
            input_tokens=len(json.dumps(request.model_dump(mode="json"), sort_keys=True).split()),
            output_tokens=len(raw.split()),
            latency_ms=int((time.perf_counter() - started) * 1000),
            stage="reconciliation",
        )

    @staticmethod
    def _reconcile_candidates(candidates: list[Any], existing_memories: list[dict[str, Any]]) -> list[Any]:
        reconciled = []
        for candidate in candidates:
            best = None
            for existing in existing_memories:
                if str(existing.get("kind", "")).casefold() != str(candidate.kind).casefold():
                    continue
                if str(existing.get("subject_key", "")).casefold() != str(candidate.subject).casefold():
                    continue
                best = existing
                break
            if best is not None and (
                candidate.statement.casefold() != str(best.get("statement", "")).casefold() or FakeExtractionProvider._looks_like_update(candidate.statement)
            ):
                ref = str(best.get("ref") or "")
                if ref:
                    reconciled.append(
                        candidate.model_copy(
                            update={
                                "existing_memory_ref": ref,
                                "intent": "supersede" if candidate.statement.casefold() != str(best.get("statement", "")).casefold() else "reinforce",
                            }
                        )
                    )
                    continue
            reconciled.append(candidate)
        return reconciled

    @staticmethod
    def _looks_like_update(statement: str) -> bool:
        text = statement.casefold()
        return any(marker in text for marker in (" now ", " used to ", " moved ", " changed ", " updated ", " prefer ", " no longer "))


class FakeSessionSummaryProvider:
    """Deterministic session summary provider for tests and offline recovery."""

    name = "fake-summary"

    def summarize(self, text: str, *, namespace_id: str, episode_id: str) -> str:
        del namespace_id, episode_id
        cleaned = " ".join(str(text or "").split())
        return cleaned[:240]


class HttpExtractionProvider:
    """Generic HTTP JSON provider - DEPRECATED, prefer OpenRouterExtractionProvider.

    Kept for backward compat with TERMYTEDB_EXTRACTION_URL. New code should use
    OpenRouterExtractionProvider which handles strict JSON schema and retry
    semantics. This class will be removed in next major version (see config/providers.py).
    """

    name = "http"

    def __init__(self, endpoint: str | None = None, model: str | None = None, api_key: str | None = None):
        self.endpoint = endpoint or os.environ.get("TERMYTEDB_EXTRACTION_URL", "")
        self.model = model or os.environ.get("TERMYTEDB_EXTRACTION_MODEL", "configured")
        self.api_key = api_key or os.environ.get("TERMYTEDB_EXTRACTION_API_KEY")
        if not self.endpoint:
            raise ValueError("an extraction endpoint is required")

    def extract(
        self,
        request: ExtractionRequest,
        timeout_seconds: float = 30.0,
        cancellation: Callable[[], bool] | None = None,
    ) -> ProviderResult:
        if cancellation and cancellation():
            raise ProviderError("extraction cancelled", retryable=True, error_class="cancelled")
        prompt = build_extraction_prompt(request)
        body = json.dumps({"model": self.model, "prompt": prompt, "schema": "extraction-v1", "stage": getattr(request, "stage", "facts")}).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        started = time.perf_counter()
        try:
            with urlopen(Request(self.endpoint, data=body, headers=headers, method="POST"), timeout=timeout_seconds) as response:
                raw_bytes = response.read()
        except HTTPError as exc:
            raise ProviderError(f"provider returned HTTP {exc.code}", retryable=exc.code >= 500, error_class="http_error") from exc
        except (TimeoutError, URLError) as exc:
            raise ProviderError("provider request failed", retryable=True, error_class="transport_error") from exc
        if cancellation and cancellation():
            raise ProviderError("extraction cancelled", retryable=True, error_class="cancelled")
        raw = raw_bytes.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict) and isinstance(payload.get("output"), str):
                payload = json.loads(payload["output"])
            parsed = _simple_response_to_extraction(payload, request)
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProviderError("provider returned invalid extraction-v1 JSON", retryable=False, error_class="invalid_output") from exc
        return ProviderResult(
            response=parsed,
            provider_name=self.name,
            model_name=self.model,
            prompt_version=parsed.prompt_version,
            raw_response_hash=hashlib.sha256(raw_bytes).hexdigest(),
            input_tokens=len(prompt.split()),
            output_tokens=len(raw.split()),
            latency_ms=int((time.perf_counter() - started) * 1000),
            stage=getattr(request, "stage", None),
        )

    def reconcile(
        self,
        request: ReconciliationRequest,
        timeout_seconds: float = 30.0,
        cancellation: Callable[[], bool] | None = None,
    ) -> ReconciliationResult:
        if cancellation and cancellation():
            raise ProviderError("reconciliation cancelled", retryable=True, error_class="cancelled")
        prompt = _build_reconciliation_prompt(request)
        body = json.dumps({"model": self.model, "prompt": prompt, "schema": "reconciliation-v1"}).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        started = time.perf_counter()
        try:
            with urlopen(Request(self.endpoint, data=body, headers=headers, method="POST"), timeout=timeout_seconds) as response:
                raw_bytes = response.read()
        except HTTPError as exc:
            raise ProviderError(f"provider returned HTTP {exc.code}", retryable=exc.code >= 500, error_class="http_error") from exc
        except (TimeoutError, URLError) as exc:
            raise ProviderError("provider request failed", retryable=True, error_class="transport_error") from exc
        if cancellation and cancellation():
            raise ProviderError("reconciliation cancelled", retryable=True, error_class="cancelled")
        raw = raw_bytes.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict) and isinstance(payload.get("output"), str):
                payload = json.loads(payload["output"])
            parsed = ReconciliationResponse.model_validate(payload)
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProviderError("provider returned invalid reconciliation-v1 JSON", retryable=False, error_class="invalid_output") from exc
        return ReconciliationResult(
            response=parsed,
            provider_name=self.name,
            model_name=self.model,
            prompt_version=parsed.prompt_version,
            raw_response_hash=hashlib.sha256(raw_bytes).hexdigest(),
            input_tokens=len(prompt.split()),
            output_tokens=len(raw.split()),
            latency_ms=int((time.perf_counter() - started) * 1000),
            stage="reconciliation",
        )


def build_session_summary_prompt(text: str, *, namespace_id: str, episode_id: str) -> list[dict[str, str]]:
    """Backward-compat wrapper - canonical implementation lives in `config.prompts`."""
    return _build_session_summary_prompt(text, namespace_id=namespace_id, episode_id=episode_id)


class OpenRouterExtractionProvider:
    """OpenAI-compatible structured extraction through OpenRouter."""

    name = "openrouter"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self.model = model or os.environ.get("TERMYTEDB_EXTRACTION_MODEL", "")
        if not self.model:
            raise ValueError("TERMYTEDB_EXTRACTION_MODEL is required")
        self.api_key = api_key or os.environ.get("TERMYTEDB_EXTRACTION_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = (
            base_url or os.environ.get("TERMYTEDB_EXTRACTION_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        ).rstrip("/")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

    def extract(
        self,
        request: ExtractionRequest,
        timeout_seconds: float = 30.0,
        cancellation: Callable[[], bool] | None = None,
    ) -> ProviderResult:
        if cancellation and cancellation():
            raise ProviderError("extraction cancelled", retryable=True, error_class="cancelled")
        prompt = build_extraction_prompt(request)
        stage = getattr(request, "stage", "facts") or "facts"
        # Use stage-specific temperature if configured via env
        try:
            temperature = float(os.environ.get("TERMYTEDB_EXTRACTION_TEMPERATURE", "0"))
        except ValueError:
            temperature = 0.0
        is_v3 = getattr(request, "extraction_schema", "v2") == "v3"
        system_content = "Return only valid JSON matching the supplied extraction-v3 schema." if is_v3 else "Return only valid JSON matching the supplied memory-list schema."
        response_fmt = _extraction_response_format_v3() if is_v3 else _extraction_response_format()
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 2500,
            "response_format": response_fmt,
            "plugins": [{"id": "response-healing"}],
        }
        started = time.perf_counter()
        max_retries = _get_retry_budget()
        payload: dict[str, Any] | None = None
        raw_bytes: bytes | None = None
        parsed: ExtractionResponse | None = None
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            elapsed = time.perf_counter() - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                raise ProviderError("extraction timeout", retryable=True, error_class="timeout")
            if cancellation and cancellation():
                raise ProviderError("extraction cancelled", retryable=True, error_class="cancelled")
            if attempt > 0:
                # Pacing between retries / stages — cancellation-aware
                _cancellable_sleep(0.1, cancellation, started, timeout_seconds)
                # recompute remaining after pacing
                elapsed = time.perf_counter() - started
                remaining = timeout_seconds - elapsed
                if remaining <= 0:
                    raise last_exc if last_exc else ProviderError("extraction timeout", retryable=True, error_class="timeout")  # type: ignore[misc]
            # Use actual remaining, but ensure a minimal viable timeout for the HTTP call
            call_timeout = max(0.5, remaining) if remaining > 0.5 else remaining
            # If remaining is tiny (<0.2) there is no point retrying — fail fast
            if call_timeout < 0.2:
                raise last_exc if last_exc else ProviderError("extraction timeout", retryable=True, error_class="timeout")  # type: ignore[misc]
            try:
                payload, raw_bytes = _openrouter_chat(self.base_url, self.api_key, body, title=f"TermyteDB Memory Extraction {stage}", timeout=call_timeout)
                choice = _message_text(payload, text_parts_only=True)
                if not choice.strip():
                    raise ValueError("empty extraction content")
                parsed = _simple_response_to_extraction(json.loads(clean_json_response(choice)), request)
                break
            except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                # A completed request with unusable model output is not an
                # ingestion failure. Retry it once, then preserve the events
                # and record an empty extraction, just as Mem0 does.
                last_exc = ProviderError("OpenRouter returned unusable extraction content", retryable=True, error_class="invalid_output")
                if attempt >= max_retries:
                    parsed = _empty_extraction_response()
                    break
                sleep_s = _retry_sleep(attempt, None)
                if sleep_s > timeout_seconds - (time.perf_counter() - started):
                    parsed = _empty_extraction_response()
                    break
                _cancellable_sleep(sleep_s, cancellation, started, timeout_seconds)
                continue
            except HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    detail = ""
                retry_after_raw = exc.headers.get("Retry-After") if hasattr(exc, "headers") and exc.headers else None
                retry_after = _parse_retry_after(retry_after_raw)
                suffix = f" response={detail}" if detail else ""
                if retry_after_raw:
                    suffix += f" retry_after={retry_after_raw}"
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                last_exc = ProviderError(f"OpenRouter returned HTTP {exc.code}{suffix}", retryable=retryable, error_class="http_error", retry_after=retry_after)
                if not retryable or attempt >= max_retries:
                    raise last_exc from exc
                sleep_s = _retry_sleep(attempt, retry_after)
                elapsed2 = time.perf_counter() - started
                remaining2 = timeout_seconds - elapsed2
                if sleep_s > remaining2:
                    raise last_exc from exc
                _cancellable_sleep(sleep_s, cancellation, started, timeout_seconds)
                continue
            except (TimeoutError, URLError, OSError) as exc:
                last_exc = ProviderError("OpenRouter request failed", retryable=True, error_class="transport_error")
                if attempt >= max_retries:
                    raise last_exc from exc
                sleep_s = _retry_sleep(attempt, None)
                elapsed2 = time.perf_counter() - started
                remaining2 = timeout_seconds - elapsed2
                if sleep_s > remaining2:
                    raise last_exc from exc
                _cancellable_sleep(sleep_s, cancellation, started, timeout_seconds)
                continue
            except Exception as exc:
                # IncompleteRead and other transport errors (connection reset etc.)
                msg = str(exc)
                is_transport = (
                    "IncompleteRead" in type(exc).__name__
                    or "IncompleteRead" in msg
                    or "Connection reset" in msg
                    or "transport" in msg.lower()
                    or "connection" in msg.lower()
                )
                if not is_transport:
                    raise
                last_exc = ProviderError("OpenRouter request failed", retryable=True, error_class="transport_error")
                if attempt >= max_retries:
                    raise last_exc from exc
                sleep_s = _retry_sleep(attempt, None)
                elapsed2 = time.perf_counter() - started
                remaining2 = timeout_seconds - elapsed2
                if sleep_s > remaining2:
                    raise last_exc from exc
                _cancellable_sleep(sleep_s, cancellation, started, timeout_seconds)
                continue
        if payload is None or raw_bytes is None:
            assert last_exc is not None
            raise last_exc  # type: ignore[misc]
        if cancellation and cancellation():
            raise ProviderError("extraction cancelled", retryable=True, error_class="cancelled")
        if parsed is None:
            parsed = _empty_extraction_response()
        actual_model = str(payload.get("model", self.model))
        usage = payload.get("usage") or {}
        return ProviderResult(
            response=parsed,
            provider_name=self.name,
            model_name=actual_model,
            prompt_version=parsed.prompt_version,
            raw_response_hash=hashlib.sha256(raw_bytes).hexdigest(),
            input_tokens=usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
            output_tokens=usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            stage=stage,
        )

    def reconcile(
        self,
        request: ReconciliationRequest,
        timeout_seconds: float = 30.0,
        cancellation: Callable[[], bool] | None = None,
    ) -> ReconciliationResult:
        if cancellation and cancellation():
            raise ProviderError("reconciliation cancelled", retryable=True, error_class="cancelled")
        prompt = _build_reconciliation_prompt(request)
        try:
            temperature = float(os.environ.get("TERMYTEDB_RECONCILIATION_TEMPERATURE", os.environ.get("TERMYTEDB_EXTRACTION_TEMPERATURE", "0")))
        except ValueError:
            temperature = 0.0
        reconciliation_model = os.environ.get("TERMYTEDB_RECONCILIATION_MODEL") or self.model
        body = {
            "model": reconciliation_model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON matching the supplied reconciliation-v1 schema."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 2500,
            "response_format": _reconciliation_response_format(),
            "plugins": [{"id": "response-healing"}],
        }
        started = time.perf_counter()
        max_retries = _get_retry_budget()
        payload: dict[str, Any] | None = None
        raw_bytes: bytes | None = None
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            elapsed = time.perf_counter() - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                raise ProviderError("reconciliation timeout", retryable=True, error_class="timeout")
            if cancellation and cancellation():
                raise ProviderError("reconciliation cancelled", retryable=True, error_class="cancelled")
            if attempt > 0:
                _cancellable_sleep(0.1, cancellation, started, timeout_seconds)
                elapsed = time.perf_counter() - started
                remaining = timeout_seconds - elapsed
                if remaining <= 0:
                    raise last_exc if last_exc else ProviderError("reconciliation timeout", retryable=True, error_class="timeout")  # type: ignore[misc]
            call_timeout = max(0.5, remaining) if remaining > 0.5 else remaining
            if call_timeout < 0.2:
                raise last_exc if last_exc else ProviderError("reconciliation timeout", retryable=True, error_class="timeout")  # type: ignore[misc]
            try:
                payload, raw_bytes = _openrouter_chat(self.base_url, self.api_key, body, title="TermyteDB Reconciliation", timeout=call_timeout)
                break
            except HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace").strip()
                except Exception:
                    detail = ""
                retry_after_raw = exc.headers.get("Retry-After") if hasattr(exc, "headers") and exc.headers else None
                retry_after = _parse_retry_after(retry_after_raw)
                suffix = f" response={detail}" if detail else ""
                if retry_after_raw:
                    suffix += f" retry_after={retry_after_raw}"
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                last_exc = ProviderError(f"OpenRouter returned HTTP {exc.code}{suffix}", retryable=retryable, error_class="http_error", retry_after=retry_after)
                if not retryable or attempt >= max_retries:
                    raise last_exc from exc
                sleep_s = _retry_sleep(attempt, retry_after)
                elapsed2 = time.perf_counter() - started
                remaining2 = timeout_seconds - elapsed2
                if sleep_s > remaining2:
                    raise last_exc from exc
                _cancellable_sleep(sleep_s, cancellation, started, timeout_seconds)
                continue
            except (TimeoutError, URLError, OSError) as exc:
                last_exc = ProviderError("OpenRouter request failed", retryable=True, error_class="transport_error")
                if attempt >= max_retries:
                    raise last_exc from exc
                sleep_s = _retry_sleep(attempt, None)
                elapsed2 = time.perf_counter() - started
                remaining2 = timeout_seconds - elapsed2
                if sleep_s > remaining2:
                    raise last_exc from exc
                _cancellable_sleep(sleep_s, cancellation, started, timeout_seconds)
                continue
            except Exception as exc:
                msg = str(exc)
                is_transport = (
                    "IncompleteRead" in type(exc).__name__
                    or "IncompleteRead" in msg
                    or "Connection reset" in msg
                    or "transport" in msg.lower()
                )
                if not is_transport:
                    raise
                last_exc = ProviderError("OpenRouter request failed", retryable=True, error_class="transport_error")
                if attempt >= max_retries:
                    raise last_exc from exc
                sleep_s = _retry_sleep(attempt, None)
                elapsed2 = time.perf_counter() - started
                remaining2 = timeout_seconds - elapsed2
                if sleep_s > remaining2:
                    raise last_exc from exc
                _cancellable_sleep(sleep_s, cancellation, started, timeout_seconds)
                continue
        if payload is None or raw_bytes is None:
            assert last_exc is not None
            raise last_exc  # type: ignore[misc]
        if cancellation and cancellation():
            raise ProviderError("reconciliation cancelled", retryable=True, error_class="cancelled")
        try:
            choice = _message_text(payload, text_parts_only=True)
            if not choice.strip():
                raise ProviderError("OpenRouter returned empty reconciliation content", retryable=True, error_class="empty_output")
            parsed = ReconciliationResponse.model_validate(json.loads(clean_json_response(choice)))
        except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("OpenRouter returned invalid reconciliation-v1 JSON", retryable=False, error_class="invalid_output") from exc
        actual_model = str(payload.get("model", reconciliation_model))
        usage = payload.get("usage") or {}
        return ReconciliationResult(
            response=parsed,
            provider_name=self.name,
            model_name=actual_model,
            prompt_version=parsed.prompt_version,
            raw_response_hash=hashlib.sha256(raw_bytes).hexdigest(),
            input_tokens=usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
            output_tokens=usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            stage="reconciliation",
        )


class OpenRouterSessionSummaryProvider:
    """OpenRouter chat completions provider for session summaries."""

    name = "openrouter-summary"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self.model = model or os.environ.get("TERMYTEDB_SUMMARY_MODEL") or os.environ.get("TERMYTEDB_EXTRACTION_MODEL") or ""
        if not self.model:
            raise ValueError("TERMYTEDB_SUMMARY_MODEL or TERMYTEDB_EXTRACTION_MODEL is required")
        self.api_key = (
            api_key or os.environ.get("TERMYTEDB_SUMMARY_API_KEY") or os.environ.get("TERMYTEDB_EXTRACTION_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        )
        self.base_url = (
            base_url
            or os.environ.get("TERMYTEDB_SUMMARY_BASE_URL")
            or os.environ.get("TERMYTEDB_EXTRACTION_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        ).rstrip("/")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

    def summarize(self, text: str, *, namespace_id: str, episode_id: str) -> str:
        prompt = build_session_summary_prompt(text, namespace_id=namespace_id, episode_id=episode_id)
        body = {
            "model": self.model,
            "messages": prompt,
            "temperature": 0,
            "max_tokens": 220,
        }
        payload, _ = _openrouter_chat(self.base_url, self.api_key, body, title="TermyteDB Session Summary", timeout=45.0)
        summary = " ".join(_message_text(payload).split())
        return summary[:400]
