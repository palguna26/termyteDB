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
    build_reconciliation_prompt as _build_reconciliation_prompt,
    build_session_summary_prompt as _build_session_summary_prompt,
    clean_json_response as _clean_json_response,
    extraction_response_format as _extraction_response_format,
    reconciliation_response_format as _reconciliation_response_format,
)

# Re-export for backward compatibility: existing imports `from memory.provider
# import build_extraction_prompt` continue to work while config is source of truth.
from ..models import (
    ExtractionRequest,
    ExtractionResponse,
    ReconciliationRequest,
    ReconciliationResponse,
)


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


def _message_text(payload: dict[str, Any], *, text_parts_only: bool = False) -> str:
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            str(part.get("text") or "") for part in content if isinstance(part, dict) and (not text_parts_only or part.get("type", "text") == "text")
        )
    if isinstance(content, dict):
        return json.dumps(content)
    return str(content or "")


def clean_json_response(value: str) -> str:
    """Backward-compat wrapper - canonical implementation lives in `config.prompts`."""
    return _clean_json_response(value)


def extraction_response_format() -> dict[str, object]:
    """Backward-compat wrapper - canonical implementation lives in `config.prompts`."""
    return _extraction_response_format()


def build_extraction_prompt(request: ExtractionRequest) -> str:
    """Backward-compat wrapper - canonical implementation lives in `config.prompts`."""
    return _build_extraction_prompt(request)


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
    def __init__(self, message: str, *, retryable: bool, error_class: str):
        super().__init__(message)
        self.retryable = retryable
        self.error_class = error_class


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

    def __init__(self, response: ExtractionResponse | None = None, reconciliation_response: ReconciliationResponse | None = None):
        self.response = response
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
        response = self.response
        stage = getattr(request, "stage", "facts") or "facts"
        if response is None and stage in self.stage_responses:
            response = self.stage_responses[stage]
        if response is None:
            from .extraction import rule_candidate_to_contract
            from .extractor import extract as rule_extract

            candidates = []
            for event_id, text in request.evidence_text.items():
                for item in rule_extract({"text": text}):
                    c = rule_candidate_to_contract(item, event_id, text)
                    # Tag with source stage for tracing
                    try:
                        c = c.model_copy(update={"source_stage": stage})  # type: ignore[arg-type]
                    except Exception:
                        pass
                    candidates.append(c)
            if request.existing_memories:
                candidates = self._reconcile_candidates(candidates, request.existing_memories)
            # Prompt version encodes stage for tracing
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
            parsed = ExtractionResponse.model_validate(payload)
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
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON matching the supplied extraction-v1 schema."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 2500,
            "response_format": extraction_response_format(),
            "plugins": [{"id": "response-healing"}],
        }
        started = time.perf_counter()
        try:
            payload, raw_bytes = _openrouter_chat(self.base_url, self.api_key, body, title=f"TermyteDB Memory Extraction {stage}", timeout=timeout_seconds)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            retry_after = exc.headers.get("Retry-After")
            suffix = f" response={detail}" if detail else ""
            if retry_after:
                suffix += f" retry_after={retry_after}"
            raise ProviderError(f"OpenRouter returned HTTP {exc.code}{suffix}", retryable=exc.code in {408, 429, 500, 502, 503, 504}, error_class="http_error") from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise ProviderError("OpenRouter request failed", retryable=True, error_class="transport_error") from exc
        if cancellation and cancellation():
            raise ProviderError("extraction cancelled", retryable=True, error_class="cancelled")
        try:
            choice = _message_text(payload, text_parts_only=True)
            if not choice.strip():
                raise ProviderError("OpenRouter returned empty extraction content", retryable=True, error_class="empty_output")
            parsed = ExtractionResponse.model_validate(json.loads(clean_json_response(choice)))
        except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("OpenRouter returned invalid extraction-v1 JSON", retryable=False, error_class="invalid_output") from exc
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
        try:
            payload, raw_bytes = _openrouter_chat(self.base_url, self.api_key, body, title="TermyteDB Reconciliation", timeout=timeout_seconds)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            retry_after = exc.headers.get("Retry-After")
            suffix = f" response={detail}" if detail else ""
            if retry_after:
                suffix += f" retry_after={retry_after}"
            raise ProviderError(f"OpenRouter returned HTTP {exc.code}{suffix}", retryable=exc.code in {408, 429, 500, 502, 503, 504}, error_class="http_error") from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise ProviderError("OpenRouter request failed", retryable=True, error_class="transport_error") from exc
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
