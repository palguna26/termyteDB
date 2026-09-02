from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from difflib import SequenceMatcher
from typing import Any
from uuid import UUID

from ..config.settings import MEMORY
from ..core.logging import log
from ..core.redaction import redact_text
from ..models import ExtractionRequest, ReconciliationRequest
from ..storage.repository import Repository
from .extraction import CandidateRejected, validate_candidate
from .extractor import payload_text
from .provider import ExtractionProvider, ProviderError, SessionSummaryProvider, default_extraction_provider


def _get_extraction_stages() -> list[str]:
    # The single prompt covers all memory types. This remains a list only for
    # compatible tracing metadata in the existing processing pipeline.
    return ["facts"]


def _is_reconciliation_enabled() -> bool:
    # Simple extraction is deliberately one LLM call. Reconciliation remains
    # available as an explicit opt-in for applications that need it.
    raw = os.environ.get("TERMYTEDB_RECONCILIATION_ENABLED", "0")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _get_extraction_schema() -> str:
    raw = os.environ.get("TERMYTEDB_EXTRACTION_SCHEMA", "v2").strip().lower()
    if raw in {"v3", "extraction-v3", "extraction_v3", "3"}:
        return "v3"
    return "v2"


def _get_max_calls() -> int:
    raw = os.environ.get("TERMYTEDB_MAX_LLM_CALLS_PER_BATCH")
    if raw is None:
        return 10


def _max_candidates_per_event() -> int:
    """Bound noisy one-call extraction without dropping distinct event facts."""
    raw = os.environ.get("TERMYTEDB_MAX_CANDIDATES_PER_EVENT")
    try:
        return max(1, int(raw)) if raw is not None else MEMORY.max_candidates_per_event
    except ValueError:
        return MEMORY.max_candidates_per_event


def _prune_event_candidates(candidates: list[Any]) -> list[Any]:
    """Keep the first high-signal, non-duplicate memories from each event.

    The model is instructed to order useful facts first.  This safety cap
    prevents a chatty turn from flooding retrieval with paraphrases.
    """
    limit = _max_candidates_per_event()
    kept: list[Any] = []
    counts: dict[str, int] = {}
    prior_statements: dict[str, list[tuple[set[str], str]]] = {}
    for candidate in candidates:
        evidence = list(getattr(candidate, "evidence", []) or [])
        event_id = str(evidence[0].event_id) if evidence else ""
        if not event_id:
            kept.append(candidate)
            continue
        terms = {term.casefold() for term in str(candidate.statement).split() if len(term) > 2}
        normalized = " ".join(str(candidate.statement).casefold().split())
        near_duplicate = any(
            (terms and prior_terms and len(terms & prior_terms) / len(terms | prior_terms) >= 0.85)
            or SequenceMatcher(None, normalized, prior_text).ratio() >= 0.90
            for prior_terms, prior_text in prior_statements.get(event_id, [])
        )
        if near_duplicate or counts.get(event_id, 0) >= limit:
            continue
        kept.append(candidate)
        counts[event_id] = counts.get(event_id, 0) + 1
        prior_statements.setdefault(event_id, []).append((terms, normalized))
    return kept


def _normalize_state_key(value: str | None) -> str | None:
    if not value:
        return None
    normalized = " ".join(str(value).strip().split()).casefold()
    if "." not in normalized:
        return None
    return normalized


def _deduplicate_v3_within_response(candidates: list[Any]) -> list[Any]:
    """Deduplicate within one extraction response; keep most specific with combined sources."""
    seen: dict[str, Any] = {}
    for cand in candidates:
        key = " ".join(str(cand.statement).casefold().split())
        labels = list(getattr(cand, "v3_source_labels", []) or [])
        if key not in seen:
            seen[key] = cand
        else:
            existing = seen[key]
            # Keep more specific: longer statement or more source events
            existing_labels = list(getattr(existing, "v3_source_labels", []) or [])
            if len(str(cand.statement)) > len(str(existing.statement)) or len(labels) > len(existing_labels):
                # Merge source labels
                merged_labels = list(dict.fromkeys(existing_labels + labels))
                try:
                    seen[key] = cand.model_copy(update={"v3_source_labels": merged_labels})
                except Exception:
                    seen[key] = cand
            else:
                # Merge missing labels into existing
                merged_labels = list(dict.fromkeys(existing_labels + labels))
                if len(merged_labels) != len(existing_labels):
                    try:
                        seen[key] = existing.model_copy(update={"v3_source_labels": merged_labels})
                    except Exception:
                        pass
    return list(seen.values())


def _enforce_session_quality_budget(
    candidates: list[Any],
    event_session_map: dict[str, str],
    included: dict[UUID, str],
) -> list[Any]:
    """Importance-based session budget: 3-8 high-value records per session."""
    archive_mode = os.environ.get("TERMYTEDB_ARCHIVE_MODE", "0").strip().lower() not in {"0", "false", "no", "off"}
    # Group candidates by primary session (first evidence event's session)
    grouped: dict[str, list[Any]] = {}
    for cand in candidates:
        evs = list(getattr(cand, "evidence", []) or [])
        # For v3, use first evidence event; for v2, single
        first_eid = str(evs[0].event_id) if evs else ""
        session = event_session_map.get(first_eid, "") if first_eid else ""
        # Fallback: use any evidence session
        if not session and evs:
            for ev in evs:
                s = event_session_map.get(str(ev.event_id), "")
                if s:
                    session = s
                    break
        session = session or "unknown"
        grouped.setdefault(session, []).append(cand)
    result: list[Any] = []
    for session, items in grouped.items():
        # Sort by importance (v3 int or float) descending, then statement length descending
        def importance_key(c):
            v = getattr(c, "v3_importance_int", None)
            if isinstance(v, int):
                return v
            # fallback to float importance scaled
            try:
                return int(float(getattr(c, "importance", 0.5)) * 5)
            except Exception:
                return 0
        items_sorted = sorted(items, key=lambda c: (-importance_key(c), -len(str(c.statement))))
        # Filter low-value 1-2 unless archive mode
        if not archive_mode:
            filtered = [c for c in items_sorted if importance_key(c) >= 3]
            # Keep at least 3 if filtered would be too few but original had more
            if len(filtered) < 3 and len(items_sorted) >= 3:
                filtered = items_sorted[:3]
            # But if all are low importance, keep filtered as is (could be empty -> omit chit-chat)
            items_sorted = filtered
        # Cap at 8 per session, allow more only when distinct and high-value
        # For now hard cap 8, but if session has many distinct high-value (importance 5), allow up to 12
        cap = 8
        high_count = sum(1 for c in items_sorted if importance_key(c) == 5)
        if high_count > 8:
            cap = min(12, len(items_sorted))
        result.extend(items_sorted[:cap])
    return result


def _resolve_v3_candidates(
    candidates: list[Any],
    request: Any,
    included: dict[UUID, str],
    repo: Any,
    namespace_id: str,
) -> tuple[list[Any], list[tuple[Any, str]]]:
    """Resolve v3 source_events labels to event IDs, attach chunks, build evidence."""
    event_labels: dict[str, UUID] = getattr(request, "event_labels", {}) or {}
    # reverse not needed; event_labels is label->UUID
    extractable_ids = set(getattr(request, "extractable_event_ids", []) or [])
    # fallback if extractable not set
    if not extractable_ids:
        extractable_ids = set(included.keys())
    # Build event->chunks map (all chunks for those events)
    resolved: list[Any] = []
    rejected: list[tuple[Any, str]] = []
    # Pre-fetch chunks for included events
    chunk_cache: dict[str, list[str]] = {}
    for eid in included:
        try:
            chunks = repo.chunks_for_events(namespace_id, [str(eid)], limit=10)
            chunk_cache[str(eid)] = [str(c["chunk_id"]) for c in chunks]
        except Exception:
            chunk_cache[str(eid)] = []
    for cand in candidates:
        labels = list(getattr(cand, "v3_source_labels", []) or [])
        if not labels:
            rejected.append((cand, "missing_source_events"))
            continue
        # Deduplicate labels preserving order
        dedup_labels = list(dict.fromkeys(labels))
        event_ids: list[UUID] = []
        has_extractable = False
        has_unknown = False
        for lab in dedup_labels:
            eid = event_labels.get(str(lab))
            if eid is None:
                has_unknown = True
                continue
            event_ids.append(eid)
            if eid in extractable_ids:
                has_extractable = True
        # Do not silently repair a mixed list such as ["e1", "invented"].
        # A successful-looking partial repair hides malformed model output and
        # makes the provenance trace claim stronger grounding than we have.
        if has_unknown:
            rejected.append((cand, "unknown_source_label"))
            continue
        if not event_ids:
            rejected.append((cand, "unknown_source_label" if has_unknown else "context_only_source"))
            continue
        if not has_extractable:
            rejected.append((cand, "context_only_source"))
            continue
        # Deduplicate event_ids
        event_ids = list(dict.fromkeys(event_ids))
        # Build evidence spans from raw source text (bounded)
        evidence_spans = []
        chunk_ids: list[str] = []
        observed_times: list[str] = []
        source_roles: set[str] = set()
        timestamps = getattr(request, "event_timestamps", {}) or {}
        roles = getattr(request, "event_roles", {}) or {}
        for eid in event_ids:
            source = included.get(eid, "")
            if not source:
                continue
            excerpt = source[:2000]
            from ..models import EvidenceSpan as _ES
            evidence_spans.append(_ES(event_id=eid, start_offset=0, end_offset=len(excerpt), excerpt=excerpt))
            chunk_ids.extend(chunk_cache.get(str(eid), []))
            ts = timestamps.get(eid)
            if ts:
                observed_times.append(str(ts))
            role = roles.get(eid)
            if role:
                source_roles.add(str(role))
        if not evidence_spans:
            rejected.append((cand, "missing_source_text"))
            continue
        # Determine primary role
        primary_role = next(iter(source_roles), "user")
        if "user" in source_roles:
            primary_role = "user"
        # Determine observed_at as latest timestamp among cited events
        # Let repository derive observed_at from evidence event time; we just attach evidence
        chunk_ids = list(dict.fromkeys(chunk_ids))
        try:
            updated = cand.model_copy(update={
                "evidence": evidence_spans,
                "source_chunk_ids": chunk_ids,
                "source_role": primary_role,
            })
        except Exception as exc:
            rejected.append((cand, f"evidence_build_failed:{exc}"))
            continue
        resolved.append(updated)
    # Deduplicate within response after resolution
    resolved = _deduplicate_v3_within_response(resolved)
    return resolved, rejected


def _ensure_stage_column(repo: Repository) -> None:
    try:
        cols = {row[1] for row in repo.db.execute("PRAGMA table_info(extraction_runs)").fetchall()}
        if "stage" not in cols:
            repo.db.execute("ALTER TABLE extraction_runs ADD COLUMN stage TEXT")
        if "candidate_count" not in cols:
            repo.db.execute("ALTER TABLE extraction_runs ADD COLUMN candidate_count INTEGER")
        if "input_event_ids" not in cols:
            repo.db.execute("ALTER TABLE extraction_runs ADD COLUMN input_event_ids TEXT")
    except Exception:
        pass


class Processor:
    def __init__(
        self,
        repository: Repository,
        logger: logging.Logger,
        provider: ExtractionProvider | None = None,
        summary_provider: SessionSummaryProvider | None = None,
    ):
        self.repository = repository
        self.logger = logger
        self.provider = provider
        self.summary_provider = summary_provider

    def _collect_multi_pass(
        self,
        provider: Any,
        namespace_id: str,
        events: list[UUID],
        included: dict[UUID, str],
        existing_memories: list[dict[str, Any]],
        deadline: float,
        *,
        event_labels: dict[str, UUID] | None = None,
        chunk_labels: dict[str, str] | None = None,
        event_chunk_labels: dict[UUID, str] | None = None,
        event_roles: dict[UUID, str] | None = None,
        extractable_event_ids: list[UUID] | None = None,
        context_event_ids: list[UUID] | None = None,
        event_timestamps: dict[UUID, str] | None = None,
        event_session_ids: dict[UUID, str] | None = None,
        extraction_schema: str | None = None,
    ) -> tuple[list[Any], list[Any], dict[str, Any]]:
        stages = _get_extraction_stages()
        max_calls = _get_max_calls()
        if len(stages) > max_calls:
            raise ProviderError(f"extraction stages {len(stages)} exceeds max {max_calls}", retryable=False, error_class="max_calls_exceeded")
        all_candidates: list[Any] = []
        provider_results: list[Any] = []
        last_error: Exception | None = None
        retryable_error: ProviderError | None = None
        for stage in stages:
            # Check cancellation deadline
            remaining = max(0.001, deadline - time.perf_counter())
            if time.perf_counter() >= deadline:
                break
            request = ExtractionRequest(
                namespace_id=namespace_id,
                events=events,
                evidence_text=included,
                event_labels=event_labels or {},
                chunk_labels=chunk_labels or {},
                event_chunk_labels=event_chunk_labels or {},
                event_roles=event_roles or {},  # type: ignore[arg-type]
                existing_memories=existing_memories,
                stage=stage,  # type: ignore[arg-type]
                extractable_event_ids=extractable_event_ids or [],
                context_event_ids=context_event_ids or [],
                event_timestamps=event_timestamps or {},
                event_session_ids=event_session_ids or {},
                extraction_schema=extraction_schema or _get_extraction_schema(),  # type: ignore[arg-type]
            )
            try:
                pr = provider.extract(
                    request,
                    timeout_seconds=remaining,
                    cancellation=lambda: time.perf_counter() >= deadline,
                )
                # Tag candidates with source_stage if missing
                for c in pr.response.candidates:
                    if getattr(c, "source_stage", None) is None:
                        try:
                            # annotate
                            all_candidates.append(c.model_copy(update={"source_stage": stage}))  # type: ignore[arg-type]
                        except Exception:
                            all_candidates.append(c)
                    else:
                        all_candidates.append(c)
                provider_results.append(pr)
                # Pacing between stages to avoid burst 429s when multi-stage is enabled
                if len(stages) > 1 and stage != stages[-1]:
                    time.sleep(0.15)
            except ProviderError as exc:
                last_error = exc
                if exc.retryable:
                    retryable_error = exc
                log(self.logger, logging.WARNING, "processing.stage_failed", namespace_id=namespace_id, stage=stage, error=str(exc), retryable=exc.retryable)
                continue
            except Exception as exc:
                last_error = exc
                log(self.logger, logging.WARNING, "processing.stage_failed", namespace_id=namespace_id, stage=stage, error=str(exc))
                continue
        if not provider_results and last_error is not None:
            raise last_error
        # Rate-limit or other retryable failures must not silently produce partial memories.
        # If any stage failed retryably, fail the whole batch so the job remains retryable.
        if retryable_error is not None and len(provider_results) != len(stages):
            log(self.logger, logging.WARNING, "processing.retryable_partial_suppressed", namespace_id=namespace_id, succeeded=len(provider_results), failed=len(stages) - len(provider_results))
            raise retryable_error
        meta = {"stages": stages, "succeeded": len(provider_results), "failed": len(stages) - len(provider_results)}
        return all_candidates, provider_results, meta

    def _apply_reconciliation(
        self,
        provider: Any,
        namespace_id: str,
        existing_memories: list[dict[str, Any]],
        candidates: list[Any],
        deadline: float,
    ) -> list[Any]:
        if not _is_reconciliation_enabled():
            return candidates
        if not candidates or not existing_memories:
            return candidates
        # Only attempt if provider has reconcile method
        if not hasattr(provider, "reconcile"):
            return candidates
        max_calls = _get_max_calls()
        # Ensure reconciliation call doesn't exceed max
        stages = _get_extraction_stages()
        if len(stages) + 1 > max_calls:
            log(self.logger, logging.WARNING, "processing.reconciliation_skipped_max_calls", namespace_id=namespace_id)
            return candidates
        try:
            req = ReconciliationRequest(
                namespace_id=namespace_id,
                existing_memories=existing_memories,
                new_candidates=candidates,
            )
            remaining = max(0.001, deadline - time.perf_counter())
            result = provider.reconcile(
                req,
                timeout_seconds=remaining,
                cancellation=lambda: time.perf_counter() >= deadline,
            )
            # Apply decisions: map candidate_index -> action/ref
            existing_by_ref = {str(m.get("ref")): m for m in existing_memories}
            # Validate and apply
            for dec in result.response.decisions:
                idx = dec.candidate_index
                if idx < 0 or idx >= len(candidates):
                    log(self.logger, logging.WARNING, "processing.reconciliation_invalid_index", namespace_id=namespace_id, index=idx)
                    continue
                # Validate action names — simplified to 5 actions
                action = str(dec.action).casefold()
                allowed = {"insert", "reinforce", "update", "supersede", "ignore"}
                if action in {"dispute", "contradiction"}:
                    action = "ignore"
                if action not in allowed:
                    log(self.logger, logging.WARNING, "processing.reconciliation_invalid_action", namespace_id=namespace_id, action=action)
                    continue
                cand = candidates[idx]
                # Validate ref if present
                ref = dec.existing_memory_ref
                if ref is not None and ref not in existing_by_ref:
                    log(self.logger, logging.WARNING, "processing.reconciliation_unknown_ref", namespace_id=namespace_id, ref=ref)
                    continue
                # Apply to candidate intent/ref; code enforces referential integrity later (unknown refs rejected)
                try:
                    updated = cand.model_copy(update={"intent": action, "existing_memory_ref": ref})
                    candidates[idx] = updated
                except Exception as exc:
                    log(self.logger, logging.WARNING, "processing.reconciliation_apply_failed", namespace_id=namespace_id, error=str(exc))
                    continue
            # Record reconciliation trace if needed (could be separate run)
            log(self.logger, logging.INFO, "processing.reconciliation_completed", namespace_id=namespace_id, decisions=len(result.response.decisions))
        except ProviderError as exc:
            log(self.logger, logging.WARNING, "processing.reconciliation_failed", namespace_id=namespace_id, error=str(exc), retryable=exc.retryable, retry_after=getattr(exc, "retry_after", None))
            if exc.retryable:
                # Retryable reconciliation failure must not silently persist partial extraction.
                raise
        except Exception as exc:
            log(self.logger, logging.WARNING, "processing.reconciliation_failed", namespace_id=namespace_id, error=str(exc))
        return candidates

    def process_events(
        self,
        namespace_id: str,
        event_ids: list[str],
        timeout_seconds: float = 30.0,
    ) -> tuple[int, int, int]:
        """Extract and store memories for one direct ingestion call."""
        if not event_ids:
            return 0, 0, 0

        events = self.repository.events_by_id(namespace_id, event_ids)
        if len(events) != len(event_ids):
            raise ValueError("one or more ingestion events are missing")

        provider = self.provider or default_extraction_provider()
        self.provider = provider
        deadline = time.perf_counter() + timeout_seconds
        included: dict[UUID, str] = {}
        current_events: dict[str, Any] = {}
        episode_ids: set[str] = set()

        for event in events:
            event_id = UUID(event["id"])
            included[event_id] = payload_text(json.loads(event["payload_json"]), event["type"], include_roles=True)
            current_events[str(event_id)] = event
            if event["episode_id"]:
                episode_ids.add(str(event["episode_id"]))

        # Current input is never split. Add only a small recent-context window.
        context_limit = len(included) + 10
        for event in events:
            for event_key, source in self.repository.extraction_window(namespace_id, event["id"], limit=4).items():
                if len(included) >= context_limit:
                    break
                included.setdefault(UUID(event_key), source)

        input_snapshot = {str(key): value for key, value in included.items()}
        # Retrieval during ingestion is only useful for the explicit reconciliation
        # call.  Skipping it avoids a remote embedding + rerank on every batch.
        existing_memories = self.repository.related_memory_context(namespace_id, "\n".join(input_snapshot.values())) if _is_reconciliation_enabled() else []
        existing_by_ref = {str(item["ref"]): str(item["memory_id"]) for item in existing_memories}

        # Multi-pass extraction
        extraction_schema = _get_extraction_schema()
        event_labels = {f"e{index + 1}": event_id for index, event_id in enumerate(included)}
        reverse_labels = {str(v): k for k, v in event_labels.items()}
        event_roles: dict[UUID, str] = {}
        for event_id, text in included.items():
            # Resolve role from stored events or fallback to current_events
            ev = current_events.get(str(event_id))
            if ev is not None:
                payload = json.loads(ev["payload_json"])
                messages = payload.get("messages", []) if isinstance(payload, dict) else []
                role = messages[0].get("role") if isinstance(messages, list) and messages and isinstance(messages[0], dict) else None
                event_roles[event_id] = role if role in {"user", "assistant"} else "user"
            else:
                # For context events, fetch via repo or default user
                try:
                    row = self.repository.db.execute("SELECT payload_json FROM events WHERE id=? AND namespace_id=?", (str(event_id), namespace_id)).fetchone()
                    if row:
                        payload = json.loads(row["payload_json"])
                        messages = payload.get("messages", []) if isinstance(payload, dict) else []
                        role = messages[0].get("role") if isinstance(messages, list) and messages and isinstance(messages[0], dict) else None
                        event_roles[event_id] = role if role in {"user", "assistant"} else "user"
                    else:
                        event_roles[event_id] = "user"
                except Exception:
                    event_roles[event_id] = "user"
        # Build v3 metadata: extractable vs context, timestamps, session ids
        extractable_uuids = [UUID(eid) for eid in event_ids]
        context_uuids = [eid for eid in included.keys() if eid not in extractable_uuids]
        event_timestamps: dict[UUID, str] = {}
        event_session_ids: dict[UUID, str] = {}
        for eid in included:
            ev = current_events.get(str(eid))
            if ev is not None:
                event_timestamps[eid] = str(ev["occurred_at"] or "")
                event_session_ids[eid] = str(ev["session_id"] or ev["stream_id"] or "")
            else:
                try:
                    row = self.repository.db.execute("SELECT occurred_at, session_id, stream_id FROM events WHERE id=? AND namespace_id=?", (str(eid), namespace_id)).fetchone()
                    if row:
                        event_timestamps[eid] = str(row["occurred_at"] or "")
                        event_session_ids[eid] = str(row["session_id"] or row["stream_id"] or "")
                except Exception:
                    pass
        chunk_labels: dict[str, str] = {}
        event_chunk_labels: dict[UUID, str] = {}
        label_index = 1
        for event_id in included:
            for chunk in self.repository.chunks_for_events(namespace_id, [str(event_id)], limit=1):
                label = f"c{label_index}"
                chunk_labels[label] = str(chunk["chunk_id"])
                event_chunk_labels[event_id] = label
                label_index += 1
        all_candidates, provider_results, meta = self._collect_multi_pass(
            provider, namespace_id, [UUID(eid) for eid in event_ids], included, existing_memories, deadline,
            event_labels=event_labels, chunk_labels=chunk_labels, event_chunk_labels=event_chunk_labels, event_roles=event_roles,
            extractable_event_ids=extractable_uuids, context_event_ids=context_uuids, event_timestamps=event_timestamps, event_session_ids=event_session_ids, extraction_schema=extraction_schema,
        )
        # LLM reconciliation (separate call) - skipped for v3 single-call by default
        if extraction_schema != "v3":
            all_candidates = self._apply_reconciliation(provider, namespace_id, existing_memories, all_candidates, deadline)

        # v3: typed multi-event grounded extraction
        v3_rejected: list[tuple[Any, str]] = []
        if extraction_schema == "v3" and all_candidates:
            # Build a light request stub for label resolution
            from ..models import ExtractionRequest as _Req

            _req_stub = _Req(
                namespace_id=namespace_id,
                events=[UUID(eid) for eid in event_ids],
                evidence_text=included,
                event_labels=event_labels,
                chunk_labels=chunk_labels,
                event_chunk_labels=event_chunk_labels,
                event_roles=event_roles,  # type: ignore[arg-type]
                existing_memories=existing_memories,
                extractable_event_ids=extractable_uuids,
                context_event_ids=context_uuids,
                event_timestamps=event_timestamps,
                event_session_ids=event_session_ids,
                extraction_schema="v3",
            )
            resolved, rejected = _resolve_v3_candidates(all_candidates, _req_stub, included, self.repository, namespace_id)
            v3_rejected = rejected
            # Apply session-level quality budget after grounding resolution but before chunk grounding
            session_map_str = {str(k): v for k, v in event_session_ids.items()}
            resolved = _enforce_session_quality_budget(resolved, session_map_str, included)
            all_candidates = resolved

        # Aggregate provider metadata for tracing
        if provider_results:
            # Merge prompt versions and token counts
            prompt_version = "+".join(pr.prompt_version for pr in provider_results)
            # Use first provider's metadata plus aggregated
            primary = provider_results[0]
            total_input = sum(pr.input_tokens or 0 for pr in provider_results if pr.input_tokens)
            total_output = sum(pr.output_tokens or 0 for pr in provider_results if pr.output_tokens)
            total_latency = sum(pr.latency_ms for pr in provider_results)
            provider_name = primary.provider_name
            model_name = primary.model_name
            # But keep candidates as merged list
        else:
            # No successful stage (should have raised), fallback
            provider_name = getattr(provider, "name", "unknown")
            model_name = getattr(provider, "model", "unknown")
            prompt_version = "unknown"
            total_input = None
            total_output = None
            total_latency = 0

        # Phase 2: chunk grounding — scope to extraction batch, reject ungrounded.
        # Only chunks covering events sent to this extraction are valid.
        # Build batch-scoped valid chunk set: chunks whose event_ids overlap included events
        included_event_ids = {str(eid) for eid in included.keys()} | {str(ev["id"]) for ev in events}
        batch_valid_chunk_ids: set[str] = set()
        chunk_session = self.repository.chunk_session_map(namespace_id)
        for row in self.repository.db.execute("SELECT chunk_id, event_ids_json FROM chunks WHERE namespace_id=?", (namespace_id,)).fetchall():
            try:
                cids = set(json.loads(row["event_ids_json"]))
            except Exception:
                cids = set()
            if cids & included_event_ids:
                batch_valid_chunk_ids.add(str(row["chunk_id"]))
        # Fallback if no chunks yet (e.g., legacy DB): allow ungrounded but don't accept cross-session forged IDs
        # For scoping, we use batch_valid if non-empty, otherwise all valid
        valid_chunk_ids = batch_valid_chunk_ids if batch_valid_chunk_ids else self.repository.chunk_ids_for_namespace(namespace_id)
        # For strict grounding, we require batch scoping when chunks exist
        scoping_ids = batch_valid_chunk_ids if batch_valid_chunk_ids else valid_chunk_ids
        event_session_map: dict[str, str] = {}
        for ev in events:
            sid = str(ev["session_id"] or ev["stream_id"] or "")
            if sid:
                event_session_map[str(ev["id"])] = sid
        grounded_candidates: list[Any] = []
        rejected_grounding: list[Any] = []
        grounded_v2 = any(pr.response.prompt_version.startswith("grounded-v2") for pr in provider_results)
        for candidate in all_candidates:
            chunk_ids = list(getattr(candidate, "source_chunk_ids", []) or [])
            if chunk_ids:
                # Scoped repair: keep only batch-covering chunk IDs
                repaired = [cid for cid in chunk_ids if cid in scoping_ids]
                if not repaired and chunk_ids:
                    log(self.logger, logging.WARNING, "processing.unknown_source_chunk_id", namespace_id=namespace_id, chunk_ids=chunk_ids)
                    # Reject: all references are fabricated or stale/cross-session
                    rejected_grounding.append(candidate)
                    continue
                # Enforce session attachment: repaired chunk must belong to a session in this batch
                if repaired:
                    batch_sessions = set(event_session_map.values())
                    # At least one repaired chunk must belong to a batch session
                    chunk_sessions = {chunk_session.get(cid, "") for cid in repaired}
                    if batch_sessions and chunk_sessions.isdisjoint(batch_sessions) and "" not in chunk_sessions:
                        log(self.logger, logging.WARNING, "processing.cross_session_chunk_id", namespace_id=namespace_id, chunk_ids=repaired)
                        rejected_grounding.append(candidate)
                        continue
                    candidate = candidate.model_copy(update={"source_chunk_ids": repaired})
                grounded_candidates.append(candidate)
            else:
                if grounded_v2:
                    rejected_grounding.append(candidate)
                    continue
                grounded_candidates.append(candidate)
        # Record rejected grounding as decisions later; for now exclude them
        # Include v3 resolution rejections
        if extraction_schema == "v3" and v3_rejected:
            # v3_rejected contains (candidate, reason); map to candidate list for recording
            # Keep candidate for decision logging; reasons will be used below
            rejected_grounding.extend([cand for cand, _ in v3_rejected])
        all_candidates = _prune_event_candidates(grounded_candidates)
        # Stash rejected for later recording
        _rejected_grounding = rejected_grounding
        _v3_rejected_map = {id(cand): reason for cand, reason in (v3_rejected if extraction_schema == "v3" else [])}

        # Resolve refs to IDs (structural integrity)
        resolved_candidates = []
        for candidate in all_candidates:
            if candidate.existing_memory_ref is None:
                resolved_candidates.append(candidate)
                continue
            existing_id = existing_by_ref.get(candidate.existing_memory_ref)
            resolved_candidates.append(
                candidate if existing_id is None else candidate.model_copy(update={"existing_memory_id": UUID(existing_id)})
            )

        # Build a pseudo response for downstream (keep candidates merged)
        # We need to record a run; use first schema version or default
        if extraction_schema == "v3":
            schema_version = "extraction-v3"
        else:
            schema_version = provider_results[0].response.schema_version if provider_results else "extraction-v1"

        run_id = str(uuid.uuid4())
        # Ensure detailed tracing columns exist (phase 6)
        _ensure_stage_column(self.repository)
        # Record aggregated run; per-stage rows could be added but aggregated keeps backward compat
        self.repository.record_run(
            namespace_id,
            {
                "id": run_id,
                "namespace_id": namespace_id,
                "input_hash": hashlib.sha256(json.dumps(input_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "provider_name": provider_name,
                "model_name": model_name,
                "prompt_version": prompt_version[:100],  # truncate to fit schema
                "schema_version": schema_version,
                "started_at": self._now(),
                "completed_at": None,
                "input_events_json": json.dumps(list(input_snapshot), separators=(",", ":")),
                "input_characters": sum(len(text) for text in input_snapshot.values()),
                "input_tokens": total_input if total_input else None,
                "output_tokens": total_output if total_output else None,
                "latency_ms": total_latency or int((time.perf_counter() - (deadline - timeout_seconds)) * 1000),
                "accepted_count": 0,
                "rejected_count": 0,
                "status": "processing",
                "error_class": None,
                "estimated_cost_usd": self._estimated_cost(total_input if total_input else None, total_output if total_output else None),
            },
        )

        accepted = rejected = 0
        # Pre-record grounding rejections (all chunk IDs unknown/cross-session) and v3 resolutions
        for _rg in _rejected_grounding:
            rejected += 1
            try:
                reason = _v3_rejected_map.get(id(_rg), "unknown_source_chunk_id")
                self.repository.record_decision(namespace_id, run_id, _rg, self._safe_fingerprint(_rg), "rejected", reason, "REJECT")
            except Exception:
                pass
        try:
            fingerprints: set[str] = set()
            validated_candidates: list[tuple[Any, Any, Any]] = []
            # Collect valid chunk IDs for validation — scoped to batch
            valid_chunks_for_validation: set[str] | None = scoping_ids if scoping_ids else None
            for candidate in resolved_candidates:
                try:
                    if candidate.existing_memory_ref is not None and candidate.existing_memory_id is None:
                        raise CandidateRejected("unknown_existing_memory_ref")
                    # Validate intent action names — simplified to 5 actions per Phase 2
                    action = str(candidate.intent).casefold() if hasattr(candidate, "intent") else "insert"
                    allowed_intents = {"insert", "reinforce", "update", "supersede", "ignore"}
                    # Backward compat: map dispute/contradiction -> ignore (no longer a distinct state)
                    if action in {"dispute", "contradiction"}:
                        action = "ignore"
                        candidate = candidate.model_copy(update={"intent": "ignore"})
                    if action not in allowed_intents:
                        raise CandidateRejected("invalid_intent")
                    validated = validate_candidate(
                        namespace_id,
                        candidate,
                        included,
                        included_chunks=valid_chunks_for_validation,
                        require_evidence=True,
                    )
                    if validated.fingerprint in fingerprints:
                        raise CandidateRejected("duplicate_candidate")
                    fingerprints.add(validated.fingerprint)
                    if not candidate.evidence:
                        raise CandidateRejected("missing_source_evidence")
                    # For v3 multi-event, allow any evidence event that is extractable
                    if extraction_schema == "v3":
                        # pick latest source event among evidence for observed_at
                        source_event = None
                        latest_ts = ""
                        for span in candidate.evidence:
                            ev = current_events.get(str(span.event_id))
                            if ev is not None:
                                ts = str(ev["occurred_at"] or "")
                                if ts >= latest_ts:
                                    latest_ts = ts
                                    source_event = ev
                        if source_event is None:
                            # fallback to first if none in batch but at least one is extractable (should not happen)
                            source_event = current_events.get(str(candidate.evidence[0].event_id))
                        if source_event is None:
                            raise CandidateRejected("evidence_not_in_ingestion_batch")
                    else:
                        source_event = current_events.get(str(candidate.evidence[0].event_id))
                        if source_event is None:
                            raise CandidateRejected("evidence_not_in_ingestion_batch")
                    validated_candidates.append((candidate, validated, source_event))
                except CandidateRejected as exc:
                    rejected += 1
                    self.repository.record_decision(
                        namespace_id, run_id, candidate, self._safe_fingerprint(candidate), "rejected", exc.reason, "REJECT"
                    )

            statements = [validated.candidate.statement for _, validated, _ in validated_candidates]
            embeddings = self.repository.embed_many(statements) if statements else []
            for (candidate, validated, source_event), embedding in zip(validated_candidates, embeddings, strict=True):
                try:
                    memory_id, action, version_id = self.repository.reconcile_candidate(
                        namespace_id, source_event, validated, run_id, embedding
                    )
                    self.repository.record_decision(
                        namespace_id, run_id, candidate, validated.fingerprint, "accepted", None, action, memory_id, version_id
                    )
                    accepted += 1
                except CandidateRejected as exc:
                    rejected += 1
                    self.repository.record_decision(
                        namespace_id, run_id, candidate, self._safe_fingerprint(candidate), "rejected", exc.reason, "REJECT"
                    )

            self.repository.finish_run(namespace_id, run_id, accepted, rejected, "completed")
        except Exception as exc:
            error_class = exc.error_class if isinstance(exc, ProviderError) else type(exc).__name__
            self.repository.finish_run(namespace_id, run_id, accepted, rejected, "failed", error_class)
            raise

        for episode_id in sorted(episode_ids):
            try:
                self.repository.refresh_episode_summary(namespace_id, episode_id, summary_provider=self.summary_provider)
            except Exception:
                log(self.logger, logging.WARNING, "processing.summary_refresh_failed", namespace_id=namespace_id, episode_id=episode_id)

        log(
            self.logger,
            logging.INFO,
            "processing.completed",
            namespace_id=namespace_id,
            event_count=len(events),
            candidates=len(resolved_candidates),
            accepted=accepted,
            rejected=rejected,
        )
        return len(events), accepted, rejected

    def process_namespace(self, namespace_id: str, limit: int = 100, lease_seconds: int = 180, timeout_seconds: float = 30.0) -> tuple[int, int, int, int, int]:
        deadline = time.perf_counter() + timeout_seconds
        jobs = self.repository.claim_jobs(namespace_id, min(limit, 100), lease_seconds)
        provider = self.provider or default_extraction_provider()
        self.provider = provider
        processed = failed = dead = accepted = rejected = 0
        episode_ids: set[str] = set()
        scoped_jobs: dict[str, list[tuple[Any, Any]]] = {}
        for job in jobs:
            event = self.repository.event_for_job(namespace_id, job["id"])
            # Events without an explicit conversation scope must not be mixed
            # with unrelated events just because they share a namespace.
            scope = str(event["stream_id"] or event["session_id"] or event["id"])
            scoped_jobs.setdefault(scope, []).append((job, event))

        # Extract a bounded session batch in one provider call. This keeps the
        # evidence window coherent and avoids one LLM request per event.
        batches: list[list[tuple[Any, Any]]] = []
        for scoped in scoped_jobs.values():
            for start in range(0, len(scoped), 20):
                batches.append(scoped[start : start + 20])

        for batch in batches:
            if time.perf_counter() >= deadline:
                break
            run_id = str(uuid.uuid4())
            started = time.perf_counter()
            batch_accepted = batch_rejected = 0
            included: dict[UUID, str] = {}
            current_jobs: dict[str, tuple[Any, Any]] = {}
            input_snapshot: dict[str, str] = {}
            try:
                for job, event in batch:
                    if event["episode_id"]:
                        episode_ids.add(str(event["episode_id"]))
                    if not self.repository.heartbeat_job(namespace_id, job["id"], lease_seconds, str(job["lease_token"])):
                        raise RuntimeError("job lease is no longer active")
                    event_id = UUID(event["id"])
                    current_jobs[str(event_id)] = (job, event)
                    # Keep retry processing semantically identical to direct
                    # ingestion: role labels are source data for assistant
                    # knowledge and must remain visible to the extractor.
                    source = payload_text(json.loads(event["payload_json"]), event["type"], include_roles=True)
                    included[event_id] = source

                # Add prior same-session turns after current events. Every
                # current event stays addressable for exact evidence offsets.
                for _job, event in batch:
                    for event_key, source in self.repository.extraction_window(namespace_id, event["id"], limit=4).items():
                        if len(included) >= 20:
                            break
                        included.setdefault(UUID(event_key), source)

                # Build v3 metadata for batch
                extraction_schema_batch = _get_extraction_schema()
                input_snapshot = {str(key): value for key, value in included.items()}
                # v3 is deliberately one extraction call with no LLM
                # reconciliation.  Searching here spends an embedding request
                # and rerank without supplying useful prompt context.
                existing_memories = (
                    self.repository.related_memory_context(namespace_id, "\n".join(input_snapshot.values()))
                    if extraction_schema_batch != "v3" and _is_reconciliation_enabled()
                    else []
                )
                existing_by_ref = {str(item["ref"]): str(item["memory_id"]) for item in existing_memories}
                event_labels_batch = {f"e{index + 1}": eid for index, eid in enumerate(included)}
                event_roles_batch: dict[UUID, str] = {}
                event_timestamps_batch: dict[UUID, str] = {}
                event_session_ids_batch: dict[UUID, str] = {}
                for eid in included:
                    ev = current_jobs.get(str(eid), (None, None))[1]
                    if ev is not None:
                        payload = json.loads(ev["payload_json"]) if isinstance(ev["payload_json"], str) else ev["payload_json"]
                        if isinstance(payload, str):
                            try:
                                payload = json.loads(payload)
                            except Exception:
                                payload = {}
                        messages = payload.get("messages", []) if isinstance(payload, dict) else []
                        role = messages[0].get("role") if isinstance(messages, list) and messages and isinstance(messages[0], dict) else None
                        event_roles_batch[eid] = role if role in {"user", "assistant"} else "user"
                        event_timestamps_batch[eid] = str(ev["occurred_at"] or "")
                        event_session_ids_batch[eid] = str(ev["session_id"] or ev["stream_id"] or "")
                    else:
                        try:
                            row = self.repository.db.execute("SELECT payload_json, occurred_at, session_id, stream_id FROM events WHERE id=? AND namespace_id=?", (str(eid), namespace_id)).fetchone()
                            if row:
                                payload = json.loads(row["payload_json"])
                                messages = payload.get("messages", []) if isinstance(payload, dict) else []
                                role = messages[0].get("role") if isinstance(messages, list) and messages and isinstance(messages[0], dict) else None
                                event_roles_batch[eid] = role if role in {"user", "assistant"} else "user"
                                event_timestamps_batch[eid] = str(row["occurred_at"] or "")
                                event_session_ids_batch[eid] = str(row["session_id"] or row["stream_id"] or "")
                            else:
                                event_roles_batch[eid] = "user"
                        except Exception:
                            event_roles_batch[eid] = "user"
                chunk_labels_batch: dict[str, str] = {}
                event_chunk_labels_batch: dict[UUID, str] = {}
                label_index_batch = 1
                for eid in included:
                    for chunk in self.repository.chunks_for_events(namespace_id, [str(eid)], limit=1):
                        label = f"c{label_index_batch}"
                        chunk_labels_batch[label] = str(chunk["chunk_id"])
                        event_chunk_labels_batch[eid] = label
                        label_index_batch += 1
                try:
                    extractable_uuids_batch = [UUID(ev["id"]) for _, ev in batch]
                except Exception:
                    extractable_uuids_batch = [UUID(x) for x in list(current_jobs.keys())]
                context_uuids_batch = [eid for eid in included.keys() if eid not in extractable_uuids_batch]
                # Multi-pass extraction for batch
                all_candidates, provider_results, meta = self._collect_multi_pass(
                    provider, namespace_id, list(included), included, existing_memories, deadline,
                    event_labels=event_labels_batch, chunk_labels=chunk_labels_batch, event_chunk_labels=event_chunk_labels_batch, event_roles=event_roles_batch,
                    extractable_event_ids=extractable_uuids_batch, context_event_ids=context_uuids_batch, event_timestamps=event_timestamps_batch, event_session_ids=event_session_ids_batch, extraction_schema=extraction_schema_batch,
                )
                # Reconciliation - skip for v3 single-call
                if extraction_schema_batch != "v3":
                    all_candidates = self._apply_reconciliation(provider, namespace_id, existing_memories, all_candidates, deadline)
                # v3 resolution for batch
                v3_rejected_batch: list[tuple[Any, str]] = []
                if extraction_schema_batch == "v3" and all_candidates:
                    from ..models import ExtractionRequest as _ReqBatch

                    _req_stub_batch = _ReqBatch(
                        namespace_id=namespace_id,
                        events=list(included.keys()),
                        evidence_text=included,
                        event_labels=event_labels_batch,
                        chunk_labels=chunk_labels_batch,
                        event_chunk_labels=event_chunk_labels_batch,
                        event_roles=event_roles_batch,  # type: ignore[arg-type]
                        existing_memories=existing_memories,
                        extractable_event_ids=extractable_uuids_batch,
                        context_event_ids=context_uuids_batch,
                        event_timestamps=event_timestamps_batch,
                        event_session_ids=event_session_ids_batch,
                        extraction_schema="v3",
                    )
                    resolved_b, rejected_b = _resolve_v3_candidates(all_candidates, _req_stub_batch, included, self.repository, namespace_id)
                    v3_rejected_batch = rejected_b
                    # Session-level quality budget for batch
                    session_map_str_batch = {str(k): v for k, v in event_session_ids_batch.items()}
                    resolved_b = _enforce_session_quality_budget(resolved_b, session_map_str_batch, included)
                    all_candidates = resolved_b
                # Phase 2: chunk grounding for batch — scoped to batch events, reject ungrounded
                included_event_ids_batch = {str(eid) for eid in included.keys()}
                batch_scoped_chunk_ids: set[str] = set()
                for brow in self.repository.db.execute("SELECT chunk_id, event_ids_json FROM chunks WHERE namespace_id=?", (namespace_id,)).fetchall():
                    try:
                        bcids = set(json.loads(brow["event_ids_json"]))
                    except Exception:
                        bcids = set()
                    if bcids & included_event_ids_batch:
                        batch_scoped_chunk_ids.add(str(brow["chunk_id"]))
                if batch_scoped_chunk_ids:
                    scoping_ids_batch = batch_scoped_chunk_ids
                else:
                    _all_chunk_ids = self.repository.chunk_ids_for_namespace(namespace_id)
                    scoping_ids_batch = _all_chunk_ids if _all_chunk_ids else set()
                # Need chunk session map for cross-session check
                chunk_session_batch = self.repository.chunk_session_map(namespace_id)
                batch_sessions_set: set[str] = set()
                for _jid, _ev in batch:
                    bs = str(_ev["session_id"] or _ev["stream_id"] or "")
                    if bs:
                        batch_sessions_set.add(bs)
                repaired_batch: list[Any] = []
                rejected_batch_grounding: list[Any] = []
                for candidate in all_candidates:
                    chunk_ids = list(getattr(candidate, "source_chunk_ids", []) or [])
                    if chunk_ids:
                        repaired = [cid for cid in chunk_ids if cid in scoping_ids_batch]
                        if not repaired and chunk_ids:
                            rejected_batch_grounding.append(candidate)
                            continue
                        if repaired:
                            repaired_sessions = {chunk_session_batch.get(cid, "") for cid in repaired}
                            if batch_sessions_set and repaired_sessions.isdisjoint(batch_sessions_set) and "" not in repaired_sessions:
                                rejected_batch_grounding.append(candidate)
                                continue
                            if repaired != chunk_ids:
                                candidate = candidate.model_copy(update={"source_chunk_ids": repaired})
                    repaired_batch.append(candidate)
                all_candidates = _prune_event_candidates(repaired_batch)
                # Merge v3 resolution rejections into batch grounding rejections
                _v3_batch_rejected_map: dict[int, str] = {}
                if extraction_schema_batch == "v3" and v3_rejected_batch:
                    for cand, reason in v3_rejected_batch:
                        _v3_batch_rejected_map[id(cand)] = reason
                        if cand not in _rejected_batch_grounding:
                            rejected_batch_grounding.append(cand)
                _rejected_batch_grounding = rejected_batch_grounding
                _v3_batch_map = _v3_batch_rejected_map
                for job, _event in batch:
                    if not self.repository.heartbeat_job(namespace_id, job["id"], lease_seconds, str(job["lease_token"])):
                        raise RuntimeError("job lease is no longer active")
                # Resolve refs
                resolved_candidates: list[Any] = []
                for candidate in all_candidates:
                    if candidate.existing_memory_ref is None:
                        resolved_candidates.append(candidate)
                        continue
                    existing_id = existing_by_ref.get(candidate.existing_memory_ref)
                    if existing_id is None:
                        resolved_candidates.append(candidate)
                        continue
                    resolved_candidates.append(candidate.model_copy(update={"existing_memory_id": uuid.UUID(existing_id)}))
                # Aggregate provider metadata
                if provider_results:
                    prompt_version = "+".join(pr.prompt_version for pr in provider_results)[:100]
                    provider_name, model_name = provider_results[0].provider_name, provider_results[0].model_name
                    total_input = sum(pr.input_tokens or 0 for pr in provider_results if pr.input_tokens)
                    total_output = sum(pr.output_tokens or 0 for pr in provider_results if pr.output_tokens)
                    provider_latency = sum(pr.latency_ms for pr in provider_results)
                    if extraction_schema_batch == "v3":
                        schema_version = "extraction-v3"
                    else:
                        schema_version = provider_results[0].response.schema_version
                else:
                    provider_name = getattr(provider, "name", "unknown")
                    model_name = getattr(provider, "model", "unknown")
                    prompt_version = "unknown"
                    total_input = None
                    total_output = None
                    provider_latency = int((time.perf_counter() - started) * 1000)
                    schema_version = "extraction-v3" if extraction_schema_batch == "v3" else "extraction-v1"
                _ensure_stage_column(self.repository)
                self.repository.record_run(
                    namespace_id,
                    {
                        "id": run_id,
                        "namespace_id": namespace_id,
                        "input_hash": hashlib.sha256(json.dumps(input_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                        "provider_name": provider_name,
                        "model_name": model_name,
                        "prompt_version": prompt_version,
                        "schema_version": schema_version,
                        "started_at": self._now(),
                        "completed_at": None,
                        "input_events_json": json.dumps(list(input_snapshot), separators=(",", ":")),
                        "input_characters": sum(len(text) for text in input_snapshot.values()),
                        "input_tokens": total_input if total_input else None,
                        "output_tokens": total_output if total_output else None,
                        "latency_ms": provider_latency or int((time.perf_counter() - started) * 1000),
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "status": "processing",
                        "error_class": None,
                        "estimated_cost_usd": self._estimated_cost(total_input if total_input else None, total_output if total_output else None),
                    },
                )
                # Count and record batch grounding rejections
                batch_rejected_grounding = len(_rejected_batch_grounding)
                for _rgb in _rejected_batch_grounding:
                    try:
                        reason = _v3_batch_map.get(id(_rgb), "unknown_source_chunk_id") if extraction_schema_batch == "v3" else "unknown_source_chunk_id"
                        self.repository.record_decision(namespace_id, run_id, _rgb, self._safe_fingerprint(_rgb), "rejected", reason, "REJECT")
                    except Exception:
                        pass
                rejected += batch_rejected_grounding
                batch_rejected += batch_rejected_grounding
                fingerprints: set[str] = set()
                validated_candidates: list[tuple[Any, Any, Any, Any]] = []
                valid_chunks_batch_set: set[str] | None = scoping_ids_batch if scoping_ids_batch else None
                for candidate in resolved_candidates:
                    try:
                        if candidate.existing_memory_ref is not None and candidate.existing_memory_id is None:
                            raise CandidateRejected("unknown_existing_memory_ref")
                        action = str(candidate.intent).casefold() if hasattr(candidate, "intent") else "insert"
                        allowed_intents = {"insert", "reinforce", "update", "supersede", "ignore"}
                        if action in {"dispute", "contradiction"}:
                            action = "ignore"
                            candidate = candidate.model_copy(update={"intent": "ignore"})
                        if action not in allowed_intents:
                            raise CandidateRejected("invalid_intent")
                        validated = validate_candidate(namespace_id, candidate, included, included_chunks=valid_chunks_batch_set)
                        if validated.fingerprint in fingerprints:
                            raise CandidateRejected("duplicate_candidate")
                        fingerprints.add(validated.fingerprint)
                        if candidate.evidence:
                            # For v3 multi-event, pick latest extractable evidence event
                            if extraction_schema_batch == "v3":
                                pick = None
                                latest_ts = ""
                                for span in candidate.evidence:
                                    key = str(span.event_id)
                                    if key in current_jobs:
                                        ev_row = current_jobs[key][1]
                                        ts = str(ev_row["occurred_at"] or "")
                                        if ts >= latest_ts:
                                            latest_ts = ts
                                            pick = current_jobs[key]
                                source_job, source_event = pick if pick is not None else current_jobs.get(str(candidate.evidence[0].event_id), batch[0])
                            else:
                                source_job, source_event = current_jobs.get(str(candidate.evidence[0].event_id), batch[0])
                        else:
                            source_job, source_event = batch[0]
                        validated_candidates.append((candidate, validated, source_job, source_event))
                    except CandidateRejected as exc:
                        rejected += 1
                        batch_rejected += 1
                        self.repository.record_decision(namespace_id, run_id, candidate, self._safe_fingerprint(candidate), "rejected", exc.reason, "REJECT")

                embedding_values = [validated.candidate.statement for _, validated, _job, _event in validated_candidates]
                embedding_vectors = self.repository.embed_many(embedding_values) if embedding_values else []
                for (candidate, validated, source_job, source_event), embedding in zip(validated_candidates, embedding_vectors, strict=True):
                    try:
                        reconciled_memory_id, action, reconciled_version_id = self.repository.reconcile_candidate(
                            namespace_id,
                            source_event,
                            validated,
                            run_id,
                            embedding,
                            job_id=str(source_job["id"]),
                            lease_token=str(source_job["lease_token"]),
                        )
                        memory_id = reconciled_memory_id
                        version_id = reconciled_version_id
                        self.repository.record_decision(namespace_id, run_id, candidate, validated.fingerprint, "accepted", None, action, memory_id, version_id)
                        accepted += 1
                        batch_accepted += 1
                    except CandidateRejected as exc:
                        rejected += 1
                        batch_rejected += 1
                        self.repository.record_decision(namespace_id, run_id, candidate, self._safe_fingerprint(candidate), "rejected", exc.reason, "REJECT")
                self.repository.finish_run(namespace_id, run_id, batch_accepted, batch_rejected, "completed")
                for job, _event in batch:
                    if self.repository.complete_job(namespace_id, job["id"], str(job["lease_token"])) is False:
                        raise RuntimeError("job lease is no longer active")
                processed += len(batch)
                log(
                    self.logger,
                    logging.INFO,
                    "processing.completed",
                    namespace_id=namespace_id,
                    job_id=str(batch[0][0]["id"]),
                    batch_jobs=len(batch),
                    candidates=len(resolved_candidates),
                    accepted=batch_accepted,
                    rejected=batch_rejected,
                )
            except Exception as exc:
                safe_error = redact_text(str(exc))
                error_class = exc.error_class if isinstance(exc, ProviderError) else type(exc).__name__
                if not self._run_exists(namespace_id, run_id) and input_snapshot and self.provider is not None:
                    _ensure_stage_column(self.repository)
                    self.repository.record_run(
                        namespace_id,
                        {
                            "id": run_id,
                            "namespace_id": namespace_id,
                            "input_hash": hashlib.sha256(json.dumps(input_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                            "provider_name": self.provider.name,
                            "model_name": self.provider.model,
                            "prompt_version": "unknown",
                            "schema_version": "extraction-v1",
                            "started_at": self._now(),
                            "completed_at": None,
                            "input_events_json": json.dumps(list(input_snapshot), separators=(",", ":")),
                            "input_characters": sum(len(text) for text in input_snapshot.values()),
                            "input_tokens": None,
                            "output_tokens": None,
                            "latency_ms": int((time.perf_counter() - started) * 1000),
                            "accepted_count": 0,
                            "rejected_count": 0,
                            "status": "processing",
                            "error_class": None,
                            "estimated_cost_usd": None,
                        },
                    )
                if self._run_exists(namespace_id, run_id):
                    self.repository.finish_run(namespace_id, run_id, batch_accepted, batch_rejected, "failed", error_class)
                for job, _event in batch:
                    retry_after = getattr(exc, "retry_after", None) if isinstance(exc, ProviderError) else None
                    # Fallback parse from message
                    if retry_after is None and "retry_after=" in safe_error:
                        try:
                            import re

                            m = re.search(r"retry_after=([0-9.]+)", safe_error)
                            if m:
                                retry_after = float(m.group(1))
                        except Exception:
                            retry_after = None
                    status = self.repository.fail_job(
                        namespace_id,
                        job["id"],
                        safe_error,
                        retryable=not isinstance(exc, ProviderError) or exc.retryable,
                        retry_after=retry_after,
                        lease_token=str(job["lease_token"]),
                    )
                    failed += 1
                    dead += status == "dead"
                    log(self.logger, logging.ERROR, "processing.failed", namespace_id=namespace_id, job_id=job["id"], status=status, error=safe_error, retry_after=retry_after)
        for episode_id in sorted(episode_ids):
            try:
                self.repository.refresh_episode_summary(namespace_id, episode_id, summary_provider=self.summary_provider)
            except Exception:
                log(self.logger, logging.WARNING, "processing.summary_refresh_failed", namespace_id=namespace_id, episode_id=episode_id)
        return processed, failed, dead, accepted, rejected

    @staticmethod
    def _estimated_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
        input_rate = os.environ.get("TERMYTEDB_INPUT_COST_PER_1K_USD")
        output_rate = os.environ.get("TERMYTEDB_OUTPUT_COST_PER_1K_USD")
        if input_tokens is None or output_tokens is None or input_rate is None or output_rate is None:
            return None
        try:
            return round((input_tokens * float(input_rate) + output_tokens * float(output_rate)) / 1000, 8)
        except ValueError:
            return None

    def _run_exists(self, namespace_id: str, run_id: str) -> bool:
        return self.repository.db.execute("SELECT 1 FROM extraction_runs WHERE id=? AND namespace_id=?", (run_id, namespace_id)).fetchone() is not None

    @staticmethod
    def _safe_fingerprint(candidate: Any) -> str:
        try:
            value = json.dumps(candidate.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        except Exception:
            value = redact_text(str(candidate))
        return hashlib.sha256(value.encode()).hexdigest()

    @staticmethod
    def _now() -> str:
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()
