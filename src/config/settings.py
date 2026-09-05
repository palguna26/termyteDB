"""Central configuration for model sizes, limits, and retrieval tuning.

All magic numbers that were previously scattered across
`memory/provider.py`, `retrieval/embedding.py`, and `storage/repository.py`
now live here so a technical founder can audit and change them in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingSettings:
    """Embedding model defaults and overrides via environment."""

    # Local CPU model
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_dimensions: int = 384
    fastembed_batch_size: int = 256

    # Remote OpenAI-compatible model
    openai_default_model: str = os.environ.get("TERMYTEDB_EMBEDDING_MODEL", "")
    openai_default_dimensions: int = int(os.environ.get("TERMYTEDB_EMBEDDING_DIMENSIONS", "1024"))
    openai_default_base_url: str = os.environ.get("TERMYTEDB_EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1")
    openai_timeout: float = 60.0
    openai_retries: int = int(os.environ.get("TERMYTEDB_EMBEDDING_RETRIES", "6"))

    # SQLite-vec virtual table
    vec_distance_metric: str = "cosine"


@dataclass(frozen=True)
class ExtractionSettings:
    """LLM extraction tuning."""

    prompt_version: str = "v1"
    schema_version: str = "extraction-v1"
    temperature: float = float(os.environ.get("TERMYTEDB_EXTRACTION_TEMPERATURE", "0") or 0)
    max_tokens: int = 2500
    summary_max_tokens: int = 220
    request_timeout: float = 30.0
    summary_timeout: float = 45.0
    # Phase 8 provider controls
    stages: tuple[str, ...] = tuple(os.environ.get("TERMYTEDB_EXTRACTION_STAGES", "facts").split(",")) if os.environ.get("TERMYTEDB_EXTRACTION_STAGES") else ("facts",)
    reconciliation_enabled: bool = os.environ.get("TERMYTEDB_RECONCILIATION_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
    summary_enabled: bool = os.environ.get("TERMYTEDB_SUMMARY_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
    max_llm_calls_per_batch: int = int(os.environ.get("TERMYTEDB_MAX_LLM_CALLS_PER_BATCH", "10") or 10)
    reconciliation_model: str = os.environ.get("TERMYTEDB_RECONCILIATION_MODEL", os.environ.get("TERMYTEDB_EXTRACTION_MODEL", "") or "")
    reconciliation_temperature: float = float(os.environ.get("TERMYTEDB_RECONCILIATION_TEMPERATURE", os.environ.get("TERMYTEDB_EXTRACTION_TEMPERATURE", "0") or 0) or 0)


@dataclass(frozen=True)
class RetrievalSettings:
    """Hybrid search + rerank."""

    rrf_k: int = 60
    lexical_overfetch: int = 5  # multiplier for FTS candidate pool (limit * 5)
    vector_overfetch: int = 5
    vector_score_floor: float = 0.6
    reranker_threshold: float = 0.25
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    default_search_limit: int = 10
    # Phase 3: hybrid chunk retrieval + reranking
    chunk_vector_weight: float = 0.6
    chunk_lexical_weight: float = 0.4
    reranker_enabled: bool = os.environ.get("TERMYTEDB_RERANKER_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
    # Optional OpenRouter reranker. Empty keeps local FlashRank as the default.
    reranking_model: str = os.environ.get("TERMYTEDB_RERANKING_MODEL", "").strip()
    reranker_max_candidates: int = int(os.environ.get("TERMYTEDB_RERANKER_MAX_CANDIDATES", "30") or 30)
    reranker_max_chars: int = int(os.environ.get("TERMYTEDB_RERANKER_MAX_CHARS", "600") or 600)
    diversity_max_per_session: int = 2
    fts_weight_identifiers: float = 1.5
    vector_weight_conceptual: float = 1.5
    # Phase 2: temporal scoring (dates influence ranking, not hard filters).
    temporal_boost_latest: float = float(os.environ.get("TERMYTEDB_TEMPORAL_BOOST_LATEST", "0.05") or 0.05)
    temporal_boost_historical: float = float(os.environ.get("TERMYTEDB_TEMPORAL_BOOST_HISTORICAL", "0.03") or 0.03)
    temporal_boost_in_range: float = float(os.environ.get("TERMYTEDB_TEMPORAL_BOOST_IN_RANGE", "0.08") or 0.08)
    temporal_penalty_future: float = float(os.environ.get("TERMYTEDB_TEMPORAL_PENALTY_FUTURE", "0.03") or 0.03)
    # Phase 3: preference signals.
    preference_boost_positive: float = float(os.environ.get("TERMYTEDB_PREFERENCE_BOOST", "0.04") or 0.04)
    preference_boost_update: float = float(os.environ.get("TERMYTEDB_PREFERENCE_UPDATE_BOOST", "0.02") or 0.02)
    # Phase 4: multi-session aggregation.
    multi_session_reserve: float = float(os.environ.get("TERMYTEDB_MULTI_SESSION_RESERVE", "0.4") or 0.4)
    multi_session_max_per_session: int = int(os.environ.get("TERMYTEDB_MULTI_SESSION_MAX_PER_SESSION", "3") or 3)
    multi_session_evidence_share: float = float(os.environ.get("TERMYTEDB_MULTI_SESSION_EVIDENCE_SHARE", "0.5") or 0.5)
    # Phase 5: latency controls.
    candidate_overfetch_cap: int = int(os.environ.get("TERMYTEDB_CANDIDATE_OVERFETCH_CAP", "100") or 100)
    chunk_search_enabled_when_no_dense: bool = os.environ.get("TERMYTEDB_CHUNK_NO_DENSE", "1").strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class MemorySettings:
    """Temporal and reconciliation defaults."""

    default_durability: str = "session"
    max_candidates_per_event: int = 3
    statement_min_chars: int = 10
    statement_max_chars: int = 150
    max_evidence_spans: int = 8


# Singletons used throughout the codebase
EMBEDDING = EmbeddingSettings()
EXTRACTION = ExtractionSettings()
RETRIEVAL = RetrievalSettings()
MEMORY = MemorySettings()
