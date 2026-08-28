"""LLM provider registry - swap models / vendors from config, not code.

Import provider classes from here. Changing the default model, endpoint,
or adding a new vendor requires editing only this file (and `settings.py`).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory.provider import ExtractionProvider, SessionSummaryProvider
    from ..retrieval.embedding import EmbeddingProvider


def get_extraction_provider(
    provider_name: str | None = None,
) -> ExtractionProvider:
    """Factory that resolves the configured extraction provider.

    Resolution order (mirrors `memory.provider.configured_extraction_provider`):
    1. explicit `provider_name`
    2. TERMYTEDB_EXTRACTION_PROVIDER env
    3. HTTP endpoint if TERMYTEDB_EXTRACTION_URL is set
    4. OpenRouter if API key present
    5. Fake (offline / tests) as fallback
    """
    from ..memory.provider import (
        FakeExtractionProvider,
        HttpExtractionProvider,
        OpenRouterExtractionProvider,
        configured_extraction_provider,
    )

    # If caller forced a name, honour it without env sniffing
    if provider_name == "fake":
        return FakeExtractionProvider()
    if provider_name == "http":
        return HttpExtractionProvider()
    if provider_name == "openrouter":
        return OpenRouterExtractionProvider()

    configured = configured_extraction_provider()
    if configured is not None:
        return configured

    # Fallback used in tests / offline mode
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("TERMYTEDB_ALLOW_FAKE_EXTRACTION") == "1":
        return FakeExtractionProvider()

    raise ValueError("no extraction provider configured; set TERMYTEDB_EXTRACTION_URL or OPENROUTER_API_KEY")


def get_summary_provider() -> SessionSummaryProvider:
    """Resolve summary provider; defaults to fake offline."""
    from ..memory.provider import FakeSessionSummaryProvider, OpenRouterSessionSummaryProvider

    # Prefer OpenRouter summary if a key/model is configured, else fake
    if os.environ.get("TERMYTEDB_SUMMARY_MODEL") or os.environ.get("TERMYTEDB_EXTRACTION_MODEL"):
        try:
            return OpenRouterSessionSummaryProvider()
        except ValueError:
            pass
    return FakeSessionSummaryProvider()


def get_embedding_provider(
    kind: str | None = None,
) -> EmbeddingProvider:
    """Resolve embedding provider.

    kind: "fastembed" | "openai" | None (auto based on env)
    """
    from ..retrieval.embedding import FastEmbedProvider, OpenAICompatibleEmbeddingProvider

    if kind == "openai":
        return OpenAICompatibleEmbeddingProvider()
    if kind == "fastembed":
        return FastEmbedProvider()

    # Auto: prefer remote if configured, else local
    if os.environ.get("TERMYTEDB_EMBEDDING_MODEL"):
        try:
            return OpenAICompatibleEmbeddingProvider()
        except ValueError:
            pass
    return FastEmbedProvider()


__all__ = ["get_embedding_provider", "get_extraction_provider", "get_summary_provider"]
