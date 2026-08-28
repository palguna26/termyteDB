"""TermyteDB centralized configuration.

Import configurables from here:

    from src.config import EMBEDDING, EXTRACTION, RETRIEVAL
    from src.config.prompts import build_extraction_prompt
    from src.config.providers import get_extraction_provider

See `settings.py`, `prompts.py`, `providers.py`, `embeddings.py` for details.
"""

from .embeddings import (
    FASTEMBED_BATCH_SIZE,
    FASTEMBED_DIMENSIONS,
    FASTEMBED_MODEL,
    OPENAI_DEFAULT_DIMENSIONS,
)
from .prompts import (
    build_extraction_prompt,
    build_session_summary_prompt,
    clean_json_response,
    extraction_response_format,
)
from .providers import get_embedding_provider, get_extraction_provider, get_summary_provider
from .settings import EMBEDDING, EXTRACTION, MEMORY, RETRIEVAL

__all__ = [
    "EMBEDDING",
    "EXTRACTION",
    "FASTEMBED_BATCH_SIZE",
    "FASTEMBED_DIMENSIONS",
    "FASTEMBED_MODEL",
    "MEMORY",
    "OPENAI_DEFAULT_DIMENSIONS",
    "RETRIEVAL",
    "build_extraction_prompt",
    "build_session_summary_prompt",
    "clean_json_response",
    "extraction_response_format",
    "get_embedding_provider",
    "get_extraction_provider",
    "get_summary_provider",
]
