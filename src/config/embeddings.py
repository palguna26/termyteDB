"""Embedding size / model configuration.

Every place that needs a dimension, model name, or batch size should import
from here. Environment overrides remain supported for 12-factor deploys.
"""

from __future__ import annotations

from .settings import EMBEDDING

# Re-export for callers that prefer flat imports
FASTEMBED_MODEL = EMBEDDING.fastembed_model
FASTEMBED_DIMENSIONS = EMBEDDING.fastembed_dimensions
FASTEMBED_BATCH_SIZE = EMBEDDING.fastembed_batch_size

OPENAI_DEFAULT_MODEL = EMBEDDING.openai_default_model
OPENAI_DEFAULT_DIMENSIONS = EMBEDDING.openai_default_dimensions
OPENAI_DEFAULT_BASE_URL = EMBEDDING.openai_default_base_url

VEC_DISTANCE_METRIC = EMBEDDING.vec_distance_metric

__all__ = [
    "FASTEMBED_BATCH_SIZE",
    "FASTEMBED_DIMENSIONS",
    "FASTEMBED_MODEL",
    "OPENAI_DEFAULT_BASE_URL",
    "OPENAI_DEFAULT_DIMENSIONS",
    "OPENAI_DEFAULT_MODEL",
    "VEC_DISTANCE_METRIC",
]
