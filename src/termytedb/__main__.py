import argparse
import os

import uvicorn

from .api.service import create_app
from .memory.provider import HttpExtractionProvider, OpenRouterExtractionProvider
from .retrieval.embedding import OpenAICompatibleEmbeddingProvider

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rate-limit-per-minute", type=int, default=None)
    parser.add_argument("--extraction-url", default=None)
    parser.add_argument("--extraction-model", default=None)
    parser.add_argument("--embedding-provider", choices=("local", "openrouter"), default="openrouter")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dimensions", type=int, default=None)
    args = parser.parse_args()
    endpoint = args.extraction_url or os.environ.get("TERMYTEDB_EXTRACTION_URL")
    if endpoint:
        provider = HttpExtractionProvider(endpoint, args.extraction_model)
    else:
        provider = OpenRouterExtractionProvider(args.extraction_model)
    embedding = None
    embedding_provider = args.embedding_provider or os.environ.get("TERMYTEDB_EMBEDDING_PROVIDER", "openrouter")
    if embedding_provider == "openrouter":
        embedding = OpenAICompatibleEmbeddingProvider(args.embedding_model, dimensions=args.embedding_dimensions)
    uvicorn.run(
        create_app(args.database, extraction_provider=provider, embedding_provider=embedding, rate_limit_per_minute=args.rate_limit_per_minute),
        host=args.host,
        port=args.port,
    )
