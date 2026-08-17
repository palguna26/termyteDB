import argparse
import os

import uvicorn

from .provider import HttpExtractionProvider
from .service import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rate-limit-per-minute", type=int, default=None)
    parser.add_argument("--extraction-url", default=None)
    parser.add_argument("--extraction-model", default=None)
    args = parser.parse_args()
    endpoint = args.extraction_url or os.environ.get("TERMYTEDB_EXTRACTION_URL")
    provider = HttpExtractionProvider(endpoint, args.extraction_model) if endpoint else None
    uvicorn.run(
        create_app(args.database, extraction_provider=provider, rate_limit_per_minute=args.rate_limit_per_minute),
        host=args.host,
        port=args.port,
    )
