# ADR 0022: Generic configurable extraction provider

## Status

Accepted

## Decision

`HttpExtractionProvider` is the optional real-provider boundary. It sends a versioned prompt and schema name to a configured HTTP JSON endpoint, validates the response as `extraction-v1`, records hashes and latency, supports cancellation checks, and classifies transport, HTTP, and invalid-output failures. The endpoint and model are configured through constructor arguments or `TERMYTEDB_EXTRACTION_URL` and `TERMYTEDB_EXTRACTION_MODEL`; an optional bearer credential comes from `TERMYTEDB_EXTRACTION_API_KEY`.

No provider-specific SDK or vendor response format is implemented. Invalid response bodies are not included in errors or logs.
