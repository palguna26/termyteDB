# Direct Benchmark Workflow Design

## Goal

Make the LongMemEval end-to-end benchmark describe and measure TermyteDB's direct synchronous ingestion workflow.

## Changes

- Rename `ingest_and_process_e2e()` to `ingest_e2e()`.
- Keep one `ingest_batch()` call per conversation session.
- Remove queue, processing-job, lease, and drain-loop language.
- Remove duplicate processing metrics that only repeat ingestion results.
- Use returned accepted and rejected candidate counts as final memory-formation results.
- Update comments, trace fields, and documentation to use direct-ingestion terms.

## Unchanged

- Dataset normalization and leakage boundaries;
- extraction and embedding providers;
- retrieval, reranking, context packing, and scoring;
- concurrency across independent benchmark samples;
- retrieval-only and judged modes.

## Validation

- Run the full test suite.
- Run Mypy and Ruff.
- Check the benchmark CLI.
- Run a small offline benchmark smoke path where possible.
