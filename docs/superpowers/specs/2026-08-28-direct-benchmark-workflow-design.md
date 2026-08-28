# Reliable Direct Benchmark Workflow Design

## Goal

Make the LongMemEval end-to-end benchmark reliably measure TermyteDB's direct synchronous ingestion workflow without flooding OpenRouter or reporting invalid scores.

## Changes

- Rename `ingest_and_process_e2e()` to `ingest_e2e()`.
- Keep one `ingest_batch()` call per conversation session.
- Remove queue, processing-job, lease, and drain-loop language.
- Remove duplicate processing metrics that only repeat ingestion results.
- Use returned accepted and rejected candidate counts as final memory-formation results.
- Update comments, trace fields, and documentation to use direct-ingestion terms.

## Run Isolation

- A run uses a new timestamped work directory by default.
- Existing databases are reused only through an explicit resume workflow.
- Duplicate events from an older run must not silently turn extraction into a no-op.

## OpenRouter Traffic Control

- All OpenRouter extraction and embedding calls share one process-wide rate limiter.
- Requests are spaced by a configurable minimum interval instead of being sent in bursts.
- End-to-end benchmark concurrency defaults to one sample at a time.
- Retryable 408, 429, and 5xx responses use bounded exponential backoff with jitter.
- Provider responses may supply a `Retry-After` delay, which takes priority when longer.
- Retry attempts and rate-limit waits are visible in benchmark output.

## Failure Reporting

- Every sample produces either a successful trace or a structured failure trace.
- Reports include total, completed, and failed sample counts.
- Score tables use completed samples only and clearly state that scope.
- A run with zero completed samples reports scores as unavailable, not zero.
- The command exits non-zero if any sample fails.

## Unchanged

- Dataset normalization and leakage boundaries;
- extraction and embedding providers;
- retrieval, reranking, context packing, and scoring;
- retrieval-only and judged modes.

## Validation

- Run the full test suite.
- Run Mypy and Ruff.
- Check the benchmark CLI.
- Test fresh run directories and explicit resume behavior.
- Test global request spacing and bounded retry behavior.
- Test partial and total failure reports and exit codes.
- Run the 30-sample micro benchmark after local validation passes.
