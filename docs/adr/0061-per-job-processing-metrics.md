# ADR 0061: Per-job extraction metrics

## Decision

Extraction-run accepted and rejected counts describe only that run. The
namespace processing response keeps separate aggregate counters.

## Reason

Mixing the counters made later runs appear to include earlier candidates and
made audit metrics inaccurate. Separate counters preserve both per-run
diagnostics and aggregate worker results.
