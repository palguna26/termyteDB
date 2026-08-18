# ADR 0057: Configurable provider cost estimates

## Decision

Persist an estimated extraction cost only when both input and output
price-per-1,000-token environment values are configured. Otherwise the value
remains `NULL`; no provider price is assumed.

## Reason

Token counts alone do not provide a monetary estimate, and provider pricing
changes. A small explicit configuration boundary gives operators useful cost
telemetry without presenting invented prices as measured results.
