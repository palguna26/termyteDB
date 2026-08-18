# ADR 0063: Retry classification

## Decision

Provider errors marked non-retryable dead-letter their job immediately.
Retryable provider errors use the persisted exponential backoff schedule;
ordinary worker exceptions remain retryable by default.

## Reason

Malformed model output cannot be repaired by repeating the same request, while
transport and timeout failures may recover. The provider boundary already
classifies these cases, so the worker must honor that classification.
