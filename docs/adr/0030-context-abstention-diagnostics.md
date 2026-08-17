# ADR 0030: Context abstention and selection diagnostics

## Status

Accepted

## Decision

Context construction filters candidates below a 0.05 combined-score floor, deduplicates statements, enforces the token budget, and returns bounded diagnostics containing candidate count, score-filter count, selected count, token budget, and exclusion reasons. An irrelevant or over-budget request can therefore return no context with an inspectable reason.

## Consequences

Callers can distinguish empty retrieval from budget exclusion and duplicate suppression. The score floor is a starting deterministic threshold and must be calibrated against labelled abstention data before quality claims.
