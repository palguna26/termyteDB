# ADR 0028: HTTP namespace authorization boundary

## Status

Accepted

## Decision

`create_app` accepts an optional namespace authorizer callback. Every namespace-bearing HTTP route calls it before repository access, including batch events, inspection, export/import, feedback, and deletion. Denied requests return 403 without checking whether the requested ID exists. The embedded engine remains independently namespace-scoped and does not depend on this callback.

## Consequences

Hosted deployments can attach authentication and policy without adding vendor-specific auth code to the engine. Local deterministic operation can omit the callback and still retains storage-level isolation.
