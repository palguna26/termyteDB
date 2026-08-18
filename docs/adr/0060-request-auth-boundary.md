# ADR 0060: Optional request authentication boundary

## Decision

The HTTP application accepts an optional request authorizer callback. A failed
check returns `401` with a request ID before route execution; namespace
authorization remains a separate service-boundary policy.

## Reason

Hosted deployments need an integration point for their identity system without
coupling the engine to a vendor or weakening namespace predicates. Local use
continues to work without an authorizer.
