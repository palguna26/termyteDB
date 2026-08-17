# ADR 0043: CLI service configuration

## Decision

The service CLI accepts the database path, optional HTTP extraction endpoint
and model, and per-minute rate limit. The extraction URL and model also accept
the documented environment variables. The default remains offline rule
extraction and no rate limit.

## Reason

Local operation must work without credentials or a network, while deployments
need a clear configuration path for the optional provider and request boundary.
Authorization stays an application-factory callback because identity policy is
deployment-specific.
