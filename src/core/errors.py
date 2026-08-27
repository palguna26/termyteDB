"""Core engine errors."""


class IdempotencyConflict(ValueError):
    """The namespace/key pair already represents different content."""
