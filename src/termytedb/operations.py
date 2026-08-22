"""Compatibility entry point for local database operations."""

from .runtime.operations import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
