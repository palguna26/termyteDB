from __future__ import annotations

from pathlib import Path

import pytest

from src import TermyteDB


@pytest.fixture
def db(tmp_path: Path):
    instance = TermyteDB(tmp_path / "termytedb.sqlite")
    yield instance
    instance.close()


def event(namespace: str, key: str, text: str, event_type: str = "conversation") -> dict[str, object]:
    return {
        "namespace_id": namespace,
        "idempotency_key": key,
        "type": event_type,
        "payload": {"text": text},
    }
