import importlib.util

import pytest

from termytedb import TermyteDB
from termytedb.retrieval.embedding import batch_dot, pack_embedding


class ConstantEmbedding:
    name = "test-constant-v1"
    dimensions = 2

    def embed(self, value: str) -> list[float]:
        return [1.0, 0.0] if "target" in value else [0.0, 1.0]


def test_embedding_provider_is_injectable_and_persisted(tmp_path):
    db = TermyteDB(tmp_path / "embedding.sqlite", embedding_provider=ConstantEmbedding())
    db.ingest({"namespace_id": "embedding", "idempotency_key": "one", "type": "note", "payload": {"text": "Decision: target."}})
    db.process("embedding")
    row = db.database.execute("SELECT provider, dimensions FROM memory_embeddings WHERE namespace_id='embedding'").fetchone()
    assert tuple(row) == ("test-constant-v1", 2)
    columns = {item["name"]: item["type"] for item in db.database.execute("PRAGMA table_info(memory_embeddings)")}
    assert columns["vector"] == "BLOB"
    assert "vector_json" not in columns
    vector = db.database.execute("SELECT vector FROM memory_embeddings WHERE namespace_id='embedding'").fetchone()[0]
    assert isinstance(vector, bytes)
    assert len(vector) == 2 * 4
    assert db.search("embedding", "target")
    db.close()


def test_binary_embedding_batch_dot_product():
    vectors = [pack_embedding([1.0, 0.0]), pack_embedding([0.0, 1.0])]
    scores = batch_dot([1.0, 0.0], vectors, 2)
    assert scores.tolist() == [1.0, 0.0]


@pytest.mark.skipif(importlib.util.find_spec("sqlite_vec") is None, reason="sqlite-vec is not installed")
def test_sqlite_vec_index_bootstraps_when_available(tmp_path):
    db = TermyteDB(tmp_path / "embedding-index.sqlite", embedding_provider=ConstantEmbedding())
    db.ingest({"namespace_id": "embedding", "idempotency_key": "one", "type": "note", "payload": {"text": "Decision: target."}})
    db.process("embedding")
    row = db.database.execute("SELECT COUNT(*) FROM memory_embedding_index WHERE namespace_id=?", ("embedding",)).fetchone()
    assert int(row[0]) == 1
    table = db.database.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embedding_index'"
    ).fetchone()
    assert table is not None
    db.close()
