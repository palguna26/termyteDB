from termytedb import TermyteDB


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
    assert db.search("embedding", "target")
    db.close()
