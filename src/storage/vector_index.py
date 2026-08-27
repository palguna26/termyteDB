"""Optional sqlite-vec index."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from ..retrieval.embedding import EmbeddingProvider, pack_embedding
from .db import Database

INDEX_TABLE = "memory_embedding_index"
META_TABLE = "memory_embedding_index_meta"


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


class SQLiteVecIndex:
    def __init__(self, database: Database, embedding: EmbeddingProvider):
        self.db = database
        self.embedding = embedding
        self.available = bool(getattr(database, "sqlite_vec_available", False))

    def ensure(self) -> bool:
        if not self.available:
            return False
        with self.db.lock, self.db.connection:
            self._ensure_schema()
            self._bootstrap_if_needed()
        return True

    def rebuild_all(self) -> None:
        if not self.available:
            return
        with self.db.lock:
            self._ensure_schema()
            self.db.execute(f"DELETE FROM {INDEX_TABLE}")
            rows = self.db.execute(
                """SELECT memory_version_id, namespace_id, provider, dimensions, vector
                FROM memory_embeddings
                WHERE provider=? AND dimensions=?
                ORDER BY namespace_id, memory_version_id""",
                (self.embedding.name, self.embedding.dimensions),
            ).fetchall()
            for row in rows:
                self.db.execute(
                    f"""INSERT INTO {INDEX_TABLE}
                    (memory_version_id, namespace_id, provider, dimensions, embedding)
                    VALUES (?, ?, ?, ?, ?)""",
                    (
                        row["memory_version_id"],
                        row["namespace_id"],
                        row["provider"],
                        row["dimensions"],
                        bytes(row["vector"]),
                    ),
                )
            self._touch_meta(len(rows))

    def upsert_row(self, memory_version_id: str, namespace_id: str, vector: bytes) -> None:
        if not self.available:
            return
        with self.db.lock:
            self._ensure_schema()
            self.db.execute(
                f"""INSERT OR REPLACE INTO {INDEX_TABLE}
                (memory_version_id, namespace_id, provider, dimensions, embedding)
                VALUES (?, ?, ?, ?, ?)""",
                (memory_version_id, namespace_id, self.embedding.name, self.embedding.dimensions, vector),
            )

    def delete_namespace(self, namespace_id: str) -> None:
        if not self.available:
            return
        with self.db.lock:
            self._ensure_schema()
            self.db.execute(f"DELETE FROM {INDEX_TABLE} WHERE namespace_id=?", (namespace_id,))

    def search(self, namespace_id: str, query_vector: list[float], limit: int) -> list[tuple[str, float]]:
        if not self.available:
            return []
        with self.db.lock:
            self._ensure_schema()
            rows = self.db.execute(
                f"""SELECT memory_version_id, distance
                FROM {INDEX_TABLE}
                WHERE namespace_id=? AND provider=? AND dimensions=? AND embedding MATCH ?
                ORDER BY distance
                LIMIT ?""",
                (
                    namespace_id,
                    self.embedding.name,
                    self.embedding.dimensions,
                    pack_embedding(query_vector),
                    limit,
                ),
            ).fetchall()
        result: list[tuple[str, float]] = []
        for row in rows:
            distance = float(row["distance"])
            score = max(0.0, 1.0 - (distance / 2.0))
            result.append((str(row["memory_version_id"]), score))
        return result

    def _ensure_schema(self) -> None:
        self.db.execute(
            f"""CREATE TABLE IF NOT EXISTS {META_TABLE} (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            provider TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            indexed_rows INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
            )"""
        )
        existing = self.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (INDEX_TABLE,),
        ).fetchone()
        if existing:
            column = self.db.execute(f"PRAGMA table_info({INDEX_TABLE})").fetchall()
            dimension = None
            for row in column:
                if row["name"] == "embedding":
                    match = re.search(r"float\[(\d+)\]", str(row["type"]))
                    if match:
                        dimension = int(match.group(1))
                    break
            if dimension != self.embedding.dimensions:
                self.db.execute(f"DROP TABLE {INDEX_TABLE}")
                existing = None
        if not existing:
            self.db.execute(
                f"""CREATE VIRTUAL TABLE {INDEX_TABLE} USING vec0(
                embedding float[{self.embedding.dimensions}] distance_metric=cosine,
                memory_version_id TEXT,
                namespace_id TEXT,
                provider TEXT,
                dimensions INTEGER
                )"""
            )

    def _bootstrap_if_needed(self) -> None:
        source_count = int(
            self.db.execute(
                "SELECT COUNT(*) FROM memory_embeddings WHERE provider=? AND dimensions=?",
                (self.embedding.name, self.embedding.dimensions),
            ).fetchone()[0]
        )
        indexed_count = int(
            self.db.execute(
                f"SELECT COUNT(*) FROM {INDEX_TABLE} WHERE provider=? AND dimensions=?",
                (self.embedding.name, self.embedding.dimensions),
            ).fetchone()[0]
        )
        meta = self.db.execute(f"SELECT provider, dimensions FROM {META_TABLE} WHERE id=1").fetchone()
        if (
            meta is None
            or str(meta["provider"]) != self.embedding.name
            or int(meta["dimensions"]) != self.embedding.dimensions
            or source_count != indexed_count
        ):
            self.rebuild_all()

    def _touch_meta(self, indexed_rows: int) -> None:
        self.db.execute(
            f"""INSERT INTO {META_TABLE}(id, provider, dimensions, indexed_rows, updated_at)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              provider=excluded.provider,
              dimensions=excluded.dimensions,
              indexed_rows=excluded.indexed_rows,
              updated_at=excluded.updated_at""",
            (self.embedding.name, self.embedding.dimensions, indexed_rows, iso_now()),
        )
