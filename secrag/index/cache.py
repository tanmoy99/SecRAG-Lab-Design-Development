"""SQLite cache for embeddings, keyed by embedder identity and exact input text.

Re-running an experiment re-embeds nothing that was embedded before; changing
the model, prefix or text produces a new key, so stale vectors are never reused.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from secrag.types import sha256_text


class EmbeddingCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")

    @staticmethod
    def key(embedder_id: str, text: str) -> str:
        return sha256_text(f"{embedder_id}\n{text}")

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for start in range(0, len(keys), 500):  # stay under SQLite's parameter limit
            batch = keys[start:start + 500]
            rows = self._conn.execute(
                f"SELECT key, vector FROM embeddings WHERE key IN ({','.join('?' * len(batch))})", batch)
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
            [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()])
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
