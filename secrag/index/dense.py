"""Dense index: exact cosine similarity with NumPy.

With a few thousand chunks exact search is fast (one matrix-vector product)
and fully deterministic, so there is no approximate-search noise in results.
"""

from __future__ import annotations

import numpy as np


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class DenseIndex:
    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("DenseIndex needs a non-empty 2-D array of vectors")
        self.vectors = l2_normalise(vectors.astype(np.float32))

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def scores(self, query_vector: np.ndarray) -> np.ndarray:
        return self.vectors @ l2_normalise(query_vector.astype(np.float32))

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Top-k (index, cosine) pairs; ties are broken by index order."""
        scores = self.scores(query_vector)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]
