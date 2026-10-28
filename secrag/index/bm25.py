"""Sparse (lexical) index: Okapi BM25 via `rank-bm25`."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

# Keeps emails, decimals and codes together: "privacy@northbridge.example", "1,285", "dg101".
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,@'_-][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its me my "
    "of on or our that the this to was we what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    """BM25 over pre-tokenised documents. Ties are broken by document order."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        if not texts:
            raise ValueError("cannot build a BM25 index over zero documents")
        self._bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)
        self.size = len(texts)

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(self.size)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.scores(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]
