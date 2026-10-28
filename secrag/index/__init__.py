"""Indexes: embedders (+ SQLite cache), exact dense index, BM25 sparse index."""

from secrag.index.bm25 import BM25Index, tokenize
from secrag.index.cache import EmbeddingCache
from secrag.index.dense import DenseIndex, l2_normalise
from secrag.index.embedders import (
    Embedder,
    HashEmbedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
)

__all__ = [
    "BM25Index", "DenseIndex", "Embedder", "EmbeddingCache", "HashEmbedder", "OllamaEmbedder",
    "SentenceTransformerEmbedder", "build_embedder", "l2_normalise", "tokenize",
]
