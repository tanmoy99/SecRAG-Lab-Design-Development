"""Text embedders: Ollama (default), sentence-transformers (alternative), hash (tests).

All embedders share one interface: `embed_documents` and `embed_queries`
return L2-normalised float32 arrays. Asymmetric models need different
prefixes for queries and documents (nomic-embed-text: "search_query: " /
"search_document: "); the prefix is part of the cached text, so changing it
never reuses old vectors.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import httpx
import numpy as np

from secrag.config import EmbedderConfig, Settings
from secrag.index.bm25 import tokenize
from secrag.index.cache import EmbeddingCache
from secrag.index.dense import l2_normalise

# Model-card prefixes, used when the config leaves the prefix as None.
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
}


class Embedder(ABC):
    def __init__(self, query_prefix: str = "", document_prefix: str = "", batch_size: int = 32,
                 cache: EmbeddingCache | None = None):
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.cache = cache
        self.calls = 0  # number of texts actually sent to the model (cache misses)

    @property
    @abstractmethod
    def embedder_id(self) -> str:
        """Identifies everything that changes the vectors, e.g. 'ollama:nomic-embed-text'."""

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed already-prefixed texts; returns shape (len(texts), dim)."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.document_prefix + t for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.query_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        keys = [EmbeddingCache.key(self.embedder_id, t) for t in texts]
        found = self.cache.get_many(keys) if self.cache else {}
        missing = [i for i, k in enumerate(keys) if k not in found]
        new: dict[str, np.ndarray] = {}
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            vectors = l2_normalise(np.asarray(self._embed_batch([texts[i] for i in batch]), dtype=np.float32))
            self.calls += len(batch)
            new.update({keys[i]: v for i, v in zip(batch, vectors)})
        if self.cache and new:
            self.cache.put_many(new)
        found.update(new)
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, 0), dtype=np.float32)


class HashEmbedder(Embedder):
    """Deterministic bag-of-words feature hashing. No model, no network: for tests only.

    Unigrams and bigrams are hashed into `dim` buckets with a sign bit, so texts
    that share words get similar vectors.
    """

    def __init__(self, dim: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    @property
    def embedder_id(self) -> str:
        return f"hash:{self.dim}"

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            for feature in tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]:
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return out


class OllamaEmbedder(Embedder):
    """Embeddings from a local Ollama server via POST /api/embed."""

    def __init__(self, model: str, host: str, timeout: float = 120.0, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.host = host
        self._client = httpx.Client(timeout=timeout)

    @property
    def embedder_id(self) -> str:
        return f"ollama:{self.model}"

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        # truncate=False: an over-long input is an error, never silently cut (cf. decision D3).
        try:
            resp = self._client.post(f"{self.host}/api/embed",
                                     json={"model": self.model, "input": texts, "truncate": False})
        except httpx.ConnectError as exc:
            raise RuntimeError(f"cannot reach Ollama at {self.host}; is `ollama serve` running?") from exc
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama /api/embed failed ({resp.status_code}): {resp.text[:300]}")
        return np.asarray(resp.json()["embeddings"], dtype=np.float32)


class SentenceTransformerEmbedder(Embedder):
    """Hugging Face embedding model on CPU (the GPU is left to Ollama)."""

    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model
        self._model = None  # loaded lazily: importing torch is slow

    @property
    def embedder_id(self) -> str:
        return f"sentence_transformers:{self.model_name}"

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)


def build_embedder(cfg: EmbedderConfig, settings: Settings, use_cache: bool = True) -> Embedder:
    default_q, default_d = KNOWN_PREFIXES.get(cfg.model, ("", ""))
    common = {
        "query_prefix": cfg.query_prefix if cfg.query_prefix is not None else default_q,
        "document_prefix": cfg.document_prefix if cfg.document_prefix is not None else default_d,
        "batch_size": cfg.batch_size,
    }
    if cfg.provider == "hash":
        # The hash embedder is instant, so it is never cached; it also ignores model prefixes.
        return HashEmbedder(dim=cfg.dim, batch_size=cfg.batch_size)
    cache = EmbeddingCache(settings.cache_dir / "embeddings.sqlite") if use_cache else None
    if cfg.provider == "ollama":
        return OllamaEmbedder(cfg.model, settings.ollama_host, cache=cache, **common)
    return SentenceTransformerEmbedder(cfg.model, cache=cache, **common)
