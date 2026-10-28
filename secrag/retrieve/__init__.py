"""Retrievers: dense, sparse (BM25), hybrid (Reciprocal Rank Fusion), optional reranking.

Every retriever maps a query string to a ranked list of `RetrievedChunk`s.
Rankings are deterministic: ties are broken by chunk order in the corpus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from secrag.config import ExperimentConfig, Settings
from secrag.index import BM25Index, DenseIndex, Embedder, build_embedder
from secrag.types import Chunk, RetrievedChunk


class Retriever(ABC):
    name: str

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("cannot retrieve from an empty corpus")
        self.chunks = chunks

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...

    def _results(self, scored: list[tuple[int, float]], key: str) -> list[RetrievedChunk]:
        return [RetrievedChunk(self.chunks[i], rank, score, {key: score})
                for rank, (i, score) in enumerate(scored, 1)]


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        super().__init__(chunks)
        self.embedder = embedder
        self.index = DenseIndex(embedder.embed_documents([c.index_text for c in chunks]))

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._results(self.index.search(self.embedder.embed_query(query), k), "dense")


class SparseRetriever(Retriever):
    name = "sparse"

    def __init__(self, chunks: list[Chunk]):
        super().__init__(chunks)
        self.index = BM25Index([c.index_text for c in chunks])

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._results(self.index.search(query, k), "bm25")


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    score(d) = sum over rankings of 1 / (k + rank_of_d), rank starting at 1.
    Ties are broken by the item's best rank in any list, then by item id.
    """
    scores: dict[int, float] = {}
    best: dict[int, int] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best[item] = min(best.get(item, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))


class HybridRetriever(Retriever):
    """Fuses the top `candidate_k` of a dense and a sparse retriever with RRF."""

    name = "hybrid"

    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever, candidate_k: int = 50, rrf_k: int = 60):
        super().__init__(dense.chunks)
        if sparse.chunks is not dense.chunks:
            raise ValueError("dense and sparse retrievers must index the same chunk list")
        self.dense, self.sparse = dense, sparse
        self.candidate_k, self.rrf_k = candidate_k, rrf_k

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        depth = max(k, self.candidate_k)
        dense_hits = self.dense.index.search(self.dense.embedder.embed_query(query), depth)
        sparse_hits = self.sparse.index.search(query, depth)
        dense_score, sparse_score = dict(dense_hits), dict(sparse_hits)
        fused = rrf_fuse([[i for i, _ in dense_hits], [i for i, _ in sparse_hits]], self.rrf_k)[:k]
        out = []
        for rank, (i, score) in enumerate(fused, 1):
            parts = {"rrf": score}
            if i in dense_score:
                parts["dense"] = dense_score[i]
            if i in sparse_score:
                parts["bm25"] = sparse_score[i]
            out.append(RetrievedChunk(self.chunks[i], rank, score, parts))
        return out


class RerankingRetriever(Retriever):
    """Re-scores the base retriever's top `candidate_k` with a cross-encoder (CPU)."""

    def __init__(self, base: Retriever, model: str, candidate_k: int = 50):
        super().__init__(base.chunks)
        self.base, self.model_name, self.candidate_k = base, model, candidate_k
        self.name = f"{base.name}+rerank"
        self._model = None

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        candidates = self.base.retrieve(query, max(k, self.candidate_k))
        scores = self._model.predict([(query, rc.chunk.index_text) for rc in candidates], show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda j: (-float(scores[j]), j))[:k]
        return [RetrievedChunk(candidates[j].chunk, rank, float(scores[j]),
                               {**candidates[j].scores, "rerank": float(scores[j]), "base_rank": candidates[j].rank})
                for rank, j in enumerate(order, 1)]


def build_retriever(cfg: ExperimentConfig, chunks: list[Chunk], settings: Settings,
                    embedder: Embedder | None = None) -> Retriever:
    """Build the retriever an experiment config asks for. Pass `embedder` to reuse one."""
    r = cfg.retrieval
    retriever: Retriever
    if r.method == "sparse":
        retriever = SparseRetriever(chunks)
    else:
        dense = DenseRetriever(chunks, embedder or build_embedder(cfg.embedder, settings))
        retriever = dense if r.method == "dense" else HybridRetriever(dense, SparseRetriever(chunks),
                                                                      r.candidate_k, r.rrf_k)
    if r.reranker:
        retriever = RerankingRetriever(retriever, r.reranker, r.candidate_k)
    return retriever


__all__ = [
    "DenseRetriever", "HybridRetriever", "RerankingRetriever", "Retriever", "SparseRetriever",
    "build_retriever", "rrf_fuse",
]
