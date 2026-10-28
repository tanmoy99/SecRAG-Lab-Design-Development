import numpy as np
import pytest

from secrag.config import EmbedderConfig, ExperimentConfig, Settings
from secrag.index import BM25Index, DenseIndex, EmbeddingCache, HashEmbedder, build_embedder, tokenize
from secrag.retrieve import DenseRetriever, HybridRetriever, SparseRetriever, build_retriever, rrf_fuse


def test_tokenizer_keeps_emails_codes_and_amounts():
    toks = tokenize("Email privacy@northbridge.example about DG101: it costs $1,285.")
    assert "privacy@northbridge.example" in toks and "dg101" in toks and "1,285" in toks
    assert "about" in toks and "it" not in toks


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(dim=64)
    a = e.embed_documents(["refund policy for courses", "something else"])
    b = HashEmbedder(dim=64).embed_documents(["refund policy for courses", "something else"])
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_dense_index_exact_cosine_and_stable_ties():
    vecs = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    hits = DenseIndex(vecs).search(np.array([2.0, 0.0]), 3)
    assert [i for i, _ in hits] == [0, 2, 1]
    assert hits[0][1] == pytest.approx(1.0)


def test_bm25_ranks_matching_text_first():
    idx = BM25Index(["refund window policy", "campus parking", "late payment fee"])
    assert idx.search("what is the late fee", 1)[0][0] == 2
    assert idx.search("the of", 2) == [(0, 0.0), (1, 0.0)]  # stopword-only query


def test_embedding_cache_avoids_recomputation(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    first = HashEmbedder(dim=32, cache=cache)
    v1 = first.embed_documents(["alpha beta", "gamma"])
    assert first.calls == 2
    second = HashEmbedder(dim=32, cache=cache)
    v2 = second.embed_documents(["gamma", "alpha beta", "delta"])
    assert second.calls == 1  # only "delta" is new
    assert np.allclose(v2[0], v1[1]) and np.allclose(v2[1], v1[0])


def test_prefixes_change_the_cache_key(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    e = HashEmbedder(dim=32, cache=cache, query_prefix="q: ", document_prefix="d: ")
    e.embed_documents(["same text"])
    e.embed_queries(["same text"])
    assert e.calls == 2


def test_build_embedder_uses_model_card_prefixes():
    settings = Settings(ollama_host="http://127.0.0.1:1", cache_dir=None)  # type: ignore[arg-type]
    e = build_embedder(EmbedderConfig(provider="ollama", model="nomic-embed-text"), settings, use_cache=False)
    assert (e.query_prefix, e.document_prefix) == ("search_query: ", "search_document: ")
    e = build_embedder(EmbedderConfig(provider="ollama", model="nomic-embed-text", query_prefix=""), settings,
                       use_cache=False)
    assert e.query_prefix == ""


def test_rrf_matches_hand_computation():
    fused = dict(rrf_fuse([[1, 2, 3], [3, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)
    assert [i for i, _ in rrf_fuse([[1, 2, 3], [3, 1]], k=60)] == [1, 3, 2]


def test_rrf_tie_break_is_deterministic():
    assert [i for i, _ in rrf_fuse([[5, 6], [6, 5]])] == [5, 6]


@pytest.mark.parametrize("method", ["dense", "sparse", "hybrid"])
def test_retrievers_return_ranked_chunks(chunks, method):
    cfg = ExperimentConfig.model_validate({"name": "t", "embedder": {"provider": "hash"},
                                           "retrieval": {"method": method}})
    retriever = build_retriever(cfg, chunks, Settings("http://127.0.0.1:1", None))  # type: ignore[arg-type]
    results = retriever.retrieve("What is the refund window for courses?", 7)
    assert [r.rank for r in results] == list(range(1, 8))
    assert len({r.chunk.chunk_id for r in results}) == 7
    assert all(results[i].score >= results[i + 1].score for i in range(6))
    # The stale forum post about refunds is a legitimate (and realistic) competitor for rank 1.
    relevant = {"policy-refunds", "news-policy-review-2026", "faq-enrolment", "forum-stale-0"}
    assert relevant & {r.chunk.doc_id for r in results[:3]}


def test_hybrid_reports_component_scores(chunks):
    dense = DenseRetriever(chunks, HashEmbedder(dim=256))
    hybrid = HybridRetriever(dense, SparseRetriever(chunks), candidate_k=20)
    top = hybrid.retrieve("Privacy Officer email", 3)[0]
    assert "rrf" in top.scores and ("dense" in top.scores or "bm25" in top.scores)


def test_hybrid_requires_same_chunks(chunks):
    with pytest.raises(ValueError):
        HybridRetriever(DenseRetriever(chunks, HashEmbedder(dim=16)), SparseRetriever(list(chunks)))
