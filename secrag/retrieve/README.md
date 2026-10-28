# `secrag.retrieve` — retrievers

`retriever.retrieve(query, k) -> list[RetrievedChunk]` (rank, score, and component scores for analysis).

| `retrieval.method` | Class | Ranking |
|---|---|---|
| `dense` | `DenseRetriever` | cosine similarity of query and chunk embeddings |
| `sparse` | `SparseRetriever` | BM25 |
| `hybrid` | `HybridRetriever` | Reciprocal Rank Fusion of dense and sparse top-`candidate_k` |

**RRF** (Cormack, Clarke & Buettcher, SIGIR 2009): `score(d) = Σ 1 / (rrf_k + rank_i(d))`, with `rrf_k = 60`.
Ties go to the item with the better best rank, then the lower chunk index, so results are deterministic.

**Reranking (optional):** set `retrieval.reranker` to a cross-encoder name (e.g.
`cross-encoder/ms-marco-MiniLM-L-6-v2`). The base retriever's top `candidate_k` are re-scored on CPU and the
original rank is kept in `scores["base_rank"]`.

Try it:

```
python -m secrag search "What is the refund window?" --method hybrid -k 5
```
