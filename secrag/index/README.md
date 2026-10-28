# `secrag.index` — embedders and indexes

| Provider (`embedder.provider`) | Class | Notes |
|---|---|---|
| `ollama` (default) | `OllamaEmbedder` | `POST /api/embed`, `truncate: false` (an over-long input is an error) |
| `sentence_transformers` | `SentenceTransformerEmbedder` | CPU only, so the GPU stays free for Ollama |
| `hash` | `HashEmbedder` | Signed feature hashing of unigrams and bigrams; deterministic, offline, tests only |

- **Prefixes:** asymmetric models need different query and document prefixes. Defaults come from the model card
  (`KNOWN_PREFIXES`: nomic `search_query: ` / `search_document: `; bge `Represent this sentence ...`) unless the
  config sets them.
- **Cache:** `EmbeddingCache` (SQLite, `.cache/embeddings.sqlite`) keys on `embedder_id + prefixed text`, so
  changing the model, the prefix or the text never reuses a stale vector.
- **Dense index:** exact cosine similarity (`vectors @ query`) with NumPy; ties are broken by corpus order.
- **Sparse index:** Okapi BM25 (`rank-bm25`, k1 = 1.5, b = 0.75). The tokenizer lower-cases, keeps emails, codes
  and amounts as single tokens (`privacy@northbridge.example`, `dg101`, `1,285`) and drops a short stopword list.
