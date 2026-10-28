# `secrag.ingest` — loading and chunking

**Loader** (`loader.py`): one markdown file per document with YAML front-matter
(`doc_id, title, source, author, trust_level, created_at, doc_type`). Files are loaded in `doc_id` order;
duplicate ids are an error. Front-matter keys that would leak ground truth (`is_poisoned`, `attack_id`,
`target_answer`, ...) are refused (decision D12).

**Chunker** (`chunking.py`): paragraph-aware, sizes in words (config `chunking.chunk_size`, `chunking.overlap`).

1. Split at blank lines; cut any paragraph longer than `chunk_size` into `chunk_size`-word pieces.
2. Pack whole paragraphs until the next would exceed `chunk_size` words of new content.
3. Start each later chunk with the last `overlap` words of the previous one.

Chunk ids are `<doc_id>#<position>`. Each chunk carries its document's metadata (title, source, trust level,
content hash) for citations and for provenance and integrity defences. The text that gets embedded and
BM25-indexed is `title + "\n\n" + chunk text` (`Chunk.index_text`).

With the defaults (120 / 20) most Northbridge pages are one chunk and course pages are two;
`python -m secrag corpus-stats` prints the distribution.
