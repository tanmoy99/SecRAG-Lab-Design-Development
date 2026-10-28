# SecRAG Lab

A local, reproducible testbed for the security of Retrieval-Augmented Generation (RAG) in small
organisations running small local LLMs: poisoning (RQ1), indirect prompt injection (RQ2) and lightweight
defences (RQ3). Design and decision log: [docs/DESIGN.md](docs/DESIGN.md).

> Work in progress (Phase 1 of 5). The full README comes in Phase 5.

## Setup (Windows, Python 3.12)

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
ollama pull nomic-embed-text qwen3:1.7b qwen3:4b qwen3:8b llama3.2:3b
.venv\Scripts\python -m secrag check-ollama
```

## Phase 1 commands

```
python -m secrag generate-data                 # synthetic corpus + QA set (seed 13; already committed in data/)
python -m secrag corpus-stats                  # document / chunk counts
python -m secrag search "What is the refund window?" --method hybrid
python -m secrag eval-retrieval --config configs/retrieval_smoke.yaml      # offline, seconds
python -m secrag eval-retrieval --config configs/retrieval_baseline.yaml   # Ollama + bge-small
python -m pytest                               # tests; Ollama tests auto-skip if the server is down
```

## Layout

```
secrag/data_gen   synthetic Northbridge corpus, facts, QA           (README inside)
secrag/ingest     front-matter markdown loader, paragraph chunker
secrag/index      embedders + SQLite cache, dense (NumPy), BM25
secrag/retrieve   dense / sparse / hybrid (RRF) / rerank
secrag/eval       answer matching, retrieval metrics, bootstrap CIs, results folders
configs/          experiment YAMLs (with grid: expansion)
data/             generated corpus and evaluator-only QA files
results/          one timestamped folder per experiment run (git-ignored)
```
