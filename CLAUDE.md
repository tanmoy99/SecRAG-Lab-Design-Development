# SecRAG Lab — notes for Claude Code

PhD research proposal (Australia) testbed for RAG security in small organisations running small local LLMs.
RQ1 poisoning, RQ2 indirect prompt injection, RQ3 lightweight defences (cost in accuracy + latency).
Research code a supervisor will read: clarity and reproducibility over cleverness.

## How we work
- Build in 5 phases. STOP after each phase: show what was built, how to run it, test results; wait for go-ahead.
  1. Skeleton, config, synthetic corpus + QA generator, ingestion, chunking, dense/BM25/hybrid retrieval, recall@k. Tests.
  2. Ollama + MockLLM generation, pipeline with citations, clean-accuracy eval. Tests.
  3. Poisoning / indirect injection / insider edit attacks, ASR + retrieval-poisoning metrics, ASR vs N experiment.
  4. Five defences (+ optional integrity check), adaptive attacks, experiment grid, plots.
  5. Streamlit demo, README, THREAT_MODEL.md, RESULTS.md.
- Living design doc (decision log D1–D17, phase tracker, architecture): https://claude.ai/artifact/8mxu73qeFp4WZ1xCygpt3j
  Update its Decision log and Phase tracker as work progresses, and keep the local copy `docs/DESIGN.md` in sync.

## Agreed decisions (details in the design doc)
- Models: qwen3:1.7b, qwen3:4b, qwen3:8b + llama3.2:3b. Qwen3 thinking OFF (`think: false`).
- Set Ollama `num_ctx` explicitly; error if a prompt would be truncated.
- Embedder default nomic-embed-text via Ollama (query/document prefixes); bge-small via sentence-transformers as alt; `hash` mock embedder for tests.
- Dense index = NumPy exact cosine; BM25 = rank-bm25; hybrid = own RRF (k=60); optional cross-encoder rerank. CPU torch (GPU left to Ollama).
- pydantic v2 + YAML configs with `grid:` expansion; argparse CLI `python -m secrag <cmd>`; SQLite response/embedding caches.
- Corpus: template-generated fictional "Northbridge Institute of Data Practice", fixed seed, ~60–100 core docs + ~150 distractors (incl. a few unverified forum posts with stale values); emails on `.example` domains. QA 80–120 items incl. ~15 unanswerable, facts with aliases + normalisation.
- Evaluator-only labels (is_poisoned) never on pipeline objects. Injected-doc trust level is an attack parameter.
- Uncertainty: bootstrap 95% CIs over questions; repetitions only for temperature > 0.
- Reduced defence grid: none, each alone, all on.
- Streamlit for the demo.

## Status (2026-10-28, Phase 1 built, awaiting review)
Environment verified (venv complete, all 5 Ollama models pulled). Phase 1 code, data, configs, tests (85 pass)
and module READMEs are written; baseline results in results/ and in the Phase 1 report (docs/DESIGN.md).
Nothing committed to git yet. Next after go-ahead: Phase 2 (LLMClient/OllamaClient/MockLLM, pipeline with
citations, clean-accuracy eval).

## Environment
Windows 11, Python 3.12 (`py -3.12`), venv at `.venv`, Ryzen 7 5800HS, 15 GB RAM, RTX 3060 Laptop 6 GB.
Ollama 0.32.9 at http://127.0.0.1:11434.
