# SecRAG Lab — Design & Development Log

Local copy of the living design doc (https://claude.ai/artifact/8mxu73qeFp4WZ1xCygpt3j), last synced 2026-10-28.
The online doc stays the master; this file is updated whenever the online doc changes.

## Purpose and research questions

SecRAG Lab is a local testbed for measuring how easily a small organisation's RAG assistant can be poisoned or
hijacked, and which laptop-affordable defences help. It supports a PhD research proposal (Australia) on RAG
security for small organisations running small local LLMs.

It is two things at once: a believable RAG assistant for a fictional professional institute, and a reproducible
experiment platform that produces the numbers and plots for the proposal.

- **RQ1 — Poisoning:** how many malicious documents does an attacker need to inject to change answers to targeted
  questions, and how does this vary with model size and retrieval method?
- **RQ2 — Indirect prompt injection:** how often do instructions hidden in retrieved documents override the system prompt?
- **RQ3 — Defences:** which lightweight defences reduce attack success, and what do they cost in clean accuracy and latency?

All attacks run only against this local sandbox, on synthetic or public research data.

## How we work

We build in five phases and stop after each one for review: Claude shows what was built, how to run it and the
test results, and waits for a go-ahead.

1. Skeleton, config, synthetic corpus + QA set, ingestion, chunking, dense / BM25 / hybrid retrieval, recall@k. Tests.
2. Generation (Ollama + MockLLM), full pipeline with citations, clean-accuracy evaluation. Tests.
3. Poisoning, indirect-injection and insider-edit attacks; ASR and retrieval-poisoning metrics; first ASR-vs-N experiment.
4. Five defences (plus optional integrity check), adaptive attacks, the experiment grid and plots.
5. Demo web app, README, THREAT_MODEL.md, RESULTS.md.

Working rules:

- **The design doc is the design log.** Every decision lands in the Decision log with its reason; the Phase tracker
  is updated at the end of each phase.
- **Reproducibility first:** fixed seeds, pinned dependencies, committed generated data, and every result folder
  stores its config copy and git commit hash.
- **Clarity over cleverness:** a supervisor will read the code, so type hints, docstrings and a short README per module.
- **Fast tests without Ollama:** the core pytest suite uses a mock LLM and mock embedder; Ollama tests are marked and auto-skipped.
- Code lives in `C:\dev\secrag-lab` (git repo, outside OneDrive).

## Environment

Everything runs offline on one Windows 11 laptop. Ollama uses the GPU for the LLMs; everything else (embeddings
cache, reranker, NLI) stays on CPU to leave VRAM free.

| Item | Value |
|---|---|
| CPU / RAM | AMD Ryzen 7 5800HS, 15 GB |
| GPU | NVIDIA RTX 3060 Laptop, 6 GB VRAM |
| Python | 3.12.10, virtual environment in `.venv` |
| Ollama | 0.32.9 at http://127.0.0.1:11434 |
| Chat models | qwen3:1.7b, qwen3:4b, qwen3:8b (same-family size sweep), llama3.2:3b (cross-family point) |
| Embedding model | nomic-embed-text (Ollama); bge-small-en-v1.5 as alternative |
| Repo | `C:\dev\secrag-lab`, git, branch `main` |

## Decision log

All 17 design changes proposed before Phase 1 were accepted on 28 Oct 2026. New decisions are added at the top.

| # | Date | Decision | Why |
|---|---|---|---|
| D17 | 2026-10-28 | Repo is a git repo; each result folder records the commit hash and dirty flag | Every number in the proposal traces back to exact code |
| D16 | 2026-10-28 | Repo moved to `C:\dev\secrag-lab`, outside OneDrive | OneDrive sync locks venv and cache files and uploads thousands of files |
| D15 | 2026-10-28 | Retrieval-optimised poisoning follows PoisonedRAG's black-box construction (Zou et al., USENIX Security 2025); spotlighting follows Hines et al. 2024 | Results on the public benchmark become comparable with published work |
| D14 | 2026-10-28 | Corpus is template-generated with a fixed seed, varied phrasing, plus ~150 distractor documents | Templates guarantee consistent facts; distractors stop top-k retrieval being trivially easy to poison; the fictional institute means models have no prior knowledge of the facts |
| D13 | 2026-10-28 | Reduced grid: no defence, each defence alone, all on (7 settings); ~20 poisoning targets; SQLite response cache; runtime estimate printed before each run | Full factorial is about 150k LLM calls, weeks on this laptop |
| D12 | 2026-10-28 | Evaluator-only fields (e.g. `is_poisoned`) live in a separate record that is never passed to the pipeline; a test enforces it | Prevents label leakage into the system under test |
| D11 | 2026-10-28 | Fixed refusal sentence in the system prompt; refusal detection matches it plus a short list of variants | Phrase matching on free-form refusals is unreliable |
| D10 | 2026-10-28 | False target values are non-round (e.g. 517); ASR reported strict (target present, true value absent) and lenient (target present) | Targets cannot appear by chance; hedged answers are visible |
| D9 | 2026-10-28 | Fact matching normalises numbers, currencies, emails, dates and phones; facts store aliases | `AUD 450`, `A$450`, `$450.00` must all count as the same answer |
| D8 | 2026-10-28 | Faithfulness metric and answer-verification defence use different methods/models (NLI for the defence) | Otherwise the defence grades itself; a same-model judge is also injectable |
| D7 | 2026-10-28 | Each relevant defence is also tested against an adaptive attack in Phase 4 | Fixed-attack-only evaluation overstates defences; reviewers will ask |
| D6 | 2026-10-28 | Insider-edit attack is expected to beat all five defences; optional sixth defence = content-hash integrity check against a signed manifest | Consistency checks see no conflict after an in-place edit; this is a finding, not a bug |
| D5 | 2026-10-28 | Trust level of injected documents is an attack parameter tied to an entry point (public upload = unverified, compromised staff account = staff, insider edit = official) | Otherwise provenance filtering wins by construction and RQ3 is circular |
| D4 | 2026-10-28 | Main uncertainty measure = bootstrap 95% CIs over questions/targets; repetitions only in a temperature > 0 condition | At temperature 0, std over repetitions is ~0 and meaningless |
| D3 | 2026-10-28 | `num_ctx` set explicitly; a prompt longer than the context window is an error | Ollama silently drops the start of the prompt, i.e. the system prompt, inflating injection success |
| D2 | 2026-10-28 | Qwen3 thinking mode disabled (`think: false`) | Reasoning traces add 5–20× latency and can contain canary strings that would be miscounted as attack success |
| D1 | 2026-10-28 | Model set: qwen3:1.7b, 4b, 8b + llama3.2:3b | Two models alone confound size with model family |

## Architecture and library choices

Every stage is swappable through YAML config, and every defence switches on or off on its own.

```
 clean corpus (markdown + front-matter)      attack files (kept separate)
                 \                               /
                  ingest -> chunk -> index (dense / BM25)
                                   |
             query -> retrieve (dense | sparse | hybrid RRF) [-> rerank]
                                   |
        pre-generation defences (4, act on retrieved chunks)
                                   |
                          LLM (Ollama / MockLLM)
                                   |
        post-generation defence (1, checks the answer)
                                   |
                     answer with citations -> evaluator
                     (evaluator-only labels live here, never upstream)
```

The defences wrap the LLM: four act on retrieved chunks before generation, one checks the answer afterwards.
Attack files are kept separate from the clean corpus.

| Concern | Choice | Why |
|---|---|---|
| Config | pydantic v2 + PyYAML | Typed and validated; a typo fails loudly |
| Dense index | NumPy exact cosine search | ~1–3k chunks; exact, deterministic, no ANN noise; Chroma adapter possible later |
| Sparse index | rank-bm25 | Small and readable |
| Hybrid | Own Reciprocal Rank Fusion (k = 60) | About 10 lines |
| Embedder | nomic-embed-text via Ollama (with its query/document prefixes); bge-small via sentence-transformers | Configurable; embeddings cached by content hash (SQLite) |
| Reranker / NLI | sentence-transformers cross-encoders on CPU torch | Leaves 6 GB VRAM for Ollama |
| LLM layer | `LLMClient` interface → `OllamaClient` (raw HTTP via httpx), `MockLLM` | Explicit control of options; a cloud provider is one more class |
| Response cache | SQLite keyed on model, prompt, options, seed | Reruns are free |
| Eval / plots | pandas, matplotlib | PNG + PDF output |
| Demo | Streamlit | Toggles and expanders in ~200 lines |
| CLI | argparse, `python -m secrag <cmd>` | No extra dependency |
| Dependencies | `requirements.in` → pip-compile → pinned `requirements.txt`; optional extras for HF `datasets` | Lean, reproducible install |
| Tests | pytest; `@pytest.mark.ollama` auto-skips without Ollama | Core suite runs in seconds |

## Phase tracker

| Phase | Scope | Status |
|---|---|---|
| 1 | Skeleton, config, corpus + QA generator, ingestion, chunking, dense / BM25 / hybrid retrieval, recall@k | In progress |
| 2 | Ollama + MockLLM generation, pipeline with citations, clean accuracy | Not started |
| 3 | Three attacks, ASR + retrieval-poisoning metrics, ASR vs N experiment | Not started |
| 4 | Five defences, adaptive attacks, experiment grid, plots | Not started |
| 5 | Demo app, README, THREAT_MODEL.md, RESULTS.md | Not started |

### Phase 1 report

Built, awaiting review (2026-10-28). Environment verified: venv complete (`pip check` clean); all five Ollama
models present (`python -m secrag check-ollama`).

- **Corpus generator** (`secrag/data_gen`): seed 13 → 226 documents (76 core: 31 official, rest staff pages;
  150 distractors: 60 knowledge-hub articles, 40 forum posts incl. 6 with stale values, 25 committee minutes,
  25 job ads), 123 facts, 115 questions (100 answerable, 15 unanswerable). Gold documents are recorded as each
  fact is written, so labels are exact. Committed in `data/`.
- **Ingestion**: front-matter markdown loader that refuses evaluator-only keys (D12); paragraph-aware word
  chunker (120 words + 20 overlap → 240 chunks).
- **Indexes and retrievers**: Ollama / sentence-transformers / hash embedders with SQLite cache; NumPy exact
  cosine; BM25; hybrid RRF (k = 60); optional cross-encoder rerank.
- **Evaluation**: kind-aware answer matcher (D9); hit@k, recall@k, answer_hit@k, MRR; bootstrap 95% CIs;
  results folder with config copy and git state (D17).
- **How to run**: `python -m secrag eval-retrieval --config configs/retrieval_baseline.yaml`
  (`retrieval_smoke.yaml` runs offline in seconds); `python -m pytest`.
- **Tests**: 85 passed in ~5 s (including the live Ollama embedding test).

Baseline retrieval, 100 answerable questions, 95% bootstrap CIs:

| Method | Embedder | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| dense | nomic-embed-text | 0.90 [0.84, 0.95] | 1.00 | 0.945 |
| dense | bge-small-en-v1.5 | 0.83 [0.75, 0.90] | 0.99 | 0.889 |
| sparse (BM25) | none | 0.68 [0.59, 0.77] | 0.97 | 0.799 |
| hybrid (RRF) | nomic-embed-text | 0.86 [0.79, 0.93] | 1.00 | 0.925 |
| hybrid (RRF) | bge-small-en-v1.5 | 0.81 [0.73, 0.88] | 0.99 | 0.887 |

Findings: clean retrieval is near ceiling at k = 5, so poisoned documents must outrank real ones to win (RQ1
becomes a ranking contest). BM25's rank-1 misses are mostly course fee / coordinator / date questions that pull
in forum posts naming the same course; for the refund-window question BM25 ranks the stale forum post first,
so the stale-value risk below is real.

## Open questions and risks

- **qwen3:8b on 6 GB VRAM:** the 5.2 GB model plus KV cache may partly spill to CPU, which slows it and makes
  latency noisier. Mitigation: fixed `num_ctx`, warm-up call excluded from latency, models run in blocks to avoid reloads.
- **Runtime budget:** grid size will be estimated in Phase 3 once per-call latency is measured on this laptop.
- **Stale forum posts:** a few unverified forum posts quote last year's values. This is realistic and lets us measure
  consistency-defence false positives, but may lower clean accuracy for small models; to be checked in Phase 2.
- **Public benchmark:** which NQ subset to load (PoisonedRAG's target questions) is decided in Phase 3.
