# `secrag.eval` — evaluation

Everything here is **evaluator-side** (decision D12): the pipeline only ever receives the question string.

| File | Purpose |
|---|---|
| `qa.py` | `QAItem` (question, gold doc ids, canonical answer, kind, unit, aliases) and JSONL I/O |
| `matching.py` | `contains_answer(text, kind, value, unit, aliases)`: kind-aware normalisation of money, dates (day first), phones, emails and numbers with units (D9) |
| `retrieval.py` | Per-question retrieval metrics |
| `stats.py` | Percentile bootstrap 95% CI over questions (D4) |
| `results.py` | Results folder with config copy, git commit + dirty flag, package versions (D17) |

## Retrieval metrics (answerable questions only)

- **hit@k**: a top-k chunk comes from a gold document. This is the headline metric.
- **recall@k**: share of gold documents in the top k. Facts stated in several documents have several gold
  documents, so recall@1 can be below 1 even for a perfect retriever.
- **answer_hit@k**: a top-k chunk from a gold document contains the answer itself.
- **MRR**: mean reciprocal rank of the first gold-document chunk.

## Results folder

`python -m secrag eval-retrieval --config <yaml>` writes `results/<timestamp>_<name>/`:
`config.yaml`, `run_info.json`, `per_question.csv` (one row per run × question), `summary.csv` / `summary.json`
(means with `_lo` / `_hi` CI bounds).
