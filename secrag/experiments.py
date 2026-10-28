"""Experiment runners. Phase 1 has one: retrieval quality over a config grid."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from secrag.config import ExperimentConfig, Run, expand_grid, get_settings, load_config, resolve_path
from secrag.eval.qa import load_qa
from secrag.eval.results import create_results_dir, write_json
from secrag.eval.retrieval import evaluate_retriever, metric_names
from secrag.eval.stats import bootstrap_ci
from secrag.index import Embedder, build_embedder
from secrag.ingest import chunk_corpus, load_corpus
from secrag.retrieve import build_retriever
from secrag.types import Chunk


def load_chunks(cfg: ExperimentConfig) -> list[Chunk]:
    docs = load_corpus([resolve_path(d) for d in cfg.data.corpus_dirs])
    return chunk_corpus(docs, cfg.chunking)


def _run_columns(run: Run) -> dict[str, Any]:
    c = run.config
    return {
        "run": run.label,
        "method": c.retrieval.method,
        "embedder": "-" if c.retrieval.method == "sparse" else f"{c.embedder.provider}:{c.embedder.model}",
        "reranker": c.retrieval.reranker or "-",
        "chunk_size": c.chunking.chunk_size,
        "overlap": c.chunking.overlap,
    }


def run_retrieval_experiment(config_path: str | Path, results_root: Path | None = None,
                             quiet: bool = False) -> tuple[Path, pd.DataFrame]:
    """Evaluate every grid point of a retrieval config; returns (results folder, summary table)."""
    cfg = load_config(config_path)
    runs = expand_grid(cfg)
    settings = get_settings()
    qa = load_qa(resolve_path(cfg.data.qa_path))
    out = create_results_dir(cfg, results_root, extra={"config_path": str(config_path), "n_runs": len(runs)})
    log = (lambda *_: None) if quiet else print
    log(f"{cfg.name}: {len(runs)} run(s), {sum(q.answerable for q in qa)} answerable questions -> {out}")

    chunk_cache: dict[str, list[Chunk]] = {}
    embedders: dict[str, Embedder] = {}
    question_rows, summary_rows = [], []
    for n, run in enumerate(runs, 1):
        c = run.config
        chunk_key = f"{c.data.corpus_dirs}|{c.chunking.model_dump_json()}"
        if chunk_key not in chunk_cache:
            chunk_cache[chunk_key] = load_chunks(c)
        chunks = chunk_cache[chunk_key]
        embedder = None
        if c.retrieval.method != "sparse":
            key = c.embedder.model_dump_json()
            embedder = embedders.setdefault(key, build_embedder(c.embedder, settings))

        t0 = time.perf_counter()
        retriever = build_retriever(c, chunks, settings, embedder=embedder)
        build_s = time.perf_counter() - t0
        rows = evaluate_retriever(retriever, qa, c.eval.ks)
        cols = _run_columns(run)
        question_rows += [{**cols, **r} for r in rows]

        summary: dict[str, Any] = {**cols, "n_chunks": len(chunks), "n_questions": len(rows),
                                   "index_build_s": round(build_s, 2),
                                   "median_latency_ms": round(float(pd.Series([r["latency_ms"] for r in rows]).median()), 2)}
        for m in metric_names(c.eval.ks):
            est = bootstrap_ci([r[m] for r in rows], c.eval.bootstrap_samples, c.eval.confidence, c.seed)
            summary[m], summary[f"{m}_lo"], summary[f"{m}_hi"] = est.mean, est.low, est.high
        summary_rows.append(summary)
        k = 5 if 5 in c.eval.ks else c.eval.ks[-1]
        log(f"  [{n}/{len(runs)}] {run.label}: hit@{k}={summary[f'hit@{k}']:.3f} "
            f"MRR={summary['rr']:.3f} ({build_s:.1f}s index)")

    summary_df = pd.DataFrame(summary_rows).rename(columns={"rr": "mrr", "rr_lo": "mrr_lo", "rr_hi": "mrr_hi"})
    pd.DataFrame(question_rows).to_csv(out / "per_question.csv", index=False, lineterminator="\n")
    summary_df.to_csv(out / "summary.csv", index=False, lineterminator="\n")
    write_json(out / "summary.json", summary_rows)
    return out, summary_df
