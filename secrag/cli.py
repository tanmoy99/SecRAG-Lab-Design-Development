"""Command-line interface: `python -m secrag <command> ...`.

Commands:
    generate-data    write the synthetic corpus and QA set (fixed seed)
    corpus-stats     document / chunk counts for a config's corpus
    search           run one query through a config's retriever and show the results
    eval-retrieval   evaluate retrieval over a config's grid; writes a results folder
    check-ollama     check the Ollama server and the models the project needs
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from secrag.config import REPO_ROOT, get_settings, load_config, resolve_path

REQUIRED_MODELS = ["nomic-embed-text", "qwen3:1.7b", "qwen3:4b", "qwen3:8b", "llama3.2:3b"]


def _generate_data(args: argparse.Namespace) -> int:
    from secrag.data_gen import generate, write_generated

    data = generate(args.seed)
    out = resolve_path(args.out_dir)
    write_generated(data, out / "corpus" / "northbridge", out / "qa" / "northbridge_qa.jsonl",
                    out / "qa" / "northbridge_facts.jsonl")
    n_un = sum(not q.answerable for q in data.qa)
    print(f"seed {args.seed}: {len(data.documents)} documents ({len(data.core_doc_ids)} core, "
          f"{len(data.documents) - len(data.core_doc_ids)} distractors), {len(data.facts)} facts, "
          f"{len(data.qa)} questions ({n_un} unanswerable) -> {out}")
    return 0


def _corpus_stats(args: argparse.Namespace) -> int:
    from secrag.experiments import load_chunks
    from secrag.ingest import load_corpus

    cfg = load_config(args.config)
    docs = load_corpus([resolve_path(d) for d in cfg.data.corpus_dirs])
    chunks = load_chunks(cfg)
    words = sorted(len(c.text.split()) for c in chunks)
    print(f"documents: {len(docs)}")
    for key, n in sorted(Counter(d.trust_level.label() for d in docs).items()):
        print(f"  trust={key}: {n}")
    print(f"chunks: {len(chunks)} (chunk_size={cfg.chunking.chunk_size}, overlap={cfg.chunking.overlap})")
    print(f"  words per chunk: min {words[0]}, median {words[len(words) // 2]}, max {words[-1]}")
    return 0


def _search(args: argparse.Namespace) -> int:
    from secrag.config import ExperimentConfig
    from secrag.experiments import load_chunks
    from secrag.retrieve import build_retriever

    cfg = load_config(args.config)
    if args.method:
        cfg = ExperimentConfig.model_validate({**cfg.model_dump(), "retrieval": {
            **cfg.retrieval.model_dump(), "method": args.method}})
    retriever = build_retriever(cfg, load_chunks(cfg), get_settings())
    for rc in retriever.retrieve(args.query, args.k):
        parts = ", ".join(f"{k}={v:.3f}" for k, v in rc.scores.items())
        print(f"\n#{rc.rank} {rc.chunk.chunk_id} [{rc.chunk.trust_level.label()}] {rc.chunk.title} ({parts})")
        print("   " + rc.chunk.text[:300].replace("\n", "\n   "))
    return 0


def _eval_retrieval(args: argparse.Namespace) -> int:
    from secrag.experiments import run_retrieval_experiment

    out, summary = run_retrieval_experiment(args.config)
    cols = ["run", "hit@1", "hit@5", "recall@5", "answer_hit@5", "mrr"]
    cols = [c for c in cols if c in summary.columns]
    print()
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nresults: {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}")
    return 0


def _check_ollama(args: argparse.Namespace) -> int:
    import httpx

    host = get_settings().ollama_host
    try:
        version = httpx.get(f"{host}/api/version", timeout=5).json()["version"]
        names = {m["name"] for m in httpx.get(f"{host}/api/tags", timeout=10).json()["models"]}
    except httpx.HTTPError as exc:
        print(f"Ollama not reachable at {host}: {exc}")
        return 1
    print(f"Ollama {version} at {host}")
    missing = 0
    for model in REQUIRED_MODELS:
        ok = model in names or f"{model}:latest" in names
        missing += not ok
        print(f"  {'ok     ' if ok else 'MISSING'} {model}")
    return 1 if missing else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m secrag", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-data", help="write the synthetic corpus and QA set")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out-dir", default="data", help="output folder (default: data/)")
    p.set_defaults(func=_generate_data)

    p = sub.add_parser("corpus-stats", help="document and chunk counts")
    p.add_argument("--config", default="configs/retrieval_baseline.yaml")
    p.set_defaults(func=_corpus_stats)

    p = sub.add_parser("search", help="retrieve chunks for one query")
    p.add_argument("query")
    p.add_argument("--config", default="configs/retrieval_baseline.yaml")
    p.add_argument("--method", choices=["dense", "sparse", "hybrid"], help="override retrieval.method")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=_search)

    p = sub.add_parser("eval-retrieval", help="evaluate retrieval over a config grid")
    p.add_argument("--config", default="configs/retrieval_baseline.yaml")
    p.set_defaults(func=_eval_retrieval)

    p = sub.add_parser("check-ollama", help="check Ollama and required models")
    p.set_defaults(func=_check_ollama)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
