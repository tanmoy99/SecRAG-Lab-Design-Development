"""Typed experiment configuration loaded from YAML.

Every experiment is one YAML file (see `configs/`). The file describes a *base*
configuration plus an optional `grid:` of overrides; the runner expands the grid
into the cartesian product of runs. All models use `extra="forbid"`, so a typo in
a YAML key is an error rather than a silently ignored setting.

Example grid (dotted paths into the config; a dict value is merged into that section):

    grid:
      retrieval.method: [dense, sparse, hybrid]
      embedder:
        - {provider: ollama, model: nomic-embed-text}
        - {provider: sentence_transformers, model: BAAI/bge-small-en-v1.5}
"""

from __future__ import annotations

import copy
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    """Which corpus and QA set to use. Paths are relative to the repo root."""

    corpus_dirs: list[str] = ["data/corpus/northbridge"]
    qa_path: str = "data/qa/northbridge_qa.jsonl"


class ChunkingConfig(StrictModel):
    """Paragraph-aware chunking, sizes measured in words."""

    chunk_size: int = Field(120, gt=0)
    overlap: int = Field(20, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.overlap >= self.chunk_size:
            raise ValueError("chunking.overlap must be smaller than chunking.chunk_size")
        return self


class EmbedderConfig(StrictModel):
    """Embedding model. `hash` is a dependency-free mock used in tests."""

    provider: Literal["ollama", "sentence_transformers", "hash"] = "ollama"
    model: str = "nomic-embed-text"
    # None = use the known default for this model (see secrag.index.embedders).
    query_prefix: str | None = None
    document_prefix: str | None = None
    batch_size: int = Field(32, gt=0)
    dim: int = Field(512, gt=0)  # only used by the hash embedder


class RetrievalConfig(StrictModel):
    method: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = Field(5, gt=0)
    candidate_k: int = Field(50, gt=0)  # per-retriever depth fed into fusion / reranking
    rrf_k: int = Field(60, gt=0)  # Reciprocal Rank Fusion constant (Cormack et al., 2009)
    reranker: str | None = None  # e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EvalConfig(StrictModel):
    ks: list[int] = [1, 3, 5, 10]  # cut-offs for recall@k / hit@k
    bootstrap_samples: int = Field(2000, gt=0)
    confidence: float = Field(0.95, gt=0, lt=1)


class ExperimentConfig(StrictModel):
    name: str
    description: str = ""
    seed: int = 13
    data: DataConfig = DataConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    eval: EvalConfig = EvalConfig()
    grid: dict[str, list[Any]] = {}


@dataclass(frozen=True)
class Run:
    """One point of an expanded grid."""

    label: str  # human-readable, e.g. "retrieval.method=dense"
    overrides: dict[str, Any]
    config: ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return ExperimentConfig.model_validate(raw)


def _set_dotted(target: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = target
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    last = keys[-1]
    if isinstance(value, dict) and isinstance(node.get(last), dict):
        node[last] = {**node[last], **value}
    else:
        node[last] = value


def _format_value(value: Any) -> str:
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}={v}" for k, v in value.items()) + "}"
    return str(value)


def expand_grid(config: ExperimentConfig) -> list[Run]:
    """Expand `config.grid` into one validated config per combination, in YAML order."""
    base = config.model_dump(exclude={"grid"})
    if not config.grid:
        return [Run(label="base", overrides={}, config=ExperimentConfig.model_validate(base))]
    keys = list(config.grid)
    runs = []
    for combo in itertools.product(*(config.grid[k] for k in keys)):
        merged = copy.deepcopy(base)
        overrides = dict(zip(keys, combo))
        for key, value in overrides.items():
            _set_dotted(merged, key, value)
        label = ";".join(f"{k}={_format_value(v)}" for k, v in overrides.items())
        runs.append(Run(label=label, overrides=overrides, config=ExperimentConfig.model_validate(merged)))
    return runs


@dataclass(frozen=True)
class Settings:
    """Machine-specific settings from the environment / .env (never from configs)."""

    ollama_host: str
    cache_dir: Path


def get_settings() -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    cache = Path(os.environ.get("SECRAG_CACHE_DIR", ".cache"))
    if not cache.is_absolute():
        cache = REPO_ROOT / cache
    return Settings(
        ollama_host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
        cache_dir=cache,
    )


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
