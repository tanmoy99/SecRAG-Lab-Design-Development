"""Result folders: every experiment run writes its config, environment and git state (decision D17).

    results/<YYYYmmdd-HHMMSS>_<experiment name>/
        config.yaml        the experiment config exactly as loaded (including its grid)
        run_info.json      git commit + dirty flag, Python and package versions, timestamps
        ...                experiment outputs (CSV tables, plots)
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from secrag.config import REPO_ROOT, ExperimentConfig

PACKAGES = ["numpy", "pydantic", "rank-bm25", "httpx", "pandas", "sentence-transformers", "torch"]


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def create_results_dir(cfg: ExperimentConfig, root: Path | None = None, extra: dict[str, Any] | None = None) -> Path:
    started = dt.datetime.now()
    root = root or REPO_ROOT / "results"
    out = root / f"{started:%Y%m%d-%H%M%S}_{cfg.name}"
    out.mkdir(parents=True, exist_ok=False)
    with open(out / "config.yaml", "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(cfg.model_dump(), fh, sort_keys=False, allow_unicode=True)
    info = {
        "experiment": cfg.name,
        "started_at": started.isoformat(timespec="seconds"),
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _versions(),
        **(extra or {}),
    }
    write_json(out / "run_info.json", info)
    return out


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
