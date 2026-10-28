import json

import numpy as np
import pandas as pd
import pytest
import yaml

from secrag.eval.qa import QAItem, load_qa, write_jsonl
from secrag.eval.retrieval import question_metrics
from secrag.eval.stats import bootstrap_ci
from secrag.experiments import run_retrieval_experiment
from secrag.types import Chunk, RetrievedChunk, TrustLevel


def _rc(doc_id: str, text: str, rank: int) -> RetrievedChunk:
    chunk = Chunk(f"{doc_id}#0", doc_id, 0, text, "t", "s", "a", TrustLevel.STAFF, "2026-01-01", "page", "h")
    return RetrievedChunk(chunk, rank, 1.0 / rank)


ITEM = QAItem("q1", "fee?", True, "x.fee", answer="AUD 450", answer_kind="money",
              aliases=("AUD 450",), fact_id="f", gold_doc_ids=("g1", "g2"))


def test_question_metrics_by_hand():
    results = [_rc("other", "nothing", 1), _rc("g2", "no value here", 2), _rc("g1", "costs $450", 3)]
    row = question_metrics(ITEM, results, [1, 2, 3])
    assert row["first_gold_rank"] == 2 and row["rr"] == 0.5
    assert (row["hit@1"], row["hit@2"], row["hit@3"]) == (0.0, 1.0, 1.0)
    assert (row["recall@2"], row["recall@3"]) == (0.5, 1.0)
    assert (row["answer_hit@2"], row["answer_hit@3"]) == (0.0, 1.0)


def test_answer_in_non_gold_doc_does_not_count():
    row = question_metrics(ITEM, [_rc("forum", "I paid $450", 1)], [1])
    assert row["answer_hit@1"] == 0.0 and row["rr"] == 0.0


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    values = np.r_[np.ones(70), np.zeros(30)]
    a, b = bootstrap_ci(values, 1000, 0.95, seed=1), bootstrap_ci(values, 1000, 0.95, seed=1)
    assert a == b
    assert a.mean == pytest.approx(0.7) and a.low < 0.7 < a.high
    assert 0.05 < a.high - a.low < 0.25  # about ±1.96 * sqrt(.21/100)
    assert bootstrap_ci([1, 1, 1]).low == 1.0


def test_qa_jsonl_round_trip(tmp_path, generated):
    write_jsonl(tmp_path / "qa.jsonl", [q.to_record() for q in generated.qa])
    assert load_qa(tmp_path / "qa.jsonl") == generated.qa


def test_pipeline_objects_carry_no_evaluator_labels():
    """Decision D12: nothing the pipeline sees has a field for ground-truth labels."""
    from dataclasses import fields

    from secrag.types import Document

    forbidden = {"is_poisoned", "poisoned", "label", "gold", "is_gold", "attack_id", "target_answer"}
    for cls in (Document, Chunk, RetrievedChunk):
        assert not forbidden & {f.name for f in fields(cls)}, cls


def test_retrieval_experiment_writes_results_folder(tmp_path):
    from secrag.config import REPO_ROOT

    out, summary = run_retrieval_experiment(REPO_ROOT / "configs" / "retrieval_smoke.yaml",
                                            results_root=tmp_path, quiet=True)
    assert {p.name for p in out.iterdir()} >= {"config.yaml", "run_info.json", "per_question.csv",
                                                 "summary.csv", "summary.json"}
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert {"commit", "dirty"} <= set(info["git"])
    assert yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))["name"] == "retrieval_smoke"
    assert list(summary["method"]) == ["dense", "sparse", "hybrid"]
    per_q = pd.read_csv(out / "per_question.csv")
    assert len(per_q) == 3 * 100
    # Sanity floor: on this corpus even lexical retrieval should find most gold documents.
    assert summary.set_index("method").loc["sparse", "hit@10"] > 0.8
