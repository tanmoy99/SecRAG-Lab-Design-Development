import pytest
from pydantic import ValidationError

from secrag.config import REPO_ROOT, ExperimentConfig, expand_grid, load_config


def test_baseline_config_loads_and_expands():
    cfg = load_config(REPO_ROOT / "configs" / "retrieval_baseline.yaml")
    runs = expand_grid(cfg)
    assert len(runs) == 6
    assert {r.config.retrieval.method for r in runs} == {"dense", "sparse", "hybrid"}
    assert all(r.config.grid == {} for r in runs)


def test_grid_dict_values_merge_into_section():
    cfg = ExperimentConfig.model_validate({
        "name": "t", "embedder": {"provider": "hash", "batch_size": 7},
        "grid": {"embedder": [{"dim": 64}, {"dim": 128}]},
    })
    runs = expand_grid(cfg)
    assert [r.config.embedder.dim for r in runs] == [64, 128]
    assert all(r.config.embedder.batch_size == 7 and r.config.embedder.provider == "hash" for r in runs)
    assert runs[0].label == "embedder={dim=64}"


def test_no_grid_gives_single_base_run():
    runs = expand_grid(ExperimentConfig(name="t"))
    assert len(runs) == 1 and runs[0].label == "base"


def test_unknown_key_is_an_error():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "t", "retrieval": {"methd": "dense"}})


def test_grid_typo_is_an_error():
    cfg = ExperimentConfig.model_validate({"name": "t", "grid": {"retrieval.methd": ["dense"]}})
    with pytest.raises(ValidationError):
        expand_grid(cfg)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"name": "t", "chunking": {"chunk_size": 20, "overlap": 20}})
