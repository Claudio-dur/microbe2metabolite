"""Paired comparisons read saved predictions only and test A vs B on identical targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m2m import paths, protocol
from m2m.config import load_config
from m2m.experiments import compare

CFG = load_config(paths.CONFIGS / "compare.yaml")


def save(model: str, target_set: str, truth: np.ndarray, preds: np.ndarray) -> None:
    result = protocol.ExternalResult(
        truth=pd.DataFrame(truth, index=[f"S{i}" for i in range(len(truth))],
                           columns=[f"m{j}" for j in range(truth.shape[1])]),
        predictions=[pd.DataFrame(p, index=[f"S{i}" for i in range(len(truth))],
                                  columns=[f"m{j}" for j in range(truth.shape[1])]) for p in preds],
        seeds=list(range(len(preds))),
        per_seed=[],
        rho_by_seed=pd.DataFrame(),
    )
    protocol.write_external_predictions(result, model=model, target_set=target_set)


def test_pairs_are_scored_from_saved_predictions(results_dir, monkeypatch):
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(40, 1))
    truth = latent + rng.normal(size=(40, 12))
    save("mgm_finetune", "T466", truth, [latent + 0.3 * rng.normal(size=(40, 12))] * 2)
    save("mgm_scratch", "T466", truth, [rng.normal(size=(40, 12))] * 2)
    cfg = {**CFG, "comparisons": [
        {"a": "mgm_finetune", "b": "mgm_scratch", "target_sets": ["T466", "TALL"],
         "question": "H1"}]}
    import m2m.data as data

    def forbidden(*args, **kwargs):
        raise AssertionError("comparisons must not open the external cohort")

    monkeypatch.setattr(data, "load_nlibd", forbidden)
    compare.run(cfg)
    rows = pd.read_csv(results_dir / compare.OUTPUT_CSV)
    assert len(rows) == 1  # TALL has no saved predictions: skipped with a note
    row = rows.iloc[0]
    assert row["delta"] > 0 and row["ci_low"] > 0 and row["question"] == "H1"


def test_mismatched_targets_are_refused(results_dir):
    rng = np.random.default_rng(1)
    save("mlp", "T466", rng.normal(size=(30, 5)), [rng.normal(size=(30, 5))])
    save("ridge", "T466", rng.normal(size=(30, 5)), [rng.normal(size=(30, 5))])
    with pytest.raises(ValueError, match="identical targets"):
        compare.compare_pair("mlp", "ridge", "T466", CFG["evaluation"])
