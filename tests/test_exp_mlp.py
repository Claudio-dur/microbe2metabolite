"""E1 MLP on synthetic data: selection by CV only, then external and learning curve."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from m2m import data, paths, protocol
from m2m.config import deep_merge, load_config
from m2m.evaluate import NullResult
from m2m.experiments import common, mlp

N_TRAIN = 45  # synthetic samples 0-44 play PRISM; 45-59 play the external cohort
N_T466 = 20  # the first 20 synthetic metabolites play the annotated set

# The real config, shrunk so that every stage runs in seconds (2 grid points, short training).
FAST = {
    "seeds": [0, 1],
    "evaluation": {"n_bootstrap": 50},
    "learning_curve": {"sizes": [20, 40]},
    "mlp": {
        "grid": {"lr": [1.0e-3], "dropout": [0.2, 0.5], "weight_decay": [1.0e-4]},
        "train": {"max_epochs": 25, "patience": 5},
    },
}


def e1_config() -> dict:
    return deep_merge(load_config(paths.CONFIGS / "e1_mlp.yaml"), FAST)


@pytest.fixture
def d1(synthetic_cohort, monkeypatch):
    """Synthetic D1: train/test split, T466 = first 20 metabolites, external opens recorded."""
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    train, test = X.index[:N_TRAIN], X.index[N_TRAIN:]
    opened: list[str] = []

    def load_test(purpose: str):
        opened.append(purpose)
        return X.loc[test], Y.loc[test]

    def no_real_data(*args, **kwargs):
        raise AssertionError("a test touched the real data loaders")

    monkeypatch.setattr(
        common, "load_d1_train", lambda: (X.loc[train], Y.loc[train], labels.loc[train])
    )
    monkeypatch.setattr(common, "load_d1_test", load_test)
    monkeypatch.setattr(data, "t466_features", lambda raw_dir=None: list(Y.columns[:N_T466]))
    monkeypatch.setattr(data, "load_prism", no_real_data)
    monkeypatch.setattr(data, "load_nlibd", no_real_data)
    return SimpleNamespace(X=X, Y=Y, labels=labels, opened=opened)


def save_fake_null_thresholds(threshold: float = 0.3) -> None:
    """E1 reads E0's null thresholds; a fixed one keeps these tests independent of E0."""
    null = NullResult(pooled_rho=np.zeros(4), threshold=threshold,
                      per_permutation_mean=np.zeros(2), quantile=0.95)
    for target_set in ("T466", "TALL"):
        protocol.save_null_threshold(data="D1", target_set=target_set, null=null, n_features=4)


def test_every_stage_runs_and_writes_its_rows(d1, results_dir):
    save_fake_null_thresholds()
    mlp.run(e1_config())

    for target_set in ("T466", "TALL"):
        table = pd.read_csv(results_dir / "selected" / f"e1_mlp__{target_set}__cv.csv")
        assert len(table) == 2  # one row per grid point
        selection = json.loads((results_dir / "selected" / f"e1_mlp__{target_set}.json")
                               .read_text())
        best = table.loc[table["cv_mean_rho"].idxmax()]
        assert selection["params"] == json.loads(best["params"])
        assert selection["cv_mean_rho"] == pytest.approx(best["cv_mean_rho"])

    summary = pd.read_csv(results_dir / "summary.csv")
    assert sorted(zip(summary["model"], summary["target_set"], strict=True)) == [
        ("mlp", "T466"), ("mlp", "TALL")
    ]
    assert (summary["n_seeds"] == 2).all()
    assert (summary["null_threshold"] == 0.3).all()
    t466 = summary.set_index("target_set").loc["T466"]
    assert t466["n_features_mean"] == N_T466
    assert np.isfinite(summary["mean_rho_mean"]).all()

    runs = pd.read_csv(results_dir / "runs.csv")
    assert (runs["stage"] == "cv").sum() == 4  # 2 grid points x 2 target sets
    assert (runs["stage"] == "external").sum() == 4  # 2 seeds x 2 target sets
    curve = pd.read_csv(results_dir / "learning_curve.csv")
    assert set(curve["model"]) == {"mlp"} and set(curve["target_set"]) == {"T466"}
    assert len(curve) == 4  # sizes 20, 40 x seeds 0, 1
    assert len(d1.opened) == 3  # external T466, external TALL, curve
    assert all("NLIBD" in purpose for purpose in d1.opened)


def test_selection_never_opens_the_external_cohort(d1, monkeypatch):
    def forbidden(purpose):
        raise AssertionError(f"model selection opened the external cohort: {purpose}")

    monkeypatch.setattr(common, "load_d1_test", forbidden)
    mlp.run(e1_config(), stage="select")


def test_cv_scores_do_not_depend_on_n_jobs(d1):
    cfg = e1_config()
    X, Y, labels = common.load_d1_train()
    folds = common.d1_folds(labels, cfg)
    grid = protocol.expand_grid(cfg["mlp"]["grid"])[:1]
    # TALL: worker processes do not see the monkeypatched T466 list (they would read data/raw).
    build = mlp.mlp_builder(cfg, "TALL")
    serial, _ = protocol.grid_search_cv(build, grid, X, Y, labels, folds, cfg, n_jobs=1)
    parallel, _ = protocol.grid_search_cv(build, grid, X, Y, labels, folds, cfg, n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_external_refuses_a_selection_outside_the_current_grid(d1):
    save_fake_null_thresholds()
    cfg = e1_config()
    stale = {"dropout": 0.3, "lr": 1.0e-2, "weight_decay": 0.0}
    table = pd.DataFrame([{"params": json.dumps(stale, sort_keys=True), "cv_mean_rho": 0.5}])
    for target_set in ("T466", "TALL"):
        protocol.save_selection(mlp.SELECTION, target_set, stale, table)
    with pytest.raises(ValueError, match="not in the current mlp.grid"):
        mlp.run(cfg, stage="external")
    assert d1.opened == []


def test_external_needs_the_selection_before_opening_the_cohort(d1):
    save_fake_null_thresholds()
    with pytest.raises(FileNotFoundError):
        mlp.run(e1_config(), stage="external")
    assert d1.opened == []


def test_external_refuses_a_selection_made_on_a_smaller_grid(d1):
    save_fake_null_thresholds()
    cfg = deep_merge(e1_config(), {"target_sets": ["T466"]})
    mlp.run(cfg, stage="select")
    grown = deep_merge(cfg, {"mlp": {"grid": {"lr": [1.0e-3, 3.0e-4]}}})
    # The old winner is still a point of the grown grid, but lr 3e-4 was never cross-validated.
    winner = protocol.load_selection(mlp.SELECTION, "T466")["params"]
    assert winner in protocol.expand_grid(grown["mlp"]["grid"])
    with pytest.raises(ValueError, match="different grid"):
        mlp.run(grown, stage="external")
    assert d1.opened == []
    assert mlp.load_checked_selection(cfg, "T466")["params"] == winner  # the searched grid is fine


def test_external_reads_every_selection_before_opening_the_cohort(d1):
    save_fake_null_thresholds()
    cfg = e1_config()
    mlp.run(deep_merge(cfg, {"target_sets": ["T466"]}), stage="select")
    with pytest.raises(FileNotFoundError):  # TALL was never selected
        mlp.run(cfg, stage="external")
    assert d1.opened == []  # T466 was not scored first
