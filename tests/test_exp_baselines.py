"""E0 baselines on synthetic data: stages end to end, the floor, the elastic net, the null."""

from __future__ import annotations

import functools
import json
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

from m2m import data, paths
from m2m.config import deep_merge, load_config
from m2m.evaluate import spearman_per_feature
from m2m.experiments import baselines, common

N_TRAIN = 45  # synthetic samples 0-44 play PRISM; 45-59 play the external cohort
N_T466 = 20  # the first 20 synthetic metabolites play the annotated set

# Small but complete versions of the real settings, so every stage runs in seconds.
FAST = {
    "seeds": [0, 1],
    "evaluation": {"n_permutations": 3, "n_bootstrap": 50},
    "learning_curve": {"sizes": [20, 40]},
    "baselines": {
        "enet": {"params": {"l1_ratio": [0.5, 0.95], "n_alphas": 10, "inner_folds": 3}},
        "rf": {"params": {"n_estimators": 10}},
    },
}


def e0_config(**override) -> dict:
    cfg = deep_merge(load_config(paths.CONFIGS / "e0_baselines.yaml"), FAST)
    return deep_merge(cfg, override)


def only(cfg: dict, *models: str) -> dict:
    """The config restricted to some baselines (faster tests of one model)."""
    return {**cfg, "baselines": {m: cfg["baselines"][m] for m in models}}


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
    return SimpleNamespace(X=X, Y=Y, labels=labels, train=train, test=test, opened=opened)


def test_every_stage_runs_and_writes_its_rows(d1, results_dir):
    cfg = e0_config()
    baselines.run(cfg)

    nulls = pd.read_csv(results_dir / "null_thresholds.csv")
    assert sorted(nulls["target_set"]) == ["T466", "TALL"]
    for row in nulls.itertuples():
        pooled = np.load(paths.DATA_PROCESSED / "null" / f"D1__{row.target_set}.npy")
        assert pooled.shape == (3 * row.n_features,)
        assert row.threshold == pytest.approx(np.quantile(pooled, 0.95))

    expected = {("mean", "T466"), ("mean", "TALL"), ("ridge", "T466"), ("ridge", "TALL"),
                ("enet", "T466"), ("rf", "T466")}
    summary = pd.read_csv(results_dir / "summary.csv")
    assert set(zip(summary["model"], summary["target_set"], strict=True)) == expected
    assert len(summary) == len(expected)
    for row in summary.itertuples():
        selection = json.loads((results_dir / "selected" / f"e0_{row.model}__{row.target_set}.json")
                               .read_text())
        assert row.cv_mean_rho == pytest.approx(selection["cv_mean_rho"])
        assert row.n_seeds == (2 if row.model == "rf" else 1)  # only the forest varies by seed
        assert row.n_features_mean == (N_T466 if row.target_set == "T466" else 40)
    # The planted signal: ridge must beat the floor on the external split.
    by_model = summary.set_index(["model", "target_set"])["mean_rho_mean"]
    assert by_model["ridge", "T466"] > by_model["mean", "T466"] + 0.1

    curve = pd.read_csv(results_dir / "learning_curve.csv")
    assert set(curve["model"]) == {"ridge"}
    assert sorted(zip(curve["n_train"], curve["seed"], strict=True)) == [
        (20, 0), (20, 1), (40, 0), (40, 1)
    ]
    # null (one per target set) + external (one per target set) + curve
    assert len(d1.opened) == 5


def test_null_opens_the_external_cohort_once_per_target_set(d1, results_dir):
    baselines.run(e0_config(), stage="null")
    assert len(d1.opened) == 2
    assert "(T466)" in d1.opened[0] and "(TALL)" in d1.opened[1]
    assert all(p.startswith("permutation null") for p in d1.opened)
    assert len(pd.read_csv(results_dir / "null_thresholds.csv")) == 2


def external_rho(pipe, d1) -> np.ndarray:
    """ρ of a fitted pipeline on the external split, against raw intensities (NaN → 0).

    Spearman is rank-based and log10(x + 1) plus z-scoring is monotone per metabolite, so raw
    intensities rank exactly like the pipeline's truth: no package scoring code is reused here.
    """
    predictions = pipe.predict(d1.X.loc[d1.test])
    truth = d1.Y.loc[d1.test, predictions.columns]
    return np.nan_to_num(spearman_per_feature(truth, predictions), nan=0.0)


def test_null_refits_ridge_on_shuffled_pairs_and_scores_the_external_truth(d1, results_dir):
    cfg = e0_config()
    baselines.run(cfg, stage="null")

    evaluation = cfg["evaluation"]
    X, Y = d1.X.loc[d1.train], d1.Y.loc[d1.train]
    build = functools.partial(common.build_ridge, cfg=cfg, target_set="T466")
    expected = []
    for p in range(evaluation["n_permutations"]):  # the permutation contract of evaluate.py
        order = np.random.default_rng([evaluation["permutation_seed"], p]).permutation(len(X))
        shuffled = X.iloc[order].set_axis(X.index, axis=0)
        expected.append(external_rho(build(cfg["ridge"], 0).fit(shuffled, Y), d1))
    pooled = np.load(paths.DATA_PROCESSED / "null" / "D1__T466.npy")
    np.testing.assert_allclose(pooled, np.concatenate(expected), rtol=0, atol=1e-12)
    row = pd.read_csv(results_dir / "null_thresholds.csv").set_index("target_set").loc["T466"]
    assert row["threshold"] == pytest.approx(np.quantile(np.concatenate(expected), 0.95))
    # Shuffling breaks the planted signal: every permutation sits far below the real ridge.
    real = external_rho(build(cfg["ridge"], 0).fit(X, Y), d1).mean()
    assert max(rho.mean() for rho in expected) < real - 0.2


def test_external_ridge_row_is_ridge_fit_on_all_training_rows(d1, results_dir):
    cfg = only(e0_config(), "ridge")
    for stage in ("null", "cv", "external"):
        baselines.run(cfg, stage=stage)

    pipe = common.build_ridge(cfg["ridge"], 0, cfg=cfg, target_set="T466")
    rho = external_rho(pipe.fit(d1.X.loc[d1.train], d1.Y.loc[d1.train]), d1)
    row = pd.read_csv(results_dir / "summary.csv").set_index(["model", "target_set"])
    assert row.loc[("ridge", "T466"), "mean_rho_mean"] == pytest.approx(rho.mean(), abs=1e-12)
    assert row.loc[("ridge", "T466"), "n_features_mean"] == len(rho)


def test_a_dry_run_null_refuses_the_real_processed_directory(d1, tmp_path, monkeypatch):
    # The pooled null feeds the null-distribution figure: a dry run must not replace the real one.
    repo = tmp_path / "repo"
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    monkeypatch.setattr(paths, "DATA_PROCESSED", repo / "data" / "processed")
    monkeypatch.setenv(common.DRY_RUN_ENV, "1")
    with pytest.raises(RuntimeError, match="M2M_DATA_PROCESSED"):
        baselines.run(e0_config(), stage="null")
    assert d1.opened == [] and not (repo / "data").exists()

    monkeypatch.setattr(paths, "DATA_PROCESSED", tmp_path / "scratch")  # redirected: allowed
    baselines.run(e0_config(), stage="null")
    assert (tmp_path / "scratch" / "null" / "D1__T466.npy").exists()


def test_external_reads_every_record_before_opening_the_cohort(d1):
    cfg = only(e0_config(), "ridge")
    baselines.run(cfg, stage="null")
    opened = len(d1.opened)
    baselines.run(deep_merge(cfg, {"baselines": {"ridge": {"target_sets": ["T466"]}}}), stage="cv")
    with pytest.raises(FileNotFoundError):  # TALL was never cross-validated
        baselines.run(cfg, stage="external")
    assert len(d1.opened) == opened  # T466 was not scored first


def test_cv_never_opens_the_external_cohort(d1, monkeypatch):
    def forbidden(purpose):
        raise AssertionError(f"model selection opened the external cohort: {purpose}")

    monkeypatch.setattr(common, "load_d1_test", forbidden)
    baselines.run(only(e0_config(), "mean", "ridge"), stage="cv")


def test_external_refuses_to_open_the_cohort_without_a_saved_selection(d1):
    with pytest.raises(FileNotFoundError):
        baselines.run(only(e0_config(), "ridge"), stage="external")
    assert d1.opened == []


def test_external_refuses_a_config_changed_since_cv(d1):
    cfg = only(e0_config(), "ridge")
    baselines.run(cfg, stage="cv")
    changed = deep_merge(cfg, {"ridge": {"n_alphas": 5}})
    with pytest.raises(ValueError, match="rerun --stage cv"):
        baselines.run(changed, stage="external")
    assert d1.opened == []


def test_mean_predictor_is_the_floor_and_raises_no_warnings(d1, results_dir):
    cfg = only(e0_config(), "mean")
    baselines.run(cfg, stage="null")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning (e.g. a 0/0 in Spearman) fails the test
        baselines.run(cfg, stage="cv")
        baselines.run(cfg, stage="external")

    for target_set, n_features in (("T466", N_T466), ("TALL", 40)):
        cv = pd.read_csv(results_dir / "selected" / f"e0_mean__{target_set}__cv.csv").iloc[0]
        assert cv["cv_n_nan"] == cv["cv_n_features"] == n_features
        assert cv["cv_mean_rho"] == 0.0
    summary = pd.read_csv(results_dir / "summary.csv")
    assert (summary["n_nan_mean"] == summary["n_features_mean"]).all()
    assert (summary["mean_rho_mean"] == 0.0).all()
    assert (summary["top50_mean_rho_mean"] == 0.0).all()
    assert (summary["n_above_null_mean"] == 0.0).all()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_per_target_elastic_net_equals_one_elastic_net_cv_per_column(rng, n_jobs):
    X = rng.normal(size=(40, 8))
    Y = X[:, :3] @ rng.normal(size=(3, 3)) + rng.normal(size=(40, 3))
    settings = dict(l1_ratio=[0.2, 0.9], n_alphas=20, eps=1e-2, inner_folds=4, inner_seed=3,
                    max_iter=5000, tol=1e-4)
    model = baselines.PerTargetElasticNet(**settings, n_jobs=n_jobs).fit(X, Y)
    X_new = rng.normal(size=(5, 8))
    predictions = model.predict(X_new)
    for j in range(Y.shape[1]):
        reference = ElasticNetCV(
            l1_ratio=[0.2, 0.9], alphas=20, eps=1e-2, max_iter=5000, tol=1e-4,
            cv=KFold(n_splits=4, shuffle=True, random_state=3),
        ).fit(X, Y[:, j])
        assert model.alpha_[j] == reference.alpha_
        assert model.l1_ratio_[j] == reference.l1_ratio_
        np.testing.assert_allclose(model.coef_[j], reference.coef_, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(predictions[:, j], reference.predict(X_new), rtol=1e-12,
                                   atol=1e-12)


def test_elastic_net_summarises_non_convergence_in_one_warning(rng):
    X, Y = rng.normal(size=(30, 6)), rng.normal(size=(30, 4))
    model = baselines.PerTargetElasticNet(
        l1_ratio=[0.5], n_alphas=5, eps=1e-3, inner_folds=3, inner_seed=0, max_iter=1, tol=1e-12
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, Y)
    assert [w.category for w in caught] == [ConvergenceWarning]
    assert model.n_not_converged_ == 4


def test_baseline_config_typos_are_rejected():
    cfg = e0_config()
    with pytest.raises(ValueError, match="unknown key"):
        baselines.configured_baselines(
            deep_merge(cfg, {"baselines": {"rf": {"external_seed": [0]}}})
        )
    with pytest.raises(ValueError, match="unknown baseline"):
        baselines.configured_baselines(deep_merge(cfg, {"baselines": {"lasso": {}}}))
    with pytest.raises(ValueError, match="not in"):
        baselines.configured_baselines(
            deep_merge(cfg, {"baselines": {"enet": {"target_sets": ["T999"]}}})
        )


def test_models_and_seeds_follow_the_config():
    models = {m.name: m for m in baselines.configured_baselines(e0_config())}
    assert list(models) == ["mean", "ridge", "enet", "rf"]
    assert models["rf"].external_seeds == (0, 1)  # every model seed
    assert models["enet"].external_seeds == (0,)
    assert models["ridge"].params == e0_config()["ridge"]
    assert models["enet"].outer_jobs(4) == 1 and models["rf"].outer_jobs(4) == 4
