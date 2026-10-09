"""The shared protocol: folds never leak, the external cohort is never needed for selection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from m2m import paths, protocol, splits
from m2m.config import load_config
from m2m.preprocess import MetabolitePreprocessor, MicrobePreprocessor

CFG = load_config(paths.CONFIGS / "base.yaml")


class SpyMicrobePreprocessor(MicrobePreprocessor):
    """Records which sample ids each fitted instance saw."""

    fitted_on: list[list[str]] = []

    def fit(self, X: pd.DataFrame) -> SpyMicrobePreprocessor:
        SpyMicrobePreprocessor.fitted_on.append(list(X.index))
        return super().fit(X)


def ridge_builder(params: dict, seed: int) -> protocol.M2MPipeline:
    return protocol.make_pipeline(
        SpyMicrobePreprocessor.from_config(CFG["microbes"]),
        MetabolitePreprocessor.from_config(CFG["metabolites"]),
        protocol.SklearnRegressor(Ridge(alpha=params["alpha"])),
        CFG,
        seed,
    )


@pytest.fixture
def folds(synthetic_cohort):
    labels = synthetic_cohort["labels"]
    return splits.make_stratified_folds(list(labels.index), list(labels), n_splits=5, seed=1)


def test_cv_fits_preprocessing_on_training_rows_only(synthetic_cohort, folds):
    SpyMicrobePreprocessor.fitted_on = []
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    protocol.grid_search_cv(ridge_builder, [{"alpha": 1.0}], X, Y, labels, folds, CFG)
    assert len(SpyMicrobePreprocessor.fitted_on) == 5
    for fold, seen in zip(range(5), SpyMicrobePreprocessor.fitted_on, strict=True):
        _, test_ids = splits.fold_train_test(folds, fold)
        assert not set(seen) & set(test_ids)


def test_cv_never_opens_the_external_cohort(synthetic_cohort, folds, monkeypatch):
    import m2m.data as data

    def forbidden(*args, **kwargs):
        raise AssertionError("model selection touched the external cohort")

    monkeypatch.setattr(data, "load_nlibd", forbidden, raising=False)
    monkeypatch.setattr(data, "load_franzosa", forbidden, raising=False)
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    table, _ = protocol.grid_search_cv(
        ridge_builder, protocol.expand_grid({"alpha": [0.1, 10.0]}), X, Y, labels, folds, CFG
    )
    assert len(table) == 2


def test_grid_search_finds_the_planted_signal(synthetic_cohort, folds):
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    table, scores = protocol.grid_search_cv(
        ridge_builder, protocol.expand_grid({"alpha": [1.0]}), X, Y, labels, folds, CFG
    )
    signal = np.mean([s.rho.reindex([f"met_{j:02d}" for j in range(10)]).mean() for s in scores])
    noise = np.mean([s.rho.reindex([f"met_{j:02d}" for j in range(10, 40)]).mean() for s in scores])
    assert signal > noise + 0.2


def test_results_do_not_depend_on_n_jobs(synthetic_cohort, folds):
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    grid = protocol.expand_grid({"alpha": [1.0, 5.0]})
    serial, _ = protocol.grid_search_cv(ridge_builder, grid, X, Y, labels, folds, CFG, n_jobs=1)
    parallel, _ = protocol.grid_search_cv(ridge_builder, grid, X, Y, labels, folds, CFG, n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_select_best_breaks_ties_by_grid_order():
    table = pd.DataFrame(
        {"params": [json.dumps({"a": 1}), json.dumps({"a": 2})], "cv_mean_rho": [0.3, 0.3]}
    )
    assert protocol.select_best(table) == {"a": 1}


def test_external_evaluation_scores_every_seed(synthetic_cohort):
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    train, test = X.index[:45], X.index[45:]
    result = protocol.evaluate_external(
        ridge_builder,
        {"alpha": 1.0},
        X.loc[train],
        Y.loc[train],
        labels.loc[train],
        X.loc[test],
        Y.loc[test],
        CFG,
        seeds=[0, 1],
        null_threshold=0.2,
    )
    assert len(result.per_seed) == 2
    assert list(result.rho_by_seed.columns) == ["rho_seed0", "rho_seed1"]
    agg = result.aggregate
    assert agg["mean_rho_ci_low"] <= agg["mean_rho_mean"] <= agg["mean_rho_ci_high"]
    assert agg["n_seeds"] == 2


def test_external_rejects_overlapping_cohorts(synthetic_cohort):
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    with pytest.raises(splits.LeakageError):
        protocol.evaluate_external(
            ridge_builder,
            {"alpha": 1.0},
            X,
            Y,
            labels,
            X.iloc[:5],
            Y.iloc[:5],
            CFG,
            seeds=[0],
            null_threshold=None,
        )


def test_recording_writes_summary_and_per_metabolite(synthetic_cohort, results_dir):
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    train, test = X.index[:45], X.index[45:]
    result = protocol.evaluate_external(
        ridge_builder,
        {"alpha": 1.0},
        X.loc[train],
        Y.loc[train],
        labels.loc[train],
        X.loc[test],
        Y.loc[test],
        CFG,
        seeds=[0],
        null_threshold=None,
    )
    protocol.record_external(
        result,
        experiment="test",
        model="ridge",
        target_set="TALL",
        data="synthetic",
        params={"alpha": 1.0},
        cfg=CFG,
        n_train=45,
        cv_mean_rho=0.1,
    )
    summary = pd.read_csv(results_dir / "summary.csv")
    assert summary.loc[0, "model"] == "ridge"
    assert (results_dir / "per_metabolite" / "ridge__TALL.csv.gz").exists()
    assert (results_dir / "runs.csv").exists()


def test_null_threshold_round_trip(results_dir):
    from m2m.evaluate import NullResult

    null = NullResult(
        pooled_rho=np.zeros(10),
        threshold=0.23,
        per_permutation_mean=np.array([0.0, 0.01]),
        quantile=0.95,
    )
    protocol.save_null_threshold(data="D1", target_set="T466", null=null, n_features=10)
    assert protocol.load_null_threshold("D1", "T466") == pytest.approx(0.23)
    with pytest.raises(KeyError):
        protocol.load_null_threshold("D2", "T466")


def test_tripwire_rejects_preprocessing_fit_on_other_rows(synthetic_cohort):
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    pipe = ridge_builder({"alpha": 1.0}, 0).fit(X.iloc[:40], Y.iloc[:40], labels.iloc[:40])
    protocol.check_fitted_on(pipe, list(X.index[:40]))  # the rows it was fit on: fine
    with pytest.raises(splits.LeakageError):
        protocol.check_fitted_on(pipe, list(X.index[:39]))


def test_cv_table_records_how_many_features_survived(synthetic_cohort, folds):
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    table, _ = protocol.grid_search_cv(ridge_builder, [{"alpha": 1.0}], X, Y, labels, folds, CFG)
    assert 0 < table.loc[0, "cv_n_input_features"] <= X.shape[1]
    assert table.loc[0, "cv_n_targets_dropped_zero"] >= 0


def test_config_path_names_the_loaded_file_not_its_parent(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nb: 2\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["config_path"] == str(tmp_path / "child.yaml")
    assert cfg["a"] == 1 and cfg["b"] == 2
