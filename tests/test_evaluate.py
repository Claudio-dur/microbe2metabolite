"""Tests for m2m.evaluate: Spearman vs scipy, summaries, bootstrap CI and permutation null."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge

from m2m import paths
from m2m.config import load_config
from m2m.evaluate import (
    NullResult,
    _ResampleRanker,
    aggregate_seeds,
    bootstrap_mean_rho,
    metric_keys,
    permutation_null,
    spearman_per_feature,
    summarize_from_cfg,
    summarize_rho,
)


def scipy_rho(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Reference: scipy.stats.spearmanr one column at a time."""
    return np.array(
        [spearmanr(y_true[:, j], y_pred[:, j]).statistic for j in range(y_true.shape[1])]
    )


# --- spearman_per_feature ---------------------------------------------------------------------


def test_spearman_matches_scipy_with_ties(rng: np.random.Generator) -> None:
    y_true = rng.integers(0, 5, size=(40, 25)).astype(float)
    y_pred = rng.integers(0, 4, size=(40, 25)) + 0.5 * y_true
    expected = scipy_rho(y_true, y_pred)
    assert np.isfinite(expected).all()
    actual = spearman_per_feature(y_true, y_pred)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_spearman_matches_scipy_zero_inflated(rng: np.random.Generator) -> None:
    n, m = 65, 60
    y_true = rng.lognormal(size=(n, m)) * (rng.random((n, m)) > 0.7)  # ~70% zeros, like LC-MS
    y_pred = np.log1p(y_true) + rng.normal(scale=0.5, size=(n, m))
    y_pred[rng.random((n, m)) < 0.3] = 0.0
    expected = scipy_rho(y_true, y_pred)
    assert np.isfinite(expected).all()
    actual = spearman_per_feature(y_true, y_pred)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_spearman_constant_column_is_nan_without_warnings(rng: np.random.Generator) -> None:
    y_true = rng.normal(size=(20, 4))
    y_pred = rng.normal(size=(20, 4))
    y_true[:, 1] = 3.0  # e.g. a metabolite that is zero in every test sample
    y_pred[:, 2] = 0.0  # e.g. a model that predicts the training mean
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rho = spearman_per_feature(y_true, y_pred)
    assert np.isnan(rho[[1, 2]]).all()
    np.testing.assert_allclose(rho[[0, 3]], scipy_rho(y_true[:, [0, 3]], y_pred[:, [0, 3]]))


def test_spearman_dataframes_must_share_labels(rng: np.random.Generator) -> None:
    truth = pd.DataFrame(
        rng.normal(size=(10, 3)), index=[f"s{i}" for i in range(10)], columns=["a", "b", "c"]
    )
    pred = 2.0 * truth + 1.0  # same ranks → ρ = 1
    np.testing.assert_allclose(spearman_per_feature(truth, pred), 1.0)
    np.testing.assert_allclose(spearman_per_feature(truth, pred.to_numpy()), 1.0)
    with pytest.raises(ValueError, match="columns"):
        spearman_per_feature(truth, pred[["b", "a", "c"]])
    with pytest.raises(ValueError, match="index"):
        spearman_per_feature(truth, pred.iloc[::-1])


def test_spearman_rejects_bad_input(rng: np.random.Generator) -> None:
    y = rng.normal(size=(10, 3))
    with_nan = y.copy()
    with_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        spearman_per_feature(y, with_nan)
    with pytest.raises(ValueError, match="shape"):
        spearman_per_feature(y, y[:, :2])
    with pytest.raises(ValueError, match="2 samples"):
        spearman_per_feature(y[:1], y[:1])


# --- summaries --------------------------------------------------------------------------------


def test_summarize_rho_hand_computed() -> None:
    # NaN → 0 gives [0.9, 0.6, 0, 0.2, -0.1]; dropping the NaN instead would give mean 0.4.
    summary = summarize_rho(
        [0.9, 0.6, np.nan, 0.2, -0.1], null_threshold=0.15, top_k=2, rho_threshold=0.5
    )
    assert summary == {
        "n_features": 5,
        "n_nan": 1,
        "mean_rho": pytest.approx(0.32),
        "median_rho": pytest.approx(0.2),
        "top2_mean_rho": pytest.approx(0.75),
        "n_rho_gt_0.5": 2,  # strictly greater
        "n_above_null": 3,
    }


def test_summarize_rho_nan_counts_as_zero_in_top_k() -> None:
    summary = summarize_rho([np.nan, -0.2, -0.5], top_k=1)
    assert summary["top1_mean_rho"] == 0.0
    assert summary["n_nan"] == 1


def test_summarize_rho_uses_all_features_when_fewer_than_top_k() -> None:
    summary = summarize_rho(np.array([0.1, 0.4, 0.7]), top_k=50, rho_threshold=0.25)
    assert summary["top50_mean_rho"] == pytest.approx(0.4)
    assert summary["n_rho_gt_0.25"] == 2
    assert np.isnan(summary["n_above_null"])  # no null threshold given
    # A missing (NaN) threshold must not read as "no feature beats the null".
    assert np.isnan(summarize_rho([0.1, 0.9], null_threshold=float("nan"))["n_above_null"])


def test_summarize_rho_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize_rho([])


def test_metric_keys_match_summary_and_base_config() -> None:
    cfg = load_config(paths.CONFIGS / "base.yaml")
    summary = summarize_from_cfg(np.array([0.3, np.nan, 0.6]), cfg, null_threshold=0.2)
    assert list(summary) == metric_keys(cfg)
    assert metric_keys(cfg) == [
        "n_features",
        "n_nan",
        "mean_rho",
        "median_rho",
        "top50_mean_rho",
        "n_rho_gt_0.5",
        "n_above_null",
    ]
    assert summary["n_above_null"] == 2


def test_aggregate_seeds_mean_and_sd() -> None:
    summaries = [
        {"model": "ridge", "mean_rho": 0.2, "n_nan": 1},
        {"model": "ridge", "mean_rho": 0.4, "n_nan": 3},
    ]
    assert aggregate_seeds(summaries) == {
        "mean_rho_mean": pytest.approx(0.3),
        "mean_rho_sd": pytest.approx(np.sqrt(0.02)),  # ddof=1
        "n_nan_mean": pytest.approx(2.0),
        "n_nan_sd": pytest.approx(np.sqrt(2.0)),
    }
    single = aggregate_seeds(summaries[:1])
    assert single["mean_rho_mean"] == pytest.approx(0.2)
    assert single["mean_rho_sd"] == 0.0


def test_aggregate_seeds_nan_statistics_and_mismatches() -> None:
    summaries = [summarize_rho([0.1, 0.3]), summarize_rho([0.2, 0.5])]  # n_above_null is NaN
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aggregated = aggregate_seeds(summaries)
    assert np.isnan(aggregated["n_above_null_mean"])
    assert aggregated["mean_rho_mean"] == pytest.approx(0.275)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        single = aggregate_seeds(summaries[:1])
    assert single["mean_rho_sd"] == 0.0
    assert np.isnan(single["n_above_null_sd"])  # a missing statistic has no SD, not SD 0
    with pytest.raises(ValueError, match="same numeric"):
        aggregate_seeds([{"mean_rho": 0.1}, {"median_rho": 0.1}])
    with pytest.raises(ValueError):
        aggregate_seeds([])


# --- bootstrap --------------------------------------------------------------------------------


def _strong_signal(rng: np.random.Generator, n: int = 40, m: int = 5, n_seeds: int = 3):
    """Truth and per-seed predictions with population Spearman ρ ≈ 0.88 per feature."""
    y_true = rng.lognormal(size=(n, m))
    preds = np.stack(
        [np.log(y_true) + rng.normal(scale=0.5, size=(n, m)) for _ in range(n_seeds)]
    )
    return y_true, preds


def _seed_mean_rho(y_true: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean([np.nan_to_num(spearman_per_feature(y_true, p)).mean() for p in preds]))


def test_bootstrap_is_deterministic_and_contains_estimate(rng: np.random.Generator) -> None:
    y_true, preds = _strong_signal(rng)
    first = bootstrap_mean_rho(y_true, preds, n_boot=300, seed=3, ci_level=0.95)
    assert first == bootstrap_mean_rho(y_true, preds, n_boot=300, seed=3, ci_level=0.95)
    assert first == bootstrap_mean_rho(y_true, list(preds), n_boot=300, seed=3, ci_level=0.95)
    assert first["ci_low"] < _seed_mean_rho(y_true, preds) < first["ci_high"]
    assert first["boot_sd"] > 0
    assert first != bootstrap_mean_rho(y_true, preds, n_boot=300, seed=4, ci_level=0.95)


def test_percentile_ci_is_shifted_toward_zero_for_many_strong_features(
    rng: np.random.Generator,
) -> None:
    """Documents a property of the percentile bootstrap for Spearman ρ, not a bug.

    Resamples contain duplicated samples (ties), which shrink ρ toward 0 by ~0.01 here. Averaging
    over many *independent* features shrinks the bootstrap SD but not that shift, so with strong
    signal and hundreds of such features the whole percentile interval sits below the estimate.
    """
    y_true, preds = _strong_signal(rng, n=40, m=300, n_seeds=1)
    result = bootstrap_mean_rho(y_true, preds, n_boot=200, seed=0, ci_level=0.95)
    assert result["ci_high"] < _seed_mean_rho(y_true, preds)


def test_percentile_ci_contains_estimate_for_correlated_features(
    rng: np.random.Generator,
) -> None:
    """Counterpart of the test above with metabolites that co-vary, as real ones do.

    Shared sample-level variation keeps the bootstrap SD up however many features are averaged
    (here ~0.03 against ~0.004 for 300 independent features), so the tie shift stays small
    relative to the interval.
    """
    n, m, n_factors = 65, 300, 3
    factors = rng.normal(size=(n, n_factors))
    loadings = rng.normal(size=(n_factors, m))
    y_true = np.exp(factors @ loadings + rng.normal(size=(n, m)))
    noisy_factors = factors + rng.normal(size=factors.shape)  # errors shared by all metabolites
    preds = np.stack([noisy_factors @ loadings + rng.normal(size=(n, m)) for _ in range(2)])
    result = bootstrap_mean_rho(y_true, preds, n_boot=200, seed=0, ci_level=0.95)
    assert result["ci_low"] < _seed_mean_rho(y_true, preds) < result["ci_high"]
    assert result["boot_sd"] > 0.01


def test_bootstrap_matches_explicit_resampling(rng: np.random.Generator) -> None:
    """The fast path must equal re-ranking every resample, ties and constant columns included."""
    n, m, n_seeds, n_boot, seed = 25, 12, 3, 60, 9
    y_true = np.round(rng.lognormal(size=(n, m)), 1)
    y_true[rng.random((n, m)) < 0.5] = 0.0
    y_true[:, 0] = 2.0  # constant truth
    preds = np.round(y_true[None] + rng.normal(size=(n_seeds, n, m)), 0)
    preds[1, :, 3] = 0.5  # one seed predicts a constant
    draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    metric = np.array(
        [
            np.mean([np.nan_to_num(spearman_per_feature(y_true[d], p[d])).mean() for p in preds])
            for d in draws
        ]
    )
    result = bootstrap_mean_rho(y_true, preds, n_boot=n_boot, seed=seed, ci_level=0.9)
    np.testing.assert_allclose(
        [result["ci_low"], result["ci_high"]], np.quantile(metric, [0.05, 0.95]), rtol=1e-12
    )
    assert result["boot_sd"] == pytest.approx(metric.std(ddof=1), rel=1e-10)


def test_resample_ranker_equals_rankdata_on_resampled_rows(rng: np.random.Generator) -> None:
    values = np.round(rng.normal(size=(30, 7)), 0)  # many ties
    values[:, 2] = 1.0
    draw = rng.integers(0, 30, size=30)
    counts = np.bincount(draw, minlength=30).astype(float)
    ranks = _ResampleRanker(values).ranks(counts)  # features × samples
    # resampled row k is a copy of sample draw[k]
    np.testing.assert_allclose(ranks.T[draw], rankdata(values[draw], axis=0))


def test_bootstrap_rejects_ambiguous_predictions(rng: np.random.Generator) -> None:
    y_true, preds = _strong_signal(rng, n=10, m=4, n_seeds=2)
    with pytest.raises(ValueError, match="list"):
        bootstrap_mean_rho(y_true, preds[0], n_boot=10, seed=0, ci_level=0.95)
    with pytest.raises(ValueError, match="list"):
        bootstrap_mean_rho(
            pd.DataFrame(y_true), pd.DataFrame(preds[0]), n_boot=10, seed=0, ci_level=0.95
        )
    with pytest.raises(ValueError, match="shape"):
        bootstrap_mean_rho(y_true, preds[:, :, :3], n_boot=10, seed=0, ci_level=0.95)


# --- permutation null -------------------------------------------------------------------------


def _linear_problem(rng: np.random.Generator, n_train: int = 150, n_test: int = 65):
    """Microbiome-like X and metabolites that depend linearly on a few features each."""
    n_features, n_targets = 40, 30
    weights = np.zeros((n_features, n_targets))
    for j in range(n_targets):
        weights[rng.choice(n_features, size=3, replace=False), j] = rng.normal(size=3)
    ids = [f"S|{i:03d}" for i in range(n_train + n_test)]
    X = pd.DataFrame(rng.normal(size=(len(ids), n_features)), index=ids).add_prefix("taxon_")
    Y = pd.DataFrame(
        X.to_numpy() @ weights + 0.3 * rng.normal(size=(len(ids), n_targets)), index=ids
    ).add_prefix("met_")
    train, test = ids[:n_train], ids[n_train:]
    return X.loc[train], Y.loc[train], X.loc[test], Y.loc[test]


def ridge_fit_predict(X: pd.DataFrame, Y: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    model = Ridge(alpha=1.0).fit(X.to_numpy(), Y.to_numpy())
    return pd.DataFrame(model.predict(X_test.to_numpy()), index=X_test.index, columns=Y.columns)


def test_permutation_null_is_far_below_real_signal(rng: np.random.Generator) -> None:
    X_train, Y_train, X_test, Y_test = _linear_problem(rng)
    real = spearman_per_feature(Y_test, ridge_fit_predict(X_train, Y_train, X_test))
    null = permutation_null(
        ridge_fit_predict,
        X_train,
        Y_train,
        X_test,
        Y_test,
        n_permutations=10,
        seed=7,
        quantile=0.95,
    )
    assert isinstance(null, NullResult)
    assert null.pooled_rho.shape == (10 * Y_test.shape[1],)
    assert null.per_permutation_mean.shape == (10,)
    assert null.threshold == pytest.approx(np.quantile(null.pooled_rho, 0.95))
    assert null.quantile == 0.95
    assert abs(null.per_permutation_mean.mean()) < 0.1  # chance level ≈ 0
    assert real.mean() > null.threshold + 0.3


def test_permutation_shuffles_only_training_microbiome_rows(rng: np.random.Generator) -> None:
    X_train, Y_train, X_test, Y_test = _linear_problem(rng, n_train=30, n_test=20)
    Y_before, X_before = Y_train.copy(), X_train.copy()
    calls: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []

    def spy(X_perm: pd.DataFrame, Y: pd.DataFrame, X_te: pd.DataFrame) -> pd.DataFrame:
        calls.append((X_perm, Y, X_te))
        return ridge_fit_predict(X_perm, Y, X_te)

    kwargs = {"seed": 5, "quantile": 0.95}
    permutation_null(spy, X_train, Y_train, X_test, Y_test, n_permutations=3, **kwargs)
    assert len(calls) == 3
    for X_perm, Y, X_te in calls:
        assert Y is Y_train and X_te is X_test
        pd.testing.assert_index_equal(X_perm.index, X_train.index)  # IDs still align with Y
        same_rows = sorted(map(tuple, X_perm.to_numpy())) == sorted(map(tuple, X_train.to_numpy()))
        assert same_rows  # the same microbiomes ...
        assert not np.array_equal(X_perm.to_numpy(), X_train.to_numpy())  # ... re-paired
    pd.testing.assert_frame_equal(Y_train, Y_before)
    pd.testing.assert_frame_equal(X_train, X_before)
    assert not calls[0][0].equals(calls[1][0])

    # Permutation p depends only on (seed, p): a shorter run repeats the first permutations.
    first_two = [X_perm for X_perm, _, _ in calls[:2]]
    calls.clear()
    permutation_null(spy, X_train, Y_train, X_test, Y_test, n_permutations=2, **kwargs)
    for earlier, again in zip(first_two, [X_perm for X_perm, _, _ in calls], strict=True):
        pd.testing.assert_frame_equal(earlier, again)


def test_permutation_null_accepts_arrays_and_checks_pairing(rng: np.random.Generator) -> None:
    X_train, Y_train, X_test, Y_test = _linear_problem(rng, n_train=40, n_test=20)

    def array_fit_predict(X: np.ndarray, Y: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        return Ridge(alpha=1.0).fit(X, Y).predict(X_te)

    kwargs = {"n_permutations": 4, "seed": 1, "quantile": 0.95}
    from_frames = permutation_null(ridge_fit_predict, X_train, Y_train, X_test, Y_test, **kwargs)
    from_arrays = permutation_null(
        array_fit_predict,
        X_train.to_numpy(),
        Y_train.to_numpy(),
        X_test.to_numpy(),
        Y_test.to_numpy(),
        **kwargs,
    )
    np.testing.assert_allclose(from_arrays.pooled_rho, from_frames.pooled_rho)
    with pytest.raises(ValueError, match="same samples"):
        permutation_null(
            ridge_fit_predict, X_train, Y_train.iloc[::-1], X_test, Y_test, **kwargs
        )


def test_basic_bootstrap_reflects_the_percentile_interval_around_the_estimate() -> None:
    rng = np.random.default_rng(3)
    latent = rng.normal(size=(65, 1))
    truth = latent + rng.normal(scale=1.0, size=(65, 40))
    preds = np.stack([latent + rng.normal(scale=1.0, size=(65, 40)) for _ in range(2)])
    kwargs = dict(n_boot=300, seed=5, ci_level=0.9)
    pct = bootstrap_mean_rho(truth, preds, method="percentile", **kwargs)
    basic = bootstrap_mean_rho(truth, preds, method="basic", **kwargs)
    est = basic["estimate"]
    assert basic["ci_low"] == pytest.approx(2 * est - pct["ci_high"])
    assert basic["ci_high"] == pytest.approx(2 * est - pct["ci_low"])
    assert basic["ci_low"] <= est <= basic["ci_high"]
    # ties in resamples pull ρ toward 0, so the bootstrap mean sits below the estimate
    assert basic["boot_mean"] < est
    with pytest.raises(ValueError, match="method"):
        bootstrap_mean_rho(truth, preds, method="bca", **kwargs)


def test_paired_bootstrap_detects_a_better_model_and_not_an_identical_one() -> None:
    from m2m.evaluate import paired_bootstrap_delta

    rng = np.random.default_rng(4)
    latent = rng.normal(size=(65, 1))
    truth = latent + rng.normal(scale=1.0, size=(65, 30))
    good = np.stack([latent + rng.normal(scale=0.5, size=(65, 30)) for _ in range(3)])
    weak = np.stack([latent + rng.normal(scale=3.0, size=(65, 30)) for _ in range(3)])
    kwargs = dict(n_boot=300, seed=2, ci_level=0.95)
    better = paired_bootstrap_delta(truth, good, weak, **kwargs)
    assert better["delta"] > 0 and better["ci_low"] > 0 and better["p_not_better"] < 0.05
    same = paired_bootstrap_delta(truth, good, good, **kwargs)
    assert same["delta"] == 0.0 and same["ci_low"] == same["ci_high"] == 0.0
    # The paired delta agrees with the difference of the two unpaired estimates.
    a = bootstrap_mean_rho(truth, good, **kwargs)["estimate"]
    b = bootstrap_mean_rho(truth, weak, **kwargs)["estimate"]
    assert better["delta"] == pytest.approx(a - b)
