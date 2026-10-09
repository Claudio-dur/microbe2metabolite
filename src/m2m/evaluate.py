"""Per-metabolite Spearman ρ, its summaries, bootstrap CI and permutation null (brief section 5).

ρ is computed per metabolite (column) across test samples (rows). A metabolite whose truth or
prediction is constant across the test samples has no defined ρ: `spearman_per_feature` returns
NaN for it, and every summary counts it as ρ = 0 and reports how many there were (`n_nan`).
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

Matrix = np.ndarray | pd.DataFrame
# fit_predict(X_train, Y_train, X_test) -> predictions for X_test, one column per target.
FitPredict = Callable[[Any, Any, Any], Matrix]


# --- per-feature Spearman ---------------------------------------------------------------------


CI_METHODS = ("percentile", "basic")


def spearman_per_feature(y_true: Matrix, y_pred: Matrix) -> np.ndarray:
    """Spearman ρ of each feature (column) across samples (rows); NaN where a column is constant.

    Ranks every column (ties get their average rank), then takes the Pearson correlation of the
    ranks column by column: scipy.stats.spearmanr's definition, vectorised over features.
    Two DataFrames must have identical index and columns (same order); otherwise shapes must
    match. Inputs must be finite: a NaN prediction is an upstream bug, not a ρ of 0.
    """
    truth, pred = _paired_matrices(y_true, y_pred)
    return _pearson_by_column(_centred_ranks(truth), _centred_ranks(pred))


# --- summaries --------------------------------------------------------------------------------


def summarize_rho(
    rho: Sequence[float] | np.ndarray | pd.Series,
    *,
    null_threshold: float | None = None,
    top_k: int = 50,
    rho_threshold: float = 0.5,
) -> dict[str, float | int]:
    """The section-5 statistics of per-feature ρ. NaN ρ count as 0 in every statistic.

    Keys: n_features, n_nan, mean_rho, median_rho, top{top_k}_mean_rho (mean of the top_k largest,
    or of all features if there are fewer), n_rho_gt_{rho_threshold:g}, and n_above_null
    (features with ρ > null_threshold; NaN when the threshold is None or NaN).
    """
    values = np.asarray(rho, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"rho must be a non-empty 1-D array, got shape {values.shape}")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    is_nan = np.isnan(values)
    scored = np.where(is_nan, 0.0, values)
    top = np.sort(scored)[::-1][:top_k]
    # A NaN threshold (e.g. a missing cell in null_thresholds.csv) must not read as "0 above null".
    no_null = null_threshold is None or np.isnan(null_threshold)
    n_above_null = np.nan if no_null else int((scored > null_threshold).sum())
    statistics = [
        int(scored.size),
        int(is_nan.sum()),
        float(scored.mean()),
        float(np.median(scored)),
        float(top.mean()),
        int((scored > rho_threshold).sum()),
        n_above_null,
    ]
    return dict(zip(_summary_keys(top_k, rho_threshold), statistics, strict=True))


def metric_keys(cfg: Mapping[str, Any]) -> list[str]:
    """Names of the `summarize_rho` statistics, in order, for the config's evaluation settings."""
    evaluation = cfg["evaluation"]
    return _summary_keys(evaluation["top_k"], evaluation["rho_threshold"])


def summarize_from_cfg(
    rho: Sequence[float] | np.ndarray | pd.Series,
    cfg: Mapping[str, Any],
    null_threshold: float | None = None,
) -> dict[str, float | int]:
    """`summarize_rho` with top_k and rho_threshold taken from cfg["evaluation"]."""
    evaluation = cfg["evaluation"]
    return summarize_rho(
        rho,
        null_threshold=null_threshold,
        top_k=evaluation["top_k"],
        rho_threshold=evaluation["rho_threshold"],
    )


def aggregate_seeds(summaries: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Mean and SD over seeds of every numeric statistic: {key}_mean and {key}_sd.

    SD uses ddof=1 (sample SD over seeds) and is 0.0 for a single seed (NaN if that seed's value
    is NaN, so a missing statistic never reads as "no spread"). Non-numeric entries (e.g. a model
    name) are skipped. Every summary must have the same numeric keys.
    """
    if not summaries:
        raise ValueError("no seed summaries to aggregate")
    keys = _numeric_keys(summaries[0])
    for summary in summaries[1:]:
        if set(_numeric_keys(summary)) != set(keys):
            raise ValueError("seed summaries do not share the same numeric statistics")
    aggregated: dict[str, float] = {}
    for key in keys:
        values = np.array([float(summary[key]) for summary in summaries])
        aggregated[f"{key}_mean"] = float(values.mean())
        # One seed: population SD, i.e. 0.0 for a finite value and NaN for a NaN one.
        aggregated[f"{key}_sd"] = float(values.std(ddof=1 if len(values) > 1 else 0))
    return aggregated


# --- bootstrap over test samples --------------------------------------------------------------


def bootstrap_mean_rho(
    y_true: Matrix,
    y_pred_by_seed: np.ndarray | Sequence[Matrix],
    *,
    n_boot: int,
    seed: int,
    ci_level: float,
    method: str = "percentile",
) -> dict[str, float]:
    """Bootstrap CI over test samples for the main metric (section 5).

    The metric is the mean over seeds of the mean ρ over features (NaN → 0). Each resample draws
    n test samples with replacement and uses the same draw for every seed, so the interval
    reflects which test samples we happened to have; seed-to-seed spread is reported separately as
    the SD over seeds. Resample b uses row b of
    `np.random.default_rng(seed).integers(0, n_samples, size=(n_boot, n_samples))`.

    `y_pred_by_seed` is a (n_seeds, n_samples, n_features) array or a list of 2-D predictions.
    Returns ci_low/ci_high for `method`, the percentile bounds whatever the method,
    the full-sample estimate, boot_mean and boot_sd (SD of the bootstrap distribution, ddof=1).

    `method="basic"` reflects the bootstrap quantiles around the estimate,
    [2·est − q_hi, 2·est − q_lo], which cancels the first-order downward bias that tied
    resamples put on Spearman ρ (see the caveat below); `"percentile"` is [q_lo, q_hi].

    Caveat: resamples contain duplicated samples (ties), which shrink Spearman ρ toward 0 by
    roughly 0.005-0.01 at 65 samples. Averaging over many *independent* features shrinks the
    bootstrap SD but not this shift, so the percentile interval can then sit entirely below the
    point estimate. Real metabolites co-vary across samples, which keeps the SD up; with
    correlated synthetic features (65 samples, mean ρ ≈ 0.3) the shift was about 0.3 SD and the
    interval contained the estimate. tests/test_evaluate.py pins down both cases.
    """
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be strictly between 0 and 1, got {ci_level}")
    if method not in CI_METHODS:
        raise ValueError(f"method must be one of {CI_METHODS}, got {method!r}")
    truth = _as_matrix(y_true, "y_true")
    predictions = _seed_predictions(y_true, y_pred_by_seed, truth.shape)
    draws = _bootstrap_draws(seed, n_boot, truth.shape[0])
    metric = _bootstrap_metric(truth, predictions, draws)
    estimate = _seed_mean_rho(truth, predictions)
    q_low, q_high, ci_low, ci_high = _interval(metric, estimate, ci_level, method)
    return {
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_low_percentile": float(q_low),
        "ci_high_percentile": float(q_high),
        "estimate": estimate,
        "boot_mean": float(metric.mean()),
        "boot_sd": float(metric.std(ddof=1)),
    }


def paired_bootstrap_delta(
    y_true: Matrix,
    y_pred_a: np.ndarray | Sequence[Matrix],
    y_pred_b: np.ndarray | Sequence[Matrix],
    *,
    n_boot: int,
    seed: int,
    ci_level: float,
    method: str = "basic",
) -> dict[str, float]:
    """Paired bootstrap of Δ = metric(A) − metric(B) on the same test samples.

    Each resample draws test samples with replacement and scores BOTH models on those same
    samples, so sample-to-sample variation common to the two models cancels: this is the test of
    "does pretraining beat the same architecture from scratch?" (H1) on the external cohort.
    The metric is `bootstrap_mean_rho`'s (mean over seeds of the mean ρ, NaN → 0); `p_not_better`
    is the share of resamples with Δ ≤ 0 (a one-sided bootstrap p-value for "A beats B").
    """
    if method not in CI_METHODS:
        raise ValueError(f"method must be one of {CI_METHODS}, got {method!r}")
    truth = _as_matrix(y_true, "y_true")
    preds_a = _seed_predictions(y_true, y_pred_a, truth.shape)
    preds_b = _seed_predictions(y_true, y_pred_b, truth.shape)
    draws = _bootstrap_draws(seed, n_boot, truth.shape[0])
    delta = _bootstrap_metric(truth, preds_a, draws) - _bootstrap_metric(truth, preds_b, draws)
    estimate = _seed_mean_rho(truth, preds_a) - _seed_mean_rho(truth, preds_b)
    q_low, q_high, ci_low, ci_high = _interval(delta, estimate, ci_level, method)
    return {
        "delta": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_low_percentile": float(q_low),
        "ci_high_percentile": float(q_high),
        "boot_sd": float(delta.std(ddof=1)),
        "p_not_better": float(np.mean(delta <= 0.0)),
    }


def _bootstrap_draws(seed: int, n_boot: int, n_samples: int) -> np.ndarray:
    """Row b = the test-sample indices of resample b (shared by every model scored with seed)."""
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}")
    return np.random.default_rng(seed).integers(0, n_samples, size=(n_boot, n_samples))


def _bootstrap_metric(
    truth: np.ndarray, predictions: Sequence[np.ndarray], draws: np.ndarray
) -> np.ndarray:
    """Mean over seeds of the mean ρ (NaN → 0) on every resample in `draws`."""
    n_samples = truth.shape[0]
    truth_ranker = _ResampleRanker(truth)
    prediction_rankers = [_ResampleRanker(pred) for pred in predictions]
    metric = np.empty(len(draws))
    for b, draw in enumerate(draws):
        counts = np.bincount(draw, minlength=n_samples).astype(float)
        truth_centred = _weighted_centre(truth_ranker.ranks(counts), counts)
        mean_rho_by_seed = [
            _resample_mean_rho(truth_centred, ranker.ranks(counts), counts)
            for ranker in prediction_rankers
        ]
        metric[b] = np.mean(mean_rho_by_seed)
    return metric


def _seed_mean_rho(truth: np.ndarray, predictions: Sequence[np.ndarray]) -> float:
    return float(np.mean([
        np.nan_to_num(spearman_per_feature(truth, pred), nan=0.0).mean() for pred in predictions
    ]))


def _interval(
    values: np.ndarray, estimate: float, ci_level: float, method: str
) -> tuple[float, float, float, float]:
    """(percentile low, percentile high, CI low, CI high) for the chosen method."""
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be strictly between 0 and 1, got {ci_level}")
    tail = (1.0 - ci_level) / 2.0
    q_low, q_high = np.quantile(values, [tail, 1.0 - tail])
    if method == "basic":
        return q_low, q_high, 2 * estimate - q_high, 2 * estimate - q_low
    return q_low, q_high, q_low, q_high


# --- permutation null -------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class NullResult:
    """Permutation-null ρ and the "above chance" threshold derived from it."""

    pooled_rho: np.ndarray  # ρ of every permutation × feature (NaN → 0), permutation-major
    threshold: float  # the `quantile` of pooled_rho
    per_permutation_mean: np.ndarray  # mean ρ of each permutation; should sit near 0
    quantile: float


def permutation_null(
    fit_predict: FitPredict,
    X_train: Matrix,
    Y_train: Matrix,
    X_test: Matrix,
    Y_test_eval: Matrix,
    *,
    n_permutations: int,
    seed: int,
    quantile: float,
) -> NullResult:
    """Chance level of per-feature ρ, by breaking the microbe-metabolite pairing in training.

    Permutation p shuffles the rows of X_train with np.random.default_rng([seed, p]) and then
    restores X_train's index, so the sample IDs still line up with Y_train while each metabolite
    profile now sits next to another person's microbiome. Y_train and the test data are never
    changed. `fit_predict(X_perm, Y_train, X_test)` must refit everything that a real run fits
    (preprocessing included) and return predictions for X_test aligned with Y_test_eval's
    columns. The ρ of all permutations are pooled (NaN → 0); the pool's `quantile` is the threshold.
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    _check_training_pairs(X_train, Y_train)
    rho_by_permutation = []
    for p in range(n_permutations):
        rng = np.random.default_rng([seed, p])
        X_permuted = _shuffle_rows_keep_index(X_train, rng.permutation(len(X_train)))
        predictions = fit_predict(X_permuted, Y_train, X_test)
        rho_by_permutation.append(spearman_per_feature(Y_test_eval, predictions))
    rho = np.nan_to_num(np.vstack(rho_by_permutation), nan=0.0)
    pooled = rho.ravel()
    return NullResult(
        pooled_rho=pooled,
        threshold=float(np.quantile(pooled, quantile)),
        per_permutation_mean=rho.mean(axis=1),
        quantile=float(quantile),
    )


# --- helpers: input checks --------------------------------------------------------------------


def _as_matrix(values: Matrix, what: str) -> np.ndarray:
    """A finite 2-D float array (samples × features) with >= 2 samples and >= 1 feature."""
    if isinstance(values, pd.DataFrame):
        array = values.to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(
            f"{what} must be 2-D (samples × features) with at least 2 samples and 1 feature; "
            f"got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{what} contains NaN or infinite values")
    return array


def _check_same_labels(y_true: Matrix, y_pred: Matrix) -> None:
    """Two DataFrames must describe the same samples and features, in the same order."""
    if isinstance(y_true, pd.DataFrame) and isinstance(y_pred, pd.DataFrame):
        if not y_true.index.equals(y_pred.index):
            raise ValueError("y_true and y_pred have different sample indexes (or order)")
        if not y_true.columns.equals(y_pred.columns):
            raise ValueError("y_true and y_pred have different feature columns (or order)")


def _paired_matrices(y_true: Matrix, y_pred: Matrix) -> tuple[np.ndarray, np.ndarray]:
    _check_same_labels(y_true, y_pred)
    truth, pred = _as_matrix(y_true, "y_true"), _as_matrix(y_pred, "y_pred")
    if truth.shape != pred.shape:
        raise ValueError(f"y_true {truth.shape} and y_pred {pred.shape} differ in shape")
    return truth, pred


def _seed_predictions(
    y_true: Matrix, y_pred_by_seed: np.ndarray | Sequence[Matrix], shape: tuple[int, ...]
) -> list[np.ndarray]:
    """Per-seed prediction matrices, each checked against y_true."""
    if isinstance(y_pred_by_seed, pd.DataFrame) or (
        isinstance(y_pred_by_seed, np.ndarray) and y_pred_by_seed.ndim != 3
    ):
        raise ValueError(
            "y_pred_by_seed must be a (n_seeds, n_samples, n_features) array or a list of "
            "per-seed 2-D predictions (wrap a single seed's predictions in a list)"
        )
    predictions = []
    for s, pred in enumerate(y_pred_by_seed):
        _check_same_labels(y_true, pred)
        matrix = _as_matrix(pred, f"y_pred_by_seed[{s}]")
        if matrix.shape != shape:
            raise ValueError(f"y_pred_by_seed[{s}] has shape {matrix.shape}, y_true {shape}")
        predictions.append(matrix)
    if not predictions:
        raise ValueError("y_pred_by_seed is empty")
    return predictions


def _check_training_pairs(X_train: Matrix, Y_train: Matrix) -> None:
    """The unpermuted training data must pair each microbiome with its own metabolome."""
    if len(X_train) != len(Y_train):
        raise ValueError(f"X_train has {len(X_train)} rows but Y_train has {len(Y_train)}")
    both_frames = isinstance(X_train, pd.DataFrame) and isinstance(Y_train, pd.DataFrame)
    if both_frames and not X_train.index.equals(Y_train.index):
        raise ValueError("X_train and Y_train must list the same samples in the same order")


def _numeric_keys(summary: Mapping[str, Any]) -> list[str]:
    return [
        key
        for key, value in summary.items()
        if isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)
    ]


def _summary_keys(top_k: int, rho_threshold: float) -> list[str]:
    """Single source of the summary names, so tables and summaries cannot drift apart."""
    return [
        "n_features",
        "n_nan",
        "mean_rho",
        "median_rho",
        f"top{top_k}_mean_rho",
        f"n_rho_gt_{rho_threshold:g}",
        "n_above_null",
    ]


def _shuffle_rows_keep_index(X: Matrix, order: np.ndarray) -> Matrix:
    """Rows reordered by `order`; a DataFrame keeps its original index (row labels stay put)."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[order].set_axis(X.index, axis=0)
    return np.asarray(X)[order]


# --- helpers: correlation ---------------------------------------------------------------------


def _centred_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks of each column (ties share their mean rank), minus the column mean."""
    ranks = rankdata(values, method="average", axis=0)
    return ranks - ranks.mean(axis=0)


def _pearson_by_column(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r per column of two column-centred matrices; NaN where a column has no spread."""
    covariance = (a * b).sum(axis=0)
    scale = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return _correlation(covariance, scale)


def _correlation(covariance: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """covariance / scale, NaN where scale is 0 (computed without divide-by-zero warnings)."""
    rho = np.full(covariance.shape, np.nan)
    defined = scale > 0
    rho[defined] = covariance[defined] / scale[defined]
    # Rounding can push |r| a hair above 1; clip as scipy does.
    return np.clip(rho, -1.0, 1.0)


# The bootstrap helpers below work feature-major (rows = features, columns = samples) so that every
# per-feature cumulative sum and gather runs along contiguous memory (about 1.6x faster).


def _weighted_centre(ranks: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Subtract each row's weighted mean (sample j counts weights[j] times)."""
    return ranks - (ranks @ weights)[:, None] / weights.sum()


def _resample_mean_rho(
    truth_centred: np.ndarray, pred_ranks: np.ndarray, counts: np.ndarray
) -> float:
    """Mean over features of ρ in one resample (NaN → 0); sample j is repeated counts[j] times."""
    pred_centred = _weighted_centre(pred_ranks, counts)
    covariance = (truth_centred * pred_centred) @ counts
    truth_spread = (truth_centred * truth_centred) @ counts
    pred_spread = (pred_centred * pred_centred) @ counts
    rho = _correlation(covariance, np.sqrt(truth_spread * pred_spread))
    return float(np.nan_to_num(rho, nan=0.0).mean())


class _ResampleRanker:
    """Average ranks of each feature inside bootstrap resamples, without re-sorting.

    A resample is given by draw counts (sample j drawn counts[j] times). Inside it, every copy of
    a value v ranks at (number of drawn values < v) + (number of drawn values == v, plus 1) / 2,
    exactly what ranking the resampled rows would give. Once each feature is sorted, both numbers
    are read off a cumulative sum of the counts, so a resample costs a few vectorised passes
    instead of a fresh sort (3-4x faster than scipy's rankdata at 65 samples × 8,848 features).
    Samples that were not drawn also get a rank, but they carry zero weight in the correlation.
    """

    def __init__(self, values: np.ndarray) -> None:
        rows = np.ascontiguousarray(values.T)  # features × samples
        n_features, n_samples = rows.shape
        order = np.argsort(rows, axis=1, kind="stable")  # sorted position -> sample
        tie_first, tie_last = _tie_runs(np.take_along_axis(rows, order, axis=1))
        # Flat indices let one 1-D gather replace np.take_along_axis (about 3x faster).
        cumulative_row = (np.arange(n_features) * (n_samples + 1))[:, None]
        rank_row = (np.arange(n_features) * n_samples)[:, None]
        self._order = order
        self._below_at = (tie_first + cumulative_row).ravel()
        self._through_at = (tie_last + 1 + cumulative_row).ravel()
        self._unsort_at = (np.argsort(order, axis=1) + rank_row).ravel()
        # Column c holds the total count of sorted positions < c (column 0 stays 0).
        self._cumulative = np.zeros((n_features, n_samples + 1))
        self._shape = rows.shape

    def ranks(self, counts: np.ndarray) -> np.ndarray:
        """(n_features, n_samples) average rank of each sample's value within the resample."""
        np.cumsum(counts[self._order], axis=1, out=self._cumulative[:, 1:])
        cumulative = self._cumulative.ravel()
        below = cumulative[self._below_at]  # drawn values smaller than this run of ties
        through = cumulative[self._through_at]  # drawn values up to the end of the run
        ranks_sorted = 0.5 * (below + through + 1.0)
        return ranks_sorted[self._unsort_at].reshape(self._shape)


def _tie_runs(sorted_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last position of each entry's run of equal values, per row of a sorted matrix."""
    n_columns = sorted_rows.shape[1]
    position = np.arange(n_columns)
    starts_run = np.ones(sorted_rows.shape, dtype=bool)
    starts_run[:, 1:] = sorted_rows[:, 1:] != sorted_rows[:, :-1]
    ends_run = np.ones(sorted_rows.shape, dtype=bool)
    ends_run[:, :-1] = starts_run[:, 1:]
    first = np.maximum.accumulate(np.where(starts_run, position, 0), axis=1)
    last_reversed = np.where(ends_run, position, n_columns - 1)[:, ::-1]
    last = np.minimum.accumulate(last_reversed, axis=1)[:, ::-1]
    return first, last
