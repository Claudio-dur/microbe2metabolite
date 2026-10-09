"""Shared experiment machinery: the one place where train/test separation is enforced.

Every experiment wraps its model in an `M2MPipeline` (preprocessing + regressor) and hands a
builder to `grid_search_cv` / `cross_validate` (PRISM folds only) or `evaluate_external` (fit on
all of PRISM, score once on the external cohort). Preprocessing is always fit inside the pipeline,
on the rows it is trained on, so no fold or cohort can leak into another's statistics.
"""

from __future__ import annotations

import itertools
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from m2m import evaluate, paths, splits, tracking

NULL_THRESHOLDS_CSV = "null_thresholds.csv"

# loky recycles a worker after large jobs (the TALL-sized fits) and says so with this warning;
# every job still completes and results are unchanged, so it is noise in every experiment log.
warnings.filterwarnings(
    "ignore", message="A worker stopped while some jobs were given to the executor"
)


class Regressor(Protocol):
    """fit/predict on numpy arrays; `needs_validation` requests an early-stopping split."""

    needs_validation: bool

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
    ) -> Any: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class SklearnRegressor:
    """Adapter for scikit-learn estimators, which train on every row they are given."""

    needs_validation = False

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, Y: np.ndarray, X_val=None, Y_val=None) -> SklearnRegressor:
        self.estimator.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=float).reshape(len(X), -1)


@dataclass
class M2MPipeline:
    """Preprocessing fit on the training rows + a regressor, operating on raw DataFrames.

    Both preprocessors are fit on the full training set passed to `fit`, *including* the rows later
    held out for early stopping: they are training rows, and this keeps the evaluated metabolite
    set identical across models (DESIGN.md decision #2).
    """

    x_prep: Any
    y_prep: Any
    regressor: Regressor
    inner_fraction: float = 0.2
    inner_seed: int = 0

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, labels: pd.Series | None = None) -> M2MPipeline:
        _check_aligned(X, Y)
        X_t = self.x_prep.fit(X).transform(X)
        Y_t = self.y_prep.fit(Y).transform(Y)
        if not self.regressor.needs_validation:
            self.regressor.fit(X_t.to_numpy(), Y_t.to_numpy())
            return self
        if labels is None:
            raise ValueError("this regressor needs an early-stopping split: pass diagnosis labels")
        fit_ids, val_ids = splits.stratified_inner_split(
            list(X.index),
            list(labels.loc[X.index]),
            fraction=self.inner_fraction,
            seed=self.inner_seed,
        )
        self.regressor.fit(
            X_t.loc[fit_ids].to_numpy(),
            Y_t.loc[fit_ids].to_numpy(),
            X_t.loc[val_ids].to_numpy(),
            Y_t.loc[val_ids].to_numpy(),
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predictions in the standardised target space of the training data."""
        preds = self.regressor.predict(self.x_prep.transform(X).to_numpy())
        return pd.DataFrame(preds, index=X.index, columns=list(self.y_prep.features_))

    def transform_targets(self, Y: pd.DataFrame) -> pd.DataFrame:
        """Truth in the same space as `predict` (monotone per metabolite for log10 targets)."""
        return self.y_prep.transform(Y)


def make_pipeline(
    x_prep: Any, y_prep: Any, regressor: Regressor, cfg: Mapping[str, Any], seed: int
) -> M2MPipeline:
    """Pipeline whose early-stopping split is seeded from the model seed plus a fixed offset."""
    inner = cfg["inner_validation"]
    return M2MPipeline(
        x_prep=x_prep,
        y_prep=y_prep,
        regressor=regressor,
        inner_fraction=float(inner["fraction"]),
        inner_seed=int(seed) + int(inner["seed_offset"]),
    )


# --------------------------------------------------------------------------------------------
# Cross-validation on the training cohort
# --------------------------------------------------------------------------------------------

PipelineBuilder = Callable[[Mapping[str, Any], int], M2MPipeline]


@dataclass
class FoldScore:
    params_index: int
    fold: int
    summary: dict[str, float]
    rho: pd.Series


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid, in a deterministic (key-sorted) order."""
    keys = sorted(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def grid_search_cv(
    build: PipelineBuilder,
    grid: Sequence[Mapping[str, Any]],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    labels: pd.Series,
    folds: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    seed: int = 0,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, list[FoldScore]]:
    """Score every parameter set on every PRISM fold; one row per parameter set.

    The selection metric is the mean over folds of each fold's mean ρ (DESIGN.md decision #6):
    fold models have different scales, so pooling their out-of-fold predictions would bias ranks.
    """
    fold_ids = sorted(int(f) for f in folds["fold"].unique())
    tasks = [(i, dict(params), fold) for i, params in enumerate(grid) for fold in fold_ids]
    scores: list[FoldScore] = Parallel(n_jobs=n_jobs)(
        delayed(_fit_score_fold)(build, params, seed, i, fold, X, Y, labels, folds, cfg)
        for i, params, fold in tasks
    )
    table = _cv_table(grid, scores, fold_ids)
    return table, scores


def cross_validate(
    build: PipelineBuilder,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    labels: pd.Series,
    folds: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    seed: int = 0,
    n_jobs: int = 1,
) -> tuple[dict[str, Any], list[FoldScore]]:
    """Cross-validate a single configuration; returns its CV row and the fold scores."""
    table, scores = grid_search_cv(
        build, [params], X, Y, labels, folds, cfg, seed=seed, n_jobs=n_jobs
    )
    return table.iloc[0].to_dict(), scores


def select_best(table: pd.DataFrame, metric: str = "cv_mean_rho") -> dict[str, Any]:
    """Highest CV score wins; ties go to the earlier grid entry (stable, documented)."""
    best_row = table.loc[table[metric].idxmax()]
    return json.loads(best_row["params"])


def _fit_score_fold(
    build: PipelineBuilder,
    params: dict[str, Any],
    seed: int,
    params_index: int,
    fold: int,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    labels: pd.Series,
    folds: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> FoldScore:
    _single_thread_torch()
    train_ids, test_ids = splits.fold_train_test(folds, fold)
    splits.assert_disjoint(train_ids, test_ids, what=f"CV fold {fold}")
    pipe = build(params, seed).fit(X.loc[train_ids], Y.loc[train_ids], labels.loc[train_ids])
    check_fitted_on(pipe, train_ids)
    truth = pipe.transform_targets(Y.loc[test_ids])
    pred = pipe.predict(X.loc[test_ids])
    rho = pd.Series(evaluate.spearman_per_feature(truth, pred), index=truth.columns)
    summary = {**evaluate.summarize_from_cfg(rho.to_numpy(), cfg), **prep_counts(pipe)}
    return FoldScore(params_index=params_index, fold=fold, summary=summary, rho=rho)


def _cv_table(
    grid: Sequence[Mapping[str, Any]], scores: Sequence[FoldScore], fold_ids: Sequence[int]
) -> pd.DataFrame:
    rows = []
    for i, params in enumerate(grid):
        mine = sorted((s for s in scores if s.params_index == i), key=lambda s: s.fold)
        fold_means = [s.summary["mean_rho"] for s in mine]
        row: dict[str, Any] = {"params": json.dumps(dict(params), sort_keys=True), **params}
        for key in mine[0].summary:
            row[f"cv_{key}"] = float(np.mean([s.summary[key] for s in mine]))
        row["cv_mean_rho_sd_over_folds"] = (
            float(np.std(fold_means, ddof=1)) if len(mine) > 1 else 0.0
        )
        row.update({f"fold{f}_mean_rho": m for f, m in zip(fold_ids, fold_means, strict=True)})
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Final evaluation on the external cohort
# --------------------------------------------------------------------------------------------


@dataclass
class ExternalResult:
    truth: pd.DataFrame
    predictions: list[pd.DataFrame]
    seeds: list[int]
    per_seed: list[dict[str, float]]
    rho_by_seed: pd.DataFrame
    aggregate: dict[str, float] = field(default_factory=dict)


def evaluate_external(
    build: PipelineBuilder,
    params: Mapping[str, Any],
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    labels_train: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    null_threshold: float | None,
    n_jobs: int = 1,
) -> ExternalResult:
    """Fit on all training rows once per seed, predict the external cohort, score it."""
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_external)(
            build, dict(params), s, X_train, Y_train, labels_train, X_test, Y_test
        )
        for s in seeds
    )
    truths = [truth for truth, _, _ in outputs]
    for other in truths[1:]:
        # Preprocessing is deterministic, so every seed must see the same evaluation targets.
        pd.testing.assert_frame_equal(truths[0], other)
    result = score_predictions(
        truths[0], [pred for _, pred, _ in outputs], cfg, seeds=seeds,
        null_threshold=null_threshold,
    )
    result.aggregate.update(outputs[0][2])  # how many inputs/targets survived the filters
    return result


def score_predictions(
    truth: pd.DataFrame,
    predictions: Sequence[pd.DataFrame],
    cfg: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    null_threshold: float | None,
) -> ExternalResult:
    """Score per-seed predictions made elsewhere (e.g. in envs/mgm) exactly like local ones."""
    rho_cols, per_seed = {}, []
    for seed, pred in zip(seeds, predictions, strict=True):
        pred = pred.loc[truth.index, truth.columns]
        rho = evaluate.spearman_per_feature(truth, pred)
        rho_cols[f"rho_seed{seed}"] = rho
        per_seed.append(evaluate.summarize_from_cfg(rho, cfg, null_threshold))
    ev = cfg["evaluation"]
    boot = evaluate.bootstrap_mean_rho(
        truth.to_numpy(),
        np.stack([p.loc[truth.index, truth.columns].to_numpy() for p in predictions]),
        n_boot=int(ev["n_bootstrap"]),
        seed=int(ev["bootstrap_seed"]),
        ci_level=float(ev["ci_level"]),
        method=str(ev["ci_method"]),
    )
    aggregate = {
        **evaluate.aggregate_seeds(per_seed),
        "mean_rho_ci_low": boot["ci_low"],
        "mean_rho_ci_high": boot["ci_high"],
        "mean_rho_ci_low_percentile": boot["ci_low_percentile"],
        "mean_rho_ci_high_percentile": boot["ci_high_percentile"],
        "mean_rho_boot_sd": boot["boot_sd"],
        "n_seeds": len(seeds),
        "null_threshold": np.nan if null_threshold is None else float(null_threshold),
    }
    rho_by_seed = pd.DataFrame(rho_cols, index=truth.columns)
    return ExternalResult(
        truth=truth,
        predictions=list(predictions),
        seeds=list(seeds),
        per_seed=per_seed,
        rho_by_seed=rho_by_seed,
        aggregate=aggregate,
    )


def _fit_predict_external(
    build: PipelineBuilder,
    params: dict[str, Any],
    seed: int,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    labels_train: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    _single_thread_torch()
    splits.assert_disjoint(list(X_train.index), list(X_test.index), what="train vs external test")
    pipe = build(params, seed).fit(X_train, Y_train, labels_train)
    check_fitted_on(pipe, list(X_train.index))
    return pipe.transform_targets(Y_test), pipe.predict(X_test), prep_counts(pipe)


# --------------------------------------------------------------------------------------------
# Recording results
# --------------------------------------------------------------------------------------------


def record_cv_table(
    table: pd.DataFrame,
    *,
    experiment: str,
    model: str,
    target_set: str,
    data: str,
    cfg: Mapping[str, Any],
    n_train: int,
    extra_tags: Mapping[str, Any] | None = None,
) -> None:
    """One runs.csv row (and MLflow run) per CV configuration.

    `extra_tags` adds columns such as the MGM environment's library versions (E2/E3 compute in
    envs/mgm, whose versions differ from the main environment's that every run records).
    """
    for _, row in table.iterrows():
        params = json.loads(row["params"])
        metrics = {k: row[k] for k in table.columns if k.startswith(("cv_", "fold"))}
        tracking.log_run(
            experiment=experiment,
            run_name=f"{model}|{target_set}|cv|{row['params']}",
            params=params,
            metrics=metrics,
            tags={
                "model": model,
                "target_set": target_set,
                "data": data,
                "stage": "cv",
                "seed": "",
                "n_train": n_train,
                **dict(extra_tags or {}),
            },
            cfg=cfg,
        )


def record_external(
    result: ExternalResult,
    *,
    experiment: str,
    model: str,
    target_set: str,
    data: str,
    params: Mapping[str, Any],
    cfg: Mapping[str, Any],
    n_train: int,
    cv_mean_rho: float | None = None,
    write_summary: bool = True,
    write_per_metabolite_file: bool = True,
    extra_tags: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """runs.csv rows per seed, per-metabolite ρ file, predictions, and the summary.csv row.

    Ablation arms pass write_summary=False / write_per_metabolite_file=False: summary.csv holds
    only the main models, one row per model × target set. Main models also keep their external
    predictions (data/processed/external/), so paired comparisons can be made afterwards without
    scoring the external cohort again.
    """
    for seed, summary in zip(result.seeds, result.per_seed, strict=True):
        tracking.log_run(
            experiment=experiment,
            run_name=f"{model}|{target_set}|external|seed{seed}",
            params=params,
            metrics=summary,
            tags={
                "model": model,
                "target_set": target_set,
                "data": data,
                "stage": "external",
                "seed": seed,
                "n_train": n_train,
                **dict(extra_tags or {}),
            },
            cfg=cfg,
        )
    if write_per_metabolite_file:
        write_per_metabolite(result, model=model, target_set=target_set)
        write_external_predictions(result, model=model, target_set=target_set)
    row = {
        "model": model,
        "target_set": target_set,
        "data": data,
        "experiment": experiment,
        "n_train": n_train,
        "n_test": len(result.truth),
        **result.aggregate,
        "cv_mean_rho": np.nan if cv_mean_rho is None else float(cv_mean_rho),
        "params": json.dumps(dict(params), sort_keys=True, default=str),
        "config": cfg.get("config_path", ""),
        "git_sha": tracking.git_sha(),
    }
    if write_summary:
        tracking.upsert_summary_row(row)
    return row


def per_metabolite_path(model: str, target_set: str) -> Path:
    """Where a model's per-metabolite external ρ lives (one naming scheme for every reader)."""
    return paths.RESULTS / "per_metabolite" / f"{model}__{target_set}.csv.gz"


def write_per_metabolite(result: ExternalResult, *, model: str, target_set: str) -> Path:
    """Per-metabolite external ρ: each seed as computed (NaN = undefined), and the mean / SD
    over seeds with NaN counted as 0, like every summary statistic."""
    out = per_metabolite_path(model, target_set)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = result.rho_by_seed.copy()
    counted = table.fillna(0.0)
    table["rho_mean"] = counted.mean(axis=1)
    table["rho_sd"] = counted.std(axis=1, ddof=1) if counted.shape[1] > 1 else 0.0
    table.index.name = "feature"
    # mtime 0: identical results give byte-identical committed files.
    table.to_csv(out, compression={"method": "gzip", "mtime": 0})
    return out


def external_predictions_path(model: str, target_set: str) -> Path:
    return paths.DATA_PROCESSED / "external" / f"{model}__{target_set}.npz"


def write_external_predictions(result: ExternalResult, *, model: str, target_set: str) -> Path:
    """Truth and per-seed predictions on the external cohort (standardised target space)."""
    out = external_predictions_path(model, target_set)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(result.truth.columns)
    np.savez_compressed(
        out,
        sample_ids=np.asarray(result.truth.index, dtype=str),
        features=np.asarray(columns, dtype=str),
        seeds=np.asarray(result.seeds),
        truth=result.truth.to_numpy(dtype=float),
        predictions=np.stack(
            [p.loc[result.truth.index, columns].to_numpy(dtype=float) for p in result.predictions]
        ),
    )
    return out


def load_external_predictions(model: str, target_set: str) -> dict[str, np.ndarray]:
    path = external_predictions_path(model, target_set)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run {model}'s external evaluation first")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_selection(
    name: str, target_set: str, best: Mapping[str, Any], table: pd.DataFrame
) -> Path:
    """Persist the chosen hyperparameters and the full CV grid they won."""
    sel_dir = paths.RESULTS / "selected"
    sel_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(sel_dir / f"{name}__{target_set}__cv.csv", index=False)
    best_row = table.loc[table["params"] == json.dumps(dict(best), sort_keys=True)].iloc[0]
    payload = {
        "name": name,
        "target_set": target_set,
        "params": dict(best),
        "cv_mean_rho": float(best_row["cv_mean_rho"]),
        "n_configs": len(table),
    }
    out = sel_dir / f"{name}__{target_set}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return out


def load_selection(name: str, target_set: str) -> dict[str, Any]:
    path = paths.RESULTS / "selected" / f"{name}__{target_set}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run model selection for {name} first")
    return json.loads(path.read_text())


def save_null_threshold(
    *, data: str, target_set: str, null: evaluate.NullResult, n_features: int
) -> None:
    tracking.upsert_csv_row(
        paths.RESULTS / NULL_THRESHOLDS_CSV,
        {
            "data": data,
            "target_set": target_set,
            "threshold": float(null.threshold),
            "quantile": float(null.quantile),
            "n_permutations": len(null.per_permutation_mean),
            "n_features": n_features,
            "null_mean_rho": float(np.mean(null.per_permutation_mean)),
            "null_sd_of_mean_rho": float(np.std(null.per_permutation_mean, ddof=1)),
            "timestamp": tracking.utc_now(),
        },
        key=("data", "target_set"),
    )


def record_null(
    null: evaluate.NullResult,
    *,
    experiment: str,
    data: str,
    target_set: str,
    n_features: int,
    cfg: Mapping[str, Any],
) -> None:
    """The permutation null as a run of its own (runs.csv + MLflow), like every other fit."""
    tracking.log_run(
        experiment=experiment,
        run_name=f"null|{data}|{target_set}",
        params={"n_permutations": len(null.per_permutation_mean), "quantile": null.quantile},
        metrics={
            "threshold": float(null.threshold),
            "null_mean_rho": float(np.mean(null.per_permutation_mean)),
            "null_sd_of_mean_rho": float(np.std(null.per_permutation_mean, ddof=1)),
            "n_features": n_features,
        },
        tags={"model": "ridge-permutation-null", "target_set": target_set, "data": data,
              "stage": "null"},
        cfg=cfg,
    )


def load_null_threshold(data: str, target_set: str) -> float:
    path = paths.RESULTS / NULL_THRESHOLDS_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the permutation null first (make baselines)")
    table = pd.read_csv(path)
    match = table[(table["data"] == data) & (table["target_set"] == target_set)]
    if match.empty:
        raise KeyError(f"no permutation-null threshold for data={data}, target_set={target_set}")
    return float(match["threshold"].iloc[0])


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def check_fitted_on(pipe: M2MPipeline, train_ids: Sequence[Any]) -> None:
    """Leakage tripwire: every preprocessor that records its rows was fit on exactly these."""
    expected = [str(i) for i in train_ids]
    # truth_prep: a pipeline scored in another target space (ablation A2) fits one more.
    for name in ("x_prep", "y_prep", "truth_prep"):
        prep = getattr(pipe, name, None)
        seen = getattr(prep, "fit_samples_", None)
        if seen is not None and [str(i) for i in seen] != expected:
            raise splits.LeakageError(f"{name} was fit on rows other than the training rows")


def prep_counts(pipe: M2MPipeline) -> dict[str, float]:
    """Inputs and targets that survived the training-set filters (brief §4: record them)."""
    dropped = getattr(pipe.y_prep, "n_dropped_zero_", None)
    return {
        "n_input_features": float(len(pipe.x_prep.features_)),
        "n_targets_dropped_zero": np.nan if dropped is None else float(dropped),
    }


def _check_aligned(X: pd.DataFrame, Y: pd.DataFrame) -> None:
    if not X.index.equals(Y.index):
        raise ValueError("microbiome and metabolome rows are not aligned by sample id")


def _single_thread_torch() -> None:
    """One intra-op thread per task: results then do not depend on n_jobs, and small models
    run faster with process-level parallelism than with thread contention."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of the main env
        return
    torch.set_num_threads(1)


# Public names for helpers that experiment modules running work out of process also need.
cv_table = _cv_table
single_thread_torch = _single_thread_torch
