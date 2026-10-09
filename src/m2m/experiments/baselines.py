"""E0: baselines on D1 (mean, ridge, elastic net, random forest) and the permutation null.

Stages, run in this order by ``python -m m2m.run configs/e0_baselines.yaml [--stage S]``:

- ``null``: per target set, the ridge baseline refit on PRISM with shuffled microbiome rows and
  scored on NLIBD (brief §5) → results/null_thresholds.csv, plus the pooled null ρ in
  data/processed/null/D1__<target_set>.npy for the figure. It comes first because every external
  score reports N(ρ above this threshold).
- ``cv``: 5-fold PRISM cross-validation of every model × target set → runs.csv rows and
  results/selected/e0_<model>__<target_set>.json. The baselines have fixed hyperparameters
  (ridge tunes its penalty per metabolite by leave-one-out, the elastic net by inner K-fold, both
  on training rows only), so the "selection" just records them with their CV mean ρ.
- ``external``: each model refit on all of PRISM and scored once on NLIBD → summary.csv.
- ``curve``: ridge on the saved learning-curve subsets → results/learning_curve.csv.

Only the final-evaluation stages (null, external, curve) open the external cohort, through
``common.load_d1_test`` and only after everything they evaluate has been loaded from disk.
"""

from __future__ import annotations

import functools
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

from m2m import evaluate, paths, protocol
from m2m.experiments import common

EXPERIMENT = "e0"
DATA = "D1"
# Selection first, then the stages that open the external cohort (DESIGN decision #24).
STAGES = ("cv", "null", "external", "curve")
NULL_DIR = "null"  # under data/processed: pooled permutation-null ρ, one .npy per target set
# Ridge is deterministic (no seed enters RidgeCV), so the null's refits all use this one.
NULL_MODEL_SEED = 0
# The mean predictor is the floor by definition: every metabolite gets its training mean.
MEAN_PARAMS = {"strategy": "mean"}
MODEL_ORDER = ("mean", "ridge", "enet", "rf")
MODEL_KEYS = ("target_sets", "external_seeds", "params")
# The elastic net parallelises over metabolites inside each fit, so its outer folds and seeds
# run one at a time (nested process pools would oversubscribe the machine).
PARALLEL_INSIDE = frozenset({"enet"})


def run(cfg: dict[str, Any], *, stage: str | None = None, n_jobs: int = 1) -> None:
    """Run the requested stage, or all of them in order."""
    actions = {"null": stage_null, "cv": stage_cv, "external": stage_external, "curve": stage_curve}
    for name in common.stages_to_run(stage, STAGES):
        start = time.perf_counter()
        actions[name](cfg, n_jobs=n_jobs)
        _say(f"stage {name} done in {time.perf_counter() - start:.1f}s")


# --------------------------------------------------------------------------------------------
# The elastic net regressor
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _TargetFit:
    """What one metabolite's ElasticNetCV leaves behind (small, so workers return it cheaply)."""

    coef: np.ndarray
    intercept: float
    alpha: float
    l1_ratio: float
    n_convergence_warnings: int


class PerTargetElasticNet:
    """One ``ElasticNetCV`` per target column, as MelonnPan fits one elastic net per metabolite.

    Every target gets its own penalty (``alpha_``) and L1/L2 mix (``l1_ratio_``), chosen by inner
    K-fold CV on the training rows it is given. The inner folds are identical for all targets (a
    seeded ``KFold``), like MelonnPan's shared ``foldid``. Targets are independent, so they are fit
    in parallel with ``n_jobs`` processes; the result does not depend on ``n_jobs``.
    """

    needs_validation = False

    def __init__(
        self,
        *,
        l1_ratio: Sequence[float],
        n_alphas: int,
        eps: float,
        inner_folds: int,
        inner_seed: int,
        max_iter: int,
        tol: float,
        n_jobs: int = 1,
    ) -> None:
        self.l1_ratio = [float(r) for r in l1_ratio]
        self.n_alphas = int(n_alphas)
        self.eps = float(eps)
        self.inner_folds = int(inner_folds)
        self.inner_seed = int(inner_seed)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_jobs = int(n_jobs)

    def make_estimator(self) -> ElasticNetCV:
        """The unfitted single-target model; ``alphas=<int>`` is the path length (sklearn ≥ 1.7)."""
        return ElasticNetCV(
            l1_ratio=list(self.l1_ratio),
            alphas=self.n_alphas,
            eps=self.eps,
            cv=KFold(n_splits=self.inner_folds, shuffle=True, random_state=self.inner_seed),
            max_iter=self.max_iter,
            tol=self.tol,
        )

    def fit(
        self, X: np.ndarray, Y: np.ndarray, X_val: Any = None, Y_val: Any = None
    ) -> PerTargetElasticNet:
        """Fit every target column of ``Y`` on ``X`` (``X_val``/``Y_val`` are not used)."""
        X, Y = _as_matrix(X, "X"), _as_matrix(Y, "Y")
        if len(X) != len(Y):
            raise ValueError(f"X has {len(X)} rows but Y has {len(Y)}")
        fits: list[_TargetFit] = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one_target)(self.make_estimator(), X, Y[:, j]) for j in range(Y.shape[1])
        )
        self.coef_ = np.vstack([f.coef for f in fits])
        self.intercept_ = np.array([f.intercept for f in fits])
        self.alpha_ = np.array([f.alpha for f in fits])
        self.l1_ratio_ = np.array([f.l1_ratio for f in fits])
        self.n_not_converged_ = int(sum(f.n_convergence_warnings > 0 for f in fits))
        if self.n_not_converged_:
            # One summary per fit instead of one warning per path point and metabolite.
            warnings.warn(
                f"elastic net: {self.n_not_converged_} of {len(fits)} targets hit "
                f"max_iter={self.max_iter} somewhere on their path (tol={self.tol}); "
                "raise baselines.enet.params.max_iter if this persists",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``X @ coef.T + intercept``: what each target's ElasticNetCV.predict returns."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("PerTargetElasticNet is not fitted; call fit() first")
        X = _as_matrix(X, "X")
        if X.shape[1] != self.coef_.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features; the model takes {self.coef_.shape[1]}")
        return X @ self.coef_.T + self.intercept_


def _fit_one_target(estimator: ElasticNetCV, X: np.ndarray, y: np.ndarray) -> _TargetFit:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(X, y)
    n_convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    for w in caught:  # recording swallowed every warning; only the convergence ones are counted
        if not issubclass(w.category, ConvergenceWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return _TargetFit(
        coef=np.asarray(estimator.coef_, dtype=float).copy(),
        intercept=float(estimator.intercept_),
        alpha=float(estimator.alpha_),
        l1_ratio=float(estimator.l1_ratio_),
        n_convergence_warnings=int(n_convergence),
    )


def _as_matrix(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (samples x columns), got shape {array.shape}")
    return array


# --------------------------------------------------------------------------------------------
# Pipeline builders: module-level functions so functools.partial pickles into joblib workers
# --------------------------------------------------------------------------------------------


def build_mean(
    params: Mapping[str, Any], seed: int, *, cfg: Mapping[str, Any], target_set: str
) -> protocol.M2MPipeline:
    regressor = protocol.SklearnRegressor(DummyRegressor(strategy=params["strategy"]))
    return _pipeline(regressor, cfg, target_set, seed)


def build_enet(
    params: Mapping[str, Any],
    seed: int,
    *,
    cfg: Mapping[str, Any],
    target_set: str,
    n_jobs: int = 1,
) -> protocol.M2MPipeline:
    # The seed is not used: the fit is deterministic given the configured inner folds.
    return _pipeline(PerTargetElasticNet(**params, n_jobs=n_jobs), cfg, target_set, seed)


def build_rf(
    params: Mapping[str, Any], seed: int, *, cfg: Mapping[str, Any], target_set: str
) -> protocol.M2MPipeline:
    # One tree-building process per forest: CV folds / seeds already run in parallel.
    forest = RandomForestRegressor(**params, random_state=int(seed), n_jobs=1)
    return _pipeline(protocol.SklearnRegressor(forest), cfg, target_set, seed)


def _pipeline(
    regressor: protocol.Regressor, cfg: Mapping[str, Any], target_set: str, seed: int
) -> protocol.M2MPipeline:
    return protocol.make_pipeline(
        common.microbe_prep(cfg), common.metabolite_prep(cfg, target_set), regressor, cfg, seed
    )


_BUILDERS = {
    "mean": build_mean,
    "ridge": common.build_ridge,
    "enet": build_enet,
    "rf": build_rf,
}


# --------------------------------------------------------------------------------------------
# The configured models
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Baseline:
    """One E0 model as configured: its fixed hyperparameters and where it is evaluated."""

    name: str
    params: dict[str, Any]
    target_sets: tuple[str, ...]
    external_seeds: tuple[int, ...]

    @property
    def selection_name(self) -> str:
        return f"e0_{self.name}"

    def builder(
        self, cfg: Mapping[str, Any], target_set: str, n_jobs: int
    ) -> protocol.PipelineBuilder:
        if self.name in PARALLEL_INSIDE:
            return functools.partial(
                _BUILDERS[self.name], cfg=cfg, target_set=target_set, n_jobs=n_jobs
            )
        return functools.partial(_BUILDERS[self.name], cfg=cfg, target_set=target_set)

    def outer_jobs(self, n_jobs: int) -> int:
        """Processes for folds / seeds: one when the model already parallelises inside a fit."""
        return 1 if self.name in PARALLEL_INSIDE else n_jobs


def configured_baselines(cfg: Mapping[str, Any]) -> list[Baseline]:
    """The models of cfg["baselines"], in a fixed order (mean, ridge, enet, rf)."""
    section = cfg["baselines"]
    unknown = sorted(set(section) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"unknown baseline(s) {unknown}; known: {list(MODEL_ORDER)}")
    known_sets = set(common.target_sets(cfg))
    models = []
    for name in MODEL_ORDER:
        if name not in section:
            continue
        spec = section[name]
        _check_keys(spec, name)
        target_sets = tuple(str(t) for t in spec["target_sets"])
        # Every external score needs that target set's null threshold, computed for cfg's sets.
        outside = sorted(set(target_sets) - known_sets)
        if outside:
            raise ValueError(f"baseline {name}: target sets {outside} are not in {known_sets}")
        seeds = spec.get("external_seeds") or common.model_seeds(cfg)
        models.append(
            Baseline(
                name=name,
                params=_model_params(name, spec, cfg),
                target_sets=target_sets,
                external_seeds=tuple(int(s) for s in seeds),
            )
        )
    return models


def _model_params(name: str, spec: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Recorded hyperparameters; ridge's live in cfg["ridge"] (shared with E5 and the ablations)."""
    if name in ("mean", "ridge"):
        if "params" in spec:  # they would be recorded but never used
            raise ValueError(f"baseline {name} takes no `params` (ridge: edit the ridge section)")
        return dict(MEAN_PARAMS) if name == "mean" else dict(cfg["ridge"])
    if "params" not in spec:
        raise ValueError(f"baseline {name} needs a `params` section")
    return dict(spec["params"])


def _check_keys(spec: Mapping[str, Any], name: str) -> None:
    unknown = sorted(set(spec) - set(MODEL_KEYS))
    if unknown:  # e.g. a misspelt `external_seeds` would silently fall back to every model seed
        raise ValueError(f"unknown key(s) {unknown} for baseline {name}; allowed: {MODEL_KEYS}")


def _baseline(cfg: Mapping[str, Any], name: str) -> Baseline:
    for model in configured_baselines(cfg):
        if model.name == name:
            return model
    raise ValueError(f"baseline {name!r} is not configured in cfg['baselines']")


# --------------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------------


def stage_null(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Permutation null of the ridge baseline per target set (sequential: ridge refits are cheap).

    The truth is NLIBD in the space of a ridge pipeline fit on the unshuffled PRISM rows; each
    permutation refits a fresh pipeline, preprocessing included, on the shuffled pairs.
    """
    require_scratch_null_dir()
    X, Y, labels = common.load_d1_train()
    evaluation = cfg["evaluation"]
    for target_set in common.target_sets(cfg):
        X_test, Y_test = common.load_d1_test(
            purpose=(
                "permutation null: ridge refit on shuffled PRISM, scored on NLIBD "
                f"({target_set})"
            )
        )
        build = functools.partial(common.build_ridge, cfg=cfg, target_set=target_set)
        params = dict(cfg["ridge"])
        y_eval = _fit(build, params, X, Y, labels).transform_targets(Y_test)
        fit_predict = functools.partial(
            _null_fit_predict, build=build, params=params, labels=labels,
            columns=list(y_eval.columns),
        )
        null = evaluate.permutation_null(
            fit_predict,
            X,
            Y,
            X_test,
            y_eval,
            n_permutations=int(evaluation["n_permutations"]),
            seed=int(evaluation["permutation_seed"]),
            quantile=float(evaluation["null_quantile"]),
        )
        protocol.save_null_threshold(
            data=DATA, target_set=target_set, null=null, n_features=y_eval.shape[1]
        )
        protocol.record_null(null, experiment=EXPERIMENT, data=DATA, target_set=target_set,
                             n_features=y_eval.shape[1], cfg=cfg)
        saved = save_pooled_null(null, target_set)
        _say(
            f"null {target_set}: threshold {null.threshold:.4f} over {y_eval.shape[1]} targets, "
            f"mean null ρ {np.mean(null.per_permutation_mean):+.4f} → {saved}"
        )


def stage_cv(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """PRISM CV of every model × target set; records the CV table and the fixed 'selection'."""
    X, Y, labels = common.load_d1_train()
    folds = common.d1_folds(labels, cfg)
    for model in configured_baselines(cfg):
        for target_set in model.target_sets:
            start = time.perf_counter()
            row, _ = protocol.cross_validate(
                model.builder(cfg, target_set, n_jobs),
                model.params,
                X,
                Y,
                labels,
                folds,
                cfg,
                seed=int(cfg["cv_seed"]),
                n_jobs=model.outer_jobs(n_jobs),
            )
            table = pd.DataFrame([row])
            protocol.record_cv_table(
                table, experiment=EXPERIMENT, model=model.name, target_set=target_set,
                data=DATA, cfg=cfg, n_train=len(X),
            )
            protocol.save_selection(model.selection_name, target_set, model.params, table)
            _say(
                f"cv {model.name} {target_set}: mean ρ {row['cv_mean_rho']:.4f} "
                f"(SD over folds {row['cv_mean_rho_sd_over_folds']:.4f}, "
                f"{row['cv_n_features']:.0f} targets) in {time.perf_counter() - start:.1f}s"
            )


def stage_external(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Refit on all of PRISM and score NLIBD once per model × target set (one load per set)."""
    X, Y, labels = common.load_d1_train()
    models = configured_baselines(cfg)
    # Everything evaluated, for every target set, is read from disk before the external cohort
    # is first opened: a missing or stale record must fail before any external score exists,
    # not after one target set was scored (a rerun would then score it on NLIBD again).
    plan = {}
    for target_set in common.target_sets(cfg):
        chosen = [m for m in models if target_set in m.target_sets]
        if chosen:
            selections = {m.name: load_checked_selection(m, target_set) for m in chosen}
            plan[target_set] = (chosen, selections, protocol.load_null_threshold(DATA, target_set))
    for target_set, (chosen, selections, threshold) in plan.items():
        X_test, Y_test = common.load_d1_test(
            purpose=(
                f"E0 final evaluation ({', '.join(selections)}): fit on PRISM, "
                f"scored on NLIBD ({target_set})"
            )
        )
        for model in chosen:
            selection = selections[model.name]
            result = protocol.evaluate_external(
                model.builder(cfg, target_set, n_jobs),
                selection["params"],
                X,
                Y,
                labels,
                X_test,
                Y_test,
                cfg,
                seeds=list(model.external_seeds),
                null_threshold=threshold,
                n_jobs=model.outer_jobs(n_jobs),
            )
            protocol.record_external(
                result, experiment=EXPERIMENT, model=model.name, target_set=target_set,
                data=DATA, params=selection["params"], cfg=cfg, n_train=len(X),
                cv_mean_rho=selection["cv_mean_rho"],
            )
            agg = result.aggregate
            _say(
                f"external {model.name} {target_set}: mean ρ {agg['mean_rho_mean']:.4f} "
                f"± {agg['mean_rho_sd']:.4f} over {agg['n_seeds']} seed(s), "
                f"CI [{agg['mean_rho_ci_low']:.4f}, {agg['mean_rho_ci_high']:.4f}]"
            )


def stage_curve(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Ridge on every saved learning-curve subset, scored on NLIBD."""
    target_set = str(cfg["learning_curve"]["target_set"])
    ridge = _baseline(cfg, "ridge")
    selection = load_checked_selection(ridge, target_set)
    threshold = protocol.load_null_threshold(DATA, target_set)
    X, Y, labels = common.load_d1_train()
    X_test, Y_test = common.load_d1_test(
        purpose=f"E0 learning curve: ridge refit on PRISM subsets, scored on NLIBD ({target_set})"
    )
    points = common.run_learning_curve(
        ridge.builder(cfg, target_set, n_jobs),
        selection["params"],
        X,
        Y,
        labels,
        X_test,
        Y_test,
        cfg,
        model=ridge.name,
        data_name=DATA,
        experiment=EXPERIMENT,
        null_threshold=threshold,
        n_jobs=n_jobs,
    )
    by_size = points.groupby("n_train")["mean_rho"].mean()
    _say("curve ridge: " + ", ".join(f"n={n}: {rho:.4f}" for n, rho in by_size.items()))


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def load_checked_selection(model: Baseline, target_set: str) -> dict[str, Any]:
    """The cv stage's record for this model, refusing it if the config changed since then."""
    selection = protocol.load_selection(model.selection_name, target_set)
    if selection["params"] != model.params:
        raise ValueError(
            f"{model.selection_name} ({target_set}) was cross-validated with "
            f"{selection['params']}, but the config now says {model.params}: rerun --stage cv"
        )
    return selection


def require_scratch_null_dir() -> None:
    """A dry-run null must not land where the real one lives (data/processed/null).

    common.dry_run() only insists that results/ is redirected, but the pooled null goes to
    data/processed, and `make figures` draws the null-distribution figure from there: a dry run
    with only M2M_RESULTS set would silently replace the real NLIBD null with a PRISM hold-out one.
    """
    repo_default = (paths.REPO_ROOT / "data" / "processed").resolve()
    if common.dry_run() and paths.DATA_PROCESSED.resolve() == repo_default:
        raise RuntimeError("dry runs must write to a scratch directory: set M2M_DATA_PROCESSED")


def save_pooled_null(null: evaluate.NullResult, target_set: str) -> str:
    """Pooled null ρ (permutation-major, NaN → 0) for the null-distribution figure."""
    out = paths.DATA_PROCESSED / NULL_DIR / f"{DATA}__{target_set}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, null.pooled_rho)
    return str(out)


def _fit(
    build: protocol.PipelineBuilder,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    labels: pd.Series,
) -> protocol.M2MPipeline:
    pipe = build(params, NULL_MODEL_SEED).fit(X, Y, labels)
    protocol.check_fitted_on(pipe, list(X.index))
    return pipe


def _null_fit_predict(
    X_permuted: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    build: protocol.PipelineBuilder,
    params: Mapping[str, Any],
    labels: pd.Series,
    columns: list[str],
) -> pd.DataFrame:
    """A fresh ridge pipeline (preprocessing included) fit on the shuffled pairs."""
    predictions = _fit(build, params, X_permuted, Y_train, labels).predict(X_test)
    if list(predictions.columns) != columns:
        # Y_train is never shuffled, so the zero filter must keep exactly the evaluated targets.
        raise ValueError("a permuted refit retained different targets than the evaluation truth")
    return predictions


def _say(message: str) -> None:
    print(f"[{EXPERIMENT}] {message}", flush=True)
