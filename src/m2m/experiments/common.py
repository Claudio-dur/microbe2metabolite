"""Helpers shared by the experiment modules: D1 data assembly, preprocessors, stages, curves."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedShuffleSplit

from m2m import data, evaluate, freeze, paths, protocol, splits, tracking
from m2m.preprocess import MetabolitePreprocessor, MicrobePreprocessor

LEARNING_CURVE_CSV = "learning_curve.csv"
LEARNING_CURVE_SUBSETS_CSV = "learning_curve_subsets.csv"
TARGET_SETS = ("T466", "TALL")

# Development-only switch: replaces the external cohort by a held-out slice of PRISM so every
# code path (including final-evaluation stages) can be exercised without opening NLIBD.
DRY_RUN_ENV = "M2M_DRY_RUN"
DRY_RUN_TEST_FRACTION = 0.25
DRY_RUN_SEED = 12345


# --- data ------------------------------------------------------------------------------------


def dry_run() -> bool:
    """True when M2M_DRY_RUN=1; refuses unless every output location points away from the repo.

    Results, processed data (nulls, embeddings, MGM task rounds, pretrained weights) and the
    MLflow store must all be redirected: a dry run must never overwrite a real run's files.
    """
    if os.environ.get(DRY_RUN_ENV) != "1":
        return False
    defaults = {
        "M2M_RESULTS": (paths.RESULTS, paths.REPO_ROOT / "results"),
        "M2M_DATA_PROCESSED": (paths.DATA_PROCESSED, paths.REPO_ROOT / "data" / "processed"),
        "M2M_MLRUNS": (paths.MLRUNS, paths.REPO_ROOT / "mlruns"),
    }
    # Inside a committed output folder counts too (e.g. M2M_RESULTS=results/dryrun).
    shared = [
        name for name, (used, real) in defaults.items()
        if used.resolve().is_relative_to(real.resolve())
    ]
    if shared:
        raise RuntimeError(f"dry runs must write to a scratch directory: set {', '.join(shared)}")
    return True


def dry_run_ids() -> tuple[list[str], list[str]]:
    """(pseudo-train, pseudo-test) D1 PRISM ids used instead of PRISM / NLIBD in dry runs."""
    prism = data.load_prism()
    ids, labels = prism.meta.index.to_numpy(), prism.meta["diagnosis"].to_numpy()
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=DRY_RUN_TEST_FRACTION, random_state=DRY_RUN_SEED
    )
    train_idx, test_idx = next(splitter.split(ids, labels))
    return sorted(ids[train_idx].tolist()), sorted(ids[test_idx].tolist())


def load_d1_train() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """PRISM microbes, metabolites and CD/UC/Control labels (the only data selection may use)."""
    prism = data.load_prism()
    if dry_run():
        prism = prism.subset(dry_run_ids()[0])
    return prism.microbes, prism.metabolites, prism.meta["diagnosis"]


def load_d1_test(purpose: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """NLIBD, opened only for a final evaluation; the reason is written to the access log."""
    if dry_run():
        # The same freeze gate as a real opening (minus the git check), so dry runs exercise it.
        freeze.require_frozen(purpose)
        pseudo = data.load_prism().subset(dry_run_ids()[1])
        return pseudo.microbes, pseudo.metabolites
    nlibd = data.load_nlibd(final_evaluation=True, purpose=purpose)
    return nlibd.microbes, nlibd.metabolites


def learning_curve_sizes(cfg: Mapping[str, Any], n_available: int) -> list[int]:
    """Configured sizes, capped at the training-set size (only binds in dry runs)."""
    sizes = sorted({min(int(n), n_available) for n in cfg["learning_curve"]["sizes"]})
    return sizes


def target_candidates(target_set: str) -> list[str] | None:
    """T466 = the annotated metabolites; TALL = every feature (the zero filter decides later)."""
    if target_set == "T466":
        return data.t466_features()
    if target_set == "TALL":
        return None
    raise ValueError(f"unknown D1 target set {target_set!r}")


def microbe_prep(
    cfg: Mapping[str, Any], features: Sequence[str] | None = None
) -> MicrobePreprocessor:
    return MicrobePreprocessor.from_config(cfg["microbes"], features=features)


def metabolite_prep(
    cfg: Mapping[str, Any], target_set: str, candidates: Sequence[str] | None = None
) -> MetabolitePreprocessor:
    chosen = candidates if candidates is not None else target_candidates(target_set)
    return MetabolitePreprocessor.from_config(cfg["metabolites"], candidate_features=chosen)


def d1_folds(labels: pd.Series, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """The single saved set of PRISM folds shared by every model."""
    return splits.prism_folds(list(labels.index), list(labels), cfg)


def model_seeds(cfg: Mapping[str, Any]) -> list[int]:
    return [int(s) for s in cfg["seeds"]]


def target_sets(cfg: Mapping[str, Any]) -> list[str]:
    return [str(t) for t in cfg.get("target_sets", TARGET_SETS)]


def stages_to_run(requested: str | None, available: Sequence[str]) -> list[str]:
    """All stages in order, or the requested ones ("cv" or "cv,external"), in canonical order."""
    if requested is None:
        return list(available)
    wanted = [s.strip() for s in requested.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in available]
    if unknown or not wanted:
        raise SystemExit(f"unknown stage(s) {unknown or requested!r}; available: {list(available)}")
    return [s for s in available if s in wanted]


# --- learning curves (E5) --------------------------------------------------------------------


def learning_curve_subsets(
    cfg: Mapping[str, Any],
    sample_ids: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create once, then reload: (n_train, seed) → D1 PRISM sample ids, stratified by diagnosis.

    Saving the subsets guarantees that ridge, the MLP, MGM (run in another environment) and the
    D2 transfer models are all trained on exactly the same people at each size. Ids default to
    D1 PRISM; tests pass synthetic ones.
    """
    if sample_ids is None or labels is None:
        X, _, y = load_d1_train()
        sample_ids, labels = list(X.index), list(y)
    path = paths.SPLITS / LEARNING_CURVE_SUBSETS_CSV
    if path.exists():
        subsets = pd.read_csv(path, dtype={"sample_id": str})
        _verify_subsets(subsets, sample_ids, cfg)
        return subsets
    rows = []
    for n in learning_curve_sizes(cfg, len(sample_ids)):
        for seed in model_seeds(cfg):
            ids = splits.stratified_subsample(list(sample_ids), list(labels), n=n, seed=seed)
            rows.extend({"n_train": int(n), "seed": seed, "sample_id": i} for i in ids)
    subsets = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    subsets.to_csv(path, index=False)
    return subsets


def subset_ids(subsets: pd.DataFrame, n_train: int, seed: int) -> list[str]:
    mask = (subsets["n_train"] == n_train) & (subsets["seed"] == seed)
    return subsets.loc[mask, "sample_id"].tolist()


def run_learning_curve(
    build: protocol.PipelineBuilder,
    params: Mapping[str, Any],
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    labels: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    model: str,
    data_name: str,
    experiment: str,
    null_threshold: float | None,
    n_jobs: int = 1,
    id_map: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Train on each saved PRISM subset, score once on the external cohort, record every point.

    `id_map` translates the saved D1 PRISM ids into this data source's ids (e.g. `PRISM|x` →
    `PRISM.x` for D2), so every data source uses the same people.
    """
    if id_map is None:
        subsets = learning_curve_subsets(cfg, list(X_train.index), list(labels.loc[X_train.index]))
    else:
        subsets = learning_curve_subsets(cfg)
    sizes = sorted(subsets["n_train"].astype(int).unique())
    points = [(n, seed) for n in sizes for seed in model_seeds(cfg)]
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_curve_point)(
            build,
            dict(params),
            seed,
            [id_map(i) if id_map else i for i in subset_ids(subsets, n, seed)],
            X_train,
            Y_train,
            labels,
            X_test,
            Y_test,
        )
        for n, seed in points
    )
    rows = []
    for (n, seed), (truth, pred, counts) in zip(points, outputs, strict=True):
        rho = evaluate.spearman_per_feature(truth, pred)
        summary = {**evaluate.summarize_from_cfg(rho, cfg, null_threshold), **counts}
        rows.append(
            record_curve_point(
                summary,
                model=model,
                data_name=data_name,
                target_set=cfg["learning_curve"]["target_set"],
                n_train=n,
                seed=seed,
                params=params,
                experiment=experiment,
                cfg=cfg,
            )
        )
    return pd.DataFrame(rows)


def record_curve_point(
    summary: Mapping[str, float],
    *,
    model: str,
    data_name: str,
    target_set: str,
    n_train: int,
    seed: int,
    params: Mapping[str, Any],
    experiment: str,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """One learning-curve point → results/learning_curve.csv (upsert) and runs.csv / MLflow."""
    tags = {
        "model": model,
        "data": data_name,
        "target_set": target_set,
        "stage": "learning_curve",
        "seed": seed,
        "n_train": n_train,
    }
    tracking.log_run(
        experiment=experiment,
        run_name=f"{model}|{target_set}|curve|n{n_train}|seed{seed}",
        params=params,
        metrics=summary,
        tags=tags,
        cfg=cfg,
    )
    row = {**tags, **summary, "experiment": experiment, "params": dict(params)}
    row.pop("stage")
    tracking.upsert_csv_row(
        paths.RESULTS / LEARNING_CURVE_CSV,
        {**row, "timestamp": tracking.utc_now()},
        key=("model", "data", "target_set", "n_train", "seed"),
    )
    return row


def _curve_point(
    build: protocol.PipelineBuilder,
    params: dict[str, Any],
    seed: int,
    ids: list[str],
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    labels: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    protocol._single_thread_torch()
    splits.assert_disjoint(ids, list(X_test.index), what="learning-curve subset vs test")
    pipe = build(params, seed).fit(X_train.loc[ids], Y_train.loc[ids], labels.loc[ids])
    protocol.check_fitted_on(pipe, ids)
    return pipe.transform_targets(Y_test), pipe.predict(X_test), protocol.prep_counts(pipe)


def _verify_subsets(
    subsets: pd.DataFrame, sample_ids: Sequence[str], cfg: Mapping[str, Any]
) -> None:
    unknown = set(subsets["sample_id"]) - set(sample_ids)
    if unknown:
        raise ValueError(f"{len(unknown)} saved learning-curve ids are not in this cohort")
    sizes = learning_curve_sizes(cfg, len(sample_ids))
    expected = {(n, s) for n in sizes for s in model_seeds(cfg)}
    saved = set(zip(subsets["n_train"].astype(int), subsets["seed"].astype(int), strict=True))
    if saved != expected:
        raise ValueError(
            "saved learning-curve subsets do not match the config; "
            "delete the file to rebuild"
        )


# --- shared model builders (E0 ridge, E1 MLP) ------------------------------------------------
# Module-level functions so `functools.partial(build_ridge, cfg=..., target_set=...)` pickles
# cleanly into joblib workers.


def ridge_regressor(cfg: Mapping[str, Any]) -> protocol.SklearnRegressor:
    """Multi-output ridge; GCV picks alpha per metabolite using training rows only."""
    from sklearn.linear_model import RidgeCV

    spec = cfg["ridge"]
    alphas = np.logspace(spec["log10_alpha_min"], spec["log10_alpha_max"], int(spec["n_alphas"]))
    return protocol.SklearnRegressor(
        RidgeCV(alphas=alphas, alpha_per_target=bool(spec["alpha_per_target"]))
    )


def build_ridge(
    params: Mapping[str, Any],
    seed: int,
    *,
    cfg: Mapping[str, Any],
    target_set: str,
    candidates: Sequence[str] | None = None,
) -> protocol.M2MPipeline:
    return protocol.make_pipeline(
        microbe_prep(cfg),
        metabolite_prep(cfg, target_set, candidates),
        ridge_regressor(cfg),
        cfg,
        seed,
    )


def mlp_regressor(params: Mapping[str, Any], seed: int, cfg: Mapping[str, Any]) -> Any:
    """E1 MLP with the grid point `params` (lr, dropout, weight_decay)."""
    from m2m.train import TorchMLPRegressor, TrainConfig

    train_cfg = TrainConfig.from_dict(
        {**cfg["mlp"]["train"], "lr": params["lr"], "weight_decay": params["weight_decay"]}
    )
    return TorchMLPRegressor(
        hidden=tuple(cfg["mlp"]["hidden"]),
        dropout=float(params["dropout"]),
        train_cfg=train_cfg,
        seed=int(seed),
    )


def build_mlp(
    params: Mapping[str, Any],
    seed: int,
    *,
    cfg: Mapping[str, Any],
    target_set: str,
    candidates: Sequence[str] | None = None,
) -> protocol.M2MPipeline:
    return protocol.make_pipeline(
        microbe_prep(cfg),
        metabolite_prep(cfg, target_set, candidates),
        mlp_regressor(params, seed, cfg),
        cfg,
        seed,
    )
