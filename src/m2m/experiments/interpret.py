"""E6 interpretation: which metabolites are predictable, and which taxa drive them (brief Q4).

Interpretation never opens the external cohort (DESIGN.md decision #14):

- ``importance`` (PRISM only): ridge, and the E1 MLP with its saved T466 hyperparameters as a
  second view, are refit inside the saved PRISM folds. The metabolites ridge predicts best in
  cross-validation, plus the best one of each configured chemical class (``named_classes``, e.g.
  bile acids) that the top list misses, are explained by permutation importance: in each fold,
  one input taxon's *preprocessed* column is shuffled across the held-out samples and the drop in
  that fold's ρ is averaged over shuffles, seeds and folds. Ridge's standardised coefficients
  cross-check the same (metabolite, taxon) pairs: the sign gives the direction, the |coef| rank
  should resemble the importance rank.
- ``classes``: ρ by chemical class, read from the per-metabolite external ρ files that the final
  evaluations already wrote. Nothing is refit and no cohort is opened.

    python -m m2m.run configs/e6_interpret.yaml [--stage importance|classes] [--n-jobs N]
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from m2m import data, evaluate, paths, protocol, splits, tracking
from m2m.experiments import common
from m2m.experiments import mlp as e1

STAGES = ("importance", "classes")
RIDGE = "ridge"
MLP = "mlp"
# Ridge has no randomness (closed-form fit, no early-stopping split): one seed describes it.
RIDGE_SEEDS = (0,)
# Rows per predict call when scoring shuffles. Larger blocks are no faster, but a worker keeps
# its peak allocation, and loky restarts workers whose memory grows by more than 300 MB.
MAX_PREDICT_ROWS = 512

OUTPUT_DIR = "interpretation"
TOP_METABOLITES_CSV = "top_metabolites.csv"
IMPORTANCE_CSV = "permutation_importance.csv"
RHO_BY_CLASS_CSV = "rho_by_class.csv"
SEED_RHO_PREFIX = "rho_seed"
COMPOUND_NAME_COLUMN = "Compound Name"  # D1 metabolome_annotation.csv
CHEMICAL_CLASS_COLUMN = "Putative.Chemical.Class"  # D2 Franzosa mtb.map.tsv
HMDB_COLUMN = "HMDB"
# `selected_by` values: why a metabolite is explained. "top{n}": among the n best by ridge CV ρ;
# "class:<name>": the best metabolite of a named chemical class that the top list does not hold.
TOP_SELECTION = "top{n}"
CLASS_PREFIX = "class:"

TOP_METABOLITE_COLUMNS = [
    "model", "metabolite", "compound_name", "chemical_class", "hmdb", "cv_rho", "selected_by",
]
IMPORTANCE_COLUMNS = [
    "model", "metabolite", "compound_name", "selected_by", "taxon", "importance_mean",
    "importance_sd", "rank", "ridge_coef", "ridge_coef_rank",
]
CLASS_COLUMNS = [
    "model", "chemical_class", "n_metabolites", "mean_rho", "median_rho", "sd_rho",
    "frac_above_null",
]


def run(cfg: Mapping[str, Any], *, stage: str | None = None, n_jobs: int = 1) -> None:
    """Run the E6 stages in order (importance → classes), or only the requested ones."""
    # A dry run must never write into the committed results/. `importance` is refused there by
    # common.load_d1_train, but `classes` loads no cohort, so nothing else would check it.
    common.dry_run()
    for name in common.stages_to_run(stage, STAGES):
        if name == "importance":
            run_importance(cfg, n_jobs=n_jobs)
        else:
            run_classes(cfg)
        _log_stage(cfg, name)


def _log_stage(cfg: Mapping[str, Any], stage: str) -> None:
    """One runs.csv row (and MLflow run) per stage, counting what it wrote."""
    outputs = {"importance": ("top_metabolites.csv", "permutation_importance.csv"),
               "classes": ("rho_by_class.csv",)}[stage]
    folder = paths.RESULTS / "interpretation"
    metrics = {
        f"rows_{name.removesuffix('.csv')}": len(pd.read_csv(folder / name))
        for name in outputs if (folder / name).exists()
    }
    tracking.log_run(
        experiment="e6", run_name=f"e6|{stage}", params=dict(cfg.get("importance", {})),
        metrics=metrics, tags={"model": "interpretation", "stage": stage}, cfg=cfg,
    )


# --------------------------------------------------------------------------------------------
# Stage 1: permutation importance inside the PRISM folds
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelView:
    """A model to explain: its pipeline builder, fixed hyperparameters and the seeds per fold."""

    name: str
    build: protocol.PipelineBuilder
    params: Mapping[str, Any]
    seeds: tuple[int, ...]


@dataclass
class FoldImportance:
    """One fit (fold × seed): held-out ρ, and how much it drops when each taxon is shuffled."""

    fold: int
    seed: int
    baseline_rho: pd.Series  # metabolite → held-out ρ (NaN → 0)
    importance: pd.DataFrame  # taxa × metabolites: baseline ρ − shuffled ρ, mean over shuffles
    coefficients: pd.DataFrame | None  # metabolites × taxa; linear regressors only


@dataclass
class Explanation:
    """One model's cross-validated view of the selected metabolites."""

    cv_rho: pd.Series  # metabolite → mean over folds of the held-out ρ
    importance: pd.DataFrame  # long: metabolite, taxon, importance_mean, importance_sd
    coefficients: pd.DataFrame | None  # metabolites × taxa, mean over folds; linear only


def run_importance(
    cfg: Mapping[str, Any], *, n_jobs: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explain the metabolites ridge predicts best, using PRISM cross-validation only.

    The top `n_top_metabolites` by ridge CV ρ are explained, then the best metabolite of each
    `named_classes` entry that the top list does not already hold. Writes
    results/interpretation/top_metabolites.csv and permutation_importance.csv (column
    `selected_by` says which rule chose each metabolite), prints a ranked summary (metabolite →
    top taxa) to compare with the literature, and returns both tables.
    """
    spec = cfg["interpret"]
    settings = spec["importance"]
    target_set = str(spec["target_set"])
    info = metabolite_info()
    # Checked before any data is loaded or model fit: a misspelt class would explain nothing.
    named_classes = check_named_classes(settings["named_classes"], info["chemical_class"])
    X, Y, labels = common.load_d1_train()
    folds = common.d1_folds(labels, cfg)
    # Resolved once in the parent: worker processes then never re-read the annotation files.
    candidates = common.target_candidates(target_set)

    ridge = ModelView(
        name=RIDGE,
        build=functools.partial(
            common.build_ridge, cfg=cfg, target_set=target_set, candidates=candidates
        ),
        params={},
        seeds=RIDGE_SEEDS,
    )
    _, scores = protocol.cross_validate(
        ridge.build, ridge.params, X, Y, labels, folds, cfg, seed=RIDGE_SEEDS[0], n_jobs=n_jobs
    )
    cv_table = cv_rho_by_metabolite(scores)
    # Several LC-MS features can carry one compound (e.g. two methods, or unresolved isomers).
    # The rule is applied once to the whole ranking, so the class picks obey it too.
    compound = info["hmdb"] if settings["one_feature_per_hmdb"] else None
    ranked = rank_metabolites(cv_table, compound)
    n_top = int(settings["n_top_metabolites"])
    class_best = best_of_classes(ranked, info["chemical_class"], named_classes)
    selected_by = selection_labels(ranked[:n_top], n_top, class_best)
    for line in class_pick_notes(class_best, selected_by, info["compound_name"]):
        print(line)
    metabolites = selected_by.index.tolist()

    views = [ridge]
    mlp = mlp_view(cfg, target_set, candidates)
    if mlp is not None:
        views.append(mlp)
    explained = {
        view.name: explain(view, metabolites, X, Y, labels, folds, settings, n_jobs=n_jobs)
        for view in views
    }
    _check_same_cv_rho(explained[RIDGE].cv_rho, cv_table.loc[metabolites, "cv_rho"])

    k = int(settings["top_k_taxa"])
    coefficients = explained[RIDGE].coefficients
    top_table = pd.concat(
        [
            top_metabolite_rows(name, e.cv_rho, info, selected_by)
            for name, e in explained.items()
        ],
        ignore_index=True,
    )
    importance_table = pd.concat(
        [
            importance_rows(name, top_taxa(e.importance, k, coefficients), info, selected_by)
            for name, e in explained.items()
        ],
        ignore_index=True,
    )
    out_dir = paths.RESULTS / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    top_table.to_csv(out_dir / TOP_METABOLITES_CSV, index=False)
    importance_table.to_csv(out_dir / IMPORTANCE_CSV, index=False)
    for line in summary_lines(top_table, importance_table, k=k):
        print(line)
    print(f"[e6] wrote {out_dir / TOP_METABOLITES_CSV} and {out_dir / IMPORTANCE_CSV}")
    return top_table, importance_table


def mlp_view(
    cfg: Mapping[str, Any], target_set: str, candidates: Sequence[str] | None
) -> ModelView | None:
    """The E1 MLP with its saved hyperparameters, or None (with a message) before E1 has run.

    The selection must pass the check E1 itself applies before evaluating it: a point of the
    current mlp.grid, chosen by a search of exactly that grid. Explaining a selection E1 would
    refuse would describe, under the name "mlp", a model that E1 never scored.
    """
    try:
        protocol.load_selection(e1.SELECTION, target_set)
    except FileNotFoundError as missing:
        print(f"[e6] skipping the MLP view: {missing}")
        return None
    selection = e1.load_checked_selection(cfg, target_set)
    build = functools.partial(
        common.build_mlp, cfg=cfg, target_set=target_set, candidates=candidates
    )
    # Every model seed is fit in every fold: a single initialisation would make the
    # importances partly a property of that draw.
    return ModelView(MLP, build, dict(selection["params"]), tuple(common.model_seeds(cfg)))


def cv_rho_by_metabolite(scores: Sequence[protocol.FoldScore]) -> pd.DataFrame:
    """Per metabolite: mean over folds of the held-out ρ (NaN → 0) and how many folds scored it.

    `scores` are the fold scores of one configuration. The zero filter is refit on every training
    fold, so a metabolite close to its threshold can be missing from some folds; `in_every_fold`
    marks the ones scored in all of them.
    """
    by_fold = pd.DataFrame({score.fold: score.rho.fillna(0.0) for score in scores})
    n_folds = by_fold.notna().sum(axis=1)
    return pd.DataFrame(
        {
            "cv_rho": by_fold.mean(axis=1),
            "n_folds": n_folds,
            "in_every_fold": n_folds == len(scores),
        }
    )


def rank_metabolites(cv_table: pd.DataFrame, compound: pd.Series | None = None) -> list[str]:
    """The metabolites scored in every fold, highest CV ρ first.

    With `compound` (feature → compound ID, e.g. HMDB), only the best feature of each compound is
    kept; features without an ID count as distinct compounds. Ties are broken by the table order
    (stable sort), so the ranking is deterministic.
    """
    eligible = cv_table[cv_table["in_every_fold"]]
    ranked = eligible.sort_values("cv_rho", ascending=False, kind="stable")
    if compound is not None:
        ids = compound.reindex(ranked.index)
        ids = ids.where(ids.notna(), ranked.index.to_series())
        ranked = ranked[~ids.duplicated().to_numpy()]
    return ranked.index.tolist()


def select_top_metabolites(
    cv_table: pd.DataFrame, n: int, compound: pd.Series | None = None
) -> list[str]:
    """The `n` best metabolites of `rank_metabolites`."""
    return rank_metabolites(cv_table, compound)[:n]


def check_named_classes(named: Sequence[str], classes: pd.Series) -> list[str]:
    """The configured class names, each required to be an exact chemical-class string.

    `classes` maps features to their Putative.Chemical.Class. A misspelt or truncated name (e.g.
    "Bile acids" for "Bile acids, alcohols and derivatives") would match no metabolite and
    silently explain nothing, so it is refused, as is a name listed twice.
    """
    names = [str(name) for name in named]
    known = set(classes.dropna())
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            f"named_classes {unknown} are not {CHEMICAL_CLASS_COLUMN} values of the Franzosa "
            "metabolite map: use the exact strings"
        )
    if len(set(names)) < len(names):
        raise ValueError(f"named_classes lists a class more than once: {names}")
    return names


def best_of_classes(
    ranked: Sequence[str], classes: pd.Series, named: Sequence[str]
) -> dict[str, str | None]:
    """Each named class → its highest-ranked metabolite in `ranked`, or None if it has none.

    `ranked` comes from `rank_metabolites`, so only metabolites scored in every fold qualify and,
    with the one-feature-per-compound rule, a compound measured in two classes counts only in the
    class of its best feature.
    """
    class_of = classes.reindex(list(ranked))
    return {
        name: next((m for m in ranked if class_of.loc[m] == name), None) for name in named
    }


def selection_labels(
    top: Sequence[str], n_top: int, class_best: Mapping[str, str | None]
) -> pd.Series:
    """metabolite → `selected_by`, in the order explained: the top list, then the class picks.

    A class whose best metabolite is already in the top list adds nothing: that metabolite keeps
    its top label. Since `top` is the head of the same ranking, a class that does add a pick has
    no metabolite in the top list.
    """
    labels = {metabolite: TOP_SELECTION.format(n=n_top) for metabolite in top}
    for name, metabolite in class_best.items():
        if metabolite is not None and metabolite not in labels:
            labels[metabolite] = f"{CLASS_PREFIX}{name}"
    return pd.Series(labels, name="selected_by", dtype=object)


def class_pick_notes(
    class_best: Mapping[str, str | None], selected_by: pd.Series, names: pd.Series
) -> list[str]:
    """One line per named class that adds no metabolite, saying why."""
    notes = []
    for name, metabolite in class_best.items():
        if metabolite is None:
            notes.append(
                f"[e6] class {name!r}: none of its metabolites is a target scored in every "
                "PRISM fold"
            )
        elif selected_by[metabolite] != f"{CLASS_PREFIX}{name}":
            shown = _compound(names.get(metabolite), metabolite)
            notes.append(
                f"[e6] class {name!r}: its best metabolite, {shown}, is already explained as "
                f"{selected_by[metabolite]}"
            )
    return notes


def _compound(name: Any, feature: str) -> str:
    """'urobilin (C18-neg_Cluster_2021)', or the feature ID alone when it has no name.

    Names are stripped: some D1 annotations end in a space ('linoleoyl ethanolamide ').
    """
    if isinstance(name, str) and name.strip():
        return f"{name.strip()} ({feature})"
    return str(feature)


def explain(
    view: ModelView,
    metabolites: Sequence[str],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    labels: pd.Series,
    folds: pd.DataFrame,
    settings: Mapping[str, Any],
    *,
    n_jobs: int = 1,
) -> Explanation:
    """Permutation importance of one model for `metabolites`, aggregated over the folds."""
    units = permutation_importance_cv(
        view.build,
        view.params,
        X,
        Y,
        labels,
        folds,
        metabolites=metabolites,
        seeds=view.seeds,
        n_repeats=int(settings["n_repeats"]),
        permutation_seed=int(settings["permutation_seed"]),
        n_jobs=n_jobs,
    )
    return Explanation(
        cv_rho=fold_mean_rho(units),
        importance=aggregate_importance(units),
        coefficients=mean_coefficients(units),
    )


def permutation_importance_cv(
    build: protocol.PipelineBuilder,
    params: Mapping[str, Any],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    labels: pd.Series,
    folds: pd.DataFrame,
    *,
    metabolites: Sequence[str],
    seeds: Sequence[int],
    n_repeats: int,
    permutation_seed: int,
    n_jobs: int = 1,
) -> list[FoldImportance]:
    """One fit per (fold, seed) on that fold's training rows; then each taxon is shuffled.

    In fold f every taxon, model and seed sees the same `n_repeats` shuffles of the held-out rows
    (drawn from `permutation_seed` and f): differences between taxa then come from the model, not
    from the draw. Results do not depend on `n_jobs`.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    fold_ids = sorted(int(f) for f in folds["fold"].unique())
    return Parallel(n_jobs=n_jobs)(
        delayed(_fold_importance)(
            build,
            dict(params),
            int(seed),
            fold,
            X,
            Y,
            labels,
            folds,
            list(metabolites),
            n_repeats,
            permutation_seed,
        )
        for fold in fold_ids
        for seed in seeds
    )


def _fold_importance(
    build: protocol.PipelineBuilder,
    params: dict[str, Any],
    seed: int,
    fold: int,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    labels: pd.Series,
    folds: pd.DataFrame,
    metabolites: list[str],
    n_repeats: int,
    permutation_seed: int,
) -> FoldImportance:
    protocol.single_thread_torch()
    train_ids, test_ids = splits.fold_train_test(folds, fold)
    splits.assert_disjoint(train_ids, test_ids, what=f"CV fold {fold}")
    pipe = build(params, seed).fit(X.loc[train_ids], Y.loc[train_ids], labels.loc[train_ids])
    protocol.check_fitted_on(pipe, train_ids)
    targets = _target_positions(pipe, metabolites, fold)
    truth = pipe.transform_targets(Y.loc[test_ids])[metabolites].to_numpy()
    # Shuffle the preprocessed column: shuffling a raw abundance would also move every other
    # taxon's CLR value (re-closure and the geometric mean), mixing the taxon with all others.
    inputs = pipe.x_prep.transform(X.loc[test_ids])
    values = inputs.to_numpy()

    def predict(rows: np.ndarray) -> np.ndarray:
        return pipe.regressor.predict(rows)[:, targets]

    baseline = _rho(truth, predict(values))
    shuffles = _shuffles(len(test_ids), n_repeats, permutation_seed, fold)
    drops = [
        baseline - _mean_shuffled_rho(predict, values, column, shuffles, truth)
        for column in range(values.shape[1])
    ]
    return FoldImportance(
        fold=fold,
        seed=seed,
        baseline_rho=pd.Series(baseline, index=metabolites),
        importance=pd.DataFrame(np.vstack(drops), index=inputs.columns, columns=metabolites),
        coefficients=_linear_coefficients(pipe, metabolites),
    )


def _target_positions(
    pipe: protocol.M2MPipeline, metabolites: Sequence[str], fold: int
) -> np.ndarray:
    positions = pd.Index(pipe.y_prep.features_).get_indexer(metabolites)
    if (positions < 0).any():
        dropped = [m for m, p in zip(metabolites, positions, strict=True) if p < 0]
        raise ValueError(
            f"fold {fold}: the zero filter dropped {dropped}; explain only metabolites that are "
            "scored in every fold"
        )
    return positions


def _shuffles(n_rows: int, n_repeats: int, seed: int, fold: int) -> np.ndarray:
    """(n_repeats, n_rows) row orders of one fold, shared by every taxon, model and seed."""
    rng = np.random.default_rng([seed, fold])
    return np.stack([rng.permutation(n_rows) for _ in range(n_repeats)])


def _mean_shuffled_rho(
    predict: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray,
    column: int,
    shuffles: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """Mean ρ per metabolite over the shuffles of one input column, several shuffles per call."""
    n_rows, n_inputs = values.shape
    per_call = max(1, MAX_PREDICT_ROWS // n_rows)
    rhos = []
    for start in range(0, len(shuffles), per_call):
        block = shuffles[start : start + per_call]
        batch = np.repeat(values[np.newaxis], len(block), axis=0)
        batch[:, :, column] = values[block, column]
        predictions = predict(batch.reshape(-1, n_inputs)).reshape(len(block), n_rows, -1)
        rhos.extend(_rho(truth, prediction) for prediction in predictions)
    return np.mean(rhos, axis=0)


def _rho(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Held-out ρ per metabolite; an undefined ρ (constant column) counts as 0, as in summaries."""
    return np.nan_to_num(evaluate.spearman_per_feature(truth, prediction), nan=0.0)


def _linear_coefficients(
    pipe: protocol.M2MPipeline, metabolites: Sequence[str]
) -> pd.DataFrame | None:
    """A linear regressor's coefficients (metabolites × taxa); None for any other regressor.

    The pipeline z-scores inputs and targets (base.yaml `standardize: true`), so these are
    standardised coefficients: comparable across taxa and metabolites.
    """
    coef = getattr(getattr(pipe.regressor, "estimator", None), "coef_", None)
    if coef is None:
        return None
    table = pd.DataFrame(
        np.atleast_2d(coef), index=pipe.y_prep.features_, columns=pipe.x_prep.features_
    )
    return table.loc[list(metabolites)]


# --- aggregation over seeds and folds ---------------------------------------------------------


def fold_mean_rho(units: Sequence[FoldImportance]) -> pd.Series:
    """Mean over folds of the held-out ρ per metabolite (the seeds of a fold averaged first)."""
    return pd.concat(_mean_over_seeds(units, "baseline_rho"), axis=1).mean(axis=1)


def aggregate_importance(units: Sequence[FoldImportance]) -> pd.DataFrame:
    """Long table (metabolite, taxon, importance_mean, importance_sd) over the folds.

    The seeds of a fold are averaged first, so the SD (ddof=1) is the fold-to-fold spread for
    every model. The prevalence filter is refit per fold: a taxon a fold dropped is never seen by
    that fold's model, so its importance there is exactly 0 (averaging only over the folds that
    kept it would overstate rarely kept taxa).
    """
    per_fold = _mean_over_seeds(units, "importance")
    taxa = _union([frame.index for frame in per_fold])
    stacked = np.stack([frame.reindex(taxa, fill_value=0.0).to_numpy() for frame in per_fold])
    mean = stacked.mean(axis=0)
    sd = stacked.std(axis=0, ddof=1) if len(per_fold) > 1 else np.full_like(mean, np.nan)
    metabolites = list(per_fold[0].columns)
    # Metabolite-major rows: (metabolite 1, every taxon), (metabolite 2, every taxon), ...
    return pd.DataFrame(
        {
            "metabolite": np.repeat(metabolites, len(taxa)),
            "taxon": np.tile(np.asarray(taxa, dtype=object), len(metabolites)),
            "importance_mean": mean.T.ravel(),
            "importance_sd": sd.T.ravel(),
        }
    )


def mean_coefficients(units: Sequence[FoldImportance]) -> pd.DataFrame | None:
    """Mean over folds of a linear model's coefficients (metabolites × taxa); None otherwise.

    As for the importance, a taxon dropped by a fold's prevalence filter counts as 0 there.
    """
    if any(unit.coefficients is None for unit in units):
        return None
    per_fold = _mean_over_seeds(units, "coefficients")
    taxa = _union([frame.columns for frame in per_fold])
    stacked = np.stack(
        [frame.reindex(columns=taxa, fill_value=0.0).to_numpy() for frame in per_fold]
    )
    return pd.DataFrame(stacked.mean(axis=0), index=per_fold[0].index, columns=taxa)


def _mean_over_seeds(units: Sequence[FoldImportance], attribute: str) -> list[Any]:
    """`attribute` of every fold, averaged over that fold's seeds; folds in ascending order."""
    by_fold: dict[int, list[Any]] = {}
    for unit in units:
        by_fold.setdefault(unit.fold, []).append(getattr(unit, attribute))
    # The fits of one fold share their taxa and metabolites, so label-aligned sums are exact.
    return [sum(parts[1:], start=parts[0]) / len(parts) for _, parts in sorted(by_fold.items())]


def _union(indexes: Sequence[pd.Index]) -> pd.Index:
    """Every label of `indexes`, in order of first appearance."""
    return pd.Index(list(dict.fromkeys(label for index in indexes for label in index)))


def _check_same_cv_rho(explained: pd.Series, cross_validated: pd.Series) -> None:
    """Tripwire: the importance fits must reproduce the CV ρ the metabolites were chosen by."""
    if not np.allclose(explained.loc[cross_validated.index], cross_validated, atol=1e-9):
        raise RuntimeError(
            "ridge's held-out ρ in the importance fits differs from its cross-validated ρ: "
            "the two passes did not use the same folds or fits"
        )


# --- tables and the printed summary -----------------------------------------------------------


def top_taxa(importance: pd.DataFrame, k: int, coefficients: pd.DataFrame) -> pd.DataFrame:
    """The k most important taxa of each metabolite (rank 1 = largest drop in ρ).

    Each pair also gets ridge's coefficient for the same (metabolite, taxon) and that
    coefficient's rank by |value| among all taxa (1 = largest): the cross-check of direction and
    rank. Ties in importance are broken by taxon name, so the table is deterministic.
    """
    coef_rank = coefficients.abs().rank(axis=1, ascending=False, method="min")
    rows = []
    for metabolite, group in importance.groupby("metabolite", sort=False):
        best = group.sort_values(["importance_mean", "taxon"], ascending=[False, True]).head(k)
        rows.append(
            best.assign(
                rank=np.arange(1, len(best) + 1),
                ridge_coef=coefficients.loc[metabolite].reindex(best["taxon"]).to_numpy(),
                ridge_coef_rank=coef_rank.loc[metabolite].reindex(best["taxon"]).to_numpy(),
            )
        )
    pairs = pd.concat(rows, ignore_index=True)
    return pairs.assign(ridge_coef_rank=pairs["ridge_coef_rank"].astype("Int64"))


def metabolite_info() -> pd.DataFrame:
    """compound_name (D1 annotation), chemical_class and hmdb (D2 Franzosa map) by D1 feature ID.

    Both files annotate features, not samples: reading them never touches a cohort.
    """
    names = data.load_annotation()[COMPOUND_NAME_COLUMN].rename("compound_name")
    mtb_map = data.load_d2_mtb_map(data.TARGET_DATASET)
    feature_ids = pd.Index([data.franzosa_feature_id(c) for c in mtb_map.index])
    if feature_ids.has_duplicates:
        raise ValueError("the Franzosa metabolite map has several rows for one D1 feature ID")
    described = pd.DataFrame(
        {
            "chemical_class": mtb_map[CHEMICAL_CLASS_COLUMN].to_numpy(),
            "hmdb": mtb_map[HMDB_COLUMN].to_numpy(),
        },
        index=feature_ids,
    )
    return described.join(names, how="outer")[["compound_name", "chemical_class", "hmdb"]]


def top_metabolite_rows(
    model: str, cv_rho: pd.Series, info: pd.DataFrame, selected_by: pd.Series
) -> pd.DataFrame:
    """top_metabolites.csv rows of one model, in the order of `cv_rho` (the order explained)."""
    described = info.reindex(cv_rho.index)
    return pd.DataFrame(
        {
            "model": model,
            "metabolite": cv_rho.index.to_numpy(),
            "compound_name": described["compound_name"].to_numpy(),
            "chemical_class": described["chemical_class"].to_numpy(),
            "hmdb": described["hmdb"].to_numpy(),
            "cv_rho": cv_rho.to_numpy(),
            "selected_by": selected_by.reindex(cv_rho.index).to_numpy(),
        }
    )[TOP_METABOLITE_COLUMNS]


def importance_rows(
    model: str, pairs: pd.DataFrame, info: pd.DataFrame, selected_by: pd.Series
) -> pd.DataFrame:
    """permutation_importance.csv rows of one model."""
    names = info["compound_name"].reindex(pairs["metabolite"]).to_numpy()
    reasons = selected_by.reindex(pairs["metabolite"]).to_numpy()
    return pairs.assign(model=model, compound_name=names, selected_by=reasons)[IMPORTANCE_COLUMNS]


def summary_lines(
    top_table: pd.DataFrame, importance_table: pd.DataFrame, *, k: int
) -> list[str]:
    """Ranked text summary per model: metabolite → its top taxa (importance, ridge sign).

    The top list is numbered in ridge's CV order; each named-class pick follows on a "class" line.
    """
    lines = []
    for model, metabolites in top_table.groupby("model", sort=False):
        pairs = importance_table[importance_table["model"] == model]
        lines.append(
            f"[e6] {model}: best-predicted metabolites (PRISM CV ρ), then the best of each named "
            f"class → top {k} taxa by permutation importance (Δρ, sign of ridge's coefficient)"
        )
        picked = metabolites["selected_by"].astype(str).str.startswith(CLASS_PREFIX).to_numpy()
        for position, row in enumerate(metabolites[~picked].itertuples(index=False), start=1):
            lines.append(f"  {position:2d}. {_metabolite_line(row, pairs)}")
        for row in metabolites[picked].itertuples(index=False):
            lines.append(f"  class {_metabolite_line(row, pairs)}")
        agree = float((pairs["ridge_coef_rank"] <= k).mean())
        noisy = float((pairs["importance_sd"] > pairs["importance_mean"]).mean())
        lines.append(
            f"      {agree:.0%} of these pairs are also ridge's top {k} by |coefficient|; "
            f"for {noisy:.0%} the SD over folds exceeds the mean importance"
        )
    return lines


def _metabolite_line(row: Any, pairs: pd.DataFrame) -> str:
    """One summary line: compound (feature) [class] ρ → each top taxon's Δρ and ridge sign.

    E.g. 'urobilin (C18-neg_Cluster_2021) [Tetrapyrroles and derivatives] ρ=0.834 → ...'.
    """
    mine = pairs[pairs["metabolite"] == row.metabolite]
    taxa = ", ".join(
        f"{p.taxon} {p.importance_mean:.3f}({_sign(p.ridge_coef)})"
        for p in mine.itertuples(index=False)
    )
    return (
        f"{_compound(row.compound_name, row.metabolite)} [{_text(row.chemical_class)}] "
        f"ρ={row.cv_rho:.3f} → {taxa}"
    )


def _sign(value: float) -> str:
    if pd.isna(value) or value == 0:
        return "0"
    return "+" if value > 0 else "-"


def _text(value: Any) -> str:
    return "unclassified" if pd.isna(value) else str(value)


# --------------------------------------------------------------------------------------------
# Stage 2: external ρ by chemical class (reads files only)
# --------------------------------------------------------------------------------------------


def run_classes(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """ρ by chemical class for every model whose external per-metabolite ρ is already written.

    Reads summary.csv, the per-metabolite ρ files, the null threshold and the metabolite
    annotation: nothing is refit and no cohort is opened. Writes rho_by_class.csv.
    """
    spec = cfg["interpret"]
    settings = spec["classes"]
    target_set = str(spec["target_set"])
    data_labels = [str(label) for label in settings["data"]]
    models = summary_models(data_labels, target_set)
    null_threshold = _null_threshold_or_none(str(settings["null_data"]), target_set)
    classes = metabolite_info()["chemical_class"]
    tables = []
    for model in models:
        path = per_metabolite_path(model, target_set)
        if not path.exists():
            print(f"[e6] {path} is missing: {model} is left out of the class table")
            continue
        table = rho_by_class(
            seed_rho(pd.read_csv(path, index_col=0)),
            classes,
            min_class_size=int(settings["min_class_size"]),
            null_threshold=null_threshold,
            pooled_label=str(settings["pooled_label"]),
            unclassified_label=str(settings["unclassified_label"]),
        )
        tables.append(table.assign(model=model))
    if not tables:
        raise FileNotFoundError(f"no per-metabolite ρ file for any {target_set} model in {models}")
    out = pd.concat(tables, ignore_index=True)[CLASS_COLUMNS]
    path = paths.RESULTS / OUTPUT_DIR / RHO_BY_CLASS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print("[e6] mean external ρ by chemical class (columns: models)")
    print(_class_pivot(out).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"[e6] wrote {path}")
    return out


def summary_models(data_labels: Sequence[str], target_set: str) -> list[str]:
    """Models with a summary.csv row for one of these data sources and this target set.

    In file order. The data label tells which inputs and null a model used: e.g. "D1" (E0, E1)
    and "D1+MGM" (E2, E3) are scored against the D1 null, "D2" (E4) against its own.
    """
    path = paths.RESULTS / tracking.SUMMARY_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the final evaluations before E6 classes")
    summary = pd.read_csv(path)
    chosen = summary[summary["data"].isin(data_labels) & (summary["target_set"] == target_set)]
    if chosen.empty:
        raise ValueError(f"{path} has no {target_set} model with data in {list(data_labels)}")
    return chosen["model"].astype(str).drop_duplicates().tolist()


def per_metabolite_path(model: str, target_set: str) -> Path:
    """The per-metabolite external ρ file of one model (protocol.write_per_metabolite)."""
    return protocol.per_metabolite_path(model, target_set)


def seed_rho(per_metabolite: pd.DataFrame) -> pd.DataFrame:
    """The per-seed ρ columns (metabolites × seeds) of a per-metabolite ρ file, NaN kept."""
    seed_columns = [c for c in per_metabolite.columns if str(c).startswith(SEED_RHO_PREFIX)]
    if not seed_columns:
        raise ValueError(f"no {SEED_RHO_PREFIX}* columns: not a per-metabolite ρ table")
    return per_metabolite[seed_columns]


def metabolite_rho(per_metabolite: pd.DataFrame) -> pd.Series:
    """External ρ per metabolite as the summaries count it: NaN → 0 in each seed, then the mean.

    The file's `rho_mean` skips NaN seeds (pandas' default), so it differs from this only for a
    metabolite that was constant in some seeds but not in others; counting those seeds as 0 makes
    the class means average back exactly to summary.csv's mean ρ.
    """
    return seed_rho(per_metabolite).fillna(0.0).mean(axis=1)


def rho_by_class(
    rho: pd.Series | pd.DataFrame,
    classes: pd.Series,
    *,
    min_class_size: int,
    null_threshold: float | None,
    pooled_label: str,
    unclassified_label: str,
) -> pd.DataFrame:
    """Summary of per-metabolite ρ by chemical class, best class first.

    `rho` is one ρ per metabolite or a metabolites × seeds table (`seed_rho`); NaN ρ count as 0
    in each seed. mean/median/sd describe each metabolite's mean over seeds. frac_above_null is
    the share of the class with ρ strictly above the permutation-null threshold in each seed,
    averaged over seeds as summary.csv averages n_above_null (NaN without a threshold). Both
    decompose the summary: n_metabolites × mean_rho (or × frac_above_null) summed over classes
    gives n_features × mean_rho_mean (or n_above_null_mean). Metabolites without a class form
    `unclassified_label`; every group with fewer than `min_class_size` metabolites is pooled
    into `pooled_label`.
    """
    by_seed = (rho.to_frame() if isinstance(rho, pd.Series) else rho).fillna(0.0)
    scored = by_seed.mean(axis=1)
    no_null = null_threshold is None or np.isnan(null_threshold)
    # Per metabolite, the share of seeds above the null. Comparing the seed-mean ρ with the
    # threshold instead would not add up to summary.csv's n_above_null for multi-seed models.
    above = None if no_null else (by_seed > null_threshold).mean(axis=1)
    labels = classes.reindex(scored.index).fillna(unclassified_label)
    sizes = labels.value_counts()
    labels = labels.where(labels.map(sizes) >= min_class_size, pooled_label)
    rows = [
        {
            "chemical_class": label,
            "n_metabolites": len(group),
            "mean_rho": float(group.mean()),
            "median_rho": float(group.median()),
            "sd_rho": float(group.std(ddof=1)),
            "frac_above_null": np.nan if above is None else float(above.loc[group.index].mean()),
        }
        for label, group in scored.groupby(labels)
    ]
    table = pd.DataFrame(rows)
    return table.sort_values("mean_rho", ascending=False, kind="stable", ignore_index=True)


def _null_threshold_or_none(data_label: str, target_set: str) -> float | None:
    try:
        return protocol.load_null_threshold(data_label, target_set)
    except (FileNotFoundError, KeyError) as missing:
        print(f"[e6] no permutation-null threshold ({missing}): frac_above_null is left empty")
        return None


def _class_pivot(table: pd.DataFrame) -> pd.DataFrame:
    """mean_rho as classes × models (with the class size), best class on average first."""
    models = table["model"].drop_duplicates().tolist()
    pivot = table.pivot(index="chemical_class", columns="model", values="mean_rho")[models]
    sizes = table.groupby("chemical_class")["n_metabolites"].max().rename("n")
    order = pivot.mean(axis=1).sort_values(ascending=False, kind="stable").index
    return pd.concat([sizes, pivot], axis=1).loc[order]
