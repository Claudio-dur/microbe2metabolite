"""E6 interpretation: planted taxa are recovered inside CV, held-out rows are never fit, and the
class summary reads finished results only (it never opens a cohort)."""

from __future__ import annotations

import copy
import functools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from m2m import data, evaluate, paths, protocol, splits
from m2m.config import load_config
from m2m.evaluate import NullResult
from m2m.experiments import common, interpret, mlp
from m2m.run import EXPERIMENT_MODULES

CFG = load_config(paths.CONFIGS / "e6_interpret.yaml")
# conftest's synthetic_cohort: met_00..met_09 depend log-linearly on taxon_00..taxon_04.
PLANTED_METABOLITES = [f"met_{j:02d}" for j in range(10)]
PLANTED_TAXA = {f"taxon_{j:02d}" for j in range(5)}


@pytest.fixture
def folds(synthetic_cohort) -> pd.DataFrame:
    labels = synthetic_cohort["labels"]
    return splits.make_stratified_folds(list(labels.index), list(labels), n_splits=5, seed=1)


@pytest.fixture
def ridge_build(synthetic_cohort) -> protocol.PipelineBuilder:
    candidates = list(synthetic_cohort["Y"].columns)
    return functools.partial(common.build_ridge, cfg=CFG, target_set="T466", candidates=candidates)


def _importance(cohort, folds, build, metabolites, **kwargs) -> list[interpret.FoldImportance]:
    settings = {"seeds": [0], "n_repeats": 10, "permutation_seed": 1, **kwargs}
    return interpret.permutation_importance_cv(
        build, {}, cohort["X"], cohort["Y"], cohort["labels"], folds,
        metabolites=metabolites, **settings,
    )


def _save_e1_selection(best: Mapping[str, float], searched: Sequence[Mapping[str, float]]) -> None:
    """Write E1's selection files as its select stage does: the winner and the grid it searched."""
    table = pd.DataFrame({
        "params": [json.dumps(dict(params), sort_keys=True) for params in searched],
        "cv_mean_rho": [0.2 if dict(params) == dict(best) else 0.1 for params in searched],
    })
    protocol.save_selection(mlp.SELECTION, "T466", best, table)


def _forbid(monkeypatch: pytest.MonkeyPatch, *targets: tuple[object, str]) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("E6 must not open this data")

    for module, name in targets:
        monkeypatch.setattr(module, name, forbidden)


def _write_annotations(
    raw: Path,
    features: Sequence[str],
    classes: Mapping[str, str],
    hmdb: Mapping[str, str] | None = None,
) -> None:
    """Minimal D1 annotation + D2 Franzosa metabolite map, laid out like data/raw.

    Feature i gets HMDB ID HMDB{i:07d} unless `hmdb` gives it another (e.g. a shared one).
    """
    d1 = raw / "MiMeNet" / "data" / "IBD"
    d1.mkdir(parents=True)
    pd.DataFrame(
        {"Retention Time": 1.0, "m/z": 100.0, "Compound Name": [f"compound {f}" for f in features],
         "Adduct": "[M+H]+"},
        index=list(features),
    ).to_csv(d1 / "metabolome_annotation.csv")
    franzosa = raw / "microbiome-metabolome-curated-data" / "data" / "processed_data"
    (franzosa / data.TARGET_DATASET).mkdir(parents=True)
    hmdb = hmdb or {}
    pd.DataFrame(
        {
            "Compound": [f"{f}: compound {f}" for f in features],
            "HMDB": [hmdb.get(f, f"HMDB{i:07d}") for i, f in enumerate(features)],
            "Putative.Chemical.Class": [classes.get(f) for f in features],
        }
    ).to_csv(franzosa / data.TARGET_DATASET / "mtb.map.tsv", sep="\t", index=False)


# --- wiring ----------------------------------------------------------------------------------


def test_config_is_wired_to_this_module() -> None:
    assert CFG["experiment"] == "e6"
    assert EXPERIMENT_MODULES["e6"] == interpret.__name__
    assert set(CFG["interpret"]) == {"target_set", "importance", "classes"}
    with pytest.raises(SystemExit):
        interpret.run(CFG, stage="no-such-stage")


def test_config_explains_one_feature_per_compound_and_a_few_named_classes() -> None:
    settings = CFG["interpret"]["importance"]
    assert settings["one_feature_per_hmdb"] is True
    named = settings["named_classes"]
    assert 2 <= len(named) <= 4 and len(set(named)) == len(named)
    assert "Bile acids, alcohols and derivatives" in named  # the brief's literature example


# --- importance ------------------------------------------------------------------------------


def test_ridge_cv_picks_the_planted_metabolites(synthetic_cohort, folds, ridge_build) -> None:
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    _, scores = protocol.cross_validate(ridge_build, {}, X, Y, labels, folds, CFG)
    cv_table = interpret.cv_rho_by_metabolite(scores)
    top = interpret.select_top_metabolites(cv_table, 10)
    assert set(top) == set(PLANTED_METABOLITES)
    assert cv_table.loc[top, "cv_rho"].is_monotonic_decreasing


def test_top_metabolites_must_be_scored_in_every_fold() -> None:
    def score(fold: int, rho: dict[str, float]) -> protocol.FoldScore:
        return protocol.FoldScore(params_index=0, fold=fold, summary={}, rho=pd.Series(rho))

    # "a" was dropped by fold 1's zero filter; "c" has an undefined ρ in fold 0 (counts as 0).
    scores = [score(0, {"a": 0.9, "b": 0.5, "c": np.nan}), score(1, {"b": 0.5, "c": 0.2})]
    cv_table = interpret.cv_rho_by_metabolite(scores)
    assert cv_table.loc["c", "cv_rho"] == pytest.approx(0.1)
    assert not cv_table.loc["a", "in_every_fold"]
    assert interpret.select_top_metabolites(cv_table, 2) == ["b", "c"]


def test_one_feature_per_compound_keeps_the_best_feature() -> None:
    cv_table = pd.DataFrame(
        {"cv_rho": [0.9, 0.8, 0.7, 0.6], "n_folds": 5, "in_every_fold": True},
        index=["a", "b", "c", "d"],
    )
    hmdb = pd.Series({"a": "HMDB1", "b": "HMDB1", "c": np.nan})  # "d" is not annotated at all
    assert interpret.select_top_metabolites(cv_table, 3) == ["a", "b", "c"]
    assert interpret.select_top_metabolites(cv_table, 3, hmdb) == ["a", "c", "d"]


def _cv_table(rho: Mapping[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"cv_rho": rho, "n_folds": 5, "in_every_fold": True})


def test_class_picks_are_each_class_best_metabolite_outside_the_top_list() -> None:
    cv_table = _cv_table({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5, "f": 0.4})
    cv_table.loc["c", "in_every_fold"] = False  # one fold's zero filter dropped it
    classes = pd.Series({"a": "A", "b": "B", "c": "C", "d": "B", "e": "D", "f": "D"})
    ranked = interpret.rank_metabolites(cv_table)
    assert ranked == ["a", "b", "d", "e", "f"]

    best = interpret.best_of_classes(ranked, classes, ["A", "B", "C", "D"])
    assert best == {"A": "a", "B": "b", "C": None, "D": "e"}
    labels = interpret.selection_labels(ranked[:1], 1, best)
    # "A" is already explained by the top list; "C" has no metabolite scored in every fold.
    assert labels.to_dict() == {"a": "top1", "b": "class:B", "e": "class:D"}
    assert labels.index.tolist() == ["a", "b", "e"]  # top list first, then classes in order
    # D1 names can end in a space; the note must not print "urobilin  (a)".
    notes = interpret.class_pick_notes(best, labels, pd.Series({"a": "urobilin "}))
    assert notes == [
        "[e6] class 'A': its best metabolite, urobilin (a), is already explained as top1",
        "[e6] class 'C': none of its metabolites is a target scored in every PRISM fold",
    ]


def test_class_picks_follow_the_one_feature_per_compound_rule() -> None:
    cv_table = _cv_table({"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6})
    # "b" and "c" measure one compound whose features fall in two classes; "b" is its best.
    hmdb = pd.Series({"a": "HMDB1", "b": "HMDB2", "c": "HMDB2", "d": "HMDB3"})
    classes = pd.Series({"a": "A", "b": "B", "c": "C", "d": "C"})
    ranked = interpret.rank_metabolites(cv_table, hmdb)
    assert interpret.best_of_classes(ranked, classes, ["B", "C"]) == {"B": "b", "C": "d"}
    # Without the rule, class C would explain the compound a second time.
    per_feature = interpret.rank_metabolites(cv_table)
    assert interpret.best_of_classes(per_feature, classes, ["C"]) == {"C": "c"}


def test_named_classes_must_be_exact_class_strings(monkeypatch, tmp_path) -> None:
    full = "Bile acids, alcohols and derivatives"
    classes = pd.Series({"a": full, "b": np.nan})
    assert interpret.check_named_classes([full], classes) == [full]
    assert interpret.check_named_classes([], classes) == []
    with pytest.raises(ValueError, match="exact strings"):
        interpret.check_named_classes(["Bile acids"], classes)
    with pytest.raises(ValueError, match="more than once"):
        interpret.check_named_classes([full, full], classes)

    # The stage refuses a misspelt class before it loads PRISM or fits anything.
    monkeypatch.setattr(paths, "DATA_RAW", tmp_path / "raw")
    _write_annotations(tmp_path / "raw", ["met_00"], {"met_00": full})
    _forbid(monkeypatch, (common, "load_d1_train"), (protocol, "cross_validate"))
    cfg = _small_cfg()
    cfg["interpret"]["importance"]["named_classes"] = ["Bile acids"]
    with pytest.raises(ValueError, match="exact strings"):
        interpret.run_importance(cfg)


def test_importance_recovers_the_planted_taxa(synthetic_cohort, folds, ridge_build) -> None:
    units = _importance(synthetic_cohort, folds, ridge_build, PLANTED_METABOLITES)
    importance = interpret.aggregate_importance(units)
    pairs = interpret.top_taxa(importance, 5, interpret.mean_coefficients(units))
    planted = pairs["taxon"].isin(PLANTED_TAXA)
    assert planted[pairs["rank"] == 1].all()
    assert planted.mean() >= 0.7  # chance level: 5 of 30 taxa
    # Ridge's own coefficients rank the same taxa first (the rank cross-check).
    assert (pairs.loc[pairs["rank"] == 1, "ridge_coef_rank"] <= 2).all()
    # Unplanted metabolites are noise: their best taxon matters far less.
    noise = interpret.aggregate_importance(_importance(synthetic_cohort, folds, ridge_build,
                                                       ["met_20", "met_30"]))
    assert noise["importance_mean"].max() < importance["importance_mean"].max() / 3


def _signed_cohort(n: int = 60, p: int = 12) -> dict[str, pd.DataFrame | pd.Series]:
    """Taxon 0 raises metabolite `up`, taxon 1 lowers metabolite `down`; two noise metabolites."""
    rng = np.random.default_rng(3)
    ids = [f"P|{i:03d}" for i in range(n)]
    rel = rng.dirichlet(np.ones(p), size=n)
    log_rel = np.log(rel)
    effect = (log_rel - log_rel.mean(axis=0)) / log_rel.std(axis=0)
    mets = np.exp(np.column_stack([
        1.5 * effect[:, 0] + 0.3 * rng.normal(size=n),
        -1.5 * effect[:, 1] + 0.3 * rng.normal(size=n),
        rng.normal(size=(n, 2)),
    ])) * 1000.0
    return {
        "X": pd.DataFrame(rel, index=ids, columns=[f"taxon_{j:02d}" for j in range(p)]),
        "Y": pd.DataFrame(mets, index=ids, columns=["up", "down", "noise_a", "noise_b"]),
        "labels": pd.Series(np.array(["CD", "UC", "Control"])[np.arange(n) % 3], index=ids),
    }


def test_ridge_coefficient_sign_gives_the_direction() -> None:
    cohort = _signed_cohort()
    labels = cohort["labels"]
    folds = splits.make_stratified_folds(list(labels.index), list(labels), n_splits=5, seed=1)
    build = functools.partial(common.build_ridge, cfg=CFG, target_set="T466",
                              candidates=list(cohort["Y"].columns))
    units = _importance(cohort, folds, build, ["up", "down"])
    pairs = interpret.top_taxa(interpret.aggregate_importance(units), 3,
                               interpret.mean_coefficients(units)).set_index(["metabolite", "rank"])
    assert pairs.loc[("up", 1), "taxon"] == "taxon_00"
    assert pairs.loc[("up", 1), "ridge_coef"] > 0
    assert pairs.loc[("down", 1), "taxon"] == "taxon_01"
    assert pairs.loc[("down", 1), "ridge_coef"] < 0


def test_importance_fits_each_fold_on_its_training_rows_only(
    synthetic_cohort, folds, ridge_build, monkeypatch
) -> None:
    seen = []
    real_check = protocol.check_fitted_on

    def spy(pipe: protocol.M2MPipeline, train_ids: Sequence[str]) -> None:
        seen.append((list(pipe.x_prep.fit_samples_), list(pipe.y_prep.fit_samples_)))
        real_check(pipe, train_ids)

    monkeypatch.setattr(protocol, "check_fitted_on", spy)
    _importance(synthetic_cohort, folds, ridge_build, PLANTED_METABOLITES[:2], seeds=[0, 1],
                n_repeats=2)
    assert len(seen) == 5 * 2  # one checked fit per (fold, seed), folds in order
    for unit, (x_rows, y_rows) in enumerate(seen):
        train_ids, test_ids = splits.fold_train_test(folds, unit // 2)
        assert x_rows == y_rows == train_ids
        assert set(x_rows).isdisjoint(test_ids)


def test_a_pipeline_that_peeks_at_held_out_rows_is_refused(
    synthetic_cohort, folds, ridge_build
) -> None:
    def leaky_build(params, seed) -> protocol.M2MPipeline:
        pipe = ridge_build(params, seed)
        honest_fit = pipe.fit

        def fit(X, Y, labels=None):
            honest_fit(X, Y, labels)
            pipe.x_prep.fit(synthetic_cohort["X"])  # refit on every sample, held-out ones too
            return pipe

        pipe.fit = fit
        return pipe

    with pytest.raises(splits.LeakageError):
        _importance(synthetic_cohort, folds, leaky_build, PLANTED_METABOLITES[:1], n_repeats=1)


def test_taxa_a_fold_dropped_count_as_zero_and_seeds_average_first() -> None:
    def unit(fold: int, seed: int, importance: dict[str, float]) -> interpret.FoldImportance:
        return interpret.FoldImportance(
            fold=fold,
            seed=seed,
            baseline_rho=pd.Series({"m": 0.5}),
            importance=pd.DataFrame({"m": importance}),
            coefficients=pd.DataFrame({taxon: [1.0] for taxon in importance}, index=["m"]),
        )

    units = [
        unit(0, 0, {"a": 0.4, "b": 0.2}),
        unit(0, 1, {"a": 0.2, "b": 0.2}),
        unit(1, 0, {"a": 0.6}),  # fold 1's prevalence filter dropped taxon b
    ]
    table = interpret.aggregate_importance(units).set_index("taxon")
    assert table.loc["a", "importance_mean"] == pytest.approx((0.3 + 0.6) / 2)
    assert table.loc["a", "importance_sd"] == pytest.approx(np.std([0.3, 0.6], ddof=1))
    assert table.loc["b", "importance_mean"] == pytest.approx((0.2 + 0.0) / 2)
    coefficients = interpret.mean_coefficients(units)
    assert coefficients.loc["m", "b"] == pytest.approx(0.5)


def test_importance_is_seeded_and_does_not_depend_on_n_jobs(
    synthetic_cohort, folds, ridge_build
) -> None:
    def table(**kwargs) -> pd.DataFrame:
        units = _importance(synthetic_cohort, folds, ridge_build, PLANTED_METABOLITES[:3],
                            n_repeats=3, **kwargs)
        return interpret.aggregate_importance(units)

    serial = table()
    pd.testing.assert_frame_equal(serial, table(n_jobs=2))
    other_shuffles = table(permutation_seed=2)
    assert not np.allclose(serial["importance_mean"], other_shuffles["importance_mean"])


def _small_cfg() -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["seeds"] = [0]
    # The synthetic metabolite maps carry no real class names; tests that need classes set them.
    cfg["interpret"]["importance"].update(n_repeats=3, top_k_taxa=3, named_classes=[])
    cfg["mlp"]["train"].update(max_epochs=20, patience=3)
    return cfg


def test_importance_stage_end_to_end(synthetic_cohort, monkeypatch, tmp_path, capsys) -> None:
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    monkeypatch.setattr(common, "load_d1_train", lambda: (X, Y, labels))
    monkeypatch.setattr(data, "t466_features", lambda raw_dir=None: list(Y.columns))
    _forbid(monkeypatch, (data, "load_nlibd"), (data, "load_franzosa"), (common, "load_d1_test"))
    monkeypatch.setattr(paths, "DATA_RAW", tmp_path / "raw")
    # met_00 is planted (so in the top 10) and met_15 is noise; the map also annotates met_99,
    # which is not a target, so class "Ghost" has no metabolite to explain.
    classes = {"met_00": "Bile acids", "met_15": "Indoles", "met_99": "Ghost"}
    _write_annotations(tmp_path / "raw", [*Y.columns, "met_99"], classes)
    cfg = _small_cfg()
    cfg["interpret"]["importance"]["named_classes"] = ["Bile acids", "Indoles", "Ghost"]
    out = paths.RESULTS / interpret.OUTPUT_DIR

    interpret.run(cfg, stage="importance")  # no E1 selection yet: ridge only
    printed = capsys.readouterr().out
    assert "skipping the MLP view" in printed
    assert ("[e6] class 'Bile acids': its best metabolite, compound met_00 (met_00), is already "
            "explained as top10") in printed
    assert ("[e6] class 'Ghost': none of its metabolites is a target scored in every PRISM "
            "fold") in printed
    assert "  class compound met_15 (met_15) [Indoles] ρ=" in printed
    top = pd.read_csv(out / interpret.TOP_METABOLITES_CSV)
    assert list(top.columns) == interpret.TOP_METABOLITE_COLUMNS
    assert set(top["model"]) == {interpret.RIDGE}
    reasons = top.set_index("metabolite")["selected_by"]
    assert set(reasons[reasons == "top10"].index) == set(PLANTED_METABOLITES)
    assert top["cv_rho"].iloc[:10].is_monotonic_decreasing  # the top list in ridge's CV order
    assert reasons.iloc[10:].to_dict() == {"met_15": "class:Indoles"}  # then the class picks
    met_00 = top.set_index("metabolite").loc["met_00"]
    assert met_00["compound_name"] == "compound met_00"  # D1 annotation
    assert met_00["chemical_class"] == "Bile acids"  # D2 map, joined on the D1 feature ID
    assert met_00["hmdb"] == "HMDB0000000"

    best = {"dropout": 0.2, "lr": 1.0e-3, "weight_decay": 1.0e-4}
    _save_e1_selection(best, protocol.expand_grid(cfg["mlp"]["grid"]))  # E1's select stage ran
    interpret.run(cfg, stage="importance")
    printed = capsys.readouterr().out
    assert "[e6] mlp:" in printed and "→ taxon_" in printed
    top = pd.read_csv(out / interpret.TOP_METABOLITES_CSV)
    pairs = pd.read_csv(out / interpret.IMPORTANCE_CSV)
    assert list(pairs.columns) == interpret.IMPORTANCE_COLUMNS
    ridge_rows = top.loc[top["model"] == interpret.RIDGE, ["metabolite", "selected_by"]]
    mlp_rows = top.loc[top["model"] == interpret.MLP, ["metabolite", "selected_by"]]
    assert mlp_rows.to_numpy().tolist() == ridge_rows.to_numpy().tolist()  # same picks, reasons
    assert len(pairs) == 2 * 11 * 3  # models × (10 top + 1 class pick) × top_k_taxa
    assert pairs.groupby(["model", "metabolite"])["rank"].apply(list).map([1, 2, 3].__eq__).all()
    assert pairs["ridge_coef"].notna().all()  # MLP pairs carry ridge's coefficient too
    reason_of = top.set_index(["model", "metabolite"])["selected_by"]
    assert pairs["selected_by"].tolist() == reason_of.loc[
        list(zip(pairs["model"], pairs["metabolite"], strict=True))
    ].tolist()


def test_importance_stage_can_keep_one_feature_per_compound(
    synthetic_cohort, monkeypatch, tmp_path
) -> None:
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    monkeypatch.setattr(common, "load_d1_train", lambda: (X, Y, labels))
    monkeypatch.setattr(data, "t466_features", lambda raw_dir=None: list(Y.columns))
    monkeypatch.setattr(paths, "DATA_RAW", tmp_path / "raw")
    # Three planted features are one compound (e.g. measured by several LC-MS methods).
    shared = {f: "HMDB9999999" for f in PLANTED_METABOLITES[:3]}
    _write_annotations(tmp_path / "raw", list(Y.columns), {}, hmdb=shared)
    cfg = _small_cfg()

    cfg["interpret"]["importance"]["one_feature_per_hmdb"] = False
    per_feature, _ = interpret.run_importance(cfg)  # one row per feature
    assert set(per_feature["metabolite"]) == set(PLANTED_METABOLITES)

    cfg["interpret"]["importance"]["one_feature_per_hmdb"] = True
    top, _ = interpret.run_importance(cfg)
    assert len(top) == 10 and top["hmdb"].is_unique
    assert len(set(top["metabolite"]) & set(shared)) == 1  # only the compound's best feature
    assert set(PLANTED_METABOLITES[3:]) <= set(top["metabolite"])


_GRID = protocol.expand_grid(CFG["mlp"]["grid"])
_OLD_POINT = {"dropout": 0.3, "lr": 1.0e-2, "weight_decay": 0.0}  # not in the current grid


@pytest.mark.parametrize(
    ("best", "searched", "e1_accepts"),
    [
        (_OLD_POINT, [_OLD_POINT], False),  # left over from an older grid
        # Still a grid point, but chosen before the grid grew: the other points were never run.
        (_GRID[0], [_GRID[0]], False),
        (_GRID[0], [_OLD_POINT, *_GRID], False),
        (_GRID[3], _GRID, True),  # E1's select stage on the current grid
    ],
    ids=["not-a-grid-point", "searched-part-of-grid", "searched-another-grid", "current-grid"],
)
def test_the_mlp_view_explains_only_a_selection_e1_would_evaluate(
    best, searched, e1_accepts
) -> None:
    """E6's "mlp" must be the model E1 scores: it accepts exactly the selections E1 accepts."""
    _save_e1_selection(best, searched)
    if e1_accepts:
        mlp.load_checked_selection(CFG, "T466")
        view = interpret.mlp_view(CFG, "T466", None)
        assert view is not None and dict(view.params) == best
        assert view.seeds == tuple(common.model_seeds(CFG))
        return
    with pytest.raises(ValueError, match="mlp.grid"):
        mlp.load_checked_selection(CFG, "T466")
    with pytest.raises(ValueError, match="mlp.grid"):
        interpret.mlp_view(CFG, "T466", None)


def test_the_mlp_view_is_skipped_before_e1_has_run(capsys) -> None:
    assert interpret.mlp_view(CFG, "T466", None) is None
    assert "skipping the MLP view" in capsys.readouterr().out


# --- classes ---------------------------------------------------------------------------------


def test_class_summary_counts_nan_as_zero_and_pools_small_classes() -> None:
    rho = pd.Series(
        [0.6, np.nan, 0.4, 0.2, 0.0, 0.5]  # A: 6 metabolites
        + [0.1, 0.3, np.nan, 0.2, 0.4]  # B: 5
        + [0.9, 0.7]  # C: 2 → Other
        + [-0.1]  # D: 1 → Other
        + [0.3, 0.3, 0.3, np.nan, 0.3],  # no class (one is not even annotated) → Unclassified
        index=[f"f{i:02d}" for i in range(19)],
    )
    classes = pd.Series(["A"] * 6 + ["B"] * 5 + ["C"] * 2 + ["D"] + [np.nan] * 4,
                        index=rho.index[:18])
    table = interpret.rho_by_class(
        rho, classes, min_class_size=5, null_threshold=0.25, pooled_label="Other",
        unclassified_label="Unclassified",
    ).set_index("chemical_class")
    assert table["n_metabolites"].to_dict() == {"A": 6, "B": 5, "Other": 3, "Unclassified": 5}
    assert table.loc["A", "mean_rho"] == pytest.approx(1.7 / 6)  # the NaN counts as 0
    assert table.loc["B", "median_rho"] == pytest.approx(0.2)
    assert table.loc["Other", "mean_rho"] == pytest.approx(0.5)
    assert table.loc["Unclassified", "sd_rho"] == pytest.approx(np.std([0.3] * 3 + [0.0, 0.3],
                                                                       ddof=1))
    assert table.loc["A", "frac_above_null"] == pytest.approx(3 / 6)
    assert table["mean_rho"].is_monotonic_decreasing
    without_null = interpret.rho_by_class(
        rho, classes, min_class_size=5, null_threshold=None, pooled_label="Other",
        unclassified_label="Unclassified",
    )
    assert without_null["frac_above_null"].isna().all()


def test_metabolite_rho_averages_back_to_the_summary_mean() -> None:
    per_metabolite = pd.DataFrame(
        {"rho_seed0": [0.5, np.nan, np.nan, 0.1], "rho_seed1": [0.3, 0.4, np.nan, 0.3]},
        index=["a", "b", "c", "d"],
    )
    per_metabolite["rho_mean"] = per_metabolite.mean(axis=1)  # as written: NaN seeds skipped
    rho = interpret.metabolite_rho(per_metabolite)
    assert rho.tolist() == pytest.approx([0.4, 0.2, 0.0, 0.2])
    per_seed = [evaluate.summarize_rho(per_metabolite[c])["mean_rho"]
                for c in ("rho_seed0", "rho_seed1")]
    assert rho.mean() == pytest.approx(np.mean(per_seed))


def test_class_share_above_null_is_counted_per_seed() -> None:
    by_seed = pd.DataFrame(
        {"rho_seed0": [0.35, 0.25, 0.5, np.nan, 0.1], "rho_seed1": [0.25, 0.35, 0.5, 0.4, 0.1]},
        index=["a1", "a2", "a3", "b1", "b2"],
    )
    classes = pd.Series(["A", "A", "A", "B", "B"], index=by_seed.index)
    table = interpret.rho_by_class(
        by_seed, classes, min_class_size=2, null_threshold=0.3, pooled_label="Other",
        unclassified_label="Unclassified",
    ).set_index("chemical_class")
    # Seed 0 has 2 metabolites above 0.3 and seed 1 has 3: summary.csv reports 2.5. The seed-mean
    # ρ (0.3, 0.3, 0.5, 0.2, 0.1) would put only one metabolite above the null.
    assert table.loc["A", "frac_above_null"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert table.loc["B", "frac_above_null"] == pytest.approx((0.5 + 0.0) / 2)
    assert (table["n_metabolites"] * table["frac_above_null"]).sum() == pytest.approx(2.5)
    assert table.loc["A", "mean_rho"] == pytest.approx((0.3 + 0.3 + 0.5) / 3)


def test_class_table_decomposes_the_summary_row(monkeypatch, tmp_path, results_dir) -> None:
    """Through the real writers: class sizes × class means / shares add up to summary.csv."""
    features = [f"met_{j:02d}" for j in range(30)]
    classes = {f: "Bile acids" if j < 12 else "Indoles" for j, f in enumerate(features[:20])}
    monkeypatch.setattr(paths, "DATA_RAW", tmp_path / "raw")
    _write_annotations(tmp_path / "raw", features, classes)
    rng = np.random.default_rng(5)
    truth = pd.DataFrame(rng.normal(size=(40, 30)), index=[f"T|{i:02d}" for i in range(40)],
                         columns=features)
    # Two seeds of different quality, so per-seed and seed-mean counts disagree near the null;
    # a constant prediction gives an undefined ρ in seed 1 only.
    predictions = [truth + rng.normal(scale=scale, size=truth.shape) for scale in (1.0, 1.5)]
    predictions[1]["met_00"] = 0.0
    threshold = 0.45
    result = protocol.score_predictions(truth, predictions, CFG, seeds=[0, 1],
                                        null_threshold=threshold)
    row = protocol.record_external(result, experiment="e1", model="mlp", target_set="T466",
                                   data="D1", params={}, cfg=CFG, n_train=100)
    null = NullResult(pooled_rho=np.zeros(4), threshold=threshold,
                      per_permutation_mean=np.zeros(2), quantile=0.95)
    protocol.save_null_threshold(data="D1", target_set="T466", null=null, n_features=30)
    seed_mean = interpret.metabolite_rho(result.rho_by_seed)
    assert (seed_mean > threshold).sum() != row["n_above_null_mean"]  # the fixture discriminates

    interpret.run(CFG, stage="classes")

    table = pd.read_csv(results_dir / interpret.OUTPUT_DIR / interpret.RHO_BY_CLASS_CSV)
    n = table["n_metabolites"]
    assert n.sum() == row["n_features_mean"] == 30
    assert (n * table["mean_rho"]).sum() / n.sum() == pytest.approx(row["mean_rho_mean"])
    assert (n * table["frac_above_null"]).sum() == pytest.approx(row["n_above_null_mean"])


def test_a_dry_run_never_writes_into_the_committed_results(monkeypatch, tmp_path) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    monkeypatch.setattr(paths, "RESULTS", repo / "results")  # the committed results/
    monkeypatch.setattr(paths, "DATA_RAW", tmp_path / "raw")
    features = [f"met_{j:02d}" for j in range(6)]
    _write_annotations(tmp_path / "raw", features, {})
    # Finished results are there, so without the guard the classes stage would write its table.
    from m2m import tracking

    tracking.upsert_summary_row({"model": "ridge", "target_set": "T466", "data": "D1"})
    _per_metabolite_file("ridge", "T466", pd.Series(0.3, index=features))
    monkeypatch.setenv(common.DRY_RUN_ENV, "1")
    with pytest.raises(RuntimeError, match="scratch"):
        interpret.run(CFG, stage="classes")
    assert not (repo / "results" / interpret.OUTPUT_DIR).exists()


def _per_metabolite_file(model: str, target_set: str, rho: pd.Series) -> None:
    table = pd.DataFrame({"rho_seed0": rho, "rho_seed1": rho})
    table["rho_mean"] = table.mean(axis=1)
    table["rho_sd"] = table[["rho_seed0", "rho_seed1"]].std(axis=1, ddof=1)
    table.index.name = "feature"
    path = interpret.per_metabolite_path(model, target_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)


def test_classes_stage_reads_finished_results_only(monkeypatch, tmp_path, results_dir) -> None:
    from m2m import tracking

    _forbid(
        monkeypatch,
        (data, "load_nlibd"), (data, "load_franzosa"), (data, "load_prism"),
        (common, "load_d1_test"), (common, "load_d1_train"),
    )
    features = [f"met_{j:02d}" for j in range(20)]
    classes = {**{f: "Bile acids" for f in features[:6]}, **{f: "Indoles" for f in features[6:8]}}
    monkeypatch.setattr(paths, "DATA_RAW", tmp_path / "raw")
    _write_annotations(tmp_path / "raw", features, classes)
    rho = pd.Series(np.linspace(-0.2, 0.7, len(features)), index=features)
    rho.iloc[0] = np.nan
    for model, target_set, data_label in [
        ("ridge", "T466", "D1"), ("mlp", "T466", "D1"), ("ridge", "TALL", "D1"),
        ("mgm_frozen_ridge", "T466", "D1+MGM"),  # MGM inputs, same D1 split, targets and null
        ("e4c_scratch", "T466", "D2"),  # other inputs and its own D2 null: not a D1 model
    ]:
        tracking.upsert_summary_row({"model": model, "target_set": target_set,
                                     "data": data_label})
        _per_metabolite_file(model, target_set, rho if model == "ridge" else rho / 2)
    null = NullResult(pooled_rho=np.zeros(4), threshold=0.3,
                      per_permutation_mean=np.zeros(2), quantile=0.95)
    protocol.save_null_threshold(data="D1", target_set="T466", null=null, n_features=20)

    interpret.run(CFG, stage="classes")

    table = pd.read_csv(results_dir / interpret.OUTPUT_DIR / interpret.RHO_BY_CLASS_CSV)
    assert list(table.columns) == interpret.CLASS_COLUMNS
    assert table["model"].drop_duplicates().tolist() == ["ridge", "mlp", "mgm_frozen_ridge"]
    ridge = table[table["model"] == "ridge"].set_index("chemical_class")
    assert ridge["n_metabolites"].to_dict() == {"Unclassified": 12, "Bile acids": 6, "Other": 2}
    bile = rho.iloc[:6].fillna(0.0)
    assert ridge.loc["Bile acids", "mean_rho"] == pytest.approx(bile.mean())
    assert ridge.loc["Unclassified", "frac_above_null"] == pytest.approx(
        (rho.iloc[8:] > 0.3).mean()
    )
