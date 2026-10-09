"""Figures (m2m.plots) and the E5 learning-curve check, on synthetic result files.

The inputs are written by the real writers (protocol.record_external, common.record_curve_point,
common.learning_curve_subsets, protocol.save_null_threshold), so a change in their columns
breaks these tests instead of silently breaking the figures.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PIL import Image

from m2m import evaluate, paths, plots, protocol, tracking
from m2m.config import deep_merge, load_config
from m2m.experiments import common
from m2m.experiments import learning_curve as e5
from m2m.run import EXPERIMENT_MODULES

FIG_CFG = load_config(paths.CONFIGS / "figures.yaml")
SEEDS = [0, 1, 2]
SIZES = [20, 40, 80]
# Small bootstrap: these tests check plumbing and figures, not interval accuracy.
FAST_CFG = deep_merge(FIG_CFG, {"evaluation": {"n_bootstrap": 50}, "seeds": SEEDS})
E5_CFG = deep_merge(
    load_config(paths.CONFIGS / "e5_learning_curves.yaml"),
    {"seeds": SEEDS, "learning_curve": {"sizes": SIZES}},
)
MODELS = {  # model: (experiment, data source), as the experiment modules record them
    "mean": ("e0", "D1"),
    "ridge": ("e0", "D1"),
    "mlp": ("e1", "D1"),
    "mgm_frozen_ridge": ("e2", "D1+MGM"),
    "mgm_random_frozen_ridge": ("e2", "D1+MGM-arch"),  # E2's random-weight control
    "mgm_finetune": ("e3", "D1+MGM"),
    "mgm_scratch": ("e3", "D1+MGM-arch"),  # E3's from-scratch control
    "e4c_scratch": ("e4", "D2"),
    "e4c_ihmp": ("e4", "D2"),
    "e4c_selfsup": ("e4", "D2"),
}
CURVE_MODELS = ("ridge", "mlp", "mgm_frozen_ridge", "mgm_finetune", "mgm_scratch", "e4c_scratch",
                "e4c_ihmp")
# The three H1 controls of the MGM models, as E2 and E3 name them.
MGM_CONTROLS = {"mgm_random_frozen_ridge": "e2", "mgm_random_frozen_mlp": "e2",
                "mgm_scratch": "e3"}
N_PRISM = 90  # synthetic PRISM ids behind the saved learning-curve subsets


@pytest.fixture(autouse=True)
def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """runs.csv rows ask git for the commit twice each; irrelevant here and slow."""
    monkeypatch.setattr(tracking, "git_sha", lambda: None)


# --------------------------------------------------------------------------------------------
# Synthetic result files
# --------------------------------------------------------------------------------------------


def write_external(model: str, target_set: str, rng: np.random.Generator) -> None:
    """summary.csv row + per-metabolite file for one model, through protocol.record_external."""
    experiment, data = MODELS[model]
    index = [f"T|{i:02d}" for i in range(30)]
    truth = pd.DataFrame(rng.normal(size=(30, 40)), index=index,
                         columns=[f"m{j:02d}" for j in range(40)])
    if model == "mean":  # constant predictions: every ρ undefined, counted as 0
        predictions = [truth * 0.0 for _ in SEEDS]
    else:
        predictions = [truth * 0.6 + rng.normal(size=truth.shape) for _ in SEEDS]
    result = protocol.score_predictions(truth, predictions, FAST_CFG, seeds=SEEDS,
                                        null_threshold=0.3)
    result.aggregate.update({"n_input_features": 25.0, "n_targets_dropped_zero": 2.0})
    protocol.record_external(
        result, experiment=experiment, model=model, target_set=target_set, data=data,
        params={"alpha": 1.0}, cfg=FAST_CFG, n_train=100,
        cv_mean_rho=None if model == "mean" else float(rng.uniform(0.3, 0.4)),
    )


def write_subsets() -> pd.DataFrame:
    """The saved (n_train, seed) subsets, created by the real E5 engine."""
    ids = [f"PRISM|{i:03d}" for i in range(N_PRISM)]
    labels = list(np.array(["CD", "UC", "Control"])[np.arange(N_PRISM) % 3])
    return common.learning_curve_subsets(E5_CFG, ids, labels)


def write_curve(
    rng: np.random.Generator,
    models: tuple[str, ...] = CURVE_MODELS,
    skip: frozenset[tuple[str, int, int]] = frozenset(),
    sizes: list[int] = SIZES,
) -> None:
    """learning_curve.csv rows through common.record_curve_point (the experiments' writer)."""
    write_subsets()
    for model in models:
        experiment, data = MODELS[model]
        for n in sizes:
            for seed in SEEDS:
                if (model, n, seed) in skip:
                    continue
                rho = rng.normal(0.05 + n / 1000, 0.2, size=40)
                summary = {**evaluate.summarize_from_cfg(rho, FIG_CFG, 0.3),
                           "n_input_features": 25.0, "n_targets_dropped_zero": 2.0}
                common.record_curve_point(
                    summary, model=model, data_name=data, target_set="T466", n_train=n,
                    seed=seed, params={"alpha": 1.0}, experiment=experiment, cfg=FIG_CFG,
                )


def write_null(rng: np.random.Generator) -> np.ndarray:
    pooled = np.clip(rng.normal(0.0, 0.15, size=20 * 40), -1, 1)
    for data in ("D1", "D2"):
        null = evaluate.NullResult(
            pooled_rho=pooled, threshold=float(np.quantile(pooled, 0.95)),
            per_permutation_mean=pooled.reshape(20, -1).mean(axis=1), quantile=0.95,
        )
        protocol.save_null_threshold(data=data, target_set="T466", null=null, n_features=40)
    folder = paths.DATA_PROCESSED / plots.NULL_DIR
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "D1__T466.npy", pooled)
    return pooled


def write_other_inputs(rng: np.random.Generator) -> None:
    """mgm_coverage.csv, interpretation/*.csv and ablations.csv with their documented columns."""
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    ids = [f"PRISM|{i}" for i in range(40)] + [f"Validation|LLDeep_{i:04d}" for i in range(20)]
    pd.DataFrame({
        "sample_id": ids,
        "n_genera_present": rng.integers(20, 40, size=60),
        "n_genera_in_vocab": rng.integers(15, 35, size=60),
        "abundance_in_vocab": rng.uniform(0.8, 1.0, size=60),
    }).to_csv(paths.RESULTS / plots.MGM_COVERAGE_CSV, index=False)
    folder = paths.RESULTS / plots.INTERPRETATION_DIR
    folder.mkdir(parents=True, exist_ok=True)
    classes = ["Bile acids", "Fatty acyls", "Indoles", "Steroids", "Other"]
    pd.DataFrame([
        {"model": model, "chemical_class": c, "n_metabolites": 3 if c == "Other" else 6,
         "mean_rho": 0.0 if model == "mean" else rng.normal(0.1, 0.1), "median_rho": 0.1,
         "sd_rho": 0.1, "frac_above_null": 0.2}
        for model in ("mean", "ridge", "mlp", "mgm_finetune") for c in classes
    ]).to_csv(folder / plots.RHO_BY_CLASS_CSV, index=False)
    taxa = [f"Genus_species_{k}" for k in range(8)]
    pd.DataFrame([
        {"model": "ridge", "metabolite": f"m{j:02d}", "compound_name": f"compound {j}",
         "taxon": taxa[(j + r) % len(taxa)], "importance_mean": 0.05 / (r + 1) - 0.004,
         "importance_sd": 0.01, "rank": r + 1, "ridge_coef": (-1) ** r * 0.3}
        for j in range(5) for r in range(4)
    ]).to_csv(folder / plots.IMPORTANCE_CSV, index=False)
    pd.DataFrame([
        {"ablation": ablation, "arm": arm, "model": model, "target_set": "T466",
         "eval_space": "log10", "is_default_arm": arm in ("clr", "0.10"), "cv_mean_rho": 0.3,
         "mean_rho_mean": 0.2, "mean_rho_sd": 0.0 if model == "ridge" else 0.02}
        for ablation, arms in {"A1_input_transform": ["clr", "log"],
                               "A3_prevalence": ["0.05", "0.10"]}.items()
        for arm in arms for model in ("ridge", "mlp")
    ]).to_csv(paths.RESULTS / plots.ABLATIONS_CSV, index=False)


@pytest.fixture
def all_inputs(rng: np.random.Generator) -> dict[str, np.ndarray]:
    for model, (_, data) in MODELS.items():
        for target_set in (("T466", "S176") if data == "D2" else ("T466", "TALL")):
            write_external(model, target_set, rng)
    write_curve(rng)
    pooled = write_null(rng)
    write_other_inputs(rng)
    return {"pooled_null": pooled}


def line_colours(fig: Figure) -> dict[str, set[str]]:
    """Model -> colours of every labelled line in the figure (legend label = model name)."""
    colours: dict[str, set[str]] = {}
    for ax in fig.axes:
        for line in ax.get_lines():
            label = str(line.get_label())
            if not label.startswith("_"):
                colours.setdefault(label.split(" (")[0], set()).add(to_hex(line.get_color()))
    return colours


# --------------------------------------------------------------------------------------------
# plots.run / draw_figures
# --------------------------------------------------------------------------------------------


def test_every_figure_is_a_png_at_the_configured_dpi(all_inputs, results_dir):
    report = plots.draw_figures(FIG_CFG)
    assert report.skipped == {}
    assert set(report.drawn) == set(plots.FIGURE_NAMES)
    for name, path in report.drawn.items():
        assert path == paths.FIGURES / f"{name}.png"
        assert path.stat().st_size > 10_000
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.info["dpi"] == pytest.approx((200, 200), abs=0.5)


def test_missing_inputs_are_skipped_with_a_reason(results_dir):
    report = plots.draw_figures(FIG_CFG)
    assert report.drawn == {}
    assert set(report.skipped) == set(plots.FIGURE_NAMES)
    assert all(reason for reason in report.skipped.values())
    assert "learning_curve.csv missing" in report.skipped["learning_curve"]
    assert not paths.FIGURES.exists() or not any(paths.FIGURES.iterdir())


def test_partial_inputs_draw_only_what_exists(rng, results_dir):
    write_external("ridge", "T466", rng)
    write_external("mlp", "T466", rng)
    report = plots.draw_figures(FIG_CFG)
    assert set(report.drawn) == {"model_comparison", "cv_vs_external", "per_metabolite_rho"}
    assert "null" in report.skipped["null_distribution"]


def test_unusable_input_is_skipped_not_fatal(results_dir):
    results_dir.mkdir(parents=True)
    pd.DataFrame({"sample_id": ["PRISM|1"], "coverage": [0.9]}).to_csv(
        paths.RESULTS / plots.MGM_COVERAGE_CSV, index=False
    )
    report = plots.draw_figures(FIG_CFG, ["mgm_coverage"])
    assert "lacks columns ['abundance_in_vocab']" in report.skipped["mgm_coverage"]


def test_run_draws_a_single_stage_and_rejects_unknown_ones(all_inputs, capsys):
    plots.run(FIG_CFG, stage="model_comparison")
    assert sorted(p.name for p in paths.FIGURES.iterdir()) == ["model_comparison.png"]
    assert "drew 1 figure" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        plots.run(FIG_CFG, stage="no_such_figure")


def test_figures_never_open_the_external_cohort(all_inputs, monkeypatch):
    import m2m.data as data

    def forbidden(*args, **kwargs):
        raise AssertionError("a figure opened the external cohort")

    for module, name in ((data, "load_nlibd"), (data, "load_franzosa"), (common, "load_d1_test")):
        monkeypatch.setattr(module, name, forbidden)
    assert set(plots.draw_figures(FIG_CFG).drawn) == set(plots.FIGURE_NAMES)
    assert e5.check(E5_CFG).complete
    assert not (paths.RESULTS / tracking.EXTERNAL_ACCESS_LOG).exists()


def test_dry_runs_are_named_in_the_titles(monkeypatch):
    assert plots.external_cohort_label(FIG_CFG) == "NLIBD"
    monkeypatch.setenv(common.DRY_RUN_ENV, "1")
    assert "DRY RUN" in plots.external_cohort_label(FIG_CFG)


def test_dry_runs_never_write_into_the_committed_results(rng, tmp_path, monkeypatch):
    """M2M_DRY_RUN=1 against the committed results/: figures and E5 refuse before writing.

    The committed folder is simulated: REPO_ROOT points at this test's tmp_path, whose results/
    is the per-test results directory (conftest), so nothing outside the test is touched.
    """
    write_other_inputs(rng)
    write_curve(rng)
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    assert paths.RESULTS.resolve() == (tmp_path / "results").resolve()
    monkeypatch.setenv(common.DRY_RUN_ENV, "1")
    with pytest.raises(RuntimeError, match="scratch directory"):
        plots.draw_figures(FIG_CFG, ["mgm_coverage"])  # a drawer that never names the cohort
    with pytest.raises(RuntimeError, match="scratch directory"):
        e5.check(E5_CFG)
    assert not paths.FIGURES.exists()
    assert not (paths.RESULTS / e5.SUMMARY_CSV).exists()


# --------------------------------------------------------------------------------------------
# One colour per model
# --------------------------------------------------------------------------------------------


def test_colour_of_a_model_is_the_same_in_every_figure(all_inputs):
    styles = plots.model_styles(FIG_CFG)
    curve = pd.read_csv(paths.RESULTS / common.LEARNING_CURVE_CSV)
    summary = pd.read_csv(paths.RESULTS / tracking.SUMMARY_CSV)
    figures = [
        plots.learning_curve(curve, styles=styles, target_set="T466", test_cohort="NLIBD"),
        plots.per_metabolite_rho(plots.load_per_metabolite("T466"), styles=styles,
                                 data_of=dict(zip(summary["model"], summary["data"], strict=True)),
                                 null_thresholds={"D1": 0.3}, target_set="T466",
                                 test_cohort="NLIBD", null_quantile=0.95),
        plots.cv_vs_external(summary, styles=styles, target_sets=["T466"], test_cohort="NLIBD",
                             cv_folds=5),
        plots.model_comparison(summary, styles=styles, target_sets=["T466", "TALL"],
                               rho_count_key="n_rho_gt_0.5_mean", rho_threshold=0.5,
                               ci_label="95% CI", test_cohort="NLIBD", null_quantile=0.95),
    ]
    seen: dict[str, set[str]] = {}
    for fig in figures:
        for model, colours in line_colours(fig).items():
            seen.setdefault(model, set()).update(colours)
    assert set(CURVE_MODELS) <= set(seen)
    for model, colours in seen.items():
        if model in MODELS:
            assert colours == {styles[model].color}, model


def test_style_assignment_is_deterministic_and_never_invents_hues():
    families = FIG_CFG["figures"]["families"]
    models = ["mean", "ridge", "enet", "rf", "mlp", "mgm_frozen_a", "mgm_frozen_b",
              "mgm_finetune", "e4c_scratch", "e4c_a", "e4c_b", "e4c_c", "mystery"]
    experiments = {"mgm_frozen_a": "e2", "mgm_frozen_b": "e2", "mgm_finetune": "e3"}
    curve = ["ridge", "mlp", "mgm_frozen_a", "mgm_finetune", "e4c_scratch", "e4c_b"]

    def build(order):
        return plots.build_model_styles(order, families=families, experiments=experiments,
                                        reference_models=["mean"], priority=curve)

    styles = build(models)
    assert build(list(reversed(models))).styles == styles.styles
    assert styles["mean"].color == plots.REFERENCE_COLOR
    # The learning-curve models take the first palette slots (in presentation order).
    assert [styles[m].color for m in styles.sort(curve)] == list(plots.PALETTE[: len(curve)])
    # Two models of one family never share a hue (families here have <= 8 models).
    by_family: dict[str, list[str]] = {}
    for model in models:
        if model != "mean":
            by_family.setdefault(styles[model].family, []).append(styles[model].color)
    assert all(len(set(c)) == len(c) for c in by_family.values())
    # (hue, marker) pairs identify every model; no hue outside the validated palette.
    pairs = [(styles[m].color, styles[m].marker) for m in models if m != "mean"]
    assert len(set(pairs)) == len(pairs)
    assert {c for c, _ in pairs} <= set(plots.PALETTE)
    assert styles["mystery"].family == plots.OTHER_FAMILY
    assert styles.sort(["e4c_a", "mlp", "ridge", "mean"]) == ["mean", "ridge", "mlp", "e4c_a"]
    with pytest.raises(KeyError, match="no style"):
        styles["not_a_model"]


def test_experiment_decides_the_family_before_the_name():
    families = FIG_CFG["figures"]["families"]
    styles = plots.build_model_styles(
        ["ridge_on_embeddings"], families=families, experiments={"ridge_on_embeddings": "e2"}
    )
    assert styles["ridge_on_embeddings"].family == "MGM"


@pytest.mark.parametrize("known_experiment", [True, False])
def test_mgm_controls_join_the_mgm_family_each_with_its_own_hue(known_experiment):
    """By experiment (summary.csv) or by name alone (learning_curve.csv, per-metabolite files),
    the controls sit next to the MGM models they control and never borrow their colours."""
    settings = FIG_CFG["figures"]
    mgm = ["mgm_frozen_ridge", "mgm_frozen_mlp", "mgm_finetune", *MGM_CONTROLS]
    experiments = (
        {"mgm_frozen_ridge": "e2", "mgm_frozen_mlp": "e2", "mgm_finetune": "e3", **MGM_CONTROLS}
        if known_experiment else {}
    )
    styles = plots.build_model_styles(
        ["ridge", "mlp", *mgm], families=settings["families"], experiments=experiments,
        model_order=settings["model_order"],
    )
    assert {styles[m].family for m in mgm} == {"MGM"}
    assert len({styles[m].color for m in mgm}) == len(mgm)
    assert set(styles.sort(mgm)) == set(mgm) and styles.sort(["mgm_scratch", "mlp"])[0] == "mlp"


# --------------------------------------------------------------------------------------------
# Individual figures
# --------------------------------------------------------------------------------------------


def test_learning_curve_axes_lines_and_bands(all_inputs):
    styles = plots.model_styles(FIG_CFG)
    curve = pd.read_csv(paths.RESULTS / common.LEARNING_CURVE_CSV)
    fig = plots.learning_curve(curve, styles=styles, target_set="T466", test_cohort="NLIBD")
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert list(ax.get_xticks()) == SIZES
    lines = {str(line.get_label()): line for line in ax.get_lines()}
    zero = lines["mean predictor (ρ = 0)"]
    assert list(zero.get_ydata()) == [0, 0] and zero.get_linestyle() == ":"
    assert lines["ridge"].get_linestyle() == "-"
    assert lines["mgm_frozen_ridge"].get_linestyle() == "-"  # D1+MGM: D1's samples and genera
    assert lines["e4c_scratch (D2)"].get_linestyle() == "--"
    assert "mean" not in lines  # the mean predictor is the zero line, not a series
    assert len(ax.collections) == len(CURVE_MODELS)  # one ± SD band per line
    title = fig.get_suptitle()
    assert "T466" in title and "NLIBD" in title


def test_curve_summary_is_mean_and_sample_sd_over_seeds():
    curve = pd.DataFrame({
        "model": ["ridge"] * 4, "data": ["D1"] * 4, "target_set": ["T466"] * 4,
        "n_train": [20, 20, 20, 40], "seed": [0, 1, 2, 0],
        "mean_rho": [0.1, 0.2, 0.4, 0.3], "n_features": [30, 32, 31, 35],
    })
    summary = plots.curve_summary(curve).set_index("n_train")
    assert list(summary.columns) == [c for c in plots.CURVE_SUMMARY_COLUMNS if c != "n_train"]
    assert summary.loc[20, "mean_rho_mean"] == pytest.approx(np.mean([0.1, 0.2, 0.4]))
    assert summary.loc[20, "mean_rho_sd"] == pytest.approx(np.std([0.1, 0.2, 0.4], ddof=1))
    assert summary.loc[20, "n_features_mean"] == pytest.approx(31.0)
    assert summary.loc[40, "mean_rho_sd"] == 0.0 and summary.loc[40, "n_seeds"] == 1


def test_null_figure_marks_the_configured_quantile(all_inputs):
    styles = plots.model_styles(FIG_CFG)
    rho = plots.load_per_metabolite("T466")["ridge"]
    pooled = all_inputs["pooled_null"]
    fig = plots.null_distribution(pooled, rho, model="ridge", styles=styles, quantile=0.95,
                                  bins=20, target_set="T466", test_cohort="NLIBD")
    vertical = [line for line in fig.axes[0].get_lines()
                if len(set(line.get_xdata())) == 1]
    assert [line.get_xdata()[0] for line in vertical] == [pytest.approx(np.quantile(pooled, 0.95))]


def test_null_figure_refuses_a_pooled_null_from_another_run(all_inputs):
    """data/processed/null/*.npy must be the null behind results/null_thresholds.csv: the two
    folders are redirected separately, and a mix-up would contradict summary.csv's N above null."""
    path = paths.DATA_PROCESSED / plots.NULL_DIR / "D1__T466.npy"
    pooled = all_inputs["pooled_null"]
    for name, other in {"shifted": pooled + 0.05, "resized": np.tile(pooled, 2)}.items():
        np.save(path, other)
        report = plots.draw_figures(FIG_CFG, ["null_distribution"])
        assert "different runs" in report.skipped.get("null_distribution", ""), name
    np.save(path, pooled)
    fig = plots._DRAWERS["null_distribution"](FIG_CFG, plots.model_styles(FIG_CFG))
    recorded = pd.read_csv(paths.RESULTS / protocol.NULL_THRESHOLDS_CSV).set_index("data")
    vertical = [line.get_xdata()[0] for line in fig.axes[0].get_lines()
                if len(set(line.get_xdata())) == 1]
    assert vertical == [pytest.approx(recorded.loc["D1", "threshold"])]
    recorded.drop(index="D1").to_csv(paths.RESULTS / protocol.NULL_THRESHOLDS_CSV)
    report = plots.draw_figures(FIG_CFG, ["null_distribution"])
    assert "no D1 T466 row" in report.skipped["null_distribution"]


def test_rho_by_class_is_a_classes_by_models_table_without_the_mean_predictor(all_inputs):
    styles = plots.model_styles(FIG_CFG)
    table = pd.read_csv(paths.RESULTS / plots.INTERPRETATION_DIR / plots.RHO_BY_CLASS_CSV)
    fig = plots.rho_by_class(table, styles=styles, test_cohort="NLIBD")
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["ridge", "mlp", "mgm_finetune"]
    rows = [t.get_text() for t in ax.get_yticklabels()]
    assert len(rows) == 5 and "Other (3)" in rows
    by_class = table[table["model"] != "mean"].groupby("chemical_class")["mean_rho"].mean()
    assert rows[0].startswith(by_class.idxmax())  # best class on top
    assert len(ax.texts) == 5 * 3  # every value is printed in its cell
    image = ax.get_images()[0]
    low, high = image.get_clim()
    assert low == -high  # diverging around ρ = 0
    with pytest.raises(plots.MissingInput):
        plots.rho_by_class(table[table["model"] == "mean"], styles=styles, test_cohort="NLIBD")


def test_per_metabolite_rho_counts_undefined_seeds_as_zero(results_dir):
    """As summary.csv does: NaN -> 0 in each seed, then the mean (not pandas' NaN-skipping)."""
    folder = paths.RESULTS / plots.PER_METABOLITE_DIR
    folder.mkdir(parents=True)
    pd.DataFrame({
        "feature": ["a", "b"], "rho_seed0": [0.4, np.nan], "rho_seed1": [np.nan, np.nan],
        "rho_mean": [0.4, np.nan], "rho_sd": [np.nan, np.nan],
    }).to_csv(folder / "mlp__T466.csv.gz", index=False)
    rho = plots.load_per_metabolite("T466")["mlp"]
    assert rho.to_dict() == {"a": pytest.approx(0.2), "b": 0.0}


def test_ecdf_panels_use_the_null_of_each_data_source(all_inputs):
    """D1+MGM models are scored against D1's null; D2 models against their own."""
    styles = plots.model_styles(FIG_CFG)
    summary = pd.read_csv(paths.RESULTS / tracking.SUMMARY_CSV)
    fig = plots.per_metabolite_rho(
        plots.load_per_metabolite("T466"), styles=styles,
        data_of=dict(zip(summary["model"], summary["data"], strict=True)),
        null_thresholds={"D1": 0.25, "D2": 0.31}, target_set="T466", test_cohort="NLIBD",
        null_quantile=0.95,
    )
    titles = {ax.get_legend().get_title().get_text() for ax in fig.axes if ax.get_legend()}
    assert titles == {"Baselines · null 95th pct: D1 0.25", "MLP · null 95th pct: D1 0.25",
                      "MGM · null 95th pct: D1 0.25", "E4 transfer · null 95th pct: D2 0.31"}


def test_ablations_mark_the_main_protocol_and_tolerate_cv_only_rows(rng, results_dir):
    write_other_inputs(rng)
    table = pd.read_csv(paths.RESULTS / plots.ABLATIONS_CSV)
    cv_only = table.drop(columns=["mean_rho_mean", "mean_rho_sd"])  # external stage not run yet
    cv_only.to_csv(paths.RESULTS / plots.ABLATIONS_CSV, index=False)
    report = plots.draw_figures(FIG_CFG, ["ablations"])
    assert "ablations" in report.drawn
    styles = plots.build_model_styles(["ridge", "mlp"], families=FIG_CFG["figures"]["families"])
    fig = plots.ablations(table, styles=styles, test_cohort="NLIBD", cv_folds=5)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels == ["clr · ridge (main)", "clr · mlp (main)", "log · ridge", "log · mlp"]


def test_scatter_labels_do_not_overlap(results_dir):
    """Every curve model within 0.015 of each other: every name still gets its own spot."""
    names = list(CURVE_MODELS)
    offsets = 0.002 * np.arange(len(names))
    summary = pd.DataFrame({
        "model": names, "target_set": "T466",
        "cv_mean_rho": 0.30 + offsets, "mean_rho_mean": 0.20 + offsets,
    })
    styles = plots.build_model_styles(names, families=FIG_CFG["figures"]["families"])
    fig = plots.cv_vs_external(summary, styles=styles, target_sets=["T466"], test_cohort="NLIBD",
                               cv_folds=5)
    renderer = plots._laid_out(fig)
    ax = fig.axes[0]
    boxes = [plots._text_box(text, renderer) for text in ax.texts]
    assert len(boxes) == len(names)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert plots._overlap_area(a, b) == 0.0
        assert plots._overlap_area(a, ax.get_window_extent(renderer)) == pytest.approx(
            plots._area(a)
        )


def test_figure_text_follows_the_protocol_settings(all_inputs):
    """The fold count and the null quantile in labels come from the config, not the code."""
    cfg = deep_merge(FIG_CFG, {"cv": {"n_splits": 4}, "evaluation": {"null_quantile": 0.99}})
    styles = plots.model_styles(cfg)

    def note(fig: Figure) -> str:
        return " ".join(fig.texts[-1].get_text().split())

    scatter = plots._DRAWERS["cv_vs_external"](cfg, styles)
    assert scatter.axes[0].get_xlabel() == "mean ρ, PRISM 4-fold CV"
    assert "CV models train on 3/4 of PRISM" in note(scatter)
    assert "99th percentile" in note(plots._DRAWERS["model_comparison"](cfg, styles))
    ecdf = plots._DRAWERS["per_metabolite_rho"](cfg, styles)
    assert "99th percentile" in note(ecdf)
    assert all("null 99th pct" in ax.get_legend().get_title().get_text()
               for ax in ecdf.axes if ax.get_legend())
    ablation = plots._DRAWERS["ablations"](cfg, styles)
    legend = ablation.axes[0].get_legend()
    assert "PRISM 4-fold CV" in [text.get_text() for text in legend.get_texts()]


def test_cell_text_takes_the_higher_contrast_ink():
    assert plots._text_colour_on("#0d366b") == plots.SURFACE  # darkest blue step
    assert plots._text_colour_on("#cde2fb") == plots.INK  # lightest blue step
    assert plots._text_colour_on(plots.NOT_RECORDED) == plots.INK


def test_spread_keeps_a_minimum_gap_and_stays_below_one():
    placed = plots._spread([0.95, 0.96, 0.2, 0.97], gap=0.05)
    assert placed[2] == pytest.approx(0.2)
    top = sorted(placed[i] for i in (0, 1, 3))
    assert top[-1] <= 1.0 and np.diff(top).min() >= 0.05 - 1e-12


# --------------------------------------------------------------------------------------------
# E5: learning-curve check and figure
# --------------------------------------------------------------------------------------------


def test_e5_check_flags_a_missing_point_and_missing_models(rng, results_dir):
    write_curve(rng, models=("ridge", "mlp", "mgm_frozen_ridge", "mgm_finetune"),
                skip=frozenset({("mlp", 40, 1)}))
    with pytest.warns(e5.CurveCheckWarning) as caught:
        result = e5.check(E5_CFG)
    assert result.missing_points == {"mlp": [(40, 1)]}
    assert result.missing_entries == ["E3 MGM architecture from scratch",  # mgm_finetune's control
                                      "E4c from scratch (D2)", "E4c best pretrained variant (D2)"]
    assert not result.complete
    messages = " ".join(str(w.message) for w in caught)
    assert "n=40/seed=1" in messages and "E4c from scratch" in messages
    summary = pd.read_csv(paths.RESULTS / e5.SUMMARY_CSV)
    assert list(summary.columns) == list(plots.CURVE_SUMMARY_COLUMNS)
    mlp_40 = summary.loc[(summary["model"] == "mlp") & (summary["n_train"] == 40)].iloc[0]
    assert mlp_40["n_seeds"] == 2


def test_e5_check_is_quiet_on_a_complete_curve(rng, results_dir, capsys):
    write_curve(rng)
    with warnings.catch_warnings():
        warnings.simplefilter("error", e5.CurveCheckWarning)
        result = e5.check(E5_CFG)
    assert result.complete
    assert result.matched["E2 head chosen for the curve"] == ["mgm_frozen_ridge"]
    assert result.matched["E3 MGM architecture from scratch"] == ["mgm_scratch"]
    assert result.matched["E4c best pretrained variant (D2)"] == ["e4c_ihmp"]
    assert result.expected_points == [(n, s) for n in SIZES for s in SEEDS]
    out = capsys.readouterr().out
    assert "e4c_ihmp (D2)" in out and "±" in out and "complete" in out
    # The summary agrees with numpy on the recorded points.
    curve = pd.read_csv(paths.RESULTS / common.LEARNING_CURVE_CSV)
    points = curve.loc[(curve["model"] == "ridge") & (curve["n_train"] == 80), "mean_rho"]
    row = result.summary.loc[(result.summary["model"] == "ridge")
                             & (result.summary["n_train"] == 80)].iloc[0]
    assert row["mean_rho_mean"] == pytest.approx(points.mean())
    assert row["mean_rho_sd"] == pytest.approx(points.std(ddof=1))


def test_e5_check_flags_points_outside_the_saved_subsets(rng, results_dir):
    write_curve(rng)
    write_curve(rng, models=("ridge",), sizes=[160])  # a stale size from another config
    with pytest.warns(e5.CurveCheckWarning, match="outside the saved subsets"):
        result = e5.check(E5_CFG)
    assert result.extra_points == {"ridge": [(160, s) for s in SEEDS]}


def test_stale_curve_points_are_left_out_of_the_summary_and_the_figure(rng, results_dir):
    """Rows of an older seed list or size are named by the check, never averaged in."""
    write_curve(rng)
    fresh = pd.read_csv(paths.RESULTS / common.LEARNING_CURVE_CSV)
    saved = fresh.loc[(fresh["model"] == "ridge") & (fresh["n_train"] == 20), "mean_rho"]
    stale = {**evaluate.summarize_from_cfg(np.full(40, 0.9), FIG_CFG, 0.3),
             "n_input_features": 25.0, "n_targets_dropped_zero": 2.0}
    for n, seed in ((20, 7), (20, 8), (160, 0)):  # not among the saved (n_train, seed) subsets
        common.record_curve_point(stale, model="ridge", data_name="D1", target_set="T466",
                                  n_train=n, seed=seed, params={}, experiment="e0", cfg=FIG_CFG)
    with pytest.warns(e5.CurveCheckWarning, match="left out of the summary and the figure"):
        result = e5.check(E5_CFG)
    assert result.extra_points == {"ridge": [(20, 7), (20, 8), (160, 0)]}
    ridge = result.summary.loc[result.summary["model"] == "ridge"].set_index("n_train")
    assert list(ridge.index) == SIZES
    assert ridge.loc[20, "n_seeds"] == len(SEEDS)
    assert ridge.loc[20, "mean_rho_mean"] == pytest.approx(saved.mean())
    # The figure (as `make figures` draws it) shows the same points and says what it left out.
    fig = plots._DRAWERS["learning_curve"](FIG_CFG, plots.model_styles(FIG_CFG))
    line = next(line for line in fig.axes[0].get_lines() if line.get_label() == "ridge")
    assert list(line.get_xdata()) == SIZES
    assert line.get_ydata()[0] == pytest.approx(saved.mean())
    note = " ".join(fig.texts[-1].get_text().split())
    assert "3 recorded point(s) outside the saved PRISM subsets" in note


def test_e5_warns_on_several_candidates_for_one_entry(rng, results_dir):
    write_curve(rng, models=(*CURVE_MODELS, "e4c_selfsup"))
    with pytest.warns(e5.CurveCheckWarning, match="2 models match E4c best pretrained"):
        result = e5.check(E5_CFG)
    assert result.matched["E4c best pretrained variant (D2)"] == ["e4c_ihmp", "e4c_selfsup"]


def save_cv_selection(name: str, cv_mean_rho: float) -> None:
    """A saved selection as the experiments write it (protocol.save_selection)."""
    table = pd.DataFrame({"params": [json.dumps({})], "cv_mean_rho": [cv_mean_rho]})
    protocol.save_selection(name, "T466", {}, table)


def test_e5_expects_the_model_the_saved_cv_selections_pick(rng, results_dir, capsys):
    write_curve(rng)  # E2 head on the curve: mgm_frozen_ridge; E4c pretrained: e4c_ihmp
    save_cv_selection("e2_mgm_frozen_ridge", 0.35)
    save_cv_selection("e2_mgm_frozen_mlp", 0.32)
    for variant, cv in (("e4c_selfsup", 0.30), ("e4c_ihmp", 0.33), ("e4c_selfsup_ihmp", 0.33)):
        save_cv_selection(variant, cv)  # a tie: the first listed (e4c_ihmp) wins, as in E4
    with warnings.catch_warnings():
        warnings.simplefilter("error", e5.CurveCheckWarning)
        result = e5.check(E5_CFG)
    assert result.complete
    assert result.picked == {"E2 head chosen for the curve": "mgm_frozen_ridge",
                             "E4c best pretrained variant (D2)": "e4c_ihmp"}
    assert "saved CV selections pick mgm_frozen_ridge" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("e2_cv", "e4_cv"),
    [
        ((0.30, 0.30), (0.30, 0.30, 0.30)),  # ties: each experiment's first listed wins
        ((0.29, 0.31), (0.20, 0.30, 0.30)),
        ((0.31, 0.29), (0.20, 0.25, 0.30)),
    ],
)
def test_e5_picks_agree_with_the_experiments_own_rules(e2_cv, e4_cv, results_dir):
    """E2's best_head and E4's best_pretrained_variant put a model on the curve; the check must
    recompute the same choice from the same saved selections (configs listed in the same order)."""
    from m2m.experiments import mgm_frozen, transfer

    for head, cv in zip(mgm_frozen.HEADS, e2_cv, strict=True):
        save_cv_selection(mgm_frozen.SELECTION_NAMES[mgm_frozen.PRETRAINED][head], cv)
        # The random-weight control has no curve: even a higher CV score never puts it there.
        save_cv_selection(mgm_frozen.SELECTION_NAMES[mgm_frozen.RANDOM][head], 0.9)
    e4_cfg = load_config(paths.CONFIGS / "e4_transfer.yaml")
    pretrained = [v for v in transfer.e4c_variants(e4_cfg) if v != "scratch"]
    for variant, cv in zip(pretrained, e4_cv, strict=True):
        save_cv_selection(f"e4c_{variant}", cv)
    picks = e5.selection_picks(E5_CFG["curve_check"]["expected"], "T466")
    head, _ = mgm_frozen.best_head("T466")
    pretrained_names = mgm_frozen.MODEL_NAMES[mgm_frozen.PRETRAINED]
    assert picks["E2 head chosen for the curve"] == pretrained_names[head]
    variant = transfer.best_pretrained_variant(e4_cfg, "T466")
    assert picks["E4c best pretrained variant (D2)"] == f"e4c_{variant}"


def test_e5_flags_a_curve_model_that_the_selections_do_not_pick(rng, results_dir):
    write_curve(rng)  # the curve has mgm_frozen_ridge ...
    save_cv_selection("e2_mgm_frozen_ridge", 0.30)
    save_cv_selection("e2_mgm_frozen_mlp", 0.34)  # ... but CV now prefers the MLP head
    with pytest.warns(e5.CurveCheckWarning) as caught:
        result = e5.check(E5_CFG)
    assert result.missing_entries == ["E2 head chosen for the curve"]
    assert result.unexpected_models == ["mgm_frozen_ridge"]
    messages = " ".join(str(w.message) for w in caught)
    assert "pick 'mgm_frozen_mlp'" in messages and "no expected entry names" in messages


def test_e5_without_a_curve_warns_and_does_not_crash(results_dir):
    with pytest.warns(e5.CurveCheckWarning, match="not found"):
        e5.run(E5_CFG, stage="check,figure")  # "subsets" needs the real PRISM cohort
    assert not (paths.RESULTS / e5.SUMMARY_CSV).exists()
    assert not (paths.FIGURES / "learning_curve.png").exists()


def test_e5_run_writes_the_summary_and_the_figure(rng, results_dir):
    write_curve(rng)
    e5.run(E5_CFG, stage="check,figure")  # "subsets" needs the real PRISM cohort
    assert (paths.RESULTS / e5.SUMMARY_CSV).exists()
    assert (paths.FIGURES / "learning_curve.png").stat().st_size > 10_000


def test_e5_figure_uses_the_shared_figure_settings():
    shared = e5.figures_config(E5_CFG)
    assert shared["experiment"] == "figures"
    assert shared["figures"] == FIG_CFG["figures"]
    assert shared["learning_curve"]["sizes"] == SIZES  # the E5 config's own curve settings


def test_figures_config_falls_back_to_the_configs_folder(tmp_path):
    """A copy of the E5 config elsewhere (e.g. a scratch override) still finds figures.yaml."""
    elsewhere = tmp_path / "e5_override.yaml"
    elsewhere.write_text(
        f"extends: {paths.CONFIGS / 'e5_learning_curves.yaml'}\nlearning_curve: {{sizes: [20]}}\n"
    )
    shared = e5.figures_config(load_config(elsewhere))
    assert shared["figures"] == FIG_CFG["figures"]
    assert shared["learning_curve"]["sizes"] == [20]


def test_expected_points_fall_back_to_the_config(results_dir, capsys):
    assert e5.expected_points(E5_CFG) == {(n, s) for n in SIZES for s in SEEDS}
    assert "not found" in capsys.readouterr().out


@pytest.mark.parametrize(
    "entry",
    [{"model": "a", "prefix": "b"}, {"what": "x"}, {"model": "a", "typo": 1}],
)
def test_expected_entries_must_name_a_model_or_a_prefix(entry):
    with pytest.raises(ValueError, match="exactly one"):
        e5.match_expected([entry], ["a"])


def test_configs_are_registered_experiments():
    assert EXPERIMENT_MODULES[FIG_CFG["experiment"]] == "m2m.plots"
    assert EXPERIMENT_MODULES[E5_CFG["experiment"]] == "m2m.experiments.learning_curve"
    assert (Path(E5_CFG["config_path"]).parent / E5_CFG["figures_config"]).exists()


def test_e5_subsets_stage_creates_the_shared_subsets(monkeypatch, capsys):
    calls = []

    def fake_subsets(cfg):
        calls.append(cfg)
        return pd.DataFrame(
            {"n_train": [20, 20, 40], "seed": [0, 1, 0], "sample_id": ["a", "b", "c"]}
        )

    monkeypatch.setattr(common, "learning_curve_subsets", fake_subsets)
    e5.run(E5_CFG, stage="subsets")
    assert calls == [E5_CFG]
    assert "sizes [20, 40]" in capsys.readouterr().out
