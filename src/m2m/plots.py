"""Every figure in results/figures/, drawn from the recorded result files only.

    python -m m2m.run configs/figures.yaml [--stage <figure name>]

draws each figure whose inputs exist and prints which were skipped and why. Nothing here fits,
scores or selects anything: the figures read what the experiments wrote (results/*.csv,
results/per_metabolite/, results/interpretation/, data/processed/null/), so drawing them never
opens the external cohort and never feeds a choice back into the models.

Presentation rules (the dataviz method: form first, colour last, palette validated by script):
- one colour and one marker per model, shared by every figure (`model_styles`), from a validated
  8-hue categorical palette; hues are never generated, and two models of one family never share
  a hue while the family has at most 8 models;
- the constant mean predictor is the grey ρ = 0 reference, not a coloured series;
- light surface, solid hairline grid, thin marks; text uses ink colours, never series colours;
- every chart names its target set and test cohort; axes are honest (log₂ only where the
  training size doubles; a zero line wherever ρ = 0 means "no better than the mean predictor").
"""

from __future__ import annotations

import functools
import math
import textwrap
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import NullLocator
from matplotlib.transforms import Bbox

from m2m import evaluate, paths, protocol, tracking
from m2m.experiments import common

# --------------------------------------------------------------------------------------------
# Design tokens (dataviz reference palette, light mode)
# --------------------------------------------------------------------------------------------

# Validated with the dataviz palette validator (light surface, adjacent pairs): worst CVD ΔE 9.1
# (target 8), worst normal-vision ΔE 19.6 (floor 15). Slots 3-5 sit below 3:1 contrast on the
# surface, so identity never rests on colour alone: every series also has a marker and a label.
# The slot order is the colour-blind-safety mechanism: never reorder or extend it.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
# Second identity channel. 7 shapes against 8 hues: a (hue, shape) pair repeats only after 56.
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NOT_RECORDED = "#f0efec"  # heatmap cells without a value (neutral grey, outside the ramp)
REFERENCE_COLOR = INK_MUTED
# One-hue sequential ramp (blue 100 -> 700) for magnitudes such as permutation importance.
SEQUENTIAL_BLUES = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
# Diverging pair for signed ρ: red below 0, blue above, the neutral grey exactly at 0.
DIVERGING = ("#e34948", NOT_RECORDED, "#2a78d6")
MIN_COLOUR_LIMIT = 0.05  # keeps an all-near-zero heatmap from saturating on noise
PREFERRED_SANS = ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans")

# Mark sizes in points (1 pt ≈ 1.33 screen px): 2 px lines, 8 px markers with a surface ring.
LINE_WIDTH = 1.5
MARKER_SIZE = 6.0
RING_WIDTH = 1.0
HAIRLINE = 0.7
BAND_ALPHA = 0.12  # a wash, never a saturated block
SD_BAR_WIDTH = 3.0
CI_BAR_WIDTH = 1.0
TITLE_SIZE = 11.5
TEXT_SIZE = 9.0
LABEL_SIZE = 8.0
NOTE_SIZE = 7.5
LINE_SPACING = 1.35  # text line height as a multiple of the font size
CHAR_WIDTH_EM = 0.5  # average glyph width of the sans fonts above, for wrapping notes
POINTS_PER_INCH = 72.0

WIDTH = 7.2  # inches: one README column (1,440 px at 200 dpi)
REFERENCE_LINE = {"color": REFERENCE_COLOR, "linestyle": ":", "linewidth": 1.2}
THRESHOLD_LINE = {"color": INK_SECONDARY, "linestyle": (0, (4, 3)), "linewidth": 1.0}
LEADER_LINE = {"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.6, "shrinkA": 1.0,
               "shrinkB": 4.0}

# Data sources as experiments record them: "D1" (MiMeNet tables, MetaPhlAn2 species), "D1+MGM"
# (MGM embeddings of D1's genera) and "D2" (curated GTDB-genus tables, E4). D2 lines are dashed.
D1 = "D1"
D2_PREFIX = "D2"
SOLID = "-"
DASHED = (0, (5, 2.5))
SEED_RHO_PREFIX = "rho_seed"  # per-seed columns of protocol.write_per_metabolite files
OTHER_FAMILY = "Other"
JITTER_SEED = 0  # strip-plot jitter only separates dots; fixed so reruns draw identical files

# Label slots tried around a scatter point, nearest first: (dx, dy) in points and alignment.
LABEL_SLOTS = (
    (6, 0, "left", "center"), (-6, 0, "right", "center"),
    (0, 7, "center", "bottom"), (0, -7, "center", "top"),
    (5, 5, "left", "bottom"), (-5, 5, "right", "bottom"),
    (5, -5, "left", "top"), (-5, -5, "right", "top"),
    (18, 12, "left", "bottom"), (-18, 12, "right", "bottom"),
    (18, -12, "left", "top"), (-18, -12, "right", "top"),
    (0, 20, "center", "bottom"), (0, -20, "center", "top"),
)
LEADER_MIN_OFFSET = 12.0  # points: labels this far from their point get a leader line
MARKER_CLEARANCE = 4.0  # points kept free around other markers when placing a label

# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------

FIGURE_NAMES = (
    "learning_curve",
    "model_comparison",
    "per_metabolite_rho",
    "null_distribution",
    "mgm_coverage",
    "rho_by_class",
    "importance_heatmap",
    "cv_vs_external",
    "ablations",
)
PER_METABOLITE_DIR = "per_metabolite"
NULL_DIR = "null"  # under DATA_PROCESSED: pooled permutation-null ρ, <data>__<target_set>.npy
# The pooled null lives in DATA_PROCESSED, its threshold in RESULTS (two separately redirected
# folders), so the figure re-derives the threshold from the pooled values and requires it to
# match the recorded one. Same values and same np.quantile: only the CSV's float text can differ.
NULL_MATCH_TOLERANCE = 1e-9
NULL_RECORD_COLUMNS = ("data", "target_set", "threshold", "quantile", "n_permutations",
                       "n_features")
MGM_COVERAGE_CSV = "mgm_coverage.csv"
INTERPRETATION_DIR = "interpretation"
RHO_BY_CLASS_CSV = "rho_by_class.csv"
IMPORTANCE_CSV = "permutation_importance.csv"
ABLATIONS_CSV = "ablations.csv"
# learning_curve.csv columns (written by common.record_curve_point) that E5 relies on.
CURVE_COLUMNS = ("model", "data", "target_set", "n_train", "seed", "mean_rho", "n_features")
CURVE_SUMMARY_COLUMNS = (
    "model", "data", "target_set", "n_train", "mean_rho_mean", "mean_rho_sd", "n_seeds",
    "n_features_mean",
)


class MissingInput(Exception):
    """A figure's input is absent or unusable; the figure is skipped with this reason."""


@dataclass
class FigureReport:
    drawn: dict[str, Path] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------


def run(cfg: Mapping[str, Any], *, stage: str | None = None, n_jobs: int = 1) -> None:
    """Draw every figure whose inputs exist; `stage` = one figure name draws only that one."""
    del n_jobs  # drawing is quick and single-process
    print_report(draw_figures(cfg, common.stages_to_run(stage, FIGURE_NAMES)))


def draw_figures(cfg: Mapping[str, Any], names: Sequence[str] = FIGURE_NAMES) -> FigureReport:
    """Draw the named figures into paths.FIGURES; missing inputs skip a figure, never crash."""
    # A dry run must refuse the committed results/ before anything is written. Only some
    # drawers name the test cohort (which also checks this), so without this call a dry run
    # would write the figures that come before the first such drawer.
    common.dry_run()
    styles = model_styles(cfg)
    report = FigureReport()
    for name in names:
        try:
            fig = _DRAWERS[name](cfg, styles)
        except MissingInput as reason:
            report.skipped[name] = str(reason)
            continue
        dpi = int(cfg["figures"]["dpi"])
        report.drawn[name] = save_figure(fig, paths.FIGURES / f"{name}.png", dpi=dpi)
    return report


def print_report(report: FigureReport) -> None:
    if report.drawn:
        # The full path: a scratch run must not read as a write into the committed results/.
        print(f"[figures] drew {len(report.drawn)} figure(s) in {paths.FIGURES}:")
        for path in report.drawn.values():
            print(f"  {path.name}")
    for name, reason in report.skipped.items():
        print(f"[figures] skipped {name}: {reason}")


def save_figure(fig: Figure, path: Path, *, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(_rc_params()):  # fonts are resolved when the figure is rendered
        fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    return path


# --------------------------------------------------------------------------------------------
# One colour per model
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelStyle:
    color: str
    marker: str
    family: str


@dataclass(frozen=True)
class ModelStyles:
    """Colour, marker and family of every model, plus the presentation order (by family)."""

    styles: Mapping[str, ModelStyle]
    rank: Mapping[str, int]
    reference: frozenset[str] = frozenset()

    def __getitem__(self, model: str) -> ModelStyle:
        if model not in self.styles:
            raise KeyError(f"no style for model {model!r}: build styles from every input file")
        return self.styles[model]

    def sort(self, models: Iterable[str]) -> list[str]:
        """Unique models in presentation order: family, then the configured order, then name."""
        return sorted(set(models), key=lambda m: self.rank[m])

    def is_reference(self, model: str) -> bool:
        return model in self.reference


def build_model_styles(
    models: Iterable[str],
    *,
    families: Sequence[Mapping[str, Any]],
    experiments: Mapping[str, str] | None = None,
    model_order: Sequence[str] = (),
    reference_models: Iterable[str] = (),
    priority: Iterable[str] = (),
) -> ModelStyles:
    """Deterministic styles: they depend on the set of models, never on which figure asks.

    Priority models (those on the learning curve) take the first palette slots, so the key
    figure always shows distinct hues in the validated order. Every other model takes the next
    slot its family has not used yet. Reference models (the mean predictor) are grey.
    """
    experiments = dict(experiments or {})
    reference, wanted_first = set(reference_models), set(priority)
    family_of = {m: _family_index(m, experiments.get(m), families) for m in set(models)}
    family_names = [str(f["name"]) for f in families] + [OTHER_FAMILY]

    def assignment_key(model: str) -> tuple:
        return (model not in wanted_first, *_presentation_key(model, family_of, model_order))

    styles: dict[str, ModelStyle] = {}
    used_in_family: dict[int, set[int]] = defaultdict(set)
    next_slot = 0
    coloured = sorted((m for m in family_of if m not in reference), key=assignment_key)
    for k, model in enumerate(coloured):
        family = family_of[model]
        slot = _free_slot(next_slot, used_in_family[family])
        used_in_family[family].add(slot)
        next_slot = slot + 1
        styles[model] = ModelStyle(PALETTE[slot], MARKERS[k % len(MARKERS)], family_names[family])
    for model in reference & family_of.keys():
        styles[model] = ModelStyle(REFERENCE_COLOR, MARKERS[0], family_names[family_of[model]])
    order = sorted(family_of, key=lambda m: _presentation_key(m, family_of, model_order))
    return ModelStyles(
        styles=styles,
        rank={m: i for i, m in enumerate(order)},
        reference=frozenset(reference & family_of.keys()),
    )


def model_styles(cfg: Mapping[str, Any]) -> ModelStyles:
    """Styles for every model named in any result file, so all figures agree on colours."""
    settings = cfg["figures"]
    models, experiments = models_in_results()
    return build_model_styles(
        models,
        families=settings["families"],
        experiments=experiments,
        model_order=settings.get("model_order", ()),
        reference_models=settings.get("reference_models", ()),
        priority=_curve_models(cfg),
    )


def models_in_results() -> tuple[set[str], dict[str, str]]:
    """Every model named in the result files, and the experiment that produced each one."""
    models: set[str] = set()
    experiments: dict[str, str] = {}
    # summary.csv first: its `experiment` is the module that evaluated the model.
    for path in (_summary_path(), _curve_path()):
        table = _model_columns(path)
        if {"model", "experiment"} <= set(table.columns):
            for model, experiment in zip(table["model"], table["experiment"], strict=True):
                experiments.setdefault(str(model), str(experiment))
        models.update(map(str, table.get("model", [])))
    for path in (_rho_by_class_path(), _importance_path(), _ablations_path()):
        models.update(map(str, _model_columns(path).get("model", [])))
    models.update(name.split("__")[0] for name in _per_metabolite_names())
    return models, experiments


def _family_index(
    model: str, experiment: str | None, families: Sequence[Mapping[str, Any]]
) -> int:
    """The experiment that produced a model decides its family; the name prefix is a fallback."""
    for i, family in enumerate(families):
        if experiment is not None and experiment in family.get("experiments", ()):
            return i
    for i, family in enumerate(families):
        if any(model.startswith(prefix) for prefix in family.get("prefixes", ())):
            return i
    return len(families)


def _order_index(model: str, model_order: Sequence[str]) -> int:
    """Position in the configured order: an exact name first, else the first matching prefix."""
    if model in model_order:
        return list(model_order).index(model)
    for i, entry in enumerate(model_order):
        if model.startswith(f"{entry}_"):
            return i
    return len(model_order)


def _presentation_key(
    model: str, family_of: Mapping[str, int], model_order: Sequence[str]
) -> tuple[int, int, str]:
    return (family_of[model], _order_index(model, model_order), model)


def _free_slot(start: int, used: set[int]) -> int:
    """First palette slot from `start` (cyclically) that the family has not used yet."""
    for step in range(len(PALETTE)):
        slot = (start + step) % len(PALETTE)
        if slot not in used:
            return slot
    return start % len(PALETTE)  # more than 8 models in one family: hues repeat, markers differ


def _curve_models(cfg: Mapping[str, Any]) -> set[str]:
    path = _curve_path()
    if not path.exists():
        return set()
    curve = pd.read_csv(path, usecols=lambda c: c in {"model", "target_set"})
    if not {"model", "target_set"} <= set(curve.columns):
        return set()
    wanted = curve["target_set"] == cfg["learning_curve"]["target_set"]
    return set(map(str, curve.loc[wanted, "model"]))


# --------------------------------------------------------------------------------------------
# Figure 1: learning curve (E5, the key figure for Q3)
# --------------------------------------------------------------------------------------------


def curve_summary(curve: pd.DataFrame) -> pd.DataFrame:
    """Mean ± SD over seeds of every learning-curve point (model × data × target set × n).

    Uses evaluate.aggregate_seeds, the aggregation behind summary.csv (SD with ddof=1).
    """
    keys = ["model", "data", "target_set", "n_train"]
    records = []
    for key, group in curve.groupby(keys, sort=True):
        stats = evaluate.aggregate_seeds(group[["mean_rho", "n_features"]].to_dict("records"))
        records.append(
            {
                **dict(zip(keys, key, strict=True)),
                "mean_rho_mean": stats["mean_rho_mean"],
                "mean_rho_sd": stats["mean_rho_sd"],
                "n_seeds": int(group["seed"].nunique()),
                "n_features_mean": stats["n_features_mean"],
            }
        )
    return pd.DataFrame(records, columns=list(CURVE_SUMMARY_COLUMNS))


def series_label(model: str, data: str) -> str:
    """Legend name of a line: D2 models say so, since their dashes need a key."""
    return f"{model} ({data})" if _is_d2(data) else model


def _styled(draw: Callable[..., Figure]) -> Callable[..., Figure]:
    """Build the figure under the shared rcParams (no global matplotlib state is touched)."""

    @functools.wraps(draw)
    def wrapper(*args: Any, **kwargs: Any) -> Figure:
        with mpl.rc_context(_rc_params()):
            return draw(*args, **kwargs)

    return wrapper


@_styled
def learning_curve(
    curve: pd.DataFrame,
    *,
    styles: ModelStyles,
    target_set: str,
    test_cohort: str,
    left_out: int = 0,
) -> Figure:
    """Mean ρ on the test cohort against the number of PRISM training samples.

    One line per model (and data source): mean over seeds, a band of ± 1 SD over seeds, D2
    models dashed, the mean predictor as the dotted ρ = 0 line. Lines end in a direct label.
    `left_out` = recorded rows of this target set that `on_saved_subsets` removed (named in the
    footer, so the figure never hides that it dropped stale points).
    """
    rows = curve.loc[curve["target_set"] == target_set]
    if rows.empty:
        stale = f" on the saved PRISM subsets ({left_out} other rows)" if left_out else ""
        raise MissingInput(f"no learning-curve rows for target set {target_set}{stale}")
    summary = curve_summary(rows)
    sizes = sorted(int(n) for n in summary["n_train"].unique())
    fig = Figure(figsize=(WIDTH, 4.9), layout="constrained")
    ax = fig.subplots()
    ax.axhline(0.0, **REFERENCE_LINE, zorder=1, label="mean predictor (ρ = 0)")
    ends = []
    for (model, data), line in _curve_series(summary, styles):
        style = styles[model]
        x = line["n_train"].to_numpy(dtype=float)
        y = line["mean_rho_mean"].to_numpy(dtype=float)
        sd = np.nan_to_num(line["mean_rho_sd"].to_numpy(dtype=float))
        ax.fill_between(x, y - sd, y + sd, color=style.color, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(
            x, y, color=style.color, linestyle=_data_linestyle(data), linewidth=LINE_WIDTH,
            marker=style.marker, markersize=MARKER_SIZE, markeredgecolor=SURFACE,
            markeredgewidth=RING_WIDTH, label=series_label(model, data), zorder=3,
        )
        ends.append((x[-1], y[-1], series_label(model, data), style.color))
    _log2_size_axis(ax, sizes)
    _include_zero(ax)
    ax.set_xlabel("PRISM training samples n (log₂ scale)")
    ax.set_ylabel(f"mean Spearman ρ on {test_cohort}")
    _legend_above(ax, ncol=4)
    note = (
        f"Each point: mean over {int(summary['n_seeds'].max())} seeds of the mean per-metabolite "
        "Spearman ρ; band = ± 1 SD over seeds. The PRISM subsets are stratified by diagnosis "
        "and shared by every model. Solid = D1 tables (MetaPhlAn2 species; the MGM models embed "
        "their genera); dashed = D2 tables (GTDB genera). Dotted line = mean predictor."
    )
    if left_out:
        note += (f" {left_out} recorded point(s) outside the saved PRISM subsets (stale rows) "
                 "are not shown.")
    _finish(fig, f"Learning curve: {target_set} metabolites, tested on {test_cohort}", note)
    _end_labels(ax, ends)
    return fig


def _curve_series(
    summary: pd.DataFrame, styles: ModelStyles
) -> list[tuple[tuple[str, str], pd.DataFrame]]:
    """(model, data) lines in presentation order; the mean predictor is the zero line instead."""
    series = [
        ((str(model), str(data)), group.sort_values("n_train"))
        for (model, data), group in summary.groupby(["model", "data"], sort=False)
        if not styles.is_reference(str(model))
    ]
    return sorted(series, key=lambda item: (styles.rank[item[0][0]], item[0][1]))


def _log2_size_axis(ax: Axes, sizes: Sequence[int]) -> None:
    """Doubling sizes are equally spaced on log₂; ticks sit at the sizes actually trained."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, labels=[str(n) for n in sizes])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(min(sizes) / 1.15, max(sizes) * 1.15)
    ax.grid(axis="x", visible=False)


def _end_labels(ax: Axes, ends: Sequence[tuple[float, float, str, str]]) -> None:
    """Name each line right of its last point; labels are spread apart and tied by a leader.

    Placed after the layout has run, so the spacing uses the real axes height.
    """
    if not ends:
        return
    _laid_out(ax.figure)
    low, high = ax.get_ylim()
    wanted = [(y - low) / (high - low) for _, y, _, _ in ends]
    for (x, y, label, color), y_text in zip(ends, _spread(wanted, _line_gap(ax)), strict=True):
        ax.annotate(
            label, xy=(x, y), xycoords="data", xytext=(1.02, y_text),
            textcoords="axes fraction", va="center", ha="left", fontsize=LABEL_SIZE, color=INK,
            arrowprops={**LEADER_LINE, "color": color, "linewidth": 0.8},
        )


# --------------------------------------------------------------------------------------------
# Figure 2: model comparison on the external cohort
# --------------------------------------------------------------------------------------------


@_styled
def model_comparison(
    summary: pd.DataFrame,
    *,
    styles: ModelStyles,
    target_sets: Sequence[str],
    rho_count_key: str,
    rho_threshold: float,
    ci_label: str,
    test_cohort: str,
    null_quantile: float,
) -> Figure:
    """Mean ρ per model and target set: seed SD (thick bar) and bootstrap CI (thin bar).

    Models are grouped by family (baselines, MLP, MGM, E4), never ranked by the test score.
    Each panel's right-hand table gives N(ρ > threshold) and N(ρ > permutation null).
    """
    sets = _ordered_present(summary["target_set"], target_sets)
    rows = _family_rows(styles.sort(summary["model"]), styles)
    fig = Figure(
        figsize=(1.6 + 2.9 * len(sets), 1.4 + 0.24 * len(rows)), layout="constrained"
    )
    grid = fig.add_gridspec(1, 2 * len(sets), width_ratios=[3.0, 1.15] * len(sets))
    first: Axes | None = None
    for i, target_set in enumerate(sets):
        ax = fig.add_subplot(grid[0, 2 * i], sharex=first, sharey=first)
        table_ax = fig.add_subplot(grid[0, 2 * i + 1], sharey=ax)
        part = summary.loc[summary["target_set"] == target_set].set_index("model")
        _comparison_panel(ax, part, rows, styles)
        _count_table(table_ax, part, rows, rho_count_key, rho_threshold)
        ax.set_title(_panel_title(target_set, part), loc="left", fontsize=TEXT_SIZE)
        ax.set_xlabel(f"mean Spearman ρ on {test_cohort}")
        if first is None:
            first = ax
            _row_labels(ax, rows)
        else:
            ax.tick_params(labelleft=False)
    return _finish(
        fig,
        f"External test on {test_cohort}: every model and target set",
        f"Dot = mean over seeds; thick bar = ± 1 SD over seeds; thin bar = {ci_label} over test "
        "samples. Dotted line = mean predictor (ρ = 0). Counts are means over seeds: "
        f"N(ρ > {rho_threshold:g}) and N(ρ > the {_nth(null_quantile)} percentile of the "
        "permutation null). Models are grouped by family, not ranked.",
    )


def _comparison_panel(
    ax: Axes, part: pd.DataFrame, rows: Sequence[tuple[str, str | None]], styles: ModelStyles
) -> None:
    ax.axvline(0.0, **REFERENCE_LINE, zorder=1)
    for y, (_, model) in enumerate(rows):
        if model is None or model not in part.index:
            continue
        row, style = part.loc[model], styles[model]
        mean = float(row["mean_rho_mean"])
        low, high = _optional(row, "mean_rho_ci_low"), _optional(row, "mean_rho_ci_high")
        sd = np.nan_to_num(_optional(row, "mean_rho_sd"))
        if np.isfinite(low) and np.isfinite(high):
            ax.hlines(y, low, high, color=style.color, linewidth=CI_BAR_WIDTH, zorder=2)
        if sd > 0:
            ax.hlines(y, mean - sd, mean + sd, color=style.color, linewidth=SD_BAR_WIDTH,
                      alpha=0.55, zorder=2)
        ax.plot(mean, y, linestyle="none", marker=style.marker, markersize=MARKER_SIZE,
                color=style.color, markeredgecolor=SURFACE, markeredgewidth=RING_WIDTH,
                label=model, zorder=3)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(len(rows) - 0.5, -0.7)


def _count_table(
    ax: Axes,
    part: pd.DataFrame,
    rows: Sequence[tuple[str, str | None]],
    rho_count_key: str,
    rho_threshold: float,
) -> None:
    """Two right-hand columns: N(ρ > threshold) and N(ρ > null), as means over seeds."""
    ax.axis("off")
    columns = ((0.3, rho_count_key, f"ρ>{rho_threshold:g}"), (0.85, "n_above_null_mean", ">null"))
    for x, key, header in columns:
        ax.text(x, 1.0, header, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=LABEL_SIZE, color=INK_SECONDARY)
        for y, (_, model) in enumerate(rows):
            if model is not None and model in part.index:
                value = _optional(part.loc[model], key)
                ax.text(x, y, _count_text(value), transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=LABEL_SIZE, color=INK)


def _family_rows(models: Sequence[str], styles: ModelStyles) -> list[tuple[str, str | None]]:
    """(label, model) rows with a header row (model None) before each family."""
    rows: list[tuple[str, str | None]] = []
    current = None
    for model in models:
        family = styles[model].family
        if family != current:
            rows.append((family, None))
            current = family
        rows.append((model, model))
    return rows


def _row_labels(ax: Axes, rows: Sequence[tuple[str, str | None]]) -> None:
    ax.set_yticks(range(len(rows)), labels=[label for label, _ in rows])
    ax.tick_params(axis="y", length=0)
    for tick, (_, model) in zip(ax.get_yticklabels(), rows, strict=True):
        if model is None:  # family header
            tick.set_fontweight("bold")
            tick.set_color(INK)


def _panel_title(target_set: str, part: pd.DataFrame) -> str:
    if "n_features_mean" not in part.columns:
        return target_set
    counts = part["n_features_mean"].dropna().round().astype(int)
    if counts.empty:
        return target_set
    low, high = counts.min(), counts.max()
    evaluated = f"{low}" if low == high else f"{low}–{high}"
    return f"{target_set} · {evaluated} metabolites"


# --------------------------------------------------------------------------------------------
# Figure 3: distribution of per-metabolite ρ
# --------------------------------------------------------------------------------------------


@_styled
def per_metabolite_rho(
    rho: Mapping[str, pd.Series],
    *,
    styles: ModelStyles,
    data_of: Mapping[str, str],
    null_thresholds: Mapping[str, float],
    target_set: str,
    test_cohort: str,
    null_quantile: float,
) -> Figure:
    """ECDF of per-metabolite ρ (as `load_per_metabolite` returns it), one panel per family,
    with the permutation-null threshold of the panel's data source."""
    models = styles.sort(rho)
    families = list(dict.fromkeys(styles[m].family for m in models))
    ncols = min(2, len(families))
    nrows = -(-len(families) // ncols)
    fig = Figure(figsize=(WIDTH, 1.0 + 3.3 * nrows), layout="constrained")
    axes = fig.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False).ravel()
    for ax, family in zip(axes, families, strict=False):
        members = [m for m in models if styles[m].family == family]
        for model in members:
            data = data_of.get(model, D1)
            ax.ecdf(rho[model].fillna(0.0).to_numpy(dtype=float), color=styles[model].color,
                    linewidth=LINE_WIDTH, linestyle=_data_linestyle(data),
                    label=series_label(model, data))
        keys = [_null_key(data_of.get(m, D1), null_thresholds) for m in members]
        sources = [key for key in dict.fromkeys(keys) if key is not None]
        for key in sources:
            ax.axvline(null_thresholds[key], **THRESHOLD_LINE)
        ax.axvline(0.0, color=AXIS, linewidth=HAIRLINE, zorder=0)
        # Above the panel the legend hides no curve; its title carries the threshold values,
        # since text on the plot would cross the curves near the threshold.
        values = ", ".join(f"{key} {null_thresholds[key]:.2f}" for key in sources)
        title = f"{family} · null {_nth(null_quantile)} pct: {values}" if sources else family
        ax.legend(title=title, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2,
                  fontsize=LABEL_SIZE, title_fontsize=TEXT_SIZE, alignment="left",
                  handlelength=2.6, borderaxespad=0.0, columnspacing=1.6)
    for ax in axes[len(families):]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[: len(families)]):
        bottom = i + ncols >= len(families)  # no panel is drawn below this one
        ax.tick_params(axis="x", labelbottom=bottom)
        if bottom:
            ax.set_xlabel(f"Spearman ρ on {test_cohort}")
        if i % ncols == 0:
            ax.set_ylabel("fraction of metabolites ≤ ρ")
    return _finish(
        fig,
        f"Per-metabolite ρ on {test_cohort} ({target_set})",
        "Empirical CDF over metabolites: a curve further right predicts more metabolites well. "
        "ρ is averaged over seeds after counting an undefined ρ (constant prediction or truth) "
        f"as 0, as in summary.csv. Dashed vertical line: {_nth(null_quantile)} percentile of the "
        "pooled permutation null of the models' data source (value in the panel title).",
    )


# --------------------------------------------------------------------------------------------
# Figure 4: permutation null vs a real model
# --------------------------------------------------------------------------------------------


@_styled
def null_distribution(
    null_rho: np.ndarray,
    model_rho: pd.Series,
    *,
    model: str,
    styles: ModelStyles,
    quantile: float,
    bins: int,
    target_set: str,
    test_cohort: str,
) -> Figure:
    """Histogram of pooled permutation-null ρ against the model's per-metabolite ρ."""
    null_rho = np.asarray(null_rho, dtype=float)
    real = model_rho.fillna(0.0).to_numpy(dtype=float)
    threshold = float(np.quantile(null_rho, quantile))
    edges = np.linspace(min(null_rho.min(), real.min()), max(null_rho.max(), real.max()), bins + 1)
    fig = Figure(figsize=(WIDTH, 4.0), layout="constrained")
    ax = fig.subplots()
    heights, _ = np.histogram(null_rho, bins=edges, density=True)
    ax.stairs(heights, edges, fill=True, color=INK_MUTED, alpha=0.35, linewidth=0,
              label=f"permutation null ({len(null_rho):,} ρ, pooled)")
    real_heights, _ = np.histogram(real, bins=edges, density=True)
    first, last = np.flatnonzero(real_heights)[[0, -1]]  # outline only where there is data
    span = slice(first, last + 1)
    color = styles[model].color
    ax.stairs(real_heights[span], edges[first : last + 2], fill=True, color=color,
              alpha=BAND_ALPHA, linewidth=0)
    ax.stairs(real_heights[span], edges[first : last + 2], baseline=0.0, color=color,
              linewidth=LINE_WIDTH, label=f"{model} ({len(real)} metabolites)")
    above = int((real > threshold).sum())
    ax.axvline(threshold, **THRESHOLD_LINE,
               label=f"{quantile:.0%} quantile of the null = {threshold:.3f}")
    ax.annotate(f"{model}: {above} of {len(real)} metabolites above it", xy=(threshold, 0.94),
                xycoords=("data", "axes fraction"), xytext=(6, 0), textcoords="offset points",
                fontsize=LABEL_SIZE, color=INK, va="center")
    # Headroom so the legend and the annotation sit above the tallest bar.
    ax.set_ylim(0.0, 1.45 * max(heights.max(), real_heights.max()))
    ax.set_xlabel(f"Spearman ρ on {test_cohort} ({target_set})")
    ax.set_ylabel("density")
    ax.legend(loc="upper left", fontsize=LABEL_SIZE)
    return _finish(
        fig,
        f"Permutation null vs {model}: chance level of per-metabolite ρ",
        "Null: the training microbiomes are shuffled against their metabolomes, the model is "
        "refit and scored; the ρ of every permutation and metabolite are pooled. Undefined ρ "
        "count as 0.",
    )


# --------------------------------------------------------------------------------------------
# Figure 5: MGM vocabulary coverage
# --------------------------------------------------------------------------------------------


@_styled
def mgm_coverage(coverage: pd.DataFrame, *, cohort_order: Sequence[str]) -> Figure:
    """Per-sample share of abundance in MGM's vocabulary: one jittered strip per cohort."""
    cohorts = _ordered_present(coverage["cohort"], cohort_order)
    rng = np.random.default_rng(JITTER_SEED)
    fig = Figure(figsize=(WIDTH, 1.6 + 0.6 * len(cohorts)), layout="constrained")
    ax = fig.subplots()
    for y, cohort in enumerate(cohorts):
        values = coverage.loc[coverage["cohort"] == cohort, "abundance_in_vocab"]
        values = values.to_numpy(dtype=float)
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(values, y + jitter, s=14, color=INK_SECONDARY, alpha=0.5, linewidths=0)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.hlines(y, q1, q3, color=INK, linewidth=SD_BAR_WIDTH, alpha=0.8)
        ax.vlines(median, y - 0.3, y + 0.3, color=INK, linewidth=LINE_WIDTH)
        ax.annotate(f"median {median:.3f} · n = {len(values)}", xy=(1.0, y),
                    xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=LABEL_SIZE, color=INK)
    ax.set_yticks(range(len(cohorts)), labels=list(cohorts))
    ax.set_ylim(len(cohorts) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=1.01)
    ax.set_xlabel("fraction of the sample's genus-level abundance in MGM's vocabulary")
    return _finish(
        fig,
        "MGM vocabulary coverage per sample",
        "Dots: samples. Thick bar: interquartile range; vertical tick: median.",
    )


# --------------------------------------------------------------------------------------------
# Figure 6: ρ by chemical class
# --------------------------------------------------------------------------------------------


@_styled
def rho_by_class(table: pd.DataFrame, *, styles: ModelStyles, test_cohort: str) -> Figure:
    """Mean ρ per chemical class (rows) and model (columns), diverging around ρ = 0.

    A heatmap rather than dots: E6 writes up to eight models per class, too many to dodge.
    The mean predictor (ρ = 0 everywhere) is left out; the class sizes are in the row labels.
    """
    shown = table.loc[~table["model"].astype(str).map(styles.is_reference)]
    if shown.empty:
        raise MissingInput("rho_by_class.csv holds only the reference (mean) model")
    models = styles.sort(shown["model"].astype(str))
    values = shown.pivot_table(index="chemical_class", columns="model", values="mean_rho",
                               aggfunc="first")
    classes = values.mean(axis=1).sort_values(ascending=False).index.tolist()
    values = values.reindex(index=classes, columns=models)
    sizes = shown.groupby("chemical_class")["n_metabolites"].max()
    matrix = values.to_numpy(dtype=float)
    limit = max(float(np.nanmax(np.abs(matrix))), MIN_COLOUR_LIMIT)
    # Missing values are blank, so they never look like ρ = 0 (the neutral grey).
    cmap = LinearSegmentedColormap.from_list("rho", DIVERGING).with_extremes(bad=SURFACE)
    fig = Figure(figsize=(max(WIDTH, 3.4 + 0.8 * len(models)), 1.9 + 0.3 * len(classes)),
                 layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-limit, vmax=limit,
                      aspect="auto", interpolation="nearest")
    for (i, j), value in np.ndenumerate(matrix):
        if np.isfinite(value):  # the table view, inside the cells
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=LABEL_SIZE - 0.5,
                    color=_text_colour_on(image.cmap(image.norm(value))))
    ax.set_xticks(range(len(models)), labels=models, rotation=35, ha="right",
                  rotation_mode="anchor")
    ax.set_yticks(range(len(classes)), labels=[f"{c} ({int(sizes[c])})" for c in classes])
    _cell_gaps(ax, len(classes), len(models))
    fig.colorbar(image, ax=ax, shrink=0.6, label=f"mean Spearman ρ on {test_cohort}")
    return _finish(
        fig,
        "Which chemical classes are predictable?",
        "Mean per-metabolite ρ of each chemical class (Putative.Chemical.Class; metabolites in "
        "brackets), classes ordered by their mean over models. Red = below 0, blue = above, "
        "blank = no value. Small classes are pooled by E6. The mean predictor (ρ = 0) is not "
        "shown.",
    )


# --------------------------------------------------------------------------------------------
# Figure 7: permutation-importance heatmap
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _ImportancePanel:
    values: pd.DataFrame  # metabolite label × taxon
    signs: pd.DataFrame  # "+", "−" or "" per cell
    has_negative: bool


@_styled
def importance_heatmap(table: pd.DataFrame, *, top_k: int) -> Figure:
    """Metabolites × the union of their top-k taxa, coloured by permutation importance.

    One panel per model. "+"/"−" marks the sign of the ridge coefficient in each top-k cell;
    grey cells have no recorded importance.
    """
    models = list(dict.fromkeys(table["model"].astype(str)))
    panels = [_importance_matrix(table.loc[table["model"] == m], top_k) for m in models]
    recorded = np.concatenate([p.values.to_numpy(dtype=float).ravel() for p in panels])
    recorded = recorded[np.isfinite(recorded)]
    if recorded.size == 0 or recorded.max() <= 0:
        raise MissingInput("permutation_importance.csv has no positive importance to colour")
    vmax = float(recorded.max())
    height = sum(0.9 + 0.26 * len(p.values) for p in panels) + 1.4
    width = max(WIDTH, 2.6 + 0.24 * max(p.values.shape[1] for p in panels))
    fig = Figure(figsize=(width, height), layout="constrained")
    axes = fig.subplots(len(panels), 1, squeeze=False,
                        height_ratios=[len(p.values) + 3 for p in panels]).ravel()
    cmap = LinearSegmentedColormap.from_list("blues", SEQUENTIAL_BLUES).with_extremes(
        bad=NOT_RECORDED
    )
    image = None
    for ax, model, panel in zip(axes, models, panels, strict=True):
        image = _heatmap_panel(ax, panel, cmap, vmax)
        ax.set_title(f"{model}: top {top_k} taxa per metabolite", loc="left", fontsize=TEXT_SIZE)
    fig.colorbar(image, ax=list(axes), shrink=0.6, label="permutation importance")
    note = (
        "Columns: every taxon that is among the top-k drivers of at least one metabolite. "
        "+/− = sign of the ridge coefficient (top-k cells only); grey = importance not recorded."
    )
    if any(p.has_negative for p in panels):
        note += " Negative importances are shown as 0."
    return _finish(fig, "Which taxa drive the best-predicted metabolites?", note)


def _importance_matrix(part: pd.DataFrame, top_k: int) -> _ImportancePanel:
    metabolites = list(dict.fromkeys(part["metabolite"].astype(str)))
    names = part.drop_duplicates("metabolite").set_index("metabolite")["compound_name"]
    top = part.loc[part["rank"] <= top_k]
    counts = top.groupby("taxon").agg(n=("metabolite", "nunique"), total=("importance_mean", "sum"))
    taxa = counts.sort_values(["n", "total"], ascending=False).index.tolist()
    values = part.pivot_table(index="metabolite", columns="taxon", values="importance_mean",
                              aggfunc="first").reindex(index=metabolites, columns=taxa)
    signs = pd.DataFrame("", index=metabolites, columns=taxa)
    if "ridge_coef" in top.columns:
        for row in top.itertuples(index=False):
            if np.isfinite(row.ridge_coef) and row.ridge_coef != 0:
                signs.loc[str(row.metabolite), row.taxon] = "+" if row.ridge_coef > 0 else "−"
    labels = [_compound_label(names.get(m), m) for m in metabolites]
    has_negative = bool((values.to_numpy(dtype=float) < 0).any())
    return _ImportancePanel(values.set_axis(labels), signs.set_axis(labels), has_negative)


def _heatmap_panel(ax: Axes, panel: _ImportancePanel, cmap: Any, vmax: float) -> Any:
    shown = panel.values.clip(lower=0.0).to_numpy(dtype=float)
    image = ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, vmin=0.0, vmax=vmax,
                      aspect="auto", interpolation="nearest")
    n_rows, n_cols = shown.shape
    for (i, j), sign in np.ndenumerate(panel.signs.to_numpy()):
        if sign:
            fill = image.cmap(image.norm(shown[i, j])) if np.isfinite(shown[i, j]) else None
            ax.text(j, i, sign, ha="center", va="center", fontsize=LABEL_SIZE,
                    color=INK if fill is None else _text_colour_on(fill))
    ax.set_xticks(range(n_cols), labels=[_taxon_label(t) for t in panel.values.columns],
                  rotation=55, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_rows), labels=list(panel.values.index))
    _cell_gaps(ax, n_rows, n_cols)
    return image


def _cell_gaps(ax: Axes, n_rows: int, n_cols: int) -> None:
    """2 px surface gaps between heatmap cells (instead of drawn borders); no ticks or spines."""
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _taxon_label(taxon: str) -> str:
    return str(taxon).replace("_", " ")


def _compound_label(name: Any, feature: str) -> str:
    return str(name) if isinstance(name, str) and name.strip() else str(feature)


# --------------------------------------------------------------------------------------------
# Figure 8: cross-validation vs external score
# --------------------------------------------------------------------------------------------


@_styled
def cv_vs_external(
    summary: pd.DataFrame,
    *,
    styles: ModelStyles,
    target_sets: Sequence[str],
    test_cohort: str,
    cv_folds: int,
) -> Figure:
    """PRISM CV mean ρ against external mean ρ. Both axes share one scale, so y = x is the
    diagonal; the distance from it is not cohort shift alone (see the note). Every point is
    named: a scatter cannot rely on more than three hues for identity."""
    # The mean predictor sits at (0, 0) by construction: it says nothing about the gap and
    # would squeeze every other point into a corner.
    scored = summary.dropna(subset=["cv_mean_rho", "mean_rho_mean"])
    rows = scored.loc[~scored["model"].astype(str).map(styles.is_reference)]
    if rows.empty:
        raise MissingInput("summary.csv has no row with both a CV and an external mean ρ")
    sets = _ordered_present(rows["target_set"], target_sets)
    values = np.concatenate([rows["cv_mean_rho"], rows["mean_rho_mean"]]).astype(float)
    pad = 0.12 * max(values.max() - values.min(), 0.05)
    low, high = values.min() - pad, values.max() + pad
    fig = Figure(figsize=(min(1.4 * WIDTH, 1.0 + 3.6 * len(sets)), 4.5), layout="constrained")
    axes = fig.subplots(1, len(sets), sharex=True, sharey=True, squeeze=False)[0]
    panels = []
    for ax, target_set in zip(axes, sets, strict=True):
        part = rows.loc[rows["target_set"] == target_set].set_index("model")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.axline((low, low), slope=1.0, **REFERENCE_LINE, zorder=1)
        points = []
        for model in styles.sort(part.index):
            style = styles[model]
            x, y = float(part.loc[model, "cv_mean_rho"]), float(part.loc[model, "mean_rho_mean"])
            ax.plot(x, y, linestyle="none", marker=style.marker, markersize=MARKER_SIZE,
                    color=style.color, markeredgecolor=SURFACE, markeredgewidth=RING_WIDTH,
                    label=model, zorder=3)
            points.append((x, y, model))
        panels.append((ax, points))
        ax.set_title(target_set, loc="left", fontsize=TEXT_SIZE)
        ax.set_xlabel(f"mean ρ, PRISM {cv_folds}-fold CV")
    axes[0].set_ylabel(f"mean ρ on {test_cohort} (mean over seeds)")
    # Two protocol effects move points off y = x besides the cohort: say so, or the figure
    # reads as a pure generalisation gap.
    note = (
        "Dotted line: y = x. The distance from it is not cohort shift alone: CV models train on "
        f"{cv_folds - 1}/{cv_folds} of PRISM (the external models on all of it), and a tuned "
        "model's CV score is the best of its hyperparameter grid, hence optimistic. Both axes "
        "use the same scale."
    )
    unscored = sorted(set(map(str, summary["model"])) - set(map(str, scored["model"])))
    if unscored:
        note += f" No CV score recorded for: {', '.join(unscored)}."
    if len(rows) < len(scored):
        note += " The mean predictor (ρ = 0 in both) is not shown."
    _finish(fig, f"Cross-validation on PRISM vs external test on {test_cohort}", note)
    for ax, points in panels:
        _point_labels(ax, points)
    return fig


def _point_labels(ax: Axes, points: Sequence[tuple[float, float, str]]) -> None:
    """Name every point: each label takes the nearest slot that overlaps no marker or other
    label and stays inside the axes (greedy, measured with the renderer after layout)."""
    if not points:
        return
    renderer = _laid_out(ax.figure)
    frame = ax.get_window_extent(renderer)
    # Other models' markers keep a clearance, so a label never sits beside the wrong point.
    reach = (MARKER_SIZE / 2 + MARKER_CLEARANCE) * ax.figure.dpi / POINTS_PER_INCH
    markers = [
        Bbox.from_extents(px - reach, py - reach, px + reach, py + reach)
        for px, py in ax.transData.transform([(x, y) for x, y, _ in points])
    ]
    labels: list[Bbox] = []
    for i, (x, y, label) in enumerate(points):
        obstacles = [box for j, box in enumerate(markers) if j != i] + labels
        dx, dy, ha, va = _free_label_slot(ax, renderer, frame, obstacles, (x, y), label)
        far = math.hypot(dx, dy) >= LEADER_MIN_OFFSET
        text = ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                           ha=ha, va=va, fontsize=LABEL_SIZE - 0.5, color=INK,
                           arrowprops=LEADER_LINE if far else None)
        labels.append(_text_box(text, renderer))


def _free_label_slot(
    ax: Axes, renderer: Any, frame: Bbox, obstacles: Sequence[Bbox], xy: tuple[float, float],
    label: str,
) -> tuple[float, float, str, str]:
    """The first slot whose text box is clear; otherwise the one with the least overlap."""
    best, best_overlap = LABEL_SLOTS[0], math.inf
    for slot in LABEL_SLOTS:
        dx, dy, ha, va = slot
        probe = ax.annotate(label, xy=xy, xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=LABEL_SIZE - 0.5)
        box = _text_box(probe, renderer)
        probe.remove()
        overlap = sum(_overlap_area(box, other) for other in obstacles)
        overlap += _area(box) - _overlap_area(box, frame)  # parts outside the axes
        if overlap == 0:
            return slot
        if overlap < best_overlap:
            best, best_overlap = slot, overlap
    return best


def _text_box(annotation: Any, renderer: Any) -> Bbox:
    """Display box of an annotation's text alone (its leader line excluded)."""
    annotation.update_positions(renderer)  # places the text; skipped by Text.get_window_extent
    return Text.get_window_extent(annotation, renderer)


def _overlap_area(a: Bbox, b: Bbox) -> float:
    both = Bbox.intersection(a, b)
    return 0.0 if both is None else _area(both)


def _area(box: Bbox) -> float:
    return max(box.width, 0.0) * max(box.height, 0.0)


# --------------------------------------------------------------------------------------------
# Figure 9: ablations
# --------------------------------------------------------------------------------------------


@_styled
def ablations(
    table: pd.DataFrame, *, styles: ModelStyles, test_cohort: str, cv_folds: int
) -> Figure:
    """CV (hollow) and external (filled, ± SD over seeds) mean ρ for every arm of each ablation."""
    groups = list(dict.fromkeys(table["ablation"].astype(str)))
    sizes = [int((table["ablation"] == g).sum()) for g in groups]
    fig = Figure(figsize=(WIDTH, 1.6 + sum(0.32 * n + 0.45 for n in sizes)), layout="constrained")
    axes = fig.subplots(len(groups), 1, sharex=True, squeeze=False,
                        height_ratios=[n + 1 for n in sizes]).ravel()
    for ax, group in zip(axes, groups, strict=True):
        part = table.loc[table["ablation"] == group].reset_index(drop=True)
        _ablation_panel(ax, part, styles)
        ax.set_title(group, loc="left", fontsize=TEXT_SIZE)
    axes[-1].set_xlabel("mean Spearman ρ")
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=MARKER_SIZE,
               markerfacecolor=SURFACE, markeredgecolor=INK_SECONDARY,
               label=f"PRISM {cv_folds}-fold CV"),
        Line2D([], [], linestyle="none", marker="o", markersize=MARKER_SIZE,
               color=INK_SECONDARY, label=f"{test_cohort} (± 1 SD over seeds)"),
    ]
    axes[0].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2,
                   fontsize=LABEL_SIZE, borderaxespad=0.2)
    return _finish(
        fig,
        "Ablations: cross-validation and external mean ρ per arm",
        "Colour and marker shape identify the model, as in every other figure. "
        "Dotted line = mean predictor (ρ = 0).",
    )


def _ablation_panel(ax: Axes, part: pd.DataFrame, styles: ModelStyles) -> None:
    ax.axvline(0.0, **REFERENCE_LINE, zorder=1)
    varying = [c for c in ("model", "target_set", "eval_space")
               if c in part.columns and part[c].nunique(dropna=True) > 1]
    labels = []
    for y, row in part.iterrows():
        style = styles[str(row["model"])]
        cv, ext = _optional(row, "cv_mean_rho"), _optional(row, "mean_rho_mean")
        sd = np.nan_to_num(_optional(row, "mean_rho_sd"))
        marks = {"linestyle": "none", "marker": style.marker, "markersize": MARKER_SIZE,
                 "markeredgecolor": style.color, "zorder": 3}
        if np.isfinite(cv):
            ax.plot(cv, y, markerfacecolor=SURFACE, markeredgewidth=1.2, **marks)
        if np.isfinite(ext):
            ax.hlines(y, ext - sd, ext + sd, color=style.color, linewidth=CI_BAR_WIDTH, zorder=2)
            ax.plot(ext, y, markerfacecolor=style.color, **marks)
        label = " · ".join([str(row["arm"]), *(str(row[c]) for c in varying)])
        # The default arm is the main protocol itself (its CV equals E0's / E1's).
        main = str(row.get("is_default_arm", "")).lower() == "true"
        labels.append(f"{label} (main)" if main else label)
    ax.set_yticks(range(len(part)), labels=labels)
    ax.set_ylim(len(part) - 0.5, -0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)


# --------------------------------------------------------------------------------------------
# Drawers: load a figure's inputs (MissingInput when absent), then draw it
# --------------------------------------------------------------------------------------------


def _draw_learning_curve(cfg: Mapping[str, Any], styles: ModelStyles) -> Figure:
    target_set = cfg["learning_curve"]["target_set"]
    curve = _read_table(_curve_path(), CURVE_COLUMNS)
    kept = on_saved_subsets(curve)
    in_set = curve["target_set"] == target_set
    left_out = int(in_set.sum() - in_set.loc[kept.index].sum())
    return learning_curve(
        kept,
        styles=styles,
        target_set=target_set,
        test_cohort=external_cohort_label(cfg),
        left_out=left_out,
    )


def _draw_model_comparison(cfg: Mapping[str, Any], styles: ModelStyles) -> Figure:
    summary = _read_table(_summary_path(), ("model", "target_set", "mean_rho_mean"))
    evaluation = cfg["evaluation"]
    return model_comparison(
        summary,
        styles=styles,
        target_sets=cfg["figures"]["target_sets"],
        rho_count_key=f"{_rho_count_metric(cfg)}_mean",
        rho_threshold=float(evaluation["rho_threshold"]),
        ci_label=f"{float(evaluation['ci_level']):.0%} bootstrap CI ({evaluation['ci_method']})",
        test_cohort=external_cohort_label(cfg),
        null_quantile=float(evaluation["null_quantile"]),
    )


def _draw_per_metabolite_rho(cfg: Mapping[str, Any], styles: ModelStyles) -> Figure:
    target_set = cfg["figures"]["main_target_set"]
    return per_metabolite_rho(
        load_per_metabolite(target_set),
        styles=styles,
        data_of=_data_sources(),
        null_thresholds=_null_thresholds(target_set),
        target_set=target_set,
        test_cohort=external_cohort_label(cfg),
        null_quantile=float(cfg["evaluation"]["null_quantile"]),
    )


def _draw_null_distribution(cfg: Mapping[str, Any], styles: ModelStyles) -> Figure:
    settings = cfg["figures"]["null_figure"]
    target_set = cfg["figures"]["main_target_set"]
    data_name = str(settings["data"])
    path = paths.DATA_PROCESSED / NULL_DIR / f"{data_name}__{target_set}.npy"
    if not path.exists():
        raise MissingInput(f"{_shown(path)} missing (pooled permutation-null ρ)")
    model = str(settings["model"])
    rho = load_per_metabolite(target_set)
    if model not in rho:
        raise MissingInput(f"no per-metabolite ρ file for {model} on {target_set}")
    recorded = _recorded_null(data_name, target_set)
    pooled = np.load(path, allow_pickle=False).ravel()
    _check_pooled_null(pooled, recorded, n_scored=len(rho[model]), source=_shown(path))
    return null_distribution(
        pooled,
        rho[model],
        model=model,
        styles=styles,
        # The quantile the recorded threshold (and so summary.csv's N above null) came from.
        quantile=float(recorded["quantile"]),
        bins=int(settings["bins"]),
        target_set=target_set,
        test_cohort=external_cohort_label(cfg),
    )


def _draw_mgm_coverage(cfg: Mapping[str, Any], styles: ModelStyles) -> Figure:
    del styles  # cohorts, not models: one neutral colour
    labels = dict(cfg["figures"]["coverage"]["cohort_labels"])
    table = _read_table(paths.RESULTS / MGM_COVERAGE_CSV, ("abundance_in_vocab",))
    if "cohort" in table.columns:
        cohort = table["cohort"].astype(str)
    else:  # derive it from the sample-id prefix: "PRISM|7122" -> PRISM
        cohort = table.iloc[:, 0].astype(str).str.split(r"[|.]", n=1, regex=True).str[0]
    table = table.assign(cohort=cohort.map(lambda c: labels.get(c, c)).to_numpy())
    return mgm_coverage(table, cohort_order=list(dict.fromkeys(labels.values())))


def _draw_rho_by_class(cfg: Mapping[str, Any], styles: ModelStyles) -> Figure:
    table = _read_table(
        _rho_by_class_path(), ("model", "chemical_class", "n_metabolites", "mean_rho")
    )
    return rho_by_class(table, styles=styles, test_cohort=external_cohort_label(cfg))


def _draw_importance_heatmap(cfg: Mapping[str, Any], styles: ModelStyles) -> Figure:
    del styles  # magnitudes: one sequential hue
    table = _read_table(
        _importance_path(),
        ("model", "metabolite", "compound_name", "taxon", "importance_mean", "rank"),
    )
    return importance_heatmap(table, top_k=int(cfg["figures"]["importance"]["top_k"]))


def _draw_cv_vs_external(cfg: Mapping[str, Any], styles: ModelStyles) -> Figure:
    summary = _read_table(_summary_path(), ("model", "target_set", "mean_rho_mean", "cv_mean_rho"))
    return cv_vs_external(
        summary, styles=styles, target_sets=cfg["figures"]["target_sets"],
        test_cohort=external_cohort_label(cfg), cv_folds=int(cfg["cv"]["n_splits"]),
    )


def _draw_ablations(cfg: Mapping[str, Any], styles: ModelStyles) -> Figure:
    # External columns appear only once the ablations' external stage has run.
    table = _read_table(_ablations_path(), ("ablation", "arm", "model", "cv_mean_rho"))
    return ablations(table, styles=styles, test_cohort=external_cohort_label(cfg),
                     cv_folds=int(cfg["cv"]["n_splits"]))


_DRAWERS: dict[str, Callable[[Mapping[str, Any], ModelStyles], Figure]] = {
    "learning_curve": _draw_learning_curve,
    "model_comparison": _draw_model_comparison,
    "per_metabolite_rho": _draw_per_metabolite_rho,
    "null_distribution": _draw_null_distribution,
    "mgm_coverage": _draw_mgm_coverage,
    "rho_by_class": _draw_rho_by_class,
    "importance_heatmap": _draw_importance_heatmap,
    "cv_vs_external": _draw_cv_vs_external,
    "ablations": _draw_ablations,
}


# --------------------------------------------------------------------------------------------
# Input helpers
# --------------------------------------------------------------------------------------------


def external_cohort_label(cfg: Mapping[str, Any]) -> str:
    """The external cohort's name, or a loud marker that a dry run used a PRISM hold-out."""
    if common.dry_run():
        return "PRISM hold-out (DRY RUN)"
    return str(cfg["figures"]["test_cohort"])


def load_per_metabolite(target_set: str) -> dict[str, pd.Series]:
    """{model: per-metabolite ρ} from results/per_metabolite/*__<set>.csv.gz.

    ρ is counted as the summaries count it: an undefined ρ is 0 in its seed, then the seeds are
    averaged (so each curve's mean is summary.csv's mean ρ). The file's own rho_mean skips
    undefined seeds instead; it is used only when no per-seed column exists.
    """
    suffix = f"__{target_set}.csv.gz"
    rho = {}
    for name in _per_metabolite_names():
        if not name.endswith(suffix):
            continue
        table = pd.read_csv(paths.RESULTS / PER_METABOLITE_DIR / name, index_col="feature")
        seeds = [c for c in table.columns if str(c).startswith(SEED_RHO_PREFIX)]
        if seeds:
            rho[name.removesuffix(suffix)] = table[seeds].fillna(0.0).mean(axis=1)
        elif "rho_mean" in table.columns:
            rho[name.removesuffix(suffix)] = table["rho_mean"].fillna(0.0)
    if not rho:
        folder = _shown(paths.RESULTS / PER_METABOLITE_DIR)
        raise MissingInput(f"no {folder}/*{suffix} file with per-seed or mean ρ columns")
    return rho


def _per_metabolite_names() -> list[str]:
    folder = paths.RESULTS / PER_METABOLITE_DIR
    return sorted(p.name for p in folder.glob("*__*.csv.gz")) if folder.exists() else []


def saved_subset_points() -> set[tuple[int, int]] | None:
    """(n_train, seed) of the saved learning-curve subsets; None before any were saved."""
    path = paths.SPLITS / common.LEARNING_CURVE_SUBSETS_CSV
    if not path.exists():
        return None
    saved = pd.read_csv(path, usecols=["n_train", "seed"]).drop_duplicates()
    return set(zip(saved["n_train"].astype(int), saved["seed"].astype(int), strict=True))


def on_saved_subsets(curve: pd.DataFrame) -> pd.DataFrame:
    """The learning-curve rows at the saved (n_train, seed) subsets (all rows if none saved).

    learning_curve.csv is upserted, so rows of an older size or seed list stay in it. They were
    trained on other PRISM samples: averaging them in would break "every model sees the same
    people at each size". E5's check names them; the summary and the figure leave them out.
    """
    points = saved_subset_points()
    if points is None:
        return curve
    keys = zip(curve["n_train"].astype(int), curve["seed"].astype(int), strict=True)
    return curve.loc[[key in points for key in keys]]


def _data_sources() -> dict[str, str]:
    """model → data source (D1/D2) from summary.csv, else from learning_curve.csv."""
    sources: dict[str, str] = {}
    for path in (_summary_path(), _curve_path()):
        if path.exists():
            table = pd.read_csv(path, usecols=lambda c: c in {"model", "data"})
            if {"model", "data"} <= set(table.columns):
                for model, data in zip(table["model"], table["data"], strict=True):
                    sources.setdefault(str(model), str(data))
    return sources


def _null_thresholds(target_set: str) -> dict[str, float]:
    """data source → permutation-null threshold for `target_set` (empty if not computed)."""
    path = paths.RESULTS / protocol.NULL_THRESHOLDS_CSV
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    if not {"data", "target_set", "threshold"} <= set(table.columns):
        return {}
    rows = table.loc[table["target_set"] == target_set]
    return {str(d): float(t) for d, t in zip(rows["data"], rows["threshold"], strict=True)}


def _recorded_null(data: str, target_set: str) -> pd.Series:
    """The null_thresholds.csv row of one data source and target set (MissingInput if absent)."""
    path = paths.RESULTS / protocol.NULL_THRESHOLDS_CSV
    table = _read_table(path, NULL_RECORD_COLUMNS)
    rows = table.loc[(table["data"].astype(str) == data)
                     & (table["target_set"].astype(str) == target_set)]
    if rows.empty:
        raise MissingInput(f"{_shown(path)} has no {data} {target_set} row: the pooled null "
                           "cannot be matched to the threshold the summaries used")
    return rows.iloc[0]


def _check_pooled_null(
    pooled: np.ndarray, recorded: pd.Series, *, n_scored: int, source: str
) -> None:
    """The pooled null must be the one behind the recorded threshold (MissingInput otherwise)."""
    n_permutations, n_features = int(recorded["n_permutations"]), int(recorded["n_features"])
    stale = "the files come from different runs"
    if pooled.size != n_permutations * n_features:
        raise MissingInput(f"{source} holds {pooled.size:,} ρ, but null_thresholds.csv records "
                           f"{n_permutations} permutations × {n_features} metabolites: {stale}")
    if n_scored != n_features:
        raise MissingInput(f"the model was scored on {n_scored} metabolites, the recorded null "
                           f"on {n_features}: {stale}")
    quantile, threshold = float(recorded["quantile"]), float(recorded["threshold"])
    derived = float(np.quantile(pooled, quantile))
    if not math.isclose(derived, threshold, rel_tol=0.0, abs_tol=NULL_MATCH_TOLERANCE):
        raise MissingInput(f"the {quantile:.0%} quantile of {source} is {derived:.4f}, but "
                           f"null_thresholds.csv records {threshold:.4f}: {stale}")


def _null_key(data: str, thresholds: Mapping[str, float]) -> str | None:
    """The permutation null a data source is scored against: its own, else its base table's
    ("D1+MGM" uses D1's: same metabolites, same test samples)."""
    for key in (str(data), str(data).split("+")[0]):
        if key in thresholds:
            return key
    return None


def _rho_count_metric(cfg: Mapping[str, Any]) -> str:
    """Name of the N(ρ > threshold) statistic, as evaluate.summarize_rho spells it."""
    return next(k for k in evaluate.metric_keys(cfg) if k.startswith("n_rho_gt_"))


def _read_table(path: Path, required: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise MissingInput(f"{_shown(path)} missing")
    table = pd.read_csv(path)
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise MissingInput(f"{_shown(path)} lacks columns {missing}")
    if table.empty:
        raise MissingInput(f"{_shown(path)} has no rows")
    return table


def _model_columns(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, usecols=lambda c: c in {"model", "experiment"})


def _summary_path() -> Path:
    return paths.RESULTS / tracking.SUMMARY_CSV


def _curve_path() -> Path:
    return paths.RESULTS / common.LEARNING_CURVE_CSV


def _rho_by_class_path() -> Path:
    return paths.RESULTS / INTERPRETATION_DIR / RHO_BY_CLASS_CSV


def _importance_path() -> Path:
    return paths.RESULTS / INTERPRETATION_DIR / IMPORTANCE_CSV


def _ablations_path() -> Path:
    return paths.RESULTS / ABLATIONS_CSV


def _shown(path: Path) -> str:
    """A short path for messages: relative to the folder that holds results/ when possible."""
    for root in (paths.RESULTS.parent, paths.DATA_PROCESSED.parent):
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


# --------------------------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------------------------


@functools.cache
def _rc_params() -> dict[str, Any]:
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in PREFERRED_SANS if name in available] or ["DejaVu Sans"]
    return {
        "font.family": "sans-serif",
        "font.sans-serif": fonts,
        "font.size": TEXT_SIZE,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": HAIRLINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": TEXT_SIZE,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": HAIRLINE,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": LABEL_SIZE,
        "ytick.labelsize": LABEL_SIZE,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",
        "savefig.facecolor": SURFACE,
    }


def _finish(fig: Figure, title: str, note: str) -> Figure:
    """Left-aligned title on top; the note (what the marks mean) wrapped into a footer band."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=TITLE_SIZE, color=INK)
    width_pt = 0.97 * fig.get_figwidth() * POINTS_PER_INCH
    wrapped = textwrap.fill(note, width=int(width_pt / (NOTE_SIZE * CHAR_WIDTH_EM)))
    height = fig.get_figheight()
    lines = wrapped.count("\n") + 1
    footer = (lines * NOTE_SIZE * LINE_SPACING / POINTS_PER_INCH + 0.12) / height
    fig.get_layout_engine().set(rect=(0.0, footer, 1.0, 1.0 - footer))
    fig.text(0.01, 0.04 / height, wrapped, ha="left", va="bottom", fontsize=NOTE_SIZE,
             color=INK_SECONDARY, linespacing=LINE_SPACING)
    return fig


def _legend_above(ax: Axes, *, ncol: int) -> None:
    """Legend in the band between the title and the axes, where it can never hide data.

    (A figure-level "outside" legend ignores the title and footer bands; an axes legend
    anchored above the axes is part of the constrained layout.)
    """
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=ncol, fontsize=LABEL_SIZE,
              handlelength=2.6, borderaxespad=0.0, columnspacing=1.6)


def _laid_out(fig: Figure) -> Any:
    """Run the layout now (Agg canvas) so pixel positions are final; returns the renderer."""
    canvas = fig.canvas if isinstance(fig.canvas, FigureCanvasAgg) else FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


def _text_colour_on(fill: Any) -> str:
    """Ink or white, whichever contrasts more with a cell's fill (WCAG relative luminance)."""
    luminance = _relative_luminance(fill)
    on_white = (_relative_luminance(SURFACE) + 0.05) / (luminance + 0.05)
    on_ink = (luminance + 0.05) / (_relative_luminance(INK) + 0.05)
    return SURFACE if on_white > on_ink else INK


def _relative_luminance(colour: Any) -> float:
    rgb = np.asarray(to_rgb(colour))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ np.array([0.2126, 0.7152, 0.0722]))


def _nth(quantile: float) -> str:
    """0.95 -> "95th": the null quantile as the brief words it ("95th percentile")."""
    return f"{quantile * 100:g}th"


def _is_d2(data: Any) -> bool:
    return str(data).startswith(D2_PREFIX)


def _data_linestyle(data: str) -> Any:
    return DASHED if _is_d2(data) else SOLID


def _include_zero(ax: Axes) -> None:
    """Keep ρ = 0 (the mean predictor) on the y axis so distances from chance are readable."""
    low, high = ax.get_ylim()
    pad = 0.05 * (high - low)
    ax.set_ylim(min(low, -pad), max(high, pad))


def _line_gap(ax: Axes) -> float:
    """Height of one label line as a fraction of the (laid-out) axes height."""
    axes_height = ax.get_position().height * ax.figure.get_figheight()
    return LABEL_SIZE * LINE_SPACING / POINTS_PER_INCH / max(axes_height, 1e-6)


def _spread(wanted: Sequence[float], gap: float) -> list[float]:
    """Positions as close as possible to `wanted`, at least `gap` apart, none above 1."""
    order = np.argsort(wanted, kind="stable")
    placed = np.asarray(wanted, dtype=float)[order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    if len(placed) and placed[-1] > 1.0:
        placed[-1] = 1.0
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - gap)
    result = np.empty_like(placed)
    result[order] = placed
    return result.tolist()


def _ordered_present(values: Iterable[Any], preferred: Sequence[str]) -> list[str]:
    """Configured order first, then any other value present (sorted)."""
    present = {str(v) for v in values if pd.notna(v)}
    return [p for p in preferred if p in present] + sorted(present - set(preferred))


def _optional(row: pd.Series, key: str) -> float:
    value = row.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _count_text(value: float) -> str:
    if not np.isfinite(value):
        return "–"
    return f"{value:.0f}" if abs(value - round(value)) < 0.05 else f"{value:.1f}"
