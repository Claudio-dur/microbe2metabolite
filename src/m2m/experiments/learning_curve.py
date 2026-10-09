"""E5 learning curves (brief Q3): completeness check, mean ± SD table and the key figure.

Each model trains and scores its own curve points in its experiment's `curve` stage
(common.run_learning_curve on the subsets saved in results/splits/learning_curve_subsets.csv),
so every model sees the same PRISM people at each size. This module never trains, scores or
opens the external cohort; it only reads results/learning_curve.csv.

Stages, in order:
    check   every expected model has every (n_train, seed) point of the saved subsets; prints
            mean ± SD over seeds and writes results/learning_curve_summary.csv. Missing models
            or points are warnings, not errors, so the figure is still drawn from what exists.
            Rows at points outside the saved subsets (an older size or seed list) are named
            and left out of the summary and the figure: they were trained on other people.
    figure  results/figures/learning_curve.png via plots (same colours as `make figures`).

Nothing here chooses a model. "The E2 head" and "the best pretrained E4c variant" are what E2's
and E4's own PRISM cross-validation selected; the check recomputes that choice from the saved
selections (CV numbers only) to verify that the curve shows it, and never compares the lines.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m2m import paths, plots, protocol, tracking
from m2m.config import deep_merge, load_config
from m2m.experiments import common

STAGES = ("subsets", "check", "figure")
SUMMARY_CSV = "learning_curve_summary.csv"
ENTRY_KEYS = {"what", "model", "prefix", "exclude", "selections"}

Point = tuple[int, int]  # (n_train, seed)


class CurveCheckWarning(UserWarning):
    """The learning curve is incomplete or inconsistent; the figure is still drawn."""


@dataclass
class CurveCheck:
    """What the check found for the configured target set."""

    target_set: str
    expected_points: list[Point]
    matched: dict[str, list[str]]  # expected entry -> models found on the curve
    picked: dict[str, str] = field(default_factory=dict)  # entry -> model its CV selection picks
    missing_entries: list[str] = field(default_factory=list)
    missing_points: dict[str, list[Point]] = field(default_factory=dict)  # "model (data)"
    extra_points: dict[str, list[Point]] = field(default_factory=dict)
    unexpected_models: list[str] = field(default_factory=list)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def complete(self) -> bool:
        problems = (self.missing_entries, self.missing_points, self.extra_points,
                    self.unexpected_models)
        return not any(problems)


def run(cfg: Mapping[str, Any], *, stage: str | None = None, n_jobs: int = 1) -> None:
    """Run the E5 stages in order (or the requested ones)."""
    del n_jobs  # reading a CSV and drawing one figure: nothing to parallelise
    for name in common.stages_to_run(stage, STAGES):
        if name == "subsets":
            subsets(cfg)
        elif name == "check":
            check(cfg)
        else:
            figure(cfg)


def subsets(cfg: Mapping[str, Any]) -> None:
    """Create (or re-verify) the saved PRISM subsets every curve model trains on.

    Run during model selection so the subsets are fixed, and committed, before any model is
    scored on the external cohort; each experiment's curve stage then only reads them.
    """
    table = common.learning_curve_subsets(cfg)
    sizes = sorted(int(n) for n in table["n_train"].unique())
    print(f"[e5] learning-curve subsets: sizes {sizes} x {table['seed'].nunique()} seeds")


# --------------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------------


def check(cfg: Mapping[str, Any]) -> CurveCheck | None:
    """Verify the recorded curve, print mean ± SD per point and write the summary CSV."""
    common.dry_run()  # a dry run must refuse the committed results/ before writing the summary
    curve = _read_curve()
    if curve is None:
        return None
    target_set = str(cfg["learning_curve"]["target_set"])
    rows = curve.loc[curve["target_set"] == target_set]
    present = sorted(set(map(str, rows["model"])))
    entries = cfg["curve_check"]["expected"]
    picked = selection_picks(entries, target_set)
    matched = match_expected(entries, present, picked=picked)
    found = {m for models in matched.values() for m in models}
    result = CurveCheck(
        target_set=target_set,
        expected_points=sorted(expected_points(cfg)),
        matched=matched,
        picked=picked,
        missing_entries=[what for what, models in matched.items() if not models],
        unexpected_models=[m for m in present if m not in found],
    )
    _check_points(rows, result)
    order = [m for models in matched.values() for m in models] + result.unexpected_models
    # Same rows as the figure: stale points are reported above, never averaged in.
    result.summary = _ordered(plots.curve_summary(plots.on_saved_subsets(curve)), order)
    out = paths.RESULTS / SUMMARY_CSV
    result.summary.to_csv(out, index=False)
    _print_table(result)
    _warn_findings(result, entries)
    tracking.log_run(
        experiment="e5",
        run_name=f"curve check|{target_set}",
        params={"sizes": cfg["learning_curve"]["sizes"]},
        metrics={"n_models": len(found), "n_missing_entries": len(result.missing_entries),
                 "n_unexpected_models": len(result.unexpected_models)},
        tags={"model": "learning-curve", "target_set": target_set, "stage": "check"}, cfg=cfg,
    )
    print(f"[e5] wrote {out}")
    return result


def expected_points(cfg: Mapping[str, Any]) -> set[Point]:
    """(n_train, seed) pairs of the saved subsets: exactly what every model was meant to run.

    Falls back to the configured sizes × seeds when no subsets were saved yet.
    """
    saved = plots.saved_subset_points()
    if saved is not None:
        return saved
    path = paths.SPLITS / common.LEARNING_CURVE_SUBSETS_CSV
    print(f"[e5] {path} not found: expecting the configured sizes × seeds")
    sizes = [int(n) for n in cfg["learning_curve"]["sizes"]]
    return {(n, s) for n in sizes for s in common.model_seeds(cfg)}


def match_expected(
    entries: Sequence[Mapping[str, Any]],
    present: Iterable[str],
    *,
    picked: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Map each expected entry to the models on the curve that it names.

    `model` names one exact model; `prefix` names every model starting with it (minus
    `exclude`), for models whose final name their own experiment decides. An entry with a
    pick from its saved selections (`picked`, see `selection_picks`) expects that model only.
    """
    present = sorted(set(present))
    picked = dict(picked or {})
    matched: dict[str, list[str]] = {}
    for entry in entries:
        unknown = set(entry) - ENTRY_KEYS
        if unknown or ("model" in entry) == ("prefix" in entry):
            raise ValueError(
                f"curve_check entry {dict(entry)} needs exactly one of `model` or `prefix` "
                f"(allowed keys: {sorted(ENTRY_KEYS)})"
            )
        what = _entry_name(entry)
        if what in matched:
            raise ValueError(f"two curve_check entries are both called {what!r}")
        if what in picked:
            matched[what] = [m for m in present if m == picked[what]]
        elif "model" in entry:
            matched[what] = [m for m in present if m == str(entry["model"])]
        else:
            excluded = set(map(str, entry.get("exclude", ())))
            prefix = str(entry["prefix"])
            matched[what] = [m for m in present if m.startswith(prefix) and m not in excluded]
    return matched


def selection_picks(entries: Sequence[Mapping[str, Any]], target_set: str) -> dict[str, str]:
    """Entry -> the model its experiment put on the curve, recomputed from saved CV selections.

    An entry's `selections` maps candidate models to their saved selection names. The pick is
    the highest saved CV mean ρ, the first listed on ties (the rule E2 and E4 apply). Only
    selection files that exist count; an entry with none falls back to name matching. These
    are cross-validation numbers on PRISM: no external score is read.
    """
    picks: dict[str, str] = {}
    for entry in entries:
        cv: dict[str, float] = {}
        for model, name in dict(entry.get("selections") or {}).items():
            try:
                selection = protocol.load_selection(str(name), target_set)
            except FileNotFoundError:
                continue
            cv[str(model)] = float(selection["cv_mean_rho"])
        if cv:
            pick = max(cv, key=cv.__getitem__)  # max keeps the first maximum: listed order
            picks[_entry_name(entry)] = pick
            scores = ", ".join(f"{m} {v:.4f}" for m, v in cv.items())
            print(f"[e5] saved CV selections pick {pick} for {_entry_name(entry)} ({scores})")
    return picks


def _read_curve() -> pd.DataFrame | None:
    path = paths.RESULTS / common.LEARNING_CURVE_CSV
    if not path.exists():
        _warn(f"{path} not found: run the models' curve stages first (nothing to check)")
        return None
    curve = pd.read_csv(path)
    missing = [c for c in plots.CURVE_COLUMNS if c not in curve.columns]
    if missing:
        _warn(f"{path} lacks columns {missing}: cannot check the learning curve")
        return None
    return curve


def _check_points(rows: pd.DataFrame, result: CurveCheck) -> None:
    """Every model on the curve (expected or not) is drawn, so each must have every point."""
    expected = set(result.expected_points)
    for (model, data), group in rows.groupby(["model", "data"], sort=True):
        name = plots.series_label(str(model), str(data))
        have = set(zip(group["n_train"].astype(int), group["seed"].astype(int), strict=True))
        finite = np.isfinite(group["mean_rho"].to_numpy(dtype=float))
        if not finite.all():
            _warn(f"{name}: {int((~finite).sum())} point(s) have no finite mean ρ")
        if expected - have:
            result.missing_points[name] = sorted(expected - have)
        if have - expected:
            result.extra_points[name] = sorted(have - expected)


def _ordered(summary: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Curve summary sorted by target set, then the expected-entry order of the models."""
    rank = {m: i for i, m in enumerate(order)}
    key = summary["model"].map(lambda m: rank.get(m, len(rank)))
    return (
        summary.assign(_rank=key)
        .sort_values(["target_set", "_rank", "model", "data", "n_train"])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def _print_table(result: CurveCheck) -> None:
    """Mean ± SD over seeds per model (rows) and training size (columns)."""
    summary = result.summary.loc[result.summary["target_set"] == result.target_set]
    seeds = len({seed for _, seed in result.expected_points})
    print(f"[e5] learning curve, target set {result.target_set}: mean ρ ± SD over seeds "
          f"(* = fewer than {seeds} seeds)")
    if summary.empty:
        print("  (no rows)")
        return
    cells = summary.assign(
        series=[plots.series_label(m, d) for m, d in zip(summary["model"], summary["data"],
                                                          strict=True)],
        cell=[
            f"{mean:.3f} ± {sd:.3f}{'*' if n < seeds else ''}"
            for mean, sd, n in zip(summary["mean_rho_mean"], summary["mean_rho_sd"],
                                   summary["n_seeds"], strict=True)
        ],
    )
    order = list(dict.fromkeys(cells["series"]))
    table = cells.pivot(index="series", columns="n_train", values="cell").reindex(order)
    table.columns = [f"n={n}" for n in table.columns]
    table.index.name = None
    print(table.fillna("—").to_string())


def _warn_findings(result: CurveCheck, entries: Sequence[Mapping[str, Any]]) -> None:
    candidates = ", ".join(result.unexpected_models) or "none"
    for what in result.missing_entries:
        spec = _entry_spec(next(e for e in entries if _entry_name(e) == what))
        if what in result.picked:
            spec = f"the saved CV selections pick {result.picked[what]!r}"
        _warn(f"no learning-curve rows for {what} ({spec}); the figure is drawn without it. "
              f"Other models on the curve: {candidates}")
    for what, models in result.matched.items():
        if len(models) > 1:
            _warn(f"{len(models)} models match {what}: {models}; all of them are drawn")
    if result.unexpected_models:
        _warn(f"models on the curve that no expected entry names (stale rows, or a choice the "
              f"saved selections do not support?): {result.unexpected_models}; they are drawn too")
    for name, points in result.missing_points.items():
        _warn(f"{name} misses {len(points)} (n_train, seed) point(s): {_points_text(points)}")
    for name, points in result.extra_points.items():
        _warn(f"{name} has {len(points)} point(s) outside the saved subsets (stale rows from an "
              f"older configuration?): {_points_text(points)}; they are left out of the summary "
              "and the figure")
    if result.complete:
        print(f"[e5] complete: {len(result.expected_points)} (n_train, seed) points for every "
              "model on the curve")


def _entry_name(entry: Mapping[str, Any]) -> str:
    return str(entry.get("what") or entry.get("model") or entry.get("prefix"))


def _entry_spec(entry: Mapping[str, Any]) -> str:
    if "model" in entry:
        return f"model {entry['model']!r}"
    excluded = list(entry.get("exclude", ()))
    return f"prefix {entry['prefix']!r}" + (f" except {excluded}" if excluded else "")


def _points_text(points: Sequence[Point]) -> str:
    return ", ".join(f"n={n}/seed={s}" for n, s in points)


def _warn(message: str) -> None:
    warnings.warn(f"[e5] {message}", CurveCheckWarning, stacklevel=3)


# --------------------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------------------


def figure(cfg: Mapping[str, Any]) -> Path | None:
    """Draw learning_curve.png with the shared figure settings (colours match every figure)."""
    report = plots.draw_figures(figures_config(cfg), ["learning_curve"])
    plots.print_report(report)
    return report.drawn.get("learning_curve")


def figures_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The presentation config named by `figures_config`, with this experiment's curve settings.

    Looked up next to the E5 config (like `extends`), then in configs/, so `make curves` and
    `make figures` draw the same file with the same colours.
    """
    name = str(cfg["figures_config"])
    folders = [Path(cfg["config_path"]).parent] if "config_path" in cfg else []
    candidates = [folder / name for folder in [*folders, paths.CONFIGS]]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"figures_config {name!r} not found in {candidates}")
    return deep_merge(load_config(path), {"learning_curve": dict(cfg["learning_curve"])})
