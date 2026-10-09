"""Render the README's results tables from results/*.csv (so no number is copied by hand).

    .venv/bin/python scripts/report_tables.py            # writes results/tables.md and prints it
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from m2m import paths, plots
from m2m.config import load_config
from m2m.experiments import common

# Display order: baselines, from-scratch neural, foundation model next to its controls, D2.
MODEL_ORDER = [
    "mean", "ridge", "enet", "rf", "mlp",
    "mgm_frozen_ridge", "mgm_random_frozen_ridge", "mgm_frozen_mlp", "mgm_random_frozen_mlp",
    "mgm_finetune", "mgm_scratch",
    "d2_ridge", "e4c_scratch", "e4c_selfsup", "e4c_ihmp", "e4c_selfsup_ihmp",
]
TARGET_SETS = ["T466", "TALL", "S176"]


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def mean_sd(row: pd.Series, key: str, digits: int = 3) -> str:
    mean, sd, n_seeds = row.get(f"{key}_mean"), row.get(f"{key}_sd"), row.get("n_seeds", 1)
    if n_seeds and n_seeds > 1 and sd is not None and not math.isnan(sd):
        return f"{fmt(mean, digits)} ± {fmt(sd, digits)}"
    return fmt(mean, digits)


def cohort_label() -> str:
    return "PRISM hold-out (DRY RUN)" if common.dry_run() else "NLIBD"


def _ordered(rows: pd.DataFrame) -> pd.DataFrame:
    rank = rows["model"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(len(MODEL_ORDER))
    return rows.assign(order=rank).sort_values(["order", "model"])


def summary_table(summary: pd.DataFrame, target_set: str) -> str:
    rows = summary[summary["target_set"] == target_set]
    if rows.empty:
        return ""
    # Each data source has its own permutation null (D1 for D1 models, D2 for the E4 models).
    nulls = ", ".join(
        f"{data} ρ > {fmt(threshold)}"
        for data, threshold in rows.groupby("data")["null_threshold"].first().items()
    )
    lines = [
        f"**{target_set}** ({int(rows['n_features_mean'].iloc[0])} metabolites evaluated on "
        f"{int(rows['n_test'].iloc[0])} {cohort_label()} samples; permutation-null thresholds: "
        f"{nulls})",
        "",
        "| Model | Data | Mean ρ | 95% CI | Median ρ | Top-50 ρ | N(ρ > 0.5) | N(ρ > null) "
        "| NaN ρ | CV mean ρ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in _ordered(rows).iterrows():
        lines.append(
            f"| {r['model']} | {r['data']} | {mean_sd(r, 'mean_rho')} | "
            f"[{fmt(r['mean_rho_ci_low'])}, {fmt(r['mean_rho_ci_high'])}] | "
            f"{mean_sd(r, 'median_rho')} | {mean_sd(r, 'top50_mean_rho')} | "
            f"{mean_sd(r, 'n_rho_gt_0.5', 1)} | {mean_sd(r, 'n_above_null', 1)} | "
            f"{fmt(r['n_nan_mean'], 1)} | {fmt(r['cv_mean_rho'])} |"
        )
    return "\n".join(lines)


def curve_table(curve: pd.DataFrame, target_set: str) -> str:
    """Mean ± SD over seeds, using only rows at the saved subsets (as the E5 figure does)."""
    curve = plots.on_saved_subsets(curve)
    curve = curve[curve["target_set"] == target_set]
    if curve.empty:
        return ""
    stats = curve.groupby(["model", "data", "n_train"])["mean_rho"].agg(["mean", "std"])
    sizes = sorted(curve["n_train"].unique())
    lines = ["| Model | Data | " + " | ".join(f"n = {n}" for n in sizes) + " |",
             "|---|---|" + "---|" * len(sizes)]
    models = _ordered(curve[["model", "data"]].drop_duplicates())
    for _, (model, data) in models[["model", "data"]].iterrows():
        cells = []
        for n in sizes:
            if (model, data, n) in stats.index:
                s = stats.loc[(model, data, n)]
                sd = 0.0 if math.isnan(s["std"]) else s["std"]
                cells.append(f"{s['mean']:.3f} ± {sd:.3f}")
            else:
                cells.append("—")
        lines.append(f"| {model} | {data} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ablation_table(ablations: pd.DataFrame) -> str:
    lines = [
        "| Ablation | Arm | Model | Scored on | Eval space | CV mean ρ | External mean ρ "
        "| N(ρ > 0.5) | Note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in ablations.iterrows():
        copied = str(r.get("external_copied_from_main", "")).lower() == "true"
        note = "main configuration (copied)" if copied else ""
        external = (f"{fmt(r['mean_rho_mean'])} ± {fmt(r['mean_rho_sd'])}"
                    if r.get("n_seeds", 1) and r.get("n_seeds", 1) > 1
                    else fmt(r["mean_rho_mean"]))
        lines.append(
            f"| {r['ablation']} | {r['arm']} | {r['model']} | {r['target_set']} | "
            f"{r['eval_space']} | {fmt(r['cv_mean_rho'])} | {external} | "
            f"{fmt(r.get('n_rho_gt_0.5_mean'), 1)} | {note} |"
        )
    return "\n".join(lines)


def paired_table(paired: pd.DataFrame) -> str:
    lines = [
        "| Question | A | B | Targets | Δ mean ρ (A − B) | 95% CI | P(Δ ≤ 0) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in paired.iterrows():
        lines.append(
            f"| {r['question']} | {r['model_a']} | {r['model_b']} | {r['target_set']} | "
            f"{r['delta']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
            f"{r['p_not_better']:.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    common.dry_run()  # a dry run must not write tables.md into the committed results/
    cfg = load_config(paths.CONFIGS / "base.yaml")
    sections = []
    summary_path = paths.RESULTS / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        sections += [t for t in (summary_table(summary, ts) for ts in TARGET_SETS) if t]
    paired_path = paths.RESULTS / "paired_comparisons.csv"
    if paired_path.exists():
        sections.append(f"**Paired comparisons on {cohort_label()}** (paired bootstrap over test "
                        "samples)\n\n" + paired_table(pd.read_csv(paired_path)))
    curve_path = paths.RESULTS / "learning_curve.csv"
    target_set = str(cfg["learning_curve"]["target_set"])
    if curve_path.exists():
        table = curve_table(pd.read_csv(curve_path), target_set)
        if table:
            sections.append(f"**Learning curve** (mean ρ on {cohort_label()}, {target_set}; "
                            "mean ± SD over 5 seeds)\n\n" + table)
    ablation_path = paths.RESULTS / "ablations.csv"
    if ablation_path.exists():
        sections.append(f"**Ablations** (external = {cohort_label()})\n\n"
                        + ablation_table(pd.read_csv(ablation_path)))
    text = "\n\n".join(sections) + "\n"
    out = args.out or paths.RESULTS / "tables.md"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
