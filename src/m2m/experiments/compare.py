"""Paired comparisons on the external cohort: does model A beat model B? (H1 and context).

Reads the predictions every main model saved when it was scored (protocol.record_external) and
runs `evaluate.paired_bootstrap_delta` on the same resampled test samples for both models. No
cohort is opened and nothing is refitted, so the brief's "one external evaluation per final
configuration" still holds.

    python -m m2m.run configs/compare.yaml
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from m2m import evaluate, paths, protocol, tracking
from m2m.experiments import common

EXPERIMENT = "compare"
STAGES = ("paired",)
OUTPUT_CSV = "paired_comparisons.csv"
ROW_KEY = ("model_a", "model_b", "target_set")


def run(cfg: Mapping[str, Any], *, stage: str | None = None, n_jobs: int = 1) -> None:
    del n_jobs  # a few bootstraps on saved predictions: nothing to parallelise
    common.dry_run()  # refuses to write into the committed results/ during a dry run
    for _ in common.stages_to_run(stage, STAGES):
        paired(cfg)


def paired(cfg: Mapping[str, Any]) -> None:
    """One row per (A, B, target set) in results/paired_comparisons.csv."""
    ev = cfg["evaluation"]
    for comparison in cfg["comparisons"]:
        for target_set in comparison["target_sets"]:
            row = compare_pair(comparison["a"], comparison["b"], target_set, ev)
            if row is None:
                continue
            row["question"] = comparison["question"]
            tracking.upsert_csv_row(
                paths.RESULTS / OUTPUT_CSV, {**row, "timestamp": tracking.utc_now()}, ROW_KEY
            )
            tracking.log_run(
                experiment=EXPERIMENT,
                run_name=f"{row['model_a']} vs {row['model_b']}|{target_set}",
                params={"n_bootstrap": ev["n_bootstrap"], "ci_method": ev["ci_method"]},
                metrics={k: row[k] for k in ("delta", "ci_low", "ci_high", "p_not_better")},
                tags={"model": f"{row['model_a']}-vs-{row['model_b']}",
                      "target_set": target_set, "stage": "paired"},
                cfg=cfg,
            )
            print(f"[{EXPERIMENT}] {row['model_a']} − {row['model_b']} ({target_set}): "
                  f"Δ mean ρ {row['delta']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}], "
                  f"P(Δ ≤ 0) {row['p_not_better']:.3f}")


def compare_pair(
    model_a: str, model_b: str, target_set: str, ev: Mapping[str, Any]
) -> dict[str, Any] | None:
    """The paired bootstrap for one pair, or None (with a note) if either side is missing."""
    try:
        a = protocol.load_external_predictions(model_a, target_set)
        b = protocol.load_external_predictions(model_b, target_set)
    except FileNotFoundError as missing:
        print(f"[{EXPERIMENT}] skipped {model_a} vs {model_b} ({target_set}): {missing}")
        return None
    _require_same_truth(a, b, f"{model_a} vs {model_b} ({target_set})")
    result = evaluate.paired_bootstrap_delta(
        a["truth"], a["predictions"], b["predictions"],
        n_boot=int(ev["n_bootstrap"]), seed=int(ev["bootstrap_seed"]),
        ci_level=float(ev["ci_level"]), method=str(ev["ci_method"]),
    )
    return {
        "model_a": model_a,
        "model_b": model_b,
        "target_set": target_set,
        **result,
        "n_features": int(a["truth"].shape[1]),
        "n_test": int(a["truth"].shape[0]),
        "n_seeds_a": int(len(a["seeds"])),
        "n_seeds_b": int(len(b["seeds"])),
    }


def _require_same_truth(
    a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], what: str
) -> None:
    """A paired test needs the same samples, metabolites and ground truth on both sides."""
    same = (
        np.array_equal(a["sample_ids"], b["sample_ids"])
        and np.array_equal(a["features"], b["features"])
        and np.allclose(a["truth"], b["truth"], rtol=0.0, atol=1e-12)
    )
    if not same:
        raise ValueError(f"{what}: the two models were not scored on identical targets")
