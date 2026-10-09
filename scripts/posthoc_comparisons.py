"""POST-HOC paired comparisons, added after the external results were seen (not pre-registered).

The pre-registered pairs are in configs/compare.yaml. The external results raised two questions
the protocol did not plan for, so they are tested here and reported as exploratory:

1. Frozen MGM + ridge scored above plain ridge on NLIBD although its PRISM CV was lower.
2. The D2 (GTDB genus) models scored above every D1 (MetaPhlAn2 species) model.

Same paired bootstrap as the pre-registered comparisons, on saved predictions only: nothing is
refitted and the external cohort is not opened. D1 and D2 name the same NLIBD people differently,
so rows are aligned by matching their ground-truth vectors, which are identical: same metabolome
measurements, standardised with the same PRISM statistics.

    .venv/bin/python scripts/posthoc_comparisons.py      # writes results/posthoc_comparisons.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from m2m import evaluate, paths, protocol
from m2m.config import load_config
from m2m.experiments import common

PAIRS = [  # (A, B, target set, question)
    ("mgm_frozen_ridge", "ridge", "T466", "Frozen MGM + ridge vs ridge on the D1 species"),
    ("mgm_frozen_ridge", "ridge", "TALL", "Frozen MGM + ridge vs ridge on the D1 species"),
    ("d2_ridge", "ridge", "T466", "Same ridge, D2 GTDB genera vs D1 MetaPhlAn2 species"),
    ("e4c_scratch", "mlp", "T466", "Same MLP from scratch, D2 GTDB genera vs D1 species"),
]
TRUTH_TOL = 1e-9


def aligned(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> np.ndarray:
    """Row order of `b` that puts each of `a`'s samples on the same person (by identical truth)."""
    if not np.array_equal(a["features"], b["features"]):
        raise ValueError("the two models were scored on different metabolites")
    distance = np.abs(a["truth"][:, None, :] - b["truth"][None, :, :]).max(axis=2)
    order = distance.argmin(axis=1)
    if len(set(order)) != len(order) or distance[np.arange(len(order)), order].max() > TRUTH_TOL:
        raise ValueError("could not match every sample one-to-one by its ground truth")
    return order


def main() -> None:
    common.dry_run()  # a dry run must not write into the committed results/
    ev = load_config(paths.CONFIGS / "base.yaml")["evaluation"]
    rows = []
    for model_a, model_b, target_set, question in PAIRS:
        a = protocol.load_external_predictions(model_a, target_set)
        b = protocol.load_external_predictions(model_b, target_set)
        order = aligned(a, b)
        result = evaluate.paired_bootstrap_delta(
            a["truth"], a["predictions"], b["predictions"][:, order, :],
            n_boot=int(ev["n_bootstrap"]), seed=int(ev["bootstrap_seed"]),
            ci_level=float(ev["ci_level"]), method=str(ev["ci_method"]),
        )
        rows.append({"model_a": model_a, "model_b": model_b, "target_set": target_set,
                     "question": question, "analysis": "post-hoc (not pre-registered)", **result})
        print(f"{model_a} - {model_b} ({target_set}): delta {result['delta']:+.4f} "
              f"[{result['ci_low']:+.4f}, {result['ci_high']:+.4f}], "
              f"P(delta <= 0) {result['p_not_better']:.3f}")
    out = paths.RESULTS / "posthoc_comparisons.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
