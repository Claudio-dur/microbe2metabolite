"""Command-line entry point.

    python -m m2m.run configs/<experiment>.yaml [--stage S] [--n-jobs N]

The config's ``experiment`` key picks the module; each experiment module exposes
``run(cfg, *, stage, n_jobs)`` and documents its stages.
"""

from __future__ import annotations

import argparse
import importlib
import time
from collections.abc import Sequence

from m2m.config import load_config

EXPERIMENT_MODULES = {
    "e0": "m2m.experiments.baselines",
    "e1": "m2m.experiments.mlp",
    "e2": "m2m.experiments.mgm_frozen",
    "e3": "m2m.experiments.mgm_finetune",
    "e4": "m2m.experiments.transfer",
    "e5": "m2m.experiments.learning_curve",
    "e6": "m2m.experiments.interpret",
    "ablations": "m2m.experiments.ablations",
    "compare": "m2m.experiments.compare",
    "figures": "m2m.plots",
}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to an experiment YAML config")
    parser.add_argument(
        "--stage", default=None, help="stage(s) to run, comma-separated (default: all, in order)"
    )
    parser.add_argument("--n-jobs", type=int, default=None, help="override cfg['n_jobs']")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    experiment = cfg.get("experiment")
    if experiment not in EXPERIMENT_MODULES:
        raise SystemExit(
            f"unknown experiment {experiment!r}; expected one of {sorted(EXPERIMENT_MODULES)}"
        )
    module = importlib.import_module(EXPERIMENT_MODULES[experiment])
    n_jobs = args.n_jobs if args.n_jobs is not None else int(cfg.get("n_jobs", 1))

    start = time.perf_counter()
    module.run(cfg, stage=args.stage, n_jobs=n_jobs)
    elapsed = time.perf_counter() - start
    print(f"[m2m.run] {experiment} stage={args.stage or 'all'} done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
