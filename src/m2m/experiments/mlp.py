"""E1: the MLP trained from scratch on D1, hyperparameters selected by PRISM cross-validation.

Stages, run in this order by ``python -m m2m.run configs/e1_mlp.yaml [--stage S]``:

- ``select``: every point of the brief's grid (``mlp.grid`` in base.yaml: lr × dropout × weight
  decay) by 5-fold PRISM CV at one model seed (``cv_seed``), separately per target set (DESIGN.md
  decision #7) → runs.csv rows and results/selected/e1_mlp__<target_set>.json.
- ``external``: the selected configuration refit on all of PRISM once per model seed (early
  stopping on an inner split of PRISM) and scored once on NLIBD → summary.csv row ``mlp``.
- ``curve``: the selected configuration of the learning-curve target set on the saved PRISM
  subsets → results/learning_curve.csv.

Architecture and training settings live in base.yaml's ``mlp`` section, shared with the
ablations and E5, so every "MLP" in the project is the same model.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from m2m import paths, protocol
from m2m.experiments import common

EXPERIMENT = "e1"
MODEL = "mlp"
SELECTION = "e1_mlp"
DATA = "D1"
STAGES = ("select", "external", "curve")


def run(cfg: dict[str, Any], *, stage: str | None = None, n_jobs: int = 1) -> None:
    """Run the requested stage, or all of them in order."""
    actions = {"select": stage_select, "external": stage_external, "curve": stage_curve}
    for name in common.stages_to_run(stage, STAGES):
        start = time.perf_counter()
        actions[name](cfg, n_jobs=n_jobs)
        _say(f"stage {name} done in {time.perf_counter() - start:.1f}s")


def mlp_builder(cfg: Mapping[str, Any], target_set: str) -> protocol.PipelineBuilder:
    return functools.partial(common.build_mlp, cfg=cfg, target_set=target_set)


def load_checked_selection(cfg: Mapping[str, Any], target_set: str) -> dict[str, Any]:
    """The saved E1 choice for a target set, refused unless the current grid is what was searched.

    A selection left over from an older grid would otherwise be evaluated (and reused by the
    ablations) as if the current grid had chosen it. Membership alone is not enough: after the
    grid grows (e.g. dropout 0.3 added, DESIGN.md decision #15) the old winner is still a grid
    point, but the new points were never cross-validated.
    """
    selection = protocol.load_selection(SELECTION, target_set)
    grid = protocol.expand_grid(cfg["mlp"]["grid"])
    if selection["params"] not in grid:
        raise ValueError(
            f"{SELECTION} ({target_set}) selected {selection['params']}, which is not in the "
            "current mlp.grid: rerun the E1 select stage"
        )
    if searched_grid(target_set) != _grid_keys(grid):
        raise ValueError(
            f"{SELECTION} ({target_set}) was selected from a different grid than the current "
            "mlp.grid: rerun the E1 select stage"
        )
    return selection


def searched_grid(target_set: str) -> list[str]:
    """The grid points of the saved E1 CV table (written by protocol.save_selection)."""
    table = pd.read_csv(paths.RESULTS / "selected" / f"{SELECTION}__{target_set}__cv.csv")
    return _grid_keys(json.loads(params) for params in table["params"])


def _grid_keys(grid: Iterable[Mapping[str, Any]]) -> list[str]:
    """Order-free, formatting-free comparison key of a list of grid points."""
    return sorted(json.dumps(dict(params), sort_keys=True) for params in grid)


def stage_select(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Grid search by PRISM CV, one selection per target set."""
    X, Y, labels = common.load_d1_train()
    folds = common.d1_folds(labels, cfg)
    grid = protocol.expand_grid(cfg["mlp"]["grid"])
    for target_set in common.target_sets(cfg):
        start = time.perf_counter()
        table, _ = protocol.grid_search_cv(
            mlp_builder(cfg, target_set),
            grid,
            X,
            Y,
            labels,
            folds,
            cfg,
            seed=int(cfg["cv_seed"]),
            n_jobs=n_jobs,
        )
        protocol.record_cv_table(
            table, experiment=EXPERIMENT, model=MODEL, target_set=target_set, data=DATA,
            cfg=cfg, n_train=len(X),
        )
        best = protocol.select_best(table)
        protocol.save_selection(SELECTION, target_set, best, table)
        scores = table.set_index("params")["cv_mean_rho"]
        _say(
            f"select {target_set}: best {best} with CV mean ρ {scores.max():.4f} "
            f"(grid range {scores.min():.4f}-{scores.max():.4f}, {len(grid)} configs) "
            f"in {time.perf_counter() - start:.1f}s"
        )


def stage_external(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Refit the selected MLP on all of PRISM per model seed; score NLIBD once per target set."""
    X, Y, labels = common.load_d1_train()
    target_sets = common.target_sets(cfg)
    # Every configuration and threshold is fixed on disk before the cohort is first opened, so a
    # missing TALL record cannot surface only after T466 was already scored on NLIBD.
    selections = {t: load_checked_selection(cfg, t) for t in target_sets}
    thresholds = {t: protocol.load_null_threshold(DATA, t) for t in target_sets}
    for target_set in target_sets:
        selection, threshold = selections[target_set], thresholds[target_set]
        X_test, Y_test = common.load_d1_test(
            purpose=(
                f"E1 final evaluation: selected MLP {selection['params']} refit on PRISM, "
                f"scored on NLIBD ({target_set})"
            )
        )
        result = protocol.evaluate_external(
            mlp_builder(cfg, target_set),
            selection["params"],
            X,
            Y,
            labels,
            X_test,
            Y_test,
            cfg,
            seeds=common.model_seeds(cfg),
            null_threshold=threshold,
            n_jobs=n_jobs,
        )
        protocol.record_external(
            result, experiment=EXPERIMENT, model=MODEL, target_set=target_set, data=DATA,
            params=selection["params"], cfg=cfg, n_train=len(X),
            cv_mean_rho=selection["cv_mean_rho"],
        )
        agg = result.aggregate
        _say(
            f"external {target_set}: mean ρ {agg['mean_rho_mean']:.4f} ± {agg['mean_rho_sd']:.4f} "
            f"over {agg['n_seeds']} seeds, CI [{agg['mean_rho_ci_low']:.4f}, "
            f"{agg['mean_rho_ci_high']:.4f}]"
        )


def stage_curve(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """The selected MLP on every saved learning-curve subset, scored on NLIBD."""
    target_set = str(cfg["learning_curve"]["target_set"])
    selection = load_checked_selection(cfg, target_set)
    threshold = protocol.load_null_threshold(DATA, target_set)
    X, Y, labels = common.load_d1_train()
    X_test, Y_test = common.load_d1_test(
        purpose=(
            f"E1 learning curve: selected MLP refit on PRISM subsets, scored on NLIBD "
            f"({target_set})"
        )
    )
    points = common.run_learning_curve(
        mlp_builder(cfg, target_set),
        selection["params"],
        X,
        Y,
        labels,
        X_test,
        Y_test,
        cfg,
        model=MODEL,
        data_name=DATA,
        experiment=EXPERIMENT,
        null_threshold=threshold,
        n_jobs=n_jobs,
    )
    by_size = points.groupby("n_train")["mean_rho"].mean()
    _say("curve mlp: " + ", ".join(f"n={n}: {rho:.4f}" for n, rho in by_size.items()))


def _say(message: str) -> None:
    print(f"[{EXPERIMENT}] {message}", flush=True)
