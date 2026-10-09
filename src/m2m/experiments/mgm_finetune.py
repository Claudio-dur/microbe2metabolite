"""E3: MGM fine-tuned, and the same network trained from scratch (the H1 control).

Both models are MGM's GPT-2 architecture + a pooled linear head on the same MGM tokens; only the
initial weights differ (brief H1: "pretrained models beat the same architecture trained from
scratch"):
  mgm_finetune  init "pretrained": MGM's weights; the last k blocks are fine-tuned (grid: k and
                the backbone learning rate; the head has its own, larger rate);
  mgm_scratch   init "random": transformers' GPT-2 initialisation of MGM's config.json, seeded
                by the model seed; every block and embedding trains with one learning rate for
                backbone and head (its own grid: 1e-5..3e-5 only nudges pretrained weights).
Everything else is shared: pooling, PRISM folds and inner early-stopping splits, target
preprocessing, fit-row centring, early stopping, the 5 final seeds and the learning-curve subsets.

Two environments share the work (docs/mgm_notes.md):
- this module (main env) fits every preprocessing step, writes task files, launches the MGM
  interpreter and does all scoring and bookkeeping through m2m.protocol;
- scripts/mgm_finetune.py (envs/mgm: torch 2.0.1, transformers 4.33) only trains and predicts.

They exchange files under data/processed/mgm/e3/<round>/, round = "cv" or "final":
  tasks/<task_id>.npz + .json   token ids, standardised targets, `init` and hyperparameters
                                (written here)
  preds/<task_id>.npz + .json   test predictions and training history (written in envs/mgm)
  preps/<name>.json             the fitted MetabolitePreprocessor of a final or curve task (shared
                                by both models): the external truth is transformed with it, in
                                this environment
  manifest.json                 the round's task list and provenance; written last, so a round
                                without it is incomplete

Stages (python -m m2m.run configs/e3_mgm_finetune.yaml [--stage S]):
  prepare-cv     one task per (model, grid point, saved PRISM fold, target set), model seed 0;
                 each target set's pooling is taken from E2's CV selection for that target set
                 (frozen MGM embeddings + ridge or MLP head)
  run-cv         train every CV task in envs/mgm (resumable: finished tasks are skipped)
  select         fold rho from the CV predictions -> CV table -> best grid point per model and
                 target set, saved under results/selected/ (PRISM only)
  prepare-final  from the saved selections: per model, 5 seeds x all of PRISM per target set,
                 plus the T466 learning-curve points (saved subsets); the test inputs are the
                 external cohort's MGM tokens (final evaluation: the read is logged)
  run-final      train the final round in envs/mgm
  evaluate       score the final predictions on the external cohort; record the learning curves

Leakage guards:
- targets are standardised by a MetabolitePreprocessor fit on the task's training rows only (the
  early-stopping rows included, DESIGN.md decision 2) and checked with a fit-ids tripwire;
- training, early-stopping and test ids must be disjoint (checked when writing a task, again
  in envs/mgm before training, and again on the preprocessor before scoring);
- only CV tasks carry test targets (PRISM fold truth); external metabolites never enter a file;
- predictions carry the content hash of the task they answer, so stale ones are never scored;
- the selection stages (prepare-cv, run-cv, select) never read the external cohort, not even
  its tokens. prepare-final reads its MGM *tokens* (inputs written by E2's embed stage, no
  metabolites) only after checking that a selection from the current CV round is saved for
  every model and target set; the read is logged in results/external_access_log.csv;
- external metabolites are opened once, in `evaluate`, after every other input has been checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m2m import data, evaluate, freeze, mgm_io, paths, protocol, splits, tracking
from m2m.config import config_hash
from m2m.experiments import common
from m2m.models import POOLING_MODES
from m2m.preprocess import MetabolitePreprocessor
from m2m.train import TrainConfig

# The initial weights of the two models (envs/mgm builds the network from this task field).
PRETRAINED, RANDOM = "pretrained", "random"
INITS = (PRETRAINED, RANDOM)
MODELS = {PRETRAINED: "mgm_finetune", RANDOM: "mgm_scratch"}
# summary.csv `data`: D1's samples and genera as MGM tokens, with MGM's weights or its
# architecture alone.
DATA_NAMES = {PRETRAINED: "D1+MGM", RANDOM: "D1+MGM-arch"}
SELECTION_NAMES = {PRETRAINED: "e3_mgm_finetune", RANDOM: "e3_mgm_scratch"}
TASK_TAGS = {PRETRAINED: "finetune", RANDOM: "scratch"}  # part of every task id
STAGES = ("prepare-cv", "run-cv", "select", "prepare-final", "run-final", "evaluate")
CV_ROUND, FINAL_ROUND = "cv", "final"
KINDS = ("cv", "final", "curve")
SPLIT_NAMES = ("train", "val", "test")
# The model seed of every CV task: E1's cv_seed (configs/e1_mlp.yaml), so E1 and E3 early-stop on
# the same inner split of each fold.
CV_SEED = 0
# Layout of the exchange files; scripts/mgm_finetune.py refuses any other version. Version 3
# added `init` (which weights a task's network starts from).
FORMAT_VERSION = 3
# The e3.train settings envs/mgm receives (lr and weight decay come from the grid and e3); any
# other key is refused rather than validated here and then silently ignored there.
TRAIN_KEYS = ("batch_size", "max_epochs", "patience", "min_delta", "min_batches_per_epoch")
POOLING_FROM_KEYS = ("selections",)
# What each model's CV grid may vary, and what e3.scratch may set. `hyperparameters` reads
# nothing else, so any other key (e.g. a scratch-only weight_decay) is refused, not ignored.
GRID_KEYS = {PRETRAINED: ("n_trainable_blocks", "backbone_lr"), RANDOM: ("lr",)}
SCRATCH_KEYS = ("grid", "n_trainable_blocks")
SCRIPT = paths.REPO_ROOT / "scripts" / "mgm_finetune.py"
TASKS_DIR, PREDS_DIR, PREPS_DIR = "tasks", "preds", "preps"
MANIFEST = "manifest.json"
# Config sections that change how a run is executed but not what it computes.
RUNTIME_KEYS = ("config_path", "mgm", "n_jobs", "mlflow")

TEST_TOKENS_PURPOSE = (
    "E3 mgm_finetune/mgm_scratch prepare-final: read the external cohort's MGM token ids (inputs "
    "only, no metabolites) to write the final-round tasks of the configurations saved by PRISM CV"
)
EXTERNAL_PURPOSE = (
    "E3 evaluate: score the CV-selected configurations of mgm_finetune and of its from-scratch "
    "control mgm_scratch (5 seeds per target set) and their T466 learning curves"
)
RUN_FINAL_PURPOSE = (
    "E3 run-final: envs/mgm trains the selected configurations on PRISM and predicts the "
    "external cohort from its MGM token ids (inputs only)"
)


class StalePredictionsError(RuntimeError):
    """Predictions on disk were computed from another version of their task file."""


# --- stages ----------------------------------------------------------------------------------


def run(cfg: dict, *, stage: str | None = None, n_jobs: int = 1) -> None:
    """Run the E3 stages in order (or only `stage`); n_jobs caps the MGM worker processes."""
    actions: dict[str, Callable[[], Any]] = {
        "prepare-cv": lambda: prepare_cv(cfg),
        "run-cv": lambda: run_mgm_round(cfg, CV_ROUND, n_jobs),
        "select": lambda: select(cfg),
        "prepare-final": lambda: prepare_final(cfg),
        "run-final": lambda: run_mgm_round(cfg, FINAL_ROUND, n_jobs),
        "evaluate": lambda: evaluate_final(cfg),
    }
    for name in common.stages_to_run(stage, STAGES):
        print(f"[e3] stage {name}", flush=True)
        actions[name]()


def prepare_cv(cfg: Mapping[str, Any]) -> Path:
    """One task per (model, grid point, PRISM fold, target set), all with model seed CV_SEED."""
    _, Y, labels = common.load_d1_train()
    ids = [str(i) for i in labels.index]
    tokens = load_prism_tokens(cfg, ids)
    target_sets = common.target_sets(cfg)
    poolings = {t: e2_pooling(cfg, t) for t in target_sets}
    grids = {
        init: {t: model_grid(cfg, init, poolings[t][0]) for t in target_sets} for init in INITS
    }
    for init, by_target in grids.items():
        for grid in by_target.values():
            for params in grid:
                # A config typo fails here, before any file is written.
                hyperparameters(params, cfg, init=init)
    folds = common.d1_folds(labels, cfg)
    fold_ids = sorted(int(f) for f in folds["fold"].unique())
    tasks = []
    for target_set in target_sets:
        for fold in fold_ids:
            train_ids, test_ids = splits.fold_train_test(folds, fold)
            # One target preprocessing per fold: both models learn and are scored on the same
            # standardised metabolites.
            prep = fit_targets(cfg, target_set, Y.loc[train_ids])
            fold_data = {
                "targets": prep.transform(Y.loc[train_ids]),
                "test_ids": test_ids,
                "test_targets": prep.transform(Y.loc[test_ids]),
                "extra": {"fold": fold, "n_targets_dropped_zero": int(prep.n_dropped_zero_)},
            }
            for init in INITS:
                for params in grids[init][target_set]:
                    tasks.append(
                        make_task(
                            task_id=cv_task_id(init, target_set, params, fold),
                            kind="cv",
                            target_set=target_set,
                            init=init,
                            seed=CV_SEED,
                            params=params,
                            cfg=cfg,
                            labels=labels,
                            tokens=tokens,
                            test_tokens=tokens,
                            **fold_data,
                        )
                    )
    round_dir = round_path(CV_ROUND)
    write_round(
        round_dir,
        tasks,
        {
            "round": CV_ROUND,
            "grids": grids,
            "pooling": {t: pooling for t, (pooling, _) in poolings.items()},
            "pooling_source": {t: source for t, (_, source) in poolings.items()},
            "target_sets": target_sets,
            "folds": fold_ids,
            "n_train_cohort": len(ids),
        },
        cfg,
    )
    return round_dir


def select(cfg: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Score the CV predictions and save the best grid point per model and target set.

    Nothing here reads the external cohort, not even its MGM tokens, so this selection stage
    can run on real data like any other: the final round is written by `prepare_final`.
    """
    round_dir = round_path(CV_ROUND)
    manifest = read_manifest(round_dir, CV_ROUND, cfg)
    _require_current_pooling(cfg, manifest)
    _require_predictions(round_dir, manifest, stage="run-cv")
    selections: dict[str, dict[str, dict[str, Any]]] = {init: {} for init in INITS}
    for init in INITS:
        for target_set in manifest["target_sets"]:
            table = cv_table(cfg, round_dir, manifest, init, target_set)
            protocol.record_cv_table(
                table,
                experiment=cfg["experiment"],
                model=MODELS[init],
                target_set=target_set,
                data=DATA_NAMES[init],
                cfg=cfg,
                n_train=int(manifest["n_train_cohort"]),
                extra_tags=mgm_io.mgm_env_tags(),
            )
            best = protocol.select_best(table)
            protocol.save_selection(SELECTION_NAMES[init], target_set, best, table)
            selection = protocol.load_selection(SELECTION_NAMES[init], target_set)
            selections[init][target_set] = selection
            print(
                f"[e3] {MODELS[init]} {target_set}: selected {best} "
                f"(CV mean rho {selection['cv_mean_rho']:.4f})",
                flush=True,
            )
    return selections


def prepare_final(cfg: Mapping[str, Any]) -> Path:
    """Final round: per model and target set, 5 seeds trained on all PRISM rows; plus each
    model's curve points.

    A final-evaluation stage: every task predicts the final test samples (the external cohort's
    MGM tokens, or PRISM's held-out pseudo-test rows in a dry run). The tokens are read last,
    once the saved selections, the curve subsets and PRISM's own inputs have all been checked.
    Training-side preprocessing is fit and saved here so that `evaluate` transforms the external
    truth with exactly the fitted object the targets came from; both models share it.
    """
    selections = saved_selections(cfg)
    target_sets = list(selections[PRETRAINED])
    curve_target = str(cfg["learning_curve"]["target_set"])
    if curve_target not in target_sets:
        raise ValueError(f"learning_curve.target_set {curve_target} was not cross-validated")
    _, Y, labels = common.load_d1_train()
    ids = [str(i) for i in labels.index]
    tokens = load_prism_tokens(cfg, ids)
    subsets = common.learning_curve_subsets(cfg, ids, list(labels.loc[ids]))
    test_tokens, test_ids, test_cohort = load_test_tokens(cfg)
    splits.assert_disjoint(ids, test_ids, what="PRISM training vs final test samples")
    round_dir = round_path(FINAL_ROUND)
    # The old round is invalid from here on (its preps go next); write_round adds the new manifest.
    (round_dir / MANIFEST).unlink(missing_ok=True)
    _remove_files(round_dir / PREPS_DIR, "*.json")  # every prep below is fit afresh
    shared = {
        "cfg": cfg,
        "labels": labels,
        "tokens": tokens,
        "test_tokens": test_tokens,
        "test_ids": test_ids,
    }
    tasks = []
    for target_set in target_sets:
        # One preprocessing per target set: every seed of both models trains on the same rows.
        name = f"final__{target_set}"
        prep, extra = fit_and_save_targets(cfg, target_set, Y.loc[ids], round_dir, name)
        targets = prep.transform(Y.loc[ids])
        for init in INITS:
            for seed in common.model_seeds(cfg):
                tasks.append(
                    make_task(
                        task_id=f"final__{target_set}__{TASK_TAGS[init]}__seed{seed}",
                        kind="final",
                        target_set=target_set,
                        init=init,
                        seed=seed,
                        params=selections[init][target_set]["params"],
                        targets=targets,
                        extra=extra,
                        **shared,
                    )
                )
    for n, seed in sorted(set(zip(subsets["n_train"], subsets["seed"], strict=True))):
        # Every curve point refits the target filter on its own subset (DESIGN decision 20).
        subset = [str(i) for i in common.subset_ids(subsets, n, seed)]
        name = f"curve__{curve_target}__n{n}__seed{seed}"
        prep, extra = fit_and_save_targets(cfg, curve_target, Y.loc[subset], round_dir, name)
        targets = prep.transform(Y.loc[subset])
        for init in INITS:
            tasks.append(
                make_task(
                    task_id=f"curve__{curve_target}__{TASK_TAGS[init]}__n{n}__seed{seed}",
                    kind="curve",
                    target_set=curve_target,
                    init=init,
                    seed=int(seed),  # the subset's seed is also the model seed, as in E1/E5
                    params=selections[init][curve_target]["params"],
                    targets=targets,
                    extra=extra,
                    **shared,
                )
            )
    write_round(
        round_dir,
        tasks,
        {
            "round": FINAL_ROUND,
            "test_cohort": test_cohort,
            "test_sample_ids": test_ids,
            "selections": {
                init: {
                    t: {"params": s["params"], "cv_mean_rho": s["cv_mean_rho"]}
                    for t, s in by_target.items()
                }
                for init, by_target in selections.items()
            },
            "n_train_cohort": len(ids),
        },
        cfg,
    )
    return round_dir


def run_mgm_round(cfg: Mapping[str, Any], round_name: str, n_jobs: int) -> None:
    """Train every pending task of a round with the MGM interpreter; fail if any is missing."""
    round_dir = round_path(round_name)
    manifest = read_manifest(round_dir, round_name, cfg)
    if round_name == FINAL_ROUND:
        # A select re-run after prepare-final would otherwise train a configuration no longer
        # selected (evaluate refuses it too, but only after the compute is spent).
        for init, by_target in manifest["selections"].items():
            for target_set in by_target:
                _current_selection(manifest, init, target_set)
        if not common.dry_run():
            # envs/mgm predicts the external cohort's token ids: gated and logged like any
            # opening, so a resumed run-final leaves its own trace.
            freeze.require_frozen(RUN_FINAL_PURPOSE)
            tracking.log_external_access(
                purpose=RUN_FINAL_PURPOSE, detail="final-round task files (token ids only)"
            )
    command = mgm_command(cfg, round_dir, n_jobs)
    print("[e3] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=paths.REPO_ROOT, check=False)
    missing = missing_predictions(round_dir, manifest["tasks"])
    if completed.returncode != 0 or missing:
        raise RuntimeError(
            f"{SCRIPT.name} exited with code {completed.returncode}; {len(missing)} of "
            f"{len(manifest['tasks'])} {round_name} tasks have no up-to-date predictions "
            f"(e.g. {missing[:3]}). Re-running the stage keeps the finished tasks."
        )


def evaluate_final(cfg: Mapping[str, Any]) -> None:
    """Score the final round on the external cohort (opened once, after every other check)."""
    round_dir = round_path(FINAL_ROUND)
    manifest = read_manifest(round_dir, FINAL_ROUND, cfg)
    _require_predictions(round_dir, manifest, stage="run-final")
    tasks = [read_task(round_dir, task_id) for task_id in manifest["tasks"]]
    predictions = {task.task_id: read_predictions(round_dir, task) for task in tasks}
    preps = {task.task_id: load_prep(round_dir, task) for task in tasks}
    test_ids = _shared_test_ids(tasks, manifest)
    selections = {
        init: {t: _current_selection(manifest, init, t) for t in by_target}
        for init, by_target in manifest["selections"].items()
    }
    null_data = str(cfg["e3"]["null_data"])
    target_sets = sorted({t for by_target in selections.values() for t in by_target})
    null = {t: protocol.load_null_threshold(null_data, t) for t in target_sets}

    _, Y_test = common.load_d1_test(purpose=EXTERNAL_PURPOSE)
    if set(map(str, Y_test.index)) != set(test_ids):
        raise ValueError(
            "the final round predicted other samples than the external cohort holds: "
            "re-run prepare-final (and run-final) with the current data"
        )
    Y_test = Y_test.set_axis([str(i) for i in Y_test.index], axis=0).loc[test_ids]
    for init, by_target in selections.items():
        for target_set, selection in by_target.items():
            final = [t for t in tasks if _is_final_of(t, init, target_set)]
            _record_final(
                cfg, init, target_set, final, predictions, preps, Y_test, selection,
                null[target_set],
            )
        curve = [t for t in tasks if t.spec["kind"] == "curve" and t.spec["init"] == init]
        _record_curve(cfg, init, curve, predictions, preps, Y_test, by_target, null)


# --- task construction -----------------------------------------------------------------------


@dataclass(frozen=True)
class Tokens:
    """MGM token ids of one cohort, one row per sample (mgm_embed.py's <cohort>_tokens.npz)."""

    sample_ids: tuple[str, ...]
    input_ids: np.ndarray
    attention_mask: np.ndarray

    @classmethod
    def load(cls, path: Path) -> Tokens:
        with np.load(path, allow_pickle=False) as npz:
            tokens = cls(
                tuple(str(s) for s in npz["sample_ids"]), npz["input_ids"], npz["attention_mask"]
            )
        n = len(tokens.sample_ids)
        if len(set(tokens.sample_ids)) != n:
            raise ValueError(f"{path} lists some samples more than once")
        if tokens.input_ids.shape != tokens.attention_mask.shape or tokens.input_ids.shape[0] != n:
            raise ValueError(f"{path}: input_ids, attention_mask and sample_ids disagree in shape")
        return tokens

    def take(self, ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(input_ids, attention_mask) of `ids`, in that order; KeyError for an unknown id."""
        position = {sample: row for row, sample in enumerate(self.sample_ids)}
        missing = [i for i in ids if i not in position]
        if missing:
            raise KeyError(f"no MGM tokens for {len(missing)} samples, e.g. {missing[:3]}")
        rows = [position[i] for i in ids]
        return self.input_ids[rows], self.attention_mask[rows]


@dataclass(frozen=True)
class Task:
    """One fit for envs/mgm: `spec` is the JSON half, `arrays` the npz half."""

    spec: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def task_id(self) -> str:
        return str(self.spec["task_id"])

    def ids(self, split: str) -> list[str]:
        return [str(i) for i in self.arrays[f"{split}_sample_ids"]]


def make_task(
    *,
    task_id: str,
    kind: str,
    target_set: str,
    init: str,
    seed: int,
    params: Mapping[str, Any],
    cfg: Mapping[str, Any],
    targets: pd.DataFrame,
    labels: pd.Series,
    tokens: Tokens,
    test_tokens: Tokens,
    test_ids: Sequence[str],
    test_targets: pd.DataFrame | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Task:
    """Assemble one task from standardised training targets (rows = all training samples).

    The training rows are split into fit / early-stopping rows exactly as protocol.M2MPipeline
    would split them for this model seed, so E1 and both E3 models stop early on the same samples.
    """
    fit_ids, val_ids = inner_split(list(targets.index), labels, seed, cfg)
    arrays: dict[str, np.ndarray] = {}
    for split, split_ids, source in (
        ("train", fit_ids, tokens),
        ("val", val_ids, tokens),
        ("test", [str(i) for i in test_ids], test_tokens),
    ):
        input_ids, attention_mask = source.take(split_ids)
        arrays[f"{split}_input_ids"] = input_ids
        arrays[f"{split}_attention_mask"] = attention_mask
        arrays[f"{split}_sample_ids"] = np.array(split_ids, dtype=str)
    # float32 is what training uses anyway; the fold truth stays float64 for exact ranks.
    arrays["Y_train"] = targets.loc[fit_ids].to_numpy(dtype=np.float32)
    arrays["Y_val"] = targets.loc[val_ids].to_numpy(dtype=np.float32)
    if test_targets is not None:
        arrays["Y_test"] = test_targets.loc[list(test_ids), targets.columns].to_numpy(np.float64)
    spec = {
        "task_id": task_id,
        "kind": kind,
        "target_set": target_set,
        "seed": int(seed),
        "n_train": len(targets),
        "params": dict(params),
        **hyperparameters(params, cfg, init=init),
        "targets": [str(c) for c in targets.columns],
        **dict(extra or {}),
    }
    return Task(spec, arrays)


def model_grid(cfg: Mapping[str, Any], init: str, pooling: str) -> list[dict[str, Any]]:
    """CV grid of one model at one pooling: (k, backbone lr) for MGM, the lr for the scratch
    network."""
    e3 = cfg["e3"]
    if init == PRETRAINED:
        where, grid = "e3.grid", e3["grid"]
    elif init == RANDOM:
        _refuse_unused_keys(e3["scratch"], SCRATCH_KEYS, "e3.scratch")
        where, grid = "e3.scratch.grid", e3["scratch"]["grid"]
    else:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")
    _refuse_unused_keys(grid, GRID_KEYS[init], where)
    return protocol.expand_grid({**grid, "pooling": [pooling]})


def _refuse_unused_keys(section: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(
            f"{where} keys {unknown} would be ignored (allowed: {list(allowed)}): weight decay, "
            "head dropout and training are set once under e3 for both models; pooling is E2's"
        )


def hyperparameters(
    params: Mapping[str, Any], cfg: Mapping[str, Any], *, init: str
) -> dict[str, Any]:
    """Everything envs/mgm needs to train one task: the initial weights, the grid point plus
    E3's fixed settings.

    Validated here (with TrainConfig's checks) so a config typo fails before any file is written,
    not hours later in the other environment.
    """
    e3 = cfg["e3"]
    train = dict(e3["train"])
    unknown = sorted(set(train) - set(TRAIN_KEYS))
    if unknown:
        raise ValueError(f"e3.train keys {unknown} are not passed to envs/mgm; use {TRAIN_KEYS}")
    if init == PRETRAINED:
        k, backbone_lr, head_lr = params["n_trainable_blocks"], params["backbone_lr"], e3["head_lr"]
    elif init == RANDOM:
        # Nothing in a random network is pretrained, so backbone and head share one rate.
        k, backbone_lr, head_lr = e3["scratch"]["n_trainable_blocks"], params["lr"], params["lr"]
    else:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")
    for lr in (backbone_lr, head_lr):
        checked = TrainConfig.from_dict({**train, "lr": lr, "weight_decay": e3["weight_decay"]})
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f"n_trainable_blocks must be an integer >= 0, got {k!r}")
    if params["pooling"] not in POOLING_MODES:
        raise ValueError(f"pooling must be one of {POOLING_MODES}, got {params['pooling']!r}")
    if not 0.0 <= float(e3["head_dropout"]) < 1.0:
        raise ValueError(f"head_dropout must be in [0, 1), got {e3['head_dropout']!r}")
    return {
        "init": init,
        "n_trainable_blocks": k,
        "backbone_lr": float(backbone_lr),
        "head_lr": float(head_lr),
        "weight_decay": float(e3["weight_decay"]),
        "head_dropout": float(e3["head_dropout"]),
        "pooling": str(params["pooling"]),
        # From the validated TrainConfig, so a setting left out of e3.train takes the same
        # default in both environments and is still written (and hashed) explicitly.
        **{key: getattr(checked, key) for key in TRAIN_KEYS},
    }


def inner_split(
    train_ids: Sequence[str], labels: pd.Series, seed: int, cfg: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    """(fit, early-stopping) ids: the split protocol.M2MPipeline carves for this model seed."""
    inner = cfg["inner_validation"]
    ids = [str(i) for i in train_ids]
    return splits.stratified_inner_split(
        ids,
        list(labels.loc[ids]),
        fraction=float(inner["fraction"]),
        seed=int(seed) + int(inner["seed_offset"]),
    )


def fit_targets(
    cfg: Mapping[str, Any], target_set: str, Y_train: pd.DataFrame
) -> MetabolitePreprocessor:
    """The target preprocessing of one training set, with the protocol's fit-ids tripwire."""
    prep = common.metabolite_prep(cfg, target_set).fit(Y_train)
    if [str(i) for i in prep.fit_samples_] != [str(i) for i in Y_train.index]:
        raise splits.LeakageError("the target preprocessor was fit on rows other than training")
    return prep


def fit_and_save_targets(
    cfg: Mapping[str, Any], target_set: str, Y_train: pd.DataFrame, round_dir: Path, name: str
) -> tuple[MetabolitePreprocessor, dict[str, Any]]:
    """Fit, save under preps/ and return the spec entries that point evaluate to the file."""
    prep = fit_targets(cfg, target_set, Y_train)
    path = prep.save(round_dir / PREPS_DIR / f"{name}.json")
    return prep, {
        "prep_file": f"{PREPS_DIR}/{path.name}",
        "prep_hash": _file_hash(path),
        "n_targets_dropped_zero": int(prep.n_dropped_zero_),
    }


def cv_task_id(init: str, target_set: str, params: Mapping[str, Any], fold: int) -> str:
    if init == PRETRAINED:
        point = f"k{params['n_trainable_blocks']}__blr{params['backbone_lr']:g}"
    else:
        point = f"lr{params['lr']:g}"
    return f"cv__{target_set}__{TASK_TAGS[init]}__{point}__fold{fold}__seed{CV_SEED}"


# --- inputs ----------------------------------------------------------------------------------


def mgm_dir() -> Path:
    return paths.DATA_PROCESSED / "mgm"


def round_path(round_name: str) -> Path:
    return mgm_dir() / "e3" / round_name


def load_prism_tokens(cfg: Mapping[str, Any], train_ids: Sequence[str]) -> Tokens:
    """PRISM's MGM tokens (E2's prepare stage); they must cover exactly the PRISM samples."""
    path = mgm_dir() / cfg["e3"]["tokens"]["train"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: E2 writes the PRISM MGM tokens (configs/e2_mgm_frozen.yaml)"
        )
    tokens = Tokens.load(path)
    expected = _all_prism_ids(train_ids)
    if set(tokens.sample_ids) != set(expected):
        raise ValueError(
            f"{path} holds {len(tokens.sample_ids)} samples, not exactly the {len(expected)} "
            "PRISM samples: regenerate it with E2"
        )
    return tokens


def load_test_tokens(cfg: Mapping[str, Any]) -> tuple[Tokens, list[str], str]:
    """(tokens, test ids, cohort label) of the final evaluation's test samples."""
    if common.dry_run():
        _, test_ids = common.dry_run_ids()
        tokens = Tokens.load(mgm_dir() / cfg["e3"]["tokens"]["train"])
        return tokens, [str(i) for i in test_ids], "PRISM pseudo-test rows (dry run)"
    path = mgm_dir() / cfg["e3"]["tokens"]["test"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: E2's embedding stage writes the external cohort's MGM tokens "
            "(configs/e2_mgm_frozen.yaml): run it, then re-run prepare-final"
        )
    freeze.require_frozen(TEST_TOKENS_PURPOSE)
    tracking.log_external_access(purpose=TEST_TOKENS_PURPOSE, detail=f"{path.name} (token ids)")
    tokens = Tokens.load(path)
    return tokens, list(tokens.sample_ids), "NLIBD"


def e2_pooling(cfg: Mapping[str, Any], target_set: str) -> tuple[str, dict[str, Any]]:
    """Pooling of whichever saved E2 selection for `target_set` has the higher PRISM CV mean
    rho (ties: first listed). Per target set, as every hyperparameter is (DESIGN decision 7)."""
    spec = cfg["e3"]["pooling_from"]
    unknown = sorted(set(spec) - set(POOLING_FROM_KEYS))
    if unknown:
        # e.g. the former `target_set` key: silently ignoring it would hide a changed rule.
        raise ValueError(
            f"e3.pooling_from keys {unknown} are not used (allowed: {list(POOLING_FROM_KEYS)}): "
            "each target set takes the pooling E2 selected for that target set"
        )
    found = {}
    for name in spec["selections"]:
        try:
            found[name] = protocol.load_selection(name, target_set)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"E3 takes its pooling from E2's CV selection, but {error}: run "
                "`python -m m2m.run configs/e2_mgm_frozen.yaml` (through select) first"
            ) from None
    winner = max(found, key=lambda name: float(found[name]["cv_mean_rho"]))
    pooling = found[winner]["params"].get("pooling")
    if pooling not in POOLING_MODES:
        raise ValueError(
            f"E2 selection {winner} ({target_set}) has params {found[winner]['params']}: "
            f"expected a 'pooling' entry in {POOLING_MODES}"
        )
    source = {name: float(sel["cv_mean_rho"]) for name, sel in found.items()}
    return str(pooling), {"chosen_from": winner, "target_set": target_set, "cv_mean_rho": source}


def saved_selections(cfg: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """{init: {target set: saved selection}}, each a point of that model's current CV grid.

    A selection left over from an older grid or pooling would otherwise be trained and scored as
    if the current cross-validation had chosen it.
    """
    manifest = read_manifest(round_path(CV_ROUND), CV_ROUND, cfg)
    _require_current_pooling(cfg, manifest)
    selections: dict[str, dict[str, dict[str, Any]]] = {}
    for init in INITS:
        selections[init] = {}
        for target_set in manifest["target_sets"]:
            grid = {_params_key(params) for params in manifest["grids"][init][target_set]}
            selection = protocol.load_selection(SELECTION_NAMES[init], target_set)
            if _params_key(selection["params"]) not in grid:
                raise ValueError(
                    f"{SELECTION_NAMES[init]} ({target_set}) holds {selection['params']}, which "
                    "is not a point of the current CV round: re-run the select stage"
                )
            selections[init][target_set] = selection
    return selections


def mgm_command(cfg: Mapping[str, Any], round_dir: Path, n_jobs: int) -> list[str]:
    """The envs/mgm command line for one round (workers capped by n_jobs and the thread budget)."""
    settings = cfg["mgm"]
    python = Path(settings["python"])
    # Absolute paths everywhere: the interpreter runs from the repository root, so a relative
    # M2M_DATA_PROCESSED would otherwise name another round directory there. The interpreter is
    # made absolute but not resolved: a venv's python is a symlink to the base interpreter, and
    # following it would leave the venv (and its packages) behind.
    python = (python if python.is_absolute() else paths.REPO_ROOT / python).absolute()
    if not python.exists():
        raise FileNotFoundError(f"MGM interpreter {python} not found: run `make mgm-env`")
    threads = int(settings["threads"])
    # Also capped by this machine's cores: the script refuses workers x threads > os.cpu_count().
    workers = max(1, min(int(settings["workers"]), int(n_jobs), (os.cpu_count() or 1) // threads))
    if workers * threads > int(settings["max_total_threads"]):
        raise ValueError(
            f"{workers} workers x {threads} threads exceeds mgm.max_total_threads "
            f"({settings['max_total_threads']})"
        )
    e3 = cfg["e3"]
    command = [
        str(python),
        "-I",  # isolated: nothing from the working directory or PYTHON* variables is imported
        str(SCRIPT.resolve()),
        "--round-dir", str(round_dir.resolve()),
        "--workers", str(workers),
        "--threads", str(threads),
        "--model-dir", str((data.d3_dir() / data.MGM_MODEL_DIR).resolve()),
        "--cache-tol", repr(float(e3["prefix_cache_tol"])),
    ]
    if not bool(e3["prefix_cache"]):
        command.append("--no-prefix-cache")
    return command


# --- reading results back --------------------------------------------------------------------


def cv_table(
    cfg: Mapping[str, Any],
    round_dir: Path,
    manifest: Mapping[str, Any],
    init: str,
    target_set: str,
) -> pd.DataFrame:
    """protocol.grid_search_cv's table (params, cv_*, fold<k>_mean_rho) of one model and target
    set, from MGM's predictions."""
    grid = manifest["grids"][init][target_set]
    grid_index = {_params_key(params): i for i, params in enumerate(grid)}
    scores = []
    for task_id in manifest["tasks"]:
        spec = _read_json(_task_paths(round_dir, task_id)[0])  # cheap: arrays only when needed
        if spec["target_set"] == target_set and spec["init"] == init:
            task = read_task(round_dir, task_id)
            index = grid_index[_params_key(task.spec["params"])]
            scores.append(cv_fold_score(cfg, round_dir, task, index))
    expected = {(i, fold) for i in range(len(grid)) for fold in manifest["folds"]}
    if {(s.params_index, s.fold) for s in scores} != expected:
        raise ValueError(
            f"the CV round does not cover every grid point x fold of {MODELS[init]} for "
            f"{target_set}"
        )
    # protocol builds the table from FoldScores, exactly as for in-process models.
    return protocol.cv_table(grid, scores, sorted(manifest["folds"]))


def cv_fold_score(
    cfg: Mapping[str, Any], round_dir: Path, task: Task, params_index: int
) -> protocol.FoldScore:
    """Per-metabolite rho of one CV task on its PRISM test fold, as protocol summarises it."""
    truth = pd.DataFrame(
        task.arrays["Y_test"], index=task.ids("test"), columns=task.spec["targets"]
    )
    pred = read_predictions(round_dir, task)
    rho = pd.Series(evaluate.spearman_per_feature(truth, pred), index=truth.columns)
    summary = {
        **evaluate.summarize_from_cfg(rho.to_numpy(), cfg),
        "n_targets_dropped_zero": float(task.spec["n_targets_dropped_zero"]),
    }
    return protocol.FoldScore(
        params_index=params_index, fold=int(task.spec["fold"]), summary=summary, rho=rho
    )


def read_predictions(round_dir: Path, task: Task) -> pd.DataFrame:
    """A task's test predictions, samples x targets, rows in the task's test order (by id)."""
    meta_path, npz_path = _prediction_paths(round_dir, task.task_id)
    if not (meta_path.exists() and npz_path.exists()):
        raise FileNotFoundError(f"no predictions for {task.task_id} in {round_dir / PREDS_DIR}")
    if _read_json(meta_path).get("task_hash") != task.spec["task_hash"]:
        raise StalePredictionsError(
            f"predictions for {task.task_id} answer an older version of the task: re-run it"
        )
    with np.load(npz_path, allow_pickle=False) as npz:
        values = np.asarray(npz["predictions"], dtype=np.float64)
        predicted_ids = [str(i) for i in npz["test_sample_ids"]]
    expected = task.ids("test")
    if len(set(predicted_ids)) != len(predicted_ids) or set(predicted_ids) != set(expected):
        raise ValueError(f"{task.task_id}: the predicted samples are not the task's test samples")
    if values.shape != (len(expected), len(task.spec["targets"])):
        raise ValueError(f"{task.task_id}: predictions have shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{task.task_id}: predictions contain NaN or infinite values")
    frame = pd.DataFrame(values, index=predicted_ids, columns=task.spec["targets"])
    return frame.loc[expected]


def load_prep(round_dir: Path, task: Task) -> MetabolitePreprocessor:
    """The saved target preprocessing of a final/curve task, checked against the task."""
    path = round_dir / task.spec["prep_file"]
    if _file_hash(path) != task.spec["prep_hash"]:
        raise ValueError(f"{path} changed after {task.task_id} was written: re-run prepare-final")
    prep = MetabolitePreprocessor.load(path)
    trained_on = set(task.ids("train")) | set(task.ids("val"))
    if set(map(str, prep.fit_samples_)) != trained_on:
        raise splits.LeakageError(f"{path} was not fit on exactly {task.task_id}'s training rows")
    if [str(f) for f in prep.features_] != task.spec["targets"]:
        raise ValueError(f"{path} and {task.task_id} disagree on the targets")
    return prep


def missing_predictions(round_dir: Path, task_ids: Sequence[str]) -> list[str]:
    """Tasks without predictions computed from their current task file."""
    return [t for t in task_ids if not _has_current_predictions(round_dir, t)]


# --- the exchange files ----------------------------------------------------------------------


def check_task(task: Task) -> None:
    """Refuse a task that could leak or that envs/mgm would misread."""
    spec, arrays, task_id = task.spec, task.arrays, task.task_id
    kind = spec["kind"]
    if kind not in KINDS:
        raise ValueError(f"{task_id}: kind must be one of {KINDS}, got {kind!r}")
    if spec.get("init") not in INITS:
        raise ValueError(f"{task_id}: init must be one of {INITS}, got {spec.get('init')!r}")
    if kind != "cv" and "Y_test" in arrays:
        raise ValueError(
            f"{task_id}: only CV tasks carry test targets (PRISM fold truth); the test samples "
            f"of a {kind} task are the external cohort, whose targets never leave evaluate"
        )
    if kind == "cv" and "Y_test" not in arrays:
        raise ValueError(f"{task_id}: a CV task needs its fold truth Y_test")
    ids = {split: task.ids(split) for split in SPLIT_NAMES}
    splits.assert_disjoint(ids["train"], ids["val"], what=f"{task_id} training vs early stopping")
    splits.assert_disjoint(
        ids["train"] + ids["val"], ids["test"], what=f"{task_id} training vs test"
    )
    for split in SPLIT_NAMES:
        shape = arrays[f"{split}_input_ids"].shape
        if len(shape) != 2 or shape[0] != len(ids[split]) or shape[0] == 0:
            raise ValueError(f"{task_id}: {split}_input_ids has shape {shape}")
        if arrays[f"{split}_attention_mask"].shape != shape:
            raise ValueError(f"{task_id}: {split} token ids and attention mask differ in shape")
    for split, key in (("train", "Y_train"), ("val", "Y_val"), ("test", "Y_test")):
        if key not in arrays:
            continue
        if arrays[key].shape != (len(ids[split]), len(spec["targets"])):
            raise ValueError(f"{task_id}: {key} has shape {arrays[key].shape}")
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{task_id}: {key} contains NaN or infinite values")


def write_task(round_dir: Path, task: Task) -> bool:
    """Write one task (npz + json); False if an identical task is already on disk.

    A task that changed gets a new content hash and loses its old predictions, so envs/mgm
    re-runs it and stale predictions can never be scored.
    """
    check_task(task)
    spec = {**task.spec, "format_version": FORMAT_VERSION}
    spec["task_hash"] = content_hash(spec, task.arrays)
    json_path, npz_path = _task_paths(round_dir, task.task_id)
    if json_path.exists() and npz_path.exists():
        if _read_json(json_path).get("task_hash") == spec["task_hash"]:
            return False
    for path in _prediction_paths(round_dir, task.task_id):
        path.unlink(missing_ok=True)
    _atomic_savez(npz_path, task.arrays)
    _atomic_write_text(json_path, json.dumps(spec, indent=1, sort_keys=True) + "\n")
    return True


def read_task(round_dir: Path, task_id: str) -> Task:
    json_path, npz_path = _task_paths(round_dir, task_id)
    spec = _read_json(json_path)
    with np.load(npz_path, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    return Task(spec, arrays)


def write_round(
    round_dir: Path, tasks: Sequence[Task], manifest: Mapping[str, Any], cfg: Mapping[str, Any]
) -> None:
    """Write a round's tasks, drop files of tasks no longer in it, then the manifest (last)."""
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("duplicate task ids in one round")
    (round_dir / MANIFEST).unlink(missing_ok=True)  # incomplete until the new manifest lands
    changed = sum(write_task(round_dir, task) for task in tasks)
    for folder in (TASKS_DIR, PREDS_DIR):
        for path in (round_dir / folder).glob("*"):
            if path.stem not in task_ids:
                path.unlink()
    payload = {
        **manifest,
        "format_version": FORMAT_VERSION,
        "experiment": cfg["experiment"],
        "dry_run": common.dry_run(),
        "settings_hash": settings_hash(cfg),
        "created": tracking.utc_now(),
        "tasks": task_ids,
    }
    _atomic_write_text(round_dir / MANIFEST, json.dumps(payload, indent=1) + "\n")
    print(
        f"[e3] {round_dir.name} round: {len(tasks)} tasks in {round_dir} "
        f"({changed} new or changed, {len(tasks) - changed} unchanged)",
        flush=True,
    )


def read_manifest(round_dir: Path, round_name: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """A complete round written under the current settings and dry-run mode."""
    path = round_dir / MANIFEST
    writer = "prepare-cv" if round_name == CV_ROUND else "prepare-final"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the {writer} stage first")
    manifest = _read_json(path)
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("round") != round_name:
        raise ValueError(f"{path} is not an E3 {round_name} round of format {FORMAT_VERSION}")
    if manifest["dry_run"] != common.dry_run():
        raise RuntimeError(
            f"the {round_name} round was written with M2M_DRY_RUN "
            f"{'on' if manifest['dry_run'] else 'off'}; re-run {writer} in the current mode"
        )
    if manifest["settings_hash"] != settings_hash(cfg):
        raise RuntimeError(
            f"the config changed since the {round_name} round was written: re-run {writer} "
            "(tasks whose content is unchanged keep their predictions)"
        )
    return manifest


def settings_hash(cfg: Mapping[str, Any]) -> str:
    """Hash of every setting that changes what is computed (not how many processes run it)."""
    return config_hash({k: v for k, v in cfg.items() if k not in RUNTIME_KEYS})


def content_hash(spec: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    """Stable hash of a task's JSON half and every array (name, dtype, shape and bytes)."""
    digest = hashlib.sha256(json.dumps(dict(spec), sort_keys=True).encode())
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key])
        digest.update(f"|{key}|{values.dtype.str}|{values.shape}|".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()[:16]


# --- helpers ---------------------------------------------------------------------------------


def _record_final(
    cfg: Mapping[str, Any],
    init: str,
    target_set: str,
    tasks: Sequence[Task],
    predictions: Mapping[str, pd.DataFrame],
    preps: Mapping[str, MetabolitePreprocessor],
    Y_test: pd.DataFrame,
    selection: Mapping[str, Any],
    null_threshold: float,
) -> None:
    """summary.csv row (mean +- SD over seeds, bootstrap CI) for one model and target set."""
    tasks = sorted(tasks, key=lambda task: task.spec["seed"])
    if not tasks or len({task.spec["prep_file"] for task in tasks}) != 1:
        raise ValueError(
            f"the final {MODELS[init]} {target_set} seeds must share one target preprocessing"
        )
    prep = preps[tasks[0].task_id]
    truth = prep.transform(Y_test)
    result = protocol.score_predictions(
        truth,
        [predictions[task.task_id] for task in tasks],
        cfg,
        seeds=[int(task.spec["seed"]) for task in tasks],
        null_threshold=null_threshold,
    )
    result.aggregate["n_targets_dropped_zero"] = float(prep.n_dropped_zero_)
    protocol.record_external(
        result,
        experiment=cfg["experiment"],
        model=MODELS[init],
        target_set=target_set,
        data=DATA_NAMES[init],
        params=selection["params"],
        cfg=cfg,
        n_train=int(tasks[0].spec["n_train"]),
        cv_mean_rho=float(selection["cv_mean_rho"]),
        extra_tags=mgm_io.mgm_env_tags(),
    )
    print(
        f"[e3] {MODELS[init]} {target_set}: external mean rho "
        f"{result.aggregate['mean_rho_mean']:.4f} +- {result.aggregate['mean_rho_sd']:.4f} over "
        f"{len(tasks)} seeds",
        flush=True,
    )


def _record_curve(
    cfg: Mapping[str, Any],
    init: str,
    tasks: Sequence[Task],
    predictions: Mapping[str, pd.DataFrame],
    preps: Mapping[str, MetabolitePreprocessor],
    Y_test: pd.DataFrame,
    selections: Mapping[str, Mapping[str, Any]],
    null: Mapping[str, float],
) -> None:
    """One learning-curve point per (n, seed) of one model; targets are filtered per subset
    (decision 20)."""
    for task in sorted(tasks, key=lambda t: (t.spec["n_train"], t.spec["seed"])):
        target_set = task.spec["target_set"]
        truth = preps[task.task_id].transform(Y_test)
        pred = predictions[task.task_id].loc[truth.index, truth.columns]
        rho = evaluate.spearman_per_feature(truth, pred)
        summary = {
            **evaluate.summarize_from_cfg(rho, cfg, null[target_set]),
            "n_targets_dropped_zero": float(preps[task.task_id].n_dropped_zero_),
        }
        common.record_curve_point(
            summary,
            model=MODELS[init],
            data_name=DATA_NAMES[init],
            target_set=target_set,
            n_train=int(task.spec["n_train"]),
            seed=int(task.spec["seed"]),
            params=selections[target_set]["params"],
            experiment=cfg["experiment"],
            cfg=cfg,
        )


def _is_final_of(task: Task, init: str, target_set: str) -> bool:
    spec = task.spec
    return spec["kind"] == "final" and spec["init"] == init and spec["target_set"] == target_set


def _current_selection(
    manifest: Mapping[str, Any], init: str, target_set: str
) -> dict[str, Any]:
    """The saved selection, which must still be the one the final round was trained with."""
    selection = protocol.load_selection(SELECTION_NAMES[init], target_set)
    trained_with = manifest["selections"][init][target_set]["params"]
    if _params_key(selection["params"]) != _params_key(trained_with):
        raise RuntimeError(
            f"the {MODELS[init]} {target_set} selection changed after the final round was "
            "written: re-run prepare-final"
        )
    return selection


def _require_current_pooling(cfg: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """The CV round must use the pooling that E2's saved selections choose now, per target set.

    The settings hash cannot see E2's results: after an E2 re-run that changed its choice,
    selecting from this round would silently keep the old pooling.
    """
    for target_set in manifest["target_sets"]:
        pooling, _ = e2_pooling(cfg, target_set)
        if pooling != manifest["pooling"][target_set]:
            raise RuntimeError(
                f"E2's saved {target_set} selections now choose pooling {pooling!r}, but the CV "
                f"round was trained with {manifest['pooling'][target_set]!r}: re-run prepare-cv "
                "and run-cv"
            )


def _shared_test_ids(tasks: Sequence[Task], manifest: Mapping[str, Any]) -> list[str]:
    expected = [str(i) for i in manifest["test_sample_ids"]]
    for task in tasks:
        if task.ids("test") != expected:
            raise ValueError(f"{task.task_id} predicts other samples than the final round's")
    return expected


def _require_predictions(round_dir: Path, manifest: Mapping[str, Any], *, stage: str) -> None:
    missing = missing_predictions(round_dir, manifest["tasks"])
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(manifest['tasks'])} tasks in {round_dir} have no up-to-date "
            f"predictions (e.g. {missing[:3]}): run the {stage} stage"
        )


def _has_current_predictions(round_dir: Path, task_id: str) -> bool:
    meta_path, npz_path = _prediction_paths(round_dir, task_id)
    json_path, _ = _task_paths(round_dir, task_id)
    if not (meta_path.exists() and npz_path.exists() and json_path.exists()):
        return False
    return _read_json(meta_path).get("task_hash") == _read_json(json_path).get("task_hash")


def _params_key(params: Mapping[str, Any]) -> str:
    return json.dumps(dict(params), sort_keys=True)


def _all_prism_ids(train_ids: Sequence[str]) -> list[str]:
    """Every PRISM id: the training rows, plus the held-out pseudo-test rows in a dry run."""
    if common.dry_run():
        pseudo_train, pseudo_test = common.dry_run_ids()
        return [str(i) for i in [*pseudo_train, *pseudo_test]]
    return [str(i) for i in train_ids]


def _task_paths(round_dir: Path, task_id: str) -> tuple[Path, Path]:
    folder = round_dir / TASKS_DIR
    return folder / f"{task_id}.json", folder / f"{task_id}.npz"


def _prediction_paths(round_dir: Path, task_id: str) -> tuple[Path, Path]:
    folder = round_dir / PREDS_DIR
    return folder / f"{task_id}.json", folder / f"{task_id}.npz"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _remove_files(folder: Path, pattern: str) -> None:
    for path in folder.glob(pattern):
        path.unlink()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file + rename, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as fh:
        fh.write(text)
    os.replace(fh.name, path)


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(fh.name, path)
