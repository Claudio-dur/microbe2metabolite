r"""Fine-tune MGM with a pooled regression head (E3) on task files; runs inside envs/mgm.

    envs/mgm/.venv/bin/python -I scripts/mgm_finetune.py \
        --round-dir data/processed/mgm/e3/cv --workers 6 --threads 2

m2m.experiments.mgm_finetune (main environment) writes the tasks, fits every preprocessing step
and does all scoring; this script only trains and predicts. A task's `init` says which network
to build: "pretrained" (MGM's weights, mgm_finetune) or "random" (the same GPT-2 architecture,
from MGM's config.json, with transformers' standard initialisation: the from-scratch control
mgm_scratch). For every task listed in <round-dir>/manifest.json that has no up-to-date
predictions (so a re-run resumes):

  1. set_seed(seed), then build the network: the seed fixes the new head's weights, and with
     init "random" the whole backbone's too;
  2. set_trainable_blocks(k): the last k blocks + final LayerNorm train, the rest is frozen (and
     stays in eval mode); a random network must train every block (k >= its layer count);
  3. trim every token tensor to its longest real sequence (D1 needs <= 48 of MGM's 512 slots);
  4. fit_model with AdamW groups (backbone lr, head lr), early stopping on the task's inner
     validation rows, best epoch restored;
  5. predict the test tokens; write preds/<task_id>.npz, then preds/<task_id>.json (last, so a
     crash never leaves a task that looks finished without predictions).

Frozen-prefix cache (0 < k < n_layer, unless --no-prefix-cache): the embeddings and frozen
blocks run once per task, and only the last k blocks + head train on their cached output
(docs/mgm_notes.md section 5: 3-4x faster per step). It is exact because frozen layers are always
in eval mode, so they compute the same output in every epoch. Each task verifies it: cached path
vs full model on one training batch before training, and on every test prediction after training
(max |diff| <= --cache-tol, recorded in the .json). Predictions always come from the full model.

Imports: m2m.models and m2m.train (torch + numpy only) and, when a network is built,
transformers. Nothing from MGM's own package: scripts/mgm_embed.py made the tokens.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import multiprocessing
import os
import platform
import sys
import tempfile
import time
import traceback
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from m2m.models import MGMRegressor, pool_hidden_states
from m2m.train import (
    TrainConfig,
    effective_batch_size,
    fit_model,
    fit_row_offset,
    predict,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "raw" / "MGM" / "mgm" / "resources" / "general_model"

# Task-file layout; must equal m2m.experiments.mgm_finetune.FORMAT_VERSION. Version 3 added
# `init` (which weights the network starts from).
FORMAT_VERSION = 3
MANIFEST = "manifest.json"
TASKS_DIR, PREDS_DIR = "tasks", "preds"
SPLITS = ("train", "val", "test")
TOKEN_ARRAYS = ("input_ids", "attention_mask", "sample_ids")
SPEC_KEYS = (
    "task_id", "task_hash", "kind", "seed", "init", "n_trainable_blocks", "backbone_lr",
    "head_lr", "weight_decay", "head_dropout", "pooling", "batch_size", "max_epochs", "patience",
    "min_delta", "min_batches_per_epoch", "targets",
)
PRETRAINED, RANDOM = "pretrained", "random"
INITS = (PRETRAINED, RANDOM)
DEFAULT_CACHE_TOL = 1e-5
PREDICT_BATCH_SIZE = 64  # inference chunk size: bounds memory, never changes the predictions
DEFAULT_WORKERS, DEFAULT_THREADS = 6, 2

# (init=, n_out=, pooling=, head_dropout=) -> a fresh MGMRegressor; tests pass a small stand-in.
ModelBuilder = Callable[..., MGMRegressor]


class TaskError(ValueError):
    """A task file this script refuses to train on."""


# --- task files ------------------------------------------------------------------------------


def round_task_ids(round_dir: Path) -> list[str]:
    """The round's tasks as listed by the main environment (its manifest is written last)."""
    path = round_dir / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: the round is incomplete; re-run its prepare")
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise TaskError(f"{path}: format {manifest.get('format_version')!r}, not {FORMAT_VERSION}")
    return [str(task_id) for task_id in manifest["tasks"]]


def load_task(round_dir: Path, task_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """A task's JSON spec and npz arrays, after `check_task`."""
    spec = json.loads((round_dir / TASKS_DIR / f"{task_id}.json").read_text())
    with np.load(round_dir / TASKS_DIR / f"{task_id}.npz", allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    check_task(spec, arrays)
    return spec, arrays


def check_task(spec: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    """Refuse malformed tasks and any overlap between training, early-stopping and test ids.

    The main environment checks the same before writing; this is the last line of defence
    before a model sees the data.
    """
    task_id = spec.get("task_id", "?")
    if spec.get("format_version") != FORMAT_VERSION:
        raise TaskError(f"{task_id}: format {spec.get('format_version')!r}, not {FORMAT_VERSION}")
    missing = [key for key in SPEC_KEYS if key not in spec]
    missing += [f"{s}_{a}" for s in SPLITS for a in TOKEN_ARRAYS if f"{s}_{a}" not in arrays]
    missing += [key for key in ("Y_train", "Y_val") if key not in arrays]
    if missing:
        raise TaskError(f"{task_id}: missing {missing}")
    if spec["init"] not in INITS:
        raise TaskError(f"{task_id}: init must be one of {INITS}, got {spec['init']!r}")
    ids = {split: [str(i) for i in arrays[f"{split}_sample_ids"]] for split in SPLITS}
    leaked = (set(ids["train"]) | set(ids["val"])) & set(ids["test"])
    if leaked:
        raise TaskError(
            f"{task_id}: {len(leaked)} test ids overlap the training/early-stopping ids, "
            f"e.g. {sorted(leaked)[:3]}"
        )
    if set(ids["train"]) & set(ids["val"]):
        raise TaskError(f"{task_id}: training and early-stopping ids overlap")
    for split in SPLITS:
        input_ids, mask = arrays[f"{split}_input_ids"], arrays[f"{split}_attention_mask"]
        n_ids = len(ids[split])
        if input_ids.ndim != 2 or input_ids.shape != mask.shape or len(input_ids) != n_ids:
            raise TaskError(f"{task_id}: {split} token arrays do not match its {n_ids} ids")
        if len(input_ids) == 0:
            raise TaskError(f"{task_id}: no {split} samples")
        _check_right_padded(mask, f"{task_id} {split}")
    for split, key in (("train", "Y_train"), ("val", "Y_val")):
        targets = arrays[key]
        if targets.shape != (len(ids[split]), len(spec["targets"])):
            raise TaskError(f"{task_id}: {key} has shape {targets.shape}")
        if not np.isfinite(targets).all():
            raise TaskError(f"{task_id}: {key} contains NaN or infinite values")


def _check_right_padded(mask: np.ndarray, what: str) -> None:
    """0/1 masks, real tokens first: what MGM's tokenizer produces and pooling relies on."""
    if not np.isin(mask, (0, 1)).all():
        raise TaskError(f"{what}: the attention mask is not 0/1")
    lengths = mask.sum(axis=1)
    if (lengths == 0).any():
        raise TaskError(f"{what}: a sample has no real token")
    right_padded = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(mask.astype(bool), right_padded):
        raise TaskError(f"{what}: tokens are not right-padded")


def is_done(round_dir: Path, task_id: str) -> bool:
    """True if the task has predictions computed from its current task file."""
    record_path = round_dir / PREDS_DIR / f"{task_id}.json"
    if not (record_path.exists() and (round_dir / PREDS_DIR / f"{task_id}.npz").exists()):
        return False
    task_hash = json.loads((round_dir / TASKS_DIR / f"{task_id}.json").read_text())["task_hash"]
    return json.loads(record_path.read_text()).get("task_hash") == task_hash


def write_predictions(
    round_dir: Path,
    task_id: str,
    test_sample_ids: Sequence[str],
    predictions: np.ndarray,
    record: Mapping[str, Any],
) -> None:
    """preds/<task_id>.npz, then the .json record (which must carry the task's hash)."""
    if "task_hash" not in record:
        raise ValueError("the prediction record must name the task_hash it answers")
    folder = round_dir / PREDS_DIR
    _atomic_savez(
        folder / f"{task_id}.npz",
        # float32 is the model's own precision, so this loses nothing and halves the file.
        predictions=np.asarray(predictions, dtype=np.float32),
        test_sample_ids=np.asarray([str(i) for i in test_sample_ids], dtype=str),
    )
    _atomic_write_text(folder / f"{task_id}.json", json.dumps(dict(record), indent=1) + "\n")


# --- model pieces ----------------------------------------------------------------------------


def trimmed(input_ids: np.ndarray, attention_mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Token tensors cut to the longest real sequence (right padding past it only costs time).

    With causal attention a real token never attends to later positions, so the padding width
    does not change its output (docs/mgm_notes.md section 4: equal up to float rounding).
    """
    width = int(attention_mask.sum(axis=1).max())
    return (
        torch.from_numpy(np.ascontiguousarray(input_ids[:, :width])).long(),
        torch.from_numpy(np.ascontiguousarray(attention_mask[:, :width])).long(),
    )


class TrainableSuffix(nn.Module):
    """The trainable end of an MGMRegressor, fed by the cached output of its frozen prefix.

    Holds the regressor's own last k blocks, final LayerNorm, head dropout and head (shared
    modules, not copies): training this module trains the regressor, and fit_model's restore of
    the best epoch lands in the regressor too.
    """

    def __init__(self, model: MGMRegressor) -> None:
        super().__init__()
        k = model.n_trainable_blocks
        if not 0 < k < model.n_layer:
            raise ValueError(f"a cached prefix needs 0 < k < {model.n_layer}, got k = {k}")
        self.blocks = nn.ModuleList(model.backbone.h[model.n_layer - k :])
        self.ln_f = model.backbone.ln_f
        self.dropout = model.dropout
        self.head = model.head
        self.pooling = model.pooling

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # GPT2Model.forward (transformers 4.33) turns the 0/1 mask into this additive form before
        # its blocks; the per-task cache check fails loudly if another version does otherwise.
        additive = attention_mask[:, None, None, :].to(hidden.dtype)
        additive = (1.0 - additive) * torch.finfo(hidden.dtype).min
        for block in self.blocks:
            hidden = block(hidden, attention_mask=additive)[0]
        pooled = pool_hidden_states(self.ln_f(hidden), attention_mask, self.pooling)
        return self.head(self.dropout(pooled))


@torch.no_grad()
def frozen_prefix(
    model: MGMRegressor, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Hidden states entering the first trainable block, computed in eval mode as in training."""
    first_trainable = model.n_layer - model.n_trainable_blocks
    was_training = model.training
    model.eval()
    try:
        output = model.backbone(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
    finally:
        model.train(was_training)
    # hidden_states[i] is the input of block i; the last entry is the final, normalised state.
    return output.hidden_states[first_trainable].detach()


def cached_path_difference(
    model: MGMRegressor,
    suffix: TrainableSuffix,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> float:
    """max |full model - cached path| over these rows, both in eval mode."""
    full = predict(model, (input_ids, attention_mask), PREDICT_BATCH_SIZE)
    prefix = frozen_prefix(model, input_ids, attention_mask)
    cached = predict(suffix, (prefix, attention_mask), PREDICT_BATCH_SIZE)
    return float(np.abs(full - cached).max())


# --- one task --------------------------------------------------------------------------------


def run_task(
    round_dir: Path,
    task_id: str,
    *,
    model_dir: Path | None,
    use_cache: bool = True,
    cache_tol: float = DEFAULT_CACHE_TOL,
    build: ModelBuilder | None = None,
) -> dict[str, Any]:
    """Train one task, predict its test rows, write the predictions; returns the .json record."""
    started = time.perf_counter()
    spec, arrays = load_task(round_dir, task_id)
    seed = int(spec["seed"])
    tokens = {
        split: trimmed(arrays[f"{split}_input_ids"], arrays[f"{split}_attention_mask"])
        for split in SPLITS
    }
    y_train = torch.from_numpy(np.asarray(arrays["Y_train"], dtype=np.float32))
    y_val = torch.from_numpy(np.asarray(arrays["Y_val"], dtype=np.float32))
    # Same fit-row centring as TorchMLPRegressor (see m2m.train.fit_row_offset): without it the
    # small learning-curve subsets early-stop on a split artefact instead of on learning.
    offset = fit_row_offset(y_train)
    y_train, y_val = y_train - offset, y_val - offset

    # Before the model exists: the seed fixes the new head's initial weights and, for a random
    # network, the backbone's too.
    set_seed(seed)
    model = (build or mgm_builder(model_dir))(
        init=spec["init"],
        n_out=len(spec["targets"]),
        pooling=spec["pooling"],
        head_dropout=float(spec["head_dropout"]),
    )
    model.set_trainable_blocks(int(spec["n_trainable_blocks"]))
    if spec["init"] == RANDOM and model.n_trainable_blocks < model.n_layer:
        # Frozen random blocks would be a random-feature model, not the from-scratch control.
        raise TaskError(
            f"{task_id}: a randomly initialised network must train all {model.n_layer} blocks, "
            f"got n_trainable_blocks = {spec['n_trainable_blocks']}"
        )
    groups = model.param_groups(
        float(spec["backbone_lr"]), float(spec["head_lr"]), float(spec["weight_decay"])
    )
    train_cfg = TrainConfig(
        lr=float(spec["head_lr"]),  # every group sets its own lr; this is only the fallback
        weight_decay=float(spec["weight_decay"]),
        batch_size=int(spec["batch_size"]),
        max_epochs=int(spec["max_epochs"]),
        patience=int(spec["patience"]),
        min_delta=float(spec["min_delta"]),
        min_batches_per_epoch=int(spec["min_batches_per_epoch"]),
    )
    cached = use_cache and 0 < model.n_trainable_blocks < model.n_layer
    checks: dict[str, float] = {}
    fit_started = time.perf_counter()
    if cached:
        suffix = TrainableSuffix(model)
        _require_same_parameters(suffix, groups)
        (train_ids, train_mask), (val_ids, val_mask) = tokens["train"], tokens["val"]
        first_batch = slice(0, train_cfg.batch_size)
        checks["before_training"] = _checked_difference(
            model, suffix, train_ids[first_batch], train_mask[first_batch], cache_tol, task_id
        )
        history = fit_model(
            suffix,
            (frozen_prefix(model, train_ids, train_mask), train_mask, y_train),
            (frozen_prefix(model, val_ids, val_mask), val_mask, y_val),
            train_cfg,
            seed=seed,
            param_groups=groups,
        )
    else:
        history = fit_model(
            model, (*tokens["train"], y_train), (*tokens["val"], y_val), train_cfg,
            seed=seed, param_groups=groups,
        )
    fit_seconds = time.perf_counter() - fit_started
    predictions = predict(model, tokens["test"], PREDICT_BATCH_SIZE) + offset.double().numpy()
    if cached:
        checks["test_predictions"] = _checked_difference(
            model, suffix, *tokens["test"], cache_tol, task_id
        )
    record = {
        "task_id": task_id,
        "task_hash": spec["task_hash"],
        "kind": spec["kind"],
        "init": spec["init"],
        "n_trainable_blocks": model.n_trainable_blocks,
        "n_trainable_parameters": sum(p.numel() for group in groups for p in group["params"]),
        "batch_size_used": effective_batch_size(len(y_train), train_cfg),
        "epochs_run": history.stopped_epoch,
        "best_epoch": history.best_epoch,
        "best_val_loss": history.best_val_loss,
        "train_loss": history.train_loss,
        "val_loss": history.val_loss,
        "fit_seconds": fit_seconds,
        "seconds_per_epoch": fit_seconds / max(history.stopped_epoch, 1),
        "seconds": time.perf_counter() - started,
        "token_width": {split: int(tokens[split][0].shape[1]) for split in SPLITS},
        "prefix_cache": cached,
        "cache_check_max_abs_diff": checks,
        "cache_tol": cache_tol,
        "torch_threads": torch.get_num_threads(),
        "versions": _versions(),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_predictions(round_dir, task_id, arrays["test_sample_ids"], predictions, record)
    return record


def random_backbone(model_dir: Path) -> nn.Module:
    """MGM's GPT-2 architecture (its config.json) with transformers' standard initialisation.

    Draws its weights from torch's global RNG, which run_task seeds with the task's seed just
    before, so a seed always gives the same network. scripts/mgm_embed.py builds E2's frozen
    random-weight control the same way after torch.manual_seed(seed): E2's network of seed s is
    this network of seed s before training (tests check that the two stay identical).
    """
    from transformers import GPT2Config, GPT2Model  # only envs/mgm has transformers

    return GPT2Model(GPT2Config.from_pretrained(str(model_dir)))


def mgm_builder(model_dir: Path | None) -> ModelBuilder:
    """Builds a task's network: MGM's pretrained backbone, or MGM's architecture at random."""
    directory = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR

    def build(*, init: str, n_out: int, pooling: str, head_dropout: float) -> MGMRegressor:
        if init == PRETRAINED:
            return MGMRegressor.from_pretrained(
                directory, n_out, pooling=pooling, head_dropout=head_dropout
            )
        if init == RANDOM:
            return MGMRegressor(
                random_backbone(directory), n_out, pooling=pooling, head_dropout=head_dropout
            )
        raise ValueError(f"init must be one of {INITS}, got {init!r}")

    return build


def _require_same_parameters(suffix: TrainableSuffix, groups: Sequence[Mapping[str, Any]]) -> None:
    """The cached path must hold exactly the parameters the optimiser updates."""
    trained = {id(p) for p in suffix.parameters() if p.requires_grad}
    optimised = {id(p) for group in groups for p in group["params"]}
    if trained != optimised:
        raise RuntimeError("the cached path and the optimiser disagree on the trainable parameters")


def _checked_difference(
    model: MGMRegressor,
    suffix: TrainableSuffix,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tol: float,
    task_id: str,
) -> float:
    difference = cached_path_difference(model, suffix, input_ids, attention_mask)
    if not difference <= tol:  # also catches NaN
        raise RuntimeError(
            f"{task_id}: the cached prefix path differs from the full model by {difference:.3g} "
            f"(tolerance {tol:g}); rerun with --no-prefix-cache and report it"
        )
    return difference


def _versions() -> dict[str, str | None]:
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:  # only the stand-in models of the main-environment tests
        transformers_version = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "numpy": np.__version__,
    }


# --- a round ---------------------------------------------------------------------------------


def run_round(
    round_dir: Path,
    *,
    workers: int,
    threads: int,
    model_dir: Path | None,
    only: str | None = None,
    use_cache: bool = True,
    cache_tol: float = DEFAULT_CACHE_TOL,
    build: ModelBuilder | None = None,
) -> list[str]:
    """Run every task of the round without up-to-date predictions; returns the failed task ids.

    workers == 1 runs in this process (and accepts a custom `build`); otherwise each task runs
    in a fresh spawned process that first sets its own torch threads, so no task inherits
    another's state and the results do not depend on how tasks were scheduled.
    """
    task_ids = round_task_ids(round_dir)
    if only is not None:
        task_ids = [t for t in task_ids if fnmatch.fnmatchcase(t, only)]
    pending = [t for t in task_ids if not is_done(round_dir, t)]
    # Longest first (fits that train the whole backbone dominate), so no worker gets a long tail.
    pending.sort(key=lambda t: _expected_cost(round_dir, t), reverse=True)
    print(
        f"[mgm_finetune] {round_dir}: {len(task_ids)} tasks, {len(task_ids) - len(pending)} "
        f"already done, {len(pending)} to run ({workers} workers x {threads} threads)",
        flush=True,
    )
    options = {"model_dir": model_dir, "use_cache": use_cache, "cache_tol": cache_tol}
    records = []
    if workers == 1:
        for task_id in pending:
            records.append(_run_safely(round_dir, task_id, build=build, **options))
            print(_describe(records[-1], len(records), len(pending)), flush=True)
    else:
        if build is not None:
            raise ValueError("a custom model builder only runs in-process (workers=1)")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(threads,),
            max_tasks_per_child=1,
        ) as pool:
            futures = [pool.submit(_run_safely, round_dir, t, **options) for t in pending]
            for future in as_completed(futures):
                records.append(future.result())
                print(_describe(records[-1], len(records), len(pending)), flush=True)
    return [record["task_id"] for record in records if "error" in record]


def _run_safely(round_dir: Path, task_id: str, **kwargs: Any) -> dict[str, Any]:
    """run_task, reporting a failure instead of stopping every other task."""
    try:
        return run_task(round_dir, task_id, **kwargs)
    except Exception:  # noqa: BLE001 - reported with its traceback, the round goes on
        return {"task_id": task_id, "error": traceback.format_exc()}


def _expected_cost(round_dir: Path, task_id: str) -> tuple[int, int]:
    spec = json.loads((round_dir / TASKS_DIR / f"{task_id}.json").read_text())
    return int(spec["n_trainable_blocks"]), len(spec["targets"])


def _describe(record: Mapping[str, Any], done: int, total: int) -> str:
    prefix = f"[mgm_finetune] {done}/{total} {record['task_id']}"
    if "error" in record:
        return f"{prefix} FAILED\n{record['error']}"
    checks = record["cache_check_max_abs_diff"]
    cache = (
        "cache on, max|diff| " + " / ".join(f"{value:.1e}" for value in checks.values())
        if record["prefix_cache"]
        else "cache off"
    )
    return (
        f"{prefix}: {record['epochs_run']} epochs (best {record['best_epoch']}, val MSE "
        f"{record['best_val_loss']:.4f}), {record['seconds']:.1f} s "
        f"({record['seconds_per_epoch']:.2f} s/epoch), {cache}"
    )


def _init_worker(threads: int) -> None:
    """First thing in every worker: torch 2.0.1 takes a thread count only before any work."""
    torch.set_num_threads(threads)
    _quiet_known_warnings()


def _quiet_known_warnings() -> None:
    # torch 2.0.1's own weights_only loader trips its TypedStorage deprecation; nothing to act on.
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temporary file + rename, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as fh:
        fh.write(text)
    os.replace(fh.name, path)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(fh.name, path)


# --- command line ----------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parallel processes")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="torch threads each")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--only", default=None, help="only task ids matching this glob (dev)")
    parser.add_argument("--no-prefix-cache", action="store_true", help="train the full model")
    parser.add_argument("--cache-tol", type=float, default=DEFAULT_CACHE_TOL)
    args = parser.parse_args(argv)

    # The first torch call: torch 2.0.1's thread pool ignores any later change (mgm_notes §5).
    torch.set_num_threads(args.threads)
    _quiet_known_warnings()
    if args.workers < 1 or args.threads < 1:
        parser.error("--workers and --threads must be >= 1")
    if args.workers * args.threads > (os.cpu_count() or 1):
        parser.error(f"{args.workers} x {args.threads} threads exceed the {os.cpu_count()} CPUs")
    failed = run_round(
        args.round_dir,
        workers=args.workers,
        threads=args.threads,
        model_dir=args.model_dir,
        only=args.only,
        use_cache=not args.no_prefix_cache,
        cache_tol=args.cache_tol,
    )
    if failed:
        print(f"[mgm_finetune] {len(failed)} task(s) failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
