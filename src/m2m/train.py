"""Seeded training loop with early stopping, shared by every neural model (E1, E3, E4).

Only numpy, torch and the standard library are imported: the MGM environment (torch 2.0.1,
numpy 1.24) runs this loop too.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import logging
import math
import numbers
import operator
import random
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from m2m.models import MLP, load_compatible_state

logger = logging.getLogger(__name__)

Batch = tuple[torch.Tensor, ...]
LossFn = Callable[[nn.Module, Batch], torch.Tensor]

# Inference chunk size: it bounds memory and never changes the predictions.
PREDICT_BATCH_SIZE = 256


def set_seed(seed: int) -> None:
    """Seed Python's, NumPy's global and torch's RNGs (weight initialisation and dropout)."""
    seed = operator.index(seed)  # random.seed rejects NumPy integers such as np.arange(5)[3]
    random.seed(seed)
    # The legacy global RNG on purpose: it is what code without an explicit generator draws from.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)


@dataclass(frozen=True)
class TrainConfig:
    """Optimiser and early-stopping settings, usually the ``train:`` block of a config."""

    lr: float = 1e-3
    weight_decay: float = 1e-4  # AdamW's decoupled weight decay
    batch_size: int = 32
    max_epochs: int = 500
    patience: int = 30  # epochs without improvement before stopping
    min_delta: float = 0.0  # an epoch improves only if its val loss < best - min_delta
    num_threads: int | None = None  # torch threads for this process; 1 in parallel fold workers
    # Small training sets (learning-curve subsets) shrink the batch so an epoch still has this
    # many optimiser steps; otherwise patience counts single steps and early stopping fires
    # before the model has moved from its initialisation. 1 = always use batch_size.
    min_batches_per_epoch: int = 1

    def __post_init__(self) -> None:
        for name in ("lr", "weight_decay", "min_delta"):
            _check_number(name, getattr(self, name), minimum=0.0)
        for name, minimum in (
            ("batch_size", 1), ("max_epochs", 1), ("patience", 1), ("min_batches_per_epoch", 1)
        ):
            _check_integer(name, getattr(self, name), minimum)
        if self.num_threads is not None:
            _check_integer("num_threads", self.num_threads, minimum=1)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TrainConfig:
        """Build from a config mapping; an unknown key raises instead of falling back silently."""
        known = [f.name for f in dataclasses.fields(cls)]
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"unknown TrainConfig keys {unknown}; valid keys are {known}")
        return cls(**values)


@dataclass
class TrainHistory:
    """Per-epoch mean losses and the early-stopping outcome.

    Epochs count from 1: ``val_loss[best_epoch - 1] == best_val_loss`` and
    ``len(train_loss) == stopped_epoch``.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = math.inf
    stopped_epoch: int = 0


def mse_loss(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Default objective: mean squared error over every sample and target."""
    *inputs, target = batch
    prediction = model(*inputs)
    if prediction.shape != target.shape:
        # F.mse_loss would broadcast e.g. (n, 1) against (n,) and return a meaningless number.
        raise ValueError(
            f"model output {tuple(prediction.shape)} does not match target {tuple(target.shape)}"
        )
    return F.mse_loss(prediction, target)


def fit_model(
    model: nn.Module,
    train_tensors: Sequence[torch.Tensor],
    val_tensors: Sequence[torch.Tensor],
    cfg: TrainConfig,
    *,
    seed: int,
    param_groups: Sequence[dict[str, Any]] | None = None,
    loss_fn: LossFn | None = None,
) -> TrainHistory:
    """Train with AdamW until the validation loss stops improving, then restore the best weights.

    ``*_tensors`` are ``(*inputs, target)`` with samples on the first axis. ``loss_fn(model,
    batch)`` returns a batch's mean loss (default: MSE of ``model(*inputs)`` against ``target``);
    it scores validation too and can read ``model.training`` to tell the two apart.
    ``param_groups`` (e.g. ``MGMRegressor.param_groups``) set per-group learning rates; groups
    without ``lr``, and the default (every trainable parameter), use ``cfg.lr``.

    ``seed`` fixes the mini-batch order through a private generator. Initialisation and dropout
    use torch's global RNG, so call ``set_seed`` before building the model (``TorchMLPRegressor``
    does) for a reproducible fit. ``cfg.num_threads`` applies to the whole process and is not
    undone afterwards (see ``_apply_num_threads``). The model is returned in eval mode.
    """
    _check_supervised(train_tensors, "train_tensors")
    _check_supervised(val_tensors, "val_tensors")
    loss_fn = loss_fn or mse_loss
    params = (
        [p for p in model.parameters() if p.requires_grad]
        if param_groups is None
        else list(param_groups)
    )
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    generator = torch.Generator().manual_seed(operator.index(seed))  # NumPy ints rejected too
    _apply_num_threads(cfg.num_threads)
    history = TrainHistory()
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    batch_size = effective_batch_size(len(train_tensors[0]), cfg)
    for epoch in range(1, cfg.max_epochs + 1):
        history.train_loss.append(
            _train_epoch(model, train_tensors, optimizer, loss_fn, batch_size, generator)
        )
        # Validation batches reuse the training batch size: it is known to fit in memory.
        val_loss = evaluate_loss(model, val_tensors, loss_fn, batch_size=cfg.batch_size)
        history.val_loss.append(val_loss)
        history.stopped_epoch = epoch
        if not math.isfinite(val_loss):
            break  # NaN/inf weights never recover: keep the best epoch so far
        if val_loss < history.best_val_loss - cfg.min_delta:
            history.best_epoch, history.best_val_loss = epoch, val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                break
    if best_state is None:
        raise FloatingPointError(
            "validation loss was not finite after the first epoch; check the inputs or lower lr"
        )
    model.load_state_dict(best_state)
    model.eval()
    logger.debug(
        "stopped after epoch %d; restored epoch %d (val loss %.4g)",
        history.stopped_epoch, history.best_epoch, history.best_val_loss,
    )
    return history


def effective_batch_size(n_train: int, cfg: TrainConfig) -> int:
    """`cfg.batch_size`, shrunk only if an epoch would have fewer than `min_batches_per_epoch`
    steps. Full-size PRISM fits (99-124 rows, batch 32: 4 steps) are unaffected at the default
    of 4; a 16-row learning-curve subset trains on batches of 4 instead of one full batch."""
    if math.ceil(n_train / cfg.batch_size) >= cfg.min_batches_per_epoch:
        return cfg.batch_size
    return max(1, math.ceil(n_train / cfg.min_batches_per_epoch))


def fit_row_offset(y_fit: torch.Tensor) -> torch.Tensor:
    """Per-target mean of the rows the model is fitted on (subtracted in training, added back
    when predicting).

    Why: targets are z-scored on all training rows, early-stopping rows included (DESIGN
    decision #2), so the early-stopping rows' mean is the fit rows' mean times -n_fit/n_val
    (exactly -4x with a 20% split). At n = 20 that offset dominates the 4-row validation loss:
    every step toward the fit-row mean raises it and early stopping keeps the untrained epoch-1
    network. Centring on the fit rows removes the artefact; it uses only fit rows (no leakage)
    and adds one constant per target, which leaves every per-metabolite Spearman ρ unchanged.
    """
    return y_fit.mean(dim=0)


def evaluate_loss(
    model: nn.Module,
    tensors: Sequence[torch.Tensor],
    loss_fn: LossFn | None = None,
    *,
    batch_size: int = PREDICT_BATCH_SIZE,
) -> float:
    """Sample-weighted mean of ``loss_fn`` over every row of ``(*inputs, target)``, in eval mode."""
    loss_fn = loss_fn or mse_loss
    n = _check_supervised(tensors, "tensors")
    total = 0.0
    with _evaluating(model):
        for start in range(0, n, batch_size):
            batch = tuple(t[start : start + batch_size] for t in tensors)
            total += loss_fn(model, batch).item() * len(batch[0])
    return total / n


def predict(
    model: nn.Module,
    inputs: Sequence[torch.Tensor] | torch.Tensor,
    batch_size: int = PREDICT_BATCH_SIZE,
) -> np.ndarray:
    """``model(*inputs)`` for every row as a float64 array, in eval mode and in chunks.

    Lower ``batch_size`` for long token sequences (MGM): attention memory grows with
    batch × length².
    """
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)
    n = _n_rows(inputs, "inputs")
    with _evaluating(model):
        chunks = [
            model(*(t[start : start + batch_size] for t in inputs))
            for start in range(0, n, batch_size)
        ]
    return torch.cat(chunks).to(torch.float64).numpy()


class TorchMLPRegressor:
    """scikit-learn-style wrapper around ``MLP`` + ``fit_model``, used by the CV protocol.

    ``needs_validation`` tells the protocol to carve an inner early-stopping split out of the
    training rows and pass it to ``fit``. Constructor arguments are stored unchanged and exposed
    through ``get_params`` so ``sklearn.base.clone`` can copy an unfitted regressor.

    ``init_state`` is an optional state_dict (e.g. an E4a autoencoder or an E4b regressor)
    loaded after construction: entries whose name or shape do not match the new MLP are skipped
    and listed in ``skipped_init_keys_``; the ones used are in ``loaded_init_keys_``. A layer
    that would load only in part (a trunk pretrained on other input features) raises instead.
    """

    needs_validation = True
    _PARAM_NAMES = ("hidden", "dropout", "train_cfg", "seed", "init_state")

    def __init__(
        self,
        *,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        train_cfg: TrainConfig | Mapping[str, Any],
        seed: int = 0,
        init_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.hidden = hidden
        self.dropout = dropout
        self.train_cfg = train_cfg
        self.seed = seed
        self.init_state = init_state

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Constructor arguments, as ``sklearn.base.clone`` expects (``deep`` is ignored)."""
        return {name: getattr(self, name) for name in self._PARAM_NAMES}

    def set_params(self, **params: Any) -> TorchMLPRegressor:
        unknown = sorted(set(params) - set(self._PARAM_NAMES))
        if unknown:
            raise ValueError(f"unknown parameters {unknown}; valid: {list(self._PARAM_NAMES)}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def fit(self, X: Any, Y: Any, X_val: Any, Y_val: Any) -> TorchMLPRegressor:
        """Train a fresh MLP on (X, Y), early-stopping on (X_val, Y_val); 2-D array-likes."""
        if X_val is None or Y_val is None:
            raise ValueError("TorchMLPRegressor needs a validation split for early stopping")
        cfg = (
            self.train_cfg
            if isinstance(self.train_cfg, TrainConfig)
            else TrainConfig.from_dict(self.train_cfg)
        )
        x, y = _as_float_tensor(X, "X"), _as_float_tensor(Y, "Y")
        x_val, y_val = _as_float_tensor(X_val, "X_val"), _as_float_tensor(Y_val, "Y_val")
        if x_val.shape[1] != x.shape[1] or y_val.shape[1] != y.shape[1]:
            raise ValueError("validation data must have the same columns as the training data")
        offset = fit_row_offset(y)
        set_seed(self.seed)
        model = MLP(x.shape[1], y.shape[1], hidden=self.hidden, dropout=self.dropout)
        if self.init_state is None:
            self.loaded_init_keys_, self.skipped_init_keys_ = [], []
        else:
            self.loaded_init_keys_, self.skipped_init_keys_ = load_compatible_state(
                model, self.init_state
            )
        self.history_ = fit_model(
            model, (x, y - offset), (x_val, y_val - offset), cfg, seed=self.seed
        )
        self.model_ = model
        self.target_offset_ = offset
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predictions (float64, one column per target) in the space the targets were given in."""
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchMLPRegressor is not fitted; call fit() first")
        x = _as_float_tensor(X, "X")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x.shape[1]} features; the model takes {self.n_features_in_}")
        # The module-level predict(), not this method; the fit-row offset is added back.
        return predict(self.model_, (x,)) + self.target_offset_.to(torch.float64).numpy()


def _train_epoch(
    model: nn.Module,
    tensors: Sequence[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    """One pass over the training rows in a fresh random order; the sample-weighted mean loss."""
    model.train()
    n = len(tensors[0])
    order = torch.randperm(n, generator=generator)
    total = 0.0
    for start in range(0, n, batch_size):
        rows = order[start : start + batch_size]
        loss = loss_fn(model, tuple(t[rows] for t in tensors))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(rows)
    return total / n


@contextlib.contextmanager
def _evaluating(model: nn.Module) -> Iterator[None]:
    """Eval mode without gradients; the model's previous mode is restored afterwards."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        model.train(was_training)


def _apply_num_threads(n: int | None) -> None:
    """Give torch ``n`` intra-op threads for the rest of this process (None: leave it alone).

    Deliberately not restored after the fit: torch's native thread pool (e.g. torch 2.0.1 on
    macOS, the MGM environment) accepts a single change made before any parallel work and
    ignores later calls. Parallel folds run in worker processes, each set on its first fit.
    """
    if n is None or torch.get_num_threads() == n:
        return
    torch.set_num_threads(n)
    if torch.get_num_threads() != n:
        warnings.warn(
            f"torch kept {torch.get_num_threads()} threads instead of num_threads={n}: this "
            "build only accepts the setting before any parallel work has started",
            RuntimeWarning,
            stacklevel=3,
        )


def _n_rows(tensors: Sequence[torch.Tensor], name: str) -> int:
    """The first-axis length shared by ``tensors``; rejects empty or ragged sets."""
    lengths = {len(t) for t in tensors}
    if len(lengths) > 1:
        raise ValueError(f"{name}: tensors disagree on the number of rows {sorted(lengths)}")
    if not lengths or lengths == {0}:
        raise ValueError(f"{name} has no rows")
    return lengths.pop()


def _check_supervised(tensors: Sequence[torch.Tensor], name: str) -> int:
    if len(tensors) < 2:
        raise ValueError(f"{name} must be (*inputs, target); got {len(tensors)} tensor(s)")
    return _n_rows(tensors, name)


def _as_float_tensor(values: Any, name: str) -> torch.Tensor:
    """2-D float32 tensor copied from an array-like (NumPy array or DataFrame); finite only."""
    # A fresh C-ordered copy: pandas hands over column-major data, and the memory layout changes
    # matmul rounding, so a DataFrame and an ndarray with equal values would fit differently.
    array = np.array(values, dtype=np.float32, order="C")
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D (samples x features), got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return torch.from_numpy(array)


def _check_number(name: str, value: Any, minimum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        # PyYAML follows YAML 1.1, where 1e-3 (no dot) is text; 1.0e-3 is a float.
        hint = " (YAML reads 1e-3 as text: write 1.0e-3)" if isinstance(value, str) else ""
        raise TypeError(f"TrainConfig.{name} must be a number, got {value!r}{hint}")
    # inf passes a plain ">= minimum" (and NaN would fail it only by accident): an infinite lr
    # trains to NaN, and min_delta = inf makes best - min_delta NaN, so no epoch could ever count.
    if not (math.isfinite(value) and value >= minimum):
        raise ValueError(f"TrainConfig.{name} must be finite and >= {minimum}, got {value}")


def _check_integer(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"TrainConfig.{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"TrainConfig.{name} must be >= {minimum}, got {value}")
