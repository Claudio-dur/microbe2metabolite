"""Training loop: seeds, early stopping, param groups, custom losses, sklearn wrapper, leakage."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from m2m import train
from m2m.models import MLP, MaskedFeatureAutoencoder
from m2m.train import (
    TorchMLPRegressor,
    TrainConfig,
    TrainHistory,
    evaluate_loss,
    fit_model,
    mse_loss,
    predict,
    set_seed,
)

Arrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# torch 2.0.1 on macOS (the MGM environment) uses it; the main environment's build uses OpenMP.
NATIVE_THREAD_POOL = "native thread pool" in torch.__config__.parallel_info()


def _linear_problem(n: int = 160, n_val: int = 40, p: int = 5, m: int = 3) -> Arrays:
    """Targets that are a noisy linear map of the inputs: (X, Y, X_val, Y_val)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n + n_val, p))
    Y = X @ rng.normal(size=(p, m)) + 0.1 * rng.normal(size=(n + n_val, m))
    return X[:n], Y[:n], X[n:], Y[n:]


def _single_weight(start: float) -> nn.Linear:
    """y = w·x with a single weight, so the loss curves can be worked out by hand."""
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(start)
    return model


def _constant_rows(target: float, n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(n, 1), torch.full((n, 1), target)


def _same_weights(a: nn.Module, b: nn.Module) -> bool:
    state_b = b.state_dict()
    return all(torch.equal(value, state_b[key]) for key, value in a.state_dict().items())


# --- seeds and configuration ---------------------------------------------------------------------


def test_set_seed_makes_python_numpy_and_torch_reproducible() -> None:
    def draw() -> tuple[float, float, float]:
        # The legacy NumPy global RNG is exactly what set_seed must cover.
        return random.random(), np.random.rand(), torch.rand(1).item()  # noqa: NPY002

    set_seed(5)
    first = draw()
    set_seed(5)
    second = draw()
    set_seed(6)
    third = draw()
    assert first == second
    assert all(a != b for a, b in zip(first, third, strict=True))


def test_train_config_from_dict_reads_known_keys() -> None:
    cfg = TrainConfig.from_dict({"lr": 3e-4, "batch_size": np.int64(16), "num_threads": 1})
    assert cfg == TrainConfig(lr=3e-4, batch_size=16, num_threads=1)
    assert (cfg.max_epochs, cfg.patience, cfg.min_delta) == (500, 30, 0.0)  # brief's defaults


def test_train_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig.from_dict({"learning_rate": 1e-3})


def test_train_config_explains_yaml_exponent_strings() -> None:
    with pytest.raises(TypeError, match=r"1\.0e-3"):
        TrainConfig.from_dict({"lr": "1e-3"})


@pytest.mark.parametrize(
    "bad",
    [
        {"lr": -1.0},
        {"lr": float("nan")},
        {"lr": math.inf},
        {"weight_decay": math.inf},
        {"min_delta": math.inf},  # best - inf is NaN: no epoch could ever count as the best
        {"min_delta": -0.1},
        {"batch_size": 0},
        {"patience": 0},
        {"max_epochs": 2.5},
        {"patience": True},
        {"num_threads": 0},
    ],
)
def test_train_config_rejects_invalid_values(bad: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        TrainConfig(**bad)


# --- fit_model -----------------------------------------------------------------------------------


def test_early_stopping_stops_when_val_loss_rises_and_restores_best_weights() -> None:
    # Training pulls the single weight from -2 towards the train target 1 (about lr per Adam
    # step); validation wants 0, so its loss w² falls until w crosses 0 and rises after.
    model = _single_weight(-2.0)
    cfg = TrainConfig(lr=0.1, weight_decay=0.0, batch_size=8, max_epochs=200, patience=5)

    history = fit_model(model, _constant_rows(1.0), _constant_rows(0.0), cfg, seed=0)

    assert history.stopped_epoch < cfg.max_epochs
    assert history.stopped_epoch == history.best_epoch + cfg.patience
    assert len(history.train_loss) == len(history.val_loss) == history.stopped_epoch
    assert history.best_val_loss == history.val_loss[history.best_epoch - 1]
    assert history.best_val_loss == min(history.val_loss)
    assert history.val_loss[-1] > history.best_val_loss  # it really got worse before stopping
    # The model holds the best epoch's weights, not the last epoch's.
    assert evaluate_loss(model, _constant_rows(0.0)) == pytest.approx(history.best_val_loss)
    assert abs(model.weight.item()) < 0.1
    assert not model.training


def test_min_delta_ignores_improvements_that_are_too_small() -> None:
    # Validation improves every epoch, but never by min_delta: only epoch 1 counts.
    model = _single_weight(-2.0)
    cfg = TrainConfig(lr=0.01, weight_decay=0.0, batch_size=8, max_epochs=100, patience=3,
                      min_delta=10.0)

    history = fit_model(model, _constant_rows(1.0), _constant_rows(1.0), cfg, seed=0)

    assert all(b < a for a, b in zip(history.val_loss, history.val_loss[1:], strict=False))
    assert history.best_epoch == 1
    assert history.stopped_epoch == 1 + cfg.patience


def test_loss_decreases_on_a_learnable_linear_problem() -> None:
    X, Y, X_val, Y_val = _linear_problem()
    cfg = TrainConfig(lr=1e-2, batch_size=32, max_epochs=300, patience=20)

    reg = TorchMLPRegressor(hidden=(32,), dropout=0.0, train_cfg=cfg, seed=0)
    reg.fit(X, Y, X_val, Y_val)

    history = reg.history_
    assert history.best_val_loss < 0.05 * history.val_loss[0]
    assert history.train_loss[-1] < 0.05 * history.train_loss[0]
    residual = ((reg.predict(X_val) - Y_val) ** 2).sum()
    assert 1 - residual / ((Y_val - Y_val.mean(0)) ** 2).sum() > 0.95  # R² on held-out rows


def test_seed_fixes_the_batch_order() -> None:
    torch.manual_seed(0)
    base = MLP(5, 3, hidden=(8,), dropout=0.0)
    X, Y, X_val, Y_val = (torch.tensor(a, dtype=torch.float32) for a in _linear_problem())
    cfg = TrainConfig(lr=1e-2, batch_size=16, max_epochs=3, patience=3)

    models = {name: copy.deepcopy(base) for name in ("a", "a_again", "b")}
    for name, seed in (("a", 0), ("a_again", 0), ("b", 1)):
        fit_model(models[name], (X, Y), (X_val, Y_val), cfg, seed=seed)

    assert _same_weights(models["a"], models["a_again"])
    assert not _same_weights(models["a"], models["b"])


def test_param_group_learning_rates_are_honoured() -> None:
    torch.manual_seed(0)
    model = MLP(5, 3, hidden=(8,), dropout=0.0)
    trunk_before = copy.deepcopy(model.trunk)
    head_before = copy.deepcopy(model.head)
    X, Y, X_val, Y_val = (torch.tensor(a, dtype=torch.float32) for a in _linear_problem())
    # A group's own lr wins; a group without one falls back to cfg.lr. AdamW scales weight
    # decay by lr too, so an lr of 0 must leave the trunk bit-for-bit unchanged.
    groups = [{"params": model.trunk.parameters(), "lr": 0.0}, {"params": model.head.parameters()}]
    cfg = TrainConfig(lr=1e-2, weight_decay=0.1, batch_size=32, max_epochs=3, patience=3)

    fit_model(model, (X, Y), (X_val, Y_val), cfg, seed=0, param_groups=groups)

    assert _same_weights(model.trunk, trunk_before)
    assert not _same_weights(model.head, head_before)


def test_custom_loss_fn_is_used_for_training_and_validation() -> None:
    calls = {"train": 0, "eval": 0}

    def l1_loss(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        calls["train" if model.training else "eval"] += 1
        x, y = batch
        return (model(x) - y).abs().mean()

    torch.manual_seed(0)
    model = MLP(5, 3, hidden=(8,), dropout=0.0)
    X, Y, X_val, Y_val = (torch.tensor(a, dtype=torch.float32) for a in _linear_problem())
    cfg = TrainConfig(lr=1e-2, batch_size=50, max_epochs=4, patience=4)

    history = fit_model(model, (X, Y), (X_val, Y_val), cfg, seed=0, loss_fn=l1_loss)

    epochs = history.stopped_epoch
    assert calls["train"] == epochs * math.ceil(len(X) / cfg.batch_size)
    assert calls["eval"] == epochs * math.ceil(len(X_val) / cfg.batch_size)
    with torch.no_grad():
        l1 = (model(X_val) - Y_val).abs().mean().item()
        mse = ((model(X_val) - Y_val) ** 2).mean().item()
    assert history.best_val_loss == pytest.approx(l1, rel=1e-5)
    assert history.best_val_loss != pytest.approx(mse, rel=1e-3)


def test_non_finite_validation_loss_stops_training_and_keeps_the_best_epoch() -> None:
    val_calls = 0

    def breaks_on_third_validation(
        model: nn.Module, batch: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        nonlocal val_calls
        loss = mse_loss(model, batch)
        if not model.training:
            val_calls += 1
            if val_calls >= 3:
                return loss * math.nan
        return loss

    model = _single_weight(-2.0)
    cfg = TrainConfig(lr=0.1, weight_decay=0.0, batch_size=8, max_epochs=50)
    rows = _constant_rows(1.0)

    history = fit_model(model, rows, rows, cfg, seed=0, loss_fn=breaks_on_third_validation)

    assert (history.stopped_epoch, history.best_epoch) == (3, 2)
    assert math.isnan(history.val_loss[-1])
    assert evaluate_loss(model, rows) == pytest.approx(history.best_val_loss)


def test_fit_model_raises_when_no_epoch_had_a_finite_loss() -> None:
    model = MLP(2, 1, hidden=(4,))
    x, y = torch.randn(6, 2), torch.full((6, 1), math.nan)
    with pytest.raises(FloatingPointError):
        fit_model(model, (x, y), (x, y), TrainConfig(max_epochs=3), seed=0)


def test_fit_model_rejects_malformed_tensors() -> None:
    model = MLP(2, 1, hidden=(4,))
    cfg = TrainConfig(max_epochs=1)
    x, y = torch.randn(6, 2), torch.randn(6, 1)
    with pytest.raises(ValueError, match="inputs, target"):
        fit_model(model, (x,), (x, y), cfg, seed=0)
    with pytest.raises(ValueError, match="disagree"):
        fit_model(model, (x, y[:5]), (x, y), cfg, seed=0)
    with pytest.raises(ValueError, match="no rows"):
        fit_model(model, (x, y), (x[:0], y[:0]), cfg, seed=0)
    with pytest.raises(ValueError, match="does not match target"):
        fit_model(model, (x, y[:, 0]), (x, y), cfg, seed=0)  # (n,) target would broadcast


@pytest.fixture
def thread_count() -> Iterator[int]:
    """torch's thread count before the test, put back afterwards where the build allows it."""
    before = torch.get_num_threads()
    yield before
    if torch.get_num_threads() != before:
        torch.set_num_threads(before)


@pytest.mark.skipif(
    NATIVE_THREAD_POOL,
    reason="torch's native thread pool fixes the thread count once parallel work has started",
)
def test_fit_model_applies_num_threads_to_the_process(thread_count: int) -> None:
    seen: list[int] = []

    def spy_loss(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        seen.append(torch.get_num_threads())
        return mse_loss(model, batch)

    wanted = 2 if thread_count == 1 else 1
    x, y = torch.randn(10, 3), torch.randn(10, 2)
    cfg = TrainConfig(max_epochs=2, num_threads=wanted)

    fit_model(MLP(3, 2, hidden=(4,)), (x, y), (x, y), cfg, seed=0, loss_fn=spy_loss)

    assert set(seen) == {wanted}
    assert torch.get_num_threads() == wanted  # a per-process setting, kept after the fit


def test_fit_model_warns_when_torch_refuses_num_threads(
    thread_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A no-op is what torch 2.0.1's native thread pool does once parallel work has started.
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    x, y = torch.randn(10, 3), torch.randn(10, 2)
    cfg = TrainConfig(max_epochs=2, num_threads=thread_count + 1)

    with pytest.warns(RuntimeWarning, match="num_threads"):
        history = fit_model(MLP(3, 2, hidden=(4,)), (x, y), (x, y), cfg, seed=0)

    assert history.stopped_epoch == 2  # training still ran, on the threads torch kept


# --- evaluation helpers --------------------------------------------------------------------------


def test_evaluate_loss_weights_every_row_equally_whatever_the_batch_size() -> None:
    torch.manual_seed(0)
    model = MLP(3, 2, hidden=(4,))
    x, y = torch.randn(10, 3), torch.randn(10, 2)
    with torch.no_grad():
        expected = ((model.eval()(x) - y) ** 2).mean().item()
    for batch_size in (1, 3, 10, 64):  # 3 leaves a last batch of one row
        loss = evaluate_loss(model, (x, y), batch_size=batch_size)
        assert loss == pytest.approx(expected, rel=1e-5)


def test_predict_runs_in_eval_mode_in_chunks_and_restores_the_mode() -> None:
    torch.manual_seed(0)
    model = MLP(3, 2, hidden=(8,), dropout=0.5).train()
    x = torch.randn(10, 3)

    full = predict(model, (x,))
    chunked = predict(model, x, batch_size=3)  # a bare tensor is accepted too

    assert full.dtype == np.float64 and full.shape == (10, 2)
    np.testing.assert_allclose(full, chunked, rtol=1e-6)
    assert model.training
    with torch.no_grad():
        np.testing.assert_allclose(full, model.eval()(x).double().numpy(), rtol=1e-6)


# --- TorchMLPRegressor ---------------------------------------------------------------------------


def test_same_seed_gives_identical_weights_and_predictions() -> None:
    X, Y, X_val, Y_val = _linear_problem()
    cfg = TrainConfig(lr=1e-2, batch_size=16, max_epochs=15, patience=5)

    def fit(seed: int) -> TorchMLPRegressor:
        # Use every global RNG first: a fit must not depend on what ran before it.
        random.random(), np.random.rand(), torch.rand(3)  # noqa: NPY002
        reg = TorchMLPRegressor(hidden=(16, 8), dropout=0.3, train_cfg=cfg, seed=seed)
        return reg.fit(X, Y, X_val, Y_val)

    # A NumPy integer seed (e.g. from np.arange) must behave exactly like the plain int.
    first, again, other = fit(3), fit(np.arange(5)[3]), fit(4)

    assert _same_weights(first.model_, again.model_)
    assert first.history_ == again.history_
    np.testing.assert_array_equal(first.predict(X_val), again.predict(X_val))
    assert not _same_weights(first.model_, other.model_)
    assert not np.allclose(first.predict(X_val), other.predict(X_val))


def test_init_state_loads_autoencoder_trunk_and_skips_a_mismatched_head() -> None:
    X, Y, X_val, Y_val = _linear_problem(p=6, m=3)
    torch.manual_seed(1)
    autoencoder = MaskedFeatureAutoencoder(6, hidden=(8, 4))
    other_head = MLP(6, 5, hidden=(8, 4)).head  # e.g. an E4b head for another target set
    init_state = {
        **autoencoder.state_dict(),
        "head.weight": other_head.weight.detach(),
        "head.bias": other_head.bias.detach(),
    }
    cfg = TrainConfig(lr=0.0, weight_decay=0.0, max_epochs=1)  # lr 0: weights stay as loaded

    reg = TorchMLPRegressor(hidden=(8, 4), dropout=0.0, train_cfg=cfg, init_state=init_state)
    reg.fit(X, Y, X_val, Y_val)
    scratch = TorchMLPRegressor(hidden=(8, 4), dropout=0.0, train_cfg=cfg).fit(X, Y, X_val, Y_val)

    trunk_keys = sorted(k for k in autoencoder.state_dict() if k.startswith("trunk."))
    decoder_keys = [k for k in autoencoder.state_dict() if k.startswith("decoder.")]
    assert reg.loaded_init_keys_ == trunk_keys
    assert reg.skipped_init_keys_ == sorted(["head.bias", "head.weight", *decoder_keys])
    for key in trunk_keys:
        assert torch.equal(reg.model_.state_dict()[key], autoencoder.state_dict()[key])
    # Loading draws no random numbers, so the fresh head equals the from-scratch one: the
    # pretrained and scratch variants differ only in the transferred weights.
    assert _same_weights(reg.model_.head, scratch.model_.head)
    assert scratch.loaded_init_keys_ == scratch.skipped_init_keys_ == []


def test_init_state_with_matching_shapes_carries_the_head_over() -> None:
    X, Y, X_val, Y_val = _linear_problem(p=6, m=3)
    torch.manual_seed(1)
    source = MLP(6, 3, hidden=(8, 4))
    cfg = TrainConfig(lr=0.0, weight_decay=0.0, max_epochs=1)

    reg = TorchMLPRegressor(hidden=(8, 4), train_cfg=cfg, init_state=source.state_dict())
    reg.fit(X, Y, X_val, Y_val)

    assert reg.skipped_init_keys_ == []
    assert _same_weights(reg.model_, source)


def test_init_state_sharing_nothing_with_the_model_is_an_error() -> None:
    X, Y, X_val, Y_val = _linear_problem()
    reg = TorchMLPRegressor(train_cfg={"max_epochs": 1}, init_state={"encoder.w": torch.zeros(2)})
    with pytest.raises(ValueError, match="shares no parameter"):
        reg.fit(X, Y, X_val, Y_val)


def test_init_state_pretrained_on_other_input_features_is_an_error() -> None:
    # E.g. an E4a encoder fit on the source genus vocabulary while the target pipeline re-filtered
    # prevalence on PRISM: only the first-layer bias and the deeper layers would have loaded.
    X, Y, X_val, Y_val = _linear_problem(p=6)
    source = MaskedFeatureAutoencoder(9, hidden=(8, 4))
    reg = TorchMLPRegressor(
        hidden=(8, 4), train_cfg={"max_epochs": 1}, init_state=source.state_dict()
    )
    with pytest.raises(ValueError, match="other input features"):
        reg.fit(X, Y, X_val, Y_val)


def test_dataframes_and_arrays_give_the_same_fit() -> None:
    # pandas stores columns contiguously; the result must not depend on that memory layout.
    arrays = _linear_problem()
    frames = [pd.DataFrame(a) for a in arrays]
    cfg = {"max_epochs": 2, "batch_size": 64}  # a plain dict, as read from YAML

    from_frames = TorchMLPRegressor(hidden=[8], train_cfg=cfg).fit(*frames)
    from_arrays = TorchMLPRegressor(hidden=[8], train_cfg=cfg).fit(*arrays)

    prediction = from_frames.predict(frames[2])
    assert prediction.shape == arrays[3].shape and prediction.dtype == np.float64
    assert _same_weights(from_frames.model_, from_arrays.model_)
    np.testing.assert_array_equal(prediction, from_arrays.predict(arrays[2]))
    assert TorchMLPRegressor.needs_validation is True
    assert not from_frames.model_.training


def test_regressor_input_errors() -> None:
    X, Y, X_val, Y_val = _linear_problem()
    reg = TorchMLPRegressor(train_cfg=TrainConfig(max_epochs=1))
    with pytest.raises(RuntimeError, match="not fitted"):
        reg.predict(X)
    with pytest.raises(ValueError, match="validation split"):
        reg.fit(X, Y, None, None)
    with_nan = Y.copy()
    with_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        reg.fit(X, with_nan, X_val, Y_val)
    with pytest.raises(ValueError, match="2-D"):
        reg.fit(X, Y[:, 0], X_val, Y_val)
    with pytest.raises(ValueError, match="same columns"):
        reg.fit(X, Y, X_val[:, :2], Y_val)
    with pytest.raises(ValueError, match="unknown TrainConfig"):
        TorchMLPRegressor(train_cfg={"epochs": 3}).fit(X, Y, X_val, Y_val)
    reg.fit(X, Y, X_val, Y_val)
    with pytest.raises(ValueError, match="features"):
        reg.predict(X[:, :2])


def test_regressor_params_round_trip_and_sklearn_clone() -> None:
    reg = TorchMLPRegressor(hidden=[16, 8], dropout=0.2, train_cfg={"max_epochs": 2}, seed=1)
    assert reg.get_params() == {
        "hidden": [16, 8],
        "dropout": 0.2,
        "train_cfg": {"max_epochs": 2},
        "seed": 1,
        "init_state": None,
    }
    assert reg.set_params(seed=7) is reg and reg.seed == 7
    with pytest.raises(ValueError, match="unknown parameters"):
        reg.set_params(lr=0.1)

    sklearn_base = pytest.importorskip("sklearn.base")
    clone = sklearn_base.clone(reg)
    assert clone is not reg and clone.get_params() == reg.get_params()
    assert not hasattr(clone, "model_")


# --- no leakage (brief section 5) ----------------------------------------------------------------


def test_validation_rows_are_scored_but_never_reach_a_gradient_step() -> None:
    # The inner split may only decide when to stop. Each row carries its index as an extra
    # tensor, so the loss can record which rows it saw with gradients enabled.
    X, Y, X_val, Y_val = (torch.tensor(a, dtype=torch.float32) for a in _linear_problem())
    train_ids = torch.arange(len(X))
    val_ids = torch.arange(len(X), len(X) + len(X_val))
    with_grad: set[int] = set()
    without_grad: set[int] = set()

    def spy_loss(model: nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        row_ids, x, y = batch
        (with_grad if torch.is_grad_enabled() else without_grad).update(row_ids.tolist())
        return mse_loss(model, (x, y))

    torch.manual_seed(0)
    cfg = TrainConfig(lr=1e-2, batch_size=32, max_epochs=5, patience=5)
    fit_model(
        MLP(5, 3, hidden=(8,)),
        (train_ids, X, Y),
        (val_ids, X_val, Y_val),
        cfg,
        seed=0,
        loss_fn=spy_loss,
    )

    assert with_grad == set(train_ids.tolist())  # every training row and nothing else
    assert without_grad == set(val_ids.tolist())  # validation rows are only ever scored


def test_regressor_trains_on_the_training_split_and_stops_on_the_validation_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Swapping the two splits passed every other test: check what the wrapper hands fit_model.
    X, Y, X_val, Y_val = _linear_problem()
    calls: list[tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]] = []
    real_fit_model = train.fit_model

    def spy_fit_model(
        model: nn.Module,
        train_tensors: Sequence[torch.Tensor],
        val_tensors: Sequence[torch.Tensor],
        cfg: TrainConfig,
        **kwargs: Any,
    ) -> TrainHistory:
        calls.append((train_tensors, val_tensors))
        return real_fit_model(model, train_tensors, val_tensors, cfg, **kwargs)

    monkeypatch.setattr(train, "fit_model", spy_fit_model)
    TorchMLPRegressor(hidden=(8,), train_cfg={"max_epochs": 2}).fit(X, Y, X_val, Y_val)

    [(train_tensors, val_tensors)] = calls
    given = (*train_tensors, *val_tensors)
    # Targets arrive centred on the TRAINING rows' mean (fit_row_offset), never the val rows'.
    offset = Y.astype(np.float32).mean(axis=0)
    expected_arrays = (X, Y.astype(np.float32) - offset, X_val, Y_val.astype(np.float32) - offset)
    for tensor, expected in zip(given, expected_arrays, strict=True):
        np.testing.assert_allclose(tensor.numpy(), expected.astype(np.float32), rtol=0, atol=1e-6)


def test_predicting_test_rows_changes_nothing_fitted_and_scores_each_row_alone() -> None:
    # Test rows may only be scored: no weight moves, and a row's prediction does not depend on
    # which other rows are predicted with it (no batch statistics), so the test cohort's
    # composition cannot leak into any single prediction.
    X, Y, X_val, Y_val = _linear_problem()
    reg = TorchMLPRegressor(hidden=(8,), dropout=0.5, train_cfg={"max_epochs": 3}, seed=0)
    reg.fit(X, Y, X_val, Y_val)
    fitted = copy.deepcopy(reg.model_.state_dict())
    X_test = np.random.default_rng(1).normal(size=(30, X.shape[1]))

    full = reg.predict(X_test)
    first_rows = reg.predict(X_test[:4])
    reversed_rows = reg.predict(X_test[::-1])

    assert all(torch.equal(fitted[key], value) for key, value in reg.model_.state_dict().items())
    np.testing.assert_allclose(first_rows, full[:4], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(reversed_rows[::-1], full, rtol=1e-5, atol=1e-6)


def test_effective_batch_size_only_shrinks_for_small_training_sets() -> None:
    from m2m.train import effective_batch_size

    cfg = TrainConfig(batch_size=32, min_batches_per_epoch=4)
    assert effective_batch_size(124, cfg) == 32  # final PRISM fit: 4 steps already
    assert effective_batch_size(99, cfg) == 32  # CV fold: 4 steps already
    assert effective_batch_size(64, cfg) == 16
    assert effective_batch_size(16, cfg) == 4
    assert effective_batch_size(3, cfg) == 1
    assert effective_batch_size(16, TrainConfig(batch_size=32)) == 32  # default: unchanged


def test_regressor_centres_targets_on_fit_rows_and_adds_the_offset_back() -> None:
    """At n = 20, z-scoring over fit + validation rows makes the 4 validation rows' mean -4x the
    fit rows' mean; without centring, early stopping keeps an untrained network (epoch 1)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6))
    Y = X[:, :2] @ rng.normal(size=(2, 3)) + 0.1 * rng.normal(size=(20, 3))
    Y = (Y - Y.mean(0)) / Y.std(0)  # standardised on all 20 rows, as the pipeline does
    fit, val = slice(0, 16), slice(16, 20)
    reg = TorchMLPRegressor(
        hidden=(16,), dropout=0.0, train_cfg=TrainConfig(lr=1e-2, max_epochs=300, patience=30),
        seed=0,
    ).fit(X[fit], Y[fit], X[val], Y[val])
    np.testing.assert_allclose(reg.target_offset_.numpy(), Y[fit].mean(0), rtol=1e-5, atol=1e-6)
    assert reg.history_.best_epoch > 5  # it learned instead of stopping at initialisation
    raw = predict(reg.model_, (torch.as_tensor(X[val], dtype=torch.float32),))
    np.testing.assert_allclose(reg.predict(X[val]), raw + Y[fit].mean(0), rtol=1e-5, atol=1e-6)
