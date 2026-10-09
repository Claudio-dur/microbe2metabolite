"""E3 (MGM fine-tuning and its from-scratch control): the files exchanged between the two
environments, selection, scoring.

Synthetic data only. scripts/mgm_finetune.py normally runs in envs/mgm; it is imported here too,
because its reader and writer are half of the file contract, and its training loop runs on a
small stand-in for MGM's GPT-2 (transformers is not installed in the main environment). One test
runs the script's real network builder in envs/mgm on a tiny GPT-2 config (skipped without it).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from m2m import evaluate, paths, protocol, splits
from m2m.config import load_config
from m2m.experiments import common
from m2m.experiments import mgm_finetune as e3
from m2m.models import MGMRegressor

CFG = load_config(paths.CONFIGS / "e3_mgm_finetune.yaml")
N_EMBD, VOCAB, WIDTH, N_LAYER = 8, 30, 12, 3
PRETRAINED, RANDOM = e3.PRETRAINED, e3.RANDOM
N_FINETUNE_GRID, N_SCRATCH_GRID = 6, 3  # (k x backbone lr) and lr points of the real config
# The world's E2 selections: each target set has its own winning pooling.
E2_POOLING = {"T466": "last", "TALL": "mean"}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mgm_finetune_script", paths.REPO_ROOT / "scripts" / "mgm_finetune.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


# --- a stand-in for transformers.GPT2Model -------------------------------------------------------


class _Block(nn.Module):
    """Causal self-attention + MLP, called like GPT2Block: (hidden, attention_mask=additive)."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(N_EMBD)
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.mlp = nn.Linear(N_EMBD, N_EMBD)
        self.drop = nn.Dropout(p)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor]:
        q, k, v = self.qkv(self.ln(hidden)).chunk(3, dim=-1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(N_EMBD)
        causal = torch.ones(hidden.shape[1], hidden.shape[1], dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        scores = scores + attention_mask[:, 0]  # (B, 1, 1, T) -> (B, 1, T): hides padded keys
        hidden = hidden + self.drop(scores.softmax(dim=-1) @ v)
        return (hidden + self.drop(torch.tanh(self.mlp(hidden))),)


class FakeGPT2(nn.Module):
    """GPT2Model's interface as MGMRegressor and the frozen-prefix cache use it, with dropout."""

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_embd=N_EMBD, n_layer=N_LAYER)
        self.wte = nn.Embedding(VOCAB, N_EMBD)
        self.wpe = nn.Embedding(WIDTH, N_EMBD)
        self.drop = nn.Dropout(p)
        self.h = nn.ModuleList(_Block(p) for _ in range(N_LAYER))
        self.ln_f = nn.LayerNorm(N_EMBD)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        additive = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
        hidden = self.drop(self.wte(input_ids) + self.wpe(torch.arange(input_ids.shape[1])))
        states = []
        for block in self.h:
            states.append(hidden)  # as transformers: the input of every block, then the output
            hidden = block(hidden, attention_mask=additive)[0]
        hidden = self.ln_f(hidden)
        states.append(hidden)
        return SimpleNamespace(
            last_hidden_state=hidden, hidden_states=tuple(states) if output_hidden_states else None
        )


def fake_mgm(*, init: str, n_out: int, pooling: str, head_dropout: float) -> MGMRegressor:
    """Stand-in for the script's builder; its backbone is drawn from the RNG run_task seeded."""
    assert init in (PRETRAINED, RANDOM)
    return MGMRegressor(FakeGPT2(), n_out=n_out, pooling=pooling, head_dropout=head_dropout)


# --- synthetic inputs ----------------------------------------------------------------------------


def synthetic_tokens(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Right-padded <bos> genera <eos> rows, like MGM's tokenizer output."""
    lengths = rng.integers(3, WIDTH + 1, size=n)
    input_ids = np.zeros((n, WIDTH), dtype=np.int64)
    for row, length in enumerate(lengths):
        input_ids[row, :length] = rng.integers(4, VOCAB, size=length)
        input_ids[row, 0], input_ids[row, length - 1] = 2, 3
    return input_ids, (input_ids != 0).astype(np.int64)


def write_token_file(path: Path, ids: list[str], rng: np.random.Generator) -> None:
    input_ids, mask = synthetic_tokens(len(ids), rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, sample_ids=np.array(ids, dtype=str), input_ids=input_ids, attention_mask=mask
    )


def tiny_task(
    rng: np.random.Generator,
    *,
    task_id: str = "cv__demo",
    kind: str = "cv",
    init: str = PRETRAINED,
    k: int = 1,
    pooling: str = "mean",
    seed: int = 0,
    train: list[str] | None = None,
    val: list[str] | None = None,
    test: list[str] | None = None,
    n_targets: int = 3,
    max_epochs: int = 3,
    train_settings: dict[str, Any] | None = None,
) -> e3.Task:
    """A hand-made task (the experiment module builds real ones from PRISM folds). For a random
    network, `k` is the config's scratch n_trainable_blocks."""
    ids = {
        "train": train or [f"tr{i}" for i in range(24)],
        "val": val or [f"va{i}" for i in range(6)],
        "test": test or [f"te{i}" for i in range(5)],
    }
    arrays: dict[str, np.ndarray] = {}
    for split, split_ids in ids.items():
        input_ids, mask = synthetic_tokens(len(split_ids), rng)
        arrays[f"{split}_input_ids"] = input_ids
        arrays[f"{split}_attention_mask"] = mask
        arrays[f"{split}_sample_ids"] = np.array(split_ids, dtype=str)
    arrays["Y_train"] = rng.normal(size=(len(ids["train"]), n_targets)).astype(np.float32)
    arrays["Y_val"] = rng.normal(size=(len(ids["val"]), n_targets)).astype(np.float32)
    if kind == "cv":
        arrays["Y_test"] = rng.normal(size=(len(ids["test"]), n_targets))
    cfg = copy.deepcopy(CFG)
    cfg["e3"]["train"].update(max_epochs=max_epochs, patience=2, batch_size=8)
    cfg["e3"]["train"].update(train_settings or {})
    if init == PRETRAINED:
        params = {"backbone_lr": 1.0e-3, "n_trainable_blocks": k, "pooling": pooling}
    else:
        cfg["e3"]["scratch"]["n_trainable_blocks"] = k
        params = {"lr": 1.0e-3, "pooling": pooling}
    spec = {
        "task_id": task_id,
        "kind": kind,
        "target_set": "T466",
        "seed": seed,
        "n_train": len(ids["train"]) + len(ids["val"]),
        "params": params,
        **e3.hyperparameters(params, cfg, init=init),
        "targets": [f"met_{j}" for j in range(n_targets)],
        "fold": 0,
        "n_targets_dropped_zero": 0,
    }
    return e3.Task(spec, arrays)


def record_for(task: e3.Task, **extra: Any) -> dict[str, Any]:
    return {"task_id": task.task_id, "task_hash": task.spec["task_hash"], **extra}


def save_e2_selection(
    name: str, *, pooling: str, cv_mean_rho: float, target_set: str = "T466"
) -> None:
    best = {"alpha": 1.0, "pooling": pooling}
    table = pd.DataFrame(
        {"params": [json.dumps(best, sort_keys=True)], "cv_mean_rho": [cv_mean_rho]}
    )
    protocol.save_selection(name, target_set, best, table)


@pytest.fixture
def world(synthetic_cohort, rng, monkeypatch) -> SimpleNamespace:
    """PRISM = the first 45 synthetic samples, the external cohort = the last 15; E0/E2 ran."""
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    prism, external = list(X.index[:45]), list(X.index[45:])
    opened: list[str] = []

    def load_d1_test(purpose: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        opened.append(purpose)
        return X.loc[external], Y.loc[external]

    monkeypatch.delenv(common.DRY_RUN_ENV, raising=False)
    monkeypatch.setattr(
        common, "load_d1_train", lambda: (X.loc[prism], Y.loc[prism], labels.loc[prism])
    )
    monkeypatch.setattr(common, "load_d1_test", load_d1_test)
    monkeypatch.setattr(
        common, "target_candidates", lambda t: list(Y.columns[:20]) if t == "T466" else None
    )
    write_token_file(paths.DATA_PROCESSED / "mgm" / "PRISM_tokens.npz", prism, rng)
    write_token_file(paths.DATA_PROCESSED / "mgm" / "NLIBD_tokens.npz", external, rng)
    # T466: E2's ridge head ("last") wins; TALL: its MLP head ("mean") does.
    save_e2_selection("e2_mgm_frozen_ridge", pooling="last", cv_mean_rho=0.30)
    save_e2_selection("e2_mgm_frozen_mlp", pooling="mean", cv_mean_rho=0.20)
    save_e2_selection("e2_mgm_frozen_ridge", pooling="last", cv_mean_rho=0.10, target_set="TALL")
    save_e2_selection("e2_mgm_frozen_mlp", pooling="mean", cv_mean_rho=0.25, target_set="TALL")
    null = evaluate.NullResult(
        pooled_rho=np.zeros(4), threshold=0.4, per_permutation_mean=np.zeros(2), quantile=0.95
    )
    for target_set in ("T466", "TALL"):
        protocol.save_null_threshold(data="D1", target_set=target_set, null=null, n_features=20)
    return SimpleNamespace(
        cfg=copy.deepcopy(CFG), X=X, Y=Y, labels=labels, prism=prism, external=external,
        opened=opened,
    )


def write_cv_predictions(round_dir: Path, rng: np.random.Generator, good: dict) -> None:
    """Fold truth + noise; tasks whose params match `good` get far less noise than the others.
    Only the keys a task's params have count, so {"lr": ...} picks a scratch point and
    {"n_trainable_blocks": ..., "backbone_lr": ...} a fine-tuning one."""
    for task_id in json.loads((round_dir / e3.MANIFEST).read_text())["tasks"]:
        task = e3.read_task(round_dir, task_id)
        params = task.spec["params"]
        matches = all(params[key] == value for key, value in good.items() if key in params)
        noise = 0.1 if matches else 3.0
        truth = task.arrays["Y_test"]
        preds = truth + noise * rng.normal(size=truth.shape)
        script.write_predictions(round_dir, task_id, task.ids("test"), preds, record_for(task))


def task_ids(round_dir: Path) -> list[str]:
    return json.loads((round_dir / e3.MANIFEST).read_text())["tasks"]


# --- the file contract ---------------------------------------------------------------------------


def test_task_round_trip_between_the_two_environments(tmp_path, rng):
    for init in (PRETRAINED, RANDOM):
        task = tiny_task(rng, init=init, task_id=f"cv__demo_{init}", k=N_LAYER)
        assert e3.write_task(tmp_path, task) is True
        assert e3.write_task(tmp_path, task) is False  # identical content: left untouched

        spec, arrays = script.load_task(tmp_path, task.task_id)  # the envs/mgm reader

        assert set(arrays) == set(task.arrays)
        for key, values in task.arrays.items():
            np.testing.assert_array_equal(arrays[key], values)
            assert arrays[key].dtype == values.dtype
        assert {key: spec[key] for key in task.spec} == json.loads(json.dumps(task.spec))
        assert spec["format_version"] == e3.FORMAT_VERSION == script.FORMAT_VERSION
        assert spec["init"] == init and e3.INITS == script.INITS
        assert spec["task_hash"] == e3.content_hash(
            {k: v for k, v in spec.items() if k != "task_hash"}, task.arrays
        )


def test_both_environments_refuse_an_unknown_init(tmp_path, rng, monkeypatch):
    task = tiny_task(rng)
    task.spec["init"] = "imagenet"
    with pytest.raises(ValueError, match="init must be one of"):
        e3.write_task(tmp_path, task)
    monkeypatch.setattr(e3, "check_task", lambda task: None)  # a writer without the guard
    e3.write_task(tmp_path, task)
    with pytest.raises(script.TaskError, match="init must be one of"):
        script.load_task(tmp_path, task.task_id)
    with pytest.raises(ValueError, match="init must be one of"):
        e3.hyperparameters({"lr": 1e-3, "pooling": "mean"}, CFG, init="imagenet")


def test_both_environments_refuse_test_ids_that_overlap_training(tmp_path, rng, monkeypatch):
    leaky_test = tiny_task(rng, train=["a", "b", "c"], val=["d", "e"], test=["e", "f"])
    leaky_val = tiny_task(rng, train=["a", "b", "c"], val=["c", "d"], test=["f", "g"])
    for task in (leaky_test, leaky_val):
        with pytest.raises(splits.LeakageError):
            e3.write_task(tmp_path, task)

    monkeypatch.setattr(e3, "check_task", lambda task: None)  # a writer without the guard
    for task in (leaky_test, leaky_val):
        e3.write_task(tmp_path, task)
        with pytest.raises(script.TaskError, match="overlap"):
            script.load_task(tmp_path, task.task_id)


def test_external_targets_never_enter_a_task_file(tmp_path, rng):
    final = tiny_task(rng, kind="final")
    final.arrays["Y_test"] = rng.normal(size=(5, 3))  # as if someone attached the test truth
    with pytest.raises(ValueError, match="only CV tasks"):
        e3.write_task(tmp_path, final)


def test_predictions_are_matched_to_the_truth_by_sample_id(tmp_path, rng):
    e3.write_task(tmp_path, tiny_task(rng))
    task = e3.read_task(tmp_path, "cv__demo")
    preds = task.arrays["Y_test"] + 0.5 * rng.normal(size=task.arrays["Y_test"].shape)
    test_ids = task.ids("test")

    def fold_rho(order: list[int]) -> pd.Series:
        rows = [test_ids[i] for i in order]
        script.write_predictions(tmp_path, task.task_id, rows, preds[order], record_for(task))
        return e3.cv_fold_score(CFG, tmp_path, task, params_index=0).rho

    pd.testing.assert_series_equal(fold_rho([0, 1, 2, 3, 4]), fold_rho([4, 3, 2, 1, 0]))

    foreign = [*test_ids[:-1], "someone_else"]
    script.write_predictions(tmp_path, task.task_id, foreign, preds, record_for(task))
    with pytest.raises(ValueError, match="not the task's test samples"):
        e3.read_predictions(tmp_path, task)

    stale = {**record_for(task), "task_hash": "an-older-version"}
    script.write_predictions(tmp_path, task.task_id, test_ids, preds, stale)
    with pytest.raises(e3.StalePredictionsError):
        e3.read_predictions(tmp_path, task)


# --- the MGM-side training loop (stand-in backbone) ----------------------------------------------


def test_script_resumes_and_reruns_only_missing_or_changed_tasks(tmp_path, rng):
    tasks = [tiny_task(rng, task_id=f"cv__demo{i}") for i in range(2)]
    e3.write_round(tmp_path, tasks, {"round": "cv"}, CFG)
    built: list[dict[str, Any]] = []

    def counting_build(**kwargs: Any) -> MGMRegressor:
        built.append(kwargs)
        return fake_mgm(**kwargs)

    def run() -> list[str]:
        return script.run_round(
            tmp_path, workers=1, threads=1, model_dir=None, build=counting_build
        )

    assert run() == [] and len(built) == 2
    assert e3.missing_predictions(tmp_path, ["cv__demo0", "cv__demo1"]) == []
    assert run() == [] and len(built) == 2  # nothing pending

    kept = tmp_path / "preds" / "cv__demo1.npz"
    before = kept.stat().st_mtime_ns
    (tmp_path / "preds" / "cv__demo0.json").unlink()
    assert run() == [] and len(built) == 3
    assert kept.stat().st_mtime_ns == before

    changed = tiny_task(np.random.default_rng(99), task_id="cv__demo1")
    e3.write_round(tmp_path, [tasks[0], changed], {"round": "cv"}, CFG)
    assert e3.missing_predictions(tmp_path, ["cv__demo0", "cv__demo1"]) == ["cv__demo1"]
    assert run() == [] and len(built) == 4


def test_a_failing_task_is_reported_while_the_others_finish(tmp_path, rng):
    tasks = [tiny_task(rng, task_id="cv__breaks", n_targets=4), tiny_task(rng, task_id="cv__works")]
    e3.write_round(tmp_path, tasks, {"round": "cv"}, CFG)

    def build(**kwargs: Any) -> MGMRegressor:
        if kwargs["n_out"] == 4:
            raise RuntimeError("simulated failure")
        return fake_mgm(**kwargs)

    failed = script.run_round(tmp_path, workers=1, threads=1, model_dir=None, build=build)

    assert failed == ["cv__breaks"]
    assert e3.missing_predictions(tmp_path, ["cv__breaks", "cv__works"]) == ["cv__breaks"]


def test_script_refuses_more_threads_than_cpus(tmp_path):
    threads_before = torch.get_num_threads()  # main() sets the process-wide thread count
    try:
        with pytest.raises(SystemExit):
            script.main(["--round-dir", str(tmp_path), "--workers", "1000", "--threads", "1"])
    finally:
        torch.set_num_threads(threads_before)


@pytest.mark.parametrize(("k", "pooling"), [(1, "mean"), (2, "last")])
def test_cached_frozen_prefix_trains_like_the_full_model(tmp_path, rng, k, pooling):
    task = tiny_task(rng, k=k, pooling=pooling, max_epochs=4)
    outputs = {}
    for use_cache in (True, False):
        round_dir = tmp_path / f"cache_{use_cache}"
        e3.write_round(round_dir, [task], {"round": "cv"}, CFG)
        record = script.run_task(
            round_dir, task.task_id, model_dir=None, use_cache=use_cache, build=fake_mgm
        )
        written = e3.read_task(round_dir, task.task_id)
        outputs[use_cache] = (record, e3.read_predictions(round_dir, written))

    cached, full = outputs[True][0], outputs[False][0]
    assert cached["prefix_cache"] and not full["prefix_cache"]
    assert set(cached["cache_check_max_abs_diff"]) == {"before_training", "test_predictions"}
    assert max(cached["cache_check_max_abs_diff"].values()) <= script.DEFAULT_CACHE_TOL
    # Same parameters, same dropout draws, same batches: the same fit up to float rounding.
    assert cached["epochs_run"] == full["epochs_run"]
    np.testing.assert_allclose(cached["val_loss"], full["val_loss"], rtol=1e-4)
    np.testing.assert_allclose(outputs[True][1], outputs[False][1], atol=1e-4)


def test_scratch_task_trains_the_whole_random_network_with_one_learning_rate(tmp_path, rng):
    task = tiny_task(rng, init=RANDOM, k=N_LAYER, max_epochs=3)
    assert task.spec["backbone_lr"] == task.spec["head_lr"] == task.spec["params"]["lr"]
    e3.write_round(tmp_path, [task], {"round": "cv"}, CFG)
    built: list[dict[str, Any]] = []
    models: list[MGMRegressor] = []

    def build(**kwargs: Any) -> MGMRegressor:
        built.append(kwargs)
        models.append(fake_mgm(**kwargs))
        return models[-1]

    record = script.run_task(tmp_path, task.task_id, model_dir=None, build=build)

    assert [kwargs["init"] for kwargs in built] == [RANDOM]
    assert record["init"] == RANDOM and not record["prefix_cache"]  # nothing frozen to cache
    model = models[0]
    assert record["n_trainable_blocks"] == model.n_layer
    assert all(p.requires_grad for p in model.parameters())
    assert record["n_trainable_parameters"] == sum(p.numel() for p in model.parameters())


def test_scratch_tasks_are_reproducible_and_their_seed_sets_the_initial_network(tmp_path, rng):
    """run_task seeds before building, so a random network depends on the task seed only."""
    initial: dict[str, dict[str, torch.Tensor]] = {}

    def build(**kwargs: Any) -> MGMRegressor:
        model = fake_mgm(**kwargs)
        initial[current] = copy.deepcopy(model.backbone.state_dict())
        return model

    predictions = {}
    for current, seed in (("first", 0), ("again", 0), ("other", 1)):
        task = tiny_task(np.random.default_rng(5), init=RANDOM, k=N_LAYER, seed=seed)
        round_dir = tmp_path / current
        e3.write_round(round_dir, [task], {"round": "cv"}, CFG)
        script.run_task(round_dir, task.task_id, model_dir=None, build=build)
        predictions[current] = e3.read_predictions(round_dir, e3.read_task(round_dir, task.task_id))

    def same(a: str, b: str) -> bool:
        return all(torch.equal(initial[a][key], initial[b][key]) for key in initial[a])

    assert same("first", "again") and not same("first", "other")
    pd.testing.assert_frame_equal(predictions["first"], predictions["again"])


def test_script_refuses_a_random_network_with_frozen_blocks(tmp_path, rng):
    task = tiny_task(rng, init=RANDOM, k=1)  # e.g. a config that set scratch k below the layers
    e3.write_round(tmp_path, [task], {"round": "cv"}, CFG)
    with pytest.raises(script.TaskError, match="must train all 3 blocks"):
        script.run_task(tmp_path, task.task_id, model_dir=None, build=fake_mgm)
    assert e3.missing_predictions(tmp_path, [task.task_id]) == [task.task_id]


@pytest.mark.parametrize(("init", "k"), [(PRETRAINED, 1), (RANDOM, N_LAYER)])
def test_script_centres_targets_on_the_fit_rows_and_adds_the_offset_back(
    tmp_path, rng, monkeypatch, init, k
):
    """Both models (MGM with its cached prefix, the scratch network without) train and early-stop
    on targets minus the fit rows' mean, as TorchMLPRegressor does; predictions get it back."""
    task = tiny_task(rng, init=init, k=k, max_epochs=2)
    task.arrays["Y_train"] = task.arrays["Y_train"] + np.float32(3.0)  # a mean worth removing
    e3.write_round(tmp_path, [task], {"round": "cv"}, CFG)
    seen: dict[str, torch.Tensor] = {}
    real_fit = script.fit_model

    def recording_fit(model, train, val, cfg, **kwargs):
        seen["train"], seen["val"] = train[-1].clone(), val[-1].clone()
        return real_fit(model, train, val, cfg, **kwargs)

    models: list[MGMRegressor] = []

    def build(**kwargs: Any) -> MGMRegressor:
        models.append(fake_mgm(**kwargs))
        return models[-1]

    monkeypatch.setattr(script, "fit_model", recording_fit)
    record = script.run_task(tmp_path, task.task_id, model_dir=None, build=build)

    assert record["prefix_cache"] == (init == PRETRAINED)
    offset = task.arrays["Y_train"].mean(axis=0)
    np.testing.assert_allclose(seen["train"].numpy(), task.arrays["Y_train"] - offset, atol=1e-5)
    np.testing.assert_allclose(seen["val"].numpy(), task.arrays["Y_val"] - offset, atol=1e-5)
    test_tokens = script.trimmed(task.arrays["test_input_ids"], task.arrays["test_attention_mask"])
    raw = script.predict(models[0], test_tokens, script.PREDICT_BATCH_SIZE)
    written = e3.read_predictions(tmp_path, e3.read_task(tmp_path, task.task_id))
    np.testing.assert_allclose(written.to_numpy(), raw + offset, rtol=1e-5, atol=1e-5)


def test_every_training_setting_reaches_envs_mgm_or_is_refused(tmp_path, rng):
    # 24 fit rows in batches of 8 is 3 steps per epoch; asking for at least 4 shrinks the batch.
    for settings, used in (({"min_batches_per_epoch": 1}, 8), ({"min_batches_per_epoch": 4}, 6)):
        task = tiny_task(rng, train_settings=settings)
        round_dir = tmp_path / f"min_batches_{used}"
        e3.write_round(round_dir, [task], {"round": "cv"}, CFG)
        record = script.run_task(round_dir, task.task_id, model_dir=None, build=fake_mgm)
        assert record["batch_size_used"] == used

    cfg = copy.deepcopy(CFG)
    cfg["e3"]["train"]["lr"] = 0.5  # would be validated, then ignored: the grid sets the lrs
    params = {"backbone_lr": 1.0e-5, "n_trainable_blocks": 1, "pooling": "mean"}
    with pytest.raises(ValueError, match="not passed to envs/mgm"):
        e3.hyperparameters(params, cfg, init=PRETRAINED)


def test_cache_check_stops_a_task_whose_cached_path_is_wrong(tmp_path, rng, monkeypatch):
    task = tiny_task(rng, k=1)
    e3.write_round(tmp_path, [task], {"round": "cv"}, CFG)
    exact = script.frozen_prefix
    monkeypatch.setattr(script, "frozen_prefix", lambda *args: exact(*args) + 0.1)

    with pytest.raises(RuntimeError, match="differs from the full model"):
        script.run_task(tmp_path, task.task_id, model_dir=None, build=fake_mgm)
    assert e3.missing_predictions(tmp_path, [task.task_id]) == [task.task_id]


# --- the experiment stages -----------------------------------------------------------------------


def test_prepare_cv_writes_one_task_per_model_grid_point_fold_and_target_set(world, monkeypatch):
    round_dir = e3.prepare_cv(world.cfg)

    manifest = json.loads((round_dir / e3.MANIFEST).read_text())
    assert len(manifest["tasks"]) == (N_FINETUNE_GRID + N_SCRATCH_GRID) * 5 * 2  # x folds x sets
    # Each target set takes the pooling of its own E2 selection (DESIGN decision 7).
    assert manifest["pooling"] == E2_POOLING
    for init, by_target in manifest["grids"].items():
        for target_set, grid in by_target.items():
            assert {params["pooling"] for params in grid} == {E2_POOLING[target_set]}, init

    task = e3.read_task(round_dir, "cv__T466__finetune__k2__blr3e-05__fold1__seed0")
    folds = common.d1_folds(world.labels.loc[world.prism], world.cfg)
    train_ids, test_ids = splits.fold_train_test(folds, 1)
    assert task.ids("test") == test_ids
    assert set(task.ids("train")) | set(task.ids("val")) == set(train_ids)
    # Targets are standardised with statistics of the fold's training rows only.
    prep = common.metabolite_prep(world.cfg, "T466").fit(world.Y.loc[train_ids])
    np.testing.assert_allclose(
        task.arrays["Y_test"], prep.transform(world.Y.loc[test_ids]).to_numpy()
    )
    np.testing.assert_allclose(
        task.arrays["Y_train"], prep.transform(world.Y.loc[task.ids("train")]), rtol=1e-6
    )

    # Early stopping uses the inner split protocol.M2MPipeline (i.e. E1) carves for E1's CV seed.
    assert load_config(paths.CONFIGS / "e1_mlp.yaml")["cv_seed"] == e3.CV_SEED
    carved: list[tuple[list[str], list[str]]] = []
    real_split = splits.stratified_inner_split

    def recording_split(*args: Any, **kwargs: Any) -> tuple[list[str], list[str]]:
        carved.append(real_split(*args, **kwargs))
        return carved[-1]

    class Idle:
        needs_validation = True

        def fit(self, X, Y, X_val=None, Y_val=None) -> Idle:
            return self

    monkeypatch.setattr(splits, "stratified_inner_split", recording_split)
    protocol.make_pipeline(
        common.microbe_prep(world.cfg), common.metabolite_prep(world.cfg, "T466"), Idle(),
        world.cfg, seed=e3.CV_SEED,
    ).fit(world.X.loc[train_ids], world.Y.loc[train_ids], world.labels.loc[train_ids])
    assert carved == [(task.ids("train"), task.ids("val"))]


def test_the_scratch_control_differs_from_finetuning_only_in_its_weights_and_rates(world):
    """Same folds, inner splits, tokens, targets and pooling; a random, fully trained network
    with one learning rate from its own grid."""
    round_dir = e3.prepare_cv(world.cfg)
    scratch = world.cfg["e3"]["scratch"]
    # Its own rates: the fine-tuning rates (1e-5..3e-5) cannot train a network from scratch.
    assert not set(scratch["grid"]["lr"]) & set(world.cfg["e3"]["grid"]["backbone_lr"])
    shared = ("seed", "weight_decay", "head_dropout", "pooling", "batch_size", "max_epochs",
              "patience", "min_delta", "min_batches_per_epoch", "targets")
    for target_set in ("T466", "TALL"):
        reference = e3.read_task(
            round_dir, f"cv__{target_set}__finetune__k8__blr1e-05__fold3__seed0"
        )
        for lr in scratch["grid"]["lr"]:
            task = e3.read_task(round_dir, f"cv__{target_set}__scratch__lr{lr:g}__fold3__seed0")
            assert task.spec["init"] == RANDOM and reference.spec["init"] == PRETRAINED
            assert task.spec["params"] == {"lr": lr, "pooling": E2_POOLING[target_set]}
            assert task.spec["backbone_lr"] == task.spec["head_lr"] == lr
            assert task.spec["n_trainable_blocks"] == scratch["n_trainable_blocks"]
            assert set(task.arrays) == set(reference.arrays)
            for key, values in reference.arrays.items():
                np.testing.assert_array_equal(task.arrays[key], values, err_msg=key)
            assert {k: task.spec[k] for k in shared} == {k: reference.spec[k] for k in shared}


def test_prepare_cv_fails_clearly_without_e2_or_with_foreign_tokens(world, rng):
    (paths.RESULTS / "selected" / "e2_mgm_frozen_mlp__TALL.json").unlink()
    with pytest.raises(FileNotFoundError, match="pooling from E2"):
        e3.prepare_cv(world.cfg)

    save_e2_selection("e2_mgm_frozen_mlp", pooling="mean", cv_mean_rho=0.2, target_set="TALL")
    write_token_file(paths.DATA_PROCESSED / "mgm" / "PRISM_tokens.npz", world.prism[:-1], rng)
    with pytest.raises(ValueError, match="not exactly the 45 PRISM samples"):
        e3.prepare_cv(world.cfg)


def test_pooling_from_refuses_the_retired_single_target_set_key(world):
    cfg = copy.deepcopy(world.cfg)
    cfg["e3"]["pooling_from"]["target_set"] = "T466"  # the old rule: T466's pooling everywhere
    with pytest.raises(ValueError, match="each target set takes the pooling"):
        e3.prepare_cv(cfg)


@pytest.mark.parametrize(
    ("path", "key", "value"),
    [
        (("scratch",), "weight_decay", 0.5),  # a scratch-only override nothing would read
        (("scratch", "grid"), "head_lr", [1.0e-3]),
        (("grid",), "pooling", ["mean"]),  # the pooling always comes from E2
    ],
)
def test_model_grids_refuse_settings_that_would_be_ignored(world, path, key, value):
    for init in (PRETRAINED, RANDOM):  # the real config's grids pass
        assert e3.model_grid(CFG, init, "mean")
    cfg = copy.deepcopy(world.cfg)
    section = cfg["e3"]
    for part in path:
        section = section[part]
    section[key] = value
    with pytest.raises(ValueError, match=rf"\['{key}'\] would be ignored"):
        e3.prepare_cv(cfg)
    assert not e3.round_path(e3.CV_ROUND).exists()  # refused before any file is written


def test_run_stage_calls_the_mgm_interpreter_and_requires_every_prediction(
    world, rng, monkeypatch, tmp_path
):
    round_dir = e3.prepare_cv(world.cfg)
    # As in a venv: bin/python is a symlink to the base interpreter, and only the symlink's own
    # path selects the venv's packages, so the command must not resolve it.
    base = tmp_path / "base" / "python3.11"
    base.parent.mkdir()
    base.touch()
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(base)
    cfg = {**world.cfg, "mgm": {**world.cfg["mgm"], "python": str(python)}}
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(e3.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="90 of 90 cv tasks have no up-to-date predictions"):
        e3.run_mgm_round(cfg, e3.CV_ROUND, n_jobs=4)
    command = commands[0]
    assert command[:3] == [str(python), "-I", str(e3.SCRIPT.resolve())]
    assert command[command.index("--round-dir") + 1] == str(round_dir.resolve())
    assert command[command.index("--workers") + 1] == "4"  # n_jobs caps the workers
    assert command[command.index("--threads") + 1] == "2"
    monkeypatch.setattr(e3.os, "cpu_count", lambda: 64)  # independent of this machine's cores
    many = e3.mgm_command(cfg, round_dir, n_jobs=64)
    assert many[many.index("--workers") + 1] == "6"  # ... but never raises them
    greedy = {**cfg, "mgm": {**cfg["mgm"], "threads": 4}}
    with pytest.raises(ValueError, match="max_total_threads"):
        e3.mgm_command(greedy, round_dir, n_jobs=8)
    monkeypatch.setattr(e3.os, "cpu_count", lambda: 4)  # a small machine: 4 cores / 2 threads
    small = e3.mgm_command(cfg, round_dir, n_jobs=64)
    assert small[small.index("--workers") + 1] == "2"

    write_cv_predictions(round_dir, rng, good={})
    e3.run_mgm_round(cfg, e3.CV_ROUND, n_jobs=4)  # every task answered: no error

    # The interpreter runs from the repository root: a relative data directory must not reach it.
    monkeypatch.setattr(paths, "DATA_PROCESSED", Path("relative") / "processed")
    relative = e3.mgm_command(cfg, e3.round_path(e3.CV_ROUND), n_jobs=1)
    assert Path(relative[relative.index("--round-dir") + 1]).is_absolute()


def test_select_picks_the_best_grid_point_of_each_model_and_reads_prism_only(
    world, rng, monkeypatch
):
    round_dir = e3.prepare_cv(world.cfg)
    good = {"n_trainable_blocks": 2, "backbone_lr": 3.0e-5, "lr": 3.0e-4}
    write_cv_predictions(round_dir, rng, good)
    token_files: list[str] = []
    real_load = e3.Tokens.load

    def recording_load(path: Path) -> e3.Tokens:
        token_files.append(Path(path).name)
        return real_load(path)

    monkeypatch.setattr(e3.Tokens, "load", recording_load)

    e3.run(world.cfg, stage="select")

    for target_set in ("T466", "TALL"):
        pooling = E2_POOLING[target_set]
        finetune = protocol.load_selection("e3_mgm_finetune", target_set)
        assert finetune["params"] == {"n_trainable_blocks": 2, "backbone_lr": 3.0e-5,
                                      "pooling": pooling}
        scratch = protocol.load_selection("e3_mgm_scratch", target_set)
        assert scratch["params"] == {"lr": 3.0e-4, "pooling": pooling}
        for name, n_points in (("e3_mgm_finetune", N_FINETUNE_GRID),
                               ("e3_mgm_scratch", N_SCRATCH_GRID)):
            table = pd.read_csv(paths.RESULTS / "selected" / f"{name}__{target_set}__cv.csv")
            assert len(table) == n_points
            for column in ("params", "cv_mean_rho", "cv_mean_rho_sd_over_folds", "fold4_mean_rho"):
                assert column in table.columns
    runs = pd.read_csv(paths.RESULTS / "runs.csv")
    assert runs.groupby("model")["data"].unique().map(list).to_dict() == {
        "mgm_finetune": ["D1+MGM"], "mgm_scratch": ["D1+MGM-arch"],
    }
    assert len(runs) == 2 * (N_FINETUNE_GRID + N_SCRATCH_GRID) and set(runs["stage"]) == {"cv"}
    # A selection stage: nothing of the external cohort is read, not even its token ids.
    assert world.opened == [] and token_files == []
    assert not (paths.RESULTS / "external_access_log.csv").exists()
    assert not (e3.round_path(e3.FINAL_ROUND) / e3.MANIFEST).exists()


def test_prepare_final_writes_the_final_round_from_the_saved_selections(world, rng, frozen):
    round_dir = e3.prepare_cv(world.cfg)
    write_cv_predictions(
        round_dir, rng, good={"n_trainable_blocks": 2, "backbone_lr": 3.0e-5, "lr": 1.0e-3}
    )
    e3.select(world.cfg)

    e3.run(world.cfg, stage="prepare-final")

    final_dir = e3.round_path(e3.FINAL_ROUND)
    tasks = [e3.read_task(final_dir, t) for t in task_ids(final_dir)]
    for init in (PRETRAINED, RANDOM):
        kinds = [task.spec["kind"] for task in tasks if task.spec["init"] == init]
        # 5 seeds x 2 target sets; curve sizes 20, 40, 45 x 5 seeds
        assert kinds.count("final") == 2 * 5 and kinds.count("curve") == 3 * 5
    expected = {
        PRETRAINED: {"n_trainable_blocks": 2, "backbone_lr": 3.0e-5},
        RANDOM: {"lr": 1.0e-3},
    }
    for task in tasks:
        assert "Y_test" not in task.arrays
        assert task.ids("test") == world.external
        pooling = E2_POOLING[task.spec["target_set"]]
        assert task.spec["params"] == {**expected[task.spec["init"]], "pooling": pooling}
    # Both models of one seed and target set train on the same rows, targets and preprocessing.
    by_id = {task.task_id: task for task in tasks}
    for pair in (("final__TALL__finetune__seed3", "final__TALL__scratch__seed3"),
                 ("curve__T466__finetune__n20__seed1", "curve__T466__scratch__n20__seed1")):
        finetune, scratch = (by_id[task_id] for task_id in pair)
        assert finetune.spec["prep_file"] == scratch.spec["prep_file"]
        for key in ("train_sample_ids", "val_sample_ids", "Y_train", "Y_val"):
            np.testing.assert_array_equal(finetune.arrays[key], scratch.arrays[key])
    assert world.opened == []  # no external metabolites before evaluate ...
    access = pd.read_csv(paths.RESULTS / "external_access_log.csv")
    assert list(access["purpose"]) == [e3.TEST_TOKENS_PURPOSE]  # ... and the token read is logged


def test_prepare_final_checks_the_selection_before_reading_the_external_tokens(world, rng):
    round_dir = e3.prepare_cv(world.cfg)
    write_cv_predictions(round_dir, rng, good={})
    e3.select(world.cfg)
    # A selection that the current CV round never evaluated (e.g. left over from an older grid).
    stale = {"n_trainable_blocks": 4, "backbone_lr": 1.0e-4, "pooling": "last"}
    table = pd.DataFrame({"params": [json.dumps(stale, sort_keys=True)], "cv_mean_rho": [0.9]})
    protocol.save_selection("e3_mgm_finetune", "TALL", stale, table)

    with pytest.raises(ValueError, match="not a point of the current CV round"):
        e3.prepare_final(world.cfg)
    assert not (paths.RESULTS / "external_access_log.csv").exists()

    (paths.RESULTS / "selected" / "e3_mgm_finetune__TALL.json").unlink()
    with pytest.raises(FileNotFoundError, match="run model selection"):
        e3.prepare_final(world.cfg)
    assert not (paths.RESULTS / "external_access_log.csv").exists()

    e3.select(world.cfg)  # both models selected again ...
    (paths.RESULTS / "selected" / "e3_mgm_scratch__T466.json").unlink()  # ... minus the control
    with pytest.raises(FileNotFoundError, match="e3_mgm_scratch"):
        e3.prepare_final(world.cfg)
    assert not (paths.RESULTS / "external_access_log.csv").exists()


def test_select_refuses_a_cv_round_trained_with_a_pooling_e2_no_longer_chooses(world, rng):
    round_dir = e3.prepare_cv(world.cfg)  # TALL: E2's MLP ("mean") had the higher CV rho
    write_cv_predictions(round_dir, rng, good={})
    save_e2_selection("e2_mgm_frozen_ridge", pooling="last", cv_mean_rho=0.50, target_set="TALL")

    with pytest.raises(RuntimeError, match="re-run prepare-cv"):
        e3.select(world.cfg)
    assert not (paths.RESULTS / "selected" / "e3_mgm_finetune__T466.json").exists()


def test_run_final_refuses_a_round_written_for_another_selection(
    world, rng, monkeypatch, frozen
):
    round_dir = e3.prepare_cv(world.cfg)
    finetune = {"n_trainable_blocks": 1, "backbone_lr": 1.0e-5}  # stays selected throughout
    write_cv_predictions(round_dir, rng, good={**finetune, "lr": 1.0e-4})
    e3.select(world.cfg)
    e3.prepare_final(world.cfg)
    # select re-run on other CV predictions (the control now prefers another lr), without
    # prepare-final.
    write_cv_predictions(round_dir, rng, good={**finetune, "lr": 1.0e-3})
    e3.select(world.cfg)
    monkeypatch.setattr(e3.subprocess, "run", lambda *a, **k: pytest.fail("MGM was started"))

    with pytest.raises(RuntimeError, match="mgm_scratch .* re-run prepare-final"):
        e3.run_mgm_round(world.cfg, e3.FINAL_ROUND, n_jobs=1)


def test_evaluate_scores_oracle_predictions_written_in_shuffled_order(world, rng, frozen):
    round_dir = e3.prepare_cv(world.cfg)
    write_cv_predictions(
        round_dir, rng, good={"n_trainable_blocks": 1, "backbone_lr": 1.0e-5, "lr": 3.0e-4}
    )
    e3.select(world.cfg)
    e3.prepare_final(world.cfg)
    final_dir = e3.round_path(e3.FINAL_ROUND)
    for task_id in task_ids(final_dir):
        task = e3.read_task(final_dir, task_id)
        truth = e3.load_prep(final_dir, task).transform(world.Y.loc[world.external])
        rows = list(reversed(world.external))
        script.write_predictions(final_dir, task_id, rows, truth.loc[rows], record_for(task))

    e3.evaluate_final(world.cfg)

    assert world.opened == [e3.EXTERNAL_PURPOSE]  # opened once, for this final evaluation
    summary = pd.read_csv(paths.RESULTS / "summary.csv").set_index(["model", "target_set"])
    assert sorted(summary.index) == [
        (model, t) for model in ("mgm_finetune", "mgm_scratch") for t in ("T466", "TALL")
    ]
    assert summary["data"].to_dict() == {
        **{("mgm_finetune", t): "D1+MGM" for t in ("T466", "TALL")},
        **{("mgm_scratch", t): "D1+MGM-arch" for t in ("T466", "TALL")},
    }
    for (model, target_set), row in summary.iterrows():
        assert row["mean_rho_mean"] == pytest.approx(1.0) and row["n_seeds"] == 5
        name = "e3_mgm_finetune" if model == "mgm_finetune" else "e3_mgm_scratch"
        selection = protocol.load_selection(name, target_set)
        assert row["cv_mean_rho"] == pytest.approx(selection["cv_mean_rho"])
        assert json.loads(row["params"]) == selection["params"]
    curve = pd.read_csv(paths.RESULTS / "learning_curve.csv")
    assert len(curve) == 2 * 15 and set(curve["n_train"]) == {20, 40, 45}
    assert curve.groupby("model")["data"].unique().map(list).to_dict() == {
        "mgm_finetune": ["D1+MGM"], "mgm_scratch": ["D1+MGM-arch"],
    }
    assert np.allclose(curve["mean_rho"], 1.0)


def test_evaluate_checks_every_input_before_opening_the_external_cohort(world, rng, frozen):
    round_dir = e3.prepare_cv(world.cfg)
    write_cv_predictions(round_dir, rng, good={})
    e3.select(world.cfg)
    final_dir = e3.prepare_final(world.cfg)
    for task_id in task_ids(final_dir):
        task = e3.read_task(final_dir, task_id)
        preds = rng.normal(size=(len(task.ids("test")), len(task.spec["targets"])))
        script.write_predictions(final_dir, task_id, task.ids("test"), preds, record_for(task))

    (paths.RESULTS / protocol.NULL_THRESHOLDS_CSV).unlink()  # E0's null stage has not run
    with pytest.raises(FileNotFoundError, match="permutation null"):
        e3.evaluate_final(world.cfg)
    stale = e3.read_task(final_dir, "final__T466__scratch__seed3")
    script.write_predictions(
        final_dir, stale.task_id, stale.ids("test"),
        rng.normal(size=(len(stale.ids("test")), len(stale.spec["targets"]))),
        {**record_for(stale), "task_hash": "an-older-version"},
    )
    with pytest.raises(FileNotFoundError, match="run the run-final stage"):
        e3.evaluate_final(world.cfg)
    assert world.opened == []


def test_dry_run_uses_prism_pseudo_test_rows_and_never_the_external_tokens(
    world, rng, monkeypatch
):
    pseudo_train, pseudo_test = world.prism[:33], world.prism[33:]
    monkeypatch.setenv(common.DRY_RUN_ENV, "1")
    monkeypatch.setattr(common, "dry_run_ids", lambda: (pseudo_train, pseudo_test))
    monkeypatch.setattr(
        common,
        "load_d1_train",
        lambda: (world.X.loc[pseudo_train], world.Y.loc[pseudo_train],
                 world.labels.loc[pseudo_train]),
    )
    (paths.DATA_PROCESSED / "mgm" / "NLIBD_tokens.npz").unlink()  # a dry run must not need it
    cfg = copy.deepcopy(world.cfg)
    cfg["e3"]["grid"] = {"n_trainable_blocks": [1], "backbone_lr": [1.0e-5]}
    cfg["e3"]["scratch"]["grid"] = {"lr": [1.0e-3]}

    round_dir = e3.prepare_cv(cfg)  # the PRISM token file covers pseudo-train + pseudo-test
    write_cv_predictions(round_dir, rng, good={})
    e3.select(cfg)
    final_dir = e3.prepare_final(cfg)

    manifest = json.loads((final_dir / e3.MANIFEST).read_text())
    assert manifest["dry_run"] is True and manifest["test_sample_ids"] == pseudo_test
    curve = [t for t in manifest["tasks"] if t.startswith("curve")]
    assert {t.split("__n")[1].split("__seed")[0] for t in curve} == {"20", "33"}  # capped at 33
    assert {t.split("__")[2] for t in curve} == {"finetune", "scratch"}
    assert not (paths.RESULTS / "external_access_log.csv").exists()
    monkeypatch.delenv(common.DRY_RUN_ENV)
    with pytest.raises(RuntimeError, match="M2M_DRY_RUN on"):
        e3.read_manifest(final_dir, e3.FINAL_ROUND, cfg)


# --- the real network builder, in envs/mgm ---------------------------------------------------

MGM_PYTHON = paths.REPO_ROOT / "envs" / "mgm" / ".venv" / "bin" / "python"
# Runs in envs/mgm (transformers lives only there). The scratch network is built as run_task
# builds it (set_seed, then mgm_builder); E2's random_backbone of the same seed must equal it.
SCRATCH_INIT_CHECK = """
import importlib.util, json, sys
import torch
from transformers import GPT2Config

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

finetune, embed = load("finetune", sys.argv[1]), load("embed", sys.argv[2])
config_dir = sys.argv[3]
tiny = GPT2Config(n_layer=2, n_embd=16, n_head=2, n_positions=12, vocab_size=30)
tiny.save_pretrained(config_dir)

def scratch(seed):
    finetune.set_seed(seed)
    build = finetune.mgm_builder(config_dir)
    return build(init="random", n_out=3, pooling="mean", head_dropout=0.1)

def same(a, b):
    a, b = a.state_dict(), b.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

first, again, other = scratch(0), scratch(0), scratch(1)
first.set_trainable_blocks(first.n_layer)
print(json.dumps({
    "same_seed": same(first, again),
    "other_seed": same(first, other),
    "equals_e2_network": same(first.backbone, embed.random_backbone(config_dir, 0)),
    "all_trainable": all(p.requires_grad for p in first.parameters()),
    "n_layer": first.n_layer,
}))
"""


@pytest.mark.skipif(not MGM_PYTHON.exists(), reason="needs envs/mgm (the env with transformers)")
def test_scratch_network_is_reproducible_and_starts_as_e2s_random_network(tmp_path):
    result = subprocess.run(
        [
            str(MGM_PYTHON), "-I", "-c", SCRATCH_INIT_CHECK,
            str(paths.REPO_ROOT / "scripts" / "mgm_finetune.py"),
            str(paths.REPO_ROOT / "scripts" / "mgm_embed.py"),
            str(tmp_path),
        ],
        capture_output=True, text=True, check=True,
    )
    checks = json.loads(result.stdout.strip().splitlines()[-1])
    assert checks == {"same_seed": True, "other_seed": False, "equals_e2_network": True,
                      "all_trainable": True, "n_layer": 2}
