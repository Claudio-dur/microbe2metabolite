"""Architectures: shapes, weight transfer by name, pooling, and freezing of the MGM backbone."""

from __future__ import annotations

import copy
import math
import pickle
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from m2m.models import (
    MLP,
    PICKLED_WEIGHTS,
    POOLING_MODES,
    MaskedFeatureAutoencoder,
    MGMRegressor,
    load_compatible_state,
    make_trunk,
    pool_hidden_states,
)
from m2m.train import TrainConfig, fit_model

N_LAYER = 4  # enough blocks to tell k = 1, 2 and "all" apart
VOCAB, MAX_LEN, N_EMBD = 12, 7, 6
LENGTHS = (7, 4, 1)  # real tokens per sequence; the rest is right padding


class _FakeBlock(nn.Module):
    def __init__(self, dim: int, p: float) -> None:
        super().__init__()
        self.mlp = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(torch.tanh(self.mlp(x)))


class FakeGPT2(nn.Module):
    """The parts of transformers.GPT2Model that MGMRegressor relies on, with heavy dropout."""

    def __init__(self, n_layer: int = N_LAYER, p: float = 0.5) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_embd=N_EMBD, n_layer=n_layer)
        self.wte = nn.Embedding(VOCAB, N_EMBD)
        self.wpe = nn.Embedding(MAX_LEN, N_EMBD)
        self.drop = nn.Dropout(p)
        self.h = nn.ModuleList(_FakeBlock(N_EMBD, p) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(N_EMBD)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        positions = torch.arange(input_ids.shape[1])
        x = self.drop(self.wte(input_ids) + self.wpe(positions))
        for block in self.h:
            x = block(x)
        return SimpleNamespace(last_hidden_state=self.ln_f(x))


def _tokens(vocab: int = VOCAB) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded batch like MGM's MicroCorpus: pad id 0 wherever the mask is 0."""
    generator = torch.Generator().manual_seed(1)
    mask = (torch.arange(MAX_LEN) < torch.tensor(LENGTHS)[:, None]).long()
    input_ids = torch.randint(1, vocab, (len(LENGTHS), MAX_LEN), generator=generator) * mask
    return input_ids, mask


def _mgm(k: int | None = None, n_out: int = 3) -> MGMRegressor:
    torch.manual_seed(0)
    model = MGMRegressor(FakeGPT2(), n_out=n_out)
    if k is not None:
        model.set_trainable_blocks(k)
    return model


def _trainable_names(module: nn.Module) -> set[str]:
    return {name for name, p in module.named_parameters() if p.requires_grad}


# --- MLP and autoencoder -------------------------------------------------------------------------


def test_mlp_default_architecture_follows_the_brief() -> None:
    model = MLP(20, 4)
    layer_types = [type(m) for m in model.trunk]
    assert layer_types == [nn.Linear, nn.ReLU, nn.Dropout] * 2
    assert [(m.in_features, m.out_features) for m in model.trunk if isinstance(m, nn.Linear)] == [
        (20, 256),
        (256, 128),
    ]
    assert all(m.p == 0.3 for m in model.trunk if isinstance(m, nn.Dropout))
    assert (model.head.in_features, model.head.out_features) == (128, 4)


def test_mlp_output_shape_with_custom_layers() -> None:
    model = MLP(10, 3, hidden=[16, 8, 4], dropout=0.2)
    assert model(torch.randn(5, 10)).shape == (5, 3)
    assert [m.out_features for m in model.trunk if isinstance(m, nn.Linear)] == [16, 8, 4]


def test_make_trunk_needs_a_hidden_layer() -> None:
    with pytest.raises(ValueError, match="at least one"):
        make_trunk(5, (), 0.1)


def test_autoencoder_reconstructs_every_input_feature() -> None:
    model = MaskedFeatureAutoencoder(12, hidden=(16, 8), decoder_hidden=10)
    assert model(torch.randn(4, 12)).shape == (4, 12)
    assert [type(m) for m in model.decoder] == [nn.Linear, nn.ReLU, nn.Linear]
    assert (model.decoder[0].in_features, model.decoder[0].out_features) == (8, 10)


def test_autoencoder_trunk_loads_into_mlp_by_name() -> None:
    torch.manual_seed(0)
    autoencoder = MaskedFeatureAutoencoder(12, hidden=(16, 8))
    mlp = MLP(12, 5, hidden=(16, 8))
    trunk_state = {k: v for k, v in autoencoder.state_dict().items() if k.startswith("trunk.")}
    assert trunk_state.keys() == {k for k in mlp.state_dict() if k.startswith("trunk.")}

    result = mlp.load_state_dict(trunk_state, strict=False)

    assert result.unexpected_keys == []
    assert sorted(result.missing_keys) == ["head.bias", "head.weight"]
    x = torch.randn(3, 12)
    torch.testing.assert_close(mlp.eval().trunk(x), autoencoder.eval().trunk(x))


def test_load_compatible_state_skips_other_names_and_shapes() -> None:
    torch.manual_seed(0)
    autoencoder = MaskedFeatureAutoencoder(6, hidden=(8, 4))
    other_head = MLP(6, 5, hidden=(8, 4)).head  # trained for another target set: 5 outputs
    state = {
        **autoencoder.state_dict(),
        "head.weight": other_head.weight,
        "head.bias": other_head.bias,
    }
    model = MLP(6, 3, hidden=(8, 4))
    head_before = copy.deepcopy(model.head.state_dict())

    loaded, skipped = load_compatible_state(model, state)

    trunk_keys = sorted(k for k in autoencoder.state_dict() if k.startswith("trunk."))
    decoder_keys = [k for k in autoencoder.state_dict() if k.startswith("decoder.")]
    assert loaded == trunk_keys
    assert skipped == sorted(["head.weight", "head.bias", *decoder_keys])
    for key in trunk_keys:
        assert torch.equal(model.state_dict()[key], autoencoder.state_dict()[key])
    for key, value in head_before.items():
        assert torch.equal(model.head.state_dict()[key], value)


def test_load_compatible_state_refuses_a_state_with_nothing_in_common() -> None:
    with pytest.raises(ValueError, match="shares no parameter"):
        load_compatible_state(MLP(6, 3, hidden=(4,)), {"encoder.weight": torch.zeros(4, 6)})


@pytest.mark.parametrize(
    ("n_in", "hidden"),
    [(8, (8, 4)), (6, (10, 4))],
    ids=["other_input_features", "other_layer_sizes"],
)
def test_load_compatible_state_refuses_a_half_loaded_layer(n_in: int, hidden: tuple) -> None:
    # E.g. an E4a trunk pretrained on another genus vocabulary: its first-layer bias still fits
    # (and with other layer sizes, a deeper bias does). Loading those alone, plus the layers
    # after them, would label a mostly untrained network "pretrained".
    torch.manual_seed(0)
    source = MaskedFeatureAutoencoder(n_in, hidden=hidden)
    model = MLP(6, 3, hidden=(8, 4))
    before = copy.deepcopy(model.state_dict())

    with pytest.raises(ValueError, match="rest of that layer matches"):
        load_compatible_state(model, source.state_dict())

    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # untouched


# --- pooling -------------------------------------------------------------------------------------


def test_pooling_matches_hand_computation_with_padding() -> None:
    torch.manual_seed(0)
    hidden = torch.randn(3, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    expected_mean = torch.stack([hidden[0, :3].mean(0), hidden[1].mean(0), hidden[2, 0]])
    expected_last = torch.stack([hidden[0, 2], hidden[1, 4], hidden[2, 0]])

    torch.testing.assert_close(pool_hidden_states(hidden, mask, "mean"), expected_mean)
    torch.testing.assert_close(pool_hidden_states(hidden, mask, "last"), expected_last)


@pytest.mark.parametrize("fill", [1e6, math.inf, math.nan])
@pytest.mark.parametrize("how", ["mean", "last"])
def test_pooling_ignores_padded_positions(how: str, fill: float) -> None:
    # Multiplying padding by a zero mask is not enough: 0 * inf and 0 * NaN are NaN.
    torch.manual_seed(0)
    hidden = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    noisy = hidden.clone()
    noisy[mask == 0] = fill
    torch.testing.assert_close(
        pool_hidden_states(noisy, mask, how), pool_hidden_states(hidden, mask, how)
    )


@pytest.mark.parametrize("length", [257, 511])
def test_last_token_is_exact_beyond_the_bfloat16_integer_range(length: int) -> None:
    # bfloat16 counts exactly only up to 256: a token count kept in the hidden dtype picked a
    # neighbouring token (257) or a padding position (511) as the "last" one.
    hidden = torch.zeros(1, 512, 2, dtype=torch.bfloat16)
    hidden[0, length - 1] = 1.0  # only the true last real token is non-zero
    mask = (torch.arange(512) < length).long()[None]
    pooled = pool_hidden_states(hidden, mask, "last")
    assert pooled.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("dtype", [torch.long, torch.float32, torch.bool])
def test_pooling_accepts_integer_float_and_bool_masks(dtype: torch.dtype) -> None:
    # MGM's MicroCorpus builds a float mask; tokenizers usually return integer ones.
    hidden = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    for how in ("mean", "last"):
        torch.testing.assert_close(
            pool_hidden_states(hidden, mask.to(dtype), how), pool_hidden_states(hidden, mask, how)
        )


def test_pooling_rejects_bad_inputs() -> None:
    hidden = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="right padding"):
        pool_hidden_states(hidden, torch.tensor([[0, 1, 1], [1, 1, 1]]), "last")
    with pytest.raises(ValueError, match="at least one real token"):
        pool_hidden_states(hidden, torch.tensor([[0, 0, 0], [1, 1, 1]]), "mean")
    with pytest.raises(ValueError, match="disagree"):
        pool_hidden_states(hidden, torch.ones(2, 4), "mean")
    with pytest.raises(ValueError, match="how must be"):
        pool_hidden_states(hidden, torch.ones(2, 3), "max")


# --- MGMRegressor with a stand-in backbone -------------------------------------------------------


@pytest.mark.parametrize("pooling", ["mean", "last"])
def test_mgm_regressor_output_shape(pooling: str) -> None:
    torch.manual_seed(0)
    model = MGMRegressor(FakeGPT2(), n_out=5, pooling=pooling)
    input_ids, mask = _tokens()
    assert model(input_ids, mask).shape == (len(LENGTHS), 5)
    assert model.head.in_features == N_EMBD


def test_mgm_regressor_rejects_unknown_pooling() -> None:
    with pytest.raises(ValueError, match="pooling"):
        MGMRegressor(FakeGPT2(), n_out=2, pooling="cls")


def test_mgm_regressor_is_fully_trainable_until_told_otherwise() -> None:
    model = _mgm()
    assert all(p.requires_grad for p in model.parameters())
    assert all(m.training for m in model.modules())


@pytest.mark.parametrize("k", [1, 2, N_LAYER])
def test_set_trainable_blocks_unfreezes_last_k_blocks_and_final_norm(k: int) -> None:
    model = _mgm(k)
    backbone = model.backbone
    expected = {
        f"h.{i}.{name}"
        for i in range(N_LAYER - k, N_LAYER)
        for name, _ in backbone.h[i].named_parameters()
    }
    expected |= {f"ln_f.{name}" for name, _ in backbone.ln_f.named_parameters()}
    if k == N_LAYER:  # fine-tuning everything includes the token and position embeddings
        expected |= {"wte.weight", "wpe.weight"}

    assert _trainable_names(backbone) == expected
    assert all(p.requires_grad for p in model.head.parameters())


def test_set_trainable_blocks_edge_values() -> None:
    model = _mgm(0)  # linear probe: the whole backbone frozen, final LayerNorm included
    assert _trainable_names(model.backbone) == set()
    assert _trainable_names(model) == {"head.weight", "head.bias"}
    model.set_trainable_blocks(N_LAYER + 3)  # more blocks than exist means "everything"
    assert all(p.requires_grad for p in model.parameters())
    model.set_trainable_blocks(1)  # re-freezing works after unfreezing
    assert not model.backbone.wte.weight.requires_grad
    with pytest.raises(ValueError, match=">= 0"):
        model.set_trainable_blocks(-1)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_param_groups_hold_only_trainable_parameters(k: int) -> None:
    model = _mgm(k)
    groups = model.param_groups(backbone_lr=2e-5, head_lr=1e-3, weight_decay=0.01)

    grouped = [p for group in groups for p in group["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in grouped} == {id(p) for p in trainable}
    assert len(grouped) == len(trainable)  # no parameter twice
    assert [g["lr"] for g in groups] == ([1e-3] if k == 0 else [2e-5, 1e-3])
    assert all(g["weight_decay"] == 0.01 for g in groups)
    head_ids = {id(p) for p in model.head.parameters()}
    assert {id(p) for p in groups[-1]["params"]} == head_ids


@pytest.mark.parametrize("k", [0, 1, 2])
def test_frozen_parts_stay_in_eval_mode_when_training(k: int) -> None:
    model = _mgm(k)
    model.train()
    backbone = model.backbone
    trainable_blocks = set(range(N_LAYER - k, N_LAYER))

    assert model.training and model.dropout.training and model.head.training
    assert not backbone.drop.training  # embedding dropout feeds the first, frozen, block
    for i, block in enumerate(backbone.h):
        assert all(m.training == (i in trainable_blocks) for m in block.modules())
    assert backbone.ln_f.training == (k > 0)

    model.eval()
    assert not any(m.training for m in model.modules())
    model.set_trainable_blocks(N_LAYER)
    model.train()
    assert all(m.training for m in model.modules())


def test_frozen_backbone_adds_no_dropout_noise_in_train_mode() -> None:
    input_ids, mask = _tokens()
    frozen = _mgm(0).train()
    states = [frozen.backbone(input_ids, mask).last_hidden_state for _ in range(2)]
    assert torch.equal(states[0], states[1])

    tuned = _mgm(N_LAYER).train()
    states = [tuned.backbone(input_ids, mask).last_hidden_state for _ in range(2)]
    assert not torch.equal(states[0], states[1])


def test_gradients_reach_only_trainable_parameters() -> None:
    model = _mgm(1).train()
    input_ids, mask = _tokens()
    model(input_ids, mask).pow(2).mean().backward()
    for name, p in model.named_parameters():
        assert (p.grad is not None) == p.requires_grad, name


def test_fit_model_updates_only_the_unfrozen_part_of_mgm() -> None:
    model = _mgm(1, n_out=2)
    before = copy.deepcopy(model.state_dict())
    input_ids, mask = _tokens()
    target = torch.randn(len(LENGTHS), 2)
    cfg = TrainConfig(lr=1e-2, weight_decay=0.01, batch_size=2, max_epochs=3, patience=3)

    fit_model(
        model,
        (input_ids, mask, target),
        (input_ids, mask, target),
        cfg,
        seed=0,
        param_groups=model.param_groups(backbone_lr=1e-3, head_lr=1e-2, weight_decay=0.01),
    )

    after = model.state_dict()
    changed = {key for key in before if not torch.equal(before[key], after[key])}
    last = N_LAYER - 1
    assert changed == {
        f"backbone.h.{last}.mlp.weight",
        f"backbone.h.{last}.mlp.bias",
        "backbone.ln_f.weight",
        "backbone.ln_f.bias",
        "head.weight",
        "head.bias",
    }
    assert not model.training  # fit_model hands the model back in eval mode


# --- real GPT-2 (only where transformers is installed, i.e. the MGM environment) -----------------


def _tiny_gpt2_config(transformers: ModuleType, n_layer: int) -> Any:
    return transformers.GPT2Config(
        vocab_size=VOCAB, n_positions=16, n_embd=8, n_layer=n_layer, n_head=2
    )


def test_mgm_regressor_wraps_a_real_gpt2_model() -> None:
    transformers = pytest.importorskip("transformers")
    n_layer = 3
    torch.manual_seed(0)
    model = MGMRegressor(transformers.GPT2Model(_tiny_gpt2_config(transformers, n_layer)), n_out=3)
    input_ids, mask = _tokens()
    assert model(input_ids, mask).shape == (len(LENGTHS), 3)

    for k in (1, 2, n_layer):
        model.set_trainable_blocks(k)
        model.train()
        backbone = model.backbone
        trainable = _trainable_names(backbone)
        blocks = {name.split(".")[1] for name in trainable if name.startswith("h.")}
        assert blocks == {str(i) for i in range(n_layer - k, n_layer)}
        assert all(p.requires_grad for p in backbone.ln_f.parameters())
        embeddings = (backbone.wte.weight.requires_grad, backbone.wpe.weight.requires_grad)
        assert embeddings == ((k == n_layer),) * 2
        assert backbone.drop.training == (k == n_layer)
        assert [b.training for b in backbone.h] == [i >= n_layer - k for i in range(n_layer)]


@pytest.mark.parametrize("pooling", POOLING_MODES)
def test_real_gpt2_rows_do_not_depend_on_padding_or_batch_mates(pooling: str) -> None:
    # E2/E3 trim padding to the longest sequence and predict in chunks: a row's output must not
    # change with its padding width or with the other rows in its batch (no cross-sample leak).
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    backbone = transformers.GPT2Model(_tiny_gpt2_config(transformers, 2))
    model = MGMRegressor(backbone, n_out=3, pooling=pooling).eval()
    input_ids, mask = _tokens()

    with torch.no_grad():
        together = model(input_ids, mask)
        alone = torch.cat(
            [model(input_ids[i : i + 1, :n], mask[i : i + 1, :n]) for i, n in enumerate(LENGTHS)]
        )

    torch.testing.assert_close(alone, together)


class _NotATensor:
    """Stands in for an arbitrary object pickled into a checkpoint (where code could hide)."""


def _save_like_mgm(directory: Path, state: dict[str, Any], config: Any) -> None:
    """MGM's general_model layout: config.json + a pickled pytorch_model.bin."""
    config.save_pretrained(directory)
    torch.save(state, directory / PICKLED_WEIGHTS)


def test_from_pretrained_reads_an_mgm_style_checkpoint_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    # MGM ships a GPT2LMHeadModel checkpoint; the regressor keeps its transformer weights.
    language_model = transformers.GPT2LMHeadModel(_tiny_gpt2_config(transformers, 2))
    _save_like_mgm(tmp_path, language_model.state_dict(), language_model.config)
    load_calls: list[dict[str, Any]] = []
    real_load = torch.load

    def spy_load(*args: Any, **kwargs: Any) -> Any:
        load_calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)

    model = MGMRegressor.from_pretrained(tmp_path, n_out=4, pooling="last")

    assert load_calls and all(call.get("weights_only") is True for call in load_calls)
    loaded = model.backbone.state_dict()
    for name, tensor in language_model.transformer.state_dict().items():
        torch.testing.assert_close(loaded[name], tensor)
    assert model.pooling == "last" and model.head.out_features == 4
    assert model.training and all(p.requires_grad for p in model.parameters())


def test_from_pretrained_refuses_a_checkpoint_carrying_python_objects(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    language_model = transformers.GPT2LMHeadModel(_tiny_gpt2_config(transformers, 2))
    state = {**language_model.state_dict(), "extra": _NotATensor()}
    _save_like_mgm(tmp_path, state, language_model.config)

    with pytest.raises(pickle.UnpicklingError):
        MGMRegressor.from_pretrained(tmp_path, n_out=4)
