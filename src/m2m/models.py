"""Network architectures: MLP regressor, masked-feature autoencoder and the MGM regression head.

Only torch and the standard library are imported: the MGM environment (torch 2.0.1, numpy 1.24)
imports this module too, so nothing here may depend on the main environment's newer stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

POOLING_MODES = ("mean", "last")
# The name transformers gives a pickled PyTorch checkpoint (MGM ships its weights in one).
PICKLED_WEIGHTS = "pytorch_model.bin"


def make_trunk(n_in: int, hidden: Sequence[int], dropout: float) -> nn.Sequential:
    """Linear → ReLU → Dropout for each hidden size.

    MLP and MaskedFeatureAutoencoder both build their encoder here, so their ``trunk.*``
    state_dict keys are identical and pretrained encoder weights load into a regressor by name.
    """
    if len(hidden) == 0:
        raise ValueError("hidden must list at least one layer size")
    layers: list[nn.Module] = []
    width = n_in
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
        width = size
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Multi-output regressor (E1): trunk of hidden layers + one linear output per target."""

    def __init__(
        self, n_in: int, n_out: int, hidden: Sequence[int] = (256, 128), dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.trunk = make_trunk(n_in, hidden, dropout)
        self.head = nn.Linear(hidden[-1], n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


class MaskedFeatureAutoencoder(nn.Module):
    """E4a pretraining network: the MLP trunk as encoder + a decoder back to every input feature.

    Masking belongs to the training loss, not to the module: the loss hides a random subset of
    inputs and scores only those positions, so the encoder must infer them from the rest.
    """

    def __init__(
        self,
        n_in: int,
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        decoder_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.trunk = make_trunk(n_in, hidden, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden[-1], decoder_hidden), nn.ReLU(), nn.Linear(decoder_hidden, n_in)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.trunk(x))


def load_compatible_state(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> tuple[list[str], list[str]]:
    """Copy the entries of ``state`` whose name and shape match ``model``.

    Returns ``(loaded, skipped)`` key lists. This is how a regressor starts from pretrained
    weights: the shared trunk loads, while an autoencoder's decoder (name not in the model) or a
    head trained for another target set (different shape) is skipped and keeps its fresh
    initialisation. Raises if nothing matches, because a "pretrained" run that silently trains
    from scratch would be mislabelled, and for the same reason refuses a layer that would load
    only in part (see ``_check_whole_layers``). The model is left untouched when it raises.
    """
    own = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in own and tuple(own[key].shape) == tuple(value.shape)
    }
    if not compatible:
        raise ValueError("init state shares no parameter (same name and shape) with the model")
    _check_whole_layers(own, state, compatible)
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible), sorted(set(state) - set(compatible))


def _check_whole_layers(
    own: Mapping[str, torch.Tensor],
    state: Mapping[str, torch.Tensor],
    compatible: Mapping[str, torch.Tensor],
) -> None:
    """Raise if some parameters of a layer would load and others not (shape mismatch).

    A trunk pretrained on other input features keeps the model's bias shape but not its weight
    shape: loading that bias, and the deeper layers on top of a random first layer, would give a
    mostly untrained network labelled "pretrained". A head for another target set mismatches in
    every parameter, so it is skipped whole and stays allowed.
    """
    loaded_layers = {_layer_name(key) for key in compatible}
    for key, value in state.items():
        if key in own and key not in compatible and _layer_name(key) in loaded_layers:
            raise ValueError(
                f"init state has {key} with shape {tuple(value.shape)} but the model expects "
                f"{tuple(own[key].shape)}, while the rest of that layer matches: was it "
                "pretrained on other input features or layer sizes?"
            )


def _layer_name(key: str) -> str:
    """'trunk.0.weight' -> 'trunk.0'; a top-level parameter is a layer of its own."""
    return key.rpartition(".")[0] or key


def pool_hidden_states(
    hidden: torch.Tensor, attention_mask: torch.Tensor, how: str = "mean"
) -> torch.Tensor:
    """Summarise token states ``(B, T, H)`` as ``(B, H)`` using real tokens only (mask != 0).

    "mean" averages the real positions. "last" takes the final real token, the only position
    of a causal model that has attended to the whole sequence; it assumes right padding, which
    is what MGM's MicroCorpus produces. Padded positions never enter the result, even if they
    hold NaN or inf.
    """
    if hidden.shape[:2] != attention_mask.shape:
        raise ValueError(
            f"hidden {tuple(hidden.shape)} and attention_mask {tuple(attention_mask.shape)} "
            "disagree on (batch, length)"
        )
    real = attention_mask != 0
    # Counted as integers, not in hidden's dtype: bfloat16 is exact only up to 256, so a longer
    # sequence would move the "last" index onto a neighbouring token or onto padding.
    lengths = real.sum(dim=1)
    if bool((lengths == 0).any()):
        raise ValueError("every sequence needs at least one real token")
    if how == "mean":
        # masked_fill rather than multiplying by the mask: 0 * NaN would still be NaN.
        summed = hidden.masked_fill(~real.unsqueeze(-1), 0.0).sum(dim=1)
        return summed / lengths.unsqueeze(-1).to(hidden.dtype)
    if how == "last":
        if bool((real[:, 1:] & ~real[:, :-1]).any()):
            raise ValueError("'last' pooling needs right padding: real tokens first, then padding")
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]
    raise ValueError(f"how must be one of {POOLING_MODES}, got {how!r}")


# The checkpoint is a GPT2LMHeadModel; its tied lm_head is the only tensor a GPT2Model drops.
EXPECTED_UNUSED_KEYS = {"lm_head.weight"}


def load_mgm_backbone(model_dir: str | Path) -> nn.Module:
    """MGM's pretrained GPT-2 transformer (no LM head), loaded without running pickled code.

    ``pytorch_model.bin`` is downloaded data: it is read with ``weights_only=True`` instead of
    the full unpickling transformers 4.33 would do. Raises if any backbone weight is missing.
    """
    # Imported here because only the MGM environment has transformers installed.
    from transformers import GPT2Model

    weights = Path(model_dir) / PICKLED_WEIGHTS
    state_dict = (
        torch.load(weights, map_location="cpu", weights_only=True)
        if weights.is_file()
        else None  # e.g. model.safetensors, a format that cannot carry code
    )
    backbone, info = GPT2Model.from_pretrained(
        str(model_dir), state_dict=state_dict, output_loading_info=True
    )
    unexpected = set(info["unexpected_keys"]) - EXPECTED_UNUSED_KEYS
    if info["missing_keys"] or unexpected:
        raise RuntimeError(
            f"MGM checkpoint mismatch: missing={info['missing_keys']}, "
            f"unexpected={sorted(unexpected)}"
        )
    return backbone


class MGMRegressor(nn.Module):
    """MGM's GPT-2 backbone, pooled over real tokens, with a linear head per metabolite (E3).

    ``backbone`` is duck-typed like ``transformers.GPT2Model``: ``.h`` (blocks), ``.ln_f``,
    ``.wte``, ``.wpe``, ``.config.n_embd``; called with ``input_ids``/``attention_mask`` it returns
    an object with ``.last_hidden_state``. Tests therefore run without transformers installed.
    Until ``set_trainable_blocks`` is called, every parameter is trainable.
    """

    def __init__(
        self, backbone: nn.Module, n_out: int, pooling: str = "mean", head_dropout: float = 0.1
    ) -> None:
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES}, got {pooling!r}")
        self.backbone = backbone
        self.pooling = pooling
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Linear(backbone.config.n_embd, n_out)
        self.n_trainable_blocks = self.n_layer
        # Hugging Face returns pretrained models in eval mode; align the backbone with this module.
        self.train()

    @property
    def n_layer(self) -> int:
        return len(self.backbone.h)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = pool_hidden_states(output.last_hidden_state, attention_mask, self.pooling)
        return self.head(self.dropout(pooled))

    def set_trainable_blocks(self, k: int) -> None:
        """Fine-tune the last ``k`` transformer blocks and the final LayerNorm; freeze the rest.

        ``k >= n_layer`` fine-tunes the whole backbone, token and position embeddings included.
        ``k = 0`` freezes all of it, final LayerNorm too (a linear probe on pooled features).
        The head is always trainable.
        """
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.n_trainable_blocks = min(k, self.n_layer)
        self.backbone.requires_grad_(False)
        for module in self._trainable_backbone_modules():
            module.requires_grad_(True)
        self.train(self.training)  # re-apply the eval rule below to the layers that just froze

    def train(self, mode: bool = True) -> MGMRegressor:
        """As ``nn.Module.train``, but frozen backbone layers always stay in eval mode.

        A frozen layer is a fixed feature extractor: dropout inside it would only add noise that
        no gradient can adapt to. Embedding dropout sits before the first block, so it counts as
        frozen unless the whole backbone is trainable.
        """
        super().train(mode)
        if mode and self.n_trainable_blocks < self.n_layer:
            self.backbone.eval()
            for module in self._trainable_backbone_modules():
                module.train()
        return self

    def param_groups(
        self, backbone_lr: float, head_lr: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        """AdamW groups: a small learning rate for pretrained layers, a larger one for the head.

        Frozen parameters are left out, so the optimiser never sees (or decays) them.
        """
        groups = []
        for part, lr in ((self.backbone, backbone_lr), (self.head, head_lr)):
            trainable = [p for p in part.parameters() if p.requires_grad]
            if trainable:  # k = 0 leaves no backbone parameter to optimise
                groups.append({"params": trainable, "lr": lr, "weight_decay": weight_decay})
        return groups

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        n_out: int,
        pooling: str = "mean",
        head_dropout: float = 0.1,
    ) -> MGMRegressor:
        """Wrap MGM's pretrained GPT-2 (``mgm/resources/general_model``) with a fresh head.

        MGM ships a ``GPT2LMHeadModel`` checkpoint; ``GPT2Model.from_pretrained`` keeps its
        transformer weights and drops the language-model head. A ``pytorch_model.bin`` is read
        here with ``weights_only=True``: it is downloaded data, and the full unpickling that
        transformers 4.33 would otherwise do can run code embedded in the file.
        """
        return cls(load_mgm_backbone(model_dir), n_out, pooling=pooling, head_dropout=head_dropout)

    def _trainable_backbone_modules(self) -> list[nn.Module]:
        """Backbone parts unfrozen by ``set_trainable_blocks`` (one rule for grads and mode)."""
        k = self.n_trainable_blocks
        if k >= self.n_layer:
            return [self.backbone]
        if k == 0:
            return []
        return [*self.backbone.h[-k:], self.backbone.ln_f]
