"""E4: own pretraining on D2 sources, then transfer to Franzosa (brief section 6, rows E4a-E4c).

Stages, in order (``--stage`` runs one, or several comma-separated):
  sources   vocabulary, pretraining table, source/target disjointness, iHMP NaN evidence
            -> results/e4_sources.json, data/processed/e4/vocabulary.json
  e4a       masked-feature autoencoder on the deduplicated source metagenomes, one per seed
            -> data/processed/e4/e4a_seed{s}.pt (+ .json, .preprocessor.json)
  e4b       iHMP genera -> shared HMDB metabolites, from scratch ("ihmp") and from E4a
            ("selfsup_ihmp"), one per seed; plus a subject-grouped CV sanity check on iHMP
            -> data/processed/e4/e4b_{variant}_seed{s}.pt (+ .json, .preprocessor.json)
  null      D2 ridge permutation null, scored on Validation -> results/null_thresholds.csv
  select    PRISM CV (the saved D1 folds) of every E4c variant and of the D2 ridge
            -> results/selected/e4c_{variant}__{T466,S176}.json, d2_ridge__*.json, runs.csv,
            and results/selected/e4c_pretrained_fingerprints.json (what it was selected with)
  external  the selected configurations, scored once on Validation -> summary.csv
  curve     E5 learning curve (T466) of e4c_scratch and the best pretrained variant by CV
            -> results/learning_curve.csv
(results/e4_pretraining.json collects the E4a/E4b losses and the iHMP sanity check.)

Leakage guards: the vocabulary and every pretraining fit see source data only, checked disjoint
from the target by ID and by profile; selection uses PRISM folds only; Validation (= NLIBD) is
opened only in null / external / curve, through the guarded loader, for configurations already
selected and saved, and only once the vocabulary, model settings and pretrained weights are
checked unchanged since selection. Every E4c variant refits the same PRISM preprocessing on each
training fold (DESIGN.md #11), so the variants differ only in their initial weights.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from m2m import data, evaluate, paths, protocol, splits, tracking, transfer_data
from m2m.config import config_hash
from m2m.experiments import common
from m2m.models import MaskedFeatureAutoencoder
from m2m.preprocess import MicrobePreprocessor
from m2m.train import (
    TorchMLPRegressor,
    TrainConfig,
    TrainHistory,
    evaluate_loss,
    fit_model,
    set_seed,
)
from m2m.transfer_data import HmdbTargetPreprocessor, SharedTargets

EXPERIMENT = "e4"
DATA_NAME = "D2"
# Selection stages first, then the three that open the external cohort (DESIGN decision #24).
STAGES = ("sources", "e4a", "e4b", "select", "null", "external", "curve")
T466, S176 = "T466", "S176"
TARGET_SETS = (T466, S176)
VARIANTS = ("scratch", "selfsup", "ihmp", "selfsup_ihmp")
E4B_VARIANTS = ("ihmp", "selfsup_ihmp")  # the E4b networks, which have a head for S176
RIDGE_MODEL = "d2_ridge"
DIAGNOSTIC_MODEL = "e4b_ihmp_diagnostic"
# Which pretrained network each E4c variant starts from (file stem in data/processed/e4/).
PRETRAINED_STEM = {"selfsup": "e4a", "ihmp": "e4b_ihmp", "selfsup_ihmp": "e4b_selfsup_ihmp"}
# The brief asks to hide 20-30% of the non-zero values (section 6, E4a).
MASK_RATIO_RANGE = (0.20, 0.30)

SOURCES_JSON = "e4_sources.json"
PRETRAINING_JSON = "e4_pretraining.json"
VOCABULARY_JSON = "vocabulary.json"
FINGERPRINTS_JSON = "e4c_pretrained_fingerprints.json"  # in results/selected/
# Config sections that define the selected models: input and target preprocessing, the MLP
# (architecture, grid, training), ridge, the early-stopping split and the E4c variants. Editing
# one after `select` makes every saved choice stale, so the final stages refuse to score them.
SELECTION_CONFIG_SECTIONS = ("microbes", "metabolites", "mlp", "ridge", "inner_validation", "e4c")
TRUNK_PREFIX, HEAD_PREFIX = "trunk.", "head."
HEAD_WEIGHT, HEAD_BIAS = "head.weight", "head.bias"


def run(cfg: dict, *, stage: str | None = None, n_jobs: int = 1) -> None:
    """Run every E4 stage in order, or only the requested ones ("select" or "select,external")."""
    # A dry run must never write into the committed results/. The PRISM-loading stages are
    # refused there by common.dry_run, but sources, e4a and e4b load no PRISM split, so nothing
    # else would stop them writing e4_sources.json, e4_pretraining.json and runs.csv rows.
    common.dry_run()
    check_config(cfg)
    for name in common.stages_to_run(stage, STAGES):
        print(f"[e4] stage {name}")
        STAGE_FUNCTIONS[name](cfg, n_jobs=n_jobs)


def check_config(cfg: Mapping[str, Any]) -> None:
    """Fail early on settings that would make a pretrained trunk not fit the E4c network."""
    trunk = list(cfg["mlp"]["hidden"])
    if list(cfg["e4a"]["hidden"]) != trunk:
        raise ValueError(f"e4a.hidden {cfg['e4a']['hidden']} must equal mlp.hidden {trunk}")
    low, high = MASK_RATIO_RANGE
    if not low <= float(cfg["e4a"]["mask_ratio"]) <= high:
        raise ValueError(f"e4a.mask_ratio must be in [{low}, {high}] (brief section 6)")
    unknown = sorted(set(cfg["e4c"]["variants"]) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown E4c variants {unknown}; expected a subset of {VARIANTS}")
    unknown = sorted(set(cfg["e4c"]["target_sets"]) - set(TARGET_SETS))
    if unknown:
        raise ValueError(f"unknown E4c target sets {unknown}; expected a subset of {TARGET_SETS}")


# --------------------------------------------------------------------------------------------
# Data access (every D2 read goes through data.py's loaders and guards)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Inputs:
    """What every E4c builder needs: the source vocabulary and the two target definitions."""

    vocabulary: list[str]
    t466: list[str]
    shared: SharedTargets

    def builder_kwargs(self) -> dict[str, Any]:
        return {"vocabulary": self.vocabulary, "t466": self.t466, "shared": self.shared}


def load_sources() -> dict[str, data.Cohort]:
    """The five source datasets, ERAWIJANTARI copies of YACHIDA dropped (DESIGN.md #9)."""
    return data.load_sources(drop_duplicates=True, with_metabolites=False)


def load_ihmp() -> data.Cohort:
    """iHMP genera + HMDB-mapped metabolites (NaN kept; see transfer_data.fill_not_detected)."""
    return data.load_d2(transfer_data.IHMP, with_metabolites=True, hmdb_only=True)


def load_shared_targets(t466: Sequence[str]) -> SharedTargets:
    return transfer_data.shared_hmdb_targets(
        data.load_d2_mtb_map(data.TARGET_DATASET), data.load_d2_mtb_map(transfer_data.IHMP), t466
    )


def load_inputs() -> Inputs:
    t466 = data.t466_features()
    return Inputs(vocabulary=load_vocabulary(), t466=t466, shared=load_shared_targets(t466))


def load_target_prism() -> data.Cohort:
    """All of Franzosa PRISM (the D2 view of the training cohort)."""
    return data.load_franzosa(part="PRISM")


def load_target_train() -> data.Cohort:
    """Franzosa PRISM; in dry runs only the D1 pseudo-train samples (mapped to D2 IDs)."""
    prism = load_target_prism()
    if common.dry_run():
        prism = prism.subset([data.d1_to_d2_prism_id(i) for i in common.dry_run_ids()[0]])
    return prism


def load_target_test(purpose: str) -> data.Cohort:
    """Franzosa Validation (= NLIBD) for a final evaluation; dry runs get the PRISM hold-out."""
    if common.dry_run():
        held_out = [data.d1_to_d2_prism_id(i) for i in common.dry_run_ids()[1]]
        return load_target_prism().subset(held_out)
    return data.load_franzosa(part="Validation", final_evaluation=True, purpose=purpose)


def open_target_test(
    cfg: Mapping[str, Any], vocabulary: Sequence[str], purpose: str
) -> data.Cohort:
    """The external cohort, after re-checking that no source sample is one of its samples."""
    test = load_target_test(purpose)
    report = transfer_data.check_source_target_disjoint(
        load_sources(),
        test.meta,
        test.microbes,
        vocabulary,
        max_correlation=float(cfg["e4"]["near_duplicate_max_correlation"]),
        target_name="the external test cohort",
    )
    print(
        f"[e4] sources vs external cohort: max profile r = {report['max_profile_correlation']:.3f}"
    )
    return test


def target_folds(cfg: Mapping[str, Any], target_ids: Sequence[str]) -> pd.DataFrame:
    """The saved D1 PRISM folds with IDs mapped to D2, so E4c is scored like every other model."""
    _, _, d1_labels = common.load_d1_train()
    folds = common.d1_folds(d1_labels, cfg)
    folds = folds.assign(sample_id=[data.d1_to_d2_prism_id(i) for i in folds["sample_id"]])
    if set(folds["sample_id"]) != set(target_ids):
        raise ValueError("the mapped D1 folds do not cover exactly the Franzosa PRISM samples")
    return folds


def weights_dir() -> Path:
    return paths.DATA_PROCESSED / EXPERIMENT


def save_vocabulary(vocabulary: Sequence[str]) -> Path:
    path = weights_dir() / VOCABULARY_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"digest": transfer_data.vocabulary_digest(vocabulary), "genera": list(vocabulary)}
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def load_vocabulary() -> list[str]:
    path = weights_dir() / VOCABULARY_JSON
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the e4 'sources' stage first")
    return list(json.loads(path.read_text())["genera"])


# --------------------------------------------------------------------------------------------
# Stage: sources
# --------------------------------------------------------------------------------------------


def stage_sources(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Vocabulary from the sources only, source/target checks, iHMP NaN evidence."""
    sources = load_sources()
    vocabulary = transfer_data.source_vocabulary(sources, cfg)
    save_vocabulary(vocabulary)
    table, keys = transfer_data.pretraining_table(sources)
    prevalent = transfer_data.prevalent_genera_by_source(sources, cfg)
    prism = load_target_prism()
    disjoint = transfer_data.check_source_target_disjoint(
        sources,
        data.load_d2_metadata(data.TARGET_DATASET),  # PRISM rows only; Validation is guarded
        prism.microbes,
        vocabulary,
        max_correlation=float(cfg["e4"]["near_duplicate_max_correlation"]),
        target_name="Franzosa PRISM",
    )
    t466 = data.t466_features()
    shared = load_shared_targets(t466)
    ihmp = load_ihmp()
    shared_compounds = [c for c in ihmp.metabolites.columns if c in shared.ihmp]
    report = {
        "datasets": {
            name: {
                "samples": len(cohort.sample_ids),
                "subjects": int(cohort.meta["subject"].nunique()),
                "prevalent_genera": len(prevalent[name]),
            }
            for name, cohort in sources.items()
        },
        "pretraining_table": {
            "samples": int(table.shape[0]),
            "genus_columns": int(table.shape[1]),
            "subjects": int(keys.nunique()),
        },
        "vocabulary": {
            "size": len(vocabulary),
            "digest": transfer_data.vocabulary_digest(vocabulary),
            "file": f"data/processed/{EXPERIMENT}/{VOCABULARY_JSON}",
        },
        "franzosa_prism_coverage": transfer_data.vocabulary_coverage(
            vocabulary, prism.microbes, cfg
        ),
        "source_target_disjointness": disjoint,
        "shared_hmdb": shared.summary(),
        "ihmp_missing_values": transfer_data.ihmp_missing_evidence(
            ihmp.metabolites, data.load_d2_mtb_map(transfer_data.IHMP), shared_compounds
        ),
    }
    _write_results_json(SOURCES_JSON, report)
    coverage = report["franzosa_prism_coverage"]
    print(
        f"[e4] vocabulary {len(vocabulary)} genera from {table.shape[0]} source samples; covers "
        f"{coverage['covered']}/{coverage['target_prevalent_genera']} Franzosa-PRISM prevalent "
        f"genera; max source-target profile r = {disjoint['max_profile_correlation']:.3f}; "
        f"{len(shared.hmdb_order)} shared HMDB IDs"
    )


# --------------------------------------------------------------------------------------------
# Stage: e4a (masked-feature self-supervised pretraining)
# --------------------------------------------------------------------------------------------


def draw_mask(maskable: torch.Tensor, ratio: float, generator: torch.Generator) -> torch.Tensor:
    """Hide round(ratio x n) of each row's maskable positions (at least one), chosen at random.

    Only ``maskable`` positions (genera present in the sample) can be hidden: an absent genus's
    CLR value comes from the pseudocount rule, not from a measurement, so reconstructing it would
    teach the encoder nothing about the community.
    """
    maskable = maskable.bool()
    n_maskable = maskable.sum(dim=1)
    n_hidden = torch.floor(ratio * n_maskable.double() + 0.5).long().clamp(min=1)
    n_hidden = torch.minimum(n_hidden, n_maskable)
    scores = torch.rand(maskable.shape, generator=generator, dtype=torch.float64)
    # Positions that may not be hidden sort after every maskable one (rand < 1 < 2).
    scores = scores.masked_fill(~maskable, 2.0)
    rank = scores.argsort(dim=1).argsort(dim=1)
    return rank < n_hidden.unsqueeze(1)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the masked positions only."""
    weights = mask.to(prediction.dtype)
    return ((prediction - target) ** 2 * weights).sum() / weights.sum().clamp(min=1.0)


class MaskedReconstructionLoss:
    """E4a objective for ``train.fit_model``: reconstruct hidden inputs from the visible ones.

    Batches are ``(inputs, positions, target)``. In training mode a fresh random ``mask_ratio``
    share of ``positions`` (the non-zero genera) is hidden on every call; in eval mode
    ``positions`` *is* the mask, so validation scores the same fixed, seeded mask every epoch and
    the early-stopping curve is not noise from re-drawn masks. Hidden inputs are set to 0, the
    training mean in z-space, and only hidden positions enter the loss.
    """

    def __init__(self, mask_ratio: float, generator: torch.Generator) -> None:
        self.mask_ratio = float(mask_ratio)
        self.generator = generator

    def __call__(self, model: torch.nn.Module, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        inputs, positions, target = batch
        if model.training:
            mask = draw_mask(positions, self.mask_ratio, self.generator)
        else:
            mask = positions.bool()
        prediction = model(inputs.masked_fill(mask, 0.0))
        return masked_mse(prediction, target, mask)


@dataclass
class AutoencoderData:
    """One seed's E4a inputs, prepared in the parent process (small arrays only)."""

    x_fit: np.ndarray
    present_fit: np.ndarray
    x_val: np.ndarray
    val_mask: np.ndarray
    fit_ids: list[str]
    val_ids: list[str]
    preprocessor: MicrobePreprocessor


def autoencoder_data(
    table: pd.DataFrame,
    keys: pd.Series,
    vocabulary: Sequence[str],
    cfg: Mapping[str, Any],
    seed: int,
) -> AutoencoderData:
    """Subject-grouped split, input scaling fit on the training part, fixed validation mask."""
    e4a = cfg["e4a"]
    fit_ids, val_ids = splits.grouped_inner_split(
        list(table.index),
        keys,
        fraction=float(e4a["val_fraction"]),
        seed=seed + int(cfg["inner_validation"]["seed_offset"]),
    )
    # The scaler sees the training part only: validation rows are also their own reconstruction
    # targets, so their statistics must not shape the space they are scored in.
    prep = common.microbe_prep(cfg, features=vocabulary).fit(table.loc[fit_ids])
    z = prep.transform(table).to_numpy(dtype=np.float32)
    present = table.reindex(columns=list(vocabulary), fill_value=0.0).to_numpy() > 0
    rows = pd.Index(table.index)
    fit_pos, val_pos = rows.get_indexer(fit_ids), rows.get_indexer(val_ids)
    generator = torch.Generator().manual_seed(seed + int(e4a["val_mask_seed_offset"]))
    val_mask = draw_mask(torch.from_numpy(present[val_pos]), float(e4a["mask_ratio"]), generator)
    return AutoencoderData(
        x_fit=z[fit_pos],
        present_fit=present[fit_pos],
        x_val=z[val_pos],
        val_mask=val_mask.numpy(),
        fit_ids=list(fit_ids),
        val_ids=list(val_ids),
        preprocessor=prep,
    )


def train_autoencoder(
    inputs: AutoencoderData, cfg: Mapping[str, Any], seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Fit one masked-feature autoencoder; returns its weights and validation metrics."""
    protocol.single_thread_torch()
    e4a = cfg["e4a"]
    x_fit, present_fit = _tensor(inputs.x_fit), _tensor(inputs.present_fit)
    x_val, val_mask = _tensor(inputs.x_val), _tensor(inputs.val_mask)
    set_seed(seed)
    model = MaskedFeatureAutoencoder(
        x_fit.shape[1],
        hidden=tuple(e4a["hidden"]),
        dropout=float(e4a["dropout"]),
        decoder_hidden=int(e4a["decoder_hidden"]),
    )
    train_masks = torch.Generator().manual_seed(seed + int(e4a["train_mask_seed_offset"]))
    loss = MaskedReconstructionLoss(float(e4a["mask_ratio"]), train_masks)
    history = fit_model(
        model,
        (x_fit, present_fit, x_fit),
        (x_val, val_mask, x_val),
        TrainConfig.from_dict(e4a["train"]),
        seed=seed,
        loss_fn=loss,
    )
    metrics = {
        **_history_metrics(history),
        # Early stopping averaged per-batch masked means; this is the exact masked MSE over every
        # hidden validation value (one batch), at the restored best epoch.
        "val_loss": evaluate_loss(model, (x_val, val_mask, x_val), loss, batch_size=len(x_val)),
        # Predicting every hidden value by its training mean (0 in z-space).
        "val_loss_mean_baseline": float((x_val[val_mask] ** 2).mean()),
        "n_fit": len(inputs.fit_ids),
        "n_val": len(inputs.val_ids),
        "hidden_per_val_sample": float(val_mask.sum(dim=1).double().mean()),
    }
    return _detached(model.state_dict()), metrics


def stage_e4a(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """One masked-feature autoencoder per model seed, on all deduplicated source samples."""
    vocabulary = load_vocabulary()
    table, keys = transfer_data.pretraining_table(load_sources())
    seeds = common.model_seeds(cfg)
    prepared = [autoencoder_data(table, keys, vocabulary, cfg, s) for s in seeds]
    del table  # the 12k-column table is no longer needed; workers get small arrays only
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(train_autoencoder)(inputs, cfg, s)
        for inputs, s in zip(prepared, seeds, strict=True)
    )
    summary = {}
    for seed, inputs, (state, metrics) in zip(seeds, prepared, outputs, strict=True):
        meta = {
            "network": "MaskedFeatureAutoencoder",
            "seed": seed,
            "vocabulary_digest": transfer_data.vocabulary_digest(vocabulary),
            "n_inputs": len(vocabulary),
            "n_val_subjects": int(keys.loc[inputs.val_ids].nunique()),
            **metrics,
        }
        stem = weights_dir() / f"e4a_seed{seed}"
        _save_network(stem, state, meta)
        inputs.preprocessor.save(stem.with_suffix(".preprocessor.json"))
        _log_pretraining(cfg, model="e4a", data_name="D2 sources", seed=seed, metrics=metrics)
        summary[str(seed)] = meta
        print(
            f"[e4] e4a seed {seed}: val masked MSE {metrics['val_loss']:.4f} "
            f"(mean baseline {metrics['val_loss_mean_baseline']:.4f}), "
            f"best epoch {metrics['best_epoch']}"
        )
    _update_results_json(PRETRAINING_JSON, "e4a", {"seeds": summary, **_mean_of(summary)})


# --------------------------------------------------------------------------------------------
# Stage: e4b (supervised pretraining on iHMP)
# --------------------------------------------------------------------------------------------


@dataclass
class SubjectGroupedPipeline(protocol.M2MPipeline):
    """M2MPipeline for a longitudinal cohort: ``labels`` are subject keys, and the early-stopping
    split keeps every subject on one side (brief section 5). As in M2MPipeline, preprocessing is
    fit on all training rows, the early-stopping ones included (DESIGN.md #2)."""

    def fit(
        self, X: pd.DataFrame, Y: pd.DataFrame, labels: pd.Series | None = None
    ) -> SubjectGroupedPipeline:
        if labels is None:
            raise ValueError("pass subject keys: the early-stopping split is grouped by subject")
        if not X.index.equals(Y.index):
            raise ValueError("microbiome and metabolome rows are not aligned by sample id")
        X_t = self.x_prep.fit(X).transform(X)
        Y_t = self.y_prep.fit(Y).transform(Y)
        fit_ids, val_ids = splits.grouped_inner_split(
            list(X.index), labels.loc[X.index], fraction=self.inner_fraction, seed=self.inner_seed
        )
        self.regressor.fit(
            X_t.loc[fit_ids].to_numpy(),
            Y_t.loc[fit_ids].to_numpy(),
            X_t.loc[val_ids].to_numpy(),
            Y_t.loc[val_ids].to_numpy(),
        )
        self.fit_ids_, self.val_ids_ = list(fit_ids), list(val_ids)
        return self


def ihmp_target_prep(cfg: Mapping[str, Any], shared: SharedTargets) -> HmdbTargetPreprocessor:
    """iHMP's shared-HMDB targets: log10(x + 1) and z-scores fit on iHMP training rows.

    The zero filter is set by ``e4b.max_zero_fraction`` (1.0 keeps every target, so the head has
    a row for each shared HMDB ID); the base threshold only counts the mostly-zero ones.
    """
    base = cfg["metabolites"]
    return HmdbTargetPreprocessor(
        feature_to_hmdb=shared.ihmp,
        hmdb_order=shared.hmdb_order,
        metabolites={**base, "max_zero_fraction": float(cfg["e4b"]["max_zero_fraction"])},
        mostly_zero_fraction=float(base["max_zero_fraction"]),
    )


def build_e4b(
    params: Mapping[str, Any],
    seed: int,
    *,
    cfg: Mapping[str, Any],
    variant: str,
    vocabulary: Sequence[str],
    shared: SharedTargets,
    weights: str,
) -> SubjectGroupedPipeline:
    """E4b network: vocabulary genera -> shared HMDB targets; "selfsup_ihmp" starts from E4a.

    ``params`` is unused (E4b has fixed settings in ``cfg["e4b"]``); it keeps the builder
    signature of the protocol, which the iHMP sanity-check CV reuses.
    """
    e4b = cfg["e4b"]
    regressor: Any = TorchMLPRegressor(
        hidden=tuple(cfg["mlp"]["hidden"]),
        dropout=float(e4b["dropout"]),
        train_cfg=TrainConfig.from_dict(e4b["train"]),
        seed=int(seed),
    )
    if variant == "selfsup_ihmp":
        trunk = load_pretrained(Path(weights), "selfsup", seed, vocabulary, carry_head=False)
        regressor = TransferRegressor(regressor, trunk)
    elif variant != "ihmp":
        raise ValueError(f"unknown E4b variant {variant!r}; expected one of {E4B_VARIANTS}")
    return SubjectGroupedPipeline(
        x_prep=common.microbe_prep(cfg, features=vocabulary),
        y_prep=ihmp_target_prep(cfg, shared),
        regressor=regressor,
        inner_fraction=float(e4b["val_fraction"]),
        inner_seed=int(seed) + int(cfg["inner_validation"]["seed_offset"]),
    )


def _fit_e4b(
    build: Callable[..., SubjectGroupedPipeline],
    seed: int,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    keys: pd.Series,
) -> SubjectGroupedPipeline:
    protocol.single_thread_torch()
    return build({}, seed).fit(X, Y, keys)


def stage_e4b(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Supervised pretraining on iHMP, one network per variant and seed, plus a sanity-check CV."""
    vocabulary = load_vocabulary()
    shared = load_shared_targets(data.t466_features())
    ihmp = load_ihmp()
    X, Y = ihmp.microbes, transfer_data.fill_not_detected(ihmp.metabolites)
    keys = transfer_data.subject_keys(ihmp)
    seeds = common.model_seeds(cfg)
    results: dict[str, Any] = {}
    for variant in E4B_VARIANTS:
        build = partial(
            build_e4b,
            cfg=cfg,
            variant=variant,
            vocabulary=vocabulary,
            shared=shared,
            weights=str(weights_dir()),
        )
        pipes = Parallel(n_jobs=n_jobs)(delayed(_fit_e4b)(build, s, X, Y, keys) for s in seeds)
        per_seed = {}
        for seed, pipe in zip(seeds, pipes, strict=True):
            per_seed[str(seed)] = _save_e4b(cfg, variant, seed, pipe, Y, vocabulary, keys)
        results[variant] = {"seeds": per_seed, **_mean_of(per_seed)}
    first = results[E4B_VARIANTS[0]]["seeds"][str(seeds[0])]
    results["n_targets"] = first["n_targets"]
    results["n_mostly_zero"] = first["n_mostly_zero"]
    results["mostly_zero_means"] = (
        f"zero in more than {cfg['metabolites']['max_zero_fraction']:.0%} of iHMP rows (kept)"
    )
    _update_results_json(PRETRAINING_JSON, "e4b", results)
    _update_results_json(
        PRETRAINING_JSON,
        "e4b_diagnostic_cv",
        ihmp_sanity_cv(cfg, X, Y, keys, vocabulary, shared, n_jobs=n_jobs),
    )


def _save_e4b(
    cfg: Mapping[str, Any],
    variant: str,
    seed: int,
    pipe: SubjectGroupedPipeline,
    Y: pd.DataFrame,
    vocabulary: Sequence[str],
    keys: pd.Series,
) -> dict[str, Any]:
    """Weights + sidecar of one E4b network; returns its summary for e4_pretraining.json."""
    regressor = pipe.regressor
    truth = pipe.transform_targets(Y)
    fit_mean = truth.loc[pipe.fit_ids_].mean(axis=0)
    metrics = {
        **_history_metrics(regressor.history_),
        # Predicting every validation target by its mean over the early-stopping fit rows.
        "val_loss_mean_baseline": float(((truth.loc[pipe.val_ids_] - fit_mean) ** 2).mean().mean()),
        "n_fit": len(pipe.fit_ids_),
        "n_val": len(pipe.val_ids_),
        "n_val_subjects": int(keys.loc[pipe.val_ids_].nunique()),
        "n_targets": len(pipe.y_prep.features_),
        "n_mostly_zero": int(pipe.y_prep.n_mostly_zero_),
        "n_init_keys_loaded": len(regressor.loaded_init_keys_),
    }
    meta = {
        "network": "MLP",
        "variant": variant,
        "seed": seed,
        "vocabulary_digest": transfer_data.vocabulary_digest(vocabulary),
        "n_inputs": len(vocabulary),
        "head_hmdb": list(pipe.y_prep.features_),  # head row i predicts head_hmdb[i]
        "loaded_init_keys": list(regressor.loaded_init_keys_),
        **metrics,
    }
    stem = weights_dir() / f"e4b_{variant}_seed{seed}"
    _save_network(stem, _detached(regressor.model_.state_dict()), meta)
    preprocessors = {"x": pipe.x_prep.to_dict(), "y": pipe.y_prep.to_dict()}
    stem.with_suffix(".preprocessor.json").write_text(json.dumps(preprocessors) + "\n")
    _log_pretraining(cfg, model=f"e4b_{variant}", data_name="iHMP", seed=seed, metrics=metrics)
    print(
        f"[e4] e4b {variant} seed {seed}: val MSE {metrics['val_loss']:.4f} "
        f"(mean baseline {metrics['val_loss_mean_baseline']:.4f}), best epoch "
        f"{metrics['best_epoch']}, {metrics['n_targets']} targets "
        f"({metrics['n_mostly_zero']} mostly zero)"
    )
    return {"seed": seed, **metrics}


def ihmp_sanity_cv(
    cfg: Mapping[str, Any],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    keys: pd.Series,
    vocabulary: Sequence[str],
    shared: SharedTargets,
    *,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Subject-grouped 5-fold CV of the "ihmp" E4b network on iHMP (seed 0).

    A sanity check that E4b learns something on its own cohort; it never selects anything.
    """
    seed = int(cfg["e4b"]["diagnostic_seed"])
    folds = splits.make_grouped_folds(
        list(X.index), keys, n_splits=int(cfg["cv"]["n_splits"]), seed=int(cfg["cv"]["seed"])
    )
    folds_per_subject = folds.groupby("group")["fold"].nunique()
    if (folds_per_subject > 1).any():
        raise splits.LeakageError("an iHMP subject spans two diagnostic CV folds")
    build = partial(
        build_e4b,
        cfg=cfg,
        variant="ihmp",
        vocabulary=vocabulary,
        shared=shared,
        weights=str(weights_dir()),
    )
    table, _ = protocol.grid_search_cv(
        build, [{}], X, Y, keys, folds, cfg, seed=seed, n_jobs=n_jobs
    )
    protocol.record_cv_table(
        table,
        experiment=EXPERIMENT,
        model=DIAGNOSTIC_MODEL,
        target_set=S176,
        data="iHMP",
        cfg=cfg,
        n_train=len(X),
    )
    row = table.iloc[0]
    fold_columns = [c for c in table.columns if c.startswith("fold") and c.endswith("_mean_rho")]
    result = {
        "note": "sanity check only; never used for any target-side choice",
        "seed": seed,
        "n_splits": int(cfg["cv"]["n_splits"]),
        "cv_mean_rho": float(row["cv_mean_rho"]),
        "cv_median_rho": float(row["cv_median_rho"]),
        "cv_mean_rho_sd_over_folds": float(row["cv_mean_rho_sd_over_folds"]),
        "fold_mean_rho": [float(row[c]) for c in fold_columns],
    }
    print(f"[e4] iHMP sanity CV (ihmp, seed {seed}): mean rho {result['cv_mean_rho']:.3f}")
    return result


# --------------------------------------------------------------------------------------------
# E4c: transfer to Franzosa
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PretrainedWeights:
    """Weights an E4c (or E4b "selfsup_ihmp") network starts from.

    ``head`` / ``head_hmdb`` are set only when an E4b head is carried over (S176): row i of the
    head predicts ``head_hmdb[i]``.
    """

    trunk: dict[str, torch.Tensor]
    head: dict[str, torch.Tensor] | None = None
    head_hmdb: tuple[str, ...] | None = None


def load_pretrained(
    weights: Path, variant: str, seed: int, vocabulary: Sequence[str], *, carry_head: bool
) -> PretrainedWeights:
    """The pretrained network of ``variant`` and ``seed`` (seed s always starts from seed s)."""
    stem = weights / f"{PRETRAINED_STEM[variant]}_seed{seed}"
    weights_file, meta_file = stem.with_suffix(".pt"), stem.with_suffix(".json")
    if not (weights_file.exists() and meta_file.exists()):
        stage = "e4a" if variant == "selfsup" else "e4b"
        raise FileNotFoundError(f"{weights_file} missing: run the e4 '{stage}' stage first")
    meta = json.loads(meta_file.read_text())
    if meta["vocabulary_digest"] != transfer_data.vocabulary_digest(vocabulary):
        # Same size, different order would load without a shape error and mean nothing.
        raise ValueError(f"{weights_file} was pretrained on another vocabulary: rerun e4a/e4b")
    # weights_only: the file is a plain state_dict; nothing in it can run code when loaded.
    state = torch.load(weights_file, map_location="cpu", weights_only=True)
    trunk = {key: value for key, value in state.items() if key.startswith(TRUNK_PREFIX)}
    if not trunk:
        raise ValueError(f"{weights_file} holds no trunk weights")
    if not carry_head:
        return PretrainedWeights(trunk=trunk)
    if HEAD_WEIGHT not in state or "head_hmdb" not in meta:
        raise ValueError(f"{weights_file} has no head to carry over")
    head = {HEAD_WEIGHT: state[HEAD_WEIGHT], HEAD_BIAS: state[HEAD_BIAS]}
    return PretrainedWeights(trunk=trunk, head=head, head_hmdb=tuple(meta["head_hmdb"]))


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Fingerprint of a network's weights (names and exact values)."""
    digest = hashlib.sha1()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def pretrained_fingerprints(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Fingerprint of every pretrained network the configured E4c variants start from."""
    fingerprints = {}
    stems = sorted({PRETRAINED_STEM[v] for v in e4c_variants(cfg) if v != "scratch"})
    for stem in stems:
        for seed in common.model_seeds(cfg):
            path = weights_dir() / f"{stem}_seed{seed}.pt"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: run the e4 'e4a' and 'e4b' stages first")
            state = torch.load(path, map_location="cpu", weights_only=True)
            fingerprints[path.name] = state_digest(state)
    return fingerprints


def selection_fingerprints(cfg: Mapping[str, Any]) -> dict[str, str]:
    """What the saved E4c choices were made with, besides PRISM itself: the vocabulary (input
    genera and their order), the config sections that define the models, and every pretrained
    network. The weights alone are not enough: a rerun `sources` stage changes every model's
    inputs, ridge and scratch included, without touching a single network file."""
    fingerprints = {VOCABULARY_JSON: transfer_data.vocabulary_digest(load_vocabulary())}
    for section in SELECTION_CONFIG_SECTIONS:
        fingerprints[f"config:{section}"] = config_hash({section: cfg[section]})
    return {**fingerprints, **pretrained_fingerprints(cfg)}


def save_fingerprints(cfg: Mapping[str, Any]) -> Path:
    """Record the vocabulary, settings and pretrained networks the model selection was made with."""
    path = paths.RESULTS / "selected" / FINGERPRINTS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection_fingerprints(cfg), indent=1, sort_keys=True) + "\n")
    return path


def check_fingerprints(cfg: Mapping[str, Any]) -> None:
    """Final scores must come from exactly what the selection was made with: retraining a
    pretrained network (say, until external scores look better), regenerating the vocabulary or
    editing a model setting after selection is refused, before the external cohort is opened."""
    path = paths.RESULTS / "selected" / FINGERPRINTS_JSON
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the e4 'select' stage first")
    saved, current = json.loads(path.read_text()), selection_fingerprints(cfg)
    changed = sorted(n for n in set(saved) | set(current) if saved.get(n) != current.get(n))
    if changed:
        raise RuntimeError(
            f"pretrained networks, vocabulary or settings changed since model selection: "
            f"{changed[:5]}; rerun the e4 'select' stage (after 'e4a' and 'e4b' if the "
            "vocabulary changed)"
        )


def head_rows(head_hmdb: Sequence[str], targets: Sequence[str]) -> list[int]:
    """Row of the pretrained head for each target, matched by HMDB ID, in target order."""
    position = {hmdb: row for row, hmdb in enumerate(head_hmdb)}
    missing = [target for target in targets if target not in position]
    if missing:
        raise KeyError(f"the pretrained head has no row for {missing[:5]}")
    return [position[target] for target in targets]


def check_pretrained_loaded(regressor: TorchMLPRegressor, *, carried_head: bool) -> None:
    """A "pretrained" network must have loaded every trunk weight (and the head if carried over);
    a silent skip would train it from scratch under a pretrained label."""
    prefixes = (TRUNK_PREFIX, HEAD_PREFIX) if carried_head else (TRUNK_PREFIX,)
    required = {key for key in regressor.model_.state_dict() if key.startswith(prefixes)}
    missing = sorted(required - set(regressor.loaded_init_keys_))
    if missing:
        raise RuntimeError(f"pretrained weights did not load {missing}")


class TransferRegressor:
    """The E1 MLP (``TorchMLPRegressor``) started from pretrained weights.

    The init state is assembled at fit time: a carried-over head keeps only the rows of the
    targets that survived this training set's zero filter, read from ``target_prep`` (the
    pipeline's own target preprocessor, fitted just before the regressor).
    """

    needs_validation = True

    def __init__(
        self,
        base: TorchMLPRegressor,
        weights: PretrainedWeights,
        target_prep: HmdbTargetPreprocessor | None = None,
    ) -> None:
        if weights.head is not None and target_prep is None:
            raise ValueError("carrying a head over needs the target preprocessor (rows by HMDB)")
        self.base = base
        self.weights = weights
        self.target_prep = target_prep

    def fit(
        self, X: np.ndarray, Y: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray
    ) -> TransferRegressor:
        self.base.set_params(init_state=self.init_state(np.shape(Y)[1]))
        self.base.fit(X, Y, X_val, Y_val)
        check_pretrained_loaded(self.base, carried_head=self.weights.head is not None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.base.predict(X)

    def init_state(self, n_targets: int) -> dict[str, torch.Tensor]:
        state = dict(self.weights.trunk)
        if self.weights.head is None:
            return state
        targets = self.target_prep.features_ if self.target_prep is not None else None
        if targets is None:
            raise RuntimeError("fit the target preprocessor before carrying the head over")
        if len(targets) != n_targets:
            raise ValueError(f"{n_targets} target columns but {len(targets)} fitted targets")
        rows = torch.as_tensor(head_rows(self.weights.head_hmdb, targets), dtype=torch.long)
        state[HEAD_WEIGHT] = self.weights.head[HEAD_WEIGHT].index_select(0, rows)
        state[HEAD_BIAS] = self.weights.head[HEAD_BIAS].index_select(0, rows)
        return state

    # What callers read from a fitted TorchMLPRegressor.
    @property
    def model_(self) -> torch.nn.Module:
        return self.base.model_

    @property
    def history_(self) -> TrainHistory:
        return self.base.history_

    @property
    def loaded_init_keys_(self) -> list[str]:
        return self.base.loaded_init_keys_


def target_prep(
    cfg: Mapping[str, Any], target_set: str, t466: Sequence[str], shared: SharedTargets
) -> Any:
    """Franzosa targets: T466 (the D1 preprocessing) or S176 (one feature per shared HMDB ID)."""
    if target_set == T466:
        return common.metabolite_prep(cfg, T466, candidates=t466)
    if target_set == S176:
        metabolites = cfg["metabolites"]
        return HmdbTargetPreprocessor(
            feature_to_hmdb=shared.franzosa,
            hmdb_order=shared.hmdb_order,
            metabolites=metabolites,
            mostly_zero_fraction=float(metabolites["max_zero_fraction"]),
        )
    raise ValueError(f"unknown E4 target set {target_set!r}; expected one of {TARGET_SETS}")


def build_e4c(
    params: Mapping[str, Any],
    seed: int,
    *,
    cfg: Mapping[str, Any],
    target_set: str,
    variant: str,
    vocabulary: Sequence[str],
    t466: Sequence[str],
    shared: SharedTargets,
    weights: str,
) -> protocol.M2MPipeline:
    """E4c pipeline: the same PRISM preprocessing for every variant (DESIGN.md #11), the E1 MLP
    with grid point ``params``, started from the pretrained network of the same ``seed``."""
    y_prep = target_prep(cfg, target_set, t466, shared)
    regressor: Any = common.mlp_regressor(params, seed, cfg)
    if variant != "scratch":
        carry_head = target_set == S176 and variant in E4B_VARIANTS
        pretrained = load_pretrained(
            Path(weights), variant, seed, vocabulary, carry_head=carry_head
        )
        regressor = TransferRegressor(regressor, pretrained, y_prep if carry_head else None)
    return protocol.make_pipeline(
        common.microbe_prep(cfg, features=vocabulary), y_prep, regressor, cfg, seed
    )


def build_d2_ridge(
    params: Mapping[str, Any],
    seed: int,
    *,
    cfg: Mapping[str, Any],
    target_set: str,
    vocabulary: Sequence[str],
    t466: Sequence[str],
    shared: SharedTargets,
) -> protocol.M2MPipeline:
    """Multi-output ridge on the same inputs and targets as E4c (baseline and permutation null)."""
    return protocol.make_pipeline(
        common.microbe_prep(cfg, features=vocabulary),
        target_prep(cfg, target_set, t466, shared),
        common.ridge_regressor(cfg),
        cfg,
        seed,
    )


def _e4c_builder(cfg: Mapping[str, Any], inputs: Inputs, target_set: str, variant: str):
    return partial(
        build_e4c,
        cfg=cfg,
        target_set=target_set,
        variant=variant,
        weights=str(weights_dir()),
        **inputs.builder_kwargs(),
    )


def _ridge_builder(cfg: Mapping[str, Any], inputs: Inputs, target_set: str):
    return partial(build_d2_ridge, cfg=cfg, target_set=target_set, **inputs.builder_kwargs())


def e4c_target_sets(cfg: Mapping[str, Any]) -> list[str]:
    return [str(t) for t in cfg["e4c"]["target_sets"]]


def e4c_variants(cfg: Mapping[str, Any]) -> list[str]:
    return [str(v) for v in cfg["e4c"]["variants"]]


# --- stage: null -----------------------------------------------------------------------------


def ridge_permutation_null(
    build: Callable[[Mapping[str, Any], int], protocol.M2MPipeline],
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> tuple[evaluate.NullResult, int]:
    """Brief section 5 null: ridge refit (preprocessing included) on microbiomes shuffled
    against metabolomes, scored on the external cohort; returns the null and its target count."""
    reference = build({}, 0).fit(X_train, Y_train)
    protocol.check_fitted_on(reference, list(X_train.index))  # it defines the external truth
    truth = reference.transform_targets(Y_test)

    def fit_predict(X_perm: pd.DataFrame, Y: pd.DataFrame, X_new: pd.DataFrame) -> pd.DataFrame:
        pipe = build({}, 0).fit(X_perm, Y)
        protocol.check_fitted_on(pipe, list(X_perm.index))
        return pipe.predict(X_new)

    ev = cfg["evaluation"]
    null = evaluate.permutation_null(
        fit_predict,
        X_train,
        Y_train,
        X_test,
        truth,
        n_permutations=int(ev["n_permutations"]),
        seed=int(ev["permutation_seed"]),
        quantile=float(ev["null_quantile"]),
    )
    return null, truth.shape[1]


def stage_null(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """D2 permutation-null thresholds (T466, S176) from the ridge baseline.

    The selection fingerprints are checked first: a vocabulary or setting changed after the
    selection phase must fail before the external cohort is opened, not after.
    """
    check_fingerprints(cfg)
    inputs = load_inputs()
    train = load_target_train()
    test = open_target_test(
        cfg, inputs.vocabulary, purpose="e4 null: D2 ridge permutation null (T466, S176)"
    )
    for target_set in e4c_target_sets(cfg):
        null, n_features = ridge_permutation_null(
            _ridge_builder(cfg, inputs, target_set),
            train.microbes,
            train.metabolites,
            test.microbes,
            test.metabolites,
            cfg,
        )
        protocol.save_null_threshold(
            data=DATA_NAME, target_set=target_set, null=null, n_features=n_features
        )
        protocol.record_null(null, experiment="e4", data=DATA_NAME, target_set=target_set,
                             n_features=n_features, cfg=cfg)
        print(f"[e4] D2 null {target_set}: threshold {null.threshold:.3f} ({n_features} targets)")


# --- stage: select ---------------------------------------------------------------------------


def stage_select(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Grid search on PRISM folds for every E4c variant and target set, plus the D2 ridge CV."""
    inputs = load_inputs()
    train = load_target_train()
    X, Y, labels = train.microbes, train.metabolites, train.meta["diagnosis"]
    folds = target_folds(cfg, list(X.index))
    grid = protocol.expand_grid(cfg["mlp"]["grid"])
    seed = int(cfg["e4c"]["selection_seed"])
    selection_fingerprints(cfg)  # fails now, not after minutes of CV, if a network is missing
    for target_set in e4c_target_sets(cfg):
        models = [(RIDGE_MODEL, _ridge_builder(cfg, inputs, target_set), [{}])]
        models += [
            (f"e4c_{variant}", _e4c_builder(cfg, inputs, target_set, variant), grid)
            for variant in e4c_variants(cfg)
        ]
        for model, build, model_grid in models:
            table, _ = protocol.grid_search_cv(
                build, model_grid, X, Y, labels, folds, cfg, seed=seed, n_jobs=n_jobs
            )
            best = protocol.select_best(table)
            protocol.save_selection(model, target_set, best, table)
            protocol.record_cv_table(
                table,
                experiment=EXPERIMENT,
                model=model,
                target_set=target_set,
                data=DATA_NAME,
                cfg=cfg,
                n_train=len(X),
            )
            print(
                f"[e4] CV {model} {target_set}: best mean rho {table['cv_mean_rho'].max():.4f} "
                f"with {json.dumps(best, sort_keys=True)}"
            )
    # Written last, once every choice is remade: a rerun interrupted midway must leave the old
    # record, which the final stages then refuse, rather than certify a mix of new and stale
    # selections as made with the current vocabulary, settings and networks (all seeds).
    save_fingerprints(cfg)


# --- stage: external -------------------------------------------------------------------------


def stage_external(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Score each selected configuration once on Validation, seed s from pretrained seed s."""
    inputs = load_inputs()
    seeds = common.model_seeds(cfg)
    # Everything that defines the evaluated models is read before the external cohort is opened:
    # a missing selection or a retrained network fails here, without an external access.
    plan = []
    for target_set in e4c_target_sets(cfg):
        # Ridge is deterministic: one seed is the whole story (its SD over seeds would be 0).
        plan.append((RIDGE_MODEL, target_set, _ridge_builder(cfg, inputs, target_set), seeds[:1]))
        plan += [
            (f"e4c_{v}", target_set, _e4c_builder(cfg, inputs, target_set, v), seeds)
            for v in e4c_variants(cfg)
        ]
    selections = {(model, ts): protocol.load_selection(model, ts) for model, ts, _, _ in plan}
    thresholds = {ts: protocol.load_null_threshold(DATA_NAME, ts) for ts in e4c_target_sets(cfg)}
    check_fingerprints(cfg)
    train = load_target_train()
    X, Y, labels = train.microbes, train.metabolites, train.meta["diagnosis"]
    test = open_target_test(
        cfg,
        inputs.vocabulary,
        purpose="e4 external: final evaluation of the selected E4c variants and d2_ridge",
    )
    for model, target_set, build, model_seeds in plan:
        selection = selections[(model, target_set)]
        result = protocol.evaluate_external(
            build,
            selection["params"],
            X,
            Y,
            labels,
            test.microbes,
            test.metabolites,
            cfg,
            seeds=model_seeds,
            null_threshold=thresholds[target_set],
            n_jobs=n_jobs,
        )
        protocol.record_external(
            result,
            experiment=EXPERIMENT,
            model=model,
            target_set=target_set,
            data=DATA_NAME,
            params=selection["params"],
            cfg=cfg,
            n_train=len(X),
            cv_mean_rho=selection["cv_mean_rho"],
        )
        print(
            f"[e4] external {model} {target_set}: mean rho "
            f"{result.aggregate['mean_rho_mean']:.4f} +/- {result.aggregate['mean_rho_sd']:.4f}"
        )


# --- stage: curve ----------------------------------------------------------------------------


def best_pretrained_variant(cfg: Mapping[str, Any], target_set: str = T466) -> str:
    """The pretrained variant with the highest *CV* mean rho (saved selections; never external)."""
    pretrained = [variant for variant in e4c_variants(cfg) if variant != "scratch"]
    if not pretrained:
        raise ValueError("no pretrained E4c variant is configured")
    cv = {v: protocol.load_selection(f"e4c_{v}", target_set)["cv_mean_rho"] for v in pretrained}
    return max(pretrained, key=lambda variant: cv[variant])  # ties: the first configured


def stage_curve(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """E5 learning curve on D2 for e4c_scratch and the best pretrained variant (by CV)."""
    target_set = str(cfg["learning_curve"]["target_set"])
    if target_set != T466:
        raise ValueError(f"the E4 learning curve is defined on T466, not {target_set}")
    # As in `external`: every saved choice is read before the external cohort is opened.
    best = best_pretrained_variant(cfg, T466)
    models = [f"e4c_{variant}" for variant in ("scratch", best)]
    params = {model: protocol.load_selection(model, T466)["params"] for model in models}
    threshold = protocol.load_null_threshold(DATA_NAME, T466)
    check_fingerprints(cfg)
    inputs = load_inputs()
    train = load_target_train()
    test = open_target_test(
        cfg,
        inputs.vocabulary,
        purpose=f"e4 curve: E5 learning curve of e4c_scratch and e4c_{best} (T466)",
    )
    for model in models:
        points = common.run_learning_curve(
            _e4c_builder(cfg, inputs, T466, model.removeprefix("e4c_")),
            params[model],
            train.microbes,
            train.metabolites,
            train.meta["diagnosis"],
            test.microbes,
            test.metabolites,
            cfg,
            model=model,
            data_name=DATA_NAME,
            experiment=EXPERIMENT,
            null_threshold=threshold,
            n_jobs=n_jobs,
            id_map=data.d1_to_d2_prism_id,
        )
        by_size = points.groupby("n_train")["mean_rho"].mean()
        print(f"[e4] curve {model}: " + ", ".join(f"n={n}: {r:.3f}" for n, r in by_size.items()))


STAGE_FUNCTIONS: dict[str, Callable[..., None]] = {
    "sources": stage_sources,
    "e4a": stage_e4a,
    "e4b": stage_e4b,
    "null": stage_null,
    "select": stage_select,
    "external": stage_external,
    "curve": stage_curve,
}


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _history_metrics(history: TrainHistory) -> dict[str, Any]:
    return {
        "val_loss": float(history.best_val_loss),
        "best_epoch": int(history.best_epoch),
        "stopped_epoch": int(history.stopped_epoch),
    }


def _tensor(values: np.ndarray) -> torch.Tensor:
    # A copy: joblib hands large arrays to workers as read-only memory maps.
    return torch.from_numpy(np.array(values, copy=True))


def _detached(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def _save_network(stem: Path, state: Mapping[str, torch.Tensor], meta: Mapping[str, Any]) -> None:
    """``<stem>.pt`` (a plain state_dict) and ``<stem>.json`` (what it was trained on)."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state), stem.with_suffix(".pt"))
    stem.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n"
    )


def _log_pretraining(
    cfg: Mapping[str, Any], *, model: str, data_name: str, seed: int, metrics: Mapping[str, Any]
) -> None:
    section = cfg["e4a"] if model == "e4a" else cfg["e4b"]
    tracking.log_run(
        experiment=EXPERIMENT,
        run_name=f"{model}|pretrain|seed{seed}",
        params={k: v for k, v in section.items() if k != "train"} | dict(section["train"]),
        metrics=metrics,
        tags={
            "model": model,
            "target_set": "" if model == "e4a" else S176,
            "data": data_name,
            "stage": "pretrain",
            "seed": seed,
            "n_train": metrics["n_fit"],
        },
        cfg=cfg,
    )


def _mean_of(per_seed: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """Mean over seeds of the validation losses (the per-seed values stay in the file)."""
    out = {}
    for key in ("val_loss", "val_loss_mean_baseline"):
        values = [float(entry[key]) for entry in per_seed.values() if key in entry]
        if values:
            out[f"{key}_mean_over_seeds"] = float(np.mean(values))
    return out


def _write_results_json(name: str, payload: Mapping[str, Any]) -> Path:
    path = paths.RESULTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def _update_results_json(name: str, key: str, value: Any) -> Path:
    path = paths.RESULTS / name
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload[key] = value
    return _write_results_json(name, payload)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
