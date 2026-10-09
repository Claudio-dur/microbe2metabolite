"""E4 experiment: masking, grouped splits, pretrained initialisation, and every stage end to end
on a synthetic D2 world (Validation must be opened only to score saved configurations)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from m2m import data, paths, protocol, transfer_data
from m2m.config import load_config
from m2m.experiments import common, transfer
from m2m.models import MLP
from m2m.preprocess import MetabolitePreprocessor, MicrobePreprocessor
from m2m.train import TorchMLPRegressor, TrainConfig

CFG = load_config(paths.CONFIGS / "e4_transfer.yaml")
IHMP = transfer_data.IHMP
ZERO_LR = TrainConfig(lr=0.0, max_epochs=1, patience=1)  # weights stay exactly at their init


# --- masking -----------------------------------------------------------------------------------


def test_draw_mask_hides_only_present_genera_at_the_configured_ratio():
    rng = np.random.default_rng(0)
    present = torch.from_numpy(rng.random((60, 40)) < rng.uniform(0.05, 0.9, size=(60, 1)))
    present[0] = False  # nothing to hide
    present[1] = False
    present[1, 7] = True  # a single present genus is still hidden (at least one)
    mask = transfer.draw_mask(present, 0.25, torch.Generator().manual_seed(1))
    assert not (mask & ~present).any()
    n_present = present.sum(dim=1).double()
    expected = torch.where(n_present > 0, torch.floor(0.25 * n_present + 0.5).clamp(min=1), 0)
    assert torch.equal(mask.sum(dim=1).double(), expected)
    again = transfer.draw_mask(present, 0.25, torch.Generator().manual_seed(1))
    other = transfer.draw_mask(present, 0.25, torch.Generator().manual_seed(2))
    assert torch.equal(mask, again) and not torch.equal(mask, other)


class _Spy(nn.Module):
    """Returns a fixed reconstruction and remembers the input it was given."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = nn.Parameter(output.clone())
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x.detach().clone())
        return self.output.expand_as(x)


def test_masked_loss_scores_hidden_positions_only_and_hides_them_from_the_encoder():
    x = torch.arange(1.0, 13.0).reshape(2, 6)
    target = x.clone()
    mask = torch.tensor([[1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=torch.bool)
    model = _Spy(torch.zeros(1, 6)).eval()  # eval mode: the second tensor *is* the mask
    loss_fn = transfer.MaskedReconstructionLoss(0.25, torch.Generator().manual_seed(0))
    loss = loss_fn(model, (x, mask, target))
    seen = model.seen[-1]
    assert torch.equal(seen[mask], torch.zeros(int(mask.sum())))  # hidden -> 0 (the mean)
    assert torch.equal(seen[~mask], x[~mask])
    assert loss.item() == pytest.approx(float((target[mask] ** 2).mean()))
    moved = target.clone()
    moved[~mask] += 100.0  # unmasked positions never enter the loss
    assert loss_fn(model, (x, mask, moved)).item() == pytest.approx(loss.item())


def test_training_masks_are_redrawn_while_the_validation_mask_stays_fixed():
    x = torch.randn(8, 20, generator=torch.Generator().manual_seed(0))
    present = torch.rand(8, 20, generator=torch.Generator().manual_seed(1)) < 0.6
    loss_fn = transfer.MaskedReconstructionLoss(0.25, torch.Generator().manual_seed(2))
    model = _Spy(torch.zeros(1, 20)).train()
    loss_fn(model, (x, present, x))
    loss_fn(model, (x, present, x))
    first, second = [(seen == 0) & (x != 0) for seen in model.seen[-2:]]
    assert not torch.equal(first, second)  # a fresh mask per training batch
    assert not (first & ~present).any() and not (second & ~present).any()
    val_mask = transfer.draw_mask(present, 0.25, torch.Generator().manual_seed(3))
    model.eval()
    assert loss_fn(model, (x, val_mask, x)).item() == loss_fn(model, (x, val_mask, x)).item()


# --- grouped splits ------------------------------------------------------------------------------


def _subject_table(n_subjects: int = 12, per_subject: int = 4, n_genera: int = 10):
    rng = np.random.default_rng(5)
    n = n_subjects * per_subject
    ids = [f"s{i:03d}" for i in range(n)]
    values = rng.lognormal(0.0, 1.0, size=(n, n_genera))
    values[rng.random(values.shape) < 0.3] = 0.0
    values[:, 0] += 0.1
    X = pd.DataFrame(
        values / values.sum(axis=1, keepdims=True),
        index=ids,
        columns=[f"g{j}" for j in range(n_genera)],
    )
    keys = pd.Series([f"D:subj{i // per_subject}" for i in range(n)], index=ids)
    Y = pd.DataFrame(rng.uniform(1, 100, size=(n, 3)), index=ids, columns=["m1", "m2", "m3"])
    return X, Y, keys


class _RecordingRegressor:
    needs_validation = True

    def fit(self, X, Y, X_val, Y_val):
        self.shapes = (X.shape, X_val.shape)
        return self

    def predict(self, X):
        return np.zeros((len(X), 3))


def test_subject_grouped_pipeline_never_splits_a_subject():
    X, Y, keys = _subject_table()
    pipe = transfer.SubjectGroupedPipeline(
        x_prep=MicrobePreprocessor(),
        y_prep=MetabolitePreprocessor(),
        regressor=_RecordingRegressor(),
        inner_fraction=0.25,
        inner_seed=3,
    ).fit(X, Y, keys)
    fit_subjects, val_subjects = set(keys[pipe.fit_ids_]), set(keys[pipe.val_ids_])
    assert fit_subjects.isdisjoint(val_subjects)
    assert len(val_subjects) == 3  # 25% of 12 subjects
    assert sorted(pipe.fit_ids_ + pipe.val_ids_) == sorted(X.index)
    assert pipe.x_prep.fit_samples_ == list(X.index)  # DESIGN.md #2: all training rows
    with pytest.raises(ValueError, match="subject"):
        pipe.fit(X, Y)


def test_autoencoder_split_and_input_scaling_use_training_subjects_only():
    X, _, keys = _subject_table()
    vocabulary = list(X.columns[:8])
    inputs = transfer.autoencoder_data(X, keys, vocabulary, CFG, seed=0)
    assert set(keys[inputs.fit_ids]).isdisjoint(set(keys[inputs.val_ids]))
    assert inputs.preprocessor.fit_samples_ == inputs.fit_ids
    assert inputs.preprocessor.features_ == vocabulary
    present_val = X.loc[inputs.val_ids, vocabulary].to_numpy() > 0
    assert inputs.val_mask.any() and not (inputs.val_mask & ~present_val).any()
    again = transfer.autoencoder_data(X, keys, vocabulary, CFG, seed=0)
    assert np.array_equal(inputs.val_mask, again.val_mask)  # the fixed, seeded mask


# --- pretrained initialisation -------------------------------------------------------------------


def _mlp_state(n_in: int, n_out: int, hidden=(8, 4), seed: int = 99) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return MLP(n_in, n_out, hidden=hidden).state_dict()


def _regressor(seed: int = 0) -> TorchMLPRegressor:
    return TorchMLPRegressor(hidden=(8, 4), dropout=0.0, train_cfg=ZERO_LR, seed=seed)


def _hmdb_problem(rng: np.random.Generator, n: int = 40):
    """Five HMDB targets; H2 and H4 are mostly zero, so the zero filter drops them."""
    ids = [f"p{i}" for i in range(n)]
    X = pd.DataFrame(
        rng.lognormal(size=(n, 6)) + 0.01, index=ids, columns=[f"g{j}" for j in range(6)]
    )
    Y = pd.DataFrame(
        rng.uniform(1, 100, size=(n, 5)), index=ids, columns=["f1", "f2", "f3", "f4", "f5"]
    )
    Y.iloc[: int(0.8 * n), [1, 3]] = 0.0
    labels = pd.Series(np.array(["CD", "UC", "Control"])[np.arange(n) % 3], index=ids)
    prep = transfer_data.HmdbTargetPreprocessor(
        feature_to_hmdb={f"f{i}": f"H{i}" for i in range(1, 6)},
        hmdb_order=["H1", "H2", "H3", "H4", "H5"],
        metabolites=CFG["metabolites"],
        mostly_zero_fraction=0.5,
    )
    return X, Y, labels, prep


def test_head_carry_over_maps_rows_by_hmdb_when_some_targets_are_filtered(rng):
    X, Y, labels, y_prep = _hmdb_problem(rng)
    pretrained = _mlp_state(6, 5)
    weights = transfer.PretrainedWeights(
        trunk={k: v for k, v in pretrained.items() if k.startswith("trunk.")},
        head={k: v for k, v in pretrained.items() if k.startswith("head.")},
        head_hmdb=("H1", "H2", "H3", "H4", "H5"),
    )
    regressor = transfer.TransferRegressor(_regressor(), weights, target_prep=y_prep)
    pipe = protocol.make_pipeline(MicrobePreprocessor(), y_prep, regressor, CFG, seed=0)
    pipe.fit(X, Y, labels)
    assert y_prep.features_ == ["H1", "H3", "H5"]
    state = regressor.model_.state_dict()
    assert torch.equal(state["head.weight"], pretrained["head.weight"][[0, 2, 4]])
    assert torch.equal(state["head.bias"], pretrained["head.bias"][[0, 2, 4]])
    assert torch.equal(state["trunk.3.weight"], pretrained["trunk.3.weight"])
    assert list(pipe.predict(X).columns) == ["H1", "H3", "H5"]


def test_head_rows_follow_the_targets_not_the_head_order():
    assert transfer.head_rows(["H1", "H2", "H3"], ["H3", "H1"]) == [2, 0]
    with pytest.raises(KeyError):
        transfer.head_rows(["H1"], ["H9"])


def test_a_pretrained_trunk_really_changes_the_starting_weights(rng):
    X = rng.normal(size=(30, 6))
    Y = rng.normal(size=(30, 2))
    pretrained = _mlp_state(6, 2)
    trunk = {k: v for k, v in pretrained.items() if k.startswith("trunk.")}
    scratch = _regressor().fit(X[:20], Y[:20], X[20:], Y[20:])
    warm = transfer.TransferRegressor(_regressor(), transfer.PretrainedWeights(trunk=trunk))
    warm.fit(X[:20], Y[:20], X[20:], Y[20:])
    for key, value in trunk.items():
        assert torch.equal(warm.model_.state_dict()[key], value)
        assert not torch.equal(scratch.model_.state_dict()[key], value)
    assert sorted(warm.loaded_init_keys_) == sorted(trunk)
    # Same seed, same head initialisation: only the trunk differs between the two.
    assert torch.equal(
        warm.model_.state_dict()["head.bias"], scratch.model_.state_dict()["head.bias"]
    )


def test_a_trunk_that_does_not_load_completely_is_an_error(rng):
    X, Y = rng.normal(size=(30, 6)), rng.normal(size=(30, 2))
    first_layer_only = {k: v for k, v in _mlp_state(6, 2).items() if k.startswith("trunk.0.")}
    warm = transfer.TransferRegressor(_regressor(), transfer.PretrainedWeights(first_layer_only))
    with pytest.raises(RuntimeError, match="did not load"):
        warm.fit(X[:20], Y[:20], X[20:], Y[20:])


def test_carrying_a_head_needs_the_target_preprocessor():
    state = _mlp_state(6, 2)
    weights = transfer.PretrainedWeights(trunk=state, head=state, head_hmdb=("H1", "H2"))
    with pytest.raises(ValueError, match="target preprocessor"):
        transfer.TransferRegressor(_regressor(), weights)


def test_load_pretrained_checks_the_vocabulary_and_returns_trunk_and_head(tmp_path):
    vocabulary = ["a", "b", "c"]
    state = _mlp_state(3, 2)
    meta = {
        "vocabulary_digest": transfer_data.vocabulary_digest(vocabulary),
        "head_hmdb": ["H1", "H2"],
    }
    transfer._save_network(tmp_path / "e4b_ihmp_seed3", state, meta)
    loaded = transfer.load_pretrained(tmp_path, "ihmp", 3, vocabulary, carry_head=True)
    assert set(loaded.trunk) == {k for k in state if k.startswith("trunk.")}
    assert loaded.head_hmdb == ("H1", "H2")
    assert torch.equal(loaded.head["head.weight"], state["head.weight"])
    trunk_only = transfer.load_pretrained(tmp_path, "ihmp", 3, vocabulary, carry_head=False)
    assert trunk_only.head is None
    with pytest.raises(ValueError, match="another vocabulary"):
        transfer.load_pretrained(tmp_path, "ihmp", 3, ["c", "b", "a"], carry_head=False)
    with pytest.raises(FileNotFoundError, match="e4b"):
        transfer.load_pretrained(tmp_path, "ihmp", 4, vocabulary, carry_head=False)


# --- configuration and selection rules -----------------------------------------------------------


def test_config_guards_the_trunk_shape_and_the_mask_ratio():
    transfer.check_config(CFG)
    bad = copy.deepcopy(CFG)
    bad["e4a"]["hidden"] = [128, 64]
    with pytest.raises(ValueError, match="mlp.hidden"):
        transfer.check_config(bad)
    bad = copy.deepcopy(CFG)
    bad["e4a"]["mask_ratio"] = 0.4
    with pytest.raises(ValueError, match="mask_ratio"):
        transfer.check_config(bad)


def test_best_pretrained_variant_is_chosen_by_cv_and_never_scratch():
    cv = {"scratch": 0.9, "selfsup": 0.2, "ihmp": 0.25, "selfsup_ihmp": 0.1}
    for variant, rho in cv.items():
        table = pd.DataFrame({"params": [json.dumps({})], "cv_mean_rho": [rho]})
        protocol.save_selection(f"e4c_{variant}", "T466", {}, table)
    assert transfer.best_pretrained_variant(CFG, "T466") == "ihmp"


# --- every stage on a synthetic D2 world ---------------------------------------------------------

N_GENERA = 14
T466 = [f"F{i:02d}" for i in range(12)]
FEATURES = [f"F{i:02d}" for i in range(16)]
# Franzosa features -> HMDB (F09 is not in iHMP; F14 is outside T466); iHMP compounds -> HMDB.
FRANZOSA_HMDB = {
    "F00": "H1",
    "F01": "H1",
    "F02": "H2",
    "F03": "H3",
    "F04": "H4",
    "F05": "H4",
    "F06": "H5",
    "F07": "H6",
    "F09": "H9",
    "F14": "H7",
}
IHMP_HMDB = {
    "c0": "H1",
    "c1": "H1",
    "c2": "H2",
    "c3": "H3",
    "c4": "H4",
    "c5": "H5",
    "c6": "H6",
    "c7": "H7",
    "c8": "H8",
}
SIZES = {
    IHMP: (40, 4),
    "YACHIDA_CRC_2019": (30, 1),
    "WANG_ESRD_2020": (30, 1),
    "ERAWIJANTARI_GASTRIC_CANCER_2020": (20, 1),
    "MARS_IBS_2020": (30, 5),
}


def _genus(i: int) -> str:
    return f"d__Bacteria;p__P;c__C;o__O;f__F;g__G{i:02d}"


def _microbes(rng, ids, genera_idx) -> pd.DataFrame:
    values = rng.lognormal(0.0, 1.2, size=(len(ids), len(genera_idx)))
    values[rng.random(values.shape) < 0.25] = 0.0
    values[:, 0] += 0.05
    return pd.DataFrame(
        values / values.sum(axis=1, keepdims=True),
        index=ids,
        columns=[_genus(i) for i in genera_idx],
    )


def _signal(rng, microbes: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Intensities driven log-linearly by the first five genera (the planted signal)."""
    logged = np.log(microbes.to_numpy()[:, :5] + 1e-4)
    driven = (logged - logged.mean(0)) @ weights
    return np.exp(driven / max(driven.std(), 1e-9) + rng.normal(scale=0.3, size=driven.shape))


def _meta(ids, subjects, dataset, diagnosis) -> pd.DataFrame:
    return pd.DataFrame(
        {"subject": subjects, "diagnosis": diagnosis, "dataset": dataset}, index=ids
    )


@dataclass
class World:
    sources: dict
    ihmp: data.Cohort
    prism: data.Cohort
    validation: data.Cohort
    d1_prism: data.Cohort
    maps: dict
    validation_purposes: list = field(default_factory=list)


def _world(target_seed: int = 0) -> World:
    rng = np.random.default_rng(42)
    sources = {}
    for k, (dataset, (n, per_subject)) in enumerate(SIZES.items()):
        ids = [f"{dataset[:3]}{i:03d}" for i in range(n)]
        subjects = [f"{dataset[:3]}subj{i // per_subject}" for i in range(n)]
        genera_idx = [i for i in range(N_GENERA) if i != 6 + k]  # each dataset lacks one genus
        microbes = _microbes(rng, ids, genera_idx)
        sources[dataset] = data.Cohort(dataset, microbes, None, _meta(ids, subjects, dataset, "x"))
    ihmp = sources[IHMP]
    mets = _signal(rng, ihmp.microbes, rng.normal(size=(5, len(IHMP_HMDB)))) * 1e4
    mets[rng.random(mets.shape) < 0.05] = np.nan  # non-detects
    ihmp = data.Cohort(
        IHMP,
        ihmp.microbes,
        pd.DataFrame(mets, index=ihmp.meta.index, columns=list(IHMP_HMDB)),
        ihmp.meta,
    )

    trng = np.random.default_rng(100 + target_seed)
    diagnosis = np.array(["CD", "UC", "Control"])[np.arange(84) % 3]
    weights = trng.normal(size=(5, len(FEATURES)))  # the same biology in PRISM and Validation

    def target(ids, labels):
        microbes = _microbes(trng, ids, list(range(N_GENERA + 1)))  # one genus no source has
        values = _signal(trng, microbes, weights) * 1e3
        values[trng.random(len(ids)) < 0.7, 6] = 0.0  # F06 (H5) mostly zero: filtered out
        mets = pd.DataFrame(values, index=ids, columns=FEATURES)
        return microbes, mets, _meta(ids, ids, data.TARGET_DATASET, labels)

    prism_ids = [f"PRISM.{7000 + i}" for i in range(60)]
    prism = data.Cohort("FRANZOSA_PRISM", *target(prism_ids, diagnosis[:60]))
    validation_ids = [f"Validation.V{i:02d}" for i in range(24)]
    validation = data.Cohort("FRANZOSA_Validation", *target(validation_ids, diagnosis[60:]))
    d1_ids = [f"PRISM|{7000 + i}" for i in range(60)]
    d1_meta = _meta(d1_ids, d1_ids, "D1", diagnosis[:60])
    d1_prism = data.Cohort("PRISM", prism.microbes.set_axis(d1_ids), None, d1_meta)
    franzosa_map = pd.DataFrame(
        {"HMDB": [FRANZOSA_HMDB.get(f) for f in FEATURES]},
        index=[f"{f}: compound {f}" for f in FEATURES],
    )
    ihmp_map = pd.DataFrame(
        {"HMDB": list(IHMP_HMDB.values()), "Method": ["C18-neg", "HILIC-pos", "C8-pos"] * 3},
        index=list(IHMP_HMDB),
    )
    maps = {data.TARGET_DATASET: franzosa_map, IHMP: ihmp_map}
    return World(sources, ihmp, prism, validation, d1_prism, maps)


def _install(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    def load_d2(dataset, raw_dir=None, **kwargs):
        assert dataset == IHMP, "E4 reads only iHMP with load_d2"
        return world.ihmp

    def load_franzosa(
        raw_dir=None, *, part, with_metabolites=True, final_evaluation=False, purpose=""
    ):
        if part == "PRISM":
            return world.prism
        if final_evaluation is not True or not purpose.strip():
            raise data.ExternalTestSetAccessError("guarded")
        world.validation_purposes.append(purpose)
        return world.validation

    def load_d2_metadata(dataset, raw_dir=None):
        assert dataset == data.TARGET_DATASET
        return world.prism.meta

    monkeypatch.delenv("M2M_DRY_RUN", raising=False)
    monkeypatch.setattr(data, "load_sources", lambda *args, **kwargs: dict(world.sources))
    monkeypatch.setattr(data, "load_d2", load_d2)
    monkeypatch.setattr(data, "load_franzosa", load_franzosa)
    monkeypatch.setattr(data, "load_d2_metadata", load_d2_metadata)
    monkeypatch.setattr(data, "load_d2_mtb_map", lambda dataset, raw_dir=None: world.maps[dataset])
    monkeypatch.setattr(data, "t466_features", lambda raw_dir=None: list(T466))
    monkeypatch.setattr(data, "load_prism", lambda raw_dir=None: world.d1_prism)


def _small_cfg() -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["seeds"] = [0, 1]
    cfg["mlp"]["hidden"] = [8, 4]
    cfg["mlp"]["grid"] = {"lr": [1.0e-3], "dropout": [0.2], "weight_decay": [1.0e-4]}
    cfg["mlp"]["train"].update(max_epochs=3, patience=2)
    cfg["e4a"].update(hidden=[8, 4], decoder_hidden=8)
    cfg["e4a"]["train"].update(max_epochs=3, patience=2, batch_size=16)
    cfg["e4b"]["train"].update(max_epochs=3, patience=2)
    cfg["evaluation"].update(n_permutations=2, n_bootstrap=20)
    cfg["learning_curve"]["sizes"] = [20, 40]
    cfg["mlflow"]["enabled"] = False
    return cfg


def test_every_stage_runs_and_validation_is_opened_only_to_score(monkeypatch, results_dir):
    world = _world()
    _install(world, monkeypatch)
    cfg = _small_cfg()
    opened = {}
    for stage in transfer.STAGES:
        world.validation_purposes.clear()
        transfer.run(cfg, stage=stage)
        opened[stage] = list(world.validation_purposes)
    for stage in ("sources", "e4a", "e4b", "select"):
        assert opened[stage] == [], f"{stage} opened the external cohort"
    for stage in ("null", "external", "curve"):
        assert len(opened[stage]) == 1 and opened[stage][0].startswith(f"e4 {stage}")

    sources = json.loads((results_dir / "e4_sources.json").read_text())
    assert sources["vocabulary"]["size"] == N_GENERA  # the target-only genus is not in it
    assert sources["shared_hmdb"]["n_hmdb"] == 6  # H1-H6: H7 only outside T466, H8/H9 one-sided
    assert sources["source_target_disjointness"]["shared_sample_or_subject_ids"] == 0
    weights = paths.DATA_PROCESSED / "e4"
    for stem in ("e4a", "e4b_ihmp", "e4b_selfsup_ihmp"):
        for seed in (0, 1):
            assert (weights / f"{stem}_seed{seed}.pt").exists()
    head = json.loads((weights / "e4b_ihmp_seed0.json").read_text())["head_hmdb"]
    assert head == ["H1", "H2", "H3", "H4", "H5", "H6"]  # E4b keeps every shared target

    pretraining = json.loads((results_dir / "e4_pretraining.json").read_text())
    assert set(pretraining) == {"e4a", "e4b", "e4b_diagnostic_cv"}
    assert np.isfinite(pretraining["e4b_diagnostic_cv"]["cv_mean_rho"])

    summary = pd.read_csv(results_dir / "summary.csv")
    models = {"d2_ridge", "e4c_scratch", "e4c_selfsup", "e4c_ihmp", "e4c_selfsup_ihmp"}
    assert set(zip(summary["model"], summary["target_set"], strict=True)) == {
        (m, t) for m in models for t in ("T466", "S176")
    }
    assert (summary["data"] == "D2").all()
    s176 = summary[summary["target_set"] == "S176"]
    assert (s176["n_features_mean"] == 5).all()  # H5 dropped by PRISM's zero filter
    # The planted signal is shared by PRISM and Validation: a correctly paired external
    # evaluation recovers it (a sample or target misalignment would give rho near 0).
    ridge = summary[summary["model"] == "d2_ridge"].set_index("target_set")
    assert (ridge["mean_rho_mean"] > 0.3).all()
    nulls = pd.read_csv(results_dir / "null_thresholds.csv")
    assert set(zip(nulls["data"], nulls["target_set"], strict=True)) == {
        ("D2", "T466"),
        ("D2", "S176"),
    }
    curve = pd.read_csv(results_dir / "learning_curve.csv")
    best = transfer.best_pretrained_variant(cfg)
    assert set(curve["model"]) == {"e4c_scratch", f"e4c_{best}"}
    assert set(curve["n_train"]) == {20, 40} and (curve["data"] == "D2").all()
    # E4c was scored on the saved D1 folds, mapped to D2 IDs.
    folds = pd.read_csv(paths.SPLITS / "prism_folds.csv")
    assert set(folds["sample_id"]) == {f"PRISM|{7000 + i}" for i in range(60)}


def test_the_vocabulary_does_not_change_with_the_target_data(monkeypatch, results_dir):
    vocabularies, coverage = [], []
    for target_seed in (0, 1):
        world = _world(target_seed=target_seed)
        _install(world, monkeypatch)
        transfer.run(_small_cfg(), stage="sources")
        vocabularies.append(transfer.load_vocabulary())
        report = json.loads((results_dir / "e4_sources.json").read_text())
        coverage.append(report["franzosa_prism_coverage"]["abundance_share_in_vocabulary"])
    assert vocabularies[0] == vocabularies[1]
    assert coverage[0] != coverage[1]  # the target data did change


def _run_stages(cfg: dict, stages: tuple[str, ...]) -> None:
    for stage in stages:
        transfer.run(cfg, stage=stage)


def test_final_stages_fail_before_opening_validation_without_a_saved_selection(
    monkeypatch, results_dir
):
    world = _world()
    _install(world, monkeypatch)
    cfg = _small_cfg()
    _run_stages(cfg, ("sources", "e4a", "e4b"))
    world.validation_purposes.clear()
    # Every evaluation stage, the null included, needs the saved selection first.
    for stage in ("null", "external", "curve"):
        with pytest.raises(FileNotFoundError, match="select"):
            transfer.run(cfg, stage=stage)
    assert world.validation_purposes == []


def test_final_stages_refuse_pretrained_networks_changed_after_selection(monkeypatch, results_dir):
    world = _world()
    _install(world, monkeypatch)
    cfg = _small_cfg()
    _run_stages(cfg, ("sources", "e4a", "e4b", "select", "null"))
    # Retraining a pretrained network after selection (e.g. a new E4a run) must not slip into
    # the final scores: the selection was made with the old one.
    path = paths.DATA_PROCESSED / "e4" / "e4a_seed1.pt"
    state = torch.load(path, weights_only=True)
    state["trunk.0.bias"] = state["trunk.0.bias"] + 1.0
    torch.save(state, path)
    world.validation_purposes.clear()
    for stage in ("external", "curve"):
        with pytest.raises(RuntimeError, match="changed since model selection"):
            transfer.run(cfg, stage=stage)
    assert world.validation_purposes == []


def test_retraining_the_pretrained_networks_reproduces_them_exactly(monkeypatch, results_dir):
    world = _world()
    _install(world, monkeypatch)
    cfg = _small_cfg()
    _run_stages(cfg, ("sources", "e4a", "e4b"))
    before = transfer.pretrained_fingerprints(cfg)
    _run_stages(cfg, ("e4a", "e4b"))
    assert transfer.pretrained_fingerprints(cfg) == before


def test_final_stages_refuse_a_vocabulary_changed_after_selection(monkeypatch, results_dir):
    world = _world()
    _install(world, monkeypatch)
    cfg = _small_cfg()
    _run_stages(cfg, ("sources", "e4a", "e4b", "select", "null"))
    # E.g. `--stage sources` rerun after a data or threshold change: the networks are untouched,
    # but every model (ridge and scratch included) would now read other input genera.
    transfer.save_vocabulary(transfer.load_vocabulary()[:-1])
    world.validation_purposes.clear()
    for stage in ("external", "curve"):
        with pytest.raises(RuntimeError, match="changed since model selection"):
            transfer.run(cfg, stage=stage)
    assert world.validation_purposes == []
    assert not (results_dir / "summary.csv").exists()


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("mlp", "grid", {"lr": [3.0e-4], "dropout": [0.5], "weight_decay": [1.0e-2]}),
        ("ridge", "n_alphas", 5),
        ("microbes", "transform", "log"),
    ],
)
def test_final_stages_refuse_settings_changed_after_selection(
    monkeypatch, results_dir, section, key, value
):
    world = _world()
    _install(world, monkeypatch)
    cfg = _small_cfg()
    _run_stages(cfg, ("sources", "e4a", "e4b", "select", "null"))
    changed = copy.deepcopy(cfg)
    changed[section][key] = value  # the saved choices were made under the old setting
    world.validation_purposes.clear()
    with pytest.raises(RuntimeError, match=f"changed since model selection.*{section}"):
        transfer.run(changed, stage="external")
    assert world.validation_purposes == []


def test_an_interrupted_selection_never_certifies_the_choices_it_did_not_remake(
    monkeypatch, results_dir
):
    world = _world()
    _install(world, monkeypatch)
    cfg = _small_cfg()
    _run_stages(cfg, ("sources", "e4a", "e4b", "select", "null"))
    changed = copy.deepcopy(cfg)
    changed["mlp"]["grid"]["lr"] = [3.0e-4]
    real_grid_search, calls = protocol.grid_search_cv, []

    def crash_on_third_model(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt  # e.g. stopped by hand: later selections keep the old grid
        return real_grid_search(*args, **kwargs)

    monkeypatch.setattr(protocol, "grid_search_cv", crash_on_third_model)
    with pytest.raises(KeyboardInterrupt):
        transfer.run(changed, stage="select")
    world.validation_purposes.clear()
    with pytest.raises(RuntimeError, match="changed since model selection"):
        transfer.run(changed, stage="external")
    assert world.validation_purposes == []


def test_a_dry_run_never_writes_into_the_committed_results(monkeypatch, tmp_path):
    world = _world()
    _install(world, monkeypatch)
    repo = tmp_path / "repo"
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    monkeypatch.setattr(paths, "RESULTS", repo / "results")  # the committed results/
    monkeypatch.setenv("M2M_DRY_RUN", "1")
    # sources, e4a and e4b load no PRISM split, so nothing else would refuse them.
    for stage in transfer.STAGES:
        with pytest.raises(RuntimeError, match="scratch"):
            transfer.run(_small_cfg(), stage=stage)
    assert not (repo / "results").exists()


def test_a_dry_run_scores_the_prism_hold_out_and_never_opens_validation(
    monkeypatch, results_dir
):
    world = _world()
    _install(world, monkeypatch)
    monkeypatch.setenv("M2M_DRY_RUN", "1")
    cfg = _small_cfg()
    _run_stages(cfg, transfer.STAGES)
    assert world.validation_purposes == []  # the fake guarded loader records every opening
    pseudo_train, pseudo_test = common.dry_run_ids()
    summary = pd.read_csv(results_dir / "summary.csv")
    assert (summary["n_test"] == len(pseudo_test)).all()
    assert (summary["n_train"] == len(pseudo_train)).all()
    folds = pd.read_csv(paths.SPLITS / "prism_folds.csv")
    assert sorted(folds["sample_id"]) == sorted(pseudo_train)  # selection never saw the hold-out
