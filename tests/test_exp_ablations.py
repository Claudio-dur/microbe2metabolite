"""Ablations A1-A4 on synthetic data: overrides, scoring spaces, reproduction of the main CV."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from m2m import data, paths, protocol
from m2m.config import deep_merge, load_config
from m2m.evaluate import spearman_per_feature
from m2m.experiments import ablations, baselines, common, mlp
from m2m.preprocess import MetabolitePreprocessor, closure, prevalent_taxa

N_TRAIN = 45  # synthetic samples 0-44 play PRISM; 45-59 play the external cohort
N_T466 = 20  # the first 20 synthetic metabolites play the annotated set
N_METABOLITES = 40

# Applied identically to the E0, E1 and ablation configs, as the real configs share base.yaml.
FAST = {
    "seeds": [0, 1],
    "evaluation": {"n_permutations": 3, "n_bootstrap": 50},
    "mlp": {
        "grid": {"lr": [1.0e-3], "dropout": [0.2], "weight_decay": [1.0e-4]},
        "train": {"max_epochs": 20, "patience": 5},
    },
}


def configs() -> SimpleNamespace:
    e0 = deep_merge(load_config(paths.CONFIGS / "e0_baselines.yaml"), FAST)
    e0["baselines"] = {"ridge": e0["baselines"]["ridge"]}  # the ablations only need ridge
    return SimpleNamespace(
        e0=e0,
        e1=deep_merge(load_config(paths.CONFIGS / "e1_mlp.yaml"), FAST),
        ablations=deep_merge(load_config(paths.CONFIGS / "ablations.yaml"), FAST),
    )


def run_main_experiments(cfgs: SimpleNamespace) -> None:
    """What the ablations build on: E0's null and ridge CV, E1's selection."""
    baselines.run(cfgs.e0, stage="null")
    baselines.run(cfgs.e0, stage="cv")
    mlp.run(cfgs.e1, stage="select")


def units_by_key(cfg: dict) -> dict[tuple[str, str, str, str], ablations.Unit]:
    return {
        (u.arm.ablation, u.arm.name, u.model, u.eval_space): u
        for u in ablations.configured_units(cfg)
    }


def fit_ridge_unit(unit: ablations.Unit, cfg: dict, d1) -> object:
    return unit.builder()(dict(cfg["ridge"]), 0).fit(d1.X_train, d1.Y_train, d1.labels_train)


@pytest.fixture
def d1(synthetic_cohort, monkeypatch):
    """Synthetic D1: train/test split, T466 = first 20 metabolites, external opens recorded."""
    X, Y, labels = synthetic_cohort["X"], synthetic_cohort["Y"], synthetic_cohort["labels"]
    train, test = X.index[:N_TRAIN], X.index[N_TRAIN:]
    opened: list[str] = []

    def load_test(purpose: str):
        opened.append(purpose)
        return X.loc[test], Y.loc[test]

    def no_real_data(*args, **kwargs):
        raise AssertionError("a test touched the real data loaders")

    monkeypatch.setattr(
        common, "load_d1_train", lambda: (X.loc[train], Y.loc[train], labels.loc[train])
    )
    monkeypatch.setattr(common, "load_d1_test", load_test)
    monkeypatch.setattr(data, "t466_features", lambda raw_dir=None: list(Y.columns[:N_T466]))
    monkeypatch.setattr(data, "load_prism", no_real_data)
    monkeypatch.setattr(data, "load_nlibd", no_real_data)
    return SimpleNamespace(
        X_train=X.loc[train], Y_train=Y.loc[train], labels_train=labels.loc[train],
        X_test=X.loc[test], Y_test=Y.loc[test], opened=opened,
    )


def test_stages_write_every_row_and_default_arms_reproduce_the_main_cv(d1, results_dir):
    cfgs = configs()
    run_main_experiments(cfgs)
    opened_before = len(d1.opened)

    ablations.run(cfgs.ablations, stage="cv")
    assert len(d1.opened) == opened_before  # CV never opens the external cohort

    # Default arms (A1 clr, A2 log10, A3 0.10) are E0 ridge / E1 MLP: identical CV mean ρ.
    main = {
        "ridge": protocol.load_selection("e0_ridge", "T466")["cv_mean_rho"],
        "mlp": protocol.load_selection("e1_mlp", "T466")["cv_mean_rho"],
    }
    defaults = [u for u in ablations.configured_units(cfgs.ablations) if u.is_main_configuration]
    assert sorted((u.arm.ablation, u.arm.name) for u in defaults if u.model == "ridge") == [
        ("A1_input_transform", "clr"), ("A2_target_transform", "log10"),
        ("A3_prevalence", "0.10"),
    ]
    for unit in defaults:
        saved = protocol.load_selection(unit.selection_name, unit.arm.label)
        assert saved["cv_mean_rho"] == main[unit.model]  # exact, not approximate

    # Default arms copy the main experiments' external rows, so those must exist first.
    baselines.run(cfgs.e0, stage="external")
    mlp.run(cfgs.e1, stage="external")
    opened_before_ablations = len(d1.opened)
    ablations.run(cfgs.ablations, stage="external")
    arm_purposes = d1.opened[opened_before_ablations:]
    # One load per non-default arm (A1 log, A2 clr, A3 0.05 and 0.20, A4): the three default
    # arms ARE E0 ridge / E1 MLP and are never scored on the external cohort a second time.
    assert len(arm_purposes) == 5
    assert any("A3_prevalence=0.20" in p for p in arm_purposes)
    assert not any("A3_prevalence=0.10" in p for p in arm_purposes)

    rows = pd.read_csv(results_dir / "ablations.csv", float_precision="round_trip")
    keys = set(zip(rows["ablation"], rows["arm"].astype(str), rows["model"], rows["eval_space"],
                   strict=True))
    expected = {(a, arm, m, "log10") for m in ("ridge", "mlp") for a, arm in [
        ("A1_input_transform", "clr"), ("A1_input_transform", "log"),
        ("A2_target_transform", "log10"), ("A2_target_transform", "clr"),
        ("A3_prevalence", "0.05"), ("A3_prevalence", "0.10"), ("A3_prevalence", "0.20"),
    ]}
    expected |= {("A2_target_transform", "clr", m, "clr") for m in ("ridge", "mlp")}
    expected |= {("A4_published_protocol", "clr_x_plus_1", m, "clr") for m in ("ridge", "mlp")}
    assert keys == expected and len(rows) == len(expected)

    assert rows["mean_rho_mean"].notna().all() and rows["cv_mean_rho"].notna().all()
    threshold = protocol.load_null_threshold("D1", "T466")
    log10_rows, clr_rows = rows[rows["eval_space"] == "log10"], rows[rows["eval_space"] == "clr"]
    assert (log10_rows["null_threshold"] == threshold).all()
    assert log10_rows["n_above_null_mean"].notna().all()
    assert clr_rows["null_threshold"].isna().all() and clr_rows["n_above_null_mean"].isna().all()
    a4 = rows[rows["ablation"] == "A4_published_protocol"]
    assert (a4["target_set"] == "T466_from_all").all() and (a4["n_features"] == N_T466).all()
    assert (rows.loc[rows["ablation"] != "A4_published_protocol", "target_set"] == "T466").all()
    assert set(rows.loc[rows["is_default_arm"], "arm"].astype(str)) == {"clr", "log10", "0.10"}
    # Default-arm rows are copies of the main summary rows; ablation runs add no summary rows.
    summary = pd.read_csv(results_dir / "summary.csv", float_precision="round_trip")
    assert not summary["model"].str.contains("|", regex=False).any()
    copied = rows[rows["external_copied_from_main"].astype(bool)]
    assert len(copied) == 6  # 3 default arms x 2 models
    for _, row in copied.iterrows():
        main_row = summary[(summary["model"] == row["model"]) & (summary["target_set"] == "T466")]
        assert row["mean_rho_mean"] == main_row["mean_rho_mean"].iloc[0]


def only_study(cfg: dict, name: str) -> dict:
    """The ablation config restricted to one study."""
    restricted = deep_merge(cfg, {})
    restricted["ablations"]["studies"] = {name: restricted["ablations"]["studies"][name]}
    return restricted


def test_a_default_arm_that_drifts_from_the_main_cv_raises(d1, results_dir):
    cfgs = configs()
    run_main_experiments(cfgs)
    drifted = only_study(deep_merge(cfgs.ablations, {"cv_seed": 1}), "A1_input_transform")
    with pytest.raises(RuntimeError, match="no longer matches the main experiment"):
        ablations.run(drifted, stage="cv")  # E1 was selected at cv_seed 0

    # The drifted unit leaves no record, so the external stage has nothing of it to score.
    unit = units_by_key(drifted)[("A1_input_transform", "clr", "mlp", "log10")]
    assert not (results_dir / "selected" / f"{unit.selection_name}__T466.json").exists()
    assert unit.run_label not in set(pd.read_csv(results_dir / "runs.csv")["model"])
    opened = len(d1.opened)
    with pytest.raises(FileNotFoundError):
        ablations.run(drifted, stage="external")
    assert len(d1.opened) == opened


def test_external_rechecks_default_arms_against_the_current_main_cv(d1, results_dir):
    cfgs = configs()
    run_main_experiments(cfgs)
    only_a1 = only_study(cfgs.ablations, "A1_input_transform")
    ablations.run(only_a1, stage="cv")
    # E1 rerun under other training settings: the same winner, another CV score.
    selection = protocol.load_selection(mlp.SELECTION, "T466")
    table = pd.read_csv(results_dir / "selected" / f"{mlp.SELECTION}__T466__cv.csv")
    table["cv_mean_rho"] += 0.01
    protocol.save_selection(mlp.SELECTION, "T466", selection["params"], table)
    opened = len(d1.opened)
    with pytest.raises(RuntimeError, match="no longer matches the main experiment"):
        ablations.run(only_a1, stage="external")
    assert len(d1.opened) == opened


def test_the_reproduction_check_only_applies_to_default_arms(d1):
    cfgs = configs()
    baselines.run(cfgs.e0, stage="cv")
    units = units_by_key(cfgs.ablations)
    main = protocol.load_selection("e0_ridge", "T466")["cv_mean_rho"]
    default = units[("A1_input_transform", "clr", "ridge", "log10")]
    ablations.check_reproduces_main(default, main, cfgs.ablations)
    with pytest.raises(RuntimeError):
        ablations.check_reproduces_main(default, main + 1e-9, cfgs.ablations)
    other = units[("A1_input_transform", "log", "ridge", "log10")]
    ablations.check_reproduces_main(other, main + 0.5, cfgs.ablations)  # a real arm may differ


def test_arm_overrides_reach_the_fitted_preprocessors(d1):
    cfg = configs().ablations
    units = units_by_key(cfg)

    def fitted(ablation: str, arm: str):
        return fit_ridge_unit(units[(ablation, arm, "ridge", "log10")], cfg, d1)

    assert fitted("A1_input_transform", "log").x_prep.transform_type == "log"
    assert fitted("A1_input_transform", "clr").x_prep.transform_type == "clr"
    relative = closure(d1.X_train)
    for arm, prevalence in (("0.05", 0.05), ("0.10", 0.10), ("0.20", 0.20)):
        pipe = fitted("A3_prevalence", arm)
        assert pipe.x_prep.min_prevalence == prevalence
        assert pipe.x_prep.features_ == prevalent_taxa(relative, 1e-4, prevalence)
        assert pipe.y_prep.transform_type == "log10"  # only the overridden setting changes
    clr = fit_ridge_unit(units[("A2_target_transform", "clr", "ridge", "clr")], cfg, d1)
    assert clr.y_prep.transform_type == "clr" and clr.x_prep.transform_type == "clr"


def test_a2_clr_arm_is_scored_against_log10_truth_of_the_same_metabolites(d1):
    cfg = configs().ablations
    units = units_by_key(cfg)
    in_clr = fit_ridge_unit(units[("A2_target_transform", "clr", "ridge", "clr")], cfg, d1)
    in_log10 = fit_ridge_unit(units[("A2_target_transform", "clr", "ridge", "log10")], cfg, d1)

    predictions = in_log10.predict(d1.X_test)
    pd.testing.assert_frame_equal(in_clr.predict(d1.X_test), predictions)  # one model
    log10 = MetabolitePreprocessor.from_config(
        cfg["metabolites"], candidate_features=list(d1.Y_test.columns[:N_T466])
    ).fit(d1.Y_train)
    truth = in_log10.transform_targets(d1.Y_test)
    pd.testing.assert_frame_equal(truth, log10.transform(d1.Y_test))

    # log10 truth ranks samples by measured intensity; per-sample CLR truth does not.
    raw = d1.Y_test[predictions.columns]
    by_intensity = spearman_per_feature(raw, predictions)
    np.testing.assert_allclose(spearman_per_feature(truth, predictions), by_intensity)
    clr_truth = in_clr.transform_targets(d1.Y_test)
    assert not np.allclose(spearman_per_feature(clr_truth, predictions), by_intensity)


def test_a4_trains_on_every_feature_and_scores_only_the_annotated_ones(d1):
    cfg = configs().ablations
    unit = units_by_key(cfg)[("A4_published_protocol", "clr_x_plus_1", "ridge", "clr")]
    assert unit.arm.label == "T466_from_all" and unit.arm.train_target_set == "TALL"
    pipe = fit_ridge_unit(unit, cfg, d1)

    assert len(pipe.y_prep.features_) == N_METABOLITES  # no zero filter: every feature an output
    annotated = list(d1.Y_test.columns[:N_T466])
    assert list(pipe.predict(d1.X_test).columns) == annotated

    def clr_x_plus_1(frame: pd.DataFrame) -> np.ndarray:
        logged = np.log(frame.to_numpy() + 1.0)
        return logged - logged.mean(axis=1, keepdims=True)  # reference: all 40 features

    train = clr_x_plus_1(d1.Y_train)
    expected = (clr_x_plus_1(d1.Y_test) - train.mean(axis=0)) / train.std(axis=0)
    truth = pipe.transform_targets(d1.Y_test)
    assert list(truth.columns) == annotated
    np.testing.assert_allclose(truth.to_numpy(), expected[:, :N_T466], rtol=1e-10, atol=1e-12)


def test_external_refuses_stale_or_missing_cv_before_opening_the_cohort(d1):
    cfgs = configs()
    run_main_experiments(cfgs)
    with pytest.raises(FileNotFoundError):
        ablations.run(cfgs.ablations, stage="external")
    assert d1.opened[-1].startswith("permutation null")  # nothing opened by the ablations

    # Records for A1 only: the external stage must fail on A2 before it scores A1's arms.
    only_a1 = only_study(cfgs.ablations, "A1_input_transform")
    ablations.run(only_a1, stage="cv")
    opened = len(d1.opened)
    with pytest.raises(FileNotFoundError):
        ablations.run(cfgs.ablations, stage="external")
    assert len(d1.opened) == opened

    # The ablation CV used E1's selection; a later re-selection on a grown grid makes it stale.
    selection = protocol.load_selection(mlp.SELECTION, "T466")
    reselected = {**selection["params"], "dropout": 0.5}
    grid_cfg = deep_merge(only_a1, {"mlp": {"grid": {"dropout": [0.2, 0.5]}}})
    table = pd.DataFrame([
        {"params": json.dumps(params, sort_keys=True), "cv_mean_rho": 0.9 if params == reselected
         else 0.1}
        for params in protocol.expand_grid(grid_cfg["mlp"]["grid"])
    ])
    protocol.save_selection(mlp.SELECTION, "T466", reselected, table)
    with pytest.raises(ValueError, match="rerun the ablations cv stage"):
        ablations.run(grid_cfg, stage="external")
    assert len(d1.opened) == opened


def test_arm_config_mistakes_are_rejected():
    cfg = configs().ablations

    def with_arm(override: dict) -> dict:
        changed = deep_merge(cfg, {})
        changed["ablations"]["studies"] = {"A9": {"arms": {"x": override}}}
        return changed

    with pytest.raises(ValueError, match="unknown microbes setting"):
        ablations.configured_arms(with_arm({"microbes": {"min_prevalance": 0.2}}))
    with pytest.raises(ValueError, match="allowed"):
        ablations.configured_arms(with_arm({"ridge": {"n_alphas": 3}}))
    with pytest.raises(ValueError, match="unknown key"):
        bad = deep_merge(cfg, {"ablations": {"studies": {"A1_input_transform": {"also": []}}}})
        ablations.configured_arms(bad)
