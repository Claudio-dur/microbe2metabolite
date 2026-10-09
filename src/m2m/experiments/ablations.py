"""Ablations A1-A4 on D1: how much do the preprocessing choices move the result?

Each arm is a config override deep-merged into the experiment config (``config.deep_merge``).
Everything it does not override (folds, model seeds, hyperparameters) is the main protocol's, so
an arm whose override changes nothing (a "default arm") is the main configuration itself: its CV
mean ρ must equal E0 ridge's / E1 MLP's. The cv stage checks that before it records the unit, and
the external stage checks the saved record again before it opens the external cohort.

- A1_input_transform: ``microbes.transform`` clr (main) vs log.
- A2_target_transform: ``metabolites.transform`` log10 (main) vs clr (MiMeNet). The CLR arm is
  scored twice: against CLR truth (eval_space "clr", MiMeNet-style) and against log10 truth of the
  same metabolites (eval_space "log10"). Per-sample CLR is not monotone in a metabolite's
  intensity (it subtracts the sample's mean log), so the two spaces rank samples differently.
- A3_prevalence: ``microbes.min_prevalence`` 0.05 / 0.10 (main) / 0.20.
- A4_published_protocol (DESIGN.md decision #18): CLR(x + 1) targets over all features with no zero
  filter, trained on all 8,848 outputs, ρ reported over the 466 annotated ("T466_from_all"). The
  inputs keep our preprocessing: the arm reproduces the published evaluation, not their models.

Models per arm: ridge (base.yaml ``ridge``) and the E1 MLP with its saved selection (the TALL one
for A4). Stages, run in order by ``python -m m2m.run configs/ablations.yaml [--stage S]``:

- ``cv``: PRISM CV per arm × model × eval space at ``cv_seed`` → runs.csv,
  results/selected/ablation__*.json and results/ablations.csv rows (CV columns only).
- ``external``: refit on all of PRISM and score NLIBD (one load per arm) → the full
  results/ablations.csv rows; runs.csv gets the per-seed rows, summary.csv nothing.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from m2m import evaluate, paths, protocol, tracking
from m2m.config import deep_merge
from m2m.experiments import common, mlp
from m2m.preprocess import METABOLITE_TRANSFORMS, MetabolitePreprocessor

EXPERIMENT = "ablations"
DATA = "D1"
STAGES = ("cv", "external")
ABLATIONS_CSV = "ablations.csv"
ROW_KEY = ("ablation", "arm", "model", "target_set", "eval_space")
# The preprocessing sections an arm may override; anything else would not be an ablation of
# preprocessing (and a misspelt section would silently leave the arm equal to the main protocol).
OVERRIDABLE = ("microbes", "metabolites")
STUDY_KEYS = ("arms", "also_score_in", "train_target_set", "score_target_set", "target_set_label")
MODEL_BUILDERS = {"ridge": common.build_ridge, "mlp": common.build_mlp}
# Where each model's main CV mean ρ is saved (E0 / E1), for the default-arm reproduction check.
MAIN_SELECTIONS = {"ridge": "e0_ridge", "mlp": mlp.SELECTION}
# summary.csv names of the main experiments a default arm reproduces exactly.
MAIN_SUMMARY_MODELS = {"ridge": "ridge", "mlp": "mlp"}


def run(cfg: dict[str, Any], *, stage: str | None = None, n_jobs: int = 1) -> None:
    """Run the requested stage, or all of them in order."""
    actions = {"cv": stage_cv, "external": stage_external}
    for name in common.stages_to_run(stage, STAGES):
        start = time.perf_counter()
        actions[name](cfg, n_jobs=n_jobs)
        _say(f"stage {name} done in {time.perf_counter() - start:.1f}s")


# --------------------------------------------------------------------------------------------
# Scoring a pipeline on a view of its targets
# --------------------------------------------------------------------------------------------


@dataclass
class ScoredView(protocol.M2MPipeline):
    """An M2MPipeline trained as usual but scored on a view of its targets.

    - ``score_candidates``: score only these of the trained outputs (A4: the 466 annotated of all
      8,848), in this order, keeping those the training filter retained.
    - ``truth_prep``: score against truth from another target preprocessor (A2: log10 truth for a
      model trained on CLR targets). It is fit in ``fit`` on exactly the training rows the
      model's own target preprocessor sees, and must retain the same targets.

    ``predict`` and ``transform_targets`` return the scored view, so the protocol's CV and
    external evaluation score it like any other pipeline.
    """

    truth_prep: MetabolitePreprocessor | None = None
    score_candidates: tuple[str, ...] | None = None

    @classmethod
    def wrap(cls, pipe: protocol.M2MPipeline, **view: Any) -> ScoredView:
        own = {f.name: getattr(pipe, f.name) for f in dataclasses.fields(protocol.M2MPipeline)}
        return cls(**own, **view)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, labels: pd.Series | None = None) -> ScoredView:
        super().fit(X, Y, labels)
        if self.truth_prep is not None:
            self.truth_prep.fit(Y)
            if list(self.truth_prep.features_) != list(self.y_prep.features_):
                raise ValueError("the scoring-space preprocessor retained different targets")
        self.scored_features_ = self._scored_features()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return super().predict(X)[self.scored_features_]

    def transform_targets(self, Y: pd.DataFrame) -> pd.DataFrame:
        prep = self.y_prep if self.truth_prep is None else self.truth_prep
        return prep.transform(Y)[self.scored_features_]

    def _scored_features(self) -> list[str]:
        trained = list(self.y_prep.features_)
        if self.score_candidates is None:
            return trained
        kept = set(trained)
        scored = [f for f in self.score_candidates if f in kept]
        if not scored:
            raise ValueError("none of the scored targets is among the trained outputs")
        return scored


def build_unit(
    params: Mapping[str, Any],
    seed: int,
    *,
    cfg: Mapping[str, Any],
    model: str,
    train_target_set: str,
    score_target_set: str,
    eval_space: str,
) -> protocol.M2MPipeline:
    """The main protocol's ridge / MLP pipeline under an arm's config, scored as the unit says."""
    pipe = MODEL_BUILDERS[model](params, seed, cfg=cfg, target_set=train_target_set)
    truth_prep = None
    if eval_space != cfg["metabolites"]["transform"]:
        # Same candidates, zero filter and training rows; only the transform of the truth changes.
        truth_cfg = deep_merge(dict(cfg), {"metabolites": {"transform": eval_space}})
        truth_prep = common.metabolite_prep(truth_cfg, train_target_set)
    score_candidates = None
    if score_target_set != train_target_set:
        score_candidates = tuple(common.target_candidates(score_target_set))
    if truth_prep is None and score_candidates is None:
        return pipe  # untouched, so a default arm is literally the E0 / E1 pipeline
    return ScoredView.wrap(pipe, truth_prep=truth_prep, score_candidates=score_candidates)


# --------------------------------------------------------------------------------------------
# Arms and evaluation units
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Arm:
    """One configuration of one ablation, on one target set."""

    ablation: str
    name: str
    override: dict[str, Any]
    cfg: dict[str, Any]  # the experiment config with the override merged in
    train_target_set: str  # what the models are trained on (TALL for A4)
    score_target_set: str  # what ρ is reported over (T466 for A4)
    label: str  # the target_set column of ablations.csv
    eval_spaces: tuple[str, ...]  # the arm's own target space first
    is_default: bool  # the override changes nothing: this arm is the main protocol


@dataclass(frozen=True, eq=False)
class Unit:
    """One ablations.csv row: an arm, a model and the target space it is scored in."""

    arm: Arm
    model: str
    eval_space: str

    @property
    def run_label(self) -> str:
        """The model column of runs.csv / MLflow, naming the arm (ablations.csv splits it out)."""
        return f"{self.model}|{self.arm.ablation}={self.arm.name}|eval={self.eval_space}"

    @property
    def selection_name(self) -> str:
        return f"ablation__{self.arm.ablation}__{self.arm.name}__{self.model}__{self.eval_space}"

    @property
    def is_main_configuration(self) -> bool:
        """Default arm, scored in its own space on its own targets: E0 / E1 exactly."""
        arm = self.arm
        natural = self.eval_space == arm.cfg["metabolites"]["transform"]
        return arm.is_default and natural and arm.train_target_set == arm.score_target_set

    def builder(self) -> protocol.PipelineBuilder:
        return functools.partial(
            build_unit,
            cfg=self.arm.cfg,
            model=self.model,
            train_target_set=self.arm.train_target_set,
            score_target_set=self.arm.score_target_set,
            eval_space=self.eval_space,
        )


def configured_arms(cfg: Mapping[str, Any]) -> list[Arm]:
    """Every arm of cfg["ablations"]["studies"], per target set, in config order."""
    section = cfg["ablations"]
    arms = []
    for ablation, study in section["studies"].items():
        _check_keys(study, STUDY_KEYS, f"ablations.studies.{ablation}")
        target_sets = _study_target_sets(study, section, ablation)
        for name, override in study["arms"].items():
            _check_override(override, cfg, f"{ablation}={name}")
            arm_cfg = deep_merge(dict(cfg), override)
            spaces = _eval_spaces(arm_cfg, study.get("also_score_in", []), ablation)
            for train_set, score_set, label in target_sets:
                arms.append(
                    Arm(
                        ablation=str(ablation),
                        name=str(name),
                        override=override,
                        cfg=arm_cfg,
                        train_target_set=train_set,
                        score_target_set=score_set,
                        label=label,
                        eval_spaces=spaces,
                        is_default=arm_cfg == cfg,
                    )
                )
    return arms


def configured_units(cfg: Mapping[str, Any]) -> list[Unit]:
    return [unit for arm in configured_arms(cfg) for unit in arm_units(arm, cfg)]


def arm_units(arm: Arm, cfg: Mapping[str, Any]) -> list[Unit]:
    models = cfg["ablations"]["models"]
    unknown = sorted(set(models) - set(MODEL_BUILDERS))
    if unknown:
        raise ValueError(f"unknown ablation model(s) {unknown}; known: {list(MODEL_BUILDERS)}")
    return [Unit(arm, model, space) for model in models for space in arm.eval_spaces]


def unit_params(unit: Unit, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Main-protocol hyperparameters: base ridge, or E1's selection for the training target set."""
    if unit.model == "ridge":
        return dict(unit.arm.cfg["ridge"])
    return dict(mlp.load_checked_selection(cfg, unit.arm.train_target_set)["params"])


def unit_seeds(unit: Unit, cfg: Mapping[str, Any]) -> list[int]:
    seeds = cfg["ablations"]["models"][unit.model].get("external_seeds")
    return [int(s) for s in seeds] if seeds else common.model_seeds(cfg)


def unit_null_threshold(unit: Unit, cfg: Mapping[str, Any]) -> float | None:
    """E0's D1 null threshold for scores in the main target space; None (NaN) in CLR space.

    The null refits ridge on log10 targets and scores it against log10 truth. Against CLR truth
    chance-level ρ is distributed differently (CLR turns tied zeros into sample-dependent values
    and couples all metabolites through each sample's mean log), so that threshold does not
    apply; a CLR null would need its own permutation runs. n_above_null is NaN there instead.
    """
    if unit.eval_space != cfg["metabolites"]["transform"]:
        return None
    return protocol.load_null_threshold(DATA, unit.arm.score_target_set)


def _study_target_sets(
    study: Mapping[str, Any], section: Mapping[str, Any], ablation: str
) -> list[tuple[str, str, str]]:
    """(train, score, label) target sets: A1-A3 use each configured set, A4 defines its own."""
    if "train_target_set" not in study:
        return [(str(t), str(t), str(t)) for t in section["target_sets"]]
    train, score = str(study["train_target_set"]), str(study["score_target_set"])
    for name in (train, score):
        if name not in common.TARGET_SETS:
            raise ValueError(f"{ablation}: unknown target set {name!r}")
    if score != train and train != "TALL":
        # Scoring a subset is meaningful only when the model was trained on every feature.
        raise ValueError(f"{ablation}: score_target_set differs from a non-TALL train set")
    return [(train, score, str(study["target_set_label"]))]


def _eval_spaces(
    arm_cfg: Mapping[str, Any], also: Sequence[str], ablation: str
) -> tuple[str, ...]:
    natural = str(arm_cfg["metabolites"]["transform"])
    for space in also:
        if space not in METABOLITE_TRANSFORMS:
            raise ValueError(f"{ablation}: unknown eval space {space!r}")
    return (natural, *(str(s) for s in also if s != natural))


def _check_override(override: Mapping[str, Any], cfg: Mapping[str, Any], where: str) -> None:
    """Only existing preprocessing settings may change, so a typo cannot fake an arm."""
    for section, values in override.items():
        if section not in OVERRIDABLE:
            raise ValueError(f"arm {where} overrides {section!r}; allowed: {OVERRIDABLE}")
        if not isinstance(values, Mapping):
            raise ValueError(f"arm {where}: the {section} override must be a mapping of settings")
        unknown = sorted(set(values) - set(cfg[section]))
        if unknown:
            raise ValueError(f"arm {where}: unknown {section} setting(s) {unknown}")


def _check_keys(section: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {where}; allowed: {list(allowed)}")


# --------------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------------


def stage_cv(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """PRISM CV of every unit at cv_seed; default arms must reproduce the main CV numbers."""
    X, Y, labels = common.load_d1_train()
    folds = common.d1_folds(labels, cfg)
    for unit in configured_units(cfg):
        start = time.perf_counter()
        params = unit_params(unit, cfg)
        row, _ = protocol.cross_validate(
            unit.builder(), params, X, Y, labels, folds, unit.arm.cfg,
            seed=int(cfg["cv_seed"]), n_jobs=n_jobs,
        )
        # Checked before anything is recorded: a default arm that drifted from the main protocol
        # must leave no CV record behind, since the external stage scores any unit that has one.
        check_reproduces_main(unit, float(row["cv_mean_rho"]), cfg)
        table = pd.DataFrame([row])
        protocol.record_cv_table(
            table, experiment=EXPERIMENT, model=unit.run_label, target_set=unit.arm.label,
            data=DATA, cfg=unit.arm.cfg, n_train=len(X),
        )
        protocol.save_selection(unit.selection_name, unit.arm.label, params, table)
        _upsert(ablation_row(unit, cfg, cv_mean_rho=row["cv_mean_rho"]))
        _say(
            f"cv {unit.run_label} {unit.arm.label}: mean ρ {row['cv_mean_rho']:.4f} "
            f"({row['cv_n_features']:.0f} targets, {row['cv_n_input_features']:.0f} taxa) "
            f"in {time.perf_counter() - start:.1f}s"
        )


def stage_external(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Refit every unit on all of PRISM and score NLIBD; the cohort is loaded once per arm."""
    X, Y, labels = common.load_d1_train()
    # Every arm's records are read (and default arms re-checked against the current E0 / E1 CV)
    # before the cohort is first opened: a missing or stale record must fail before any arm has
    # an external score, not after a few arms were already scored on NLIBD.
    plan = []
    for arm in configured_arms(cfg):
        units = arm_units(arm, cfg)
        saved = [load_checked_unit(unit, cfg) for unit in units]
        for unit, selection in zip(units, saved, strict=True):
            check_reproduces_main(unit, float(selection["cv_mean_rho"]), cfg)
        # A default arm IS E0 ridge / E1 MLP, already scored on NLIBD once: copy that row instead
        # of opening the cohort again ("once per final configuration", brief section 5).
        copied = {unit: main_external_row(unit) for unit in units if unit.is_main_configuration}
        plan.append((arm, units, saved, [unit_null_threshold(unit, cfg) for unit in units], copied))
    for arm, units, saved, thresholds, copied in plan:
        for unit, selection in zip(units, saved, strict=True):
            if unit in copied:
                _upsert(ablation_row(unit, cfg, cv_mean_rho=selection["cv_mean_rho"],
                                     aggregate=copied[unit], copied_from_main=True))
                _say(f"external {unit.run_label} {arm.label}: copied from the main "
                     f"{MAIN_SUMMARY_MODELS[unit.model]} row (same configuration)")
        if all(unit in copied for unit in units):
            continue
        X_test, Y_test = common.load_d1_test(
            purpose=(
                f"ablation {arm.ablation}={arm.name}: {', '.join(cfg['ablations']['models'])} "
                f"refit on PRISM with this arm's preprocessing, scored on NLIBD ({arm.label})"
            )
        )
        for unit, selection, threshold in zip(units, saved, thresholds, strict=True):
            if unit in copied:
                continue
            result = protocol.evaluate_external(
                unit.builder(), selection["params"], X, Y, labels, X_test, Y_test, arm.cfg,
                seeds=unit_seeds(unit, cfg), null_threshold=threshold, n_jobs=n_jobs,
            )
            protocol.record_external(
                result, experiment=EXPERIMENT, model=unit.run_label, target_set=arm.label,
                data=DATA, params=selection["params"], cfg=arm.cfg, n_train=len(X),
                cv_mean_rho=selection["cv_mean_rho"], write_summary=False,
                write_per_metabolite_file=False,
            )
            _upsert(ablation_row(unit, cfg, cv_mean_rho=selection["cv_mean_rho"], result=result))
            agg = result.aggregate
            _say(
                f"external {unit.run_label} {arm.label}: mean ρ {agg['mean_rho_mean']:.4f} "
                f"± {agg['mean_rho_sd']:.4f} (CV {selection['cv_mean_rho']:.4f})"
            )


# --------------------------------------------------------------------------------------------
# Checks and rows
# --------------------------------------------------------------------------------------------


def check_reproduces_main(unit: Unit, cv_mean_rho: float, cfg: Mapping[str, Any]) -> None:
    """A default arm is E0 ridge / E1 MLP itself, so its CV mean ρ must equal theirs.

    A mismatch means the ablation config (or code path) drifted from the main protocol, which
    would make every arm-vs-default comparison meaningless, so it raises.
    """
    if not unit.is_main_configuration:
        return
    main_name = MAIN_SELECTIONS[unit.model]
    try:
        main = protocol.load_selection(main_name, unit.arm.train_target_set)["cv_mean_rho"]
    except FileNotFoundError:
        _say(f"{unit.run_label}: not compared with {main_name} (run its CV stage first)")
        return
    tolerance = float(cfg["ablations"]["reproduce_tolerance"])
    difference = abs(cv_mean_rho - float(main))
    if difference > tolerance:
        raise RuntimeError(
            f"default arm {unit.run_label} ({unit.arm.label}) has CV mean ρ {cv_mean_rho!r}, "
            f"but {main_name} has {main!r} (|diff| {difference:.3g} > {tolerance:g}): the "
            "ablation config no longer matches the main experiment (cv_seed, folds, sections)"
        )
    _say(f"{unit.run_label} reproduces {main_name} CV mean ρ {float(main):.6f} exactly")


def main_external_row(unit: Unit) -> dict[str, Any]:
    """The main experiment's summary.csv row that a default-arm unit reproduces.

    Read before any cohort is opened, so a missing E0 / E1 external stage fails cleanly.
    """
    path = paths.RESULTS / tracking.SUMMARY_CSV
    model = MAIN_SUMMARY_MODELS[unit.model]
    summary = (
        pd.read_csv(path, float_precision="round_trip")  # copies must be exact, not rounded
        if path.exists()
        else pd.DataFrame(columns=["model", "target_set"])
    )
    match = summary[
        (summary["model"] == model) & (summary["target_set"] == unit.arm.train_target_set)
    ]
    if match.empty:
        raise FileNotFoundError(
            f"no {model} / {unit.arm.train_target_set} row in {path}: run the main experiment's "
            "external stage first (the default arm copies it instead of re-scoring NLIBD)"
        )
    return match.iloc[0].to_dict()


def load_checked_unit(unit: Unit, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The cv stage's record of a unit, refused if its hyperparameters changed since then."""
    selection = protocol.load_selection(unit.selection_name, unit.arm.label)
    expected = unit_params(unit, cfg)
    if selection["params"] != expected:
        raise ValueError(
            f"{unit.selection_name} was cross-validated with {selection['params']}, but the main "
            f"protocol now uses {expected}: rerun the ablations cv stage"
        )
    return selection


def ablation_row(
    unit: Unit,
    cfg: Mapping[str, Any],
    *,
    cv_mean_rho: float,
    result: protocol.ExternalResult | None = None,
    aggregate: Mapping[str, Any] | None = None,
    copied_from_main: bool = False,
) -> dict[str, Any]:
    """One results/ablations.csv row; external columns only once the external stage ran.

    External metrics come from `result` (scored here) or `aggregate` (a main-experiment summary
    row copied for a default arm, flagged by `copied_from_main`).
    """
    arm = unit.arm
    row: dict[str, Any] = {
        "ablation": arm.ablation,
        "arm": arm.name,
        "model": unit.model,
        "target_set": arm.label,
        "eval_space": unit.eval_space,
        "is_default_arm": arm.is_default,
        "override": arm.override,
        "cv_mean_rho": float(cv_mean_rho),
    }
    external = result.aggregate if result is not None else aggregate
    if external is not None:
        row.update(_external_columns(external, cfg))
        row["external_copied_from_main"] = copied_from_main
    row["timestamp"] = tracking.utc_now()
    return row


def _external_columns(agg: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
    columns = {
        "mean_rho_mean": agg["mean_rho_mean"],
        "mean_rho_sd": agg["mean_rho_sd"],
        "mean_rho_ci_low": agg["mean_rho_ci_low"],
        "mean_rho_ci_high": agg["mean_rho_ci_high"],
    }
    # Seed means of the other section-5 statistics (top-k, N(ρ > threshold), N above null, ...).
    for key in evaluate.metric_keys(cfg):
        if key not in ("n_features", "mean_rho"):
            columns[f"{key}_mean"] = agg[f"{key}_mean"]
    columns.update(
        n_features=int(agg["n_features_mean"]),
        n_input_features=int(agg["n_input_features"]),
        n_seeds=int(agg["n_seeds"]),
        null_threshold=agg["null_threshold"],
    )
    return columns


def _upsert(row: Mapping[str, Any]) -> None:
    tracking.upsert_csv_row(paths.RESULTS / ABLATIONS_CSV, row, key=ROW_KEY)


def _say(message: str) -> None:
    print(f"[{EXPERIMENT}] {message}", flush=True)
