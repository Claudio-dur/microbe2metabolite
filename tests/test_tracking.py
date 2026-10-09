"""Run bookkeeping (m2m.tracking): CSV append/upsert, run rows, the audit log and MLflow."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from m2m import config, paths, tracking

TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
SHA = "abc1234"
# MLflow exports the active store and experiment to these variables for child processes.
MLFLOW_ENV_VARS = ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _header(path: Path) -> list[str]:
    with path.open(newline="") as fh:
        return next(csv.reader(fh))


@pytest.fixture
def fixed_sha(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin the commit so rows are predictable and no git subprocess runs."""
    monkeypatch.setattr(tracking, "git_sha", lambda: SHA)
    return SHA


@pytest.fixture
def mlflow_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """MLflow switched on (conftest disables it) and writing to a throwaway SQLite store."""
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setattr(paths, "MLRUNS", tmp_path / "mlruns")
    monkeypatch.delenv("M2M_DISABLE_MLFLOW")
    saved = {var: os.environ.get(var) for var in MLFLOW_ENV_VARS}
    yield mlflow
    # Put the variables back so later tests and subprocesses never see the temporary store.
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


# --- CSV helpers ---------------------------------------------------------------------------------


def test_append_csv_row_creates_the_file_then_appends(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "runs.csv"  # missing parent directories are created
    tracking.append_csv_row(path, {"a": 1, "b": "x"})
    tracking.append_csv_row(path, {"b": "y", "a": 2})  # key order does not matter

    assert _header(path) == ["a", "b"]
    assert _rows(path) == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_append_csv_row_rewrites_the_file_with_the_union_of_columns(tmp_path: Path) -> None:
    path = tmp_path / "runs.csv"
    tracking.append_csv_row(path, {"a": 1, "b": 2})
    tracking.append_csv_row(path, {"a": 3, "c": 4})  # new column: old rows get a blank
    tracking.append_csv_row(path, {"c": 5})  # missing columns are left blank

    assert _header(path) == ["a", "b", "c"]
    assert _rows(path) == [
        {"a": "1", "b": "2", "c": ""},
        {"a": "3", "b": "", "c": "4"},
        {"a": "", "b": "", "c": "5"},
    ]


def test_append_csv_row_stores_containers_as_sorted_json(tmp_path: Path) -> None:
    path = tmp_path / "runs.csv"
    tracking.append_csv_row(
        path, {"grid": {"lr": 1e-3, "dropout": 0.2}, "seeds": [0, 1], "pair": (1, 2)}
    )
    row = _rows(path)[0]
    assert row["grid"] == '{"dropout": 0.2, "lr": 0.001}'
    assert json.loads(row["seeds"]) == [0, 1]
    assert json.loads(row["pair"]) == [1, 2]


def test_upsert_csv_row_replaces_the_matching_key_and_keeps_the_others(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    key = ("model", "target_set")
    tracking.upsert_csv_row(path, {"model": "ridge", "target_set": "T466", "mean_rho": 0.1}, key)
    tracking.upsert_csv_row(path, {"model": "ridge", "target_set": "TALL", "mean_rho": 0.2}, key)
    tracking.upsert_csv_row(
        path, {"model": "ridge", "target_set": "T466", "mean_rho": 0.3, "n_nan": 5}, key
    )

    assert _header(path) == ["model", "target_set", "mean_rho", "n_nan"]
    assert [(r["target_set"], r["mean_rho"], r["n_nan"]) for r in _rows(path)] == [
        ("TALL", "0.2", ""),
        ("T466", "0.3", "5"),  # the replaced row moves to the end
    ]


def test_upsert_csv_row_matches_numeric_keys(tmp_path: Path) -> None:
    path = tmp_path / "learning_curve.csv"
    key = ("model", "n_train")
    tracking.upsert_csv_row(path, {"model": "ridge", "n_train": 20, "mean_rho": 0.1}, key)
    tracking.upsert_csv_row(path, {"model": "ridge", "n_train": 20, "mean_rho": 0.2}, key)

    assert [r["mean_rho"] for r in _rows(path)] == ["0.2"]


def test_upsert_summary_row_keeps_one_row_per_model_and_target_set(results_dir: Path) -> None:
    tracking.upsert_summary_row({"model": "mlp", "target_set": "T466", "mean_rho": 0.25})
    tracking.upsert_summary_row({"model": "mlp", "target_set": "TALL", "mean_rho": 0.15})
    tracking.upsert_summary_row({"model": "mlp", "target_set": "T466", "mean_rho": 0.27})

    rows = _rows(results_dir / tracking.SUMMARY_CSV)
    assert sorted((r["target_set"], r["mean_rho"]) for r in rows) == [
        ("T466", "0.27"),
        ("TALL", "0.15"),
    ]
    assert all(TIMESTAMP.match(r["timestamp"]) for r in rows)


# --- run rows ------------------------------------------------------------------------------------


def test_log_run_writes_one_row_with_tags_metrics_and_params_as_json(
    results_dir: Path, fixed_sha: str
) -> None:
    cfg = {"train": {"lr": 1e-3}, "config_path": "configs/e1_mlp.yaml"}
    tracking.log_run(
        experiment="e1_mlp",
        run_name="T466-fold0",
        params={"lr": 1e-3, "hidden": [256, 128]},
        metrics={"mean_rho": 0.21, "n_rho_gt_0.5": 3},
        tags={"model": "mlp", "target_set": "T466", "seed": 0},
        cfg=cfg,
    )

    path = results_dir / tracking.RUNS_CSV
    [row] = _rows(path)
    assert _header(path)[:3] == ["timestamp", "experiment", "run_name"]
    assert TIMESTAMP.match(row["timestamp"])
    assert (row["experiment"], row["run_name"]) == ("e1_mlp", "T466-fold0")
    assert (row["model"], row["target_set"], row["seed"]) == ("mlp", "T466", "0")
    assert row["params"] == '{"hidden": [256, 128], "lr": 0.001}'  # one JSON column, sorted
    assert (row["mean_rho"], row["n_rho_gt_0.5"]) == ("0.21", "3")
    assert row["config_hash"] == config.config_hash(cfg)
    assert row["git_sha"] == fixed_sha


def test_log_run_without_a_config_to_an_explicit_csv(tmp_path: Path, fixed_sha: str) -> None:
    path = tmp_path / "custom_runs.csv"
    tracking.log_run(
        experiment="e0", run_name="ridge", params={}, metrics={"mean_rho": 0.1}, runs_csv=path
    )
    [row] = _rows(path)
    assert (row["params"], row["config_hash"], row["mean_rho"]) == ("{}", "", "0.1")


def test_log_run_with_mlflow_disabled_never_creates_a_store(
    results_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_sha: str
) -> None:
    monkeypatch.setattr(paths, "MLRUNS", tmp_path / "mlruns")
    assert os.environ["M2M_DISABLE_MLFLOW"] == "1"  # set by conftest for every test

    tracking.log_run(
        experiment="e1", run_name="r", params={}, metrics={}, cfg={"mlflow": {"enabled": True}}
    )

    assert not (tmp_path / "mlruns").exists()
    assert len(_rows(results_dir / tracking.RUNS_CSV)) == 1


def test_mlflow_enabled_checks_the_environment_before_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("M2M_DISABLE_MLFLOW")
    assert tracking.mlflow_enabled(None)
    assert tracking.mlflow_enabled({"mlflow": {"enabled": True}})
    assert not tracking.mlflow_enabled({"mlflow": {"enabled": False}})
    monkeypatch.setenv("M2M_DISABLE_MLFLOW", "1")
    assert not tracking.mlflow_enabled({"mlflow": {"enabled": True}})


# --- external-access audit log -------------------------------------------------------------------


def test_log_external_access_appends_one_row_per_access(results_dir: Path, fixed_sha: str) -> None:
    tracking.log_external_access(purpose="final_evaluation", detail="E1 mlp T466")
    tracking.log_external_access(purpose="final_evaluation", detail="E0 ridge T466")

    rows = _rows(results_dir / tracking.EXTERNAL_ACCESS_LOG)
    assert [(r["purpose"], r["detail"], r["git_sha"]) for r in rows] == [
        ("final_evaluation", "E1 mlp T466", fixed_sha),
        ("final_evaluation", "E0 ridge T466", fixed_sha),
    ]
    assert all(TIMESTAMP.match(r["timestamp"]) for r in rows)


def test_log_external_access_to_an_explicit_path(tmp_path: Path, fixed_sha: str) -> None:
    path = tmp_path / "audit.csv"
    tracking.log_external_access(purpose="data_check", path=path)
    assert [(r["purpose"], r["detail"]) for r in _rows(path)] == [("data_check", "")]


# --- environment facts ---------------------------------------------------------------------------


def test_library_versions_reports_python_platform_and_installed_libraries() -> None:
    versions = tracking.library_versions()

    assert versions["python"] == platform.python_version()
    assert versions["platform"]
    for name in tracking.TRACKED_LIBRARIES:
        try:
            expected = metadata.version(name)
        except metadata.PackageNotFoundError:
            assert name not in versions  # e.g. transformers outside the MGM environment
        else:
            assert versions[name] == expected
    assert {"torch", "numpy", "pandas", "scikit-learn", "scipy"} <= versions.keys()


def test_git_sha_is_none_outside_a_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("GIT_DIR", "GIT_WORK_TREE"):  # set when tests run from a git hook
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    assert tracking.git_sha() is None


def test_utc_now_is_an_iso8601_utc_timestamp() -> None:
    assert TIMESTAMP.match(tracking.utc_now())


def test_tracking_uri_is_a_sqlite_file_inside_mlruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "MLRUNS", tmp_path / "mlruns")
    assert tracking.tracking_uri() == f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}"
    assert (tmp_path / "mlruns").is_dir()


# --- MLflow switched on --------------------------------------------------------------------------


def test_log_run_records_an_mlflow_run_when_enabled(
    mlflow_store: ModuleType, results_dir: Path, fixed_sha: str
) -> None:
    cfg = {"mlflow": {"enabled": True, "experiment_prefix": "test"}}
    tracking.log_run(
        experiment="e1_mlp",
        run_name="T466-fold0",
        params={"lr": 1e-3, "hidden": [256, 128]},
        metrics={"mean_rho": 0.25, "nan_metric": math.nan, "flag": True, "note": "text"},
        tags={"model": "mlp", "seed": 0},
        cfg=cfg,
    )

    mlflow_store.set_tracking_uri(tracking.tracking_uri())
    runs = mlflow_store.search_runs(experiment_names=["test-e1_mlp"])
    assert len(runs) == 1
    run = runs.iloc[0]
    assert run["tags.mlflow.runName"] == "T466-fold0"
    assert (run["params.lr"], run["params.hidden"]) == ("0.001", "[256, 128]")
    assert run["metrics.mean_rho"] == pytest.approx(0.25)
    # Only finite real numbers become MLflow metrics; runs.csv keeps everything.
    assert not {"metrics.nan_metric", "metrics.flag", "metrics.note"} & set(runs.columns)
    assert (run["tags.model"], run["tags.seed"]) == ("mlp", "0")
    assert run["tags.config_hash"] == config.config_hash(cfg)
    assert run["tags.git_sha"] == fixed_sha
    assert run["tags.torch"] == metadata.version("torch")
    assert run["tags.python"] == platform.python_version()
    assert run["artifact_uri"].startswith((paths.MLRUNS / "artifacts").resolve().as_uri())
    assert len(_rows(results_dir / tracking.RUNS_CSV)) == 1


def test_mlflow_keeps_numpy_metrics(
    mlflow_store: ModuleType, results_dir: Path, fixed_sha: str
) -> None:
    tracking.log_run(
        experiment="e0",
        run_name="ridge",
        params={},
        metrics={"n_rho_gt_0.5": np.int64(3), "mean_rho": np.float32(0.5)},
        cfg={"mlflow": {"enabled": True, "experiment_prefix": "test"}},
    )

    mlflow_store.set_tracking_uri(tracking.tracking_uri())
    run = mlflow_store.search_runs(experiment_names=["test-e0"]).iloc[0]
    assert run["metrics.n_rho_gt_0.5"] == 3
    assert run["metrics.mean_rho"] == pytest.approx(0.5)
