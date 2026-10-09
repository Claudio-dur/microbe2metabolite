"""Run bookkeeping: results/runs.csv, results/summary.csv, MLflow and the external-access log.

MLflow is imported lazily so that `m2m.data` (which records external-test access) stays cheap
to import. Logging always happens in the parent process; worker processes only return results,
so the CSV files and the SQLite store never see concurrent writers.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import numbers
import os
import platform
import subprocess
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

from m2m import paths
from m2m.config import config_hash

RUNS_CSV = "runs.csv"
SUMMARY_CSV = "summary.csv"
EXTERNAL_ACCESS_LOG = "external_access_log.csv"
TRACKED_LIBRARIES = ("torch", "numpy", "pandas", "scikit-learn", "scipy", "mlflow", "transformers")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# What decides a run's numbers: '-dirty' means one of these differs from the commit. The run's
# own outputs (results/*.csv, figures, the access log) are left out, or every row written during
# an evaluation would look dirty; the frozen selections stay in.
PROVENANCE_PATHS = (
    "src", "scripts", "configs", "Makefile", "pyproject.toml", "envs/mgm/requirements.txt",
    "results/selected", "results/splits",
)


def git_sha() -> str | None:
    """Current commit, suffixed with '-dirty' when code, configs or selections are uncommitted."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", *PROVENANCE_PATHS],
            cwd=paths.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def library_versions() -> dict[str, str]:
    """Installed versions, read from package metadata so nothing heavy gets imported."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def append_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append one row; if new columns appear, rewrite the file with the union of columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {key: _csv_value(value) for key, value in row.items()}
    if not path.exists():
        _write_rows(path, list(row), [row])
        return
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        columns = list(reader.fieldnames or [])
        existing = list(reader)
    new_columns = [key for key in row if key not in columns]
    if new_columns:
        _write_rows(path, columns + new_columns, [*existing, row])
        return
    with path.open("a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=columns).writerow(row)


def upsert_csv_row(path: Path, row: Mapping[str, Any], key: tuple[str, ...]) -> None:
    """Replace the row whose `key` columns match (or append), keeping the other rows."""
    row = {k: _csv_value(v) for k, v in row.items()}
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    if path.exists():
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            columns = list(reader.fieldnames or [])
            # Compare as written text: the file holds "20" where the caller passes 20.
            wanted = tuple(_csv_text(row[k]) for k in key)
            rows = [r for r in reader if tuple(r.get(k) for k in key) != wanted]
    columns += [k for k in row if k not in columns]
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(path, columns, rows)


def log_external_access(*, purpose: str, detail: str = "", path: Path | None = None) -> None:
    """Audit trail: every opening of the external test cohort, with the reason given."""
    append_csv_row(
        path or paths.RESULTS / EXTERNAL_ACCESS_LOG,
        {"timestamp": utc_now(), "purpose": purpose, "detail": detail, "git_sha": git_sha()},
    )


def log_run(
    *,
    experiment: str,
    run_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    tags: Mapping[str, Any] | None = None,
    cfg: Mapping[str, Any] | None = None,
    runs_csv: Path | None = None,
) -> None:
    """Record one run in results/runs.csv and, unless disabled, as an MLflow run.

    `tags` are the descriptive columns (model, target_set, stage, seed, n_train, ...);
    `params` are hyperparameters, stored as one JSON column so runs.csv keeps a stable layout.
    """
    tags = dict(tags or {})
    sha = git_sha()
    cfg_hash = config_hash(dict(cfg)) if cfg else ""
    row = {
        "timestamp": utc_now(),
        "experiment": experiment,
        "run_name": run_name,
        **tags,
        "params": json.dumps(dict(params), sort_keys=True, default=str),
        **dict(metrics),
        "config_hash": cfg_hash,
        "git_sha": sha,
    }
    append_csv_row(runs_csv or paths.RESULTS / RUNS_CSV, row)
    if mlflow_enabled(cfg):
        _log_mlflow(
            experiment=_mlflow_experiment_name(experiment, cfg),
            run_name=run_name,
            params=params,
            metrics=metrics,
            tags={**tags, "config_hash": cfg_hash, "git_sha": sha or "unknown"},
        )


def upsert_summary_row(
    row: Mapping[str, Any],
    *,
    key: tuple[str, ...] = ("model", "target_set"),
    path: Path | None = None,
) -> None:
    """results/summary.csv holds exactly one row per model × target set."""
    upsert_csv_row(path or paths.RESULTS / SUMMARY_CSV, {**row, "timestamp": utc_now()}, key)


def mlflow_enabled(cfg: Mapping[str, Any] | None) -> bool:
    if os.environ.get("M2M_DISABLE_MLFLOW") == "1":
        return False
    return bool(((cfg or {}).get("mlflow") or {}).get("enabled", True))


def tracking_uri() -> str:
    """SQLite store inside mlruns/ (MLflow >= 3.7 treats the ./mlruns file store as legacy)."""
    paths.MLRUNS.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{paths.MLRUNS / 'mlflow.db'}"


def _mlflow_experiment_name(experiment: str, cfg: Mapping[str, Any] | None) -> str:
    prefix = ((cfg or {}).get("mlflow") or {}).get("experiment_prefix", "m2m")
    return f"{prefix}-{experiment}"


def _log_mlflow(
    *,
    experiment: str,
    run_name: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    tags: Mapping[str, Any],
) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    if mlflow.get_experiment_by_name(experiment) is None:
        artifacts = (paths.MLRUNS / "artifacts" / experiment).resolve().as_uri()
        mlflow.create_experiment(experiment, artifact_location=artifacts)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        # MLflow caps param values at a few hundred characters; long lists are truncated, not lost
        # (the full JSON is always in runs.csv).
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if _is_finite_number(v)})
        mlflow.set_tags({**{k: str(v) for k, v in tags.items()}, **library_versions()})


def _is_finite_number(value: Any) -> bool:
    """Real numbers including numpy scalars (np.int64 counts, float32 metrics); not bools."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _csv_text(value: Any) -> str:
    """The text csv.DictWriter writes for a value (None → empty string)."""
    return "" if value is None else str(_csv_value(value))


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _write_rows(path: Path, columns: list[str], rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in columns})
