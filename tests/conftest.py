"""Test-wide isolation: every test writes results to a temp dir and never touches MLflow.

The environment variables must be set before any `m2m` module is imported, because
`m2m.paths` resolves its locations at import time.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="m2m-tests-"))
os.environ.setdefault("M2M_RESULTS", str(_TMP_ROOT / "results"))
os.environ.setdefault("M2M_DATA_PROCESSED", str(_TMP_ROOT / "processed"))
os.environ.setdefault("M2M_MLRUNS", str(_TMP_ROOT / "mlruns"))
os.environ["M2M_DISABLE_MLFLOW"] = "1"
# Behave like CI on a developer machine too: tests are synthetic, so the real raw data must be
# unreachable (a test that silently read it would pass here and fail on GitHub), and a dry-run
# switch left over in the shell must not leak in (tests that need it set it themselves).
os.environ["M2M_DATA_RAW"] = str(_TMP_ROOT / "no_raw_data")
os.environ.pop("M2M_DRY_RUN", None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(autouse=True)
def _isolated_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test writes to its own directories, including the paths derived from RESULTS at
    import time (SPLITS, SELECTED, FIGURES), so no test can touch the committed results/."""
    from m2m import paths

    out = tmp_path / "results"
    monkeypatch.setattr(paths, "RESULTS", out)
    monkeypatch.setattr(paths, "SPLITS", out / "splits")
    monkeypatch.setattr(paths, "SELECTED", out / "selected")
    monkeypatch.setattr(paths, "FIGURES", out / "figures")
    monkeypatch.setattr(paths, "DATA_PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(paths, "MLRUNS", tmp_path / "mlruns")
    return out


@pytest.fixture
def results_dir(_isolated_outputs: Path) -> Path:
    """The per-test results/ directory, for tests that inspect written files."""
    return _isolated_outputs


@pytest.fixture
def synthetic_cohort(rng: np.random.Generator) -> dict[str, pd.DataFrame | pd.Series]:
    """Small compositional microbiome + zero-inflated metabolome with a planted signal.

    60 samples, 30 taxa (relative abundances not summing to 1, like D1), 40 metabolites of which
    the first 10 depend log-linearly on taxa 0-4 (no zeros) and the other 30 are zero-inflated
    noise; diagnosis labels in three classes.
    """
    n, p, m = 60, 30, 40
    ids = [f"S|{i:03d}" for i in range(n)]
    raw = rng.lognormal(mean=0.0, sigma=1.5, size=(n, p))
    raw[rng.random((n, p)) < 0.35] = 0.0
    raw[:, :5] += rng.lognormal(0.0, 0.5, size=(n, 5))  # keep signal taxa prevalent
    rel = raw / raw.sum(axis=1, keepdims=True) * rng.uniform(0.5, 1.0, size=(n, 1))
    log_signal = np.log(rel[:, :5] + 1e-6) @ rng.normal(size=(5, 10))
    mets = np.exp(rng.normal(size=(n, m)) * 0.5)
    mets[:, :10] = np.exp(1.0 * (log_signal - log_signal.mean(0)) / log_signal.std(0))
    mets *= 1000.0
    # Zero-inflate only the noise metabolites: a zero becomes an extreme outlier after log10(x+1)
    # and would bury the planted signal in a 60-sample cohort.
    noise_zeros = rng.random((n, m)) < 0.2
    noise_zeros[:, :10] = False
    mets[noise_zeros] = 0.0
    X = pd.DataFrame(rel, index=ids, columns=[f"taxon_{j:02d}" for j in range(p)])
    Y = pd.DataFrame(mets, index=ids, columns=[f"met_{j:02d}" for j in range(m)])
    labels = pd.Series(np.array(["CD", "UC", "Control"])[np.arange(n) % 3], index=ids,
                       name="diagnosis")
    return {"X": X, "Y": Y, "labels": labels}


@pytest.fixture
def frozen(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Pretend the selection phase is frozen and committed; records every gated purpose."""
    from m2m import freeze

    purposes: list[str] = []
    monkeypatch.setattr(freeze, "require_frozen", purposes.append)
    return purposes
