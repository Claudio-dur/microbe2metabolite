"""Freezing the selection phase: no external access unless every choice is fixed and committed."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from m2m import freeze, paths


@pytest.fixture
def selection_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repo-like tree with one selection output per required family and two configs."""
    root = tmp_path / "repo"
    results, configs = root / "results", root / "configs"
    monkeypatch.setattr(paths, "REPO_ROOT", root)
    monkeypatch.setattr(paths, "RESULTS", results)
    monkeypatch.setattr(paths, "SELECTED", results / "selected")
    monkeypatch.setattr(paths, "SPLITS", results / "splits")
    monkeypatch.setattr(paths, "CONFIGS", configs)
    for pattern in freeze.REQUIRED.values():
        target = results / pattern.replace("*", "x__T466")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{pattern}\n")
    configs.mkdir(parents=True)
    (configs / "base.yaml").write_text("seeds: [0]\n")
    (configs / "e1_mlp.yaml").write_text("extends: base.yaml\n")
    monkeypatch.setenv(freeze.DRY_RUN_ENV, "1")  # no git checks unless a test asks for them
    return root


def test_write_then_check_round_trip(selection_phase: Path) -> None:
    out = freeze.write_manifest()
    manifest = freeze.check()
    assert out == paths.SELECTED / freeze.MANIFEST_NAME
    assert "splits/prism_folds.csv" in manifest["artefacts"]
    assert set(manifest["configs"]) == {"base.yaml", "e1_mlp.yaml"}


def test_freezing_needs_every_selection_family(selection_phase: Path) -> None:
    (paths.SELECTED / "e1_mlp__x__T466.json").unlink()
    with pytest.raises(freeze.SelectionNotFrozenError, match="E1 MLP"):
        freeze.write_manifest()


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda: (paths.CONFIGS / "base.yaml").write_text("seeds: [1]\n"), "config base.yaml"),
        (lambda: (paths.SELECTED / "e0_x__T466.json").write_text("{}"), "changed"),
        (lambda: (paths.SELECTED / "e1_mlp__late.json").write_text("{}"), "added after"),
        (lambda: (paths.SPLITS / "prism_folds.csv").unlink(), "missing"),
    ],
)
def test_any_drift_after_the_freeze_is_refused(selection_phase: Path, change, message) -> None:
    freeze.write_manifest()
    change()
    with pytest.raises(freeze.SelectionNotFrozenError, match=message):
        freeze.require_frozen("E1 external")


def test_no_manifest_means_no_external_access(selection_phase: Path) -> None:
    with pytest.raises(freeze.SelectionNotFrozenError, match="refusing external access"):
        freeze.require_frozen("E0 null")


def test_outside_dry_runs_the_freeze_must_be_committed(
    selection_phase: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(freeze.DRY_RUN_ENV)
    root = selection_phase
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    freeze.write_manifest()
    with pytest.raises(freeze.SelectionNotFrozenError, match="not committed"):
        freeze.check()

    subprocess.run([*git, "-C", str(root), "add", "-A"], check=True)
    subprocess.run([*git, "-C", str(root), "commit", "-q", "-m", "freeze"], check=True)
    freeze.check()  # committed and unchanged: allowed

    (paths.CONFIGS / "e1_mlp.yaml").write_text("extends: base.yaml\nseeds: [9]\n")
    with pytest.raises(freeze.SelectionNotFrozenError):
        freeze.check()
