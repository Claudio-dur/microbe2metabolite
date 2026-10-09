"""Freezing model selection before the external evaluation (docs/DESIGN.md decision #24).

`make select` ends with ``python -m m2m.freeze write``: a manifest that hashes every selection
artefact (results/selected/, results/splits/) and every experiment config. Every later opening of
the external cohort (except the section-3 data checks) calls `require_frozen`, which refuses unless

1. the manifest exists and nothing it lists has changed (no config or selection drift),
2. no selection artefact appeared after the freeze, and
3. outside dry runs, the manifest, the selections and the code are committed and unmodified.

So no external score can be produced from choices that were not fixed, and committed, first.
Kept free of `m2m.experiments` imports because `m2m.data` (the guard) imports this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from m2m import paths, tracking

MANIFEST_NAME = "FROZEN.json"
DRY_RUN_ENV = "M2M_DRY_RUN"  # the switch of m2m.experiments.common.dry_run()
# Selection-phase outputs that must exist before freezing: at least one file per pattern.
REQUIRED = {
    "PRISM folds": "splits/prism_folds.csv",
    "learning-curve subsets": "splits/learning_curve_subsets.csv",
    "E0 baselines": "selected/e0_*.json",
    "E1 MLP": "selected/e1_mlp__*.json",
    "E2 frozen MGM": "selected/e2_mgm_frozen_*.json",
    "E2 random-weights control": "selected/e2_mgm_random_frozen_*.json",
    "E3 fine-tuned MGM": "selected/e3_mgm_finetune__*.json",
    "E3 from-scratch control": "selected/e3_mgm_scratch__*.json",
    "E4 transfer": "selected/e4c_*.json",
    "ablations": "selected/ablation__*.json",
}
# What an external evaluation computes from: it must be the committed state of these paths.
COMMITTED_INPUTS = (
    "src", "scripts", "configs", "Makefile", "pyproject.toml", "envs/mgm/requirements.txt",
    "results/selected", "results/splits",
)


class SelectionNotFrozenError(RuntimeError):
    """The external cohort was requested before the selection phase was frozen (and committed)."""


def manifest_path() -> Path:
    return paths.SELECTED / MANIFEST_NAME


def write_manifest() -> Path:
    """Hash every selection artefact and config; refuse if a selection family is missing."""
    missing = [
        label for label, pattern in REQUIRED.items() if not list(paths.RESULTS.glob(pattern))
    ]
    if missing:
        raise SelectionNotFrozenError(
            f"cannot freeze: no selection output for {missing}; run `make select` first"
        )
    # Content only (no timestamp or commit): re-deriving identical selections, e.g. `make all`
    # from a fresh clone, rewrites an identical manifest and the committed freeze stays valid.
    manifest = {
        "dry_run": _dry_run(),
        "artefacts": _hashes(_artefacts(), paths.RESULTS),
        "configs": _hashes(_configs(), paths.CONFIGS),
    }
    out = manifest_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[freeze] {tracking.utc_now()}: {len(manifest['artefacts'])} selection artefacts and "
          f"{len(manifest['configs'])} configs frozen in {out}")
    if not manifest["dry_run"]:
        print("[freeze] commit results/selected and results/splits before `make evaluate`")
    return out


def check(*, require_committed: bool | None = None) -> dict[str, Any]:
    """The manifest, verified against the files on disk (and, by default outside dry runs, git)."""
    path = manifest_path()
    if not path.exists():
        raise SelectionNotFrozenError(
            f"{path} not found: run `make select` (it ends by freezing the selections)"
        )
    manifest = json.loads(path.read_text())
    problems = _drift("selection artefact", manifest["artefacts"], _artefacts(), paths.RESULTS)
    problems += _drift("config", manifest["configs"], _configs(), paths.CONFIGS)
    if problems:
        raise SelectionNotFrozenError(
            "the selection phase changed after it was frozen: " + "; ".join(problems[:10])
        )
    if require_committed if require_committed is not None else not _dry_run():
        _require_committed(path)
    return manifest


def require_frozen(purpose: str) -> None:
    """Gate for every opening of the external cohort; the message names what was refused."""
    try:
        check()
    except SelectionNotFrozenError as err:
        raise SelectionNotFrozenError(f"refusing external access ({purpose}): {err}") from err


def _artefacts() -> list[Path]:
    files = [p for root in (paths.SELECTED, paths.SPLITS) if root.exists()
             for p in root.rglob("*") if p.is_file()]
    return sorted(p for p in files if p.name != MANIFEST_NAME)


def _configs() -> list[Path]:
    return sorted(paths.CONFIGS.glob("*.yaml"))


def _hashes(files: Sequence[Path], root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha256(p) for p in files}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drift(kind: str, frozen: dict[str, str], files: Sequence[Path], root: Path) -> list[str]:
    current = _hashes(files, root)
    problems = [f"{kind} {name} is missing" for name in frozen if name not in current]
    problems += [f"{kind} {name} was added after the freeze" for name in current
                 if name not in frozen]
    problems += [f"{kind} {name} changed" for name, digest in current.items()
                 if name in frozen and frozen[name] != digest]
    return problems


def _require_committed(manifest: Path) -> None:
    """The manifest is in HEAD and every input of the evaluation is committed and unmodified."""
    root = paths.REPO_ROOT
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(manifest.relative_to(root))],
        cwd=root, capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        raise SelectionNotFrozenError(
            "the frozen selections are not committed: commit results/selected and "
            "results/splits after `make select`, then run `make evaluate`"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *COMMITTED_INPUTS],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if status:
        raise SelectionNotFrozenError(
            "uncommitted changes to the code, configs or selections since the freeze:\n" + status
        )


def _dry_run() -> bool:
    return os.environ.get(DRY_RUN_ENV) == "1"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("write", "check"))
    args = parser.parse_args(argv)
    if args.action == "write":
        write_manifest()
    else:
        manifest = check()
        print(f"[freeze] OK: {len(manifest['artefacts'])} artefacts and "
              f"{len(manifest['configs'])} configs unchanged since the freeze"
              + ("" if manifest["dry_run"] else ", and committed"))


if __name__ == "__main__":
    main()
