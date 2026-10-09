"""Filesystem locations used across the project.

Every location can be redirected with an environment variable so tests can write to a
temporary directory instead of the committed results/ folder.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("M2M_ROOT", Path(__file__).resolve().parents[2]))

DATA_RAW = Path(os.environ.get("M2M_DATA_RAW", REPO_ROOT / "data" / "raw"))
DATA_PROCESSED = Path(os.environ.get("M2M_DATA_PROCESSED", REPO_ROOT / "data" / "processed"))
RESULTS = Path(os.environ.get("M2M_RESULTS", REPO_ROOT / "results"))
CONFIGS = REPO_ROOT / "configs"

FIGURES = RESULTS / "figures"
SPLITS = RESULTS / "splits"
SELECTED = RESULTS / "selected"
MLRUNS = Path(os.environ.get("M2M_MLRUNS", REPO_ROOT / "mlruns"))

# Raw-data layout produced by scripts/download_data.sh.
D1_DIR = DATA_RAW / "MiMeNet" / "data" / "IBD"
D2_DIR = DATA_RAW / "microbiome-metabolome-curated-data" / "data" / "processed_data"
D3_DIR = DATA_RAW / "MGM"
