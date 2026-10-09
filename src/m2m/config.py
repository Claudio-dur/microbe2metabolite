"""YAML experiment configs with single inheritance.

An experiment config may start with ``extends: base.yaml``; the child is deep-merged over the
parent so shared settings (preprocessing, CV, metrics) live in one place.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` updated recursively with ``override`` (neither input is mutated)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving ``extends`` relative to the config's own directory."""
    path = Path(path)
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    parent_name = cfg.pop("extends", None)
    if parent_name is not None:
        cfg = deep_merge(load_config(path.parent / parent_name), cfg)
    # Assigned, not defaulted: the parent's own config_path must not survive the merge.
    cfg["config_path"] = str(path)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Short stable hash of a config, ignoring where it was loaded from."""
    payload = {k: v for k, v in cfg.items() if k != "config_path"}
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]
