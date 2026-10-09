"""Cross-validation folds, inner early-stopping splits and subsamples, always as sample IDs.

Every function takes and returns sample IDs, never row positions, so a split survives any
re-ordering or filtering of the data tables, can be saved to disk, and can be checked for leakage
with `assert_disjoint`. Sample IDs are handled as strings.

Labels and groups can be given either as a sequence in the same order as `sample_ids`, or as a
pandas Series indexed by sample ID (matched by ID; extra entries are ignored). The second form
lets callers pass the full metadata column together with a subset of IDs.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
)

from m2m import paths

PRISM_FOLDS_FILE = "prism_folds.csv"
MAX_REPORTED_IDS = 10  # how many offending IDs an error message lists
# Digits kept before rounding a split size up: float noise such as 0.07 * 100 = 7.000000000000001
# must not add an extra sample to the held-out side.
_SIZE_ROUNDING_DIGITS = 9

# One value per sample: aligned with `sample_ids` by position, or a Series indexed by sample ID.
PerSample = Sequence[Hashable] | np.ndarray | pd.Series


class LeakageError(AssertionError):
    """The same sample (or subject) sits on both sides of a split that must be disjoint."""


# --- outer folds ------------------------------------------------------------------------------


def make_stratified_folds(
    sample_ids: Iterable[Hashable], labels: PerSample, *, n_splits: int, seed: int
) -> pd.DataFrame:
    """Stratified K-fold assignment: DataFrame[sample_id, label, fold], rows in input order.

    Folds are numbered 0..n_splits-1; each sample is in exactly one test fold.
    """
    ids = _as_ids(sample_ids)
    label_values = _per_sample(ids, labels, "labels")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = _fold_numbers(splitter.split(np.zeros(len(ids)), label_values), len(ids))
    return pd.DataFrame({"sample_id": ids, "label": label_values, "fold": fold})


def make_grouped_folds(
    sample_ids: Iterable[Hashable], groups: PerSample, *, n_splits: int, seed: int
) -> pd.DataFrame:
    """Grouped K-fold assignment (a group never spans two folds): DataFrame[sample_id, group, fold].

    Used for longitudinal cohorts, where a group is a subject. scikit-learn's
    GroupKFold(shuffle=True) shuffles the groups with `seed` and deals the same *number of groups*
    to each fold, so sample counts per fold vary with group sizes. The unshuffled GroupKFold would
    balance sample counts instead, but it ignores the seed (every seed would give the same folds).
    """
    ids = _as_ids(sample_ids)
    group_values = _per_sample(ids, groups, "groups")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = _fold_numbers(splitter.split(np.zeros(len(ids)), groups=group_values), len(ids))
    return pd.DataFrame({"sample_id": ids, "group": group_values, "fold": fold})


def save_folds(folds: pd.DataFrame, path: str | Path) -> None:
    """Write a fold table as CSV, creating the directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(path, index=False)


def load_folds(path: str | Path) -> pd.DataFrame:
    """Read a fold table written by `save_folds`. IDs, labels and groups come back as strings."""
    # keep_default_na=False: an ID or label such as "NA" must stay a string, not become NaN.
    folds = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not {"sample_id", "fold"} <= set(folds.columns):
        raise ValueError(f"{path} is not a fold table: it needs 'sample_id' and 'fold' columns")
    if folds["sample_id"].duplicated().any():
        raise ValueError(f"{path} lists some sample IDs more than once")
    return folds.assign(fold=folds["fold"].astype(int))


def prism_folds(
    sample_ids: Iterable[Hashable],
    labels: PerSample,
    cfg: Mapping[str, Any],
    *,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """The PRISM outer CV folds shared by every model: created once, then reloaded and verified.

    Default file: results/splits/prism_folds.csv. The file is the source of truth, so every model
    is scored on identical folds. It is reused only if it covers exactly the same sample IDs with
    the same labels and has cfg["cv"]["n_splits"] folds; otherwise ValueError. Rows are returned
    in the order of `sample_ids`, with the same dtypes whether the file was just created or not.
    """
    path = Path(path) if path is not None else paths.SPLITS / PRISM_FOLDS_FILE
    ids = _as_ids(sample_ids)
    label_values = _per_sample(ids, labels, "labels")
    n_splits = int(cfg["cv"]["n_splits"])
    if not path.exists():
        created = make_stratified_folds(
            ids, label_values, n_splits=n_splits, seed=int(cfg["cv"]["seed"])
        )
        save_folds(created, path)
    saved = load_folds(path)
    _verify_saved_folds(saved, ids, label_values, n_splits, path)
    return saved.set_index("sample_id").loc[ids].reset_index()


def fold_train_test(folds: pd.DataFrame, fold: int) -> tuple[list[str], list[str]]:
    """(training IDs, test IDs) of one fold, both in the order of the fold table."""
    fold_numbers = folds["fold"].to_numpy()
    available = sorted(set(fold_numbers.tolist()))
    if fold not in available:
        raise ValueError(f"fold {fold} is not in the table (folds: {available})")
    ids = folds["sample_id"].to_numpy()
    in_test = fold_numbers == fold
    return ids[~in_test].tolist(), ids[in_test].tolist()


# --- inner splits (early stopping) ------------------------------------------------------------


def stratified_inner_split(
    sample_ids: Iterable[Hashable], labels: PerSample, *, fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split training IDs into (fit, validation) for early stopping, stratified by label.

    Validation gets ceil(fraction * n) samples, at least 1, and fit keeps at least 1. Both lists
    are in input order.
    """
    ids = _as_ids(sample_ids)
    label_values = _per_sample(ids, labels, "labels")
    n_val = _n_held_out(fraction, len(ids))
    class_sizes = pd.Series(label_values).value_counts()
    n_classes = len(class_sizes)
    # Stratifying needs >= 2 members per class and room for every class on both sides. Small
    # inputs (e.g. a 20-sample learning-curve subset) can miss that; a plain random split is
    # still a valid early-stopping split (it is all training data), just a noisier one, so we
    # fall back to it instead of failing.
    can_stratify = class_sizes.min() >= 2 and min(n_val, len(ids) - n_val) >= n_classes
    if can_stratify:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=seed)
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=n_val, random_state=seed)
    fit_pos, val_pos = next(splitter.split(np.zeros(len(ids)), label_values))
    return _ids_at(ids, fit_pos), _ids_at(ids, val_pos)


def grouped_inner_split(
    sample_ids: Iterable[Hashable], groups: PerSample, *, fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split IDs into (fit, validation) with every group (subject) on one side only.

    As in scikit-learn's GroupShuffleSplit, `fraction` is the share of *groups* held out
    (ceil, at least one group); the share of samples varies with group sizes. Input order kept.
    """
    ids = _as_ids(sample_ids)
    group_values = _per_sample(ids, groups, "groups")
    n_val_groups = _n_held_out(fraction, pd.Series(group_values).nunique())
    splitter = GroupShuffleSplit(n_splits=1, test_size=n_val_groups, random_state=seed)
    fit_pos, val_pos = next(splitter.split(np.zeros(len(ids)), groups=group_values))
    return _ids_at(ids, fit_pos), _ids_at(ids, val_pos)


# --- subsampling (learning curves) ------------------------------------------------------------


def stratified_subsample(
    sample_ids: Iterable[Hashable], labels: PerSample, *, n: int, seed: int
) -> list[str]:
    """Exactly `n` IDs, drawn per class in proportion to class sizes; returned in input order.

    Class quotas use largest-remainder rounding; when n >= number of classes every class keeps at
    least one sample (a learning-curve subset must not lose a diagnosis). Within a class, samples
    come from the front of a seeded permutation, so for one seed the subsets for increasing n are
    nested whenever no class quota shrinks (true for the PRISM sizes 20/40/80/155).
    """
    ids = _as_ids(sample_ids)
    label_values = _per_sample(ids, labels, "labels")
    if not 1 <= n <= len(ids):
        raise ValueError(f"n must be between 1 and the number of samples ({len(ids)}), got {n}")
    if n == len(ids):
        return ids
    positions_by_class = [
        np.flatnonzero(label_values == label) for label in sorted(set(label_values.tolist()))
    ]
    quotas = _class_quotas([len(pos) for pos in positions_by_class], n)
    rng = np.random.default_rng(seed)
    # Permute every class even when its quota is 0, so the random stream (and hence nesting
    # across n) does not depend on the quotas.
    permuted = [rng.permutation(pos) for pos in positions_by_class]
    chosen = [perm[:quota] for perm, quota in zip(permuted, quotas, strict=True)]
    return _ids_at(ids, np.concatenate(chosen))


# --- leakage checks ---------------------------------------------------------------------------


def assert_disjoint(a: Iterable[Hashable], b: Iterable[Hashable], *, what: str) -> None:
    """Raise LeakageError if any ID appears in both `a` and `b` (e.g. training vs test IDs).

    IDs also match by their string form, as everywhere in this module: an ID parsed as a number
    from one table (7123) and read as text from another ("7123") is the same sample, and a
    leakage check must fail closed rather than miss it.
    """
    a_ids, b_ids = set(a), set(b)
    as_text = {str(i) for i in a_ids} & {str(i) for i in b_ids}
    overlap = {str(i) for i in a_ids & b_ids} | as_text  # the first set keeps 1 == 1.0 matches
    if overlap:
        examples = _first(sorted(overlap))
        raise LeakageError(f"{what}: {len(overlap)} IDs are on both sides, e.g. {examples}")


# --- helpers ----------------------------------------------------------------------------------


def _as_ids(sample_ids: Iterable[Hashable]) -> list[str]:
    """Sample IDs as strings; missing or duplicated IDs are an error (an ID names one sample)."""
    index = pd.Index(list(sample_ids))
    if index.hasnans:
        raise ValueError("sample IDs contain missing values")
    ids = [str(i) for i in index]
    as_str = pd.Index(ids)
    duplicated = as_str[as_str.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"duplicated sample IDs, e.g. {_first(duplicated)}")
    return ids


def _per_sample(ids: list[str], values: PerSample, what: str) -> np.ndarray:
    """One value per ID: a Series is matched by sample ID, anything else by position."""
    if isinstance(values, pd.Series):
        by_id = values.set_axis(values.index.map(str))
        if not by_id.index.is_unique:
            raise ValueError(f"{what}: the Series index repeats some sample IDs")
        missing = [i for i in ids if i not in by_id.index]
        if missing:
            raise ValueError(f"{what} missing for {len(missing)} sample IDs: {_first(missing)}")
        out = by_id.loc[ids].to_numpy()
    else:
        out = np.asarray(values)
        if out.shape != (len(ids),):
            raise ValueError(f"{what} has shape {out.shape}; expected one value per sample")
    if pd.isna(out).any():
        raise ValueError(f"{what} contain missing values")
    return out


def _fold_numbers(splits: Iterator[tuple[np.ndarray, np.ndarray]], n_samples: int) -> np.ndarray:
    """Fold number of each position, from a scikit-learn splitter's (train, test) pairs."""
    fold = np.full(n_samples, -1)
    for k, (_, test_pos) in enumerate(splits):
        fold[test_pos] = k
    return fold


def _n_held_out(fraction: float, n_items: int) -> int:
    """ceil(fraction * n_items), but at least 1 and leaving at least 1 item on the other side."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be strictly between 0 and 1, got {fraction}")
    if n_items < 2:
        raise ValueError(f"need at least 2 samples (or groups) to split, got {n_items}")
    n_held_out = math.ceil(round(fraction * n_items, _SIZE_ROUNDING_DIGITS))
    return min(max(n_held_out, 1), n_items - 1)


def _ids_at(ids: list[str], positions: np.ndarray) -> list[str]:
    """IDs at the given positions, in input order (so outputs never depend on draw order)."""
    return [ids[p] for p in np.sort(positions)]


def _class_quotas(class_sizes: Sequence[int], n: int) -> np.ndarray:
    """Split `n` into per-class integer quotas proportional to `class_sizes` (n < total).

    Largest-remainder (Hamilton) rounding in integer arithmetic, so ties are exact and go to the
    class listed first. Then, if n >= number of classes, each class left at 0 takes one slot from
    the class that is furthest above its exact share.
    """
    sizes = np.asarray(class_sizes, dtype=np.int64)
    total = int(sizes.sum())
    share_x_total = sizes * n  # each class's exact share, multiplied by `total` to stay integer
    quotas = share_x_total // total
    leftover = n - int(quotas.sum())
    largest_remainders = np.argsort(-(share_x_total % total), kind="stable")[:leftover]
    quotas[largest_remainders] += 1
    if n >= len(sizes):
        for empty in np.flatnonzero(quotas == 0):
            # A class with a single slot cannot donate it (it would become empty in turn).
            excess = np.where(quotas >= 2, quotas * total - share_x_total, np.iinfo(np.int64).min)
            donor = int(np.argmax(excess))
            quotas[donor] -= 1
            quotas[empty] += 1
    return quotas


def _verify_saved_folds(
    saved: pd.DataFrame, ids: list[str], label_values: np.ndarray, n_splits: int, path: Path
) -> None:
    """Raise ValueError (saying how to regenerate) if saved folds no longer match the data."""
    problems = []
    saved_ids = set(saved["sample_id"].tolist())
    missing, extra = sorted(set(ids) - saved_ids), sorted(saved_ids - set(ids))
    if missing:
        problems.append(f"{len(missing)} samples are not in the file, e.g. {_first(missing)}")
    if extra:
        problems.append(f"{len(extra)} samples in the file are not in the data: {_first(extra)}")
    if "label" not in saved.columns:
        problems.append("the file has no 'label' column")
    elif not missing:
        saved_labels = saved.set_index("sample_id")["label"]
        current = zip(ids, label_values, strict=True)
        changed = [i for i, label in current if saved_labels[i] != str(label)]
        if changed:
            problems.append(f"{len(changed)} samples changed label, e.g. {_first(changed)}")
    n_saved = saved["fold"].nunique()
    if n_saved != n_splits:
        problems.append(f"the file has {n_saved} folds but cv.n_splits is {n_splits}")
    if problems:
        raise ValueError(
            f"Saved folds {path} do not match the current samples/config: {'; '.join(problems)}. "
            f"If the change is intended, delete {path} and rerun to regenerate the folds, then "
            "re-run every model so that all of them share the new folds."
        )


def _first(items: Sequence[Hashable]) -> list[Hashable]:
    """The first few items, for error messages."""
    return list(items[:MAX_REPORTED_IDS])
