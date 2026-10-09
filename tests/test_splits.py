"""Tests for m2m.splits: folds, inner splits, subsampling and leakage checks (synthetic IDs)."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from m2m import paths
from m2m.splits import (
    LeakageError,
    assert_disjoint,
    fold_train_test,
    grouped_inner_split,
    load_folds,
    make_grouped_folds,
    make_stratified_folds,
    prism_folds,
    save_folds,
    stratified_inner_split,
    stratified_subsample,
)

PRISM_COUNTS = {"CD": 68, "UC": 53, "Control": 34}
CV_CFG = {"cv": {"n_splits": 5, "seed": 20261008}}


def prism_like(seed: int = 0) -> tuple[list[str], list[str]]:
    """155 PRISM-style IDs with the real diagnosis counts, in shuffled order."""
    ids = [f"PRISM|{7000 + i}" for i in range(155)]
    labels = np.repeat(list(PRISM_COUNTS), list(PRISM_COUNTS.values()))
    np.random.default_rng(seed).shuffle(labels)
    return ids, labels.tolist()


def in_input_order(subset: list[str], ids: list[str]) -> bool:
    position = {sample: k for k, sample in enumerate(ids)}
    return subset == sorted(subset, key=position.__getitem__)


# --- stratified folds -------------------------------------------------------------------------


def test_stratified_folds_partition_and_balance() -> None:
    ids, labels = prism_like()
    folds = make_stratified_folds(ids, labels, n_splits=5, seed=1)
    assert list(folds.columns) == ["sample_id", "label", "fold"]
    assert folds["sample_id"].tolist() == ids
    assert folds["label"].tolist() == labels
    assert sorted(folds["fold"].unique()) == [0, 1, 2, 3, 4]
    test_sets = [fold_train_test(folds, k)[1] for k in range(5)]
    assert sorted(sum(test_sets, [])) == sorted(ids)  # every sample in exactly one test fold
    per_fold = pd.crosstab(folds["label"], folds["fold"])
    for label, counts in per_fold.iterrows():
        assert counts.max() - counts.min() <= 1
        assert (abs(counts - PRISM_COUNTS[label] / 5) <= 1).all()


def test_stratified_folds_are_deterministic() -> None:
    ids, labels = prism_like()
    first = make_stratified_folds(ids, labels, n_splits=5, seed=1)
    pd.testing.assert_frame_equal(first, make_stratified_folds(ids, labels, n_splits=5, seed=1))
    other_seed = make_stratified_folds(ids, labels, n_splits=5, seed=2)
    assert not first["fold"].equals(other_seed["fold"])


def test_labels_series_is_matched_by_sample_id() -> None:
    ids, labels = prism_like()
    shuffled = pd.Series(labels, index=ids).iloc[::-1]
    with_extra = pd.concat([shuffled, pd.Series({"PRISM|9999": "CD"})])
    by_position = make_stratified_folds(ids, labels, n_splits=5, seed=1)
    by_id = make_stratified_folds(ids, with_extra, n_splits=5, seed=1)
    pd.testing.assert_frame_equal(by_position, by_id)
    with pytest.raises(ValueError, match="missing"):
        make_stratified_folds([*ids, "PRISM|1"], shuffled, n_splits=5, seed=1)


def test_bad_ids_or_labels_are_rejected() -> None:
    ids, labels = prism_like()
    with pytest.raises(ValueError, match="duplicated"):
        make_stratified_folds([*ids, ids[0]], [*labels, "CD"], n_splits=5, seed=1)
    with pytest.raises(ValueError, match="one value per sample"):
        make_stratified_folds(ids, labels[:-1], n_splits=5, seed=1)
    with pytest.raises(ValueError, match="missing values"):
        make_stratified_folds(ids, [*labels[:-1], None], n_splits=5, seed=1)


def test_fold_train_test_is_disjoint_and_covering() -> None:
    ids, labels = prism_like()
    folds = make_stratified_folds(ids, labels, n_splits=5, seed=1)
    for k in range(5):
        train, test = fold_train_test(folds, k)
        assert_disjoint(train, test, what=f"fold {k}")
        assert sorted(train + test) == sorted(ids)
        assert in_input_order(train, ids) and in_input_order(test, ids)
    with pytest.raises(ValueError, match="fold 7"):
        fold_train_test(folds, 7)


# --- saving and the PRISM folds ---------------------------------------------------------------


def test_save_load_roundtrip_keeps_ids_and_labels_as_strings(tmp_path: Path) -> None:
    folds = pd.DataFrame(
        {"sample_id": ["PRISM|7122", "007", "x"], "label": ["NA", "CD", "UC"], "fold": [0, 1, 0]}
    )
    path = tmp_path / "nested" / "folds.csv"
    save_folds(folds, path)
    pd.testing.assert_frame_equal(load_folds(path), folds)


def test_prism_folds_created_once_then_reloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "SPLITS", tmp_path / "splits")
    ids, labels = prism_like()
    created = prism_folds(ids, labels, CV_CFG)
    assert (tmp_path / "splits" / "prism_folds.csv").exists()
    expected = make_stratified_folds(ids, labels, n_splits=5, seed=20261008)
    pd.testing.assert_frame_equal(created, expected)
    pd.testing.assert_frame_equal(prism_folds(ids, labels, CV_CFG), created)
    # The saved assignment is reused when the same samples arrive in another order.
    reordered = prism_folds(ids[::-1], labels[::-1], CV_CFG)
    assert reordered["sample_id"].tolist() == ids[::-1]
    pd.testing.assert_frame_equal(reordered.iloc[::-1].reset_index(drop=True), created)


def test_prism_folds_refuse_stale_files(tmp_path: Path) -> None:
    path = tmp_path / "folds.csv"
    ids, labels = prism_like()
    prism_folds(ids, labels, CV_CFG, path=path)
    relabelled = ["UC" if label == "CD" else label for label in labels]
    with pytest.raises(ValueError, match=r"changed label.*delete .* regenerate"):
        prism_folds(ids, relabelled, CV_CFG, path=path)
    with pytest.raises(ValueError, match="not in the file"):
        prism_folds([*ids, "PRISM|9999"], [*labels, "CD"], CV_CFG, path=path)
    with pytest.raises(ValueError, match="not in the data"):
        prism_folds(ids[1:], labels[1:], CV_CFG, path=path)
    with pytest.raises(ValueError, match="n_splits"):
        prism_folds(ids, labels, {"cv": {"n_splits": 10, "seed": 1}}, path=path)


# --- inner splits -----------------------------------------------------------------------------


def test_stratified_inner_split_is_disjoint_covering_and_stratified() -> None:
    ids, labels = prism_like()
    train, _ = fold_train_test(make_stratified_folds(ids, labels, n_splits=5, seed=1), 0)
    all_labels = pd.Series(labels, index=ids)  # full metadata column, matched by ID
    fit, val = stratified_inner_split(train, all_labels, fraction=0.2, seed=1000)
    assert_disjoint(fit, val, what="inner split")
    assert sorted(fit + val) == sorted(train)
    assert len(val) == math.ceil(0.2 * len(train))
    assert in_input_order(fit, train) and in_input_order(val, train)
    train_counts = Counter(all_labels[train])
    val_counts = Counter(all_labels[val])
    for label, count in train_counts.items():
        assert abs(val_counts[label] - len(val) * count / len(train)) <= 1
    assert (fit, val) == stratified_inner_split(train, all_labels, fraction=0.2, seed=1000)
    assert val != stratified_inner_split(train, all_labels, fraction=0.2, seed=1001)[1]


def test_stratified_inner_split_falls_back_when_too_small() -> None:
    ids = [f"s{i}" for i in range(10)]
    fit, val = stratified_inner_split(ids, ["A"] * 9 + ["B"], fraction=0.2, seed=0)  # B has 1
    assert len(val) == 2 and sorted(fit + val) == sorted(ids)
    six = ids[:6]
    fit, val = stratified_inner_split(six, ["A", "A", "B", "B", "C", "C"], fraction=0.2, seed=0)
    assert len(val) == 2 and sorted(fit + val) == sorted(six)  # 2 val < 3 classes
    fit, val = stratified_inner_split(ids[:3], ["A", "B", "C"], fraction=0.2, seed=0)
    assert len(val) == 1 and len(fit) == 2  # at least one validation sample
    with pytest.raises(ValueError, match="at least 2"):
        stratified_inner_split(ids[:1], ["A"], fraction=0.2, seed=0)
    with pytest.raises(ValueError, match="fraction"):
        stratified_inner_split(ids, ["A"] * 10, fraction=1.0, seed=0)


def test_grouped_inner_split_keeps_subjects_together() -> None:
    ids = [f"S{i}" for i in range(60)]
    subjects = [f"subject{i % 17}" for i in range(60)]  # 17 subjects with 3-4 samples each
    subject_of = dict(zip(ids, subjects, strict=True))
    fit, val = grouped_inner_split(ids, subjects, fraction=0.25, seed=3)
    assert_disjoint({subject_of[i] for i in fit}, {subject_of[i] for i in val}, what="subjects")
    assert sorted(fit + val) == sorted(ids)
    assert len({subject_of[i] for i in val}) == math.ceil(0.25 * 17)
    assert in_input_order(fit, ids) and in_input_order(val, ids)
    assert (fit, val) == grouped_inner_split(ids, subjects, fraction=0.25, seed=3)
    assert val != grouped_inner_split(ids, subjects, fraction=0.25, seed=4)[1]


def test_grouped_folds_never_split_a_group() -> None:
    ids = [f"S{i}" for i in range(100)]
    groups = [f"subject{g}" for g in np.random.default_rng(0).integers(0, 23, size=100)]
    folds = make_grouped_folds(ids, groups, n_splits=5, seed=1)
    assert list(folds.columns) == ["sample_id", "group", "fold"]
    assert folds["sample_id"].tolist() == ids
    assert sorted(folds["fold"].unique()) == [0, 1, 2, 3, 4]
    assert (folds.groupby("group")["fold"].nunique() == 1).all()
    groups_per_fold = folds.groupby("fold")["group"].nunique()
    assert groups_per_fold.max() - groups_per_fold.min() <= 1  # shuffle=True balances groups
    pd.testing.assert_frame_equal(folds, make_grouped_folds(ids, groups, n_splits=5, seed=1))
    other_seed = make_grouped_folds(ids, groups, n_splits=5, seed=2)
    assert not folds["fold"].equals(other_seed["fold"])


# --- subsampling ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (20, {"CD": 9, "UC": 7, "Control": 4}),  # exact shares 8.77 / 6.84 / 4.39
        (40, {"CD": 17, "UC": 14, "Control": 9}),  # 17.55 / 13.68 / 8.77
        (80, {"CD": 35, "UC": 27, "Control": 18}),  # 35.10 / 27.35 / 17.55
    ],
)
def test_stratified_subsample_largest_remainder(n: int, expected: dict[str, int]) -> None:
    ids, labels = prism_like()
    chosen = stratified_subsample(ids, labels, n=n, seed=0)
    assert len(chosen) == len(set(chosen)) == n
    label_of = dict(zip(ids, labels, strict=True))
    assert Counter(label_of[i] for i in chosen) == expected
    assert in_input_order(chosen, ids)


def test_stratified_subsample_all_ids_nested_and_deterministic() -> None:
    ids, labels = prism_like()
    assert stratified_subsample(ids, labels, n=155, seed=3) == ids
    subsets = [set(stratified_subsample(ids, labels, n=n, seed=4)) for n in (20, 40, 80, 155)]
    assert all(small <= large for small, large in zip(subsets, subsets[1:], strict=False))
    first = stratified_subsample(ids, labels, n=40, seed=4)
    assert first == stratified_subsample(ids, labels, n=40, seed=4)
    assert first != stratified_subsample(ids, labels, n=40, seed=5)
    for bad_n in (0, 156):
        with pytest.raises(ValueError, match="between 1"):
            stratified_subsample(ids, labels, n=bad_n, seed=0)


def test_stratified_subsample_keeps_every_class() -> None:
    ids = [f"s{i}" for i in range(100)]
    labels = ["A"] * 50 + ["B"] * 49 + ["C"]  # C's exact share of 10 is 0.1
    label_of = dict(zip(ids, labels, strict=True))
    chosen = stratified_subsample(ids, labels, n=10, seed=0)
    assert Counter(label_of[i] for i in chosen) == {"A": 5, "B": 4, "C": 1}


def test_stratified_subsample_random_configurations() -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        sizes = rng.integers(1, 30, size=rng.integers(2, 6))
        labels = np.repeat([f"class{k}" for k in range(len(sizes))], sizes).tolist()
        ids = [f"s{i}" for i in range(len(labels))]
        n = int(rng.integers(1, len(ids) + 1))
        chosen = stratified_subsample(ids, labels, n=n, seed=int(rng.integers(1000)))
        assert len(chosen) == len(set(chosen)) == n
        if n >= len(sizes):
            assert {labels[int(i[1:])] for i in chosen} == set(labels)


# --- leakage check ----------------------------------------------------------------------------


def test_assert_disjoint() -> None:
    assert_disjoint(["a", "b"], ["c"], what="ok")
    assert issubclass(LeakageError, AssertionError)
    train = [f"id{i}" for i in range(20)]
    test = [f"id{i}" for i in range(8, 30)]
    with pytest.raises(LeakageError, match="train/test: 12 IDs") as caught:
        assert_disjoint(train, test, what="train/test")
    assert len(re.findall(r"id\d+", str(caught.value))) == 10  # lists at most 10 overlaps


def test_assert_disjoint_matches_ids_across_types() -> None:
    """An ID parsed as a number in one table and read as text in another is the same sample."""
    with pytest.raises(LeakageError, match="source vs target: 1 IDs"):
        assert_disjoint([7122, 7123], ["7123", "9999"], what="source vs target")
    with pytest.raises(LeakageError):
        assert_disjoint(np.array([10, 11]), pd.Index(["11"]), what="numpy vs pandas")
    with pytest.raises(LeakageError):
        assert_disjoint([1.0], [1], what="equal values still match")
    assert_disjoint([1, 2], ["3"], what="no overlap")
