"""E4 data assembly: source-only vocabulary, source/target checks, HMDB targets, iHMP NaN policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from m2m import data, paths, splits, transfer_data
from m2m.config import load_config
from m2m.transfer_data import HmdbTargetPreprocessor

CFG = load_config(paths.CONFIGS / "e4_transfer.yaml")
IHMP, YACHIDA, MARS = "iHMP_IBDMDB_2019", "YACHIDA_CRC_2019", "MARS_IBS_2020"


def _genus(name: str) -> str:
    return f"d__Bacteria;p__Firmicutes;c__Clostridia;o__O;f__F;g__{name}"


def _cohort(
    dataset: str,
    ids: list[str],
    subjects: list[str],
    abundances: np.ndarray,
    genera: list[str],
    metabolites: pd.DataFrame | None = None,
) -> data.Cohort:
    microbes = pd.DataFrame(abundances, index=ids, columns=[_genus(g) for g in genera])
    meta = pd.DataFrame({"subject": subjects, "diagnosis": "x", "dataset": dataset}, index=ids)
    return data.Cohort(dataset, microbes, metabolites, meta)


def _random_cohort(
    dataset: str, prefix: str, n: int, genera: list[str], seed: int, per_subject: int = 1
) -> data.Cohort:
    rng = np.random.default_rng(seed)
    values = rng.lognormal(0.0, 1.5, size=(n, len(genera)))
    values[rng.random(values.shape) < 0.3] = 0.0
    values[:, 0] += 0.1  # no empty profile
    values /= values.sum(axis=1, keepdims=True)
    ids = [f"{prefix}{i:03d}" for i in range(n)]
    subjects = [f"{prefix}subj{i // per_subject}" for i in range(n)]
    return _cohort(dataset, ids, subjects, values, genera)


# --- vocabulary --------------------------------------------------------------------------------


def test_vocabulary_is_the_sorted_union_of_each_sources_prevalent_genera():
    # Genus "rare" is present (above 1e-4) in 1 of 20 YACHIDA samples: below 10% prevalence.
    n = 20
    yachida = np.full((n, 3), 0.3)
    yachida[:, 2] = 0.0
    yachida[0, 2] = 0.2
    mars = np.full((n, 2), 0.5)
    sources = {
        YACHIDA: _cohort(
            YACHIDA,
            [f"y{i}" for i in range(n)],
            [f"y{i}" for i in range(n)],
            yachida,
            ["Zeta", "Alpha", "rare"],
        ),
        MARS: _cohort(
            MARS,
            [f"m{i}" for i in range(n)],
            [f"s{i // 4}" for i in range(n)],
            mars,
            ["Alpha", "Beta"],
        ),
    }
    vocabulary = transfer_data.source_vocabulary(sources, CFG)
    assert vocabulary == sorted([_genus("Alpha"), _genus("Beta"), _genus("Zeta")])


def test_vocabulary_depends_on_sources_only():
    genera = ["A", "B", "C", "D"]
    sources = {
        YACHIDA: _random_cohort(YACHIDA, "y", 30, genera, seed=1),
        MARS: _random_cohort(MARS, "m", 30, genera[:3], seed=2, per_subject=5),
    }
    before = transfer_data.source_vocabulary(sources, CFG)
    # The target is never an argument; and a sources mapping that includes it is refused.
    target = _random_cohort(data.TARGET_DATASET, "PRISM.", 20, ["A", "E", "F"], seed=3)
    with pytest.raises(ValueError, match="never include the target"):
        transfer_data.source_vocabulary({**sources, data.TARGET_DATASET: target}, CFG)
    assert transfer_data.source_vocabulary(sources, CFG) == before
    assert transfer_data.vocabulary_digest(before) != transfer_data.vocabulary_digest(before[::-1])


def test_pretraining_table_fills_unreported_genera_and_keys_subjects_per_dataset():
    sources = {
        IHMP: _cohort(
            IHMP, ["h1", "h2"], ["S1", "S1"], np.array([[0.5, 0.5], [1.0, 0.0]]), ["A", "B"]
        ),
        MARS: _cohort(MARS, ["m1"], ["S1"], np.array([[0.25, 0.75]]), ["B", "C"]),
    }
    table, keys = transfer_data.pretraining_table(sources)
    assert list(table.columns) == sorted(_genus(g) for g in "ABC")
    assert table.loc["m1", _genus("A")] == 0.0
    assert table.loc["h1", _genus("C")] == 0.0
    assert table.loc["m1", _genus("C")] == 0.75
    # The same raw subject ID in two datasets is two people.
    assert keys.to_dict() == {"h1": f"{IHMP}:S1", "h2": f"{IHMP}:S1", "m1": f"{MARS}:S1"}


def test_pretraining_table_rejects_sample_ids_that_collide_across_sources():
    sources = {
        IHMP: _cohort(IHMP, ["x1"], ["S1"], np.array([[1.0]]), ["A"]),
        MARS: _cohort(MARS, ["x1"], ["S2"], np.array([[1.0]]), ["A"]),
    }
    with pytest.raises(ValueError, match="collide"):
        transfer_data.pretraining_table(sources)


def test_vocabulary_coverage_counts_covered_target_genera():
    target = _cohort(
        "T",
        ["t1", "t2"],
        ["t1", "t2"],
        np.array([[0.5, 0.5, 0.0], [0.4, 0.3, 0.3]]),
        ["A", "B", "Zz"],
    )
    coverage = transfer_data.vocabulary_coverage([_genus("A"), _genus("Q")], target.microbes, CFG)
    assert coverage["target_prevalent_genera"] == 3
    assert coverage["covered"] == 1
    assert coverage["missing_genera"] == ["B", "Zz"]
    assert coverage["abundance_share_in_vocabulary"]["mean"] == pytest.approx(0.45)


# --- source / target disjointness ----------------------------------------------------------------


@pytest.fixture
def two_cohorts():
    genera = [f"G{i}" for i in range(12)]
    sources = {
        YACHIDA: _random_cohort(YACHIDA, "y", 25, genera, seed=10),
        MARS: _random_cohort(MARS, "m", 20, genera, seed=11, per_subject=4),
    }
    target = _random_cohort(data.TARGET_DATASET, "PRISM.", 15, genera, seed=12)
    vocabulary = transfer_data.source_vocabulary(sources, CFG)
    return sources, target, vocabulary


def _check(sources, target, vocabulary, threshold=0.995):
    return transfer_data.check_source_target_disjoint(
        sources,
        target.meta,
        target.microbes,
        vocabulary,
        max_correlation=threshold,
        target_name="target",
    )


def test_disjoint_cohorts_pass_and_report_the_largest_profile_correlation(two_cohorts):
    sources, target, vocabulary = two_cohorts
    report = _check(sources, target, vocabulary)
    assert report["shared_sample_or_subject_ids"] == 0
    assert -1.0 <= report["max_profile_correlation"] < 0.995
    assert report["n_source_samples"] == 45


@pytest.mark.parametrize("column", ["sample", "subject"])
def test_disjointness_check_raises_on_a_planted_shared_id(two_cohorts, column):
    sources, target, vocabulary = two_cohorts
    meta = target.meta.copy()
    if column == "sample":
        meta = meta.rename(index={meta.index[0]: "y003"})
        microbes = target.microbes.rename(index={target.microbes.index[0]: "y003"})
    else:
        meta.loc[meta.index[0], "subject"] = "msubj2"  # a MARS subject
        microbes = target.microbes
    planted = data.Cohort(target.name, microbes, None, meta)
    with pytest.raises(splits.LeakageError, match="IDs"):
        _check(sources, planted, vocabulary)


def test_disjointness_check_raises_on_a_republished_profile_under_a_new_id(two_cohorts):
    sources, target, vocabulary = two_cohorts
    microbes = target.microbes.copy()
    copied = sources[MARS].microbes.iloc[[3]]
    microbes.iloc[0] = copied.reindex(columns=microbes.columns, fill_value=0.0).iloc[0] * 0.7
    planted = data.Cohort(target.name, microbes, None, target.meta)
    with pytest.raises(splits.LeakageError, match="near-identical"):
        _check(sources, planted, vocabulary)


def test_max_profile_correlation_finds_the_most_similar_pair():
    a = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    b = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 1.0]])  # row 0 is flat: correlation 0
    best, i, j = transfer_data.max_profile_correlation(a, b)
    assert (i, j) == (1, 1)
    assert best == pytest.approx(1.0)


# --- shared HMDB targets -------------------------------------------------------------------------


def test_shared_hmdb_targets_use_t466_features_only():
    franzosa_map = pd.DataFrame(
        {"HMDB": ["HMDB01", "HMDB01", "HMDB02", "HMDB03", None]},
        index=["F1: a", "F2: a", "F3: b", "F9: c", "F4: d"],
    )
    ihmp_map = pd.DataFrame(
        {"HMDB": ["HMDB01", "HMDB03", "HMDB07"], "Method": "C18-neg"},
        index=["c1", "c3", "c7"],
    )
    shared = transfer_data.shared_hmdb_targets(
        franzosa_map, ihmp_map, t466=["F1", "F2", "F3", "F4"]
    )
    # HMDB03 is measured by F9, which is not in T466; HMDB02 is not in iHMP.
    assert shared.hmdb_order == ("HMDB01",)
    assert shared.franzosa == {"F1": "HMDB01", "F2": "HMDB01"}
    assert shared.ihmp == {"c1": "HMDB01"}


def test_representative_feature_rule():
    Y = pd.DataFrame(
        {
            # HMDB-A: f1 has 1 zero, f2 has 2 missing (NaN counts as not detected) -> f1.
            "f1": [0.0, 5.0, 6.0, 7.0],
            "f2": [np.nan, 0.0, 60.0, 70.0],
            # HMDB-B: both detected everywhere -> the higher median (g2), whatever the order.
            "g1": [1.0, 2.0, 3.0, 4.0],
            "g2": [10.0, 20.0, 30.0, 40.0],
            # HMDB-C: identical columns -> the first name (h1).
            "h2": [1.0, 2.0, 3.0, 4.0],
            "h1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    mapping = {"f1": "A", "f2": "A", "g1": "B", "g2": "B", "h1": "C", "h2": "C"}
    assert transfer_data.representative_features(Y, mapping) == {"A": "f1", "B": "g2", "C": "h1"}
    reordered = Y[list(reversed(Y.columns))]
    assert transfer_data.representative_features(reordered, mapping) == {
        "A": "f1",
        "B": "g2",
        "C": "h1",
    }


def test_representative_features_need_a_column_for_every_hmdb_id():
    Y = pd.DataFrame({"f1": [1.0, 2.0]})
    with pytest.raises(KeyError, match="HMDB"):
        transfer_data.representative_features(Y, {"f1": "A", "missing": "B"})


def _hmdb_prep(max_zero_fraction: float = 0.5) -> HmdbTargetPreprocessor:
    return HmdbTargetPreprocessor(
        feature_to_hmdb={"f1": "H1", "f2": "H1", "g1": "H2", "k1": "H3"},
        hmdb_order=["H1", "H2", "H3"],
        metabolites={**CFG["metabolites"], "max_zero_fraction": max_zero_fraction},
        mostly_zero_fraction=0.5,
    )


def _hmdb_table(rng: np.random.Generator, n: int) -> pd.DataFrame:
    Y = pd.DataFrame(
        {
            "f1": rng.uniform(1, 100, n),
            "f2": rng.uniform(1, 100, n),
            "g1": rng.uniform(1, 100, n),
            "k1": rng.uniform(1, 100, n),
            "other": rng.uniform(1, 100, n),
        },
        index=[f"s{i}" for i in range(n)],
    )
    Y.iloc[: n // 2, 0] = 0.0  # f1 often missing -> f2 represents H1
    Y.iloc[: (3 * n) // 4, 3] = 0.0  # H3 is mostly zero
    return Y


def test_hmdb_targets_are_named_by_hmdb_in_head_order_and_zero_filtered(rng):
    Y = _hmdb_table(rng, 20)
    prep = _hmdb_prep().fit(Y)
    assert prep.representatives_ == {"H1": "f2", "H2": "g1", "H3": "k1"}
    assert prep.features_ == ["H1", "H2"]  # H3 dropped by the zero filter
    assert prep.n_dropped_zero_ == 1
    assert prep.n_mostly_zero_ == 1
    out = prep.transform(Y)
    expected = np.log10(Y["f2"] + 1.0)
    assert np.allclose(out["H1"], (expected - expected.mean()) / expected.std(ddof=0))
    assert prep.fit_samples_ == list(Y.index)


def test_hmdb_targets_keep_everything_when_the_zero_filter_is_off(rng):
    prep = _hmdb_prep(max_zero_fraction=1.0).fit(_hmdb_table(rng, 20))
    assert prep.features_ == ["H1", "H2", "H3"]
    assert prep.n_mostly_zero_ == 1  # still counted


def test_hmdb_target_choices_never_depend_on_the_rows_they_transform(rng):
    train, test = _hmdb_table(rng, 20), _hmdb_table(rng, 10)
    prep = _hmdb_prep().fit(train)
    before = (dict(prep.representatives_), list(prep.features_), prep.inner_.mean_.copy())
    flipped = test.copy()
    flipped["f2"] = 0.0  # would make f1 the representative if test rows were used
    prep.transform(flipped)
    assert (prep.representatives_, prep.features_) == before[:2]
    assert np.array_equal(prep.inner_.mean_, before[2])


# --- iHMP missing values -------------------------------------------------------------------------


def test_ihmp_missing_evidence_tells_whole_method_gaps_from_scattered_nan():
    Y = pd.DataFrame(
        {
            "a1": [10.0, np.nan, 30.0, 40.0],
            "a2": [12.0, np.nan, 31.0, 44.0],
            "b1": [np.nan, 25.0, 35.0, 45.0],
            "b2": [15.0, 26.0, 36.0, 46.0],
        },
        index=list("pqrs"),
    )
    mtb_map = pd.DataFrame(
        {"Method": ["HILIC-pos", "HILIC-pos", "C18-neg", "C18-neg"]}, index=list(Y.columns)
    )
    evidence = transfer_data.ihmp_missing_evidence(Y, mtb_map, shared_compounds=["b1", "b2"])
    assert evidence["n_missing"] == 3
    assert evidence["n_missing_in_whole_missing_methods"] == 2  # sample q lacks all of HILIC-pos
    assert evidence["n_missing_scattered"] == 1
    assert evidence["per_method"]["HILIC-pos"]["samples_missing_the_whole_method"] == 1
    assert evidence["missing_fraction_shared_hmdb_features"] == pytest.approx(1 / 8)
    assert evidence["n_exact_zeros"] == 0
    assert evidence["policy"] == transfer_data.IHMP_MISSING_POLICY


def test_not_detected_values_become_zero_and_then_count_for_the_zero_filter():
    Y = pd.DataFrame({"c1": [np.nan, 5.0, np.nan], "c2": [1.0, 2.0, 3.0]}, index=list("abc"))
    filled = transfer_data.fill_not_detected(Y)
    assert filled["c1"].tolist() == [0.0, 5.0, 0.0]
    assert Y["c1"].isna().sum() == 2  # the input is not modified
