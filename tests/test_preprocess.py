"""Tests for m2m.preprocess: fit on training rows only, transform anything, serialise exactly."""

from __future__ import annotations

import copy
import json
import math
import pickle
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from m2m import paths
from m2m.config import load_config
from m2m.preprocess import (
    FittedPreprocessor,
    MetabolitePreprocessor,
    MicrobePreprocessor,
    StandardScalerDF,
    fit_preprocessor,
    half_min_nonzero_pseudocount,
    prevalent_taxa,
    transform,
)

N_TRAIN = 40
TRANSFORM_PAIRS = [("clr", "log10"), ("log", "log10"), ("clr", "clr"), ("log", "clr")]


def make_cfg(x_transform: str = "clr", y_transform: str = "log10") -> dict:
    """The preprocessing sections of configs/base.yaml, with selectable transforms."""
    return {
        "microbes": {
            "abundance_threshold": 1e-4,
            "min_prevalence": 0.10,
            "transform": x_transform,
            "standardize": True,
        },
        "metabolites": {"transform": y_transform, "max_zero_fraction": 0.5, "standardize": True},
    }


@pytest.fixture
def data(synthetic_cohort: dict) -> dict[str, pd.DataFrame]:
    """The shared synthetic cohort plus one rare taxon and one mostly-zero metabolite.

    Both extra columns must be removed by the filters, so the tests also exercise the
    "fitted on train, absent from the fitted features" paths. Rows 0-39 train, 40-59 test.
    """
    X, Y = synthetic_cohort["X"], synthetic_cohort["Y"]
    rare = np.zeros(len(X))
    rare[[0, 45]] = 1e-3  # present in 1 of 40 training rows (2.5% < 10%)
    sparse = np.zeros(len(Y))
    sparse[::4] = 50.0  # zero in 75% of rows (> 50%)
    X = X.assign(rare_taxon=rare)
    Y = Y.assign(mostly_zero=sparse)
    return {
        "X_train": X.iloc[:N_TRAIN],
        "Y_train": Y.iloc[:N_TRAIN],
        "X_test": X.iloc[N_TRAIN:],
        "Y_test": Y.iloc[N_TRAIN:],
    }


# --------------------------------------------------------------------------------------------
# The leakage guarantee required by the brief (section 4)
# --------------------------------------------------------------------------------------------


def mutated_test_sets(
    data: dict[str, pd.DataFrame], rng: np.random.Generator
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Test sets that differ wildly from the real one; none may move a fitted parameter."""
    X, Y = data["X_test"], data["Y_test"]
    yield "original", X, Y
    yield (
        "extreme values",
        (X * 1e6).assign(taxon_00=1e12),
        (Y * 1e6).assign(met_00=1e15),
    )
    yield "all zeros", X * 0.0, Y * 0.0
    rows = rng.permutation(len(X))
    yield (
        "shuffled rows and columns",
        X.iloc[rows, rng.permutation(X.shape[1])],
        Y.iloc[rows, rng.permutation(Y.shape[1])],
    )
    yield (
        "extra and missing columns",
        X.drop(columns=["taxon_01", "taxon_02"]).assign(unseen_taxon=rng.uniform(0, 1, len(X))),
        Y.assign(unseen_metabolite=rng.uniform(0, 1e5, len(Y))),
    )
    yield "single sample", X.iloc[[0]], Y.iloc[[0]]
    yield "no samples", X.iloc[:0], Y.iloc[:0]
    yield (
        "train and test stacked",
        pd.concat([data["X_train"], X]),
        pd.concat([data["Y_train"], Y]),
    )


@pytest.mark.parametrize(("x_transform", "y_transform"), TRANSFORM_PAIRS)
def test_changing_test_data_never_changes_fitted_parameters(
    data: dict, rng: np.random.Generator, x_transform: str, y_transform: str
) -> None:
    cfg = make_cfg(x_transform, y_transform)
    prep = fit_preprocessor(cfg, data["X_train"], data["Y_train"])
    fitted_before = copy.deepcopy(prep.to_dict())
    X_train_t, Y_train_t = transform(prep, data["X_train"], data["Y_train"])

    for name, X_other, Y_other in mutated_test_sets(data, rng):
        X_t, Y_t = transform(prep, X_other, Y_other)
        assert np.isfinite(X_t.to_numpy()).all() and np.isfinite(Y_t.to_numpy()).all(), name
        assert prep.to_dict() == fitted_before, name
        X_again, Y_again = transform(prep, data["X_train"], data["Y_train"])
        pdt.assert_frame_equal(X_again, X_train_t, check_exact=True, obj=name)
        pdt.assert_frame_equal(Y_again, Y_train_t, check_exact=True, obj=name)

    # Rejected inputs leave no trace either.
    with pytest.raises(KeyError):
        transform(prep, data["X_test"], data["Y_test"].drop(columns=prep.y.features_[:3]))
    X_nan = data["X_test"].copy()
    X_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        transform(prep, X_nan, data["Y_test"])
    assert prep.to_dict() == fitted_before

    # The fit saw training rows only, and the same rows always give the same parameters.
    assert prep.x.fit_samples_ == prep.y.fit_samples_ == list(data["X_train"].index)
    assert fit_preprocessor(cfg, data["X_train"], data["Y_train"]).to_dict() == fitted_before


@pytest.mark.parametrize(("x_transform", "y_transform"), TRANSFORM_PAIRS)
def test_each_sample_is_transformed_on_its_own(
    data: dict, x_transform: str, y_transform: str
) -> None:
    # Transductive leakage stores nothing, so the parameter check above cannot see it: e.g.
    # z-scoring a test batch with its own mean/std, or a fallback pseudocount taken from the
    # batch. Either would make a sample's features depend on the samples it is batched with.
    prep = fit_preprocessor(make_cfg(x_transform, y_transform), data["X_train"], data["Y_train"])
    X, Y = data["X_test"].copy(), data["Y_test"].copy()
    X.iloc[0] = 0.0  # no kept taxon: this sample needs the fitted fallback pseudocount
    Y.iloc[1] = 0.0  # likewise for the metabolite CLR
    X_t, Y_t = transform(prep, X, Y)
    for i in range(len(X)):
        X_i, Y_i = transform(prep, X.iloc[[i]], Y.iloc[[i]])
        pdt.assert_frame_equal(X_i, X_t.iloc[[i]], rtol=1e-12, atol=1e-12)
        pdt.assert_frame_equal(Y_i, Y_t.iloc[[i]], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("x_transform", "y_transform"), TRANSFORM_PAIRS)
def test_outputs_follow_labels_not_positions(
    data: dict, rng: np.random.Generator, x_transform: str, y_transform: str
) -> None:
    # Predictions and truth are joined by sample ID downstream, and each fitted mean/std belongs
    # to a named feature: reordering the input may only reorder the output rows.
    prep = fit_preprocessor(make_cfg(x_transform, y_transform), data["X_train"], data["Y_train"])
    X, Y = data["X_test"], data["Y_test"]
    X_t, Y_t = transform(prep, X, Y)
    assert X_t.index.equals(X.index) and Y_t.index.equals(Y.index)
    rows = rng.permutation(len(X))
    X_s, Y_s = transform(
        prep, X.iloc[rows, rng.permutation(X.shape[1])], Y.iloc[rows, rng.permutation(Y.shape[1])]
    )
    pdt.assert_frame_equal(X_s, X_t.iloc[rows], rtol=1e-12, atol=1e-12)
    pdt.assert_frame_equal(Y_s, Y_t.iloc[rows], rtol=1e-12, atol=1e-12)


def test_fitted_arrays_are_read_only_also_in_copies(data: dict) -> None:
    prep = fit_preprocessor(make_cfg(), data["X_train"], data["Y_train"])
    expected = transform(prep, data["X_test"], data["Y_test"])
    # Copies are what joblib workers receive (and from_dict is what load returns); they must
    # behave exactly like the original.
    restored = FittedPreprocessor.from_dict(prep.to_dict())
    for other in (prep, pickle.loads(pickle.dumps(prep)), copy.deepcopy(prep), restored):
        for array in (other.x.mean_, other.x.std_, other.y.mean_, other.y.std_):
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0.0
        got = transform(other, data["X_test"], data["Y_test"])
        for got_frame, want_frame in zip(got, expected, strict=True):
            pdt.assert_frame_equal(got_frame, want_frame, check_exact=True)


def test_fit_and_transform_never_mutate_inputs(data: dict, rng: np.random.Generator) -> None:
    originals = {key: frame.copy(deep=True) for key, frame in data.items()}
    for x_transform, y_transform in TRANSFORM_PAIRS:
        cfg = make_cfg(x_transform, y_transform)
        prep = fit_preprocessor(cfg, data["X_train"], data["Y_train"])
        transform(prep, data["X_test"], data["Y_test"])
        prep.y.inverse_standardize(prep.y.transform(data["Y_test"]))
    embeddings = pd.DataFrame(rng.normal(size=(20, 4)), index=data["X_test"].index)
    embeddings_before = embeddings.copy(deep=True)
    StandardScalerDF().fit(embeddings).transform(embeddings)

    for key, frame in data.items():
        pdt.assert_frame_equal(frame, originals[key], check_exact=True)
    pdt.assert_frame_equal(embeddings, embeddings_before, check_exact=True)


# --------------------------------------------------------------------------------------------
# Pseudocount rule
# --------------------------------------------------------------------------------------------


def test_half_min_nonzero_pseudocount_per_row_with_fallback() -> None:
    values = np.array([[0.0, 0.2, 0.4], [0.0, 0.0, 0.0], [0.1, 0.0, 0.3]])
    pseudocount = half_min_nonzero_pseudocount(values, fallback=0.01)
    np.testing.assert_allclose(pseudocount, [0.1, 0.01, 0.05])
    # No columns at all: every row falls back.
    np.testing.assert_array_equal(half_min_nonzero_pseudocount(np.zeros((2, 0)), 0.5), [0.5, 0.5])


@pytest.mark.parametrize(
    ("values", "fallback"),
    [(np.array([0.1, 0.2]), 0.1), (np.ones((2, 2)), 0.0), (np.ones((2, 2)), np.nan)],
)
def test_half_min_nonzero_pseudocount_rejects_bad_input(
    values: np.ndarray, fallback: float
) -> None:
    with pytest.raises(ValueError):
        half_min_nonzero_pseudocount(values, fallback)


def test_pseudocount_comes_from_kept_taxa_of_the_same_sample() -> None:
    # "trace" is absent from training, so it is not kept; its tiny test value must not become
    # the pseudocount. Expected CLR for kept values (a, 0) with c = a/2: ±ln(3)/2, any a.
    X_train = pd.DataFrame(
        {"a": [0.5, 0.3, 0.6, 0.2], "b": [0.5, 0.7, 0.4, 0.8], "trace": [0.0] * 4},
        index=["p", "q", "r", "s"],
    )
    X_test = pd.DataFrame({"a": [0.3], "b": [0.0], "trace": [1e-6]}, index=["t"])

    clr = MicrobePreprocessor(standardize=False).fit(X_train)
    assert clr.features_ == ["a", "b"]
    half_log3 = math.log(3) / 2
    np.testing.assert_allclose(clr.transform(X_test).loc["t"], [half_log3, -half_log3], rtol=1e-12)

    log = MicrobePreprocessor(transform="log", standardize=False).fit(X_train)
    a_rel = 0.3 / (0.3 + 1e-6)  # re-closed over all of the sample's columns
    expected = [math.log10(1.5 * a_rel), math.log10(0.5 * a_rel)]
    np.testing.assert_allclose(log.transform(X_test).loc["t"], expected, rtol=1e-12)


def test_fallback_pseudocount_covers_samples_without_kept_taxa() -> None:
    X_train = pd.DataFrame(
        {"a": [0.6, 0.2, 0.5], "b": [0.4, 0.8, 0.5], "trace": [0.0, 0.0, 1e-5]},
        index=["p", "q", "r"],
    )
    X_test = pd.DataFrame(
        {"a": [0.0, 0.0], "b": [0.0, 0.0], "trace": [0.3, 0.0]}, index=["only_trace", "empty"]
    )
    clr = MicrobePreprocessor(standardize=False).fit(X_train)
    assert clr.features_ == ["a", "b"]  # 1e-5 never exceeds the 1e-4 presence threshold
    # Half of the smallest positive *kept* training value (0.2), not of the trace (1e-5).
    assert clr.fallback_pseudocount_ == pytest.approx(0.1)
    np.testing.assert_allclose(clr.transform(X_test), 0.0, atol=1e-12)

    log = MicrobePreprocessor(transform="log", standardize=False).fit(X_train)
    np.testing.assert_allclose(log.transform(X_test), math.log10(0.1), rtol=1e-12)


# --------------------------------------------------------------------------------------------
# Microbes
# --------------------------------------------------------------------------------------------


def test_prevalence_filter_counts_training_rows_above_the_threshold() -> None:
    n = 20
    X = pd.DataFrame(
        {
            "two_of_twenty": [0.01] * 2 + [0.0] * (n - 2),  # 10% → kept (>= min_prevalence)
            "everywhere": [0.5] * n,
            "one_of_twenty": [0.01] + [0.0] * (n - 1),  # 5% → dropped
            "faint": [2e-5] * n,  # non-zero everywhere but never above 1e-4 → dropped
        }
    )
    assert MicrobePreprocessor().fit(X).features_ == ["two_of_twenty", "everywhere"]
    # Ablation A3: a lower prevalence threshold keeps the 5% taxon, still in input order.
    assert MicrobePreprocessor(min_prevalence=0.05).fit(X).features_ == [
        "two_of_twenty",
        "everywhere",
        "one_of_twenty",
    ]
    # Brief §4 says present *above* 1e-4: a value equal to the threshold is not presence.
    at_threshold = pd.DataFrame({"equal": [1e-4] * 10, "above": [1.0001e-4] * 10})
    assert prevalent_taxa(at_threshold, abundance_threshold=1e-4, min_prevalence=0.10) == [
        "above"
    ]


def test_filters_drop_the_rare_columns_of_the_fixture(data: dict) -> None:
    prep = fit_preprocessor(make_cfg(), data["X_train"], data["Y_train"])
    assert "rare_taxon" not in prep.x.features_
    assert "rare_taxon" in prep.x.input_features_
    assert "mostly_zero" not in prep.y.features_


def test_fixed_feature_list_is_used_in_its_order(data: dict) -> None:
    vocabulary = ["taxon_05", "not_in_training", "taxon_01"]
    prep = MicrobePreprocessor(features=vocabulary).fit(data["X_train"])
    assert prep.features_ == vocabulary
    assert list(prep.transform(data["X_test"]).columns) == vocabulary
    with pytest.raises(ValueError, match="none of the fixed"):
        MicrobePreprocessor(features=["nope"]).fit(data["X_train"])


def test_microbe_clr_rows_sum_to_zero_before_standardizing(data: dict) -> None:
    prep = MicrobePreprocessor(standardize=False).fit(data["X_train"])
    np.testing.assert_allclose(prep.transform(data["X_test"]).sum(axis=1), 0.0, atol=1e-10)


def test_microbe_clr_is_scale_invariant(data: dict, rng: np.random.Generator) -> None:
    prep = MicrobePreprocessor().fit(data["X_train"])
    X = data["X_test"]
    expected = prep.transform(X)
    scaled = X.mul(rng.uniform(1e-3, 1e3, size=len(X)), axis=0)
    # A new taxon changes every re-closed value of its sample by the same factor.
    diluted = X.assign(unseen_taxon=rng.uniform(0.0, 10.0, size=len(X)))
    for other in (scaled, diluted):
        np.testing.assert_allclose(prep.transform(other), expected, rtol=1e-9, atol=1e-9)


def test_clr_and_log_differ_only_by_per_sample_centring(data: dict) -> None:
    clr = MicrobePreprocessor(standardize=False).fit(data["X_train"]).transform(data["X_test"])
    log = (
        MicrobePreprocessor(transform="log", standardize=False)
        .fit(data["X_train"])
        .transform(data["X_test"])
    )
    centred = log.sub(log.mean(axis=1), axis=0) * math.log(10)  # CLR uses the natural log
    np.testing.assert_allclose(clr, centred, atol=1e-10)


@pytest.mark.parametrize("x_transform", ["clr", "log"])
def test_microbe_training_features_have_mean_zero_and_unit_std(
    data: dict, x_transform: str
) -> None:
    out = MicrobePreprocessor(transform=x_transform).fit_transform(data["X_train"])
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=0, ddof=0), 1.0, rtol=1e-12)


@pytest.mark.parametrize("x_transform", ["clr", "log"])
def test_missing_microbe_columns_count_as_absent(data: dict, x_transform: str) -> None:
    prep = MicrobePreprocessor(transform=x_transform).fit(data["X_train"])
    X = data["X_test"]
    pdt.assert_frame_equal(
        prep.transform(X.drop(columns=["taxon_03"])),
        prep.transform(X.assign(taxon_03=0.0)),
        check_exact=True,
    )


def test_microbe_transform_rejects_unrelated_or_invalid_tables(data: dict) -> None:
    prep = MicrobePreprocessor().fit(data["X_train"])
    other_table = pd.DataFrame({"g__Unrelated": [0.4, 0.6]}, index=["u", "v"])
    with pytest.raises(ValueError, match="shares no column"):
        prep.transform(other_table)
    with pytest.raises(ValueError, match="negative"):
        prep.transform(data["X_test"] - 1.0)
    duplicated = pd.concat([data["X_test"], data["X_test"][["taxon_00"]]], axis=1)
    with pytest.raises(ValueError, match="duplicated"):
        prep.transform(duplicated)


# --------------------------------------------------------------------------------------------
# Metabolites
# --------------------------------------------------------------------------------------------


def zero_filter_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "half_zero": [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],  # 50% zeros → kept
            "most_zero": [0, 0, 0, 0, 0, 0, 2, 3, 4, 5],  # 60% zeros → dropped
            "no_zero": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        dtype=float,
    )


def test_zero_filter_keeps_exactly_half_zero_and_drops_more() -> None:
    prep = MetabolitePreprocessor().fit(zero_filter_table())
    assert prep.retained_ == prep.features_ == ["half_zero", "no_zero"]
    assert (prep.n_candidates_, prep.n_dropped_zero_) == (3, 1)


def test_candidate_features_keep_their_order() -> None:
    candidates = ["no_zero", "most_zero", "half_zero"]
    prep = MetabolitePreprocessor(candidate_features=candidates).fit(zero_filter_table())
    assert prep.features_ == ["no_zero", "half_zero"]
    assert prep.retained_ == ["half_zero", "no_zero"]  # input column order
    assert (prep.n_candidates_, prep.n_dropped_zero_) == (3, 1)
    assert list(prep.transform(zero_filter_table()).columns) == ["no_zero", "half_zero"]


def test_candidates_missing_from_training_data_are_an_error() -> None:
    with pytest.raises(KeyError, match="not_measured"):
        MetabolitePreprocessor(candidate_features=["no_zero", "not_measured"]).fit(
            zero_filter_table()
        )


def test_log10_targets_match_the_formula(data: dict) -> None:
    Y_train, Y_test = data["Y_train"], data["Y_test"]
    raw = MetabolitePreprocessor(standardize=False).fit(Y_train)
    features = raw.features_
    np.testing.assert_allclose(raw.transform(Y_test), np.log10(Y_test[features] + 1.0))

    z = MetabolitePreprocessor().fit(Y_train).transform(Y_test)
    log_train = np.log10(Y_train[features] + 1.0)
    expected = (np.log10(Y_test[features] + 1.0) - log_train.mean()) / log_train.std(ddof=0)
    np.testing.assert_allclose(z, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("y_transform", ["log10", "clr"])
def test_metabolite_training_targets_have_mean_zero_and_unit_std(
    data: dict, y_transform: str
) -> None:
    out = MetabolitePreprocessor(transform=y_transform).fit_transform(data["Y_train"])
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(out.std(axis=0, ddof=0), 1.0, rtol=1e-12)


def test_constant_training_feature_maps_to_zero() -> None:
    Y = pd.DataFrame({"constant": [7.0] * 6, "varying": [1.0, 2, 3, 4, 5, 6]})
    prep = MetabolitePreprocessor().fit(Y)
    assert prep.std_[0] == 1.0
    np.testing.assert_allclose(prep.transform(Y)["constant"], 0.0, atol=1e-12)


def test_metabolite_clr_rows_sum_to_zero_and_ignore_scale(
    data: dict, rng: np.random.Generator
) -> None:
    prep = MetabolitePreprocessor(transform="clr", standardize=False).fit(data["Y_train"])
    assert prep.clr_reference_ == prep.retained_ == prep.features_
    Y = data["Y_test"]
    out = prep.transform(Y)
    np.testing.assert_allclose(out.sum(axis=1), 0.0, atol=1e-10)
    scaled = Y.mul(rng.uniform(1e-3, 1e3, size=len(Y)), axis=0)
    np.testing.assert_allclose(prep.transform(scaled), out, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("standardize", [False, True])
def test_target_sets_share_one_clr_reference(data: dict, standardize: bool) -> None:
    Y_train, Y_test = data["Y_train"], data["Y_test"]
    subset = list(reversed(Y_train.columns[:15]))  # a "T466" inside "TALL", in its own order
    tall = MetabolitePreprocessor(transform="clr", standardize=standardize).fit(Y_train)
    t466 = MetabolitePreprocessor(
        transform="clr", standardize=standardize, candidate_features=subset
    ).fit(Y_train)
    assert t466.clr_reference_ == tall.clr_reference_ == tall.retained_
    assert len(t466.features_) < len(tall.features_)
    np.testing.assert_allclose(
        t466.transform(Y_test), tall.transform(Y_test)[t466.features_], rtol=1e-12, atol=1e-12
    )


def test_missing_targets_raise_key_error_naming_them(data: dict) -> None:
    Y_train, Y_test = data["Y_train"], data["Y_test"]
    log10 = MetabolitePreprocessor(candidate_features=["met_00", "met_01"]).fit(Y_train)
    with pytest.raises(KeyError, match="met_01"):
        log10.transform(Y_test.drop(columns=["met_01"]))
    log10.transform(Y_test.drop(columns=["met_05"]))  # not a target: not needed

    # For CLR the whole reference is needed, including columns that are not targets.
    clr = MetabolitePreprocessor(transform="clr", candidate_features=["met_00"]).fit(Y_train)
    with pytest.raises(KeyError, match="met_05"):
        clr.transform(Y_test.drop(columns=["met_05"]))


def test_metabolite_nan_is_rejected_where_it_matters(data: dict) -> None:
    Y_nan = data["Y_train"].copy()
    Y_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        MetabolitePreprocessor().fit(Y_nan)

    prep = MetabolitePreprocessor(candidate_features=["met_01", "met_02"]).fit(data["Y_train"])
    Y_test = data["Y_test"].copy()
    Y_test.loc[Y_test.index[0], "met_10"] = np.nan  # an unused column
    prep.transform(Y_test)
    Y_test.loc[Y_test.index[0], "met_01"] = np.nan  # a target
    with pytest.raises(ValueError, match="NaN"):
        prep.transform(Y_test)


@pytest.mark.parametrize("y_transform", ["log10", "clr"])
def test_inverse_standardize_undoes_the_z_score(data: dict, y_transform: str) -> None:
    scaled = MetabolitePreprocessor(transform=y_transform).fit(data["Y_train"])
    unscaled = MetabolitePreprocessor(transform=y_transform, standardize=False).fit(
        data["Y_train"]
    )
    z = scaled.transform(data["Y_test"])
    expected = unscaled.transform(data["Y_test"])
    pdt.assert_frame_equal(scaled.inverse_standardize(z), expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(scaled.inverse_standardize(z.to_numpy()), expected, atol=1e-12)
    with pytest.raises(ValueError, match="shape"):
        scaled.inverse_standardize(z.to_numpy()[:, :2])


# --------------------------------------------------------------------------------------------
# StandardScalerDF
# --------------------------------------------------------------------------------------------


def test_standard_scaler_df_scales_by_column_name(rng: np.random.Generator) -> None:
    values = rng.normal(3.0, 2.0, size=(30, 5))
    values[:, 4] = 1.5  # a constant embedding dimension
    E = pd.DataFrame(values, index=[f"s{i}" for i in range(30)])  # integer column labels
    scaler = StandardScalerDF().fit(E.iloc[:20])
    train = scaler.transform(E.iloc[:20])
    np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(train.std(axis=0, ddof=0).iloc[:4], 1.0, rtol=1e-12)
    np.testing.assert_allclose(train[4], 0.0, atol=1e-12)
    test = E.iloc[20:]
    # Test rows use the training mean/std, never their own (one row is scaled as in a batch).
    varying = E.iloc[:20, :4]
    expected = (test.iloc[:, :4] - varying.mean()) / varying.std(ddof=0)
    pdt.assert_frame_equal(scaler.transform(test).iloc[:, :4], expected, rtol=1e-12, atol=1e-12)
    pdt.assert_frame_equal(scaler.transform(test.iloc[[3]]), scaler.transform(test).iloc[[3]])
    pdt.assert_frame_equal(scaler.transform(test.iloc[:, ::-1]), scaler.transform(test))
    with pytest.raises(KeyError):
        scaler.transform(test.drop(columns=[2]))


# --------------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------------


def _embeddings(data: dict) -> pd.DataFrame:
    X = pd.concat([data["X_train"], data["X_test"]])
    return pd.DataFrame(
        np.log(X.to_numpy()[:, :6] + 1e-3), index=X.index, columns=[f"emb_{i}" for i in range(6)]
    )


FACTORIES: dict[str, Callable[[dict], tuple]] = {
    "microbe clr": lambda d: (MicrobePreprocessor().fit(d["X_train"]), d["X_test"]),
    "microbe log unscaled": lambda d: (
        MicrobePreprocessor(transform="log", standardize=False).fit(d["X_train"]),
        d["X_test"],
    ),
    "microbe vocabulary": lambda d: (
        MicrobePreprocessor(features=["taxon_09", "absent", "taxon_02"]).fit(d["X_train"]),
        d["X_test"],
    ),
    "metabolite log10": lambda d: (MetabolitePreprocessor().fit(d["Y_train"]), d["Y_test"]),
    "metabolite clr subset": lambda d: (
        MetabolitePreprocessor(
            transform="clr", candidate_features=list(d["Y_train"].columns[::-3])
        ).fit(d["Y_train"]),
        d["Y_test"],
    ),
    "embedding scaler": lambda d: (
        StandardScalerDF().fit(_embeddings(d).iloc[:N_TRAIN]),
        _embeddings(d).iloc[N_TRAIN:],
    ),
}


@pytest.mark.parametrize("name", list(FACTORIES))
def test_round_trips_reproduce_transforms_exactly(data: dict, tmp_path: Path, name: str) -> None:
    prep, test_input = FACTORIES[name](data)
    payload = prep.to_dict()
    text = json.dumps(payload, allow_nan=False)  # JSON-serialisable as is
    expected = prep.transform(test_input)
    restored = [
        type(prep).from_dict(json.loads(text)),
        type(prep).load(prep.save(tmp_path / "p.json")),
    ]
    for other in restored:
        assert other.to_dict() == payload
        pdt.assert_frame_equal(other.transform(test_input), expected, check_exact=True)


@pytest.mark.parametrize("x_transform", ["clr", "log"])
def test_fitted_preprocessor_round_trip(data: dict, tmp_path: Path, x_transform: str) -> None:
    prep = fit_preprocessor(make_cfg(x_transform, "clr"), data["X_train"], data["Y_train"])
    loaded = FittedPreprocessor.load(prep.save(tmp_path / "nested" / "prep.json"))
    assert loaded.to_dict() == prep.to_dict()
    for got, want in zip(
        transform(loaded, data["X_test"], data["Y_test"]),
        transform(prep, data["X_test"], data["Y_test"]),
        strict=True,
    ):
        pdt.assert_frame_equal(got, want, check_exact=True)


def test_fitted_preprocessor_accepts_an_embedding_scaler(data: dict) -> None:
    E = _embeddings(data)
    prep = FittedPreprocessor(
        x=StandardScalerDF().fit(E.iloc[:N_TRAIN]),
        y=MetabolitePreprocessor().fit(data["Y_train"]),
    )
    restored = FittedPreprocessor.from_dict(json.loads(json.dumps(prep.to_dict())))
    assert isinstance(restored.x, StandardScalerDF)
    X_t, Y_t = transform(restored, E.iloc[N_TRAIN:], data["Y_test"])
    assert X_t.shape == (20, 6) and Y_t.shape[0] == 20


def test_unfitted_objects_round_trip_their_parameters() -> None:
    for prep in (
        MicrobePreprocessor(min_prevalence=0.2, transform="log", features=["a", "b"]),
        MetabolitePreprocessor(transform="clr", max_zero_fraction=0.3, candidate_features=["m"]),
        StandardScalerDF(),
    ):
        restored = type(prep).from_dict(json.loads(json.dumps(prep.to_dict())))
        assert restored.get_params() == prep.get_params()
        assert not restored.is_fitted


def test_loading_another_kind_is_rejected(data: dict) -> None:
    payload = MicrobePreprocessor().fit(data["X_train"]).to_dict()
    with pytest.raises(ValueError, match="MetabolitePreprocessor"):
        MetabolitePreprocessor.from_dict(payload)
    with pytest.raises(ValueError, match="format_version"):
        MicrobePreprocessor.from_dict({**payload, "format_version": 999})


# --------------------------------------------------------------------------------------------
# fit_preprocessor / transform / configuration
# --------------------------------------------------------------------------------------------


def test_fit_preprocessor_uses_config_sections_and_feature_lists(data: dict) -> None:
    cfg = make_cfg("log", "clr")
    cfg["microbes"]["min_prevalence"] = 0.2
    targets = list(data["Y_train"].columns[:12]) + ["mostly_zero"]
    vocabulary = list(data["X_train"].columns[:7])
    prep = fit_preprocessor(
        cfg, data["X_train"], data["Y_train"], target_features=targets, x_features=vocabulary
    )
    assert prep.x.get_params()["min_prevalence"] == 0.2
    assert (prep.x.transform_type, prep.y.transform_type) == ("log", "clr")
    assert prep.x.features_ == vocabulary
    assert prep.y.features_ == targets[:12]  # mostly_zero fails the zero filter
    assert prep.summary() == {
        "n_input_features": 7,
        "n_target_candidates": 13,
        "n_targets": 12,
        "n_targets_dropped_zero": 1,
    }
    X_t, Y_t = transform(prep, data["X_test"])
    assert Y_t is None and X_t.shape == (20, 7)


def test_misaligned_microbes_and_metabolites_are_rejected(data: dict) -> None:
    with pytest.raises(ValueError, match="same samples"):
        fit_preprocessor(make_cfg(), data["X_train"], data["Y_train"].iloc[::-1])
    prep = fit_preprocessor(make_cfg(), data["X_train"], data["Y_train"])
    with pytest.raises(ValueError, match="same samples"):
        transform(prep, data["X_test"], data["Y_train"])


def test_from_config_accepts_the_base_config() -> None:
    cfg = load_config(paths.CONFIGS / "base.yaml")
    x = MicrobePreprocessor.from_config(cfg["microbes"])
    y = MetabolitePreprocessor.from_config(cfg["metabolites"])
    assert x.get_params() == {**cfg["microbes"], "features": None}
    assert y.get_params() == {**cfg["metabolites"], "candidate_features": None}


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="min_prevalance"):
        MicrobePreprocessor.from_config({"min_prevalance": 0.05})  # misspelt
    with pytest.raises(ValueError, match="transform"):
        MicrobePreprocessor(transform="ilr")
    with pytest.raises(ValueError, match="transform"):
        MetabolitePreprocessor(transform="log2")
    with pytest.raises(ValueError, match="max_zero_fraction"):
        MetabolitePreprocessor(max_zero_fraction=1.5)
    with pytest.raises(ValueError, match="abundance_threshold"):
        MicrobePreprocessor(abundance_threshold=-1.0)
    with pytest.raises(TypeError, match="standardize"):
        MicrobePreprocessor(standardize="yes")
    with pytest.raises(TypeError, match="sequence of names"):
        MicrobePreprocessor(features="taxon_01")
    with pytest.raises(ValueError, match="duplicated"):
        MetabolitePreprocessor(candidate_features=["m", "m"])


@pytest.mark.parametrize("make", [MicrobePreprocessor, MetabolitePreprocessor, StandardScalerDF])
def test_transform_before_fit_is_an_error(make: Callable, data: dict) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        make().transform(data["X_test"])


def test_clone_is_unfitted_with_the_same_parameters(data: dict) -> None:
    prep = MicrobePreprocessor(transform="log", min_prevalence=0.3).fit(data["X_train"])
    copy_ = prep.clone()
    assert copy_.get_params() == prep.get_params() and not copy_.is_fitted
    assert repr(copy_).endswith("[unfitted]")
    assert repr(prep).endswith(f"[fitted, {len(prep.features_)} features]")


def test_fixed_pseudocount_reproduces_published_clr_plus_one() -> None:
    """mNODE/MiMeNet use CLR(x + 1) over every feature (docs/published_numbers.md)."""
    Y = pd.DataFrame(
        [[0.0, 9.0, 99.0], [4.0, 0.0, 1.0]], index=["a", "b"], columns=["m1", "m2", "m3"]
    )
    prep = MetabolitePreprocessor(
        transform="clr", max_zero_fraction=1.0, standardize=False, pseudocount=1.0
    ).fit(Y)
    logged = np.log(Y.to_numpy() + 1.0)
    expected = logged - logged.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(prep.transform(Y).to_numpy(), expected)
    assert MetabolitePreprocessor.from_dict(prep.to_dict()).pseudocount == 1.0
    with pytest.raises(ValueError, match="pseudocount"):
        MetabolitePreprocessor(pseudocount=0.0)
    with pytest.raises(ValueError, match="pseudocount"):
        MetabolitePreprocessor(pseudocount="one")
