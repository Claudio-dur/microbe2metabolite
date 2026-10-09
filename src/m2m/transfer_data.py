"""E4 data assembly (no training): source vocabulary, pretraining table, source/target checks and
the shared-HMDB targets.

Leakage rules implemented here (brief sections 3 and 5; DESIGN.md decisions 9-12):
- the vocabulary and the pretraining table come from the five deduplicated *source* datasets only;
  the target cohort (Franzosa) appears only as the other side of the disjointness checks;
- every choice made on targets (the representative LC-MS feature of each HMDB ID, the zero filter,
  the z-scores) is fit on the training rows passed to ``fit``, never on the rows it is applied to.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from m2m import data, preprocess, splits
from m2m.preprocess import MetabolitePreprocessor

IHMP = "iHMP_IBDMDB_2019"
SUBJECT_KEY_SEPARATOR = ":"
# An all-zero profile has CLR 0 whatever its pseudocount, so any positive value works here.
EMPTY_PROFILE_PSEUDOCOUNT = 1.0
# How many of the most-missing iHMP compounds the NaN evidence lists by name.
N_LISTED_COMPOUNDS = 8
# iHMP leaves non-detected compounds empty (NaN) where Franzosa writes 0; see ihmp_missing_evidence.
IHMP_MISSING_POLICY = "not_detected_as_zero"
IHMP_MISSING_RATIONALE = (
    "iHMP's NaN are non-detects, so they become 0 (Franzosa's encoding of a non-detect on the same "
    "Broad platform) and the usual zero filter then applies: no sample lacks a whole LC-MS method "
    "(every NaN is scattered), the table holds no exact zero at all, compounds with more NaN have "
    "lower detected intensities, most of the most-missing compounds are drugs that most people do "
    "not take (furosemide, hydrochlorothiazide, sulfapyridine, gabapentin), and partly-missing "
    "compounds show a longer low tail of detected values (left-censoring at a detection limit). "
    "Masking NaN out of the loss would instead treat absence as unknown."
)


# --------------------------------------------------------------------------------------------
# Vocabulary and pretraining table (sources only)
# --------------------------------------------------------------------------------------------


def prevalent_genera_by_source(
    sources: Mapping[str, data.Cohort], cfg: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Each source's prevalent genera (base.yaml `microbes` thresholds) on its own rows."""
    _require_sources_only(sources)
    microbes = cfg["microbes"]
    return {
        name: preprocess.prevalent_taxa(
            preprocess.closure(cohort.microbes),
            float(microbes["abundance_threshold"]),
            float(microbes["min_prevalence"]),
        )
        for name, cohort in sources.items()
    }


def source_vocabulary(sources: Mapping[str, data.Cohort], cfg: Mapping[str, Any]) -> list[str]:
    """E4 input genera: union of every source's prevalent genera, sorted (DESIGN.md #10).

    A union of per-source sets (rather than prevalence over the pooled sources) keeps genera that
    are characteristic of a single cohort. Sorting makes the order, which is the input order of
    every pretrained network, independent of dataset order.
    """
    vocabulary = sorted(set().union(*prevalent_genera_by_source(sources, cfg).values()))
    if not vocabulary:
        raise ValueError("no genus is prevalent in any source dataset")
    return vocabulary


def vocabulary_digest(vocabulary: Sequence[str]) -> str:
    """Short fingerprint of the vocabulary *and its order*, stored with every pretrained network."""
    return hashlib.sha1("\n".join(vocabulary).encode()).hexdigest()[:16]


def subject_keys(cohort: data.Cohort) -> pd.Series:
    """'<dataset>:<subject>' per sample: unique across datasets, whose subject IDs could collide."""
    meta = cohort.meta
    keys = meta["dataset"].astype(str) + SUBJECT_KEY_SEPARATOR + meta["subject"].astype(str)
    return keys.rename("subject_key")


def pretraining_table(sources: Mapping[str, data.Cohort]) -> tuple[pd.DataFrame, pd.Series]:
    """Every source sample x the union of the sources' genus columns, with its subject key.

    Values are the published relative abundances; a genus that a dataset does not report is
    absent there (0). Columns are sorted so the table does not depend on dataset order.
    """
    _require_sources_only(sources)
    table = pd.concat([cohort.microbes for cohort in sources.values()], axis=0, join="outer")
    if not table.index.is_unique:
        duplicated = table.index[table.index.duplicated()].unique().tolist()
        raise ValueError(f"sample IDs collide across source datasets, e.g. {duplicated[:5]}")
    table = table.fillna(0.0)
    table = table.reindex(columns=sorted(table.columns))
    keys = pd.concat([subject_keys(cohort) for cohort in sources.values()])
    return table, keys.loc[table.index]


def vocabulary_coverage(
    vocabulary: Sequence[str], target_microbes: pd.DataFrame, cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """How much of a target cohort the source vocabulary covers (descriptive only).

    Nothing here feeds back into the vocabulary: it is reported, never used for fitting.
    """
    microbes = cfg["microbes"]
    relative = preprocess.closure(target_microbes)
    prevalent = preprocess.prevalent_taxa(
        relative, float(microbes["abundance_threshold"]), float(microbes["min_prevalence"])
    )
    in_vocabulary = set(vocabulary)
    missing = [genus for genus in prevalent if genus not in in_vocabulary]
    share = relative.reindex(columns=list(vocabulary), fill_value=0.0).sum(axis=1)
    return {
        "n_samples": int(len(target_microbes)),
        "target_prevalent_genera": len(prevalent),
        "covered": len(prevalent) - len(missing),
        "covered_fraction": (len(prevalent) - len(missing)) / len(prevalent) if prevalent else 0.0,
        "missing_genera": [data.gtdb_genus(genus) for genus in missing],
        "abundance_share_in_vocabulary": {
            "mean": float(share.mean()),
            "median": float(share.median()),
            "min": float(share.min()),
        },
    }


def _require_sources_only(sources: Mapping[str, data.Cohort]) -> None:
    if not sources:
        raise ValueError("no source dataset given")
    unknown = sorted(set(sources) - set(data.SOURCE_DATASETS))
    if unknown:
        raise ValueError(
            f"{unknown} are not source datasets: pretraining data must never include the target "
            f"{data.TARGET_DATASET} (brief section 5)"
        )


# --------------------------------------------------------------------------------------------
# Source / target disjointness (brief section 5)
# --------------------------------------------------------------------------------------------


def vocabulary_clr(microbes: pd.DataFrame, vocabulary: Sequence[str]) -> np.ndarray:
    """CLR profile of each sample over the vocabulary (per-sample pseudocount; no fitting)."""
    relative = preprocess.closure(microbes).reindex(columns=list(vocabulary), fill_value=0.0)
    values = relative.to_numpy(dtype=float)
    pseudocount = preprocess.half_min_nonzero_pseudocount(values, EMPTY_PROFILE_PSEUDOCOUNT)
    return preprocess.centred_log_ratio(values, pseudocount)


def max_profile_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Largest Pearson correlation between a row of ``a`` and a row of ``b``, and its rows."""
    correlation = _unit_rows(a) @ _unit_rows(b).T
    i, j = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    return float(correlation[i, j]), int(i), int(j)


def check_source_target_disjoint(
    sources: Mapping[str, data.Cohort],
    target_meta: pd.DataFrame,
    target_microbes: pd.DataFrame,
    vocabulary: Sequence[str],
    *,
    max_correlation: float,
    target_name: str,
) -> dict[str, Any]:
    """Raise LeakageError if the target shares an ID or a near-identical profile with the sources.

    IDs: no source sample or subject ID may equal a target sample or subject ID (compared as one
    pool, so a sample ID reused as a subject ID is caught too). Profiles: the largest Pearson
    correlation between a source and a target vocabulary-CLR profile must not exceed
    ``max_correlation``; a re-published metagenome correlates at 1.0 (the ERAWIJANTARI copies of
    YACHIDA samples do) whatever its sample ID. Returns the numbers for the report.
    """
    _require_sources_only(sources)
    source_ids = set().union(
        *(set(c.meta.index) | set(c.meta["subject"].astype(str)) for c in sources.values())
    )
    target_ids = set(target_meta.index) | set(target_meta["subject"].astype(str))
    splits.assert_disjoint(
        source_ids, target_ids, what=f"pretraining sources vs {target_name} (sample/subject IDs)"
    )
    unchecked = target_microbes.index.difference(target_meta.index)
    if len(unchecked):
        raise ValueError(f"{len(unchecked)} {target_name} profiles have no metadata row to check")
    source_rows = [sample for cohort in sources.values() for sample in cohort.sample_ids]
    source_profiles = np.vstack([vocabulary_clr(c.microbes, vocabulary) for c in sources.values()])
    target_profiles = vocabulary_clr(target_microbes, vocabulary)
    best, i, j = max_profile_correlation(source_profiles, target_profiles)
    if best > max_correlation:
        raise splits.LeakageError(
            f"source sample {source_rows[i]} and {target_name} sample {target_microbes.index[j]} "
            f"have near-identical genus profiles (Pearson r = {best:.4f} > {max_correlation}): "
            "the same metagenome may be in both"
        )
    return {
        "target": target_name,
        "n_source_samples": len(source_rows),
        "n_target_samples": int(len(target_microbes)),
        "shared_sample_or_subject_ids": 0,
        "max_profile_correlation": best,
        "max_correlation_pair": [str(source_rows[i]), str(target_microbes.index[j])],
        "max_correlation_threshold": float(max_correlation),
        # Reference: the most similar two *different people* of the target cohort.
        "max_correlation_within_target": _max_offdiagonal_correlation(target_profiles),
    }


def _unit_rows(values: np.ndarray) -> np.ndarray:
    centred = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    # A flat profile has no defined correlation; as a zero row it correlates 0 with everything.
    return centred / np.where(norms > 0, norms, 1.0)


def _max_offdiagonal_correlation(profiles: np.ndarray) -> float:
    if len(profiles) < 2:
        return float("nan")
    unit = _unit_rows(profiles)
    correlation = unit @ unit.T
    np.fill_diagonal(correlation, -np.inf)
    return float(correlation.max())


# --------------------------------------------------------------------------------------------
# Shared HMDB targets (iHMP <-> Franzosa)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedTargets:
    """HMDB IDs measured in both iHMP and Franzosa's T466, and the LC-MS features measuring them.

    ``hmdb_order`` (sorted) is the column order of every S176 target table and the row order of
    every pretrained E4b head, which is what lets a head be carried over row by row.
    """

    hmdb_order: tuple[str, ...]
    franzosa: dict[str, str]  # Franzosa feature ID (D1 naming, as in T466) -> HMDB ID
    ihmp: dict[str, str]  # iHMP compound (mtb.tsv column) -> HMDB ID

    def summary(self) -> dict[str, Any]:
        def per_hmdb(mapping: Mapping[str, str]) -> dict[str, int]:
            counts = pd.Series(list(mapping.values())).value_counts().value_counts().sort_index()
            return {f"{n}_features": int(k) for n, k in counts.items()}

        return {
            "n_hmdb": len(self.hmdb_order),
            "franzosa_features": len(self.franzosa),
            "ihmp_features": len(self.ihmp),
            "franzosa_hmdb_by_feature_count": per_hmdb(self.franzosa),
            "ihmp_hmdb_by_feature_count": per_hmdb(self.ihmp),
        }


def shared_hmdb_targets(
    franzosa_map: pd.DataFrame, ihmp_map: pd.DataFrame, t466: Sequence[str]
) -> SharedTargets:
    """HMDB IDs shared by iHMP and Franzosa, the Franzosa side restricted to T466 features.

    ``*_map`` are mtb.map.tsv tables indexed by compound (data.load_d2_mtb_map). Franzosa
    compounds are renamed to their D1 feature IDs, the names its metabolite table uses.
    """
    in_t466 = set(t466)
    franzosa_hmdb = franzosa_map["HMDB"].dropna()
    franzosa = {
        data.franzosa_feature_id(compound): str(hmdb)
        for compound, hmdb in franzosa_hmdb.items()
        if data.franzosa_feature_id(compound) in in_t466
    }
    ihmp_all = {str(compound): str(hmdb) for compound, hmdb in ihmp_map["HMDB"].dropna().items()}
    shared = sorted(set(franzosa.values()) & set(ihmp_all.values()))
    if not shared:
        raise ValueError("iHMP and Franzosa's T466 share no HMDB ID")
    keep = set(shared)
    return SharedTargets(
        hmdb_order=tuple(shared),
        franzosa={feature: hmdb for feature, hmdb in franzosa.items() if hmdb in keep},
        ihmp={compound: hmdb for compound, hmdb in ihmp_all.items() if hmdb in keep},
    )


def ihmp_missing_evidence(
    metabolites: pd.DataFrame, mtb_map: pd.DataFrame, shared_compounds: Sequence[str]
) -> dict[str, Any]:
    """What iHMP's NaN are: whole LC-MS methods missing per sample, or scattered non-detects.

    ``mtb_map`` must have the ``Method`` column of iHMP's mtb.map.tsv. Returns the evidence behind
    IHMP_MISSING_POLICY, written to results/e4_sources.json.
    """
    missing = metabolites.isna()
    method = mtb_map["Method"].reindex(metabolites.columns)
    if method.isna().any():
        raise ValueError("every iHMP compound needs its LC-MS method in mtb.map.tsv")
    per_method, whole_method_cells = {}, 0
    for name in sorted(method.unique()):
        block = missing[method.index[method == name]]
        whole = block.all(axis=1)
        whole_method_cells += int(block.loc[whole].to_numpy().sum())
        per_method[str(name)] = {
            "n_features": int(block.shape[1]),
            "missing_fraction": float(block.to_numpy().mean()),
            "samples_missing_the_whole_method": int(whole.sum()),
        }
    n_missing = int(missing.to_numpy().sum())
    missing_fraction = missing.mean(axis=0)
    detected_median = metabolites.median(axis=0, skipna=True)
    measured = detected_median.notna() & (detected_median > 0)
    rho = spearmanr(missing_fraction[measured], np.log10(detected_median[measured])).statistic
    return {
        "n_samples": int(metabolites.shape[0]),
        "n_features": int(metabolites.shape[1]),
        "missing_fraction": float(missing.to_numpy().mean()),
        "n_missing": n_missing,
        "n_missing_in_whole_missing_methods": whole_method_cells,
        "n_missing_scattered": n_missing - whole_method_cells,
        "per_method": per_method,
        "n_exact_zeros": int((metabolites == 0).to_numpy().sum()),
        "min_detected_value": float(np.nanmin(metabolites.to_numpy(dtype=float))),
        "spearman_missing_fraction_vs_log10_median_detected": float(rho),
        "left_tail_log10_q05_over_median": _left_tail_by_missingness(metabolites, missing_fraction),
        "most_missing_compounds": {
            str(name): float(value)
            for name, value in missing_fraction.sort_values(ascending=False)
            .head(N_LISTED_COMPOUNDS)
            .items()
        },
        "missing_fraction_shared_hmdb_features": float(
            missing[list(shared_compounds)].to_numpy().mean()
        ),
        "policy": IHMP_MISSING_POLICY,
        "rationale": IHMP_MISSING_RATIONALE,
    }


def _left_tail_by_missingness(
    metabolites: pd.DataFrame, missing_fraction: pd.Series
) -> dict[str, float]:
    """Median over compounds of log10(5th percentile / median) of the *detected* values, for
    compounds that are never missing vs partly missing: a detection limit censors the low tail."""

    def tail(columns: pd.Index) -> float:
        if len(columns) == 0:
            return float("nan")
        detected = metabolites[columns]
        ratio = np.log10(detected.quantile(0.05) / detected.median())
        return float(ratio.median())

    never = missing_fraction.index[missing_fraction == 0]
    partly = missing_fraction.index[(missing_fraction > 0) & (missing_fraction < 1)]
    return {"never_missing": tail(never), "partly_missing": tail(partly)}


def fill_not_detected(metabolites: pd.DataFrame) -> pd.DataFrame:
    """Apply IHMP_MISSING_POLICY: a missing iHMP intensity is a non-detect, i.e. 0.

    A fixed per-cell rule (nothing is fitted), so applying it to all rows at once cannot leak.
    """
    return metabolites.fillna(0.0)


def representative_features(
    Y_train: pd.DataFrame, feature_to_hmdb: Mapping[str, str]
) -> dict[str, str]:
    """One LC-MS feature per HMDB ID, chosen on training rows only (DESIGN.md #12).

    The feature detected in the most training rows wins (lowest fraction of zeros or NaN); ties go
    to the highest median intensity (NaN counted as 0), then to the feature name, so the choice
    never depends on column order. Summing features instead would add intensities from LC-MS
    methods that are not on a common scale.
    """
    features = [feature for feature in Y_train.columns if feature in feature_to_hmdb]
    if not features:
        raise ValueError("none of the mapped features is a column of Y_train")
    values = Y_train[features]
    candidates = pd.DataFrame(
        {
            "feature": features,
            "hmdb": [feature_to_hmdb[feature] for feature in features],
            "not_detected": (values.isna() | (values == 0)).mean(axis=0).to_numpy(),
            "median": values.fillna(0.0).median(axis=0).to_numpy(),
        }
    )
    best = candidates.sort_values(
        ["hmdb", "not_detected", "median", "feature"], ascending=[True, True, False, True]
    ).drop_duplicates("hmdb", keep="first")
    chosen = dict(zip(best["hmdb"], best["feature"], strict=True))
    absent = sorted(set(feature_to_hmdb.values()) - set(chosen))
    if absent:
        raise KeyError(f"no feature column for {len(absent)} HMDB IDs, e.g. {absent[:5]}")
    return chosen


class HmdbTargetPreprocessor:
    """Targets named by HMDB ID (the S176 set): fit(train) -> transform(any), like
    MetabolitePreprocessor.

    fit: choose each HMDB ID's representative feature on the training rows, rename it to its
    HMDB ID, then MetabolitePreprocessor (log10(x + 1), zero filter, z-score) on the same rows.
    Columns follow ``hmdb_order`` minus the targets the zero filter drops, so a pretrained head
    can be carried over row by row. ``n_mostly_zero_`` counts the targets zero in more than
    ``mostly_zero_fraction`` of training rows, whether or not the zero filter drops them (E4b keeps
    them all, so that its head has a row for every shared HMDB ID).
    """

    def __init__(
        self,
        *,
        feature_to_hmdb: Mapping[str, str],
        hmdb_order: Sequence[str],
        metabolites: Mapping[str, Any],
        mostly_zero_fraction: float,
    ) -> None:
        self.hmdb_order = [str(hmdb) for hmdb in hmdb_order]
        wanted = set(self.hmdb_order)
        self.feature_to_hmdb = {f: h for f, h in feature_to_hmdb.items() if h in wanted}
        self.metabolites = dict(metabolites)
        self.mostly_zero_fraction = float(mostly_zero_fraction)
        self.representatives_: dict[str, str] | None = None
        self.n_mostly_zero_: int | None = None
        self.inner_: MetabolitePreprocessor | None = None

    def fit(self, Y: pd.DataFrame) -> Self:
        representatives = representative_features(Y, self.feature_to_hmdb)
        unmapped = [hmdb for hmdb in self.hmdb_order if hmdb not in representatives]
        if unmapped:
            raise KeyError(f"no feature measures {len(unmapped)} HMDB IDs, e.g. {unmapped[:5]}")
        by_hmdb = self._by_hmdb(Y, representatives)
        inner = MetabolitePreprocessor.from_config(
            self.metabolites, candidate_features=self.hmdb_order
        ).fit(by_hmdb)
        zero_fraction = (by_hmdb == 0).mean(axis=0)
        # Assigned last, so a failed fit never leaves a half-fitted object behind.
        self.representatives_ = representatives
        self.n_mostly_zero_ = int((zero_fraction > self.mostly_zero_fraction).sum())
        self.inner_ = inner
        return self

    def transform(self, Y: pd.DataFrame) -> pd.DataFrame:
        if self.inner_ is None or self.representatives_ is None:
            raise RuntimeError("HmdbTargetPreprocessor is not fitted: call fit() on training data")
        return self.inner_.transform(self._by_hmdb(Y, self.representatives_))

    # The attributes the protocol reads from a target preprocessor (predict, tripwire, counts).
    @property
    def features_(self) -> list[str] | None:
        return None if self.inner_ is None else list(self.inner_.features_)

    @property
    def fit_samples_(self) -> list[Any] | None:
        return None if self.inner_ is None else self.inner_.fit_samples_

    @property
    def n_dropped_zero_(self) -> int | None:
        return None if self.inner_ is None else self.inner_.n_dropped_zero_

    def to_dict(self) -> dict[str, Any]:
        """Fitted state as JSON-ready objects (representatives + the inner preprocessor)."""
        return {
            "kind": type(self).__name__,
            "hmdb_order": list(self.hmdb_order),
            "representatives": dict(self.representatives_ or {}),
            "n_mostly_zero": self.n_mostly_zero_,
            "inner": None if self.inner_ is None else self.inner_.to_dict(),
        }

    def _by_hmdb(self, Y: pd.DataFrame, representatives: Mapping[str, str]) -> pd.DataFrame:
        columns = [representatives[hmdb] for hmdb in self.hmdb_order]
        return Y[columns].set_axis(self.hmdb_order, axis=1)
