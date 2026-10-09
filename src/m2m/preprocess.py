"""Preprocessing as fit(train) → transform(any) objects (section 4 of the brief).

Leakage contract: every fitted quantity (kept taxa, retained metabolites, the fallback
pseudocount, the z-score mean/std) is computed in `fit` from training rows only and is never
changed afterwards. `transform` is a pure function of its input and those parameters, so
validation or test data cannot move them (tests/test_preprocess.py checks this with deliberately
mutated test sets). Fitted parameters serialise to JSON (`to_dict` / `save` / `load`) together
with the IDs of the samples they were fitted on, so a saved run can be audited and re-applied.

Choices a reviewer should know about (docs/DESIGN.md, decisions 2-3):
- CLR uses the natural log. The base only rescales each feature, which z-scoring removes.
- The pseudocount is per sample (½ × smallest non-zero kept value), so it needs no fitting and
  CLR is unchanged by re-closure. The fitted fallback only covers samples in which every kept
  value is zero.
- Microbial inputs are re-closed over all of their own columns, and fitted taxa missing from an
  input count as absent (0). Metabolite targets have no such default: a missing one is an error.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd

__all__ = [
    "FittedPreprocessor",
    "MetabolitePreprocessor",
    "MicrobePreprocessor",
    "StandardScalerDF",
    "centred_log_ratio",
    "closure",
    "fit_preprocessor",
    "half_min_nonzero_pseudocount",
    "prevalent_taxa",
    "transform",
]

FORMAT_VERSION = 1  # bump when the JSON layout changes; loading another version fails loudly

MICROBE_TRANSFORMS = ("clr", "log")
METABOLITE_TRANSFORMS = ("log10", "clr")
MICROBE_CONFIG_KEYS = ("abundance_threshold", "min_prevalence", "transform", "standardize")
METABOLITE_CONFIG_KEYS = ("transform", "max_zero_fraction", "standardize", "pseudocount")
# Default CLR pseudocount rule; a positive number instead reproduces published "CLR(x + c)"
# protocols (the mNODE/MiMeNet comparison uses c = 1, docs/published_numbers.md).
HALF_MIN_NONZERO = "half_min_nonzero"

# A training feature whose std is this small relative to its magnitude is constant up to
# floating-point rounding; dividing by that rounding noise would turn it into arbitrary ±1 values.
CONSTANT_STD_RTOL = 1e-12

# Error messages name at most this many missing or duplicated labels (TALL has thousands).
MAX_LISTED_NAMES = 20


# --------------------------------------------------------------------------------------------
# Stateless building blocks
# --------------------------------------------------------------------------------------------


def closure(X: pd.DataFrame) -> pd.DataFrame:
    """Rescale each row to sum to 1. All-zero rows stay all-zero instead of becoming NaN."""
    totals = X.sum(axis=1).to_numpy(dtype=float)
    safe = np.where(totals > 0, totals, 1.0)
    return X.div(safe, axis=0)


def prevalent_taxa(
    X_rel: pd.DataFrame, abundance_threshold: float, min_prevalence: float
) -> list[str]:
    """Taxa whose relative abundance exceeds `abundance_threshold` in ≥ `min_prevalence` of rows.

    Callers must pass training rows only: this is a fitted quantity.
    """
    prevalence = (X_rel > abundance_threshold).mean(axis=0)
    return prevalence.index[prevalence >= min_prevalence].tolist()


def half_min_nonzero_pseudocount(values: np.ndarray, fallback: float) -> np.ndarray:
    """Per-row pseudocount: half of the row's smallest strictly positive value.

    Rows without any positive value get `fallback`. Because the rule scales with the row, a CLR
    built on it does not change when a row is multiplied by a constant (e.g. by re-closure).
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D samples × features array, got shape {values.shape}")
    if not (np.isfinite(fallback) and fallback > 0):
        raise ValueError(f"fallback pseudocount must be positive and finite, got {fallback!r}")
    smallest = np.where(values > 0, values, np.inf).min(axis=1, initial=np.inf)
    return np.where(np.isfinite(smallest), 0.5 * smallest, float(fallback))


def centred_log_ratio(values: np.ndarray, pseudocount: np.ndarray) -> np.ndarray:
    """CLR of each row: ln(x + c) minus its row mean, with c the pseudocount of that row."""
    logged = np.log(values + pseudocount[:, None])
    return logged - logged.mean(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------------
# Shared machinery: serialisation, validation, standardisation
# --------------------------------------------------------------------------------------------


class _JsonSerializable(ABC):
    """JSON persistence: subclasses define `to_dict` / `from_dict`, this adds `save` / `load`."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """All parameters (and fitted state) as JSON-serialisable built-in Python objects."""

    @classmethod
    @abstractmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Rebuild an object from `to_dict()` output."""

    def save(self, path: str | Path) -> Path:
        """Write `to_dict()` as JSON. NaN/inf would make invalid JSON, so they raise instead."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), allow_nan=False) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Read an object written by `save`."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def _header(self) -> dict[str, Any]:
        return {"kind": type(self).__name__, "format_version": FORMAT_VERSION}

    @classmethod
    def _check_header(cls, payload: Mapping[str, Any]) -> None:
        kind, version = payload.get("kind"), payload.get("format_version")
        if kind != cls.__name__:
            raise ValueError(f"expected a saved {cls.__name__}, got kind={kind!r}")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported format_version {version!r} (expected {FORMAT_VERSION})")


class _Preprocessor(_JsonSerializable):
    """Common surface of the fit(train) → transform(any) objects."""

    features_: list[Any] | None

    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> Self:
        """Learn every parameter from training rows."""

    @abstractmethod
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted parameters to any rows; never changes the fitted state."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Constructor arguments, so `type(p)(**p.get_params())` is an unfitted copy."""

    @property
    def is_fitted(self) -> bool:
        return self.features_ is not None

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """`fit(frame)` then `transform(frame)`: the training data on the fitted scale."""
        return self.fit(frame).transform(frame)

    def clone(self) -> Self:
        """Unfitted copy with the same parameters (one per CV fold, so folds never share state)."""
        return type(self)(**self.get_params())

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted: call fit() on training data")

    def __setstate__(self, state: dict[str, Any]) -> None:
        # pickle and deepcopy (e.g. shipping a fitted object to a joblib worker) restore arrays
        # as writeable; re-freeze them so the read-only guarantee holds in every process.
        self.__dict__.update(state)
        for name in ("mean_", "std_"):
            if self.__dict__.get(name) is not None:
                self.__dict__[name] = _frozen(self.__dict__[name])

    def __repr__(self) -> str:
        shown = ", ".join(f"{k}={_short_repr(v)}" for k, v in self.get_params().items())
        state = f"fitted, {len(self.features_)} features" if self.is_fitted else "unfitted"
        return f"{type(self).__name__}({shown}) [{state}]"


def _short_repr(value: Any) -> str:
    return f"<{len(value)} names>" if isinstance(value, list) else repr(value)


def _checked_values(frame: pd.DataFrame, what: str, *, nonnegative: bool = True) -> np.ndarray:
    """The table as floats, after the checks every preprocessor relies on."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{what} must be a pandas DataFrame, got {type(frame).__name__}")
    if frame.columns.has_duplicates:
        duplicated = frame.columns[frame.columns.duplicated()].unique().tolist()
        raise ValueError(f"{what} has duplicated columns: {_preview(duplicated)}")
    non_numeric = [
        c for c, dtype in frame.dtypes.items() if not pd.api.types.is_numeric_dtype(dtype)
    ]
    if non_numeric:
        raise ValueError(f"{what} has non-numeric columns: {_preview(non_numeric)}")
    values = frame.to_numpy(dtype=float)
    if np.isnan(values).any():
        # The NaN policy (drop / impute) is a data decision made upstream, not a silent default.
        raise ValueError(f"{what} contains NaN; handle missing values before preprocessing")
    if np.isinf(values).any():
        raise ValueError(f"{what} contains infinite values")
    if nonnegative and (values < 0).any():
        raise ValueError(f"{what} contains negative values; abundances and intensities are >= 0")
    return values


def _training_values(frame: pd.DataFrame, what: str, *, nonnegative: bool = True) -> np.ndarray:
    values = _checked_values(frame, what, nonnegative=nonnegative)
    if values.shape[0] == 0:
        raise ValueError(f"cannot fit on an empty {what}")
    return values


def _require_columns(frame: pd.DataFrame, columns: Sequence[Hashable], what: str) -> None:
    """Raise KeyError naming the expected columns that `frame` lacks."""
    present = set(frame.columns)
    missing = [c for c in columns if c not in present]
    if missing:
        raise KeyError(f"{what} lacks {len(missing)} required column(s): {_preview(missing)}")


def _check_aligned(X: pd.DataFrame, Y: pd.DataFrame) -> None:
    """Microbes and metabolites must describe the same samples, row for row."""
    if not X.index.equals(Y.index):
        raise ValueError("X and Y must index the same samples in the same order")


def _preview(labels: Sequence[Hashable]) -> str:
    shown = ", ".join(repr(label) for label in labels[:MAX_LISTED_NAMES])
    hidden = len(labels) - MAX_LISTED_NAMES
    return f"{shown}, ... ({hidden} more)" if hidden > 0 else shown


def _fit_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and population std (ddof=0, as scikit-learn's StandardScaler).

    A constant column gets std 1, so it maps to 0 instead of being divided by zero.
    """
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    constant = std <= CONSTANT_STD_RTOL * np.maximum(np.abs(mean), 1.0)
    return _frozen(mean), _frozen(np.where(constant, 1.0, std))


def _frozen(values: Any) -> np.ndarray:
    """Read-only float copy: a fitted parameter cannot be modified in place by later code."""
    array = np.array(values, dtype=float)
    array.setflags(write=False)
    return array


def _frozen_or_none(values: Any) -> np.ndarray | None:
    return None if values is None else _frozen(values)


def _list_or_none(values: np.ndarray | None) -> list[float] | None:
    return None if values is None else values.tolist()


def _plain_labels(labels: Iterable[Hashable]) -> list[Any]:
    """Labels as built-in Python objects (numpy scalars unwrapped) so JSON round-trips them."""
    return [label.item() if isinstance(label, np.generic) else label for label in labels]


def _unique_labels(labels: Sequence[Hashable], what: str) -> tuple[Any, ...]:
    if isinstance(labels, str):
        raise TypeError(f"{what} must be a sequence of names, not the single string {labels!r}")
    unique = tuple(_plain_labels(labels))
    if not unique:
        raise ValueError(f"{what} is empty")
    if len(set(unique)) != len(unique):
        raise ValueError(f"{what} contains duplicated names")
    return unique


def _check_config_keys(section: Mapping[str, Any], allowed: Sequence[str], name: str) -> None:
    """Reject unknown keys: a misspelt option would otherwise silently keep its default value."""
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in config section '{name}'; allowed: {allowed}")


def _check_choice(value: str, choices: Sequence[str], name: str) -> str:
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")
    return value


def _check_fraction(value: Any, name: str) -> float:
    value = float(value)  # float() also accepts '1e-4', which PyYAML reads as a string
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _check_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return bool(value)


# --------------------------------------------------------------------------------------------
# Microbes
# --------------------------------------------------------------------------------------------


class MicrobePreprocessor(_Preprocessor):
    """Microbial abundances → CLR (or log10) features, z-scored with training statistics.

    fit: re-close each training row over all of its taxa → keep taxa above `abundance_threshold`
    in at least `min_prevalence` of training rows (or the fixed `features` list, e.g. a pretraining
    vocabulary) → per-sample pseudocount → CLR or log10 → per-feature mean and std.
    transform: the same steps with the fitted taxa and statistics, on any samples.
    """

    def __init__(
        self,
        *,
        abundance_threshold: float = 1e-4,
        min_prevalence: float = 0.10,
        transform: str = "clr",
        standardize: bool = True,
        features: Sequence[str] | None = None,
    ) -> None:
        self.abundance_threshold = float(abundance_threshold)
        if not (np.isfinite(self.abundance_threshold) and self.abundance_threshold >= 0):
            raise ValueError(f"abundance_threshold must be >= 0, got {abundance_threshold!r}")
        self.min_prevalence = _check_fraction(min_prevalence, "min_prevalence")
        # Stored as `transform_type` because `transform` is the method.
        self.transform_type = _check_choice(transform, MICROBE_TRANSFORMS, "transform")
        self.standardize = _check_bool(standardize, "standardize")
        self.features = None if features is None else _unique_labels(features, "features")

        self.input_features_: list[Any] | None = None
        self.features_: list[Any] | None = None
        self.fallback_pseudocount_: float | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.fit_samples_: list[Any] | None = None

    @classmethod
    def from_config(
        cls, section: Mapping[str, Any], features: Sequence[str] | None = None
    ) -> Self:
        """Build from the `microbes` section of an experiment config."""
        _check_config_keys(section, MICROBE_CONFIG_KEYS, "microbes")
        return cls(**section, features=features)

    def get_params(self) -> dict[str, Any]:
        return {
            "abundance_threshold": self.abundance_threshold,
            "min_prevalence": self.min_prevalence,
            "transform": self.transform_type,
            "standardize": self.standardize,
            "features": None if self.features is None else list(self.features),
        }

    def fit(self, X: pd.DataFrame) -> Self:
        """Learn the kept taxa, fallback pseudocount and z-score statistics from training rows."""
        _training_values(X, "X")
        relative = closure(X)
        features = self._select_features(relative)
        kept = _kept_abundances(relative, features)
        positive = kept[kept > 0]
        if positive.size == 0:
            raise ValueError("none of the kept taxa is present in any training sample")
        # For samples whose kept taxa are all zero: half the smallest training value, i.e. below
        # anything seen in training. Only "log" depends on it; CLR maps such a sample to all zeros.
        fallback = float(0.5 * positive.min())
        mean, std = (None, None)
        if self.standardize:
            mean, std = _fit_mean_std(self._log_scale(kept, fallback))

        # Assigned last, so a failed fit never leaves a half-fitted object behind.
        self.input_features_ = _plain_labels(X.columns)
        self.features_ = features
        self.fallback_pseudocount_ = fallback
        self.mean_, self.std_ = mean, std
        self.fit_samples_ = _plain_labels(X.index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fitted features of any samples; fitted taxa absent from X count as zero."""
        self._check_fitted()
        _checked_values(X, "X")
        if not set(self.features_) & set(X.columns):
            raise ValueError(
                "X shares no column with the fitted taxa (wrong table, e.g. species vs genera?)"
            )
        kept = _kept_abundances(closure(X), self.features_)
        values = self._log_scale(kept, self.fallback_pseudocount_)
        if self.standardize:
            values = (values - self.mean_) / self.std_
        return pd.DataFrame(values, index=X.index, columns=pd.Index(self.features_))

    def to_dict(self) -> dict[str, Any]:
        fitted = None
        if self.is_fitted:
            fitted = {
                "input_features": list(self.input_features_),
                "features": list(self.features_),
                "fallback_pseudocount": self.fallback_pseudocount_,
                "mean": _list_or_none(self.mean_),
                "std": _list_or_none(self.std_),
                "fit_samples": list(self.fit_samples_),
            }
        return {**self._header(), "params": self.get_params(), "fitted": fitted}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        cls._check_header(payload)
        prep = cls(**payload["params"])
        fitted = payload.get("fitted")
        if fitted is not None:
            prep.input_features_ = list(fitted["input_features"])
            prep.features_ = list(fitted["features"])
            prep.fallback_pseudocount_ = float(fitted["fallback_pseudocount"])
            prep.mean_ = _frozen_or_none(fitted["mean"])
            prep.std_ = _frozen_or_none(fitted["std"])
            prep.fit_samples_ = list(fitted["fit_samples"])
        return prep

    def _select_features(self, relative: pd.DataFrame) -> list[Any]:
        if self.features is not None:
            if not set(self.features) & set(relative.columns):
                raise ValueError("none of the fixed `features` is a column of the training data")
            return list(self.features)
        features = prevalent_taxa(relative, self.abundance_threshold, self.min_prevalence)
        if not features:
            raise ValueError("no taxon passes the prevalence filter on the training data")
        return _plain_labels(features)

    def _log_scale(self, kept: np.ndarray, fallback: float) -> np.ndarray:
        pseudocount = half_min_nonzero_pseudocount(kept, fallback)
        if self.transform_type == "clr":
            return centred_log_ratio(kept, pseudocount)
        # "log" uses the same pseudocount, so ablation A1 isolates CLR's per-sample centring.
        return np.log10(kept + pseudocount[:, None])


def _kept_abundances(relative: pd.DataFrame, features: Sequence[Hashable]) -> np.ndarray:
    """Relative abundances of the fitted taxa; taxa missing from the table count as absent."""
    return relative.reindex(columns=list(features), fill_value=0.0).to_numpy(dtype=float)


# --------------------------------------------------------------------------------------------
# Metabolites
# --------------------------------------------------------------------------------------------


class MetabolitePreprocessor(_Preprocessor):
    """Metabolite intensities → log10(x + 1) (or CLR) targets, z-scored with training statistics.

    fit: retain features that are zero in at most `max_zero_fraction` of training rows → keep the
    `candidate_features` (e.g. T466; default: every column) that were retained, in candidate order
    → transform → per-target mean and std. For CLR the per-sample reference is every retained
    feature, so target sets fitted on the same training rows (T466, TALL) share one denominator.
    """

    def __init__(
        self,
        *,
        transform: str = "log10",
        max_zero_fraction: float = 0.5,
        standardize: bool = True,
        candidate_features: Sequence[str] | None = None,
        pseudocount: str | float = HALF_MIN_NONZERO,
    ) -> None:
        # Stored as `transform_type` because `transform` is the method.
        self.transform_type = _check_choice(transform, METABOLITE_TRANSFORMS, "transform")
        self.max_zero_fraction = _check_fraction(max_zero_fraction, "max_zero_fraction")
        self.standardize = _check_bool(standardize, "standardize")
        self.pseudocount = _check_pseudocount(pseudocount)
        self.candidate_features = (
            None
            if candidate_features is None
            else _unique_labels(candidate_features, "candidate_features")
        )

        self.retained_: list[Any] | None = None
        self.features_: list[Any] | None = None
        self.clr_reference_: list[Any] | None = None
        self.fallback_pseudocount_: float | None = None
        self.n_candidates_: int | None = None
        self.n_dropped_zero_: int | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.fit_samples_: list[Any] | None = None

    @classmethod
    def from_config(
        cls, section: Mapping[str, Any], candidate_features: Sequence[str] | None = None
    ) -> Self:
        """Build from the `metabolites` section of an experiment config."""
        _check_config_keys(section, METABOLITE_CONFIG_KEYS, "metabolites")
        return cls(**section, candidate_features=candidate_features)

    def get_params(self) -> dict[str, Any]:
        return {
            "transform": self.transform_type,
            "max_zero_fraction": self.max_zero_fraction,
            "standardize": self.standardize,
            "candidate_features": (
                None if self.candidate_features is None else list(self.candidate_features)
            ),
            "pseudocount": self.pseudocount,
        }

    def fit(self, Y: pd.DataFrame) -> Self:
        """Learn the retained targets, CLR reference and z-score statistics from training rows."""
        values = _training_values(Y, "Y")
        candidates = (
            _plain_labels(Y.columns)
            if self.candidate_features is None
            else list(self.candidate_features)
        )
        # A candidate absent from Y is a caller bug; silently dropping it would be miscounted as
        # a zero-filter drop.
        _require_columns(Y, candidates, "Y (candidate_features)")
        zero_fraction = (values == 0).sum(axis=0) / values.shape[0]  # count / n: 50% is exactly 0.5
        retained = _plain_labels(Y.columns[zero_fraction <= self.max_zero_fraction])
        retained_set = set(retained)
        features = [c for c in candidates if c in retained_set]
        if not features:
            raise ValueError(
                f"every candidate target is zero in more than {self.max_zero_fraction:.0%} "
                "of training samples"
            )
        reference, fallback = None, None
        if self.transform_type == "clr":
            reference = retained
            fallback = _clr_fallback(Y[reference].to_numpy(dtype=float))
        mean, std = (None, None)
        if self.standardize:
            mean, std = _fit_mean_std(self._scale(Y, features, reference, fallback))

        # Assigned last, so a failed fit never leaves a half-fitted object behind.
        self.retained_ = retained
        self.features_ = features
        self.clr_reference_ = reference
        self.fallback_pseudocount_ = fallback
        self.n_candidates_ = len(candidates)
        self.n_dropped_zero_ = len(candidates) - len(features)
        self.mean_, self.std_ = mean, std
        self.fit_samples_ = _plain_labels(Y.index)
        return self

    def transform(self, Y: pd.DataFrame) -> pd.DataFrame:
        """Targets of any samples on the fitted scale; KeyError if a needed feature is missing."""
        self._check_fitted()
        needed = self.features_ if self.clr_reference_ is None else self.clr_reference_
        _require_columns(Y, needed, "Y")
        # Only the columns used are checked: a NaN in an unused column is not our concern.
        _checked_values(Y[needed], "Y")
        values = self._scale(Y, self.features_, self.clr_reference_, self.fallback_pseudocount_)
        if self.standardize:
            values = (values - self.mean_) / self.std_
        return pd.DataFrame(values, index=Y.index, columns=pd.Index(self.features_))

    def inverse_standardize(self, Z: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Map standardised targets (e.g. predictions) back to the log10 / CLR scale."""
        self._check_fitted()
        if isinstance(Z, pd.DataFrame):
            _require_columns(Z, self.features_, "Z")
            values = self._unstandardize(Z[self.features_].to_numpy(dtype=float))
            return pd.DataFrame(values, index=Z.index, columns=pd.Index(self.features_))
        values = np.asarray(Z, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.features_):
            raise ValueError(f"expected shape (n, {len(self.features_)}), got {values.shape}")
        return self._unstandardize(values)

    def to_dict(self) -> dict[str, Any]:
        fitted = None
        if self.is_fitted:
            fitted = {
                "retained": list(self.retained_),
                "features": list(self.features_),
                "clr_reference": None if self.clr_reference_ is None else list(self.clr_reference_),
                "fallback_pseudocount": self.fallback_pseudocount_,
                "n_candidates": self.n_candidates_,
                "n_dropped_zero": self.n_dropped_zero_,
                "mean": _list_or_none(self.mean_),
                "std": _list_or_none(self.std_),
                "fit_samples": list(self.fit_samples_),
            }
        return {**self._header(), "params": self.get_params(), "fitted": fitted}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        cls._check_header(payload)
        prep = cls(**payload["params"])
        fitted = payload.get("fitted")
        if fitted is not None:
            prep.retained_ = list(fitted["retained"])
            prep.features_ = list(fitted["features"])
            reference = fitted["clr_reference"]
            prep.clr_reference_ = None if reference is None else list(reference)
            fallback = fitted["fallback_pseudocount"]
            prep.fallback_pseudocount_ = None if fallback is None else float(fallback)
            prep.n_candidates_ = int(fitted["n_candidates"])
            prep.n_dropped_zero_ = int(fitted["n_dropped_zero"])
            prep.mean_ = _frozen_or_none(fitted["mean"])
            prep.std_ = _frozen_or_none(fitted["std"])
            prep.fit_samples_ = list(fitted["fit_samples"])
        return prep

    def _scale(
        self,
        Y: pd.DataFrame,
        features: list[Any],
        reference: list[Any] | None,
        fallback: float | None,
    ) -> np.ndarray:
        if self.transform_type == "log10":
            # The +1 keeps zeros at 0 without a fitted pseudocount.
            return np.log10(Y[features].to_numpy(dtype=float) + 1.0)
        ref_values = Y[reference].to_numpy(dtype=float)
        if self.pseudocount == HALF_MIN_NONZERO:
            pseudocount = half_min_nonzero_pseudocount(ref_values, fallback)
        else:
            pseudocount = np.full(len(ref_values), float(self.pseudocount))
        clr = centred_log_ratio(ref_values, pseudocount)
        return clr[:, pd.Index(reference).get_indexer(features)]

    def _unstandardize(self, values: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return values.copy()
        return values * self.std_ + self.mean_


def _check_pseudocount(value: str | float) -> str | float:
    if value == HALF_MIN_NONZERO:
        return HALF_MIN_NONZERO
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"pseudocount must be {HALF_MIN_NONZERO!r} or a positive number, got {value!r}"
        )
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"pseudocount must be positive and finite, got {value!r}")
    return float(value)


def _clr_fallback(reference_values: np.ndarray) -> float:
    """Pseudocount for samples with no positive reference value: half the smallest in training."""
    positive = reference_values[reference_values > 0]
    if positive.size == 0:
        raise ValueError("the CLR reference has no positive value in the training data")
    return float(0.5 * positive.min())


# --------------------------------------------------------------------------------------------
# Plain z-scoring (MGM embeddings)
# --------------------------------------------------------------------------------------------


class StandardScalerDF(_Preprocessor):
    """Column-wise z-scoring of a samples × features table (e.g. MGM embeddings).

    Unlike scikit-learn's StandardScaler it keeps labels: transform selects the fitted columns by
    name, so a reordered table cannot be silently scaled with another column's statistics.
    """

    def __init__(self) -> None:
        self.features_: list[Any] | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.fit_samples_: list[Any] | None = None

    def get_params(self) -> dict[str, Any]:
        return {}

    def fit(self, E: pd.DataFrame) -> Self:
        """Learn per-column mean and std (ddof=0; constant columns get std 1) from training rows."""
        mean, std = _fit_mean_std(_training_values(E, "E", nonnegative=False))
        self.features_ = _plain_labels(E.columns)
        self.mean_, self.std_ = mean, std
        self.fit_samples_ = _plain_labels(E.index)
        return self

    def transform(self, E: pd.DataFrame) -> pd.DataFrame:
        """z-score any rows with the training statistics; KeyError if a fitted column is missing."""
        self._check_fitted()
        _require_columns(E, self.features_, "E")
        values = _checked_values(E[self.features_], "E", nonnegative=False)
        z = (values - self.mean_) / self.std_
        return pd.DataFrame(z, index=E.index, columns=pd.Index(self.features_))

    def to_dict(self) -> dict[str, Any]:
        fitted = None
        if self.is_fitted:
            fitted = {
                "features": list(self.features_),
                "mean": self.mean_.tolist(),
                "std": self.std_.tolist(),
                "fit_samples": list(self.fit_samples_),
            }
        return {**self._header(), "params": self.get_params(), "fitted": fitted}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        cls._check_header(payload)
        scaler = cls(**payload["params"])
        fitted = payload.get("fitted")
        if fitted is not None:
            scaler.features_ = list(fitted["features"])
            scaler.mean_ = _frozen(fitted["mean"])
            scaler.std_ = _frozen(fitted["std"])
            scaler.fit_samples_ = list(fitted["fit_samples"])
        return scaler


# --------------------------------------------------------------------------------------------
# Paired input/target preprocessing
# --------------------------------------------------------------------------------------------


@dataclass
class FittedPreprocessor(_JsonSerializable):
    """The fitted input (`x`) and target (`y`) preprocessing of one training set."""

    x: MicrobePreprocessor | StandardScalerDF
    y: MetabolitePreprocessor

    def summary(self) -> dict[str, int]:
        """Feature counts to log with every run (brief §4: record how many targets remain)."""
        return {
            "n_input_features": len(self.x.features_),
            "n_target_candidates": self.y.n_candidates_,
            "n_targets": len(self.y.features_),
            "n_targets_dropped_zero": self.y.n_dropped_zero_,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._header(), "x": self.x.to_dict(), "y": self.y.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        cls._check_header(payload)
        x_kind = payload["x"].get("kind")
        if x_kind not in _INPUT_PREPROCESSORS:
            raise ValueError(f"unknown input preprocessor kind {x_kind!r}")
        x = _INPUT_PREPROCESSORS[x_kind].from_dict(payload["x"])
        return cls(x=x, y=MetabolitePreprocessor.from_dict(payload["y"]))


_INPUT_PREPROCESSORS: dict[str, type[MicrobePreprocessor] | type[StandardScalerDF]] = {
    "MicrobePreprocessor": MicrobePreprocessor,
    "StandardScalerDF": StandardScalerDF,
}


def fit_preprocessor(
    cfg: Mapping[str, Any],
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    *,
    target_features: Sequence[str] | None = None,
    x_features: Sequence[str] | None = None,
) -> FittedPreprocessor:
    """Fit microbe and metabolite preprocessing on one training set.

    Settings come from `cfg["microbes"]` and `cfg["metabolites"]`; `target_features` restricts
    the targets (e.g. T466) and `x_features` fixes the input taxa (e.g. a pretraining vocabulary).
    Pass training rows only: everything learned here is applied unchanged to other samples.
    """
    _check_aligned(X_train, Y_train)
    x = MicrobePreprocessor.from_config(cfg["microbes"], features=x_features).fit(X_train)
    y = MetabolitePreprocessor.from_config(
        cfg["metabolites"], candidate_features=target_features
    ).fit(Y_train)
    return FittedPreprocessor(x=x, y=y)


def transform(
    prep: FittedPreprocessor, X: pd.DataFrame, Y: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Apply fitted preprocessing to any samples: (inputs, targets or None when Y is None)."""
    if Y is not None:
        _check_aligned(X, Y)
    X_t = prep.x.transform(X)
    Y_t = None if Y is None else prep.y.transform(Y)
    return X_t, Y_t
