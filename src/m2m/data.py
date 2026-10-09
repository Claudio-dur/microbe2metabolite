"""Raw-data loaders (D1, D2, D3), the external-test guard and the section-3 data checks.

Every loader takes ``raw_dir``, the folder that ``scripts/download_data.sh`` fills (default
``paths.DATA_RAW``, read at call time so tests can point it at a synthetic tree):

- D1 = ``raw_dir/MiMeNet/data/IBD``: MiMeNet's IBD tables. PRISM trains, NLIBD is the external test.
- D2 = ``raw_dir/microbiome-metabolome-curated-data/data/processed_data``: FRANZOSA_IBD_2019 (the
  same two cohorts, profiled as GTDB genera) and five source datasets for pretraining.
- D3 = ``raw_dir/MGM``: MGM's resources, read as JSON/CSV only. MGM's code is never imported here.

The external test cohort (D1 NLIBD = D2 Franzosa ``Validation.*``) leaves this module only through
``load_nlibd`` and ``load_franzosa(part="Validation")`` with ``final_evaluation=True``; every such
access is appended to ``results/external_access_log.csv``.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from m2m import paths, preprocess, tracking
from m2m.config import load_config

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------

D2_DATASETS = (
    "FRANZOSA_IBD_2019",
    "iHMP_IBDMDB_2019",
    "YACHIDA_CRC_2019",
    "WANG_ESRD_2020",
    "ERAWIJANTARI_GASTRIC_CANCER_2020",
    "MARS_IBS_2020",
)
TARGET_DATASET = "FRANZOSA_IBD_2019"
SOURCE_DATASETS = tuple(name for name in D2_DATASETS if name != TARGET_DATASET)
LONGITUDINAL_DATASETS = ("iHMP_IBDMDB_2019", "MARS_IBS_2020")
DIAGNOSES = ("CD", "UC", "Control")
FRANZOSA_PARTS = ("PRISM", "Validation")

SAMPLE_ID = "sample_id"
REQUIRED_META_COLUMNS = ("subject", "diagnosis", "dataset")

# D1's diagnosis_PRISM.csv only says IBD/Healthy; the 3-class D2 labels must collapse onto it.
IBD_STATUS = {"CD": "IBD", "UC": "IBD", "Control": "Healthy"}
NOT_READ = "not read"  # metadata value when only the external cohort's inputs were loaded

# iHMP's metabolome is 81,867 LC-MS features wide; only the HMDB-mapped ones are ever used, so the
# loader refuses to read the full table (brief section 8).
HMDB_ONLY_REQUIRED = ("iHMP_IBDMDB_2019",)

# ERAWIJANTARI re-published 53 YACHIDA metagenomes and flags them in its metadata. The YACHIDA copy
# is kept, otherwise the same metagenome could sit on both sides of a pretraining split.
DUPLICATED_IN = "ERAWIJANTARI_GASTRIC_CANCER_2020"
DUPLICATE_FLAG = "Shared.w.YACHIDA_2019"
DUPLICATE_KEPT_IN = "YACHIDA_CRC_2019"

# Two copies of the same measurement differ only by text round-off (~1e-13 here); real
# replicate measurements differ by orders of magnitude more.
IDENTITY_TOL = 1e-6

# GTDB splits polyphyletic genera with suffixes ("Faecalibacterium_A"); MGM's vocabulary does not.
_GTDB_GENUS = re.compile(r"(?:^|;)g__([^;]*)")
_GTDB_SUFFIX = re.compile(r"_[A-Z]+$")

MGM_RESOURCES = Path("mgm") / "resources"
MGM_MODEL_DIR = MGM_RESOURCES / "general_model"


class ExternalTestSetAccessError(RuntimeError):
    """The external test cohort was requested outside a final evaluation."""


class DataFactError(AssertionError):
    """The downloaded data does not match a fact the code relies on."""


# --------------------------------------------------------------------------------------------
# Cohort
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class Cohort:
    """One cohort: microbes, optional metabolites and metadata, with identical rows.

    ``microbes`` holds the values as published (not re-closed) and ``metabolites`` raw
    intensities; ``meta`` has at least ``subject``, ``diagnosis`` and ``dataset``.
    """

    name: str
    microbes: pd.DataFrame
    metabolites: pd.DataFrame | None
    meta: pd.DataFrame

    def __post_init__(self) -> None:
        if not self.meta.index.is_unique:
            raise ValueError(f"{self.name}: duplicate sample IDs in meta")
        missing = [col for col in REQUIRED_META_COLUMNS if col not in self.meta.columns]
        if missing:
            raise ValueError(f"{self.name}: meta lacks columns {missing}")
        tables = {"microbes": self.microbes}
        if self.metabolites is not None:
            tables["metabolites"] = self.metabolites
        for label, table in tables.items():
            if not table.index.equals(self.meta.index):
                raise ValueError(f"{self.name}: {label} rows differ from meta rows (IDs or order)")

    @property
    def sample_ids(self) -> list[str]:
        return self.meta.index.tolist()

    def subset(self, ids: Iterable[str]) -> Cohort:
        """The given samples, in the given order."""
        ids = list(ids)
        unknown = pd.Index(ids).difference(self.meta.index)
        if len(unknown):
            raise KeyError(f"{self.name}: unknown sample IDs, e.g. {unknown[:3].tolist()}")
        metabolites = None if self.metabolites is None else self.metabolites.loc[ids]
        return Cohort(self.name, self.microbes.loc[ids], metabolites, self.meta.loc[ids])

    def __repr__(self) -> str:
        def shape(table: pd.DataFrame | None) -> str:
            return "None" if table is None else f"{table.shape[0]}x{table.shape[1]}"

        return (
            f"Cohort({self.name!r}, microbes={shape(self.microbes)}, "
            f"metabolites={shape(self.metabolites)}, meta={shape(self.meta)})"
        )


# --------------------------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------------------------


def _raw_root(raw_dir: Path | None) -> Path:
    return Path(raw_dir) if raw_dir is not None else paths.DATA_RAW


def d1_dir(raw_dir: Path | None = None) -> Path:
    return _raw_root(raw_dir) / "MiMeNet" / "data" / "IBD"


def d2_dir(raw_dir: Path | None = None) -> Path:
    return _raw_root(raw_dir) / "microbiome-metabolome-curated-data" / "data" / "processed_data"


def d3_dir(raw_dir: Path | None = None) -> Path:
    return _raw_root(raw_dir) / "MGM"


# --------------------------------------------------------------------------------------------
# External-test guard
# --------------------------------------------------------------------------------------------


def _guard_external(
    final_evaluation: bool, purpose: str, detail: str, *, data_check: bool = False
) -> None:
    """Refuse the external test cohort unless this is a final evaluation; log every access.

    A final evaluation also needs the selection phase frozen and committed (m2m.freeze); only
    the section-3 data checks (`data_check=True`), which fit and score nothing, are exempt.
    """
    if final_evaluation is not True:
        raise ExternalTestSetAccessError(
            f"{detail} is the external test cohort. It is opened only to score a final, fixed "
            "configuration: pass final_evaluation=True and a purpose (the access is logged)."
        )
    if not purpose.strip():
        # The access log is the only evidence that the cohort was used once per final
        # configuration, so an unexplained access is refused rather than logged as "unspecified".
        raise ExternalTestSetAccessError(f"opening {detail} requires a non-empty purpose")
    if not data_check:
        from m2m import freeze  # local: freeze is only needed once the guard has passed

        freeze.require_frozen(purpose)
    tracking.log_external_access(purpose=purpose, detail=detail)


# --------------------------------------------------------------------------------------------
# D1: MiMeNet IBD tables (features x samples CSVs)
# --------------------------------------------------------------------------------------------


def load_prism(raw_dir: Path | None = None) -> Cohort:
    """D1 PRISM, the training cohort: 155 samples x 200 species and x 8,848 LC-MS features.

    ``diagnosis`` (CD/UC/Control) comes from the D2 Franzosa metadata because D1 only says
    IBD/Healthy (kept as ``ibd_status``); the two must agree.
    """
    folder = d1_dir(raw_dir)
    microbes = _read_d1_table(folder / "microbiome_PRISM.csv")
    metabolites = _read_d1_table(folder / "metabolome_PRISM.csv")
    _require_same_rows(microbes, metabolites, "D1 PRISM microbiome vs metabolome")
    samples = microbes.index
    ibd_status = _labels_for(samples, _read_d1_labels(folder / "diagnosis_PRISM.csv"))
    d2_labels = load_d2_metadata(TARGET_DATASET, raw_dir)["diagnosis"]
    diagnosis = pd.Series(
        d2_labels.reindex([d1_to_d2_prism_id(s) for s in samples]).to_numpy(), index=samples
    )
    _require_known_diagnoses(diagnosis, "D2 Franzosa labels of D1 PRISM")
    _require_consistent_ibd_status(diagnosis, ibd_status)
    meta = _d1_meta(samples, diagnosis, ibd_status)
    return Cohort("PRISM", microbes, metabolites, meta)


def load_nlibd(
    raw_dir: Path | None = None,
    *,
    final_evaluation: bool = False,
    purpose: str = "",
    with_metabolites: bool = True,
    data_check: bool = False,
) -> Cohort:
    """D1 NLIBD, the external test cohort (65 samples). Guarded: see the module docstring.

    ``diagnosis`` comes from external_diagnosis.csv; ``ibd_status`` is derived from it so that
    PRISM and NLIBD share one metadata layout. ``with_metabolites=False`` reads the microbiome
    only (e.g. to embed the inputs with MGM), so the metabolome file is not opened at all.
    """
    detail = "D1 NLIBD" if with_metabolites else "D1 NLIBD (microbiome only)"
    _guard_external(final_evaluation, purpose, detail=detail, data_check=data_check)
    folder = d1_dir(raw_dir)
    microbes = _read_d1_table(folder / "microbiome_external.csv")
    samples = microbes.index
    if not with_metabolites:
        # Inputs only: neither the metabolome nor the diagnoses are read.
        unread = pd.Series(NOT_READ, index=samples)
        return Cohort("NLIBD", microbes, None, _d1_meta(samples, unread, unread))
    metabolites = _read_d1_table(folder / "metabolome_external.csv")
    _require_same_rows(microbes, metabolites, "D1 NLIBD microbiome vs metabolome")
    diagnosis = _labels_for(samples, _read_d1_labels(folder / "external_diagnosis.csv"))
    _require_known_diagnoses(diagnosis, "external_diagnosis.csv")
    meta = _d1_meta(samples, diagnosis, diagnosis.map(IBD_STATUS))
    return Cohort("NLIBD", microbes, metabolites, meta)


def load_annotation(raw_dir: Path | None = None) -> pd.DataFrame:
    """The 466 annotated metabolites (Retention Time, m/z, Compound Name, Adduct) by feature ID."""
    annotation = pd.read_csv(d1_dir(raw_dir) / "metabolome_annotation.csv", index_col=0)
    _require_unique(annotation.index, "metabolome_annotation.csv feature IDs")
    return annotation.rename_axis(index="feature_id")


def t466_features(raw_dir: Path | None = None) -> list[str]:
    """The T466 target set: annotated feature IDs, in metabolome column order."""
    annotated = set(load_annotation(raw_dir).index)
    stored = pd.read_csv(d1_dir(raw_dir) / "metabolome_PRISM.csv", usecols=[0])
    features = stored.iloc[:, 0].tolist()
    missing = annotated.difference(features)
    if missing:
        raise DataFactError(f"{len(missing)} annotated features are not in the metabolome")
    return [feature for feature in features if feature in annotated]


def species_to_genus(names: Iterable[str]) -> list[str]:
    """MetaPhlAn2 species -> first name token ("Bacteroides_vulgatus" -> "Bacteroides").

    Unclassified species are named after a higher rank, so the token is not always a genus
    ("Lachnospiraceae_bacterium_1_1_57FAA" -> "Lachnospiraceae", a family).
    """
    return [str(name).split("_", 1)[0] for name in names]


def d1_to_d2_prism_id(d1_id: str) -> str:
    """'PRISM|7122' -> 'PRISM.7122': D1 separates cohort and sample with '|', D2 with '.'."""
    prefix = "PRISM|"
    if not d1_id.startswith(prefix):
        raise ValueError(f"not a D1 PRISM sample ID: {d1_id!r}")
    return f"PRISM.{d1_id[len(prefix):]}"


def _read_d1_table(path: Path) -> pd.DataFrame:
    """A D1 features x samples CSV, returned as samples x features floats."""
    stored = pd.read_csv(path, index_col=0)
    table = stored.T.astype(float).rename_axis(index=SAMPLE_ID, columns=None)
    _require_unique(table.columns, f"{path.name} feature IDs")
    return table


def _read_d1_labels(path: Path) -> pd.Series:
    """The single label column of a D1 diagnosis CSV, indexed by sample ID."""
    table = pd.read_csv(path, index_col=0)
    if table.shape[1] != 1:
        raise DataFactError(f"{path.name}: expected one label column, found {table.shape[1]}")
    _require_unique(table.index, f"{path.name} sample IDs")
    return table.iloc[:, 0]


def _labels_for(samples: pd.Index, labels: pd.Series) -> pd.Series:
    """Labels in the order of ``samples``; every sample must have one."""
    aligned = pd.Series(labels.reindex(samples).to_numpy(), index=samples)
    missing = aligned.index[aligned.isna()]
    if len(missing):
        raise DataFactError(f"no label for {len(missing)} samples, e.g. {missing[:3].tolist()}")
    return aligned.astype(str)


def _require_known_diagnoses(diagnosis: pd.Series, source: str) -> None:
    unknown = set(diagnosis.fillna("<missing>")) - set(DIAGNOSES)
    if unknown:
        raise DataFactError(f"{source}: labels outside {DIAGNOSES}: {sorted(unknown)}")


def _require_consistent_ibd_status(diagnosis: pd.Series, ibd_status: pd.Series) -> None:
    implied = diagnosis.map(IBD_STATUS)
    conflicts = diagnosis.index[implied.to_numpy() != ibd_status.to_numpy()]
    if len(conflicts):
        raise DataFactError(
            f"D2 diagnosis contradicts D1 IBD/Healthy for {len(conflicts)} samples, "
            f"e.g. {conflicts[:3].tolist()}"
        )


def _d1_meta(samples: pd.Index, diagnosis: pd.Series, ibd_status: pd.Series) -> pd.DataFrame:
    # Each D1 person has exactly one sample, so the sample ID doubles as the subject ID.
    return pd.DataFrame(
        {
            "subject": samples.to_numpy(),
            "diagnosis": diagnosis.to_numpy(),
            "ibd_status": ibd_status.to_numpy(),
            "dataset": "D1",
        },
        index=samples,
    )


# --------------------------------------------------------------------------------------------
# D2: curated microbiome-metabolome collection (samples x features TSVs)
# --------------------------------------------------------------------------------------------


def load_d2_metadata(dataset: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Metadata indexed by sample ID (str), adding ``subject``, ``diagnosis`` and ``dataset``.

    For the target dataset only the ``PRISM.*`` rows are returned: the ``Validation.*`` rows are
    the external test cohort, reachable only through the guarded ``load_franzosa``.
    """
    meta = _read_metadata(_d2_folder(dataset, raw_dir), dataset)
    if dataset == TARGET_DATASET:
        meta = meta.loc[_part_ids(meta.index, "PRISM")]
    return meta


def load_d2_mtb_map(dataset: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """mtb.map.tsv (HMDB, KEGG, ...) indexed by ``Compound``, i.e. the mtb.tsv column names."""
    return _read_mtb_map(_d2_folder(dataset, raw_dir))


def load_d2(
    dataset: str,
    raw_dir: Path | None = None,
    *,
    with_metabolites: bool = False,
    hmdb_only: bool = False,
) -> Cohort:
    """A source dataset: genus relative abundances (full GTDB strings), optionally metabolites.

    ``hmdb_only`` reads only the HMDB-mapped metabolite columns (required for iHMP). Missing
    metabolite values stay NaN; the policy for them is decided downstream. The target dataset
    is refused here because it contains the external test cohort: use ``load_franzosa``.
    """
    if dataset == TARGET_DATASET:
        raise ExternalTestSetAccessError(
            f"{TARGET_DATASET} contains the external test cohort; use load_franzosa(part='PRISM') "
            "(or part='Validation' with final_evaluation=True for the final evaluation)."
        )
    return _load_d2_cohort(dataset, raw_dir, with_metabolites=with_metabolites, hmdb_only=hmdb_only)


def load_franzosa(
    raw_dir: Path | None = None,
    *,
    part: Literal["PRISM", "Validation"],
    with_metabolites: bool = True,
    final_evaluation: bool = False,
    purpose: str = "",
    data_check: bool = False,
) -> Cohort:
    """D2 view of PRISM (155) or of the external Validation cohort (65, guarded like NLIBD).

    Metabolite columns are renamed to the D1 feature IDs, so T466 means the same features.
    """
    if part not in FRANZOSA_PARTS:
        raise ValueError(f"part must be one of {FRANZOSA_PARTS}, got {part!r}")
    if part == "Validation":
        _guard_external(
            final_evaluation, purpose, detail="D2 Franzosa Validation", data_check=data_check
        )
    full = _load_d2_cohort(
        TARGET_DATASET, raw_dir, with_metabolites=with_metabolites, hmdb_only=False
    )
    cohort = full.subset(_part_ids(full.meta.index, part))
    _require_known_diagnoses(cohort.meta["diagnosis"], f"{TARGET_DATASET} {part}")
    metabolites = cohort.metabolites
    if metabolites is not None:
        metabolites = metabolites.rename(columns=franzosa_feature_id)
        _require_unique(metabolites.columns, "renamed Franzosa metabolite IDs")
    return Cohort(f"FRANZOSA_{part}", cohort.microbes, metabolites, cohort.meta)


def load_sources(
    raw_dir: Path | None = None,
    *,
    drop_duplicates: bool = True,
    with_metabolites: bool = False,
    hmdb_only: bool = True,
) -> dict[str, Cohort]:
    """The five source datasets, keyed by name in ``SOURCE_DATASETS`` order.

    ``drop_duplicates`` removes the ERAWIJANTARI copies of YACHIDA metagenomes. With
    ``with_metabolites``, ``hmdb_only`` (the default) keeps only HMDB-mapped compounds, the
    ones comparable across platforms; iHMP cannot be read without it.
    """
    cohorts = {
        dataset: load_d2(
            dataset, raw_dir, with_metabolites=with_metabolites, hmdb_only=hmdb_only
        )
        for dataset in SOURCE_DATASETS
    }
    return drop_duplicate_samples(cohorts) if drop_duplicates else cohorts


def drop_duplicate_samples(cohorts: Mapping[str, Cohort]) -> dict[str, Cohort]:
    """Drop the ERAWIJANTARI samples its metadata flags as YACHIDA metagenomes.

    Only done when YACHIDA is loaded too; otherwise nothing is duplicated.
    """
    out = dict(cohorts)
    if DUPLICATED_IN in out and DUPLICATE_KEPT_IN in out:
        cohort = out[DUPLICATED_IN]
        shared = _bool_flag(cohort.meta, DUPLICATE_FLAG)
        out[DUPLICATED_IN] = cohort.subset(cohort.meta.index[~shared])
    return out


def gtdb_genus(column: str, *, strip_suffix: bool = False) -> str:
    """Genus name of a GTDB taxonomy string: "d__...;g__Faecalibacterium_A" -> "Faecalibacterium_A".

    ``strip_suffix`` also drops GTDB's split suffix ("Faecalibacterium_A" -> "Faecalibacterium"),
    which is how other taxonomies (and MGM's vocabulary) name the genus.
    """
    found = _GTDB_GENUS.search(column)
    if found is None or not found.group(1):
        raise ValueError(f"no genus (g__) in taxonomy string {column!r}")
    genus = found.group(1)
    return _GTDB_SUFFIX.sub("", genus) if strip_suffix else genus


def franzosa_feature_id(compound: str) -> str:
    """'C18-neg_Cluster_0001: NA' -> 'C18-neg_Cluster_0001': D2 appends a name to D1's ID."""
    return compound.split(": ", 1)[0]


def match_nlibd_to_validation(
    d1_metabolites: pd.DataFrame, d2_metabolites: pd.DataFrame, *, tol: float = IDENTITY_TOL
) -> pd.Series:
    """Map D1 NLIBD sample IDs to D2 Validation IDs by identical metabolomes.

    The two releases name these samples differently (``Validation|LLDeep_0001`` vs
    ``Validation.UMCG...``), but carry the same measurements. Two samples match when the
    largest absolute difference of their log1p values is below ``tol``; the match must be
    one-to-one, otherwise ``DataFactError``.
    """
    if set(d1_metabolites.columns) != set(d2_metabolites.columns):
        raise DataFactError("D1 and D2 metabolomes do not have the same features")
    d1_values = np.log1p(d1_metabolites.to_numpy(dtype=float))
    d2_values = np.log1p(d2_metabolites[d1_metabolites.columns].to_numpy(dtype=float))
    close = _max_abs_diff(d1_values, d2_values) < tol
    hits = close.sum(axis=1)
    problems = []
    if (hits != 1).any():
        problems.append(
            f"{int((hits == 0).sum())} D1 samples match no D2 sample and "
            f"{int((hits > 1).sum())} match several"
        )
    matched = d2_metabolites.index[close.argmax(axis=1)]
    if (hits == 1).all() and matched.duplicated().any():
        problems.append("several D1 samples match the same D2 sample")
    if problems:
        raise DataFactError("NLIBD-Validation match is not one-to-one: " + "; ".join(problems))
    return pd.Series(
        matched.to_numpy(),
        index=pd.Index(d1_metabolites.index, name="nlibd_id"),
        name="validation_id",
    )


def _d2_folder(dataset: str, raw_dir: Path | None) -> Path:
    if dataset not in D2_DATASETS:
        raise ValueError(f"unknown D2 dataset {dataset!r}; expected one of {D2_DATASETS}")
    return d2_dir(raw_dir) / dataset


def _d2_file(folder: Path, name: str) -> Path:
    """``<name>.tsv``, or ``<name>.tsv.zip`` for the tables the upstream repository zips."""
    for candidate in (folder / f"{name}.tsv", folder / f"{name}.tsv.zip"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"neither {name}.tsv nor {name}.tsv.zip in {folder}")


def _table_columns(folder: Path, name: str) -> list[str]:
    """Feature names of a D2 samples x features table, parsed from its header line only.

    The csv module reads that one line in milliseconds; pandas (nrows=0) would build an empty
    frame with one Series per column, which takes seconds for iHMP's 81,867 columns.
    """
    path = _d2_file(folder, name)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as member:
            text = io.TextIOWrapper(member, encoding="utf-8", newline="")
            header = next(csv.reader(text, delimiter="\t"))
    else:
        with path.open(encoding="utf-8", newline="") as text:
            header = next(csv.reader(text, delimiter="\t"))
    return header[1:]


def _read_samples_table(
    folder: Path, name: str, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """A D2 samples x features table as floats, indexed by sample ID (str)."""
    usecols = None if columns is None else ["Sample", *columns]
    table = pd.read_csv(_d2_file(folder, name), sep="\t", usecols=usecols, dtype={"Sample": str})
    table = table.set_index("Sample").rename_axis(index=SAMPLE_ID, columns=None)
    _require_unique(table.index, f"{folder.name}/{name} sample IDs")
    return table.astype(float)


def _read_metadata(folder: Path, dataset: str) -> pd.DataFrame:
    # IDs are read as text: YACHIDA's numeric IDs must not become integers.
    meta = pd.read_csv(folder / "metadata.tsv", sep="\t", dtype={"Sample": str, "Subject": str})
    for column in ("Sample", "Subject", "Study.Group"):
        if meta[column].isna().any():
            raise DataFactError(f"{dataset}/metadata.tsv: missing {column} values")
    meta = meta.set_index("Sample").rename_axis(index=SAMPLE_ID)
    _require_unique(meta.index, f"{dataset}/metadata.tsv sample IDs")
    return meta.assign(
        subject=meta["Subject"], diagnosis=meta["Study.Group"].astype(str), dataset=dataset
    )


def _read_mtb_map(folder: Path) -> pd.DataFrame:
    mtb_map = pd.read_csv(folder / "mtb.map.tsv", sep="\t", dtype={"Compound": str})
    if mtb_map["Compound"].isna().any():
        raise DataFactError(f"{folder.name}/mtb.map.tsv: missing Compound values")
    mtb_map = mtb_map.set_index("Compound")
    _require_unique(mtb_map.index, f"{folder.name}/mtb.map.tsv compounds")
    return mtb_map


def _load_d2_cohort(
    dataset: str, raw_dir: Path | None, *, with_metabolites: bool, hmdb_only: bool
) -> Cohort:
    folder = _d2_folder(dataset, raw_dir)
    meta = _read_metadata(folder, dataset)
    microbes = _align_rows(_read_samples_table(folder, "genera"), meta.index, "genera")
    metabolites = None
    if with_metabolites:
        mtb = _read_metabolites(folder, dataset, hmdb_only=hmdb_only)
        metabolites = _align_rows(mtb, meta.index, "mtb")
    return Cohort(dataset, microbes, metabolites, meta)


def _read_metabolites(folder: Path, dataset: str, *, hmdb_only: bool) -> pd.DataFrame:
    if dataset in HMDB_ONLY_REQUIRED and not hmdb_only:
        raise ValueError(f"{dataset}'s metabolome is ~82k features wide: pass hmdb_only=True")
    if not hmdb_only:
        return _read_samples_table(folder, "mtb")
    mtb_map = _read_mtb_map(folder)
    hmdb_compounds = mtb_map.index[mtb_map["HMDB"].notna()]
    absent = hmdb_compounds.difference(_table_columns(folder, "mtb"))
    if len(absent):
        raise DataFactError(f"{dataset}: {len(absent)} HMDB-mapped compounds are not in mtb")
    # usecols: the unmapped columns are never parsed (they are ~99% of iHMP's table).
    return _read_samples_table(folder, "mtb", columns=hmdb_compounds.tolist())


def _align_rows(table: pd.DataFrame, samples: pd.Index, label: str) -> pd.DataFrame:
    """``table`` reordered to ``samples``; both must hold exactly the same sample IDs."""
    if set(table.index) != set(samples):
        raise DataFactError(f"{label}: sample IDs differ from metadata.tsv")
    return table.loc[samples]


def _part_ids(samples: Iterable[str], part: str) -> list[str]:
    return [sample for sample in samples if sample.startswith(f"{part}.")]


def _bool_flag(meta: pd.DataFrame, column: str) -> np.ndarray:
    values = meta[column]
    if values.dtype != bool:
        raise DataFactError(f"{column} should be TRUE/FALSE, found dtype {values.dtype}")
    return values.to_numpy()


# --------------------------------------------------------------------------------------------
# D3: MGM resources (JSON/CSV only)
# --------------------------------------------------------------------------------------------


def load_mgm_config(raw_dir: Path | None = None) -> dict[str, Any]:
    """MGM's GPT-2 configuration (general_model/config.json)."""
    with (d3_dir(raw_dir) / MGM_MODEL_DIR / "config.json").open() as fh:
        return json.load(fh)


def load_mgm_phylogeny(raw_dir: Path | None = None) -> pd.DataFrame:
    """MGM's genus vocabulary with the corpus mean/std per genus, indexed by token 'g__<Genus>'."""
    return pd.read_csv(d3_dir(raw_dir) / MGM_RESOURCES / "phylogeny.csv", index_col=0)


def mgm_token(genus: str) -> str:
    """MGM's vocabulary token for a genus name."""
    return f"g__{genus}"


# --------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------


def _require_unique(index: pd.Index, label: str) -> None:
    duplicated = index[index.duplicated()]
    if len(duplicated):
        raise DataFactError(f"{label}: duplicates, e.g. {duplicated[:3].tolist()}")


def _require_same_rows(a: pd.DataFrame, b: pd.DataFrame, label: str) -> None:
    if not a.index.equals(b.index):
        raise DataFactError(f"{label}: sample IDs or their order differ")


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """max_j |a[i, j] - b[k, j]| for every row pair (i, k), one row of ``a`` at a time."""
    rows = [np.abs(b - row).max(axis=1) for row in a]
    return np.array(rows, dtype=float).reshape(len(a), len(b))


def _has_identical_row(a: pd.DataFrame, b: pd.DataFrame, tol: float = IDENTITY_TOL) -> np.ndarray:
    """For each row of ``a``: does ``b`` hold the same profile? Absent columns count as 0."""
    columns = a.columns.union(b.columns)
    a_values = a.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=float)
    b_values = b.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=float)
    return (_max_abs_diff(a_values, b_values) < tol).any(axis=1)


# --------------------------------------------------------------------------------------------
# Section-3 data facts
# --------------------------------------------------------------------------------------------

FACTS_FILE = "data_facts.yaml"
CHECK_PURPOSE = "section-3 data checks (no modelling)"
IHMP = "iHMP_IBDMDB_2019"
MGM_ARCHITECTURE_KEYS = ("model_type", "n_layer", "n_embd", "n_head", "n_positions", "vocab_size")
MGM_FILES = (
    MGM_MODEL_DIR / "pytorch_model.bin",
    MGM_MODEL_DIR / "config.json",
    MGM_RESOURCES / "MicroTokenizer.pkl",
    MGM_RESOURCES / "phylogeny.csv",
)


@dataclass(frozen=True)
class FactResult:
    """One checked fact: what configs/data_facts.yaml expects and what the data shows."""

    key: str
    description: str
    expected: Any
    observed: Any
    passed: bool
    tol: float | None = None
    brief: Any = None
    note: str = ""


class _FactData:
    """Everything the facts look at, loaded on first use."""

    def __init__(self, raw_dir: Path | None, microbe_cfg: Mapping[str, Any]) -> None:
        self.raw_dir = raw_dir
        self.abundance_threshold = float(microbe_cfg["abundance_threshold"])
        self.min_prevalence = float(microbe_cfg["min_prevalence"])
        self._cache: dict[str, tuple[bool, Any]] = {}

    def _once(self, name: str, load: Callable[[], Any]) -> Any:
        # Failures are cached too: a broken guarded loader then runs (and logs an external
        # access) once, however many facts depend on it.
        if name not in self._cache:
            try:
                self._cache[name] = (True, load())
            except Exception as exc:
                self._cache[name] = (False, exc)
        ok, value = self._cache[name]
        if not ok:
            raise value
        return value

    @property
    def prism(self) -> Cohort:
        return self._once("prism", lambda: load_prism(self.raw_dir))

    @property
    def nlibd(self) -> Cohort:
        return self._once(
            "nlibd",
            lambda: load_nlibd(
                self.raw_dir, final_evaluation=True, purpose=CHECK_PURPOSE, data_check=True
            ),
        )

    @property
    def annotation(self) -> pd.DataFrame:
        return self._once("annotation", lambda: load_annotation(self.raw_dir))

    @property
    def t466(self) -> list[str]:
        return self._once("t466", lambda: t466_features(self.raw_dir))

    @property
    def franzosa_prism(self) -> Cohort:
        return self._once("franzosa_prism", lambda: load_franzosa(self.raw_dir, part="PRISM"))

    @property
    def franzosa_validation(self) -> Cohort:
        return self._once(
            "franzosa_validation",
            lambda: load_franzosa(
                self.raw_dir, part="Validation", final_evaluation=True, purpose=CHECK_PURPOSE,
                data_check=True,
            ),
        )

    @property
    def sources(self) -> dict[str, Cohort]:
        return self._once(
            "sources",
            lambda: load_sources(self.raw_dir, drop_duplicates=False, with_metabolites=True),
        )

    @property
    def d2_meta(self) -> dict[str, pd.DataFrame]:
        """Metadata of all six D2 datasets; Franzosa = PRISM + Validation (all 220 samples)."""
        return self._once("d2_meta", lambda: self._all_six(lambda cohort: cohort.meta))

    @property
    def genera(self) -> dict[str, pd.DataFrame]:
        """Genus tables of all six D2 datasets; Franzosa = PRISM + Validation."""
        return self._once("genera", lambda: self._all_six(lambda cohort: cohort.microbes))

    def _all_six(self, table: Callable[[Cohort], pd.DataFrame]) -> dict[str, pd.DataFrame]:
        franzosa = pd.concat([table(self.franzosa_prism), table(self.franzosa_validation)])
        return {TARGET_DATASET: franzosa, **{ds: table(c) for ds, c in self.sources.items()}}

    @property
    def prevalent(self) -> dict[str, set[str]]:
        return self._once("prevalent", self._prevalent)

    def _prevalent(self) -> dict[str, set[str]]:
        return {
            dataset: set(
                preprocess.prevalent_taxa(
                    preprocess.closure(table), self.abundance_threshold, self.min_prevalence
                )
            )
            for dataset, table in self.genera.items()
        }

    @property
    def nlibd_match(self) -> pd.Series:
        return self._once(
            "nlibd_match",
            lambda: match_nlibd_to_validation(
                self.nlibd.metabolites, self.franzosa_validation.metabolites
            ),
        )

    def mtb_map(self, dataset: str) -> pd.DataFrame:
        return self._once(f"mtb_map:{dataset}", lambda: load_d2_mtb_map(dataset, self.raw_dir))

    @property
    def mgm_config(self) -> dict[str, Any]:
        return self._once("mgm_config", lambda: load_mgm_config(self.raw_dir))

    @property
    def mgm_vocabulary(self) -> set[str]:
        return self._once(
            "mgm_vocabulary", lambda: set(load_mgm_phylogeny(self.raw_dir).index)
        )


def _shape(table: pd.DataFrame | None) -> list[int]:
    if table is None:
        raise DataFactError("table not loaded")
    return list(table.shape)


def _value_range(table: pd.DataFrame | None, *, skipna: bool = False) -> list[float]:
    """[min, max] of a table. A missing value turns both into NaN, so the fact fails, unless
    ``skipna`` (for tables whose missing values are expected and counted by a fact of their own).
    """
    values = np.asarray(table, dtype=float)
    low, high = (np.nanmin, np.nanmax) if skipna else (np.min, np.max)
    return [float(low(values)), float(high(values))]


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(label): int(n) for label, n in values.value_counts().items()}


def _id_format(sample_id: str) -> str:
    """'PRISM|7122' -> 'PRISM|#': the ID with every run of digits replaced by '#'."""
    return re.sub(r"\d+", "#", str(sample_id))


def _id_formats(ids: Iterable[str]) -> dict[str, int]:
    """How many IDs follow each format."""
    return dict(sorted(Counter(_id_format(sample) for sample in ids).items()))


def _diagnosis_by_id_format(cohort: Cohort) -> dict[str, dict[str, int]]:
    """Diagnosis counts per ID format (the format reveals the recruitment cohort)."""
    formats = np.array([_id_format(sample) for sample in cohort.sample_ids])
    diagnosis = cohort.meta["diagnosis"]
    return {fmt: _counts(diagnosis[formats == fmt]) for fmt in sorted(set(formats))}


def _validation_id_agreement(d: _FactData) -> dict[str, int]:
    """Where an NLIBD ID also exists in D2 (after '|' -> '.'), does the metabolome match pick it?"""
    d2_ids = set(d.franzosa_validation.sample_ids)
    same_id = {s: s.replace("|", ".") for s in d.nlibd.sample_ids if s.replace("|", ".") in d2_ids}
    return {
        "ids_in_both_releases": len(same_id),
        "matched_to_the_same_id": sum(d.nlibd_match[s] == d2_id for s, d2_id in same_id.items()),
    }


def _same_columns(a: pd.DataFrame | None, b: pd.DataFrame | None) -> bool:
    return a is not None and b is not None and a.columns.equals(b.columns)


def _n_samples(cohorts: Mapping[str, Cohort]) -> int:
    return sum(len(cohort.sample_ids) for cohort in cohorts.values())


def _duplicate_report(d: _FactData) -> dict[str, int]:
    """Does the ERAWIJANTARI flag agree with genus-profile identity to a YACHIDA sample?"""
    cohort, twin = d.sources[DUPLICATED_IN], d.sources[DUPLICATE_KEPT_IN]
    flagged = _bool_flag(cohort.meta, DUPLICATE_FLAG)
    identical = _has_identical_row(cohort.microbes, twin.microbes)
    return {
        "flagged": int(flagged.sum()),
        "flagged_identical_to_yachida": int((flagged & identical).sum()),
        "unflagged_identical_to_yachida": int((~flagged & identical).sum()),
    }


def _source_target_overlap(d: _FactData) -> dict[str, int]:
    target = d.d2_meta[TARGET_DATASET]
    sources = [meta for dataset, meta in d.d2_meta.items() if dataset != TARGET_DATASET]
    source_samples = set().union(*(meta.index for meta in sources))
    source_subjects = set().union(*(meta["subject"] for meta in sources))
    return {
        "samples": len(source_samples & set(target.index)),
        "subjects": len(source_subjects & set(target["subject"])),
    }


def _shared_hmdb(d: _FactData) -> set[str]:
    franzosa, ihmp = d.mtb_map(TARGET_DATASET), d.mtb_map(IHMP)
    return set(franzosa["HMDB"].dropna()) & set(ihmp["HMDB"].dropna())


def _shared_hmdb_inside_t466(d: _FactData) -> int:
    """Shared HMDB IDs whose Franzosa features all belong to T466."""
    franzosa = d.mtb_map(TARGET_DATASET)
    shared = franzosa[franzosa["HMDB"].isin(_shared_hmdb(d))]
    t466 = set(d.t466)
    inside = pd.Series(
        [franzosa_feature_id(compound) in t466 for compound in shared.index],
        index=shared["HMDB"].to_numpy(),
    )
    return int(inside.groupby(level=0).all().sum())


def _prism_metabolome_difference(d: _FactData) -> float:
    """Largest |D1 - D2| over the PRISM metabolome, matching samples by ID and features by name."""
    d1 = d.prism.metabolites
    d2 = d.franzosa_prism.metabolites.loc[[d1_to_d2_prism_id(s) for s in d1.index], d1.columns]
    return float(np.abs(d2.to_numpy() - d1.to_numpy()).max())


def _d1_genera_in_mgm(d: _FactData) -> int:
    genera = set(species_to_genus(d.prism.microbes.columns))
    return sum(mgm_token(genus) in d.mgm_vocabulary for genus in genera)


def _in_mgm_vocabulary(columns: Iterable[str], vocabulary: set[str]) -> list[str]:
    """GTDB genus columns whose suffix-stripped name is an MGM token."""
    return [
        col for col in columns if mgm_token(gtdb_genus(col, strip_suffix=True)) in vocabulary
    ]


def _mgm_d2_name_coverage(d: _FactData) -> float:
    """Fraction of Franzosa's prevalent genera that MGM's vocabulary contains."""
    prevalent = d.prevalent[TARGET_DATASET]
    return len(_in_mgm_vocabulary(prevalent, d.mgm_vocabulary)) / len(prevalent)


def _mgm_d2_abundance_coverage(d: _FactData) -> float:
    """Mean over Franzosa samples of the abundance share in genera MGM's vocabulary contains.

    All genus columns count, not only the prevalent ones: this is the share of each sample's
    abundance that survives MGM's vocabulary filter.
    """
    genera = preprocess.closure(d.genera[TARGET_DATASET])
    covered = _in_mgm_vocabulary(genera.columns, d.mgm_vocabulary)
    return float(genera[covered].sum(axis=1).mean())


def _missing_mgm_files(d: _FactData) -> list[str]:
    root = d3_dir(d.raw_dir)
    return [str(path) for path in MGM_FILES if not (root / path).exists()]


# One entry per fact in configs/data_facts.yaml: fact ID -> how to observe it.
_FACT_CHECKS: dict[str, Callable[[_FactData], Any]] = {
    # D1: MiMeNet IBD tables
    "d1.prism_microbes_shape": lambda d: _shape(d.prism.microbes),
    "d1.prism_metabolites_shape": lambda d: _shape(d.prism.metabolites),
    "d1.prism_metabolite_range": lambda d: _value_range(d.prism.metabolites),
    "d1.prism_id_format": lambda d: _id_formats(d.prism.sample_ids),
    "d1.prism_ibd_status_counts": lambda d: _counts(d.prism.meta["ibd_status"]),
    "d1.prism_diagnosis_counts": lambda d: _counts(d.prism.meta["diagnosis"]),
    "d1.nlibd_microbes_shape": lambda d: _shape(d.nlibd.microbes),
    "d1.nlibd_metabolites_shape": lambda d: _shape(d.nlibd.metabolites),
    "d1.nlibd_metabolite_range": lambda d: _value_range(d.nlibd.metabolites),
    "d1.nlibd_id_format": lambda d: _id_formats(d.nlibd.sample_ids),
    "d1.nlibd_diagnosis_counts": lambda d: _counts(d.nlibd.meta["diagnosis"]),
    "d1.nlibd_diagnosis_by_id_format": lambda d: _diagnosis_by_id_format(d.nlibd),
    "d1.nlibd_same_species_as_prism": lambda d: _same_columns(d.nlibd.microbes, d.prism.microbes),
    "d1.nlibd_same_features_as_prism": lambda d: _same_columns(
        d.nlibd.metabolites, d.prism.metabolites
    ),
    "d1.annotation_shape": lambda d: _shape(d.annotation),
    "d1.annotated_features_in_metabolome": lambda d: len(d.t466),
    "d1.species_first_token_genera": lambda d: len(set(species_to_genus(d.prism.microbes.columns))),
    # D2: curated microbiome-metabolome collection
    "d2.samples": lambda d: {dataset: len(meta) for dataset, meta in d.d2_meta.items()},
    "d2.subjects": lambda d: {ds: int(meta["subject"].nunique()) for ds, meta in d.d2_meta.items()},
    "d2.franzosa_parts": lambda d: {
        "PRISM": len(d.franzosa_prism.sample_ids),
        "Validation": len(d.franzosa_validation.sample_ids),
    },
    "d2.franzosa_prism_id_format": lambda d: _id_formats(d.franzosa_prism.sample_ids),
    "d2.franzosa_validation_id_format": lambda d: _id_formats(d.franzosa_validation.sample_ids),
    "d2.prism_ids_shared_with_d1": lambda d: len(
        {d1_to_d2_prism_id(s) for s in d.prism.sample_ids} & set(d.franzosa_prism.sample_ids)
    ),
    "d2.genera_row_sum_max_deviation": lambda d: max(
        float((table.sum(axis=1) - 1.0).abs().max()) for table in d.genera.values()
    ),
    "d2.prevalent_genera": lambda d: {
        dataset: len(d.prevalent[dataset]) for dataset in (TARGET_DATASET, IHMP)
    },
    "d2.prevalent_genera_all_six": lambda d: len(set.intersection(*d.prevalent.values())),
    "d2.prevalent_genera_union": lambda d: len(set.union(*d.prevalent.values())),
    "d2.prevalent_genera_franzosa_ihmp": lambda d: len(
        d.prevalent[TARGET_DATASET] & d.prevalent[IHMP]
    ),
    "d2.source_metagenomes": lambda d: _n_samples(d.sources),
    "d2.source_duplicates": _duplicate_report,
    "d2.source_unique_metagenomes": lambda d: _n_samples(drop_duplicate_samples(d.sources)),
    "d2.source_target_shared_ids": _source_target_overlap,
    "d2.ihmp_metabolite_features": lambda d: len(
        _table_columns(_d2_folder(IHMP, d.raw_dir), "mtb")
    ),
    "d2.ihmp_hmdb_mapped_features": lambda d: _shape(d.sources[IHMP].metabolites)[1],
    "d2.ihmp_hmdb_nan_fraction": lambda d: float(
        np.isnan(d.sources[IHMP].metabolites.to_numpy(dtype=float)).mean()
    ),
    "d2.ihmp_hmdb_value_range": lambda d: _value_range(d.sources[IHMP].metabolites, skipna=True),
    "d2.hmdb_shared_franzosa_ihmp": lambda d: len(_shared_hmdb(d)),
    "d2.hmdb_shared_inside_t466": _shared_hmdb_inside_t466,
    "d2.franzosa_features_are_d1_features": lambda d: _same_columns(
        d.franzosa_prism.metabolites, d.prism.metabolites
    ),
    "d2.prism_metabolome_max_abs_diff_vs_d1": _prism_metabolome_difference,
    "d2.nlibd_validation_one_to_one_matches": lambda d: len(d.nlibd_match),
    "d2.nlibd_validation_ids_agree": _validation_id_agreement,
    # D3: MGM
    "mgm.files_missing": _missing_mgm_files,
    "mgm.architecture": lambda d: {key: d.mgm_config.get(key) for key in MGM_ARCHITECTURE_KEYS},
    "mgm.phylogeny_genera": lambda d: len(d.mgm_vocabulary),
    "mgm.special_tokens": lambda d: int(d.mgm_config["vocab_size"]) - len(d.mgm_vocabulary),
    "mgm.d1_genera_in_vocabulary": _d1_genera_in_mgm,
    "mgm.d2_prevalent_name_coverage": _mgm_d2_name_coverage,
    "mgm.d2_abundance_coverage": _mgm_d2_abundance_coverage,
}
FACT_KEYS: tuple[str, ...] = tuple(_FACT_CHECKS)


def load_fact_specs(facts_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The expected values of configs/data_facts.yaml, keyed by fact ID."""
    path = Path(facts_path) if facts_path is not None else paths.CONFIGS / FACTS_FILE
    with path.open() as fh:
        document = yaml.safe_load(fh)
    specs = document.get("facts") if isinstance(document, Mapping) else None
    # An empty or mis-keyed file would otherwise check nothing and report success.
    if not isinstance(specs, Mapping) or not specs:
        raise ValueError(f"{path.name}: no facts listed under a top-level 'facts:' mapping")
    invalid = [
        key for key, spec in specs.items() if not isinstance(spec, dict) or "expected" not in spec
    ]
    if invalid:
        raise ValueError(f"{path.name}: facts without an 'expected' value: {invalid}")
    return {str(key): spec for key, spec in specs.items()}


def evaluate_section3_facts(
    raw_dir: Path | None = None, facts_path: Path | None = None
) -> list[FactResult]:
    """Observe every fact listed in the facts file; never raises for a failing fact."""
    specs = load_fact_specs(facts_path)
    microbe_cfg = load_config(paths.CONFIGS / "base.yaml")["microbes"]
    fact_data = _FactData(raw_dir, microbe_cfg)
    return [_evaluate_fact(key, spec, fact_data) for key, spec in specs.items()]


def format_fact_report(results: Sequence[FactResult]) -> list[str]:
    """A markdown table: fact | expected | observed | status | note."""
    lines = ["| Fact | Expected | Observed | Status | Note |", "|---|---|---|---|---|"]
    for result in results:
        expected = _format_value(result.expected)
        if result.tol is not None:
            expected += f" ± {result.tol:g}"
        observed = _format_value(_ordered_like(result.observed, result.expected))
        cells = [result.description, expected, observed, _status(result), _note(result)]
        lines.append("| " + " | ".join(_cell(text) for text in cells) + " |")
    return lines


def check_section3_facts(
    raw_dir: Path | None = None, facts_path: Path | None = None
) -> list[str]:
    """Check every section-3 fact: the markdown report, or DataFactError listing all failures."""
    results = evaluate_section3_facts(raw_dir, facts_path)
    failed = [result for result in results if not result.passed]
    if failed:
        listing = "\n".join(
            f"- {r.key}: expected {_format_value(r.expected)}, observed {_format_value(r.observed)}"
            for r in failed
        )
        raise DataFactError(f"{len(failed)} of {len(results)} data facts failed:\n{listing}")
    return format_fact_report(results)


def _evaluate_fact(key: str, spec: Mapping[str, Any], data: _FactData) -> FactResult:
    expected, tol = spec["expected"], spec.get("tol")
    check = _FACT_CHECKS.get(key)
    if check is None:
        observed, passed = "error: no check is implemented for this fact", False
    else:
        try:
            observed = _plain(check(data))
            passed = _matches(observed, expected, tol)
        except Exception as exc:  # a failing fact must not hide the others
            observed, passed = f"error: {type(exc).__name__}: {exc}", False
    return FactResult(
        key=key,
        description=str(spec.get("description", key)),
        expected=expected,
        observed=observed,
        passed=passed,
        tol=None if tol is None else float(tol),
        brief=spec.get("brief"),
        note=str(spec.get("note", "")),
    )


def _plain(value: Any) -> Any:
    """numpy/pandas values -> plain Python, so they compare equal to YAML values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple | np.ndarray | pd.Index):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _matches(observed: Any, expected: Any, tol: float | None) -> bool:
    """Exact equality, or |observed - expected| <= tol element-wise when a tolerance is given."""
    if tol is None:
        return observed == expected
    if isinstance(expected, Mapping):
        return (
            isinstance(observed, Mapping)
            and observed.keys() == expected.keys()
            and all(_matches(observed[key], value, tol) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(_matches(o, e, tol) for o, e in zip(observed, expected, strict=True))
        )
    try:
        return abs(float(observed) - float(expected)) <= float(tol)
    except (TypeError, ValueError):
        return False


def _ordered_like(observed: Any, expected: Any) -> Any:
    """Show dict keys (also nested ones) in the order the facts file lists them."""
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        keys = [k for k in expected if k in observed] + [k for k in observed if k not in expected]
        return {key: _ordered_like(observed[key], expected.get(key)) for key in keys}
    return observed


def _format_value(value: Any, *, nested: bool = False) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, Mapping):
        items = ", ".join(f"{k}: {_format_value(v, nested=True)}" for k, v in value.items())
        return f"{{{items}}}" if nested else items
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item, nested=True) for item in value) + "]"
    return str(value)


def _status(result: FactResult) -> str:
    if not result.passed:
        return "FAIL"
    return "pass (brief differs)" if result.brief is not None else "pass"


def _note(result: FactResult) -> str:
    parts = []
    if result.brief is not None:
        parts.append(f"brief: {_format_value(result.brief)}.")
    if result.note:
        parts.append(result.note)
    return " ".join(parts)


def _cell(text: str) -> str:
    # A literal '|' (as in D1 sample IDs) would split the markdown cell.
    return text.replace("|", "\\|").replace("\n", " ")
