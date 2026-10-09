"""m2m.data on a tiny synthetic raw-data tree (D1, D2, MGM); data/raw is never touched."""

from __future__ import annotations

import json
import runpy
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from m2m import data, paths

# --- synthetic cohort definitions ------------------------------------------------------------

PRISM_IDS = ["PRISM|7122", "PRISM|7147", "PRISM|7150", "PRISM|7153"]
PRISM_IBD_STATUS = ["IBD", "IBD", "Healthy", "IBD"]  # D1 diagnosis_PRISM.csv
PRISM_DIAGNOSIS = ["CD", "UC", "Control", "CD"]  # D2 Study.Group of PRISM.<x>
NLIBD_IDS = ["Validation|LLDeep_0001", "Validation|UMCGIBD00002"]
NLIBD_DIAGNOSIS = ["Control", "CD"]
# D2 names the IBD patient differently and lists the two samples in the other order.
VALIDATION_IDS = ["Validation.UMCG9189151", "Validation.LLDeep_0001"]
VALIDATION_DIAGNOSIS = ["CD", "Control"]
EXPECTED_MATCH = {
    "Validation|LLDeep_0001": "Validation.LLDeep_0001",
    "Validation|UMCGIBD00002": "Validation.UMCG9189151",
}

SPECIES = [
    "Bacteroides_vulgatus",
    "Faecalibacterium_prausnitzii",
    "Lachnospiraceae_bacterium_1_1_57FAA",
]
FEATURES = ["C18-neg_Cluster_0001", "C18-neg_Cluster_0002", "HILIC-pos_Cluster_0003"]
FRANZOSA_COMPOUNDS = [
    "C18-neg_Cluster_0001: NA",
    "C18-neg_Cluster_0002: NA",
    "HILIC-pos_Cluster_0003: glycine",
]
ANNOTATED = ["HILIC-pos_Cluster_0003", "C18-neg_Cluster_0001"]  # sorted by name, not file order
PRISM_METABOLOME = np.array(
    [[0.0, 12.5, 300.0], [5.0, 0.0, 1000.0], [7.5, 3.0, 0.0], [1.0, 2.0, 3.0]]
)
NLIBD_METABOLOME = np.array([[10.0, 0.0, 20.0], [0.0, 30.0, 40.0]])

TAXONOMY = "d__Bacteria;p__Firmicutes_A;c__Clostridia;o__Oscillospirales;f__Ruminococcaceae"
GENERA = [f"{TAXONOMY};g__{genus}" for genus in ("Faecalibacterium_A", "CAG-83", "Bacteroides")]
FAECALIBACTERIUM_A, CAG_83, BACTEROIDES = GENERA
# Fixed genus shares that make the prevalence and MGM-coverage facts hand-computable:
FRANZOSA_CAG_83_SHARE = 0.25  # CAG-83 is the one genus MGM lacks -> abundance coverage 0.75
IHMP_CAG_83_SHARE = 5e-5  # present, but below the 1e-4 prevalence threshold

# iHMP's first mapped compound shares glycine's HMDB ID with Franzosa (a T466 feature), the
# second shares HMDB0004705 with a Franzosa feature outside T466.
IHMP_HMDB = ["HILp_TF14__glycine", "C18n_QI06__12.13-diHOME"]
IHMP_UNMAPPED = "C18n_QI99__unknown"
WANG_UNMAPPED = "Cyclotrisiloxane, hexamethyl-"


# --- writers ---------------------------------------------------------------------------------


def _write_d1_table(path: Path, table: pd.DataFrame) -> None:
    """Stored like D1: features x samples, with MiMeNet's corner label."""
    table.T.rename_axis("# Feature / Sample").to_csv(path)


def _write_d1_labels(path: Path, ids: list[str], labels: list[str]) -> None:
    pd.DataFrame({"0": labels}, index=ids).to_csv(path)


def _write_tsv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def _write_zipped_tsv(path: Path, table: pd.DataFrame) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(path.name.removesuffix(".zip"), table.to_csv(sep="\t", index=False))


def _genera(ids: list[str], rng: np.random.Generator) -> pd.DataFrame:
    values = rng.uniform(0.1, 1.0, size=(len(ids), len(GENERA)))
    table = pd.DataFrame(values / values.sum(axis=1, keepdims=True), columns=GENERA)
    return table.assign(Sample=ids)[["Sample", *GENERA]]


def _with_share(table: pd.DataFrame, genus: str, share: float) -> pd.DataFrame:
    """``genus`` gets the same share in every sample; the other genera fill the rest (sum 1)."""
    others = [column for column in GENERA if column != genus]
    rest = table[others].div(table[others].sum(axis=1), axis=0) * (1.0 - share)
    return table.assign(**{genus: share}).assign(**{col: rest[col] for col in others})


def _metadata(
    dataset: str, ids: list[str], subjects: list, groups: list[str], **extra: list
) -> pd.DataFrame:
    columns = {"Dataset": dataset, "Sample": ids, "Subject": subjects, "Study.Group": groups}
    return pd.DataFrame({**columns, **extra})


def _write_d1(folder: Path, rng: np.random.Generator) -> None:
    folder.mkdir(parents=True)
    prism_microbes = pd.DataFrame(rng.uniform(0, 0.3, (4, 3)), index=PRISM_IDS, columns=SPECIES)
    nlibd_microbes = pd.DataFrame(rng.uniform(0, 0.3, (2, 3)), index=NLIBD_IDS, columns=SPECIES)
    _write_d1_table(folder / "microbiome_PRISM.csv", prism_microbes)
    _write_d1_table(folder / "microbiome_external.csv", nlibd_microbes)
    _write_d1_table(
        folder / "metabolome_PRISM.csv",
        pd.DataFrame(PRISM_METABOLOME, index=PRISM_IDS, columns=FEATURES),
    )
    _write_d1_table(
        folder / "metabolome_external.csv",
        pd.DataFrame(NLIBD_METABOLOME, index=NLIBD_IDS, columns=FEATURES),
    )
    _write_d1_labels(folder / "diagnosis_PRISM.csv", PRISM_IDS, PRISM_IBD_STATUS)
    _write_d1_labels(folder / "external_diagnosis.csv", NLIBD_IDS, NLIBD_DIAGNOSIS)
    annotation = pd.DataFrame(
        {
            "Metabolomic Feature": ANNOTATED,
            "Retention Time": [3.8, 1.2],
            "m/z": [76.04, 259.07],
            "Compound Name": ["glycine", "a lipid"],
            "Adduct": ["[M+H]+", "[M-H]-"],
        }
    )
    annotation.to_csv(folder / "metabolome_annotation.csv", index=False)


def _write_franzosa(folder: Path, rng: np.random.Generator) -> None:
    prism = [d1_id.replace("|", ".") for d1_id in PRISM_IDS]
    ids = prism + VALIDATION_IDS
    _write_tsv(
        folder / "metadata.tsv",
        _metadata(
            data.TARGET_DATASET, ids, ids, PRISM_DIAGNOSIS + VALIDATION_DIAGNOSIS,
            Age=[30, 41, 52, 63, 25, 36],
        ),
    )
    # Rows in a different order from metadata.tsv, as in several real tables.
    genera = _with_share(_genera(ids[::-1], rng), CAG_83, FRANZOSA_CAG_83_SHARE)
    _write_tsv(folder / "genera.tsv", genera)
    metabolome = np.vstack([PRISM_METABOLOME, NLIBD_METABOLOME[[1, 0]]])
    mtb = pd.DataFrame(metabolome, columns=FRANZOSA_COMPOUNDS).assign(Sample=ids)
    _write_tsv(folder / "mtb.tsv", mtb[["Sample", *FRANZOSA_COMPOUNDS]])
    mtb_map = pd.DataFrame(
        {"Compound": FRANZOSA_COMPOUNDS, "HMDB": [None, "HMDB0004705", "HMDB0000123"]}
    )
    _write_tsv(folder / "mtb.map.tsv", mtb_map)


def _write_source(
    folder: Path,
    meta: pd.DataFrame,
    genera: pd.DataFrame,
    mtb: pd.DataFrame,
    hmdb: dict[str, str | None],
    *,
    zipped_mtb: bool = False,
) -> None:
    _write_tsv(folder / "metadata.tsv", meta)
    _write_tsv(folder / "genera.tsv", genera)
    if zipped_mtb:
        _write_zipped_tsv(folder / "mtb.tsv.zip", mtb)
    else:
        _write_tsv(folder / "mtb.tsv", mtb)
    mtb_map = pd.DataFrame({"Compound": list(hmdb), "HMDB": list(hmdb.values())})
    _write_tsv(folder / "mtb.map.tsv", mtb_map)


def _write_sources(root: Path, rng: np.random.Generator) -> None:
    ihmp = ["CSM5MCVN", "CSM5MCWE", "CSM5MCX3"]
    ihmp_mtb = pd.DataFrame(
        {
            "Sample": ihmp,
            IHMP_HMDB[0]: [5.0, np.nan, 7.0],
            # Text in an unmapped column: parsing it would fail, so it proves usecols skipped it.
            IHMP_UNMAPPED: ["junk", "junk", "junk"],
            IHMP_HMDB[1]: [1.0e9, 2.0e9, np.nan],
        }
    )
    _write_source(
        root / "iHMP_IBDMDB_2019",
        _metadata("iHMP_IBDMDB_2019", ihmp, ["C3002", "C3002", "C3009"], ["CD", "CD", "nonIBD"]),
        _with_share(_genera(ihmp, rng), CAG_83, IHMP_CAG_83_SHARE),
        ihmp_mtb,
        {IHMP_HMDB[0]: "HMDB0000123", IHMP_UNMAPPED: None, IHMP_HMDB[1]: "HMDB0004705"},
        zipped_mtb=True,
    )

    yachida = ["10025", "10030", "10040"]
    yachida_genera = _genera(yachida, rng)
    _write_source(
        root / "YACHIDA_CRC_2019",
        _metadata(
            "YACHIDA_CRC_2019", yachida, [10025, 10030, 10040], ["Healthy"] * 3,
            **{"Shared.w.ERAWIJANTARI_2020": ["TRUE", "FALSE", "FALSE"]},
        ),
        yachida_genera,
        pd.DataFrame({"Sample": yachida, "C00024_Acetyl CoA": [1.0, 2.0, 3.0]}),
        {"C00024_Acetyl CoA": "HMDB0001206"},
    )

    # ERAWIJANTARI's first sample is YACHIDA's 10025, published again (identical profile).
    erawijantari = ["10025.Healthy", "10050.Gastrectomy"]
    erawijantari_genera = _genera(erawijantari, rng)
    erawijantari_genera.loc[0, GENERA] = yachida_genera.loc[0, GENERA].to_numpy()
    _write_source(
        root / "ERAWIJANTARI_GASTRIC_CANCER_2020",
        _metadata(
            "ERAWIJANTARI_GASTRIC_CANCER_2020", erawijantari, erawijantari,
            ["Healthy", "Gastrectomy"], **{"Shared.w.YACHIDA_2019": ["TRUE", "FALSE"]},
        ),
        erawijantari_genera,
        pd.DataFrame({"Sample": erawijantari, "-_2-Hydroxyisobutyrate": [0.5, 0.7]}),
        {"-_2-Hydroxyisobutyrate": "HMDB0000729"},
    )

    wang = ["CON-001", "ESRD-001"]
    _write_source(
        root / "WANG_ESRD_2020",
        _metadata("WANG_ESRD_2020", wang, wang, ["Control", "ESRD"]),
        _genera(wang[::-1], rng),  # stored in the other order: the loader must realign
        pd.DataFrame({"Sample": wang, "Benzaldehyde": [1.0, 2.0], WANG_UNMAPPED: ["x", "y"]}),
        {"Benzaldehyde": "HMDB0006115", WANG_UNMAPPED: None},
    )

    mars = ["1.T.0", "1.T.1"]
    _write_source(
        root / "MARS_IBS_2020",
        _metadata("MARS_IBS_2020", mars, [10007539, 10007539], ["D", "D"]),
        _with_share(_genera(mars, rng), BACTEROIDES, 0.0),  # absent -> not prevalent here
        pd.DataFrame({"Sample": mars, "NMR_Lactate": [3.0, 4.0]}),
        {"NMR_Lactate": "HMDB0000190"},
    )


def _write_mgm(folder: Path) -> None:
    resources = folder / data.MGM_RESOURCES
    model = folder / data.MGM_MODEL_DIR
    model.mkdir(parents=True)
    config = {"model_type": "gpt2", "n_layer": 2, "n_embd": 8, "n_head": 2, "n_positions": 16,
              "vocab_size": 6}
    (model / "config.json").write_text(json.dumps(config))
    (model / "pytorch_model.bin").write_bytes(b"")  # presence only; never loaded
    (resources / "MicroTokenizer.pkl").write_bytes(b"")
    phylogeny = pd.DataFrame(
        {"mean": [0.1, 0.2], "std": [0.01, 0.02]},
        index=pd.Index(["g__Bacteroides", "g__Faecalibacterium"], name="#SampleID"),
    )
    phylogeny.to_csv(resources / "phylogeny.csv")


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """A synthetic data/raw with the real folder layout and file formats."""
    rng = np.random.default_rng(0)
    root = tmp_path / "raw"
    _write_d1(data.d1_dir(root), rng)
    d2 = data.d2_dir(root)
    _write_franzosa(d2 / data.TARGET_DATASET, rng)
    _write_sources(d2, rng)
    _write_mgm(data.d3_dir(root))
    return root


def _access_log(results: Path) -> pd.DataFrame:
    return pd.read_csv(results / "external_access_log.csv")


def _written_genera(raw_root: Path, dataset: str) -> pd.DataFrame:
    """A genera.tsv exactly as the fixture stored it (file row order), indexed by Sample."""
    path = data.d2_dir(raw_root) / dataset / "genera.tsv"
    return pd.read_csv(path, sep="\t", dtype={"Sample": str}).set_index("Sample")


# --- D1 --------------------------------------------------------------------------------------


def test_load_prism_transposes_and_takes_three_class_diagnosis_from_d2(raw_root: Path) -> None:
    prism = data.load_prism(raw_root)

    assert prism.name == "PRISM"
    assert prism.microbes.shape == (4, 3) and prism.metabolites.shape == (4, 3)
    assert prism.sample_ids == PRISM_IDS
    assert prism.microbes.index.name == data.SAMPLE_ID
    assert prism.microbes.columns.tolist() == SPECIES
    np.testing.assert_array_equal(prism.metabolites.to_numpy(), PRISM_METABOLOME)
    assert prism.meta["diagnosis"].tolist() == PRISM_DIAGNOSIS
    assert prism.meta["ibd_status"].tolist() == PRISM_IBD_STATUS
    assert prism.meta["subject"].tolist() == PRISM_IDS
    assert set(prism.meta["dataset"]) == {"D1"}


def test_load_prism_rejects_d2_labels_that_contradict_d1(raw_root: Path) -> None:
    path = data.d2_dir(raw_root) / data.TARGET_DATASET / "metadata.tsv"
    meta = pd.read_csv(path, sep="\t")
    meta.loc[meta["Sample"] == "PRISM.7150", "Study.Group"] = "UC"  # D1 says Healthy
    meta.to_csv(path, sep="\t", index=False)

    with pytest.raises(data.DataFactError, match="contradicts"):
        data.load_prism(raw_root)


def test_unknown_or_missing_labels_are_data_fact_errors(
    raw_root: Path, results_dir: Path, frozen: list[str]
) -> None:
    folder = data.d1_dir(raw_root)
    _write_d1_labels(folder / "external_diagnosis.csv", NLIBD_IDS, ["control", "CD"])
    with pytest.raises(data.DataFactError, match="control"):
        data.load_nlibd(raw_root, final_evaluation=True, purpose="unit test")

    path = data.d2_dir(raw_root) / data.TARGET_DATASET / "metadata.tsv"
    meta = pd.read_csv(path, sep="\t")
    meta[meta["Sample"] != "PRISM.7122"].to_csv(path, sep="\t", index=False)
    with pytest.raises(data.DataFactError, match="<missing>"):
        data.load_prism(raw_root)


def test_loaders_default_to_paths_data_raw_at_call_time(
    raw_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "DATA_RAW", raw_root)
    assert data.load_prism().sample_ids == PRISM_IDS


def test_load_nlibd_is_refused_without_final_evaluation(raw_root: Path, results_dir: Path) -> None:
    with pytest.raises(data.ExternalTestSetAccessError):
        data.load_nlibd(raw_root)
    with pytest.raises(data.ExternalTestSetAccessError):
        data.load_nlibd(raw_root, final_evaluation="yes", purpose="truthy is not enough")
    assert not (results_dir / "external_access_log.csv").exists()


def test_load_nlibd_logs_every_allowed_access(
    raw_root: Path, results_dir: Path, frozen: list[str]
) -> None:
    nlibd = data.load_nlibd(raw_root, final_evaluation=True, purpose="final ridge evaluation")
    data.load_nlibd(raw_root, final_evaluation=True, purpose="final MLP evaluation")
    with pytest.raises(data.ExternalTestSetAccessError, match="purpose"):
        data.load_nlibd(raw_root, final_evaluation=True, purpose="  ")

    assert nlibd.sample_ids == NLIBD_IDS
    assert nlibd.meta["diagnosis"].tolist() == NLIBD_DIAGNOSIS
    assert nlibd.meta["ibd_status"].tolist() == ["Healthy", "IBD"]
    np.testing.assert_array_equal(nlibd.metabolites.to_numpy(), NLIBD_METABOLOME)
    log = _access_log(results_dir)
    assert log["purpose"].tolist() == ["final ridge evaluation", "final MLP evaluation"]
    assert set(log["detail"]) == {"D1 NLIBD"}


def test_annotation_and_t466_follow_metabolome_order(raw_root: Path) -> None:
    annotation = data.load_annotation(raw_root)
    assert annotation.shape == (2, 4)
    assert annotation.columns.tolist() == ["Retention Time", "m/z", "Compound Name", "Adduct"]
    assert data.t466_features(raw_root) == ["C18-neg_Cluster_0001", "HILIC-pos_Cluster_0003"]


def test_species_to_genus_takes_the_first_name_token() -> None:
    names = ["Lachnospiraceae_bacterium_1_1_57FAA", "Bacteroides_vulgatus", "Akkermansia"]
    assert data.species_to_genus(names) == ["Lachnospiraceae", "Bacteroides", "Akkermansia"]


def test_d1_to_d2_prism_id() -> None:
    assert data.d1_to_d2_prism_id("PRISM|7122") == "PRISM.7122"
    with pytest.raises(ValueError):
        data.d1_to_d2_prism_id("Validation|LLDeep_0001")


# --- D2 --------------------------------------------------------------------------------------


def test_load_d2_metadata_keeps_ids_as_text_and_hides_validation(raw_root: Path) -> None:
    yachida = data.load_d2_metadata("YACHIDA_CRC_2019", raw_root)
    assert yachida.index.tolist() == ["10025", "10030", "10040"]  # not integers
    assert yachida["subject"].tolist() == ["10025", "10030", "10040"]
    assert {"Subject", "Study.Group", "Shared.w.ERAWIJANTARI_2020"} <= set(yachida.columns)
    assert set(yachida["dataset"]) == {"YACHIDA_CRC_2019"}

    mars = data.load_d2_metadata("MARS_IBS_2020", raw_root)
    assert mars["subject"].tolist() == ["10007539", "10007539"]

    franzosa = data.load_d2_metadata(data.TARGET_DATASET, raw_root)
    assert franzosa.index.tolist() == [d1_id.replace("|", ".") for d1_id in PRISM_IDS]


def test_load_d2_refuses_the_target_and_unknown_datasets(raw_root: Path) -> None:
    with pytest.raises(data.ExternalTestSetAccessError):
        data.load_d2(data.TARGET_DATASET, raw_root)
    with pytest.raises(ValueError, match="unknown"):
        data.load_d2("NOT_A_DATASET", raw_root)


def test_load_d2_aligns_genera_rows_to_metadata(raw_root: Path) -> None:
    wang = data.load_d2("WANG_ESRD_2020", raw_root)
    written = _written_genera(raw_root, "WANG_ESRD_2020")

    assert written.index.tolist() == ["ESRD-001", "CON-001"]  # file order differs from metadata
    assert wang.sample_ids == ["CON-001", "ESRD-001"]
    assert wang.metabolites is None
    assert wang.microbes.columns.tolist() == GENERA
    # Values must travel with their sample ID, not just the index labels.
    pd.testing.assert_frame_equal(wang.microbes, written.loc[wang.sample_ids], check_names=False)
    np.testing.assert_allclose(wang.microbes.sum(axis=1), 1.0)


def test_hmdb_only_reads_mapped_columns_of_a_zipped_table_and_keeps_nan(raw_root: Path) -> None:
    ihmp = data.load_d2("iHMP_IBDMDB_2019", raw_root, with_metabolites=True, hmdb_only=True)

    assert sorted(ihmp.metabolites.columns) == sorted(IHMP_HMDB)
    assert ihmp.metabolites.isna().sum().sum() == 2
    assert ihmp.metabolites.dtypes.eq(float).all()
    with pytest.raises(ValueError, match="hmdb_only"):
        data.load_d2("iHMP_IBDMDB_2019", raw_root, with_metabolites=True)


def test_hmdb_only_never_parses_unmapped_columns(raw_root: Path) -> None:
    # WANG's unmapped column holds text: reading it would fail the float conversion.
    wang = data.load_d2("WANG_ESRD_2020", raw_root, with_metabolites=True, hmdb_only=True)
    assert wang.metabolites.columns.tolist() == ["Benzaldehyde"]
    with pytest.raises(ValueError):
        data.load_d2("WANG_ESRD_2020", raw_root, with_metabolites=True, hmdb_only=False)


def test_load_d2_mtb_map_is_indexed_by_compound(raw_root: Path) -> None:
    mtb_map = data.load_d2_mtb_map(data.TARGET_DATASET, raw_root)
    assert mtb_map.index.tolist() == FRANZOSA_COMPOUNDS
    assert mtb_map["HMDB"].notna().sum() == 2


def test_load_franzosa_prism_uses_d1_feature_ids(raw_root: Path) -> None:
    prism = data.load_franzosa(raw_root, part="PRISM")

    assert prism.name == "FRANZOSA_PRISM"
    assert prism.sample_ids == [d1_id.replace("|", ".") for d1_id in PRISM_IDS]
    assert prism.metabolites.columns.tolist() == FEATURES
    np.testing.assert_array_equal(prism.metabolites.to_numpy(), PRISM_METABOLOME)
    # genera.tsv is stored in reverse order: each sample keeps its own genus profile.
    written = _written_genera(raw_root, data.TARGET_DATASET)
    pd.testing.assert_frame_equal(prism.microbes, written.loc[prism.sample_ids], check_names=False)
    assert set(prism.meta["dataset"]) == {data.TARGET_DATASET}
    with pytest.raises(ValueError):
        data.load_franzosa(raw_root, part="NLIBD")


def test_load_franzosa_validation_is_guarded_and_logged(
    raw_root: Path, results_dir: Path, frozen: list[str]
) -> None:
    with pytest.raises(data.ExternalTestSetAccessError):
        data.load_franzosa(raw_root, part="Validation")
    assert not (results_dir / "external_access_log.csv").exists()

    validation = data.load_franzosa(
        raw_root, part="Validation", final_evaluation=True, purpose="unit test"
    )
    assert validation.sample_ids == VALIDATION_IDS
    log = _access_log(results_dir)
    assert log[["purpose", "detail"]].to_numpy().tolist() == [
        ["unit test", "D2 Franzosa Validation"]
    ]


def test_load_sources_drops_flagged_duplicates(raw_root: Path) -> None:
    deduplicated = data.load_sources(raw_root)
    everything = data.load_sources(raw_root, drop_duplicates=False)

    assert list(deduplicated) == list(data.SOURCE_DATASETS)
    erawijantari = "ERAWIJANTARI_GASTRIC_CANCER_2020"
    assert everything[erawijantari].sample_ids == ["10025.Healthy", "10050.Gastrectomy"]
    assert deduplicated[erawijantari].sample_ids == ["10050.Gastrectomy"]
    assert deduplicated["YACHIDA_CRC_2019"].sample_ids == ["10025", "10030", "10040"]
    # Without its twin dataset nothing is duplicated, so nothing is dropped.
    alone = data.drop_duplicate_samples({erawijantari: everything[erawijantari]})
    assert alone[erawijantari].sample_ids == ["10025.Healthy", "10050.Gastrectomy"]


def test_load_sources_with_metabolites_restricts_to_hmdb(raw_root: Path) -> None:
    sources = data.load_sources(raw_root, with_metabolites=True)
    assert sources["WANG_ESRD_2020"].metabolites.columns.tolist() == ["Benzaldehyde"]
    assert sources["iHMP_IBDMDB_2019"].metabolites.shape == (3, 2)


@pytest.mark.parametrize(
    ("column", "genus", "stripped"),
    [
        (f"{TAXONOMY};g__Faecalibacterium_A", "Faecalibacterium_A", "Faecalibacterium"),
        (f"{TAXONOMY};g__Clostridium_AQ", "Clostridium_AQ", "Clostridium"),
        (f"{TAXONOMY};g__CAG-83", "CAG-83", "CAG-83"),
        (f"{TAXONOMY};g__UBA1775", "UBA1775", "UBA1775"),
        ("g__Bacteroides", "Bacteroides", "Bacteroides"),
    ],
)
def test_gtdb_genus(column: str, genus: str, stripped: str) -> None:
    assert data.gtdb_genus(column) == genus
    assert data.gtdb_genus(column, strip_suffix=True) == stripped


@pytest.mark.parametrize("column", [TAXONOMY, f"{TAXONOMY};g__"])
def test_gtdb_genus_requires_a_genus(column: str) -> None:
    with pytest.raises(ValueError):
        data.gtdb_genus(column)


# --- NLIBD <-> Validation matching -----------------------------------------------------------


def _metabolomes(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    d1 = pd.DataFrame(
        rng.lognormal(3.0, 2.0, size=(4, 6)),
        index=[f"Validation|LLDeep_{i:04d}" for i in range(4)],
        columns=[f"f{j}" for j in range(6)],
    )
    order = [2, 0, 3, 1]
    d2 = pd.DataFrame(d1.to_numpy()[order], index=[f"Validation.UMCG{i}" for i in order],
                      columns=d1.columns)
    return d1, d2


def test_match_recovers_permuted_samples_despite_text_round_off(rng: np.random.Generator) -> None:
    d1, d2 = _metabolomes(rng)
    d2 = d2[d2.columns[::-1]] * (1 + 1e-12)  # other column order, float round-off

    match = data.match_nlibd_to_validation(d1, d2)

    expected = {f"Validation|LLDeep_{i:04d}": f"Validation.UMCG{i}" for i in range(4)}
    assert match.to_dict() == expected
    assert match.index.name == "nlibd_id" and match.name == "validation_id"


def test_match_rejects_ambiguous_missing_and_shared_matches(rng: np.random.Generator) -> None:
    d1, d2 = _metabolomes(rng)
    ambiguous = pd.concat([d2, d2.iloc[[0]].rename(index=lambda s: s + "_copy")])
    with pytest.raises(data.DataFactError, match="several"):
        data.match_nlibd_to_validation(d1, ambiguous)

    missing = d2.copy()
    missing.iloc[0] = missing.iloc[0] + 1.0
    with pytest.raises(data.DataFactError, match="match no"):
        data.match_nlibd_to_validation(d1, missing)

    twins = d1.copy()
    twins.iloc[1] = twins.iloc[0]  # two D1 samples, one D2 sample
    with pytest.raises(data.DataFactError, match="same D2 sample"):
        data.match_nlibd_to_validation(twins, d2.drop(index="Validation.UMCG1"))

    with pytest.raises(data.DataFactError, match="features"):
        data.match_nlibd_to_validation(d1, d2.drop(columns="f0"))


def test_match_on_the_synthetic_tree(
    raw_root: Path, results_dir: Path, frozen: list[str]
) -> None:
    nlibd = data.load_nlibd(raw_root, final_evaluation=True, purpose="unit test")
    validation = data.load_franzosa(
        raw_root, part="Validation", final_evaluation=True, purpose="unit test"
    )
    match = data.match_nlibd_to_validation(nlibd.metabolites, validation.metabolites)
    assert match.to_dict() == EXPECTED_MATCH


# --- Cohort ----------------------------------------------------------------------------------


def _cohort(with_metabolites: bool = True) -> data.Cohort:
    ids = pd.Index(["a", "b", "c"], name=data.SAMPLE_ID)
    microbes = pd.DataFrame({"t1": [0.1, 0.2, 0.3]}, index=ids)
    metabolites = pd.DataFrame({"m1": [1.0, 2.0, 3.0]}, index=ids) if with_metabolites else None
    meta = pd.DataFrame(
        {"subject": ["s1", "s1", "s2"], "diagnosis": ["CD", "CD", "UC"], "dataset": "X"},
        index=ids,
    )
    return data.Cohort("X", microbes, metabolites, meta)


def test_cohort_subset_keeps_the_given_order() -> None:
    subset = _cohort().subset(["c", "a"])
    assert subset.sample_ids == ["c", "a"]
    assert subset.microbes["t1"].tolist() == [0.3, 0.1]
    assert subset.metabolites["m1"].tolist() == [3.0, 1.0]
    assert subset.meta["subject"].tolist() == ["s2", "s1"]
    assert _cohort(with_metabolites=False).subset(["b"]).metabolites is None
    with pytest.raises(KeyError):
        _cohort().subset(["a", "zz"])


def test_cohort_validates_its_rows_and_meta_columns() -> None:
    cohort = _cohort()
    with pytest.raises(ValueError, match="rows"):
        data.Cohort("X", cohort.microbes.iloc[::-1], cohort.metabolites, cohort.meta)
    with pytest.raises(ValueError, match="rows"):
        data.Cohort("X", cohort.microbes, cohort.metabolites.iloc[:2], cohort.meta)
    with pytest.raises(ValueError, match="columns"):
        data.Cohort("X", cohort.microbes, None, cohort.meta.drop(columns="subject"))
    with pytest.raises(ValueError, match="duplicate"):
        cohort.subset(["a", "a"])
    assert "microbes=3x1" in repr(cohort)


# --- MGM resources ---------------------------------------------------------------------------


def test_mgm_resources_are_read_as_json_and_csv(raw_root: Path) -> None:
    assert data.load_mgm_config(raw_root)["n_layer"] == 2
    phylogeny = data.load_mgm_phylogeny(raw_root)
    assert phylogeny.index.tolist() == ["g__Bacteroides", "g__Faecalibacterium"]
    assert data.mgm_token("Bacteroides") in phylogeny.index


# --- section-3 checks ------------------------------------------------------------------------

FRANZOSA, IHMP = data.TARGET_DATASET, "iHMP_IBDMDB_2019"
YACHIDA, WANG, MARS = "YACHIDA_CRC_2019", "WANG_ESRD_2020", "MARS_IBS_2020"
ERAWIJANTARI = "ERAWIJANTARI_GASTRIC_CANCER_2020"

# Every implemented check, with the value worked out by hand from the fixture above, so that a
# check returning a constant or looking at the wrong table fails here (CI has no real data).
SYNTHETIC_FACTS = {
    # D1
    "d1.prism_microbes_shape": {"expected": [4, 3]},
    "d1.prism_metabolites_shape": {"expected": [4, 3]},
    "d1.prism_metabolite_range": {"expected": [0.0, 1000.0], "tol": 1e-9},
    "d1.prism_id_format": {"expected": {"PRISM|#": 4}},
    "d1.prism_ibd_status_counts": {"expected": {"IBD": 3, "Healthy": 1}},
    "d1.prism_diagnosis_counts": {"expected": {"CD": 2, "UC": 1, "Control": 1}},
    "d1.nlibd_microbes_shape": {"expected": [2, 3]},
    "d1.nlibd_metabolites_shape": {"expected": [2, 3]},
    "d1.nlibd_metabolite_range": {"expected": [0.0, 40.0], "tol": 1e-9},
    "d1.nlibd_id_format": {"expected": {"Validation|LLDeep_#": 1, "Validation|UMCGIBD#": 1}},
    "d1.nlibd_diagnosis_counts": {"expected": {"Control": 1, "CD": 1}},
    "d1.nlibd_diagnosis_by_id_format": {
        "expected": {"Validation|LLDeep_#": {"Control": 1}, "Validation|UMCGIBD#": {"CD": 1}}
    },
    "d1.nlibd_same_species_as_prism": {"expected": True},
    "d1.nlibd_same_features_as_prism": {"expected": True},
    "d1.annotation_shape": {"expected": [2, 4]},
    "d1.annotated_features_in_metabolome": {"expected": 2},
    "d1.species_first_token_genera": {"expected": 3},
    # D2
    "d2.samples": {
        "expected": {FRANZOSA: 6, IHMP: 3, YACHIDA: 3, WANG: 2, ERAWIJANTARI: 2, MARS: 2}
    },
    "d2.subjects": {
        "expected": {FRANZOSA: 6, IHMP: 2, YACHIDA: 3, WANG: 2, ERAWIJANTARI: 2, MARS: 1}
    },
    "d2.franzosa_parts": {"expected": {"PRISM": 4, "Validation": 2}},
    "d2.franzosa_prism_id_format": {"expected": {"PRISM.#": 4}},
    "d2.franzosa_validation_id_format": {
        "expected": {"Validation.LLDeep_#": 1, "Validation.UMCG#": 1}
    },
    "d2.prism_ids_shared_with_d1": {"expected": 4},
    "d2.genera_row_sum_max_deviation": {"expected": 0.0, "tol": 1e-9},
    # CAG-83 is below 1e-4 in iHMP and Bacteroides absent from MARS; all else is prevalent.
    "d2.prevalent_genera": {"expected": {FRANZOSA: 3, IHMP: 2}},
    "d2.prevalent_genera_all_six": {"expected": 1},
    "d2.prevalent_genera_union": {"expected": 3},
    "d2.prevalent_genera_franzosa_ihmp": {"expected": 2},
    "d2.source_metagenomes": {"expected": 12},
    "d2.source_duplicates": {
        "expected": {
            "flagged": 1, "flagged_identical_to_yachida": 1, "unflagged_identical_to_yachida": 0
        }
    },
    "d2.source_unique_metagenomes": {"expected": 11},
    "d2.source_target_shared_ids": {"expected": {"samples": 0, "subjects": 0}},
    "d2.ihmp_metabolite_features": {"expected": 3},
    "d2.ihmp_hmdb_mapped_features": {"expected": 2},
    "d2.ihmp_hmdb_nan_fraction": {"expected": 2 / 6, "tol": 1e-9},
    "d2.ihmp_hmdb_value_range": {"expected": [5.0, 2.0e9], "tol": 1e-6},
    "d2.hmdb_shared_franzosa_ihmp": {"expected": 2},  # glycine and HMDB0004705
    "d2.hmdb_shared_inside_t466": {"expected": 1},  # glycine only
    "d2.franzosa_features_are_d1_features": {"expected": True},
    "d2.prism_metabolome_max_abs_diff_vs_d1": {"expected": 0.0, "tol": 1e-6},
    "d2.nlibd_validation_one_to_one_matches": {"expected": 2},
    "d2.nlibd_validation_ids_agree": {
        "expected": {"ids_in_both_releases": 1, "matched_to_the_same_id": 1}
    },
    # D3: MGM
    "mgm.files_missing": {"expected": []},
    "mgm.architecture": {
        "expected": {
            "model_type": "gpt2", "n_layer": 2, "n_embd": 8, "n_head": 2, "n_positions": 16,
            "vocab_size": 6,
        }
    },
    "mgm.phylogeny_genera": {"expected": 2},
    "mgm.special_tokens": {"expected": 4},
    "mgm.d1_genera_in_vocabulary": {"expected": 2},  # Bacteroides, Faecalibacterium
    # Faecalibacterium_A (suffix stripped) and Bacteroides are MGM tokens, CAG-83 is not.
    "mgm.d2_prevalent_name_coverage": {"expected": 2 / 3, "tol": 1e-9},
    "mgm.d2_abundance_coverage": {"expected": 1.0 - FRANZOSA_CAG_83_SHARE, "tol": 1e-9},
}
MGM_VOCABULARY_FACTS = {
    "mgm.files_missing",  # phylogeny.csv is one of the expected files
    "mgm.phylogeny_genera",
    "mgm.special_tokens",
    "mgm.d1_genera_in_vocabulary",
    "mgm.d2_prevalent_name_coverage",
    "mgm.d2_abundance_coverage",
}


def _facts_file(tmp_path: Path, facts: dict) -> Path:
    path = tmp_path / "facts.yaml"
    path.write_text(yaml.safe_dump({"facts": facts}))
    return path


def test_check_section3_facts_passes_on_matching_data(
    raw_root: Path, results_dir: Path, tmp_path: Path
) -> None:
    report = data.check_section3_facts(raw_root, _facts_file(tmp_path, SYNTHETIC_FACTS))

    assert len(report) == 2 + len(SYNTHETIC_FACTS)
    assert all(row.endswith("|") for row in report)
    assert "PRISM\\|#: 4" in "\n".join(report)  # '|' inside a cell is escaped
    log = _access_log(results_dir)
    assert set(log["purpose"]) == {data.CHECK_PURPOSE}
    assert sorted(log["detail"]) == ["D1 NLIBD", "D2 Franzosa Validation"]  # each opened once


def test_check_section3_facts_collects_every_failure(
    raw_root: Path, results_dir: Path, tmp_path: Path
) -> None:
    facts = {
        **SYNTHETIC_FACTS,
        "d1.prism_metabolites_shape": {"expected": [4, 8848]},
        "mgm.phylogeny_genera": {"expected": 9665},
        "d9.made_up_fact": {"expected": 1},
    }
    with pytest.raises(data.DataFactError) as failure:
        data.check_section3_facts(raw_root, _facts_file(tmp_path, facts))

    message = str(failure.value)
    assert "3 of" in message
    for key in ("d1.prism_metabolites_shape", "mgm.phylogeny_genera", "d9.made_up_fact"):
        assert key in message
    assert "d1.prism_microbes_shape" not in message


def test_synthetic_facts_cover_every_check() -> None:
    # A new check must come with a hand-computed synthetic value, or it is never tested in CI.
    assert set(SYNTHETIC_FACTS) == set(data.FACT_KEYS)


def test_a_failing_loader_fails_only_its_facts(
    raw_root: Path, results_dir: Path, tmp_path: Path
) -> None:
    (data.d3_dir(raw_root) / data.MGM_RESOURCES / "phylogeny.csv").unlink()
    results = data.evaluate_section3_facts(raw_root, _facts_file(tmp_path, SYNTHETIC_FACTS))
    failed = {r.key for r in results if not r.passed}
    assert failed == MGM_VOCABULARY_FACTS


def _replace_in_tables(folder: Path, column: str, old: str, new: str) -> None:
    """Rename one ID wherever ``column`` appears in a D2 dataset's metadata/genera/mtb."""
    for name in ("metadata.tsv", "genera.tsv", "mtb.tsv"):
        table = pd.read_csv(folder / name, sep="\t", dtype=str)
        if column in table.columns:
            table[column] = table[column].replace(old, new)
            table.to_csv(folder / name, sep="\t", index=False)


def test_source_target_id_overlap_is_detected(
    raw_root: Path, results_dir: Path, tmp_path: Path
) -> None:
    """Brief section 5: no source sample or subject ID may appear in the target cohort."""
    fact = "d2.source_target_shared_ids"
    facts = _facts_file(tmp_path, {fact: SYNTHETIC_FACTS[fact]})
    assert data.evaluate_section3_facts(raw_root, facts)[0].passed

    d2 = data.d2_dir(raw_root)
    _replace_in_tables(d2 / WANG, "Sample", "CON-001", "PRISM.7122")  # a PRISM sample ID
    _replace_in_tables(d2 / MARS, "Subject", "10007539", "Validation.LLDeep_0001")  # a subject
    result = data.evaluate_section3_facts(raw_root, facts)[0]

    assert result.observed == {"samples": 1, "subjects": 1}
    assert not result.passed


@pytest.mark.parametrize(
    "text",
    ["", "facts:\n", "facts: {}\n", "fact:\n  d1.prism_microbes_shape:\n    expected: [5, 3]\n"],
)
def test_a_facts_file_without_facts_is_refused(
    raw_root: Path, results_dir: Path, tmp_path: Path, text: str
) -> None:
    # Otherwise `make data` would check nothing and still exit 0.
    path = tmp_path / "facts.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match="no facts"):
        data.load_fact_specs(path)
    with pytest.raises(ValueError, match="no facts"):
        data.check_section3_facts(raw_root, path)


def test_d1_value_range_fails_on_missing_values(
    raw_root: Path, results_dir: Path, tmp_path: Path
) -> None:
    path = data.d1_dir(raw_root) / "metabolome_PRISM.csv"
    stored = pd.read_csv(path, index_col=0)
    stored.iloc[0, 1] = np.nan  # one missing intensity; min and max stay 0 and 1000
    stored.to_csv(path)
    fact = "d1.prism_metabolite_range"
    facts = _facts_file(tmp_path, {fact: SYNTHETIC_FACTS[fact]})

    result = data.evaluate_section3_facts(raw_root, facts)[0]

    assert not result.passed
    assert np.isnan(result.observed).all()


def test_tolerances_apply_elementwise() -> None:
    assert data._matches([0.0, 1.0], [0.0, 1.0 + 1e-9], 1e-6)
    assert not data._matches([0.0, 1.0], [0.0, 1.1], 1e-6)
    assert data._matches({"a": 0.426}, {"a": 0.42}, 0.01)
    assert not data._matches({"a": 0.426, "b": 1}, {"a": 0.42}, 0.01)
    assert data._matches({"x": 1}, {"x": 1}, None)


def test_shipped_facts_file_covers_exactly_the_implemented_checks() -> None:
    specs = data.load_fact_specs()  # configs/data_facts.yaml, not raw data
    assert set(specs) == set(data.FACT_KEYS)
    assert all(spec.get("description") for spec in specs.values())


def test_check_data_script_writes_the_report_and_sets_the_exit_code(
    raw_root: Path, results_dir: Path, tmp_path: Path
) -> None:
    script = paths.REPO_ROOT / "scripts" / "check_data.py"
    main = runpy.run_path(str(script), run_name="check_data")["main"]
    good = _facts_file(tmp_path, SYNTHETIC_FACTS)
    assert main(["--raw-dir", str(raw_root), "--facts", str(good)]) == 0
    report = (results_dir / "data_checks.md").read_text()
    assert "| Fact | Expected | Observed | Status | Note |" in report
    assert "0 failed" in report

    bad = _facts_file(tmp_path, {"d1.prism_microbes_shape": {"expected": [5, 3]}})
    assert main(["--raw-dir", str(raw_root), "--facts", str(bad)]) == 1
    assert "| FAIL |" in (results_dir / "data_checks.md").read_text()


def test_final_evaluation_requires_a_frozen_selection(raw_root: Path, results_dir: Path) -> None:
    """Outside the data checks, the external cohort opens only after `make select` froze (and
    committed) the selections: here nothing is frozen, so the guard refuses and logs nothing."""
    from m2m import freeze

    with pytest.raises(freeze.SelectionNotFrozenError, match="final MLP evaluation"):
        data.load_nlibd(raw_root, final_evaluation=True, purpose="final MLP evaluation")
    with pytest.raises(freeze.SelectionNotFrozenError):
        data.load_franzosa(raw_root, part="Validation", final_evaluation=True, purpose="E4 x")
    assert not (results_dir / "external_access_log.csv").exists()
    # The section-3 data checks fit and score nothing: they are the only exemption.
    data.load_nlibd(raw_root, final_evaluation=True, purpose="data checks", data_check=True)
    assert (results_dir / "external_access_log.csv").exists()


def test_gate_sees_every_final_evaluation_purpose(
    raw_root: Path, results_dir: Path, frozen: list[str]
) -> None:
    data.load_nlibd(raw_root, final_evaluation=True, purpose="E1 external")
    nlibd = data.load_nlibd(
        raw_root, final_evaluation=True, purpose="E2 embed", with_metabolites=False
    )
    assert frozen == ["E1 external", "E2 embed"]
    # Inputs only: neither the metabolome nor the diagnoses were read.
    assert nlibd.metabolites is None
    assert set(nlibd.meta["diagnosis"]) == {data.NOT_READ}
