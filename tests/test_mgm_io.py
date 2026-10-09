"""D1 species → MGM genus tables and vocabulary coverage."""

from __future__ import annotations

import pandas as pd
import pytest

from m2m import mgm_io


@pytest.fixture
def species() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Bacteroides_dorei": [0.30, 0.0],
            "Bacteroides_vulgatus": [0.10, 0.2],
            "Lachnospiraceae_bacterium_1_1_57FAA": [0.05, 0.0],
            "Faecalibacterium_prausnitzii": [0.15, 0.0],
        },
        index=pd.Index(["S|1", "S|2"], name="sample_id"),
    )


def test_species_are_summed_into_prefixed_genera(species: pd.DataFrame) -> None:
    table = mgm_io.d1_genus_table(species)
    assert list(table.columns) == ["g__Bacteroides", "g__Faecalibacterium", "g__Lachnospiraceae"]
    assert table.loc["S|1", "g__Bacteroides"] == pytest.approx(0.40)
    assert table.index.name == "sample_id"
    # MGM renormalises within its own vocabulary, so the table must keep the published scale.
    assert table.sum(axis=1).tolist() == pytest.approx(species.sum(axis=1).tolist())


def test_coverage_counts_only_vocabulary_genera(species: pd.DataFrame) -> None:
    table = mgm_io.d1_genus_table(species)
    cov = mgm_io.vocabulary_coverage(table, ["g__Bacteroides", "g__Faecalibacterium"])
    assert cov.loc["S|1", "n_genera_present"] == 3
    assert cov.loc["S|1", "n_genera_in_vocab"] == 2
    assert cov.loc["S|1", "abundance_in_vocab"] == pytest.approx(0.55 / 0.60)
    assert cov.loc["S|2", "abundance_in_vocab"] == pytest.approx(1.0)


def test_coverage_of_an_empty_sample_is_zero_not_nan() -> None:
    table = pd.DataFrame({"g__Bacteroides": [0.0]}, index=["S|0"])
    cov = mgm_io.vocabulary_coverage(table, ["g__Bacteroides"])
    assert cov.loc["S|0", "abundance_in_vocab"] == 0.0
