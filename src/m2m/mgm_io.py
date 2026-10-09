"""Main-environment side of the MGM bridge: genus tables in, coverage report out.

MGM itself (tokenisation, GPT-2 forward passes, fine-tuning) runs in envs/mgm; the two
environments only exchange files under data/processed/mgm/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from m2m import paths
from m2m.data import species_to_genus

GENUS_PREFIX = "g__"


def d1_genus_table(microbes: pd.DataFrame) -> pd.DataFrame:
    """Sum D1 species columns into `g__<Genus>` columns (genus = first token of the name).

    Values stay as published (no re-closure): MGM renormalises within its own vocabulary, and
    the 8 higher-rank labels (e.g. `Lachnospiraceae`) simply fall outside that vocabulary.
    """
    genera = [GENUS_PREFIX + g for g in species_to_genus(microbes.columns)]
    table = microbes.T.groupby(genera, sort=True).sum().T
    table.index.name = microbes.index.name
    return table


def mgm_vocabulary(raw_dir: Path | None = None) -> list[str]:
    """The 9,665 genus tokens MGM knows (index of its phylogeny.csv)."""
    phylogeny = (raw_dir or paths.DATA_RAW) / "MGM" / "mgm" / "resources" / "phylogeny.csv"
    return pd.read_csv(phylogeny, index_col=0).index.tolist()


def vocabulary_coverage(genus_table: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame:
    """Per sample: genera present, how many MGM can tokenise, and the abundance share they carry."""
    in_vocab = genus_table.columns.isin(set(vocabulary))
    present = genus_table > 0
    total = genus_table.sum(axis=1)
    covered = genus_table.loc[:, in_vocab].sum(axis=1)
    return pd.DataFrame(
        {
            "n_genera_present": present.sum(axis=1),
            "n_genera_in_vocab": present.loc[:, in_vocab].sum(axis=1),
            "abundance_in_vocab": (covered / total.where(total > 0)).fillna(0.0),
        },
        index=genus_table.index,
    )


def mgm_env_tags() -> dict[str, str]:
    """The MGM environment's library versions, as run tags for E2/E3 (every run also records the
    main environment's versions, which did not compute the embeddings or fine-tuned models).

    Read from the manifest scripts/mgm_embed.py writes; empty if nothing was embedded yet.
    """
    manifest = paths.DATA_PROCESSED / "mgm" / "PRISM_manifest.json"
    if not manifest.exists():
        return {}
    recorded = json.loads(manifest.read_text())
    keys = ("python", "torch", "transformers", "pandas")
    return {"mgm_env_versions": json.dumps({k: recorded.get(k) for k in keys}, sort_keys=True)}
