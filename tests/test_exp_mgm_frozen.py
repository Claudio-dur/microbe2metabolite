"""E2 (frozen MGM and its random-weight control): embedding files, the envs/mgm calls, and every
stage on synthetic data.

Loaders are monkeypatched with a synthetic PRISM / NLIBD pair whose metabolites depend on a few
latent factors. MGM's pretrained embeddings carry them only in "last" pooling, so CV has a right
answer to find. Of the random-weight networks, only init seed 1 carries them (in both poolings),
so a model seed that read another seed's network would show. The envs/mgm subprocess is
intercepted and MGM's interpreter and model files are empty stand-ins, so nothing here reads
data/raw. Two tests run code in envs/mgm itself: the random-init check (a tiny GPT-2 config,
skipped without the MGM environment) and the opt-in real-MGM test (M2M_TEST_MGM_ENV=1), which
reads MGM's weights and tokenizer from data/raw.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from m2m import config, data, mgm_io, paths, protocol, splits
from m2m.evaluate import NullResult
from m2m.experiments import common, mgm_frozen
from m2m.preprocess import StandardScalerDF
from m2m.run import EXPERIMENT_MODULES

N_PRISM, N_NLIBD, N_DIM, N_LATENT = 48, 18, 256, 3
SPECIES = [
    "Bacteroides_vulgatus",
    "Bacteroides_dorei",
    "Faecalibacterium_prausnitzii",
    "Escherichia_coli",
    "Prevotella_copri",
    "Lachnospiraceae_bacterium_1_1_57FAA",  # a family name, outside MGM's vocabulary as in D1
]
OUT_OF_VOCABULARY = "Lachnospiraceae_bacterium_1_1_57FAA"
VOCABULARY = ["g__Akkermansia", "g__Bacteroides", "g__Escherichia", "g__Faecalibacterium",
              "g__Prevotella"]
METABOLITES = [f"met_{j:02d}" for j in range(24)]
T466 = METABOLITES[:16]  # stands in for the annotated metabolites
N_SIGNAL = 8  # metabolites driven by the latent factors
PRETRAINED, RANDOM = mgm_frozen.PRETRAINED, mgm_frozen.RANDOM
HEAD_MODELS = ("mgm_frozen_ridge", "mgm_frozen_mlp")
CONTROL_MODELS = ("mgm_random_frozen_ridge", "mgm_random_frozen_mlp")
INIT_SEEDS = (0, 1)  # the test config's seeds [0, 1]; the CV seed 0 is among them
INFORMATIVE_NETWORK = 1  # the only random network whose embeddings carry the latent factors


# --- synthetic world -------------------------------------------------------------------------


def _embeddings(
    latent: np.ndarray, rng: np.random.Generator, projection: np.ndarray | None, poolings: tuple
) -> dict[str, np.ndarray]:
    """Embedding arrays: the listed poolings encode the latent factors, the others are noise."""
    n = len(latent)
    arrays = {}
    for pooling, key in mgm_frozen.EMBEDDING_ARRAYS.items():
        noise = rng.normal(size=(n, N_DIM))
        arrays[key] = latent @ projection + 0.1 * noise if pooling in poolings else noise
    return arrays


def _cohort(name: str, prefix: str, n: int, rng: np.random.Generator, biology: dict) -> tuple:
    """A D1-like cohort plus its embeddings: MGM's ("last" encodes the latent factors) and one
    set per random network (only INFORMATIVE_NETWORK encodes them)."""
    ids = pd.Index([f"{prefix}|{i:03d}" for i in range(n)], name=data.SAMPLE_ID)
    microbes = rng.dirichlet(np.ones(len(SPECIES)), size=n) * rng.uniform(0.6, 1.0, size=(n, 1))
    latent = rng.normal(size=(n, N_LATENT))
    log_mets = rng.normal(scale=0.3, size=(n, len(METABOLITES)))
    log_mets[:, :N_SIGNAL] += latent @ biology["metabolites"]
    metabolites = np.exp(log_mets) * 100.0
    noise_zeros = rng.random(metabolites.shape) < 0.1
    noise_zeros[:, :N_SIGNAL] = False  # a zero would be an outlier that buries the signal
    metabolites[noise_zeros] = 0.0
    diagnosis = np.array(data.DIAGNOSES)[np.arange(n) % len(data.DIAGNOSES)]
    meta = pd.DataFrame(
        {
            "subject": ids,
            "diagnosis": diagnosis,
            "ibd_status": [data.IBD_STATUS[d] for d in diagnosis],
            "dataset": "D1",
        },
        index=ids,
    )
    cohort = data.Cohort(
        name,
        pd.DataFrame(microbes, index=ids, columns=SPECIES),
        pd.DataFrame(metabolites, index=ids, columns=METABOLITES),
        meta,
    )
    pretrained = _embeddings(latent, rng, biology["embeddings"], ("last",))
    random_networks = {
        seed: _embeddings(
            latent, rng, biology["random"],
            ("mean", "last") if seed == INFORMATIVE_NETWORK else (),
        )
        for seed in INIT_SEEDS
    }
    return cohort, pretrained, random_networks


def write_embeddings(
    cohort: str,
    ids: list[str],
    arrays: dict,
    order: np.ndarray | None = None,
    init_seed: int | None = None,
) -> None:
    """An embeddings file laid out like scripts/mgm_embed.py writes it (MGM's, or one random
    network's with `init_seed`)."""
    order = np.arange(len(ids)) if order is None else order
    path = mgm_frozen.embeddings_path(cohort, init_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=np.array(ids, dtype=str)[order],
        emb_mean=arrays["emb_mean"][order].astype(np.float32),
        emb_last=arrays["emb_last"][order].astype(np.float32),
    )


def write_cohort_embeddings(
    world: SimpleNamespace, cohort: str, init_seeds: tuple[int, ...] = INIT_SEEDS
) -> None:
    """MGM's embeddings and those of the random networks `init_seeds` for one cohort."""
    source = world.prism if cohort == "PRISM" else world.nlibd
    pretrained, random_networks = world.embeddings[cohort]
    write_embeddings(cohort, source.sample_ids, pretrained)
    for seed in init_seeds:
        write_embeddings(cohort, source.sample_ids, random_networks[seed], init_seed=seed)


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, rng: np.random.Generator) -> SimpleNamespace:
    """Synthetic PRISM and NLIBD behind the real loader names; NLIBD keeps its access guard."""
    biology = {
        "metabolites": rng.normal(size=(N_LATENT, N_SIGNAL)),
        "embeddings": rng.normal(size=(N_LATENT, N_DIM)),
        "random": rng.normal(size=(N_LATENT, N_DIM)),
    }
    prism, *prism_emb = _cohort("PRISM", "PRISM", N_PRISM, rng, biology)
    nlibd, *nlibd_emb = _cohort("NLIBD", "Validation", N_NLIBD, rng, biology)
    purposes: list[str] = []

    def load_nlibd(raw_dir=None, *, final_evaluation=False, purpose="", with_metabolites=True):
        if final_evaluation is not True or not purpose.strip():
            raise data.ExternalTestSetAccessError("NLIBD opened outside a final evaluation")
        purposes.append(purpose)
        if not with_metabolites:  # embed-external must read the inputs only
            return data.Cohort(nlibd.name, nlibd.microbes, None, nlibd.meta)
        return nlibd

    monkeypatch.setattr(data, "load_prism", lambda raw_dir=None: prism)
    monkeypatch.setattr(data, "load_nlibd", load_nlibd)
    monkeypatch.setattr(data, "t466_features", lambda raw_dir=None: list(T466))
    monkeypatch.setattr(mgm_io, "mgm_vocabulary", lambda raw_dir=None: list(VOCABULARY))
    return SimpleNamespace(
        prism=prism,
        nlibd=nlibd,
        prism_emb=prism_emb[0],
        embeddings={"PRISM": prism_emb, "NLIBD": nlibd_emb},
        nlibd_opened=purposes,
    )


@pytest.fixture
def mgm_install(tmp_path: Path) -> dict[str, str]:
    """Empty stand-ins for MGM's interpreter and model files: prepare's install check passes,
    and the subprocess that would use them is intercepted."""
    python = tmp_path / "mgm-env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    python.chmod(0o755)
    model_dir = tmp_path / "mgm-model"
    model_dir.mkdir()
    for name in (mgm_frozen.MODEL_CONFIG_FILE, mgm_frozen.MODEL_WEIGHT_FILES[0]):
        (model_dir / name).touch()
    return {"python": str(python), "model_dir": str(model_dir)}


@pytest.fixture
def cfg(mgm_install: dict[str, str]) -> dict[str, Any]:
    """The real E2 config, shrunk so every stage runs in seconds, with the stand-in MGM install."""
    loaded = config.load_config(paths.CONFIGS / "e2_mgm_frozen.yaml")
    return config.deep_merge(
        loaded,
        {
            "seeds": list(INIT_SEEDS),
            "mlp": {
                "grid": {"lr": [1.0e-3], "dropout": [0.2], "weight_decay": [1.0e-4]},
                "train": {"max_epochs": 40, "patience": 5},
            },
            "learning_curve": {"sizes": [20, 155]},
            "mgm": mgm_install,
        },
    )


def command_args(command: list[str]) -> dict[str, str]:
    """The `--option value` pairs after the interpreter and the script."""
    return dict(zip(command[2::2], command[3::2], strict=True))


def fake_mgm_embed(command: list[str], drop: int = 0) -> None:
    """Stands in for scripts/mgm_embed.py: per genus table found, MGM's embeddings file or (with
    --init random) one file per init seed, minus the first `drop` samples (MGM losing some)."""
    args = command_args(command)
    seeds = [None]
    if args["--init"] == RANDOM:
        seeds = [int(s) for s in args["--init-seeds"].split(",")]
    for table in sorted(Path(args["--inputs"]).glob("*_genus.csv")):
        ids = [str(i) for i in pd.read_csv(table, index_col=0).index][drop:]
        for seed in seeds:
            rng = np.random.default_rng(len(ids))
            arrays = {key: rng.normal(size=(len(ids), N_DIM)) for key in ("emb_mean", "emb_last")}
            write_embeddings(table.name.removesuffix("_genus.csv"), ids, arrays, init_seed=seed)


def intercept_mgm(
    monkeypatch: pytest.MonkeyPatch, embed: Callable[[list[str]], None] = fake_mgm_embed
) -> list[tuple[list[str], dict]]:
    """Replaces the envs/mgm subprocess by `embed`; other commands (e.g. git) still run."""
    calls: list[tuple[list[str], dict]] = []
    real_run: Callable = subprocess.run

    def fake_run(command, *args, **kwargs):
        if len(command) < 2 or command[1] != mgm_frozen.EMBED_SCRIPT:
            return real_run(command, *args, **kwargs)
        calls.append((list(command), kwargs))
        embed(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.fixture
def mgm_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict]]:
    return intercept_mgm(monkeypatch)


def save_selection(
    head: str, target_set: str, params: dict, cv_mean_rho: float, init: str = PRETRAINED
) -> None:
    table = pd.DataFrame(
        {"params": [json.dumps(params, sort_keys=True)], "cv_mean_rho": [cv_mean_rho]}
    )
    protocol.save_selection(mgm_frozen.SELECTION_NAMES[init][head], target_set, params, table)


def save_null_thresholds(threshold: float = 0.3) -> None:
    null = NullResult(
        pooled_rho=np.zeros(4), threshold=threshold, per_permutation_mean=np.zeros(2), quantile=0.95
    )
    for target_set in ("T466", "TALL"):
        protocol.save_null_threshold(data="D1", target_set=target_set, null=null, n_features=4)


# --- config and builders ---------------------------------------------------------------------


def test_config_routes_e2_to_this_module() -> None:
    cfg = config.load_config(paths.CONFIGS / "e2_mgm_frozen.yaml")
    assert EXPERIMENT_MODULES[cfg["experiment"]] == "m2m.experiments.mgm_frozen"
    assert set(cfg["mgm"]) == {"python", "model_dir", "max_len", "batch_size", "threads"}
    assert cfg["head"]["mlp"]["hidden"] == [128]
    # Model seed s of the control reads network s, and the CV runs at seed 0.
    assert mgm_frozen.random_init_seeds(cfg) == sorted({0, *cfg["seeds"]})


def test_mlp_head_takes_the_head_layers_and_e1_training(cfg: dict[str, Any]) -> None:
    params = {"lr": 3.0e-4, "dropout": 0.5, "weight_decay": 1.0e-2, "pooling": "mean"}
    pipe = mgm_frozen.build_mlp_head(params, 3, cfg=cfg, target_set="TALL")
    head = pipe.regressor
    assert (head.hidden, head.dropout, head.seed) == ((128,), 0.5, 3)
    assert (head.train_cfg.lr, head.train_cfg.weight_decay) == (3.0e-4, 1.0e-2)
    assert head.train_cfg.batch_size == cfg["mlp"]["train"]["batch_size"]
    assert isinstance(pipe.x_prep, StandardScalerDF)
    assert pipe.inner_seed == 3 + cfg["inner_validation"]["seed_offset"]


def test_random_control_heads_read_the_network_of_their_model_seed(
    cfg: dict[str, Any], rng: np.random.Generator
) -> None:
    """Same heads, scaling and inner split; only the input block changes with the seed."""
    blocks = {seed: rng.normal(size=(12, 4)) + 10 * seed for seed in INIT_SEEDS}
    wide = pd.concat(
        [pd.DataFrame(values, columns=[f"init{s}_emb_{j:03d}" for j in range(4)])
         for s, values in blocks.items()],
        axis=1,
    ).set_axis([f"S|{i}" for i in range(12)])
    for head, params in (("ridge", {"pooling": "last"}),
                         ("mlp", {"lr": 1e-3, "dropout": 0.2, "weight_decay": 1e-4,
                                  "pooling": "last"})):
        build = mgm_frozen.HEAD_BUILDERS[head]  # TALL: no annotation file to read
        pipe = build(params, 1, cfg=cfg, target_set="TALL", init=RANDOM)
        assert isinstance(pipe.x_prep, mgm_frozen.InitSeedScaler)
        assert pipe.inner_seed == 1 + cfg["inner_validation"]["seed_offset"]
        scaled = pipe.x_prep.fit(wide.iloc[:8]).transform(wide)
        assert pipe.x_prep.features_ == [f"init1_emb_{j:03d}" for j in range(4)]
        assert pipe.x_prep.fit_samples_ == list(wide.index[:8])
        plain = StandardScalerDF().fit(wide.iloc[:8, 4:]).transform(wide.iloc[:, 4:])
        pd.testing.assert_frame_equal(scaled, plain)
    with pytest.raises(KeyError, match="init seed 2"):
        mgm_frozen.InitSeedScaler(2).fit(wide)
    with pytest.raises(ValueError, match="init must be one of"):
        mgm_frozen.embedding_scaler("pretrianed", 0)


def test_grids_put_the_pooling_in_every_parameter_set(cfg: dict[str, Any]) -> None:
    assert mgm_frozen.head_grid("ridge", cfg, "last") == [{"pooling": "last"}]
    full = config.load_config(paths.CONFIGS / "e2_mgm_frozen.yaml")
    mlp = mgm_frozen.head_grid("mlp", full, "mean")
    assert len(mlp) == 8 and {p["pooling"] for p in mlp} == {"mean"}


def test_embed_command_resolves_config_paths_against_the_repository_root() -> None:
    full = config.load_config(paths.CONFIGS / "e2_mgm_frozen.yaml")
    command = mgm_frozen.embed_command(full)
    root = paths.REPO_ROOT
    assert command[:2] == [str(root / "envs/mgm/.venv/bin/python"), "scripts/mgm_embed.py"]
    args = command_args(command)
    assert args["--model-dir"] == str(root / "data/raw/MGM/mgm/resources/general_model")
    assert args["--init"] == "pretrained" and "--init-seeds" not in args
    random = command_args(mgm_frozen.embed_command(full, RANDOM, [0, 1, 2]))
    assert (random["--init"], random["--init-seeds"]) == ("random", "0,1,2")


# --- embeddings ------------------------------------------------------------------------------


def test_load_embeddings_returns_the_pooling_asked_for_in_cohort_order(
    world: SimpleNamespace, rng: np.random.Generator
) -> None:
    ids = world.prism.sample_ids
    write_embeddings("PRISM", ids, world.prism_emb, order=rng.permutation(len(ids)))
    for pooling, key in (("mean", "emb_mean"), ("last", "emb_last")):
        emb = mgm_frozen.load_embeddings("PRISM", pooling)
        assert emb.shape == (N_PRISM, N_DIM)
        assert list(emb.index) == ids
        assert emb.columns[0] == "emb_000" and emb.columns[-1] == "emb_255"
        np.testing.assert_array_equal(emb.to_numpy(), world.prism_emb[key].astype(np.float32))


def test_random_networks_sit_side_by_side_in_the_control_inputs(
    world: SimpleNamespace, rng: np.random.Generator
) -> None:
    ids = world.prism.sample_ids
    random_networks = world.embeddings["PRISM"][1]
    for seed in INIT_SEEDS:  # files in different row orders: rows are aligned by sample id
        write_embeddings("PRISM", ids, random_networks[seed], order=rng.permutation(len(ids)),
                         init_seed=seed)
    wide = mgm_frozen.embedding_inputs("PRISM", "last", RANDOM, INIT_SEEDS)
    assert list(wide.index) == ids and wide.shape == (N_PRISM, len(INIT_SEEDS) * N_DIM)
    for seed in INIT_SEEDS:
        block = wide[[f"init{seed}_emb_{j:03d}" for j in range(N_DIM)]].to_numpy()
        np.testing.assert_array_equal(block, random_networks[seed]["emb_last"].astype(np.float32))
    write_embeddings("PRISM", ids[1:], {k: v[1:] for k, v in random_networks[1].items()},
                     init_seed=1)
    with pytest.raises(ValueError, match="1 samples missing"):
        mgm_frozen.embedding_inputs("PRISM", "last", RANDOM, INIT_SEEDS)
    with pytest.raises(ValueError, match="at least one init seed"):
        mgm_frozen.random_init_embeddings("PRISM", "last", [])


def test_load_embeddings_rejects_unknown_poolings_and_stale_prism_files(
    world: SimpleNamespace,
) -> None:
    ids = world.prism.sample_ids
    with pytest.raises(FileNotFoundError, match="prepare stage"):
        mgm_frozen.load_embeddings("PRISM", "mean")
    write_embeddings("PRISM", ids, world.prism_emb)
    with pytest.raises(ValueError, match="pooling"):
        mgm_frozen.load_embeddings("PRISM", "max")
    partial = {key: values[1:] for key, values in world.prism_emb.items()}
    write_embeddings("PRISM", ids[1:], partial)  # e.g. a file embedded from a dry-run subset
    with pytest.raises(ValueError, match="1 samples missing"):
        mgm_frozen.load_embeddings("PRISM", "mean")


# --- prepare ---------------------------------------------------------------------------------


def test_prepare_writes_genus_tables_and_coverage_then_calls_envs_mgm(
    world: SimpleNamespace, cfg: dict[str, Any], mgm_calls: list, results_dir: Path
) -> None:
    mgm_frozen.run(cfg, stage="prepare")

    [(pretrained, kwargs), (random, random_kwargs)] = mgm_calls
    mgm_dir = (paths.DATA_PROCESSED / "mgm").resolve()
    assert pretrained[:2] == [cfg["mgm"]["python"], "scripts/mgm_embed.py"]
    shared = {
        "--inputs": str(mgm_dir / "inputs"),
        "--out": str(mgm_dir),
        "--model-dir": cfg["mgm"]["model_dir"],
        "--max-len": "512",
        "--batch-size": "64",
        "--threads": "4",
    }
    assert command_args(pretrained) == {**shared, "--init": "pretrained"}
    # Second, after the tokens exist: the same inputs through the random networks 0 and 1.
    assert random[:2] == pretrained[:2]
    assert command_args(random) == {**shared, "--init": "random", "--init-seeds": "0,1"}
    assert kwargs == random_kwargs == {"cwd": paths.REPO_ROOT, "check": True}
    # Model selection's prepare stage never opens the external cohort.
    assert world.nlibd_opened == []

    inputs = mgm_dir / "inputs"
    assert sorted(p.name for p in inputs.glob("*")) == ["PRISM_genus.csv"]
    genus = pd.read_csv(inputs / "PRISM_genus.csv", index_col=0)
    assert list(genus.index) == world.prism.sample_ids
    assert genus.loc["PRISM|000", "g__Bacteroides"] == pytest.approx(
        world.prism.microbes.loc["PRISM|000", ["Bacteroides_vulgatus", "Bacteroides_dorei"]].sum()
    )

    coverage = pd.read_csv(results_dir / "mgm_coverage.csv")
    assert list(coverage.columns) == [
        "cohort", "sample_id", "n_genera_present", "n_genera_in_vocab", "abundance_in_vocab"
    ]
    assert coverage.groupby("cohort").size().to_dict() == {"PRISM": N_PRISM}
    prism = coverage[coverage["cohort"] == "PRISM"].set_index("sample_id").loc[genus.index]
    microbes = world.prism.microbes
    outside = microbes[OUT_OF_VOCABULARY] / microbes.sum(axis=1)
    np.testing.assert_allclose(prism["abundance_in_vocab"], 1.0 - outside)
    assert (prism["n_genera_present"] == 5).all() and (prism["n_genera_in_vocab"] == 4).all()

    # The final-evaluation stage embeds NLIBD's inputs: one logged opening, PRISM rows kept.
    mgm_calls.clear()
    mgm_frozen.run(cfg, stage="embed-external")
    assert world.nlibd_opened == [mgm_frozen.EMBED_PURPOSE]
    assert len(mgm_calls) == 2
    assert sorted(p.name for p in inputs.glob("*")) == ["NLIBD_genus.csv"]
    coverage = pd.read_csv(results_dir / "mgm_coverage.csv")
    assert coverage.groupby("cohort").size().to_dict() == {"NLIBD": N_NLIBD, "PRISM": N_PRISM}


def test_prepare_in_a_dry_run_embeds_all_of_prism_and_never_opens_nlibd(
    world: SimpleNamespace, cfg: dict[str, Any], mgm_calls: list, results_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(common.DRY_RUN_ENV, "1")
    stale = paths.DATA_PROCESSED / "mgm" / "inputs" / "NLIBD_genus.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("sample_id,g__Bacteroides\nValidation|x,1.0\n")  # left by a real run

    mgm_frozen.prepare(cfg)

    assert world.nlibd_opened == []
    assert not stale.exists()  # mgm_embed.py would otherwise re-embed NLIBD inputs
    assert set(pd.read_csv(results_dir / "mgm_coverage.csv")["cohort"]) == {"PRISM"}
    # The pseudo-test rows of a dry run are PRISM rows, so all of PRISM is embedded.
    assert list(mgm_frozen.load_embeddings("PRISM", "last").index) == world.prism.sample_ids
    pseudo_test = common.dry_run_ids()[1]
    assert list(mgm_frozen.external_embeddings("mean").index) == pseudo_test
    control = mgm_frozen.external_embeddings("mean", RANDOM, INIT_SEEDS)
    assert list(control.index) == pseudo_test and control.shape[1] == len(INIT_SEEDS) * N_DIM


@pytest.mark.parametrize("losing", [PRETRAINED, RANDOM])
def test_prepare_fails_when_mgm_returns_fewer_samples(
    world: SimpleNamespace, cfg: dict[str, Any], monkeypatch: pytest.MonkeyPatch, losing: str
) -> None:
    monkeypatch.setenv(common.DRY_RUN_ENV, "1")

    def embed(command: list[str]) -> None:
        fake_mgm_embed(command, drop=int(command_args(command)["--init"] == losing))

    intercept_mgm(monkeypatch, embed=embed)
    with pytest.raises(RuntimeError, match="do not cover"):
        mgm_frozen.prepare(cfg)


@pytest.mark.parametrize("missing", ["interpreter", "model weights", "vocabulary"])
def test_prepare_checks_the_mgm_install_before_opening_nlibd(
    world: SimpleNamespace, cfg: dict[str, Any], mgm_calls: list, results_dir: Path,
    monkeypatch: pytest.MonkeyPatch, missing: str,
) -> None:
    """embed-external opens NLIBD to embed it: a broken MGM setup must fail before that access
    (and before anything is written), not after it; prepare (PRISM only) fails the same way."""
    if missing == "interpreter":
        Path(cfg["mgm"]["python"]).unlink()
        expected = "make mgm-env"
    elif missing == "model weights":
        (Path(cfg["mgm"]["model_dir"]) / mgm_frozen.MODEL_WEIGHT_FILES[0]).unlink()
        expected = "make data"
    else:
        def no_vocabulary(raw_dir: Path | None = None) -> list[str]:
            raise FileNotFoundError("phylogeny.csv missing")

        monkeypatch.setattr(mgm_io, "mgm_vocabulary", no_vocabulary)
        expected = "phylogeny"

    with pytest.raises(FileNotFoundError, match=expected):
        mgm_frozen.embed_external(cfg)
    with pytest.raises(FileNotFoundError, match=expected):
        mgm_frozen.prepare(cfg)

    assert world.nlibd_opened == [] and mgm_calls == []
    assert not (paths.DATA_PROCESSED / "mgm").exists()
    assert not (results_dir / "mgm_coverage.csv").exists()


# --- select / external / curve ---------------------------------------------------------------


def test_stages_select_on_prism_then_score_nlibd_once_per_stage(
    world: SimpleNamespace, cfg: dict[str, Any], results_dir: Path
) -> None:
    write_cohort_embeddings(world, "PRISM")
    mgm_frozen.run(cfg, stage="select")  # no NLIBD embeddings exist yet: selection needs none

    assert world.nlibd_opened == []
    for init, names in mgm_frozen.SELECTION_NAMES.items():
        for head, name in names.items():
            for target_set in ("T466", "TALL"):
                cv = pd.read_csv(results_dir / "selected" / f"{name}__{target_set}__cv.csv")
                assert sorted(cv["pooling"]) == ["last", "mean"]
                if init == PRETRAINED:
                    pooling = protocol.load_selection(name, target_set)["params"]["pooling"]
                    assert pooling == "last", f"{head} {target_set} missed the informative pooling"
                else:  # selected on network 0, which carries no signal
                    assert cv["cv_mean_rho"].max() < 0.2, f"{name} {target_set} saw network 1"
    runs = pd.read_csv(results_dir / "runs.csv")
    assert set(runs["model"]) == set(HEAD_MODELS) | set(CONTROL_MODELS)
    assert set(runs.loc[runs["model"].isin(CONTROL_MODELS), "data"]) == {"D1+MGM-arch"}
    assert set(runs.loc[runs["model"].isin(HEAD_MODELS), "data"]) == {"D1+MGM"}
    assert set(runs["stage"]) == {"cv"}

    write_cohort_embeddings(world, "NLIBD")
    save_null_thresholds()
    mgm_frozen.run(cfg, stage="external")
    mgm_frozen.run(cfg, stage="curve")

    assert world.nlibd_opened == [mgm_frozen.EXTERNAL_PURPOSE, mgm_frozen.CURVE_PURPOSE]
    summary = pd.read_csv(results_dir / "summary.csv").set_index(["model", "target_set"])
    models = (*HEAD_MODELS, *CONTROL_MODELS)
    assert sorted(summary.index) == sorted((m, t) for m in models for t in ("T466", "TALL"))
    assert set(summary["data"].xs("mgm_random_frozen_mlp")) == {"D1+MGM-arch"}
    assert set(summary["data"].xs("mgm_frozen_mlp")) == {"D1+MGM"}
    assert (summary["n_train"] == N_PRISM).all() and (summary["n_test"] == N_NLIBD).all()
    assert summary["n_seeds"].xs("mgm_frozen_ridge").tolist() == [1, 1]
    for model in ("mgm_frozen_mlp", *CONTROL_MODELS):  # the control's ridge has an init seed
        assert summary["n_seeds"].xs(model).tolist() == [2, 2]
    assert summary["n_above_null_mean"].notna().all()
    pretrained_rows = summary.loc[list(HEAD_MODELS)]
    assert all(json.loads(p)["pooling"] == "last" for p in pretrained_rows["params"])
    assert summary.loc[("mgm_frozen_ridge", "T466"), "mean_rho_mean"] > 0.2  # signal transfers

    # Model seed s of the control trained and predicted with network s: only network 1 informs.
    runs = pd.read_csv(results_dir / "runs.csv")
    external = runs[(runs["stage"] == "external") & (runs["model"] == "mgm_random_frozen_ridge")]
    by_seed = external[external["target_set"] == "T466"].set_index("seed")["mean_rho"]
    assert by_seed.loc[INFORMATIVE_NETWORK] > 0.2
    assert by_seed.loc[INFORMATIVE_NETWORK] - by_seed.loc[0] > 0.15

    points = pd.read_csv(results_dir / "learning_curve.csv")
    head, _ = mgm_frozen.best_head("T466")
    assert set(points["model"]) == {mgm_frozen.MODEL_NAMES[PRETRAINED][head]}
    assert set(points["data"]) == {"D1+MGM"}
    assert sorted(points["n_train"].unique()) == [20, N_PRISM] and len(points) == 2 * 2


def test_select_scores_the_saved_prism_folds(
    world: SimpleNamespace, cfg: dict[str, Any], results_dir: Path
) -> None:
    """Every model shares results/splits/prism_folds.csv: a saved file (here made with another
    seed than the config's) must drive E2's fold scores and must not be rewritten. The control's
    CV is a plain CV on network 0, the network of the CV's model seed."""
    write_cohort_embeddings(world, "PRISM", init_seeds=(0,))  # selection needs network 0 only
    labels = world.prism.meta["diagnosis"]
    saved = splits.make_stratified_folds(list(labels.index), list(labels), n_splits=5, seed=99)
    folds_file = paths.SPLITS / splits.PRISM_FOLDS_FILE
    splits.save_folds(saved, folds_file)
    before = folds_file.read_bytes()
    cfg = config.deep_merge(cfg, {"target_sets": ["T466"]})

    mgm_frozen.select(cfg)

    fold_columns = [f"fold{f}_mean_rho" for f in range(5)]
    for name, embeddings in (
        ("e2_mgm_frozen_ridge", mgm_frozen.load_embeddings("PRISM", "last")),
        ("e2_mgm_random_frozen_ridge", mgm_frozen.load_embeddings("PRISM", "last", init_seed=0)),
    ):
        expected, _ = protocol.grid_search_cv(
            mgm_frozen.builder("ridge", cfg, "T466"),
            [{"pooling": "last"}],
            embeddings,
            world.prism.metabolites,
            labels,
            saved,
            cfg,
        )
        table = pd.read_csv(results_dir / "selected" / f"{name}__T466__cv.csv")
        np.testing.assert_allclose(
            table.set_index("pooling").loc["last", fold_columns].to_numpy(dtype=float),
            expected.loc[0, fold_columns].to_numpy(dtype=float),
        )
    assert folds_file.read_bytes() == before


def test_dry_run_scores_the_prism_hold_out_without_nlibd(
    world: SimpleNamespace, cfg: dict[str, Any], results_dir: Path,
    monkeypatch: pytest.MonkeyPatch, frozen: list[str],
) -> None:
    monkeypatch.setenv(common.DRY_RUN_ENV, "1")
    cfg = config.deep_merge(cfg, {"target_sets": ["T466"]})
    write_cohort_embeddings(world, "PRISM")

    for stage in ("select", "external", "curve"):  # no permutation null: tolerated in dry runs
        mgm_frozen.run(cfg, stage=stage)

    assert world.nlibd_opened == []
    train_ids, test_ids = common.dry_run_ids()
    summary = pd.read_csv(results_dir / "summary.csv")
    assert set(summary["model"]) == set(HEAD_MODELS) | set(CONTROL_MODELS)
    assert (summary["n_train"] == len(train_ids)).all()
    assert (summary["n_test"] == len(test_ids)).all()
    assert summary["n_above_null_mean"].isna().all()
    points = pd.read_csv(results_dir / "learning_curve.csv")
    assert sorted(points["n_train"].unique()) == [20, len(train_ids)]


@pytest.mark.parametrize("stage", ["external", "curve"])
def test_final_stages_check_prerequisites_before_opening_nlibd(
    world: SimpleNamespace, cfg: dict[str, Any], stage: str
) -> None:
    write_cohort_embeddings(world, "PRISM")
    with pytest.raises(FileNotFoundError, match="model selection"):
        mgm_frozen.run(cfg, stage=stage)
    mlp_params = {"dropout": 0.2, "lr": 1.0e-3, "pooling": "mean", "weight_decay": 1.0e-4}
    for target_set in ("T466", "TALL"):
        save_selection("ridge", target_set, {"pooling": "last"}, 0.1)
        save_selection("mlp", target_set, mlp_params, 0.1)
    if stage == "external":  # the controls' selections are needed too
        with pytest.raises(FileNotFoundError, match="e2_mgm_random_frozen_ridge"):
            mgm_frozen.run(cfg, stage=stage)
        for target_set in ("T466", "TALL"):
            save_selection("ridge", target_set, {"pooling": "last"}, 0.1, init=RANDOM)
            save_selection("mlp", target_set, mlp_params, 0.1, init=RANDOM)
    with pytest.raises(FileNotFoundError, match="permutation null"):
        mgm_frozen.run(cfg, stage=stage)
    save_null_thresholds()
    with pytest.raises(FileNotFoundError, match="NLIBD_embeddings"):
        mgm_frozen.run(cfg, stage=stage)
    if stage == "external":  # MGM's NLIBD embeddings alone are not enough for the control
        write_embeddings("NLIBD", world.nlibd.sample_ids, world.embeddings["NLIBD"][0])
        with pytest.raises(FileNotFoundError, match="NLIBD_embeddings_random_seed0"):
            mgm_frozen.run(cfg, stage=stage)
    assert world.nlibd_opened == []


def test_curve_follows_the_head_with_the_higher_cv_score() -> None:
    mlp_params = {"dropout": 0.2, "lr": 1.0e-3, "pooling": "last", "weight_decay": 1.0e-4}
    save_selection("ridge", "T466", {"pooling": "mean"}, 0.10)
    save_selection("mlp", "T466", mlp_params, 0.12)
    save_selection("ridge", "T466", {"pooling": "mean"}, 0.50, init=RANDOM)  # never on the curve
    assert mgm_frozen.best_head("T466") == ("mlp", {
        "name": "e2_mgm_frozen_mlp", "target_set": "T466", "params": mlp_params,
        "cv_mean_rho": 0.12, "n_configs": 1,
    })
    save_selection("mlp", "T466", mlp_params, 0.10)
    assert mgm_frozen.best_head("T466")[0] == "ridge"  # a tie goes to the simpler head


# --- the real MGM environment ----------------------------------------------------------------

MGM_PYTHON = paths.REPO_ROOT / "envs" / "mgm" / ".venv" / "bin" / "python"
MGM_WEIGHTS = paths.REPO_ROOT / "data/raw/MGM/mgm/resources/general_model/pytorch_model.bin"
EMBED_SCRIPT = paths.REPO_ROOT / "scripts" / "mgm_embed.py"
# Opt-in, because the test suite runs on synthetic data and never reads data/raw; this test
# needs MGM's real weights and tokenizer from data/raw/MGM (and the envs/mgm interpreter).
REAL_MGM_SWITCH = "M2M_TEST_MGM_ENV"

# Runs in envs/mgm (transformers lives only there): random_backbone on a tiny GPT-2 config.
RANDOM_INIT_CHECK = """
import importlib.util, json, sys
import torch
from transformers import GPT2Config

script, config_dir = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("mgm_embed", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tiny = GPT2Config(n_layer=2, n_embd=16, n_head=2, n_positions=12, vocab_size=30)
tiny.save_pretrained(config_dir)
first, again, other = (module.random_backbone(config_dir, seed) for seed in (0, 0, 1))

def same(a, b):
    a, b = a.state_dict(), b.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)

print(json.dumps({"same_seed": same(first, again), "other_seed": same(first, other),
                  "eval_mode": not first.training, "n_layer": len(first.h)}))
"""


@pytest.mark.skipif(not MGM_PYTHON.exists(), reason="needs envs/mgm (the env with transformers)")
def test_random_networks_are_reproducible_in_envs_mgm(tmp_path: Path) -> None:
    """Same init seed, same weights: embeddings of a seed never change between prepare runs."""
    result = subprocess.run(
        [str(MGM_PYTHON), "-I", "-c", RANDOM_INIT_CHECK, str(EMBED_SCRIPT), str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    checks = json.loads(result.stdout.strip().splitlines()[-1])
    assert checks == {"same_seed": True, "other_seed": False, "eval_mode": True, "n_layer": 2}


@pytest.mark.skipif(
    os.environ.get(REAL_MGM_SWITCH) != "1" or not (MGM_PYTHON.exists() and MGM_WEIGHTS.exists()),
    reason=f"opt-in ({REAL_MGM_SWITCH}=1); needs envs/mgm and MGM's model files in data/raw",
)
def test_envs_mgm_embeds_a_synthetic_genus_table(
    world: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(common.DRY_RUN_ENV, "1")  # PRISM only: the synthetic NLIBD is not needed
    real_cfg = config.load_config(paths.CONFIGS / "e2_mgm_frozen.yaml")
    mgm_frozen.prepare(real_cfg)

    mean = mgm_frozen.load_embeddings("PRISM", "mean")
    last = mgm_frozen.load_embeddings("PRISM", "last")
    assert mean.shape == last.shape == (N_PRISM, N_DIM)
    assert list(mean.index) == world.prism.sample_ids
    assert not np.allclose(mean.to_numpy(), last.to_numpy())
    manifest = json.loads((paths.DATA_PROCESSED / "mgm" / "PRISM_manifest.json").read_text())
    # <bos> + 4 in-vocabulary genera (Lachnospiraceae is dropped) + <eos>
    assert manifest["tokens_per_sample"]["max"] == 6
    seeds = mgm_frozen.random_init_seeds(real_cfg)
    control = mgm_frozen.embedding_inputs("PRISM", "mean", RANDOM, seeds)
    assert control.shape == (N_PRISM, len(seeds) * N_DIM)
    first = control.iloc[:, :N_DIM].to_numpy()
    assert not np.allclose(first, mean.to_numpy())  # random weights, not MGM's
    assert not np.allclose(first, control.iloc[:, N_DIM : 2 * N_DIM].to_numpy())
    random_manifest = json.loads(
        (paths.DATA_PROCESSED / "mgm" / "PRISM_manifest_random.json").read_text()
    )
    assert random_manifest["init_seeds"] == seeds and random_manifest["init"] == "random"
