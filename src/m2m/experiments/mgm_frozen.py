"""E2: frozen MGM embeddings with a ridge head and a small MLP head (brief section 6), plus the
same models on a random-weight copy of MGM's network (the H1 control).

MGM runs in its own pinned environment (envs/mgm, scripts/mgm_embed.py); this module only swaps
files with it: genus tables go in, per-sample embeddings come out (data/processed/mgm/).

One frozen GPT-2 architecture (MGM's config.json: 8 layers, 256-d, 8 heads, 512 positions,
vocabulary 9,669) embeds the same MGM tokens with two sets of weights:
  pretrained  MGM's weights                       -> mgm_frozen_ridge, mgm_frozen_mlp (D1+MGM)
  random      transformers' GPT-2 initialisation  -> mgm_random_frozen_ridge, ..._mlp (D1+MGM-arch)
Everything after the embeddings (poolings, heads, CV selection, external protocol) is shared, so
comparing the two isolates what MGM's pretrained weights add; comparing either with ridge or the
MLP on D1 also mixes in the architecture and MGM's tokenisation (genus presence ordered by
corpus z-score).

Stages, in order (``python -m m2m.run configs/e2_mgm_frozen.yaml [--stage S]``); the first two
are model selection (`make select`, PRISM only), the last three the final evaluation:

  prepare         D1 species -> the PRISM genus table, MGM vocabulary coverage per sample ->
                  results/mgm_coverage.csv, then envs/mgm embeds every PRISM sample (mean-pooled
                  and last-token hidden states, both saved) with MGM's weights, then with one
                  random-weight network per init seed (same tokens). Never opens NLIBD.
  select          PRISM 5-fold CV of both heads on z-scored embeddings, for both weight sets. The
                  pooling (mean or last) is a hyperparameter, selected together with the head's
                  own ones, per target set. The random control is selected on the network of
                  init seed 0.
  embed-external  the same for NLIBD's genus table (its coverage rows, its tokens for E3 and its
                  embeddings by MGM and by every random network): opens NLIBD's microbiome only.
                  Dry runs skip it: their pseudo-test rows are PRISM rows, embedded by prepare.
  external        fit each head's selected configuration on all of PRISM, score it once on
                  NLIBD. Model seed s of the random control uses the network of init seed s.
  curve           E5 learning curve of the pretrained head with the higher CV mean rho on the
                  curve's target set.

Why embedding NLIBD (embed-external) cannot leak: the networks are frozen (MGM's weights, or
random weights fixed by their seed before any sample is seen), and MGM's input normalisation uses
its own corpus statistics, so every sample's embedding depends on that sample alone (the padded
length moves it only by float rounding, ~1e-6). The only thing fitted on embeddings, the
per-column scaler, sits inside the CV pipeline (training rows only).
"""

from __future__ import annotations

import functools
import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from m2m import data, mgm_io, paths, protocol
from m2m.experiments import common
from m2m.models import PICKLED_WEIGHTS, POOLING_MODES
from m2m.preprocess import StandardScalerDF
from m2m.train import TorchMLPRegressor, TrainConfig

EXPERIMENT = "e2"
# "prepare" embeds PRISM only, so model selection never opens the external cohort;
# "embed-external" (a final-evaluation stage) embeds NLIBD's inputs just before scoring.
STAGES = ("prepare", "select", "embed-external", "external", "curve")
# Same metabolites and same test samples as D1, so E0's D1 permutation null is the right chance
# level for the embeddings too.
NULL_DATA = "D1"

PRISM, NLIBD = "PRISM", "NLIBD"
HEADS = ("ridge", "mlp")  # also the tie-break order when the curve stage picks a head
# The frozen network's weights: MGM's, or random ones for the same architecture (H1 control).
PRETRAINED, RANDOM = "pretrained", "random"
INITS = (PRETRAINED, RANDOM)
MODEL_NAMES = {
    PRETRAINED: {"ridge": "mgm_frozen_ridge", "mlp": "mgm_frozen_mlp"},
    RANDOM: {"ridge": "mgm_random_frozen_ridge", "mlp": "mgm_random_frozen_mlp"},
}
SELECTION_NAMES = {
    PRETRAINED: {"ridge": "e2_mgm_frozen_ridge", "mlp": "e2_mgm_frozen_mlp"},
    RANDOM: {"ridge": "e2_mgm_random_frozen_ridge", "mlp": "e2_mgm_random_frozen_mlp"},
}
# summary.csv `data`: D1's samples and genera as MGM tokens, embedded by MGM, or by its
# architecture alone.
DATA_NAMES = {PRETRAINED: "D1+MGM", RANDOM: "D1+MGM-arch"}
# Model seed of every CV fit (E1's cv_seed). The random control's model seed is also the init seed
# of the network it reads, so it is selected on init seed 0.
CV_SEED = 0
# The pretrained ridge head has no randomness (fixed embeddings, alpha from GCV on the training
# rows): one fit is enough.
RIDGE_SEEDS = (0,)

POOLINGS = POOLING_MODES  # ("mean", "last"), as computed by scripts/mgm_embed.py
EMBEDDING_ARRAYS = {"mean": "emb_mean", "last": "emb_last"}  # keys in every embeddings file
EMBEDDING_COLUMN = "emb_{:03d}"
# Column prefix of one random network's embeddings in the random control's input table.
INIT_BLOCK = "init{}_"

MGM_SUBDIR = "mgm"
INPUTS_SUBDIR = "inputs"
GENUS_SUFFIX = "_genus.csv"
EMBED_SCRIPT = "scripts/mgm_embed.py"  # run with cwd = repository root
# What models.load_mgm_backbone needs in the model directory: the config, plus the weights in
# either format it reads (the random networks need only the config).
MODEL_CONFIG_FILE = "config.json"
MODEL_WEIGHT_FILES = (PICKLED_WEIGHTS, "model.safetensors")
COVERAGE_CSV = "mgm_coverage.csv"
COVERAGE_STATS = ("n_genera_present", "n_genera_in_vocab", "abundance_in_vocab")

EMBED_PURPOSE = "MGM tokenisation/embedding of NLIBD inputs (transform only; nothing is fitted)"
EXTERNAL_PURPOSE = (
    "E2 final evaluation: frozen-MGM ridge and MLP heads and their random-weight controls "
    "(configurations selected by PRISM CV)"
)
CURVE_PURPOSE = "E5 learning curve of the frozen-MGM head selected by PRISM CV (E2 curve stage)"


def run(cfg: Mapping[str, Any], *, stage: str | None = None, n_jobs: int = 1) -> None:
    """Run every stage in order, or only `stage`."""
    common.dry_run()  # a dry run must not write into the committed results/ or data/processed/
    actions = {
        "prepare": prepare,
        "select": select,
        "embed-external": embed_external,
        "external": external,
        "curve": curve,
    }
    for name in common.stages_to_run(stage, STAGES):
        print(f"[{EXPERIMENT}] stage {name}")
        actions[name](cfg, n_jobs=n_jobs)


# --------------------------------------------------------------------------------------------
# Locations and embeddings
# --------------------------------------------------------------------------------------------


def mgm_dir() -> Path:
    """Where envs/mgm writes tokens, embeddings and manifests (read at call time for tests)."""
    return paths.DATA_PROCESSED / MGM_SUBDIR


def inputs_dir() -> Path:
    return mgm_dir() / INPUTS_SUBDIR


def embeddings_path(cohort: str, init_seed: int | None = None) -> Path:
    """MGM's embeddings of a cohort, or (`init_seed`) those of one random-weight network."""
    if init_seed is None:
        return mgm_dir() / f"{cohort}_embeddings.npz"
    return mgm_dir() / f"{cohort}_embeddings_random_seed{init_seed}.npz"


def repo_path(path: str | Path) -> Path:
    """Config paths are relative to the repository root, not to the caller's directory."""
    path = Path(path)
    return path if path.is_absolute() else paths.REPO_ROOT / path


def random_init_seeds(cfg: Mapping[str, Any]) -> list[int]:
    """The random networks to embed with: the CV's, and one per model seed (model seed s of the
    control reads the network of init seed s)."""
    return sorted({CV_SEED, *common.model_seeds(cfg)})


def load_embeddings(cohort: str, pooling: str, init_seed: int | None = None) -> pd.DataFrame:
    """Frozen embeddings of one cohort: samples x 256 (emb_000, ...), index = sample ids.

    `pooling` is "mean" (average over real tokens) or "last" (the final <eos> token); with
    `init_seed` the random-weight network of that seed replaces MGM's weights. PRISM must cover
    exactly the PRISM cohort and comes back in cohort order, so a stale or partial file fails
    here instead of silently shrinking the training set.
    """
    if pooling not in EMBEDDING_ARRAYS:
        raise ValueError(f"pooling must be one of {list(EMBEDDING_ARRAYS)}, got {pooling!r}")
    path = embeddings_path(cohort, init_seed)
    ids, values = _read_embedding_file(path, EMBEDDING_ARRAYS[pooling])
    columns = [EMBEDDING_COLUMN.format(j) for j in range(values.shape[1])]
    frame = pd.DataFrame(values, index=pd.Index(ids, name=data.SAMPLE_ID), columns=columns)
    return _in_prism_order(frame, path.name) if cohort == PRISM else frame


def random_init_embeddings(
    cohort: str, pooling: str, init_seeds: Sequence[int]
) -> pd.DataFrame:
    """The random control's input table: the networks of `init_seeds` side by side (columns
    init<s>_emb_000, ...). `InitSeedScaler` gives each model seed its own network's block."""
    if not init_seeds:
        raise ValueError("the random-weight control needs at least one init seed")
    blocks = [
        load_embeddings(cohort, pooling, seed).add_prefix(INIT_BLOCK.format(seed))
        for seed in init_seeds
    ]
    reference = blocks[0].index
    for seed, block in zip(init_seeds, blocks, strict=True):
        if set(block.index) != set(reference):
            raise ValueError(
                f"{cohort}: the random networks of init seeds {init_seeds[0]} and {seed} embedded "
                f"different samples: re-run the {EXPERIMENT} prepare stage"
            )
    return pd.concat([block.loc[reference] for block in blocks], axis=1)


def embedding_inputs(
    cohort: str, pooling: str, init: str = PRETRAINED, init_seeds: Sequence[int] = ()
) -> pd.DataFrame:
    """What the heads read for one weight set: MGM's embeddings, or the random networks'."""
    if init == PRETRAINED:
        return load_embeddings(cohort, pooling)
    if init == RANDOM:
        return random_init_embeddings(cohort, pooling, init_seeds)
    raise ValueError(f"init must be one of {INITS}, got {init!r}")


def external_embeddings(
    pooling: str, init: str = PRETRAINED, init_seeds: Sequence[int] = ()
) -> pd.DataFrame:
    """Inputs of the external cohort: NLIBD, or the PRISM hold-out rows in a dry run."""
    if common.dry_run():
        return embedding_inputs(PRISM, pooling, init, init_seeds).loc[common.dry_run_ids()[1]]
    return embedding_inputs(NLIBD, pooling, init, init_seeds)


def _read_embedding_file(path: Path, array: str) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the {EXPERIMENT} prepare stage first")
    with np.load(path, allow_pickle=False) as npz:
        ids = [str(i) for i in npz["sample_ids"]]
        values = np.asarray(npz[array], dtype=float)
    if values.ndim != 2 or values.shape[0] != len(ids):
        raise ValueError(f"{path.name}: {array} has shape {values.shape} for {len(ids)} samples")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: duplicated sample ids")
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: {array} contains NaN or infinite values")
    return ids, values


def _in_prism_order(frame: pd.DataFrame, file_name: str) -> pd.DataFrame:
    expected = data.load_prism().sample_ids
    missing = pd.Index(expected).difference(frame.index)
    extra = frame.index.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            f"{file_name} does not match the PRISM cohort ({len(missing)} samples missing, "
            f"{len(extra)} extra): re-run the {EXPERIMENT} prepare stage"
        )
    return frame.loc[expected]


def _rows(embeddings: pd.DataFrame, ids: pd.Index, what: str) -> pd.DataFrame:
    """Embedding rows for `ids`, in that order, so they line up with the metabolite table."""
    missing = ids.difference(embeddings.index)
    if len(missing):
        raise KeyError(
            f"{what}: no MGM embedding for {len(missing)} samples, e.g. {missing[:3].tolist()}"
        )
    return embeddings.loc[ids]


# --------------------------------------------------------------------------------------------
# Stage: prepare (genus tables, coverage, MGM embedding in envs/mgm)
# --------------------------------------------------------------------------------------------


def prepare(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """PRISM genus table + coverage, embedded in envs/mgm with MGM's weights and with the random
    networks (`n_jobs` unused: MGM's threads are cfg["mgm"]["threads"]). Never opens NLIBD."""
    embed_cohorts(cfg, genus_tables())


def embed_external(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """NLIBD's genus table embedded the same way (a final-evaluation stage: it opens the cohort).

    Only microbiome inputs are read and nothing is fitted (MGM normalises with its own corpus
    statistics). The MGM installation and vocabulary are checked before the cohort is opened,
    so a broken setup never costs a logged access. Dry runs need nothing: their pseudo-test rows
    are PRISM rows, already embedded by `prepare`.
    """
    if common.dry_run():
        print(f"[{EXPERIMENT}] dry run: pseudo-test rows are PRISM rows, nothing to embed")
        return
    require_mgm_install(cfg)
    mgm_io.mgm_vocabulary()
    nlibd = data.load_nlibd(final_evaluation=True, purpose=EMBED_PURPOSE, with_metabolites=False)
    embed_cohorts(cfg, {NLIBD: mgm_io.d1_genus_table(nlibd.microbes)})


def embed_cohorts(cfg: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]) -> None:
    """Coverage rows for these cohorts, then their MGM and random-network embeddings."""
    require_mgm_install(cfg)
    vocabulary = mgm_io.mgm_vocabulary()
    write_genus_tables(tables)
    write_coverage(coverage_table(tables, vocabulary))
    run_mgm_embed(cfg, PRETRAINED)
    # The same tokens through the same architecture with random weights: a forward pass takes
    # about a second per cohort and network.
    init_seeds = random_init_seeds(cfg)
    run_mgm_embed(cfg, RANDOM, init_seeds)
    check_embedded(tables, init_seeds)


def require_mgm_install(cfg: Mapping[str, Any]) -> None:
    """Fail early, with the fix, if the MGM interpreter or its model files are missing."""
    mgm = cfg["mgm"]
    python = repo_path(mgm["python"])
    if not (python.is_file() and os.access(python, os.X_OK)):
        raise FileNotFoundError(
            f"MGM interpreter {python} not found: build the MGM environment with `make mgm-env`"
        )
    model_dir = repo_path(mgm["model_dir"])
    has_weights = any((model_dir / name).is_file() for name in MODEL_WEIGHT_FILES)
    if not ((model_dir / MODEL_CONFIG_FILE).is_file() and has_weights):
        raise FileNotFoundError(
            f"MGM model files missing in {model_dir} (need {MODEL_CONFIG_FILE} and one of "
            f"{list(MODEL_WEIGHT_FILES)}): download them with `make data`"
        )


def genus_tables() -> dict[str, pd.DataFrame]:
    """The PRISM genus table to embed in `prepare` (NLIBD's is built by `embed_external`).

    PRISM is the whole cohort even in a dry run (where `common.load_d1_train()` drops the
    pseudo-test rows), because a dry run's pseudo-test rows are PRISM rows and need embeddings.
    """
    return {PRISM: mgm_io.d1_genus_table(data.load_prism().microbes)}


def write_genus_tables(tables: Mapping[str, pd.DataFrame]) -> Path:
    """One `<cohort>_genus.csv` per cohort in the MGM inputs folder; returns the folder.

    mgm_embed.py embeds every genus table it finds, so tables of other cohorts are removed
    first: each stage embeds exactly the cohorts it was asked for (their earlier embedding files
    stay in place).
    """
    folder = inputs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for path in folder.glob(f"*{GENUS_SUFFIX}"):
        if path.name.removesuffix(GENUS_SUFFIX) not in tables:
            print(f"[{EXPERIMENT}] removing stale MGM input {path}")
            path.unlink()
    for cohort, table in tables.items():
        table.to_csv(folder / f"{cohort}{GENUS_SUFFIX}")
    return folder


def coverage_table(
    tables: Mapping[str, pd.DataFrame], vocabulary: Sequence[str]
) -> pd.DataFrame:
    """Per sample: genera present, genera MGM can tokenise and the abundance share they carry."""
    frames = []
    for cohort, table in tables.items():
        per_sample = mgm_io.vocabulary_coverage(table, list(vocabulary))
        per_sample = per_sample.rename_axis(index="sample_id").reset_index()
        frames.append(per_sample.assign(cohort=cohort))
    return pd.concat(frames, ignore_index=True)[["cohort", "sample_id", *COVERAGE_STATS]]


def write_coverage(coverage: pd.DataFrame) -> Path:
    """results/mgm_coverage.csv (one row per cohort x sample) plus a printed summary.

    PRISM and NLIBD are embedded in different stages, so rows of other cohorts already in the
    file are kept and only the cohorts written now are replaced.
    """
    out = paths.RESULTS / COVERAGE_CSV
    out.parent.mkdir(parents=True, exist_ok=True)
    written = coverage
    if out.exists():
        kept = pd.read_csv(out, dtype={"sample_id": str})
        kept = kept[~kept["cohort"].isin(coverage["cohort"].unique())]
        coverage = pd.concat([kept, coverage], ignore_index=True)
    coverage.to_csv(out, index=False)
    print_coverage(written)  # only this stage's cohorts (selection never prints NLIBD's)
    return out


def print_coverage(coverage: pd.DataFrame) -> None:
    print(f"[{EXPERIMENT}] MGM vocabulary coverage per sample (min / median):")
    for cohort, group in coverage.groupby("cohort", sort=False):
        stats = "; ".join(
            f"{name} {group[name].min():.3g} / {group[name].median():.3g}"
            for name in COVERAGE_STATS
        )
        print(f"  {cohort} (n={len(group)}): {stats}")


def embed_command(
    cfg: Mapping[str, Any], init: str = PRETRAINED, init_seeds: Sequence[int] = ()
) -> list[str]:
    """The envs/mgm call that tokenises and embeds every genus table in the inputs folder, with
    MGM's weights or (init "random") with one random network per seed in `init_seeds`."""
    mgm = cfg["mgm"]
    command = [
        str(repo_path(mgm["python"])),
        EMBED_SCRIPT,
        # Absolute: the script runs from the repository root, whatever our working directory.
        "--inputs", str(inputs_dir().resolve()),
        "--out", str(mgm_dir().resolve()),
        "--model-dir", str(repo_path(mgm["model_dir"])),
        "--max-len", str(int(mgm["max_len"])),
        "--batch-size", str(int(mgm["batch_size"])),
        "--threads", str(int(mgm["threads"])),
        "--init", init,
    ]
    if init == RANDOM:
        command += ["--init-seeds", ",".join(str(int(seed)) for seed in init_seeds)]
    return command


def run_mgm_embed(
    cfg: Mapping[str, Any], init: str = PRETRAINED, init_seeds: Sequence[int] = ()
) -> None:
    command = embed_command(cfg, init, init_seeds)
    # Flushed so our log lines stay ahead of the child's when stdout is a pipe or a file.
    print(f"[{EXPERIMENT}] embedding ({init} weights): {shlex.join(command)}", flush=True)
    try:
        subprocess.run(command, cwd=paths.REPO_ROOT, check=True)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"cannot start {command[0]} ({err}): build the MGM environment with `make mgm-env`"
        ) from err


def check_embedded(tables: Mapping[str, pd.DataFrame], init_seeds: Sequence[int]) -> None:
    """Every sample of every genus table came back embedded by MGM and by every random network,
    with finite values for both poolings (MGM would silently drop a sample with no
    in-vocabulary genus)."""
    for cohort, table in tables.items():
        expected = {str(i) for i in table.index}
        for init_seed in (None, *init_seeds):
            path = embeddings_path(cohort, init_seed)
            for array in EMBEDDING_ARRAYS.values():
                ids, _ = _read_embedding_file(path, array)
                if set(ids) != expected:
                    raise RuntimeError(
                        f"{cohort}: MGM embeddings in {path.name} do not cover the genus table"
                    )


# --------------------------------------------------------------------------------------------
# Heads (pipeline builders)
# --------------------------------------------------------------------------------------------
# Module-level functions so `functools.partial(...)` pickles into joblib workers. `params`
# always carries the pooling too: the caller passes that pooling's embeddings as X, so the
# builders themselves ignore it.


class InitSeedScaler:
    """The random control's input scaling: z-scores ONE random network's embeddings.

    The control's input table holds several networks side by side (columns init<s>_emb_*); the
    builders pass the model seed as `init_seed`, so model seed s always reads the network of init
    seed s. The CV (model seed 0) therefore selects on network 0, and the final fits of seeds
    0-4 use networks 0-4: their SD over seeds covers the randomness of the initialisation, which
    the pretrained embeddings do not have. Fit on training rows only, like StandardScalerDF.
    """

    def __init__(self, init_seed: int) -> None:
        self.init_seed = int(init_seed)
        self.scaler = StandardScalerDF()

    @property
    def features_(self) -> list[Any] | None:
        return self.scaler.features_

    @property
    def fit_samples_(self) -> list[Any] | None:
        return self.scaler.fit_samples_

    def fit(self, E: pd.DataFrame) -> InitSeedScaler:
        self.scaler.fit(self._own_network(E))
        return self

    def transform(self, E: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(self._own_network(E))

    def _own_network(self, E: pd.DataFrame) -> pd.DataFrame:
        prefix = INIT_BLOCK.format(self.init_seed)
        columns = [c for c in E.columns if str(c).startswith(prefix)]
        if not columns:
            raise KeyError(
                f"no embeddings of the random network of init seed {self.init_seed} among the "
                "inputs: model seed s of the random control needs the network of init seed s"
            )
        return E[columns]


def embedding_scaler(init: str, seed: int) -> StandardScalerDF | InitSeedScaler:
    """z-scoring of the heads' inputs on the training rows (the random control: its own network)."""
    if init == PRETRAINED:
        return StandardScalerDF()
    if init == RANDOM:
        return InitSeedScaler(seed)
    raise ValueError(f"init must be one of {INITS}, got {init!r}")


def mlp_head(params: Mapping[str, Any], seed: int, cfg: Mapping[str, Any]) -> TorchMLPRegressor:
    """E1's training recipe (base `mlp.train`, grid point `params`), `head.mlp.hidden` layers."""
    train_cfg = TrainConfig.from_dict(
        {**cfg["mlp"]["train"], "lr": params["lr"], "weight_decay": params["weight_decay"]}
    )
    return TorchMLPRegressor(
        hidden=tuple(cfg["head"]["mlp"]["hidden"]),
        dropout=float(params["dropout"]),
        train_cfg=train_cfg,
        seed=int(seed),
    )


def build_ridge_head(
    params: Mapping[str, Any],
    seed: int,
    *,
    cfg: Mapping[str, Any],
    target_set: str,
    candidates: Sequence[str] | None = None,
    init: str = PRETRAINED,
) -> protocol.M2MPipeline:
    """Multi-output ridge (alpha per metabolite by GCV) on embeddings z-scored on training rows."""
    return protocol.make_pipeline(
        embedding_scaler(init, seed),
        common.metabolite_prep(cfg, target_set, candidates),
        common.ridge_regressor(cfg),
        cfg,
        seed,
    )


def build_mlp_head(
    params: Mapping[str, Any],
    seed: int,
    *,
    cfg: Mapping[str, Any],
    target_set: str,
    candidates: Sequence[str] | None = None,
    init: str = PRETRAINED,
) -> protocol.M2MPipeline:
    """Small MLP on embeddings z-scored on training rows (early stopping on an inner split)."""
    return protocol.make_pipeline(
        embedding_scaler(init, seed),
        common.metabolite_prep(cfg, target_set, candidates),
        mlp_head(params, seed, cfg),
        cfg,
        seed,
    )


HEAD_BUILDERS = {"ridge": build_ridge_head, "mlp": build_mlp_head}


def builder(
    head: str, cfg: Mapping[str, Any], target_set: str, init: str = PRETRAINED
) -> protocol.PipelineBuilder:
    """Picklable builder; the T466 candidate list is read once here, not once per fit."""
    return functools.partial(
        HEAD_BUILDERS[head],
        cfg=cfg,
        target_set=target_set,
        candidates=common.target_candidates(target_set),
        init=init,
    )


def head_grid(head: str, cfg: Mapping[str, Any], pooling: str) -> list[dict[str, Any]]:
    """CV grid at one pooling: ridge tunes alpha itself (GCV); the MLP uses E1's grid."""
    grid: dict[str, list[Any]] = {"pooling": [pooling]}
    if head == "mlp":
        grid.update({name: list(values) for name, values in cfg["mlp"]["grid"].items()})
    return protocol.expand_grid(grid)


# --------------------------------------------------------------------------------------------
# Stage: select (PRISM cross-validation only)
# --------------------------------------------------------------------------------------------


def select(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """CV every weight set x head x pooling x grid point on the saved PRISM folds; save the
    winners. The random control reads the network of init seed CV_SEED (the CV's model seed)."""
    _, Y, labels = common.load_d1_train()
    folds = common.d1_folds(labels, cfg)
    for init in INITS:
        inputs = {
            p: _rows(embedding_inputs(PRISM, p, init, [CV_SEED]), Y.index, PRISM)
            for p in POOLINGS
        }
        for target_set in common.target_sets(cfg):
            for head in HEADS:
                table = cv_table(
                    head, target_set, inputs, Y, labels, folds, cfg, init=init, n_jobs=n_jobs
                )
                best = protocol.select_best(table)
                protocol.save_selection(SELECTION_NAMES[init][head], target_set, best, table)
                protocol.record_cv_table(
                    table,
                    experiment=EXPERIMENT,
                    model=MODEL_NAMES[init][head],
                    target_set=target_set,
                    data=DATA_NAMES[init],
                    cfg=cfg,
                    n_train=len(Y),
                    extra_tags=mgm_io.mgm_env_tags(),
                )
                _print_selection(MODEL_NAMES[init][head], target_set, table, best)


def cv_table(
    head: str,
    target_set: str,
    inputs: Mapping[str, pd.DataFrame],
    Y: pd.DataFrame,
    labels: pd.Series,
    folds: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    init: str = PRETRAINED,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """One CV row per (pooling, head hyperparameters): the pooling decides which table is X."""
    build = builder(head, cfg, target_set, init)
    tables = [
        protocol.grid_search_cv(
            build, head_grid(head, cfg, pooling), X, Y, labels, folds, cfg,
            seed=CV_SEED, n_jobs=n_jobs,
        )[0]
        for pooling, X in inputs.items()
    ]
    return pd.concat(tables, ignore_index=True)


def _print_selection(
    model: str, target_set: str, table: pd.DataFrame, best: Mapping[str, Any]
) -> None:
    by_pooling = table.groupby("pooling", sort=False)["cv_mean_rho"].max()
    scores = ", ".join(f"{pooling} {rho:.4f}" for pooling, rho in by_pooling.items())
    print(
        f"[{EXPERIMENT}] {model} {target_set}: best CV mean rho by pooling: {scores}; "
        f"selected {dict(best)}"
    )


# --------------------------------------------------------------------------------------------
# Stages: external and curve (final evaluation)
# --------------------------------------------------------------------------------------------
# Both check every prerequisite (selections, null threshold, embedding files) before opening the
# external cohort, so a missing piece never costs an access that produces no result.


def external(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """Fit each selected configuration (both weight sets, both heads) on PRISM and score it
    once per target set."""
    target_sets = common.target_sets(cfg)
    selections = {
        (init, head, t): protocol.load_selection(SELECTION_NAMES[init][head], t)
        for init in INITS
        for t in target_sets
        for head in HEADS
    }
    nulls = {t: null_threshold(t) for t in target_sets}
    _require_final_embeddings(common.model_seeds(cfg))
    _, Y_train, labels = common.load_d1_train()
    _, Y_test = common.load_d1_test(purpose=EXTERNAL_PURPOSE)
    for (init, head, target_set), selection in selections.items():
        params = selection["params"]
        seeds = external_seeds(init, head, cfg)
        X_train, X_test = final_inputs(
            params["pooling"], Y_train.index, Y_test.index, init=init, init_seeds=seeds
        )
        result = protocol.evaluate_external(
            builder(head, cfg, target_set, init),
            params,
            X_train,
            Y_train,
            labels,
            X_test,
            Y_test,
            cfg,
            seeds=seeds,
            null_threshold=nulls[target_set],
            n_jobs=n_jobs,
        )
        row = protocol.record_external(
            result,
            experiment=EXPERIMENT,
            model=MODEL_NAMES[init][head],
            target_set=target_set,
            data=DATA_NAMES[init],
            params=params,
            cfg=cfg,
            n_train=len(Y_train),
            cv_mean_rho=selection["cv_mean_rho"],
            extra_tags=mgm_io.mgm_env_tags(),
        )
        print(
            f"[{EXPERIMENT}] {MODEL_NAMES[init][head]} {target_set}: external mean rho "
            f"{row['mean_rho_mean']:.4f} +/- {row['mean_rho_sd']:.4f} over {row['n_seeds']} "
            f"seed(s), 95% CI [{row['mean_rho_ci_low']:.4f}, {row['mean_rho_ci_high']:.4f}] "
            f"(CV {selection['cv_mean_rho']:.4f})"
        )


def curve(cfg: Mapping[str, Any], *, n_jobs: int = 1) -> None:
    """E5 learning curve (saved PRISM subsets) for the pretrained head with the higher CV mean
    rho (the brief's "best E2"; the random control has no curve)."""
    target_set = cfg["learning_curve"]["target_set"]
    head, selection = best_head(target_set)
    null = null_threshold(target_set)
    _require_final_embeddings()
    _, Y_train, labels = common.load_d1_train()
    _, Y_test = common.load_d1_test(purpose=CURVE_PURPOSE)
    params = selection["params"]
    X_train, X_test = final_inputs(params["pooling"], Y_train.index, Y_test.index)
    points = common.run_learning_curve(
        builder(head, cfg, target_set),
        params,
        X_train,
        Y_train,
        labels,
        X_test,
        Y_test,
        cfg,
        model=MODEL_NAMES[PRETRAINED][head],
        data_name=DATA_NAMES[PRETRAINED],
        experiment=EXPERIMENT,
        null_threshold=null,
        n_jobs=n_jobs,
    )
    by_size = points.groupby("n_train")["mean_rho"].agg(["mean", "std"])
    print(
        f"[{EXPERIMENT}] learning curve of {MODEL_NAMES[PRETRAINED][head]} ({target_set}):\n"
        f"{by_size}"
    )


def best_head(target_set: str) -> tuple[str, dict[str, Any]]:
    """The pretrained head whose saved selection has the higher CV mean rho; ties go to the
    first of HEADS."""
    names = SELECTION_NAMES[PRETRAINED]
    selections = {h: protocol.load_selection(names[h], target_set) for h in HEADS}
    head = max(HEADS, key=lambda h: selections[h]["cv_mean_rho"])  # max keeps the first maximum
    return head, selections[head]


def external_seeds(init: str, head: str, cfg: Mapping[str, Any]) -> list[int]:
    """Model seeds of the final fits.

    The pretrained ridge head has no randomness, so one fit is enough. The random control's
    ridge has: model seed s reads the network of init seed s, so it is refit for every seed and
    the reported SD over seeds covers the randomness of the initialisation.
    """
    if init == PRETRAINED and head == "ridge":
        return list(RIDGE_SEEDS)
    return common.model_seeds(cfg)


def null_threshold(target_set: str) -> float | None:
    """E0's D1 permutation-null threshold (same targets and test samples as D1+MGM).

    A real evaluation without it fails, before the external cohort is opened. A dry run may lack
    it (E0 is a separate experiment) and then reports n_above_null as NaN.
    """
    try:
        return protocol.load_null_threshold(NULL_DATA, target_set)
    except (FileNotFoundError, KeyError):
        if not common.dry_run():
            raise
        print(f"[{EXPERIMENT}] dry run without a {NULL_DATA} {target_set} null: n_above_null=NaN")
        return None


def final_inputs(
    pooling: str,
    train_ids: pd.Index,
    test_ids: pd.Index,
    *,
    init: str = PRETRAINED,
    init_seeds: Sequence[int] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(PRISM, external) inputs for one pooling, row-aligned with the metabolite tables."""
    X_train = _rows(embedding_inputs(PRISM, pooling, init, init_seeds), train_ids, PRISM)
    X_test = external_embeddings(pooling, init, init_seeds)
    extra = X_test.index.difference(test_ids)
    if len(extra):
        raise ValueError(f"{len(extra)} external embeddings have no metabolome: stale file?")
    return X_train, _rows(X_test, test_ids, "external cohort")


def _require_final_embeddings(init_seeds: Sequence[int] = ()) -> None:
    """Every embedding file a final stage reads (MGM's, plus the random networks of
    `init_seeds`, for PRISM and the external cohort) exists before that cohort is opened."""
    cohorts = (PRISM,) if common.dry_run() else (PRISM, NLIBD)
    missing = [
        embeddings_path(cohort, seed).name
        for cohort in cohorts
        for seed in (None, *init_seeds)
        if not embeddings_path(cohort, seed).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} missing in {mgm_dir()}: run the {EXPERIMENT} prepare stage first"
        )
