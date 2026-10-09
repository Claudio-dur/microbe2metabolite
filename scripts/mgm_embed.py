"""Tokenise genus tables with MGM's own MicroCorpus and embed them with a frozen GPT-2 backbone.

Runs inside envs/mgm (torch 2.0.1, transformers 4.33.3, pandas 2.0.3), never in the main env:

    envs/mgm/.venv/bin/python scripts/mgm_embed.py \
        --inputs data/processed/mgm/inputs --out data/processed/mgm \
        --model-dir data/raw/MGM/mgm/resources/general_model [--init random --init-seeds 0,1,2]

For every ``<cohort>_genus.csv`` in --inputs (samples × ``g__Genus`` columns, written by
m2m.experiments.mgm_frozen) it writes, in --out:

--init pretrained (default): MGM's pretrained weights
  <cohort>_tokens.npz      sample_ids, input_ids, attention_mask (padding trimmed to the longest
                           sample; consumed by E3 fine-tuning)
  <cohort>_embeddings.npz  sample_ids, emb_mean, emb_last (last-layer hidden states averaged over
                           non-padding tokens / taken at the final <eos> token; consumed by E2)
  <cohort>_manifest.json   provenance: versions, model dir, token-length statistics

--init random --init-seeds S: E2's H1 control, MGM's architecture with random weights
  <cohort>_embeddings_random_seed<s>.npz  the same arrays for every seed s in S: the network is
                           GPT2Model(GPT2Config.from_pretrained(--model-dir)), i.e. transformers'
                           standard GPT-2 initialisation, built right after torch.manual_seed(s)
  <cohort>_manifest_random.json           provenance of those files
  The tokens are the pretrained run's: they are recomputed, must equal <cohort>_tokens.npz and
  are not rewritten, so run --init pretrained first.

Nothing is fitted here: MGM's normalisation uses its own corpus statistics (phylogeny.csv), so
embedding the external cohort cannot leak information into training.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import platform
import sys
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from m2m.models import load_mgm_backbone, pool_hidden_states

REPO_ROOT = Path(__file__).resolve().parents[1]
MGM_RESOURCES = REPO_ROOT / "data" / "raw" / "MGM" / "mgm" / "resources"

PRETRAINED, RANDOM = "pretrained", "random"
INITS = (PRETRAINED, RANDOM)
GENUS_SUFFIX = "_genus.csv"

# The tokenizer pickle comes from a downloaded repository. Unpickling can execute arbitrary
# code, so only the two globals the file is known to reference are allowed (docs/mgm_notes.md §3).
TOKENIZER_GLOBALS = {
    ("src.MicroCorpus", "MicroTokenizer"),
    ("transformers.tokenization_utils", "Trie"),
}


class CohortTokens(NamedTuple):
    """MGM's token ids of one cohort, one row per sample, right-padded."""

    sample_ids: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def import_mgm():
    """Import MGM's classes without letting its global `warnings.filterwarnings("ignore")` stick."""
    with warnings.catch_warnings():
        from mgm.src.MicroCorpus import MicroCorpus
        from mgm.src.utils import CustomUnpickler
    return MicroCorpus, CustomUnpickler


def load_tokenizer(resources: Path):
    """MGM pickles its tokenizer class; CustomUnpickler maps the class name back, we allowlist."""
    _, CustomUnpickler = import_mgm()

    class AllowlistUnpickler(CustomUnpickler):
        def find_class(self, module: str, name: str):
            if (module, name) not in TOKENIZER_GLOBALS:
                raise pickle.UnpicklingError(f"unexpected global in tokenizer: {module}.{name}")
            return super().find_class(module, name)

    with (resources / "MicroTokenizer.pkl").open("rb") as fh:
        return AllowlistUnpickler(fh).load()


def tokenize(
    genus_table: pd.DataFrame, tokenizer, resources: Path, max_len: int
) -> CohortTokens:
    """MGM's own preprocessing + tokenisation; ids and masks trimmed to the longest row."""
    MicroCorpus, _ = import_mgm()

    # MicroCorpus prints a progress bar and summary lines; keep this script's stdout readable.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        corpus = MicroCorpus(
            tokenizer=tokenizer,
            abu=genus_table.copy(),
            phylogeny_path=str(resources / "phylogeny.csv"),
            max_len=max_len,
            preprocess=True,
        )
    kept = [str(s) for s in corpus.data.index]
    dropped = sorted(set(map(str, genus_table.index)) - set(kept))
    if dropped:
        # MGM silently drops samples with no in-vocabulary genus; we would rather fail loudly.
        raise RuntimeError(f"MGM dropped {len(dropped)} all-zero samples, e.g. {dropped[:5]}")
    input_ids = corpus.tokens
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    width = int(attention_mask.sum(dim=1).max())
    return CohortTokens(
        kept, input_ids[:, :width].contiguous(), attention_mask[:, :width].contiguous()
    )


def random_backbone(model_dir: Path, seed: int) -> nn.Module:
    """MGM's GPT-2 architecture (its config.json) with transformers' standard initialisation.

    Seeded right before construction, so a seed always gives the same weights. E3's scratch
    control (scripts/mgm_finetune.py) builds its backbone the same way after seeding with the
    model seed: this frozen network of seed s is that network of seed s before any training.
    """
    from transformers import GPT2Config, GPT2Model  # only envs/mgm has transformers

    torch.manual_seed(seed)
    return GPT2Model(GPT2Config.from_pretrained(str(model_dir))).eval()


@torch.no_grad()
def embed(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen forward pass; mean and last-token pooling share models.pool_hidden_states with E3."""
    model.eval()
    means, lasts = [], []
    for start in range(0, len(input_ids), batch_size):
        ids = input_ids[start : start + batch_size]
        mask = attention_mask[start : start + batch_size]
        hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
        means.append(pool_hidden_states(hidden, mask, "mean"))
        lasts.append(pool_hidden_states(hidden, mask, "last"))
    return torch.cat(means).numpy(), torch.cat(lasts).numpy()


# --- outputs ---------------------------------------------------------------------------------


def tokens_path(out: Path, cohort: str) -> Path:
    return out / f"{cohort}_tokens.npz"


def embeddings_path(out: Path, cohort: str, init_seed: int | None = None) -> Path:
    """The pretrained embeddings, or (with `init_seed`) those of one random-weight network."""
    if init_seed is None:
        return out / f"{cohort}_embeddings.npz"
    return out / f"{cohort}_embeddings_random_seed{init_seed}.npz"


def save_embeddings(path: Path, tokens: CohortTokens, model, batch_size: int) -> int:
    """Embed one cohort and write sample_ids, emb_mean and emb_last; returns the width."""
    emb_mean, emb_last = embed(model, tokens.input_ids, tokens.attention_mask, batch_size)
    np.savez_compressed(
        path,
        sample_ids=np.array(tokens.sample_ids, dtype=str),
        emb_mean=emb_mean,
        emb_last=emb_last,
    )
    return int(emb_mean.shape[1])


def require_same_tokens(path: Path, tokens: CohortTokens) -> None:
    """The random-weight control must embed exactly the tokens of the pretrained run, which are
    also the tokens E3 trains on: otherwise the control would differ in more than its weights."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run --init pretrained first (the random-weight control embeds the "
            "pretrained run's tokens)"
        )
    with np.load(path, allow_pickle=False) as saved:
        same = (
            [str(s) for s in saved["sample_ids"]] == tokens.sample_ids
            and np.array_equal(saved["input_ids"], tokens.input_ids.numpy())
            and np.array_equal(saved["attention_mask"], tokens.attention_mask.numpy())
        )
    if not same:
        raise RuntimeError(
            f"{path} holds other tokens than these genus tables give: re-run --init pretrained"
        )


def write_manifest(
    path: Path, cohort: str, tokens: CohortTokens, embedding_dim: int, args: argparse.Namespace
) -> dict:
    lengths = tokens.attention_mask.sum(dim=1).numpy()
    manifest = {
        "cohort": cohort,
        "init": args.init,
        "n_samples": len(tokens.sample_ids),
        "tokens_per_sample": {"min": int(lengths.min()), "mean": float(lengths.mean()),
                              "max": int(lengths.max())},
        "embedding_dim": embedding_dim,
        "model_dir": str(args.model_dir),
        "max_len": args.max_len,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "pandas": pd.__version__,
    }
    if args.init == RANDOM:
        manifest["init_seeds"] = list(args.init_seeds)
        manifest["files"] = [embeddings_path(path.parent, cohort, s).name for s in args.init_seeds]
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def embed_pretrained(cohorts: dict[str, CohortTokens], args: argparse.Namespace) -> None:
    """Tokens, pretrained embeddings and manifest of every cohort."""
    model = load_mgm_backbone(args.model_dir)
    for cohort, tokens in cohorts.items():
        np.savez_compressed(
            tokens_path(args.out, cohort),
            sample_ids=np.array(tokens.sample_ids, dtype=str),
            input_ids=tokens.input_ids.numpy(),
            attention_mask=tokens.attention_mask.numpy(),
        )
        width = save_embeddings(embeddings_path(args.out, cohort), tokens, model, args.batch_size)
        manifest = write_manifest(
            args.out / f"{cohort}_manifest.json", cohort, tokens, width, args
        )
        print(f"[mgm_embed] {cohort}: {manifest['n_samples']} samples, "
              f"tokens/sample {manifest['tokens_per_sample']}")


def embed_random(cohorts: dict[str, CohortTokens], args: argparse.Namespace) -> None:
    """Embeddings of every cohort with one random-weight network per init seed."""
    for cohort, tokens in cohorts.items():  # checked before any network is built
        require_same_tokens(tokens_path(args.out, cohort), tokens)
    widths: dict[str, int] = {}
    for seed in args.init_seeds:
        model = random_backbone(args.model_dir, seed)
        for cohort, tokens in cohorts.items():
            path = embeddings_path(args.out, cohort, seed)
            widths[cohort] = save_embeddings(path, tokens, model, args.batch_size)
    for cohort, tokens in cohorts.items():
        write_manifest(args.out / f"{cohort}_manifest_random.json", cohort, tokens,
                       widths[cohort], args)
        print(f"[mgm_embed] {cohort}: {len(tokens.sample_ids)} samples embedded by "
              f"{len(args.init_seeds)} random-weight networks (init seeds {args.init_seeds})")


# --- command line ----------------------------------------------------------------------------


def seed_list(text: str) -> list[int]:
    """'0,1,2' -> [0, 1, 2]: distinct non-negative integers, in the order given."""
    try:
        seeds = [int(part) for part in text.split(",")]
    except ValueError:
        message = f"expected comma-separated integers, got {text!r}"
        raise argparse.ArgumentTypeError(message) from None
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(f"seeds must be distinct and >= 0, got {text!r}")
    return seeds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MGM_RESOURCES / "general_model")
    parser.add_argument("--resources", type=Path, default=MGM_RESOURCES)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--threads", type=int, default=4, help="torch threads (overrides OMP_NUM_THREADS)"
    )
    parser.add_argument(
        "--init", choices=INITS, default=PRETRAINED,
        help="MGM's pretrained weights, or its architecture with random weights (E2's control)",
    )
    parser.add_argument(
        "--init-seeds", type=seed_list, default=None,
        help="--init random only: comma-separated seeds, one random network each (e.g. 0,1,2)",
    )
    args = parser.parse_args(argv)
    if (args.init == RANDOM) != (args.init_seeds is not None):
        parser.error("--init-seeds is required with --init random and refused otherwise")

    # torch 2.0.1 ignores thread changes after its first parallel work, so set them first.
    torch.set_num_threads(args.threads)
    # torch 2.0.1's own weights_only loader trips its TypedStorage deprecation; nothing to act on.
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = sorted(args.inputs.glob(f"*{GENUS_SUFFIX}"))
    if not inputs:
        sys.exit(f"no *{GENUS_SUFFIX} files in {args.inputs}")
    tokenizer = load_tokenizer(args.resources)
    cohorts = {}
    for path in inputs:
        table = pd.read_csv(path, index_col=0)
        table.index = table.index.astype(str)
        cohort = path.name.removesuffix(GENUS_SUFFIX)
        cohorts[cohort] = tokenize(table, tokenizer, args.resources, args.max_len)
    if args.init == PRETRAINED:
        embed_pretrained(cohorts, args)
    else:
        embed_random(cohorts, args)


if __name__ == "__main__":
    main()
