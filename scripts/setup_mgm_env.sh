#!/usr/bin/env bash
# Build the separate, pinned Python environment for MGM (brief section 3, D3) in envs/mgm/.venv.
#
# MGM needs 2023-era libraries (torch 2.0.1, transformers 4.33.3, pandas 2.0.3) that cannot share
# the main environment. Scripts that run MGM (scripts/mgm_embed.py, scripts/mgm_finetune.py) use
# this interpreter and import m2m.models / m2m.train from this repository. See docs/mgm_notes.md.
#
# Idempotent: re-running reuses the venv, re-applies the pins, reinstalls MGM and m2m and repeats
# the smoke test. Needs data/raw/MGM (scripts/download_data.sh) and network or a warm package cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${MGM_VENV:-${ROOT}/envs/mgm/.venv}"   # MGM_VENV redirects the venv, e.g. to test this script
REQUIREMENTS="${ROOT}/envs/mgm/requirements.txt"
MGM_SRC="${ROOT}/data/raw/MGM"
PYTHON_VERSION="3.11"
# requirements.txt is a complete lock for this platform only. Elsewhere torch 2.0.1 needs extra
# wheels (CUDA libraries and triton on Linux), so the pins are installed with dependency resolution.
LOCK_PLATFORM="Darwin-arm64"

log() { printf '[setup_mgm_env] %s\n' "$*"; }
die() { printf '[setup_mgm_env] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: scripts/setup_mgm_env.sh [--recreate]
  --recreate   delete envs/mgm/.venv first and build it from scratch
Environment: MGM_VENV=<dir> builds the venv there instead (used to test this script).
While the script runs, MGM jobs using the venv can fail: it reinstalls torch, MGM and m2m.
EOF
}

# Every argument is checked before anything runs: a stray word must not be ignored next to
# --recreate, which deletes the venv.
recreate=0
for arg in "$@"; do
  case "${arg}" in
    --recreate) recreate=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: ${arg}" ;;
  esac
done

[[ -f "${MGM_SRC}/setup.py" ]] || die "no MGM clone at ${MGM_SRC}; run scripts/download_data.sh"
# Step 1 may delete VENV; with MGM_VENV pointing at the wrong folder that must not happen.
if [[ -e "${VENV}" && ! -f "${VENV}/pyvenv.cfg" && -n "$(ls -A "${VENV}" 2>/dev/null)" ]]; then
  die "${VENV} exists but is not a virtual environment; refusing to delete it"
fi

have_uv=0
command -v uv >/dev/null 2>&1 && have_uv=1

venv_python_version() {
  "${VENV}/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true
}

# Install into the venv with uv when available, otherwise with the venv's own pip.
venv_install() {
  if ((have_uv)); then
    uv pip install --python "${VENV}" "$@"
  else
    "${VENV}/bin/python" -m pip install --disable-pip-version-check "$@"
  fi
}

# --- 1. virtual environment ----------------------------------------------------------------------
if [[ -e "${VENV}" ]] && { ((recreate)) || [[ "$(venv_python_version)" != "${PYTHON_VERSION}" ]]; }
then
  log "removing ${VENV}"
  rm -rf "${VENV}"
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  log "creating ${VENV} with Python ${PYTHON_VERSION}"
  if ((have_uv)); then
    uv venv --python "${PYTHON_VERSION}" "${VENV}"
  elif command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
    "python${PYTHON_VERSION}" -m venv "${VENV}"
  else
    die "need uv or python${PYTHON_VERSION} on PATH"
  fi
fi
# A venv made by uv has no pip; the pip fallback needs one.
if ((!have_uv)) && ! "${VENV}/bin/python" -m pip --version >/dev/null 2>&1; then
  "${VENV}/bin/python" -m ensurepip --upgrade
fi
log "using $("${VENV}/bin/python" --version) at ${VENV}"

# --- 2. pinned dependencies ----------------------------------------------------------------------
if ((have_uv)) && [[ "$(uname -s)-$(uname -m)" == "${LOCK_PLATFORM}" ]]; then
  # Exact match to the lock; this also removes MGM and m2m, which steps 3 and 4 reinstall.
  log "syncing the venv to envs/mgm/requirements.txt"
  uv pip sync --python "${VENV}" "${REQUIREMENTS}"
else
  log "installing envs/mgm/requirements.txt (the lock was verified on ${LOCK_PLATFORM} only)"
  venv_install -r "${REQUIREMENTS}"
fi

# --- 3. MGM from the local clone, without its dependencies ---------------------------------------
# A regular (not editable) install: MGM finds its weights, tokenizer and phylogeny table through
# pkg_resources, which then resolves inside site-packages, independent of data/raw. The build runs
# on a throwaway copy of the files setup.py needs: an in-place build writes build/ and *.egg-info
# into data/raw/MGM, and the copy leaves out the __pycache__ folders the upstream repo commits.
build_dir="$(mktemp -d "${TMPDIR:-/tmp}/mgm-build.XXXXXX")"
trap 'rm -rf "${build_dir}"' EXIT
tar -C "${MGM_SRC}" --exclude=__pycache__ -cf - \
  setup.py MANIFEST.in README.md requirements.txt LICENSE mgm | tar -C "${build_dir}" -xf -
mgm_commit="$(git -C "${MGM_SRC}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "installing MGM (commit ${mgm_commit}) without dependencies"
if ((have_uv)); then
  uv pip install --python "${VENV}" --no-deps --reinstall-package microformer-mgm "${build_dir}"
else
  venv_install --no-deps --force-reinstall "${build_dir}"
fi

# --- 4. this repository, without dependencies (m2m.models / m2m.train run here) ------------------
log "installing m2m (editable) without dependencies"
venv_install --no-deps -e "${ROOT}"

# --- 5. smoke test -------------------------------------------------------------------------------
log "smoke test"
"${VENV}/bin/python" -I - "${ROOT}" <<'PY'
"""MGM resources, tokenizer, input pipeline and pretrained weights; m2m imports from this repo."""
import contextlib
import importlib
import importlib.util
import io
import math
import pickle
import platform
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

with warnings.catch_warnings():  # importing mgm runs warnings.filterwarnings("ignore") globally
    import mgm
    from mgm.CLI.CLI_utils import find_pkg_resource
    from mgm.src.MicroCorpus import MicroCorpus
    from mgm.src.utils import CustomUnpickler

import pandas as pd
import torch
import transformers
from transformers import GPT2Model

# Facts verified on 2026-10-08 (docs/mgm_notes.md); a failure means the MGM clone or a pin changed.
VOCAB_SIZE = 9669  # 4 special tokens + 9,665 genera
SPECIAL_IDS = {"<pad>": 0, "<mask>": 1, "<bos>": 2, "<eos>": 3}
ARCHITECTURE = {
    "n_layer": 8, "n_embd": 256, "n_head": 8, "n_positions": 512, "vocab_size": VOCAB_SIZE,
}
N_PARAMS = 8_924_928  # GPT2Model, i.e. without the language-model head
TOKENIZER_GLOBALS = {
    ("src.MicroCorpus", "MicroTokenizer"),  # redirected to mgm.src.MicroCorpus by CustomUnpickler
    ("transformers.tokenization_utils", "Trie"),
}
repo_root = Path(sys.argv[1]).resolve()


def check(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"SMOKE TEST FAILED: {message}")
    print(f"  ok  {message}")


class AllowlistUnpickler(CustomUnpickler):
    """MGM's tokenizer unpickler, refusing any class the pickle is not known to contain."""

    def find_class(self, module: str, name: str):
        if (module, name) not in TOKENIZER_GLOBALS:
            raise pickle.UnpicklingError(f"unexpected object in the tokenizer: {module}.{name}")
        return super().find_class(module, name)


site_packages = Path(torch.__file__).resolve().parents[1]
resources = Path(find_pkg_resource("resources/phylogeny.csv")).parent
check(resources.is_relative_to(site_packages),
      f"MGM {mgm.__version__} resources resolve to {resources}")

with open(resources / "MicroTokenizer.pkl", "rb") as fh:
    tokenizer = AllowlistUnpickler(fh).load()
check(len(tokenizer.vocab) == VOCAB_SIZE, f"tokenizer loads with {VOCAB_SIZE} tokens")
check(all(tokenizer.vocab[tok] == i for tok, i in SPECIAL_IDS.items()),
      f"special ids {SPECIAL_IDS}")

# MicroCorpus on one hand-made sample (docs/mgm_notes.md section 2). E2/E3 rely on these rules:
# abundances are renormalised within MGM's vocabulary (the family-level column drops out),
# genera are ordered by corpus z-score rather than abundance, and <pad> fills the right side.
toy = pd.DataFrame({"g__Bacteroides": [0.40], "g__Faecalibacterium": [0.20],
                    "g__Escherichia": [0.01], "f__Lachnospiraceae": [0.39]}, index=["toy"])
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    corpus = MicroCorpus(tokenizer=tokenizer, abu=toy.copy(), preprocess=True)
stats = pd.read_csv(resources / "phylogeny.csv", index_col=0).loc["g__Bacteroides"]
in_vocab_total = toy.drop(columns="f__Lachnospiraceae").loc["toy"].sum()
share = toy.loc["toy", "g__Bacteroides"] / in_vocab_total
check(math.isclose(corpus.data.loc["toy", "g__Bacteroides"],
                   (share - stats["mean"]) / stats["std"], rel_tol=1e-9),
      "MicroCorpus renormalises within the vocabulary, then z-scores with phylogeny.csv")
by_z_score = ["<bos>", "g__Escherichia", "g__Faecalibacterium", "g__Bacteroides", "<eos>"]
row = corpus.tokens[0]
check(row[:len(by_z_score)].tolist() == tokenizer.encode(by_z_score)
      and not bool(row[len(by_z_score):].any()) and row.numel() == ARCHITECTURE["n_positions"],
      "MicroCorpus orders genera by z-score, not abundance, and right-pads with <pad> to 512")

model_dir = resources / "general_model"
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="TypedStorage is deprecated")  # torch 2.0.1 internals
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
check(torch.equal(state["lm_head.weight"], state["transformer.wte.weight"]),
      "checkpoint is a plain state dict; its LM head is tied to the token embeddings")

transformers.logging.set_verbosity_error()  # the loading report is checked explicitly instead
model, info = GPT2Model.from_pretrained(str(model_dir), output_loading_info=True)
check(not (info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]),
      "GPT2Model.from_pretrained finds every weight it needs")
check(info["unexpected_keys"] == ["lm_head.weight"], "only lm_head.weight is left unused")
config = {key: getattr(model.config, key) for key in ARCHITECTURE}
check(config == ARCHITECTURE, f"architecture {config}")
n_params = sum(p.numel() for p in model.parameters())
check(n_params == N_PARAMS, f"{n_params:,} parameters")

sentence = ["<bos>", "g__Bacteroides", "g__Faecalibacterium", "g__Prevotella", "<eos>"]
ids = torch.tensor([tokenizer.encode(sentence)])
model.eval()
with torch.no_grad():
    first = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
    second = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
check(tuple(first.shape) == (1, len(sentence), 256) and bool(torch.isfinite(first).all()),
      "forward pass gives finite hidden states of shape (1, 5, 256)")
check(torch.equal(first, second), "eval-mode forward pass is deterministic")

import m2m  # noqa: E402

check(Path(m2m.__file__).resolve().is_relative_to(repo_root / "src"),
      f"m2m imports from {repo_root}")
for name in ("m2m.paths", "m2m.config", "m2m.models", "m2m.train"):
    if importlib.util.find_spec(name) is None:
        print(f"  --  {name} does not exist yet; skipped")
        continue
    try:
        importlib.import_module(name)
        error = ""
    except Exception as exc:  # any import failure means MGM scripts cannot use this module
        error = f" -> {type(exc).__name__}: {exc}"
    check(not error, f"{name} imports in the MGM env{error}")

packages = ("torch", "transformers", "numpy", "pandas")
versions = ", ".join(f"{pkg} {version(pkg)}" for pkg in packages)
print(f"MGM environment OK: Python {platform.python_version()}, {versions}, MGM {mgm.__version__}")
PY

log "done; run MGM code with ${VENV}/bin/python"
