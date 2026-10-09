#!/usr/bin/env bash
# Fetch the three public sources (D1, D2, D3) into data/raw/.
# Idempotent: an existing clone is checked against its pinned commit instead of re-downloaded.
# data/ is gitignored, so nothing fetched here is ever committed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${ROOT}/data/raw"
mkdir -p "${RAW}"

# Commits pinned on 2026-10-08 so the shape asserts in src/m2m/data.py stay meaningful.
# Set an *_SHA to an empty string to track the default branch instead.
MIMENET_URL="https://github.com/YDaiLab/MiMeNet.git"
MIMENET_SHA="${MIMENET_SHA-f9064b54d61c40d12207db896bc137341969aef4}"
CURATED_URL="https://github.com/borenstein-lab/microbiome-metabolome-curated-data.git"
CURATED_SHA="${CURATED_SHA-89a519d8c832008fbc6e650453e83e2f04858d02}"
MGM_URL="https://github.com/HUST-NingKang-Lab/MGM.git"
MGM_SHA="${MGM_SHA-bee1469fe13116d1d09e74a6169b29f6cd5a3c26}"

CURATED_DATASETS=(
  FRANZOSA_IBD_2019
  iHMP_IBDMDB_2019
  YACHIDA_CRC_2019
  WANG_ESRD_2020
  ERAWIJANTARI_GASTRIC_CANCER_2020
  MARS_IBS_2020
)

log() { printf '[download_data] %s\n' "$*"; }

# Shallow-clone $1 into $2 and, if $3 is non-empty, move to that exact commit.
shallow_clone() {
  local url="$1" dest="$2" sha="$3"
  if [[ ! -d "${dest}/.git" ]]; then
    log "cloning ${url} -> ${dest}"
    git clone --depth 1 --quiet "${url}" "${dest}"
  fi
  pin_commit "${dest}" "${sha}"
}

pin_commit() {
  local dest="$1" sha="$2"
  if [[ -z "${sha}" ]]; then
    log "$(basename "${dest}") at $(git -C "${dest}" rev-parse HEAD) (unpinned)"
    return
  fi
  if [[ "$(git -C "${dest}" rev-parse HEAD)" != "${sha}" ]]; then
    log "checking out pinned ${sha} in $(basename "${dest}")"
    git -C "${dest}" fetch --depth 1 --quiet origin "${sha}"
    git -C "${dest}" checkout --quiet "${sha}"
  fi
  log "$(basename "${dest}") pinned at ${sha}"
}

# D1: MiMeNet repository (~27 MB); only data/IBD/ is used.
shallow_clone "${MIMENET_URL}" "${RAW}/MiMeNet" "${MIMENET_SHA}"

# D2: the full curated-data repo is ~1.6 GB, so fetch blobs lazily and check out six folders only.
CURATED_DIR="${RAW}/microbiome-metabolome-curated-data"
if [[ ! -d "${CURATED_DIR}/.git" ]]; then
  log "sparse-cloning ${CURATED_URL}"
  git clone --filter=blob:none --sparse --quiet "${CURATED_URL}" "${CURATED_DIR}"
fi
sparse_paths=()
for ds in "${CURATED_DATASETS[@]}"; do sparse_paths+=("data/processed_data/${ds}"); done
git -C "${CURATED_DIR}" sparse-checkout set "${sparse_paths[@]}"
if [[ -n "${CURATED_SHA}" && "$(git -C "${CURATED_DIR}" rev-parse HEAD)" != "${CURATED_SHA}" ]]; then
  git -C "${CURATED_DIR}" fetch --quiet origin "${CURATED_SHA}"
  git -C "${CURATED_DIR}" checkout --quiet "${CURATED_SHA}"
fi
log "microbiome-metabolome-curated-data at $(git -C "${CURATED_DIR}" rev-parse HEAD)"

# D3: MGM ships its weights and tokenizer inside the repository (mgm/resources/).
shallow_clone "${MGM_URL}" "${RAW}/MGM" "${MGM_SHA}"

log "done; run 'make check-data' to assert every fact in section 3 of the brief"
