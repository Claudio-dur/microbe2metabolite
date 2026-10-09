# MGM notes: environment, input pipeline, model, speed

Everything here was checked on 2026-10-08 against MGM commit `bee1469` (microformer-mgm 0.5.8) in
`envs/mgm/.venv`, then re-checked the same day by an independent audit. That audit re-measured
section 5 and corrected the padding statement in section 4. Paths like `MicroCorpus.py:128` point
into `data/raw/MGM/mgm/` (`src/` or `CLI/`). The audit scripts were throwaway. The facts that can
break are re-asserted by the smoke test in `scripts/setup_mgm_env.sh`: tokenizer, MicroCorpus
renormalisation, ordering and padding, weights, and architecture.

## 1. Environment

```bash
scripts/setup_mgm_env.sh              # build or refresh (idempotent), ends with a smoke test
scripts/setup_mgm_env.sh --recreate   # delete envs/mgm/.venv and rebuild from scratch
envs/mgm/.venv/bin/python scripts/mgm_embed.py ...   # run MGM code with this interpreter
```

Call the interpreter directly. `uv run` would sync the main project's environment instead.

- **Do not run the script while MGM jobs are running.** `--recreate` leaves no environment for
  about 25 s. A plain re-run also uninstalls and reinstalls torch, MGM and m2m for a few seconds
  (see Quirks).
- Every argument is checked before anything runs, so a stray word next to `--recreate` stops the
  script. It also refuses to delete an existing `MGM_VENV` folder that is not a virtual
  environment.

**Pins** (`envs/mgm/requirements.txt`, a full lock of 47 packages, CPython 3.11):
- MGM's own pins, unchanged: numpy 1.24.3, pandas 2.0.3, pytorch_lightning 2.0.6,
  scikit_learn 1.3.1, setuptools 68.0.0, torch 2.0.1, tqdm 4.65.0, transformers 4.33.3.
- accelerate 0.23.0. MGM asks for >=0.20.1, and transformers 4.33.3's `accelerate` extra asks for
  >=0.20.3. 0.23.0 came out the same month.
- scipy 1.11.4 (supports numpy <1.28) and pyyaml 6.0.3.
- Hugging Face packages: tokenizers 0.13.3, huggingface-hub 0.17.3, safetensors 0.3.3.
- sympy 1.12 and mpmath 1.3.0, because mpmath 1.4 raises DeprecationWarnings inside sympy 1.12.
- torchmetrics 1.0.3 and lightning-utilities 0.9.0, matched to lightning 2.0.6.
- Every pinned version installs from a wheel on macOS arm64, so no pin had to be replaced.
- On this platform, resolving the file reproduces its 47 pins exactly. On Linux, torch 2.0.1 also
  pulls CUDA wheels and triton, so the script resolves dependencies there instead of syncing.
- Why a separate environment: MGM's `groupby(axis=1)` (`MicroCorpus.py:129`) raises `TypeError`
  under the main env's pandas 3.0.6. Under pandas 2.0.3 it runs with no warning, even under
  `-W error`.

**How MGM is installed.** It is a regular install, not editable, built from a temporary copy of the
files `setup.py` needs. Both modes let `CLI_utils.py:18` (`find_pkg_resource`, via `pkg_resources`)
resolve `resources/`:
- **Regular:** resources resolve to `site-packages/mgm/resources/`, because MANIFEST.in ships them.
- **Editable:** resources resolve to `data/raw/MGM/mgm/resources/`. setuptools' finder maps only
  `mgm`, so the clone root is not on `sys.path`.

Regular was chosen for three reasons:
1. Building in place writes `build/` and `microformer_mgm.egg-info/` into `data/raw/MGM`.
2. Importing from the clone rewrites the committed `*.cpython-311.pyc` files and adds new ones,
   which was tested on a copy. The repository commits 21 `.pyc` files. All are timestamp-checked,
   so stale ones are recompiled rather than executed. MANIFEST.in:7-9 keeps them out of the wheel.
3. The environment then no longer depends on the clone.

Both the uv path and the pip fallback build from that temporary copy, so `data/raw/MGM` stays
untouched (`git status` was clean after both).

**Quirks**
- The torch 2.0.1 arm64 wheel (`cp311-none-macosx_11_0_arm64`) declares `macosx_11_0_x86_64` in its
  `WHEEL` metadata, although its binaries are arm64.
  - So `uv pip check` reports "built for a different platform".
  - Every `uv pip sync` also reinstalls torch, which costs about 0.5 s.
- `uv pip check` also lists m2m's main-environment pins (torch 2.14.1, pandas 3.0.6, mlflow, ...)
  as unmet. That is expected: m2m is installed with `--no-deps`. Only `m2m.models`, `m2m.train`,
  `m2m.paths` and `m2m.config` are meant to run here, and the smoke test imports each of them.
- Build times:
  - about 25 s from scratch with a warm uv cache;
  - 6-7 s when re-run;
  - 41 s through the pip fallback with a warm pip cache, 64 s when wheels must be downloaded.
  - The pip fallback installs the same 47 versions, plus pip.

## 2. MicroCorpus (`MicroCorpus.py`)

What it does, confirmed by reading the code and by running it on a hand-made table. The tokens
match an independent re-implementation for every toy and every PRISM sample.

| Step | Line | Observed behaviour |
|---|---|---|
| Genus extraction | 128 | First match of `(g__[A-Za-z0-9_]+)`. A full lineage `...;g__Bacteroides;s__...` gives `g__Bacteroides`. `f__Lachnospiraceae` gives NaN and is dropped. `g__CAG-83` gives `g__CAG`. `g__Faecalibacterium_A` and `g__Bacteroides_vulgatus` are captured whole and are not in the vocabulary. **Pass genus-level names**: species columns are not collapsed. |
| Merge duplicates | 129 | `groupby(axis=1).sum()`: two Bacteroides columns (0.30 + 0.10) become 0.40. |
| Restrict to vocabulary | 132-133 | Left-merge onto the `phylogeny.csv` index gives 9,665 columns in phylogeny order. Other genera are dropped; missing ones are set to 0. |
| Drop all-zero samples | 135-136 | Checked **after** the restriction, so a sample with only out-of-vocabulary abundance is dropped too. It is only reported by a `print`. |
| Renormalise | 140 | To relative abundance **within the vocabulary**: 0.40 / 0.61 in-vocab = 0.656. |
| z-score | 142-145 | `(x - mean) / std` with `phylogeny.csv` statistics (no zero std; min 1.95e-7). `zero_values` = z of 0 = `-mean/std`. |
| Drop zeros | 102 | `sample > zero_values` (strict), so a present genus below the corpus mean is kept (z = -0.37 > -0.43). |
| Order | 103 | By **z-score, descending, not abundance**. Toy sample: Escherichia (1.6 %, z 9.1), then Faecalibacterium (32.8 %, z 4.3), then Bacteroides (65.6 %, z 3.8). |
| Wrap | 106-112 | `<bos>` + genera + `<eos>`, then `tokenizer.encode`, which is just `convert_tokens_to_ids` with no extra special tokens. |
| Pad / truncate | 116-119 | Right-padded with `<pad>` (0) to `max_len=512`. Over 510 genera, `<eos>` is cut off (tested with 600). |
| Outputs | 87, 89-95 | `corpus.tokens` is int64 `(n, 512)`. `corpus.data` is the z-scored `(n, 9665)` table and its index gives the kept sample IDs. `corpus[i]` returns `input_ids` and a float `attention_mask`. Lengths are only printed: recompute them as `(tokens != 0).sum(1)`. |

On PRISM the z-score order differs a lot from abundance order. The first token is not the most
abundant genus in 105 of 155 samples, and Kendall's tau between token order and abundance order
has median 0.50 (min 0.12).

**Surprises**
- **Any `import mgm...` silences every warning in the process.** `CLI/main.py:7` runs
  `warnings.filterwarnings('ignore')`, and `mgm/__init__.py:1` → `CLI/__init__.py:1` import it.
  Verified: m2m.train's RuntimeWarning about an ignored thread setting disappears.
- `MicroCorpus(abu=df)` renames the caller's `df.columns` in place (line 63 keeps a reference,
  line 128 assigns).
- A sample named `zero` collides with the sentinel row (lines 142 and 145) and is silently dropped.
- A one-column table raises `TypeError`: `.squeeze()` turns the labels into a scalar (line 128).
- The default `phylogeny_path` is resolved when the module is imported (line 48), so importing
  `MicroCorpus` already needs the resources.
- Output noise:
  - Two `print`s to stdout (lines 83-86 and 136).
  - A tqdm bar to stderr (line 77). tqdm 4.65 predates the `TQDM_DISABLE` variable.
  - `corpus[i]` raises a UserWarning ("To copy construct from a tensor...", line 94).
  - `import pkg_resources` (`CLI_utils.py:4`) raises a DeprecationWarning. It is hidden by
    default and shown under pytest or `-W default`.
- No FutureWarning from pandas 2.0.3.
- `preprocess=False` keeps genera above each genus' *minimum across samples*, not above 0
  (lines 137-138). Do not use it.

**Quiet call for downstream scripts** (verified in the MGM env with `n_threads = 1`):

```python
import contextlib
import io
import pickle
import warnings

import torch

torch.set_num_threads(n_threads)  # first torch call: torch 2.0.1 ignores later changes (section 5)

with warnings.catch_warnings():  # undoes mgm's process-wide warnings.filterwarnings("ignore")
    from mgm.CLI.CLI_utils import find_pkg_resource
    from mgm.src.MicroCorpus import MicroCorpus
    from mgm.src.utils import CustomUnpickler

TOKENIZER_GLOBALS = {
    ("src.MicroCorpus", "MicroTokenizer"),
    ("transformers.tokenization_utils", "Trie"),
}

class AllowlistUnpickler(CustomUnpickler):
    def find_class(self, module, name):
        if (module, name) not in TOKENIZER_GLOBALS:
            raise pickle.UnpicklingError(f"unexpected object in the tokenizer: {module}.{name}")
        return super().find_class(module, name)

def load_tokenizer():
    with open(find_pkg_resource("resources/MicroTokenizer.pkl"), "rb") as fh:
        return AllowlistUnpickler(fh).load()

def build_corpus(genus_table, tokenizer):
    """samples x 'g__Genus' table -> MicroCorpus; its prints and progress bar come back as text."""
    log = io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        corpus = MicroCorpus(tokenizer=tokenizer, abu=genus_table.copy(), preprocess=True)
    return corpus, log.getvalue()
```

Align outputs on `corpus.data.index`, never on input order. Then trim `corpus.tokens` to the
longest real sequence (section 4).

## 3. Tokenizer (`resources/MicroTokenizer.pkl`)

- **Vocabulary:** 9,669 tokens. `<pad>`=0, `<mask>`=1, `<bos>`=2, `<eos>`=3, then the 9,665 genera
  (ids 4-9668), in exactly the row order of `phylogeny.csv`, which is alphabetical. For example,
  `g__Bacteroides` is 964, `g__Faecalibacterium` 3317, `g__Escherichia` 3180, `g__Prevotella` 7158
  and `g__Akkermansia` 266. There is no unknown-token id, so encoding an unknown genus raises
  `KeyError`. MicroCorpus filters genera before encoding, so it never hits this.
- **Vocabulary names the regex cannot produce:** 235 genus tokens contain `_` (for example
  `g__Candidatus_Accumulibacter`), so names with spaces must be converted to underscores. Three
  tokens contain characters the line-128 regex never matches (`g__Pseudo-nitzschia`, ...).
- **Loading:**
  - A plain `pickle.load` fails, because the pickle refers to `src.MicroCorpus.MicroTokenizer`.
  - `CustomUnpickler` (`utils.py:42-46`) redirects that class by name only and otherwise
    unpickles anything.
  - A static opcode scan and the allowlist run both show the file references only the two globals
    above, so loading with the allowlist is safe.
- **Weights pickle** (`general_model/pytorch_model.bin`): references only `OrderedDict`,
  `FloatStorage` and `_rebuild_tensor_v2`, so `torch.load(..., weights_only=True)` works.
  `training_args.bin` pickles TrainingArguments and accelerate objects; it is never needed.

## 4. Model (`resources/general_model`)

- **Loading:**
  - `GPT2Model.from_pretrained(dir)` loads the GPT2LMHeadModel checkpoint with
    `missing_keys=[]` and `unexpected_keys=['lm_head.weight']`. That tensor is identical to `wte`,
    so nothing is lost.
  - The report is logged at INFO, because GPT2Model is not in `config.architectures`
    (transformers `modeling_utils.py:3636-3639`). Loading is therefore silent: check
    `output_loading_info=True` instead of reading logs.
  - The model comes back in **eval mode** (`modeling_utils.py:3205`).
- **Size:** 8,924,928 parameters:
  - wte 2,475,264 and wpe 131,072;
  - 8 blocks × 789,760;
  - ln_f 512.

  The architecture is 8 layers, 256-d, 8 heads, 512 positions, `gelu_new`, dropout 0.1
  (resid/embd/attn). With a 466-output head the total is 9,044,690 trainable parameters. Training
  only the last block, ln_f and that head leaves 910,034.
- **Determinism:**
  - In eval mode, outputs are bit-identical across calls and across 1, 4 and 14 threads (separate
    processes).
  - With causal attention and right padding, padding cannot change real-token outputs in exact
    arithmetic.
  - In float32 they were bit-identical for padded lengths 48, 64 and 512, and for batch sizes 1, 8
    and 16.
  - Trimming a 13-token sample to its own length moved them by up to 1.3e-6, so bitwise equality
    needs the same padded length.
  - **Trim each batch to its longest sequence**: the result is the same up to rounding, and much
    faster (section 5).
  - Train mode is stochastic (dropout) but reproducible under the same `torch.manual_seed`.
- **Pooling:**
  - "Last token" must index the `<eos>` position (length - 1), not position -1, which is padding.
  - Mean pooling must exclude padding. `m2m.models.pool_hidden_states` does both.
  - Always pass `attention_mask`; transformers warns once otherwise.

## 5. CPU speed (E3 compute plan)

**Setup:**
- Apple M4 Max (10 performance + 4 efficiency cores), torch 2.0.1, BLAS = Accelerate, ATen
  native thread pool.
- Model: `m2m.models.MGMRegressor` with a 466-output head, after `set_trainable_blocks(8)` or `(1)`.
- Input: batch 16 of right-padded sequences of 10-48 tokens, the D1 range.
- One step is zero_grad + forward + MSE + backward + AdamW.
- Each cell is the median over 30 steps for L=64 or 8 steps for L=512, after 3 warm-up steps.
- Each cell ran in its own process. A row timed after another model in the same process ran up
  to 30 % slower.
- Other jobs kept the machine's load average at 2-5, so expect ±15 %.

| ms per step | L=64, 1 thread | L=64, 4 threads | L=512, 1 thread | L=512, 4 threads |
|---|---|---|---|---|
| train, all blocks trainable | 188 | 119 | 3,145 | 1,906 |
| train, last block (+ ln_f, head) | 92 | 53 | 1,501 | 730 |
| train, last block, cached frozen prefix | 24 | 16 | 384 | 235 |
| forward only, eval (E2 embeddings) | 76 | 41 | 1,268 | 554 |

The first audit's last-block row (133 / 87 / 2,225 / 1,380 ms) did not reproduce. In 6 re-runs
it was 86-107 ms at L=64 with 1 thread. The other rows agree with the first audit within 30 %.

- **Thread count is fixed once per process.** torch 2.0.1 (native thread pool) accepts one
  `torch.set_num_threads` call, and only before any multi-threaded work. Loading MGM's weights
  counts as such work; imports do not. Later calls print only a C++ warning, and
  `get_num_threads()` keeps its old value (14 by default here).
  - `OMP_NUM_THREADS=n` sets only the starting value and does not use up that one call. So
    `OMP_NUM_THREADS=1 ... scripts/mgm_embed.py --threads 4` runs with 4 threads, because the
    script calls `torch.set_num_threads(4)` first.
  - `m2m.train._apply_num_threads` warns about an ignored setting. mgm's global ignore filter
    swallows that warning unless mgm was imported inside `warnings.catch_warnings()`.
  - Fix, either:
    - call `torch.set_num_threads` before anything else; or
    - launch with `OMP_NUM_THREADS=n` and never call it.
  - Both were verified. The main env's torch 2.14.1 (OpenMP backend) does not have this
    limitation.
- **Parallel workers.** Aggregate throughput at L=64, all blocks:
  - one process with 4 threads: 8.4 steps/s;
  - two such processes at once: 12.3 steps/s (163 ms each);
  - eight 1-thread processes at once: 22.0-22.6 steps/s (350-370 ms each);
  - `VECLIB_MAXIMUM_THREADS=1` on those eight made no clear difference (22.8 steps/s).
- **Plan:**
  - D1 sequences are at most 48 tokens, so always trim to the batch maximum: L=64 instead of 512 is
    13-17 times cheaper.
  - Training only the last block (+ ln_f, head) roughly halves a step (51-62 % less). Backward and
    AdamW stop at the last block, but the forward pass still runs through all 8 blocks.
  - For k=1 or k=2, run the frozen blocks once per fit and cache their output.
    - A k=1 step then drops from 53 to 16 ms (L=64, 4 threads), 3-4 times faster.
    - The cache is exact because `MGMRegressor.train()` keeps frozen blocks, including embedding
      dropout, in eval mode.
    - The cached path matched the full model to 0.0 in both audits.
  - One fit takes epochs × ceil(n_inner_train / 16) steps, about 7 per epoch on a PRISM fold. For
    example, 100 epochs of full fine-tuning take about 1.5 min in one 4-thread process, validation
    included.
  - E2 embeds all 155 PRISM samples in under a second.
  - The Apple-GPU (MPS) backend exists but was not tested.

## 6. D1 check (PRISM training cohort only; no external file was read)

`data/raw/MiMeNet/data/IBD/microbiome_PRISM.csv` was transposed, and species (`Genus_species`) were
summed into `g__Genus` by their first token. MicroCorpus was then run with `preprocess=True`.

- **Genera:** 67 genera, of which 59 are in the vocabulary. The 8 outside it are higher ranks:
  Lachnospiraceae, Erysipelotrichaceae, Peptostreptococcaceae, Ruminococcaceae, Bacteroidales,
  Burkholderiales, Clostridiales and Clostridiaceae. Lachnospiraceae has the largest share, 1.9 %
  of abundance on average. 58 genera appear in the tokens: `Methanobrevibacter` is in the
  vocabulary but zero in every PRISM sample.
- **Samples:** 155 kept and 0 dropped. MicroCorpus takes 0.03 s.
- **Tokens per sample:**
  - including `<bos>`/`<eos>`: min 10, mean 32.45, max 48;
  - genus tokens only: min 8, mean 30.45, max 46.
- **In-vocabulary abundance per sample** (share of the sample's 200-species total, which itself
  ranges from 0.143 to 1.0 with median 0.993):
  - min 0.765 (`PRISM|8734`), median 0.983, mean 0.974;
  - 7 samples are below 0.90.

## 7. Open points

- Resolved: `scripts/download_data.sh` pins `MGM_SHA=bee1469fe13116d1d09e74a6169b29f6cd5a3c26`,
  the commit these facts and the smoke-test constants describe.
- The lock is complete for macOS arm64 only. The Linux path (`install -r`, which pulls torch's
  CUDA and triton wheels) has not been tested.
- The installed MGM records only a deleted temporary folder as its source (`direct_url.json`). The
  setup log prints the commit, but the environment itself does not.
