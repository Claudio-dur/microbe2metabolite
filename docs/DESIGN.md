# Design: modules, contracts and decisions

This is the engineering contract of the repository: what each module does, its public API, where
results go, and every non-obvious decision (with the alternatives that were rejected). The
scientific protocol itself is sections 4–6 of the project brief; this file explains how the code
implements it. Background notes: `docs/mgm_notes.md` (MGM internals, speed) and
`docs/published_numbers.md` (what the published PRISM → NLIBD numbers measure).

---

## 1. Data facts (re-checked on 2026-10-08 by `scripts/check_data.py`)

Expected values live in `configs/data_facts.yaml`; `make data` fails if any of them changes.
`results/data_checks.md` is the full report (49 facts, all passing; 6 differ from the brief's wording).

| Fact | Brief | Observed | Status |
|---|---|---|---|
| D1 microbiome PRISM / external | 200 × 155 / 200 × 65 | same | ✅ |
| D1 metabolome PRISM / external | 8,848 × 155 / 8,848 × 65, values 0–385,626 | same (external max 361,959) | ✅ |
| D1 annotation | 466 × 3 | **466 × 4** (extra `Adduct` column) | ⚠️ harmless |
| D1 `diagnosis_PRISM.csv` | CD / UC / control | **`IBD` (121) / `Healthy` (34) only** | ⚠️ CD/UC taken from D2 metadata |
| D1 external labels | (not listed) | `external_diagnosis.csv`: CD 20, UC 23, Control 22 | ✅ matches brief counts |
| PRISM diagnosis counts | 68 CD, 53 UC, 34 controls | same (D2 Franzosa `Study.Group`, consistent with D1 IBD/Healthy) | ✅ |
| NLIBD ID families | `Validation\|LLDeep_0001` | **two families**: 22 `LLDeep_` (all controls) + 43 `UMCGIBD` (all CD/UC) | ⚠️ diagnosis is confounded with recruitment cohort in the external test set |
| D1 ↔ D2 IDs | `PRISM\|7122` ↔ `PRISM.7122` | PRISM ✅; the 22 LLDeep IDs are identical in both releases; the 43 UMCG IDs differ (`UMCGIBD…` vs `UMCG…`) | ✅ matched by metabolome identity, 65/65 one-to-one (the LLDeep ones match themselves) |
| D1 vs D2 metabolome values | — | identical up to 1e-13 | ✅ same measurements |
| D2 sample / subject counts | Franzosa 220/220, iHMP 382/105, Yachida 347/347, Wang 287/287, Erawijantari 96/96, MARS 444/75 | same | ✅ |
| Prevalent genera (≥10% of samples > 1e-4) | Franzosa 350, iHMP 262, all-six 213, union 460, Franzosa∩iHMP 241 | same (Franzosa's 350 is over all 220 samples; PRISM alone gives 354) | ✅ |
| Source metagenomes | 1,556, all usable | 1,556, but **53 Erawijantari samples are the same metagenomes as Yachida samples** (flag `Shared.w.YACHIDA_2019`, identical profiles) | ⚠️ deduplicated → 1,503 |
| iHMP metabolome | zipped, 81,867 features | same; 507 HMDB-mapped features; **intensities 5–3e10 with 8.2% NaN** | ⚠️ NaN policy, per-cohort scaling |
| HMDB IDs shared Franzosa ∩ iHMP | 176 | 176 (all inside T466) | ✅ |
| MGM architecture | GPT-2, 8 layers, 256-d, 8 heads, 512 positions, vocab 9,669 | same (9,665 genera + 4 special tokens; 8.9 M parameters) | ✅ |
| D1 species → genera → in MGM vocab | 200 → 67 → 59 | same; the 8 misses are higher ranks (e.g. `Lachnospiraceae`, `Clostridiales`) | ✅ |
| MGM coverage of D2 genera | 42% of names, 81% of abundance | names: 0.426 of the 350 prevalent genera **after stripping GTDB suffixes** (`Faecalibacterium_A` → `Faecalibacterium`); abundance: 0.807 over all genus columns (0.792 over the prevalent ones) | ✅ (method clarified) |
| MGM input order | "sort from most to least abundant" | MGM sorts by **corpus z-score**, not abundance (first token ≠ most abundant genus in 105/155 PRISM samples) | ⚠️ we use MGM's own code |
| Published reference numbers | mNODE 0.294 / 0.642 / 93 | these are the **preprint**; the journal version reports 0.287 / 0.622 / 69 on the 466 annotated metabolites, with early stopping on NLIBD itself | ⚠️ see `docs/published_numbers.md` |

---

## 2. Package layout

```
src/m2m/
  paths.py          locations (env-overridable for tests and dry runs)
  config.py         YAML loading with `extends:` inheritance, config hashing
  data.py           loaders, the external-test guard, section-3 checks
  preprocess.py     fit(train) → transform(any) objects for microbes and metabolites
  splits.py         stratified / grouped folds, inner early-stopping splits, subsampling, disjointness
  evaluate.py       per-metabolite Spearman, summaries, permutation null, bootstrap CI
  models.py         MLP, masked-feature autoencoder, MGM regression head, safe MGM loading
  train.py          seeded training loop with early stopping (torch only; also runs in envs/mgm)
  tracking.py       runs.csv / summary.csv / MLflow / external-access audit log
  protocol.py       the shared CV and external-evaluation machinery every experiment uses
  run.py            CLI: `python -m m2m.run configs/<experiment>.yaml [--stage S[,S...]]`
  mgm_io.py         D1 → MGM genus tables, MGM vocabulary coverage
  transfer_data.py  D2 source/target assembly for E4 (vocabulary, HMDB targets, ID checks)
  plots.py          every figure in results/figures/
  experiments/
    common.py       D1 data assembly, shared ridge/MLP builders, learning-curve engine, dry runs
    baselines.py    E0 (mean, ridge, elastic net, random forest) + the D1 permutation null
    mlp.py          E1
    mgm_frozen.py   E2 + its random-weights control (runs scripts/mgm_embed.py in envs/mgm)
    mgm_finetune.py E3 + its from-scratch control (runs scripts/mgm_finetune.py in envs/mgm)
    transfer.py     E4a / E4b / E4c + the D2 permutation null
    learning_curve.py  E5: shared subsets, completeness check, the key figure
    interpret.py    E6
    ablations.py    A1–A4
scripts/
  download_data.sh, check_data.py, setup_mgm_env.sh, mgm_embed.py, mgm_finetune.py, report_tables.py
```

`m2m/__init__.py`, `models.py` and `train.py` import nothing beyond `torch`/`numpy`/stdlib at module
level, because the MGM environment (torch 2.0.1, numpy 1.24, pandas 2.0) imports them.

---

## 3. Module contracts

### data.py
- `Cohort(name, microbes, metabolites, meta)`: frozen dataclass; rows aligned across the three.
- `load_prism()`: D1 PRISM, transposed to samples × features; CD/UC/Control from D2 Franzosa
  metadata, asserted consistent with D1's IBD/Healthy file.
- `load_nlibd(*, final_evaluation=False, purpose="")` and `load_franzosa(part="Validation", ...)`:
  **raise `ExternalTestSetAccessError` unless `final_evaluation=True` and a non-empty `purpose`
  is given**; every access is appended to `results/external_access_log.csv`.
- `load_d2(dataset)` refuses the target dataset (use `load_franzosa`), and
  `load_d2_metadata(FRANZOSA)` returns PRISM rows only, so the external labels are reachable only
  through the guard. `load_sources(drop_duplicates=True)` returns the five deduplicated sources;
  iHMP metabolites are only loaded restricted to HMDB-mapped columns.
- `check_section3_facts()` collects every failed fact and raises `DataFactError` listing them; an
  empty facts file is an error (it would otherwise "pass" by checking nothing).

### preprocess.py (section 4 of the brief)
- `MicrobePreprocessor`: fit = re-close rows → prevalence filter on training rows (or a fixed
  `features` list such as the E4 vocabulary, in that order) → CLR/log with a per-sample
  pseudocount (½ × smallest non-zero kept value) → mean/std; transform reuses them.
- `MetabolitePreprocessor`: fit = zero filter on training rows → log10(x+1) or CLR (reference = all
  retained columns, shared by T466 and TALL; `pseudocount` is the per-sample rule or a fixed number
  for the published-protocol arm) → mean/std. A candidate target missing from Y is an error.
- Properties enforced by tests: fitted parameters never change after `fit`; a sample's output
  depends only on that sample and the fitted parameters (no batch statistics); outputs are aligned
  by label, not position; fitted objects record their training IDs (`fit_samples_`);
  `from_config` rejects unknown keys (a typo would otherwise fall back silently).

### splits.py / evaluate.py
- Folds, inner splits and subsets are ID-based; `prism_folds` treats `results/splits/prism_folds.csv`
  as the source of truth (the seed matters only when the file is first created). Grouped folds use
  `GroupKFold(shuffle=True)`, which balances subjects (not samples) per fold.
- `assert_disjoint` matches IDs both as raw values and as strings.
- `spearman_per_feature` (rank columns, then column-wise Pearson; constant column → NaN) is tested
  against `scipy.stats.spearmanr` and refuses NaN/inf input. `summarize_rho` counts NaN as ρ = 0
  and reports how many.
- `bootstrap_mean_rho`: resample test samples; metric = mean over seeds of the mean ρ. The CI is the
  **basic (reflected) bootstrap** interval; the percentile bounds are reported alongside.
- `permutation_null`: shuffle the training microbiome rows against the metabolome rows, refit,
  score the external cohort; pooled 95th percentile. Its per-permutation mean ρ and SD are saved
  next to the threshold (the null is wide because metabolites co-vary; see decision #30).

### models.py / train.py
- `MLP` and `MaskedFeatureAutoencoder` share the trunk structure, so pretrained trunks transfer by
  name; `load_compatible_state` refuses a layer that would load only in part.
- `MGMRegressor`: GPT-2 backbone + pooled linear head; `set_trainable_blocks(k)` unfreezes the last
  k blocks and the final LayerNorm (k = 0: linear probe); frozen parts stay in eval mode.
  `load_mgm_backbone` reads the checkpoint with `weights_only=True` (no pickled code runs).
- `fit_model(..., *, seed)`: AdamW, MSE by default, early stopping on validation loss, best weights
  restored; `min_batches_per_epoch` shrinks the batch only for tiny training sets.
- `TorchMLPRegressor` (and the E3 script) **centre targets on the fit rows** (`fit_row_offset`)
  and add the offset back when predicting (decision #21).

### protocol.py / experiments/common.py
- `M2MPipeline(x_prep, y_prep, regressor)`: fits preprocessing on the outer training rows, carves the
  inner early-stopping split if the regressor needs one, fits the regressor.
- `grid_search_cv` / `cross_validate` (PRISM folds only), `evaluate_external`, `score_predictions`,
  `cv_table`. Every fit is followed by a **tripwire** (`check_fitted_on`): both preprocessors must
  have been fit on exactly the training IDs, else `LeakageError`. The counts that survive the
  filters are recorded with every CV row, external row and curve point.
- `common.run_learning_curve`: trains every model on the same saved PRISM subsets
  (`results/splits/learning_curve_subsets.csv`, created in the selection phase), mapping IDs for D2.
- **Dry runs** (`M2M_DRY_RUN=1`, development only): the external cohort is replaced by a held-out
  slice of PRISM, so final-evaluation code can be exercised without opening NLIBD. They refuse to
  run unless results, processed data and the MLflow store are all redirected outside the repo's
  output folders.

### freeze.py
- `write_manifest()` (end of `make select`): hashes every file in `results/selected/` and
  `results/splits/` and every `configs/*.yaml` into `results/selected/FROZEN.json`; refuses if any
  experiment family has no selection output. Content only, so re-deriving identical selections
  rewrites an identical manifest.
- `require_frozen(purpose)`: called by the external-data guard (`data._guard_external`) for every
  opening except the section-3 data checks, by E3's NLIBD token reads and, in dry runs, by
  `common.load_d1_test`. Refuses unless the manifest exists, no listed file changed, none was
  added, and (outside dry runs) the manifest is committed and `git status` shows no change to the
  code, configs or selections.

### Experiments and their stages
| Experiment | Selection phase (PRISM only) | Evaluation phase (opens the external cohort) |
|---|---|---|
| E0 baselines | `cv` | `null`, `external`, `curve` |
| E1 MLP | `select` | `external`, `curve` |
| E2 frozen MGM + random-weights control | `prepare` (PRISM embeddings), `select` | `embed-external`, `external`, `curve` |
| E3 fine-tuned MGM + from-scratch control | `prepare-cv`, `run-cv`, `select` | `prepare-final` (NLIBD tokens), `run-final`, `evaluate` |
| E4 transfer | `sources`, `e4a`, `e4b`, `select` | `null`, `external`, `curve` |
| Ablations A1–A4 | `cv` | `external` |
| E5 learning curves | `subsets` | `check`, `figure` |
| E6 interpretation | `importance` | `classes` (reads per-metabolite results only) |
| Freeze | `python -m m2m.freeze write` (manifest of hashes) | `python -m m2m.freeze check`, then every opening re-checks it |
| Paired comparisons | — | `compare` (reads saved predictions only) |

E3 runs across environments: the main env writes task files (tokens, inner split, targets
standardised on the task's training rows) under `data/processed/mgm/e3/<round>/tasks/`, envs/mgm
trains and writes `preds/`, and the main env scores them with the same protocol functions. Each
round has a manifest with a settings hash; later stages refuse a round written under other settings
or in the other dry-run mode. External targets never enter a task file.

---

## 4. Results files

| File | Written by | Content |
|---|---|---|
| `results/data_checks.md` | check_data | every section-3 assert and its observed value |
| `results/splits/prism_folds.csv` | splits | the 5 PRISM folds used by every model |
| `results/splits/learning_curve_subsets.csv` | E5 `subsets` | the PRISM subsets for every (n, seed) |
| `results/selected/*.json`, `*__cv.csv` | selection stages | chosen hyperparameters + the CV grid they won; E0 and ablation units store their fixed parameters and CV score; `e4c_pretrained_fingerprints.json` fingerprints the E4 vocabulary, configs and pretrained networks |
| `results/runs.csv` | every run | one row per CV config / external seed / learning-curve point |
| `results/summary.csv` | external evaluations | one row per model × target set (mean ± SD over seeds, bootstrap CI) |
| `results/null_thresholds.csv` | E0 (D1), E4 (D2) | permutation-null 95th percentile per data × target set |
| `results/per_metabolite/*.csv.gz` | external evaluations | per-metabolite ρ on NLIBD (per seed and mean) |
| `results/learning_curve.csv`, `learning_curve_summary.csv` | curve stages, E5 `check` | mean ρ on NLIBD vs training size, per seed and summarised |
| `results/ablations.csv` | A1–A4 | CV and external metrics per (ablation, arm, model, target set, eval space) |
| `results/mgm_coverage.csv` | E2 `prepare` / `embed-external` | MGM vocabulary coverage per sample |
| `results/e4_sources.json`, `e4_pretraining.json` | E4 | source counts, vocabulary, iHMP NaN evidence, pretraining losses, iHMP sanity CV |
| `results/interpretation/*.csv` | E6 | top metabolites, permutation importance, ρ by chemical class |
| `results/external_access_log.csv` | data loaders, E2/E3 token reads | every time the external cohort was opened, and why |
| `results/selected/FROZEN.json` | `m2m.freeze write` | hashes of every selection artefact and config at the freeze |
| `results/paired_comparisons.csv` | compare | Δ mean ρ (A − B) with a paired-bootstrap CI and P(Δ ≤ 0) per pair |
| `data/processed/external/*.npz` (not committed) | external evaluations | truth and per-seed predictions on the external cohort |
| `results/figures/*.png`, `results/tables.md` | figures, report | figures and the README's tables |

---

## 5. Decisions and rejected alternatives

| # | Decision | Rejected alternative | Why |
|---|---|---|---|
| 1 | Stratify PRISM folds by CD/UC/Control from D2 metadata | IBD vs Healthy from D1 | The brief asks for 3-class stratification; D1 only has 2 classes; D2 labels are consistent with D1. |
| 2 | Preprocessing is fit on the whole outer training set (including the inner early-stopping split) | Fit on the inner-train part only | Keeps the evaluated metabolite set identical across models (the zero filter would otherwise differ between ridge and the MLP). The inner split is training data, so no test information leaks. |
| 3 | CLR pseudocount = ½ smallest non-zero value *of the kept taxa* in that sample | Global pseudocount fit on training | Per-sample rule needs no fitting (cannot leak) and is scale-invariant, so CLR ignores re-closure. |
| 4 | Early stopping on inner-validation MSE | Inner-validation mean Spearman | MSE is smooth and is the training objective; Spearman on ~30 samples is noisy. |
| 5 | Final models train on 80% of PRISM (20% kept for early stopping) | Refit on 100% for the best epoch count | Matches the brief's rule literally; refit would need an epoch-scaling heuristic. |
| 6 | CV selection metric = mean over folds of the fold's mean ρ | Pooled out-of-fold ρ | Per-fold models have different scales; pooling them biases rank correlations. |
| 7 | Hyperparameters selected separately for T466 and TALL | Select on T466, reuse for TALL | Cheap; E3 also takes each target set's pooling from E2's selection for that target set. |
| 8 | MGM input built with MGM's own `MicroCorpus` in `envs/mgm` | Re-implementing it | MGM orders tokens by corpus z-score, not raw abundance, and renormalises within its vocabulary; using its code avoids silent mismatches. |
| 9 | Deduplicate the 53 Erawijantari ∩ Yachida metagenomes | Keep all 1,556 | Duplicates would sit on both sides of the grouped pretraining validation split. |
| 10 | E4 vocabulary = union of per-source prevalent genera (sources only) | Prevalence over the pooled sources | Keeps genera that are characteristic of one cohort; fit never sees Franzosa. |
| 11 | All E4c variants refit the same target-side preprocessing on PRISM | Keep the source scaler for pretrained variants | Then the only difference between variants is the initialisation: a controlled comparison. |
| 12 | One representative LC-MS feature per HMDB ID (lowest zero fraction, then highest median, on training data) | Sum intensities across methods | Different LC-MS methods are not on a common scale. |
| 13 | Permutation null computed per data source (D1, D2) and target set | One global threshold | Ties from zero inflation change the null; it costs seconds. |
| 14 | Interpretation (E6) uses PRISM cross-validation only | Permutation importance on NLIBD | Keeps the external cohort out of everything except the final scores. |
| 15 | The MLP grid follows the brief (dropout {0.2, 0.5}); the "dropout 0.3" in the architecture line is the class default only | Add 0.3 to the grid | The brief's grid is explicit; flagged as an open question. |
| 16 | Basic (reflected) bootstrap CI | Percentile CI | Resamples contain duplicated samples, whose ties pull Spearman ρ toward 0; the percentile interval inherits that bias (≈0.3 SD low), the reflected one cancels it to first order. Percentile bounds are still reported. |
| 17 | Dry-run mode for development | Developing final-evaluation code against NLIBD | Every final-evaluation path is exercised on a PRISM hold-out first, so the real external evaluation runs once. |
| 18 | Ablation A4 reproduces the published evaluation protocol (CLR(x+1) targets over all 8,848 features, no zero filter, scored on the 466 annotated) | Comparing published numbers with ours directly | Gives one like-for-like comparison with mNODE/MiMeNet (still context: mNODE early-stopped on NLIBD and reported one run). |
| 19 | Downloaded pickles are never fully unpickled (tokenizer: allowlisted globals; weights: `weights_only=True`) | `pickle.load` / transformers' default loader | Unpickling runs code; the files come from a third-party repository. |
| 20 | Learning-curve points refit the zero filter on each subset and record how many targets remain | Fix the target set from all 155 PRISM samples | "n labelled samples" must mean the model saw only those n samples' metabolomes (at n = 20 the kept sets overlap the full-data set with Jaccard 0.95–0.98). |
| 21 | Early-stopping regressors centre targets on their fit rows and add the offset back | Leave the targets as standardised on fit + validation rows | With a 20% split the validation rows' mean is −4 × the fit rows' mean; at n = 20 that offset dominates the validation loss and early stopping kept the untrained epoch-1 network (MLP ρ ≈ 0.01). Centring uses fit rows only and adds one constant per target, which leaves every Spearman ρ unchanged; PRISM check at n = 20: ρ 0.26, best epoch ≈ 159. |
| 22 | At least 4 optimiser steps per epoch (`min_batches_per_epoch`) | Always the configured batch size | Learning-curve subsets otherwise get one step per epoch; same accuracy, half the epochs. Full-data fits are unaffected. |
| 23 | Ablation default arms copy E0 ridge / E1 MLP's external rows | Re-scoring them on NLIBD | They are the same configurations; the brief allows one external evaluation per final configuration. Their CV is still recomputed and must match E0 / E1 exactly. |
| 24 | Two-phase run: `make select` (no external access) → commit the selections → `make evaluate` | Per-experiment order (select then evaluate, one experiment at a time) | Every choice is frozen, and committed, before any model is scored on the external cohort; the commit is the evidence. |
| 25 | E2 pooling (mean vs last token) is a CV hyperparameter of each head | Fixing mean pooling a priori | Both are defensible for a causal transformer; CV decides. |
| 26 | MGM controls: a random-weights frozen GPT-2 (E2) and a GPT-2 trained from random weights (E3), same tokens, folds and protocol | Comparing MGM only with ridge / MLP | H1 is about pretraining versus the same architecture from scratch; without these, MGM's result mixes pretraining with architecture and tokenisation. The from-scratch network gets its own lr grid {1e-4, 3e-4, 1e-3}: 1e-5–3e-5 cannot train it. |
| 27 | iHMP missing metabolite values = not detected (0) | Mask them out of the loss | No sample lacks a whole LC-MS method, the table has no zeros at all, and NaN concentrates in low-intensity compounds and drugs: missing means below detection. |
| 28 | E4 transfer guards: a selection fingerprint (vocabulary digest, model-defining config sections, every pretrained network) is written last by `select` and checked before any external access; trunk (and carried-over head) keys must load | Trusting files on disk | A regenerated vocabulary or network would otherwise be scored as if it were the selected one, and a silently skipped layer would turn "pretrained" into "scratch". |
| 29 | E6 explains one feature per compound and the best compound of three named classes (bile acids, indoles, straight-chain fatty acids), chosen by ridge's PRISM CV; importance permutes the z-scored CLR input column | Raw-abundance permutation with re-applied preprocessing | Standard input permutation; caveat: the permuted row leaves the CLR constraint, so importances are model-specific explanations, not causal claims. |
| 30 | Keep the brief's pooled permutation null | A null on independent ranks | The pooled null is wide because a model trained on shuffled pairs still produces predictions correlated with many co-varying metabolites at once; N(ρ > null) is therefore conservative, which is stated with it. |
| 31 | Elastic net per metabolite with MelonnPan's l1-ratio floor (0.05) | Extending the grid toward ridge | Comparable to MelonnPan; most metabolites pick the floor, and ~16 per fold get a constant prediction (counted as ρ = 0). |
| 32 | The freeze is enforced in code: every external opening (except the data checks) needs a committed, unchanged manifest of the selection phase | A documented convention plus a manual commit | Removes the honour system: no external score can come from a choice, config or code made after the freeze. It also catches config drift between `select` and `evaluate` for every experiment at once. |
| 33 | `git_sha`'s "-dirty" covers code, configs and selections only | The whole tree | The run writes its own outputs (summary.csv, the access log); counting them would mark every evaluation row dirty and hide real code changes. |
| 34 | External predictions are kept (not committed) and pairs of models are compared with a paired bootstrap over test samples | Comparing two models' separate CIs | H1 asks whether A beats B on the same people; pairing removes the shared sample-to-sample variation, and reading saved predictions needs no second opening of the cohort. |
| 35 | E4b early-stops on one subject-grouped hold-out (20% of iHMP subjects); a subject-grouped 5-fold CV is a diagnostic only | Early stopping by GroupKFold (the brief's wording) | One pretrained network per seed is needed for transfer; k networks per seed would need an ensemble or a refit rule. Subjects never straddle the split either way. |
| 36 | One CV model seed (0) for every experiment's selection | A seed per experiment | E0/E1/E2/E3 and the ablations then share identical inner early-stopping splits, which the ablations' exact-reproduction check relies on. |
| 37 | E3's predictions are bitwise reproducible at the configured thread count (`mgm.threads: 2`); threads are not in the settings hash | Hashing the thread count | A different thread count changes float rounding only; it is documented rather than forcing re-training on another machine. |
| 38 | Evaluation stages do not skip already-scored configurations | Resume logic that skips them | Re-scoring is deterministic and every opening is logged; after a failure, rerun the failed stage rather than all of `make evaluate`. |
