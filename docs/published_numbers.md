# Published PRISM → NLIBD reference numbers: what they measure

This note answers the brief's question (section 3, "Published reference numbers", and section 12):
*which metabolites, which preprocessing and which protocol produced the mNODE and MiMeNet numbers,
and can our results be compared with them?* Every claim below has a source and, where possible, a
check against the authors' own code and result files (section 5). Checked on 2026-10-08, then
re-verified the same day by an independent fact-check pass: every source was re-fetched and every
number recomputed. The corrections that pass made are listed at the end of the appendix.

## 0. Bottom line

- **The brief quotes the preprint.** The peer-reviewed version (Nature Machine Intelligence, 2023)
  reports different mNODE numbers: mean ρ **0.287**, top-50 mean ρ **0.622**, N(ρ > 0.5) **69**,
  instead of 0.294 / 0.642 / 93. MiMeNet's numbers did not change (0.191 / 0.512 / 28). Cite the
  journal version.
- **Metabolite set: the 466 annotated metabolites.** The models were *trained* on all 8,848 LC-MS
  features, but the metrics are computed only on the 466 annotated ones. The journal text says so,
  and the authors' per-metabolite result file reproduces 0.287 / 0.622 / 69 exactly on that subset.
- **mNODE's preprocessing: CLR(x + 1) per sample for both microbes and metabolites.** For metabolites
  the CLR runs across all 8,848 features. There is no zero/prevalence filter and no z-scoring. The
  input is all 200 MetaPhlAn2 species. The baselines in mNODE's Fig. 3 add a z-score on top of the CLR
  (SI §1). A per-metabolite z-score does not change Spearman ρ.
- **Protocol:** train on PRISM, test on NLIBD, hyperparameters chosen by 5-fold CV on PRISM. However,
  the final mNODE model is **early-stopped on the NLIBD test cohort itself**. Each published mNODE
  value comes from **one training run**. Across the authors' own 50 re-runs (Extended Data Fig. 8,
  read from the figure), the median is about **0.263 / 0.611 / 57.5**. The published mean ρ and
  N(ρ > 0.5) coincide with the single highest re-run (the upper outlier). The published top-50 value
  (0.622) lies inside the upper whisker (maximum ≈0.629). Whether the published run is one of the 50
  cannot be checked.
- **Verdict: no published number is like-for-like with our protocol.** Report all of them as
  context. Make the like-for-like comparisons against our own baselines (ridge, elastic net) under
  the identical protocol. Paste-ready README text is in section 7.

## 1. The numbers and where they appear

| Source | Where | Method | Mean ρ | Top-50 mean ρ | N(ρ > 0.5) | Metabolites scored | Status |
|---|---|---|---|---|---|---|---|
| mNODE preprint, bioRxiv 2022.06.23.497381 v1 (24 Jun 2022) | Results, "Superior performance of mNODE on real data"; Fig. 3a1–a3 | mNODE | 0.294 | 0.642 | 93 | Not stated in the text; annotated by magnitude (see section 2) | **Superseded**; not reproducible from the authors' files |
| same | same | MiMeNet (run by mNODE authors) | 0.191 | 0.512 | 28 | same | Same values as the journal |
| mNODE journal, *Nat Mach Intell* 5:284–293 (13 Mar 2023); author manuscript PMC10786629 | Same section; Fig. 3a1–a3; Extended Data Fig. 7a4/a7 | mNODE | **0.287** | **0.622** | **69** | **466 annotated** (stated) | **Current**; reproduced exactly (section 5) |
| same | same | MiMeNet (run by mNODE authors) | 0.191 | 0.512 | 28 | 466 annotated | Current; not reproducible (no baseline code published) |
| same | Fig. 3a (bar heights, read from figure ±0.01 / ±1) | MelonnPan / sparse NED / ResNet (run by mNODE authors) | ≈0.18 / ≈0.15 / ≈0.20 | ≈0.50 / ≈0.45 / ≈0.57 | ≈17 / ≈12 / ≈48 | 466 annotated | Read from the figure only |
| same | Extended Data Fig. 8 (boxplots, n = 50 re-runs; read from figure) | mNODE, 50 re-runs | median ≈0.263 (IQR 0.258–0.268) | median ≈0.611 (IQR 0.604–0.618) | median ≈57.5 (IQR 56–61) | 466 annotated (inferred: the caption does not say so, but its N(ρ > 0.3) ≈ 208 fits only the annotated set) | The single upper outliers, ≈0.289 and ≈69, match the headline values. The headline top-50 (0.622) is inside the whisker (maximum ≈0.629) |
| MiMeNet paper, Reiman et al. *PLoS Comput Biol* 17:e1009021 (2021) | Results, "MiMeNet outperforms models in MelonnPan"; Fig. 4D | MiMeNet / MelonnPan (own runs) | **0.275** / 0.168 | not reported | not reported | 466 annotated | Mean of 10 trained models |
| same | same; S4 Fig D | MiMeNet / MelonnPan | — | — | 308 / 186 "well-predicted" (ρ above a shuffled-data null cutoff of 0.136); 160 shared | Annotated (inferred: MelonnPan was run on the annotated set only, and the code's external count is over annotated metabolites, `MiMeNet_train.py` L1094–1097) | Different threshold |
| same | S3 Table ("Evaluation of the IBD (External) data") | MiMeNet on CLR-transformed data / MiMeNet on RA-transformed data / MelonnPan (RA block only) | 0.24 / 0.21 / 0.17 | — | — | Not stated | Context only. The two blocks differ in the *data* transform. IBD microbes stay RA in the CLR block (S1 Table caption), so 0.24 vs 0.21 mainly reflects the target transform, not the input |
| MelonnPan paper, Mallick et al. *Nat Commun* 10:3136 (2019) | Results, "MelonnPan accurately predicts metabolites from metagenomes"; Supplementary Fig. 1 | MelonnPan (UniRef90 gene-family input) | not reported | not reported | **107 of 199** annotated with ρ ≥ 0.3 (not 0.5) | The 199 annotated standards that passed its filter *and* were well predicted in PRISM training CV (Supplementary Fig. 1B: 248 trained → 199 → 107 validated) | Different metric and inputs. The text says "≥0.3"; the figure legend says "r > 0.3" |

**What changed between versions.** Only mNODE's own numbers changed: 0.294 → 0.287, 0.642 → 0.622,
93 → 69. The baseline bars in Fig. 3a are identical in both versions within measurement precision
(pixel-measured). The text of the MiMeNet comparison is also unchanged. The journal version also
changed three other things:

- it states the annotated-only rule explicitly;
- it widens the λ grid from {1e-4, 1e-3, 1e-2} to {1e-6, …, 1e-2};
- it changes early stopping from "test loss increasing" to "test mean ρ decreasing" (section 4).

Code release v1.0.0 (Zenodo 10.5281/zenodo.7602940) is commit `8677bf9` (27 Oct 2022), titled
"Updated the mNODE algorithm with expanded hyperparameter ranges". This is the code behind the
journal numbers.

## 2. Which metabolites and which microbes

**Metabolites.**

- Journal: *"Though all metabolites are used in the training of ML methods, we only included annotated
  metabolites in the metric calculation to be in line with previous studies to ensure a fair
  comparison."* It also states: *"Out of 8848 unique metabolites, 466 are annotated."*
- Preprint: the text is ambiguous. It defines the metric as *"the mean SCC, [ρ̄], of all metabolites
  measured in experiments"* (both versions keep this sentence in the overview). The scatterplot is
  over annotated metabolites: *"we directly compared ρ of annotated metabolites via a scatterplot in
  Supplementary Fig. S1."*
- Evidence that the preprint also used the annotated subset:
  - The preprint-era script (`mNODE.jl` at commit `07afef6`, 23 Jun 2022) prints
    "The mean Spearman C.C. for all annotated metabolites" and
    "The number of annotated metabolites with Spearman C.C. > 0.5".
  - A count of 93 metabolites with ρ > 0.5 is only plausible for the annotated set. Over all 8,848
    features, the authors' own files give 1,237 (journal run) to 1,639 (June 2022 run).
- **How the code selects the subset.** `compound_names.csv` holds the annotation name for each of the
  8,848 rows and `nan` for unannotated rows (8,382 `nan`, 466 names). Julia's `readdlm` reads `nan`
  as `NaN`: untyped cells go through `jl_try_substrtod`, which accepts `nan` (section 9). Because
  `NaN == NaN` is false, the filter `corr_test[findall(compound_names.==compound_names)]` keeps
  exactly the 466 annotated rows (`mNODE.jl` L92, L162).

**Microbes.**

- All 200 MetaPhlAn2 species in MiMeNet's `microbiome_PRISM.csv` / `microbiome_external.csv`, with no
  prevalence filter. The paper says *"In total, 200 microbial taxa"*.
- mNODE's copies of the PRISM files are byte-identical to our D1 copy (md5 checked for
  `metabolome_PRISM.csv`, `microbiome_PRISM.csv`, `metabolome_annotation.csv` and
  `diagnosis_PRISM.csv`).
- For comparison:
  - our microbe filter (> 1e-4 in ≥ 10% of PRISM) keeps 151 of the 200 species;
  - MiMeNet's own rule (present in > 10% of samples) keeps 163.

## 3. Preprocessing

| Method (as run where) | Microbes | Metabolites (targets) | Feature filter | Scaling | Source |
|---|---|---|---|---|---|
| **mNODE** (both versions) | CLR(x + 1) per sample over the 200 species. Because D1 values are fractions (column sums ≤ 1), this is almost linear in relative abundance: r = 0.997 with centred raw abundance on PRISM | CLR(x + 1) per sample across **all 8,848** features | None | None | Paper Methods: *"The CLR (Centered Log-Ratio) transformation is applied to microbial abundances and metabolite concentrations."*; `data_processing.py` L46–57; verified on `processed_data/y_train.csv` (section 5) |
| MiMeNet, sparse NED, ResNet (as run by the mNODE authors) | CLR then z-score | CLR then z-score | Not stated | z-score | mNODE SI §1: *"The Z-score transformation is applied to the CLR (Centered Log-Ratio) transformed microbial abundances and metabolite concentrations."* |
| MelonnPan (as run by the mNODE authors) | Quantile-normal ("rank") transform of normalised abundances | Arcsine square root of normalised profiles | Not stated | — | mNODE SI §1: *"It applies the rank-transformation to the normalized microbial abundances … Normalized metabolite profiles are arcsine square root transformed"* |
| **MiMeNet, own paper / code** | Raw relative abundance (`run.sh`: `-micro_norm None`) | CLR(x + 1) per sample, across the 8,847 features that pass the filter | Drop features present in ≤ 10% of PRISM samples. On PRISM this drops 1 (unannotated) of the 8,848 metabolite features and 37 of the 200 species | z-score (`StandardScaler` fit on training data) | Paper Methods: *"Any input or output feature that is present in less than 10% of samples was removed … A pseudocount of 1 was added … The only exception was for the IBD (PRISM) and IBD (External) microbe values, which were obtained in relative abundance (RA)."*; `MiMeNet_train.py` L192–208 (filter), L228–256 (transforms), L348–354 and L987–992 (z-score) |
| **MelonnPan, own paper** | ~1M UniRef90 gene families (HUMAnN2) → 814 after filtering, quantile-normal transform | Relative abundance, arcsine square root; 2,818 features after filtering | > 0.01% in ≥ 10% of samples | — | *"both gene families and metabolites, of very low relative abundance and prevalence (<0.01% in ≥10% of samples) were removed, leaving 2818 metabolites and 814 gene families for final modelling"* |
| **Ours** | Re-close; 151 of 200 species pass > 1e-4 in ≥ 10% of PRISM; CLR with a per-sample half-minimum pseudocount | log10(x + 1); targets zero in > 50% of PRISM dropped: **T466 → 427** annotated, **TALL → 6,948** features | Fit on PRISM | z-score (PRISM statistics) | `configs/base.yaml`. Counts computed from full PRISM, and reproduced by fitting the project's own `m2m.preprocess` classes. The pipeline's results record should still state them |

**Why the target scale matters for Spearman.**

- On our side, log10(x + 1) and z-scoring are strictly increasing *per metabolite*. Our ρ is therefore
  exactly the Spearman correlation with the raw measured intensity, with tied zeros kept as ties.
- On mNODE's side, CLR subtracts a *per-sample* mean log, which re-orders samples within a metabolite.
- On PRISM, the across-sample Spearman between raw intensity and CLR(x + 1) value has median 0.959
  over the 466 annotated metabolites (10th percentile 0.898, minimum 0.739). The two ground truths
  agree closely but are not the same. For a metabolite that is zero in a sample, the CLR value is
  −mean log(x + 1) of that sample, so it partly encodes overall metabolome depth. For the 39
  zero-heavy annotated metabolites we drop, the CLR profiles have a median Spearman of 0.40 with that
  sample-level quantity on PRISM (0.20 for the 427 we keep).

## 4. Protocol

| Aspect | mNODE (journal / code v1.0.0) | MiMeNet (own paper / code) | Ours |
|---|---|---|---|
| Train / test | Train on all of PRISM (155), test on NLIBD (65): *"all data … related to individuals in PRISM are used as the training set and the external validation cohort NLIBD/LLDeep is used as the test set."* | Same | Same |
| Hyperparameters | 5-fold CV on PRISM, criterion = mean ρ over annotated metabolites. Grid: Nh ∈ {32, 64, 128}; λ ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2} (preprint: {1e-4, 1e-3, 1e-2}). The folds are contiguous blocks of PRISM samples in sorted-ID order: MLDataUtils `kfolds` does not shuffle, and `np.intersect1d` sorts the IDs | Random search (20 sets) with nested 5-fold CV on the first CV partition; IBD choice: 1 layer × 512, λ = 0.001, dropout 0.5 | Stratified 5-fold CV on PRISM (saved folds) |
| Early stopping | **On the test set.** The final fit passes NLIBD as `X_test, y_test` (`mNODE.jl` L231). Training stops when NLIBD mean annotated ρ fell in more than 12 of the last 20 epochs (L144–149), with a maximum of 100 epochs and no best-weight restore. Paper: *"The training stops if the mean SCC … of annotated metabolites ρ̄ on the validation/test set starts to decrease within the past 20 epochs."* Preprint-era code (`07afef6` L255) stopped on the NLIBD *loss* instead. During CV, it stops on the held-out fold | Keras `EarlyStopping` on a 20% validation split of the *training* data, patience 40, best weights restored (`src/models/MiMeNet.py` L49–52). It never sees the test set | Inner 20% split of PRISM |
| Repeats / seeds | One run per reported number. No seed is set in `mNODE.jl`. 50 re-runs are shown separately (ED Fig. 8) | Mean over 10 trained models (Fig. 4D). The code's single "final model" is chosen by a training-only cluster-consensus score. As coded, every candidate gets the same score, because L1012–1014 reuse a stale loop variable, so run 0 is always selected | Mean ± SD over 5 seeds |
| Metric | Spearman per metabolite across the 65 NLIBD samples; mean, mean of the 50 largest, count > 0.5 (strict). No NaNs occur because CLR values are never constant. The script itself prints the 51st-largest annotated ρ (labelled "Top 100"), not the top-50 mean. The published 0.622 equals the mean of the 50 largest values in the result file | Spearman per metabolite; mean; "well-predicted" = above the 95th percentile of a shuffled-data CV null (0.136 for IBD) | Spearman per metabolite on NLIBD, NaN → 0; same three summaries plus median and N(above null) |

The baselines in mNODE's Fig. 3 were tuned by 5-fold CV on PRISM (SI §1). The SI describes MiMeNet's
early stopping as *"the loss on the validation/test set has not improved within the past 40 epochs"*.
Sparse NED and ResNet train for a fixed 100 epochs, and MelonnPan (elastic net) has no epoch-based
stopping. So among the baselines, only MiMeNet's stopping rule could involve NLIBD. The mNODE
repository contains no baseline code, so which set was used, and how many runs, cannot be checked.

## 5. What the mNODE repository confirms

Repository: https://github.com/wt1005203/mNODE (full history cloned; nothing executed).

| Check | Result |
|---|---|
| Journal numbers from `predictions/PRISM+NLIBD/metabolites_corr.csv` (8,848 rows, D1 row order) restricted to the 466 annotated rows | mean **0.2871**, top-50 mean **0.6221**, N(ρ > 0.5) **69**. Exact match with the journal text. Over all 8,848 rows: 0.261 / 0.713 / 1,237 |
| Is that file Spearman? For untied Spearman with n = 65, (1 − ρ)·n(n² − 1)/6 must be an integer | 100% of the 8,848 values satisfy this for n = 65. These are Spearman coefficients over all 65 NLIBD samples |
| Preprint-era file `best_fitted_metabolites_corr.csv` (`43bd258`, 22 Jun 2022) | Annotated: 0.277 / 0.680 / 102. This does **not** match the preprint (0.294 / 0.642 / 93); no committed file reproduces the preprint numbers. Essentially none of its values satisfy the untied-Spearman condition for any n from 10 to 300: none for n = 65, and at most 1 of 8,848 for any n (chance level). This is consistent with the June 2022 scripts computing `cor(...)`, i.e. **Pearson** (`07afef6` `mNODE.jl` L133, L268; the same in `43bd258` and `d352694`). Whether the preprint figure itself was Pearson or Spearman cannot be determined |
| Preprocessing on the saved training data | `processed_data/y_train.csv` (155 × 8,848) equals CLR(x + 1) of `metabolome_PRISM.csv` (max abs diff 5e-15). `X_train.csv` (155 × 200) equals CLR(x + 1) of `microbiome_PRISM.csv` (max abs diff 3e-15) |
| Journal numbers re-scored on **our** subsets (same published per-metabolite ρ) | T466 after our zero filter (427): **0.290 / 0.619 / 66**. TALL-like set (6,948): 0.258 / 0.705 / 921. The 39 dropped annotated metabolites: mean 0.254. The subset alone moves the mean by +0.003 |

## 6. Verdict per number

Our protocol:

- PRISM-fit log10(x + 1) + z-score targets;
- targets zero in > 50% of PRISM dropped (T466 = 427 annotated, TALL = 6,948);
- Spearman per metabolite on NLIBD, NaN → 0;
- mean over 5 seeds;
- NLIBD never used before the final evaluation.

| Reference number | Like-for-like with ours? | What differs |
|---|---|---|
| mNODE 0.287 / 0.622 / 69 (journal) | **No: context** | (1) CLR(x + 1) target scale over 8,848 features vs log10 (rank-equivalent to raw intensity); (2) 466 vs 427 metabolites, and mNODE trains on all 8,848 outputs; (3) **early stopping on NLIBD**: the test cohort decides when training stops, so this is not a clean held-out estimate (probably optimistic; not quantified); (4) a single run whose mean ρ and N(ρ > 0.5) match the best of the authors' 50 re-runs (median ≈0.263 / 0.611 / 57.5), so it overstates a typical run. Shared: the same cohorts, the same 200-species input table (mNODE uses all 200 species; our filter keeps 151) and the same per-metabolite Spearman definition |
| mNODE 0.294 / 0.642 / 93 (preprint) | **No: do not cite** | Everything above, plus: superseded by the journal, not reproducible from the authors' files, and the preprint-era script computed Pearson |
| mNODE 50-run median ≈0.263 / ≈0.611 / ≈57.5 (ED Fig. 8) | **No: closest in spirit** (multi-run, like our seed mean) | Same scale, subset and test-set early-stopping issues; values read off a boxplot |
| MiMeNet 0.191 / 0.512 / 28 (mNODE's Fig. 3) | **No: context** | CLR + z-score targets, 466 annotated, run by a third party with unpublished code and protocol details. It is far below MiMeNet's own 0.275 for the same split |
| MelonnPan ≈0.18 / 0.50 / 17, sparse NED ≈0.15 / 0.45 / 12, ResNet ≈0.20 / 0.57 / 48 (mNODE's Fig. 3) | **No: context** | Read from the figure; third-party runs; MelonnPan here uses species, not its original gene families |
| MiMeNet own 0.275 (MelonnPan 0.168) | **No: context** | CLR(x + 1) targets, relative-abundance input, 10% presence filter, 466 annotated. Clean protocol, averaged over 10 models |
| MiMeNet own 308 vs 186 "well-predicted" | **No** | Threshold is a null cutoff of 0.136, not 0.5 |
| MelonnPan own 107 / 199 with ρ ≥ 0.3 | **No** | Gene-family inputs, different filter, a CV-preselected denominator, threshold 0.3 |

**What would be closer, if wanted:** score an extra evaluation arm on mNODE's scale. That means
per-sample CLR(x + 1) of the NLIBD truth over all 8,848 features (a per-sample transform, so nothing
is fitted on NLIBD), no zero filter, metrics over all 466 annotated, and a model that outputs all
8,848 features. That would match target scale and subset. It still would not match mNODE's test-set
early stopping or its single-run reporting, so it would remain context.

## 7. Wording for the README

**Short form** (caption or footnote under the results table):

> Published numbers on the same PRISM → NLIBD split (mNODE, Wang et al. 2023: mean ρ 0.287, top-50
> ρ 0.622, 69 metabolites with ρ > 0.5; MiMeNet in the same figure: 0.191 / 0.512 / 28) are context,
> not a benchmark. They score CLR-transformed abundances over all 466 annotated metabolites, and
> mNODE's final model was early-stopped on the NLIBD test cohort.

**Long form** (Methods or Limitations section):

> **Published reference numbers (context only).** On the same PRISM → NLIBD split, using the same
> 200-species MetaPhlAn2 table, mNODE (Wang et al., *Nat. Mach. Intell.* 2023, Fig. 3a) reports a
> mean Spearman ρ of 0.287, a top-50 mean ρ of 0.622 and 69 metabolites with ρ > 0.5, over the 466
> annotated metabolites. The MiMeNet baseline in the same figure scores 0.191 / 0.512 / 28. The
> bioRxiv preprint reported 0.294 / 0.642 / 93 for mNODE; the journal version replaces it. These
> values are not a like-for-like benchmark for ours, for four reasons:
>
> 1. **Target scale.** mNODE scores CLR(x + 1) values computed across all 8,848 LC-MS features of
>    each sample. We score log10(x + 1) intensities, so our ρ ranks samples by measured intensity.
> 2. **Metabolite set.** mNODE scores all 466 annotated metabolites and trains on all 8,848 features.
>    Our T466 keeps the 427 annotated metabolites that are non-zero in at least half of PRISM.
> 3. **Early stopping.** mNODE's final model stops training based on its mean ρ on the NLIBD test
>    cohort itself, so the published values are not clean held-out estimates. We never use NLIBD
>    before the final evaluation.
> 4. **Repeats.** Each published value is a single training run. Across the authors' 50 re-runs
>    (Extended Data Fig. 8), the median is about 0.26 / 0.61 / 57.5. The published mean ρ and
>    N(ρ > 0.5) match the best of those re-runs. We report the mean over 5 seeds.
>
> Re-scoring mNODE's published per-metabolite ρ on our 427-metabolite set gives 0.290 / 0.619 / 66,
> so the subset alone changes little. The published run sits at the top of the authors' own 50-run
> distribution (median mean ρ about 0.26 vs 0.287). Test-set early stopping probably adds optimism of
> unknown size. The effect of the different target scale has unknown sign. MiMeNet's own
> paper reports a mean ρ of 0.275 on the annotated metabolites for this split (mean of 10 models,
> CLR targets). We therefore make like-for-like comparisons only against our own baselines under the
> identical protocol, and we do not claim to beat or match mNODE.

## 8. What could not be verified

- **The preprint values** (0.294 / 0.642 / 93) cannot be reproduced from any file the authors
  committed. Whether that figure used Pearson (as the June 2022 script does) or Spearman is unknown.
  Checking it would need NLIBD measurements (mNODE's `y_test.csv`), which this package does not open.
- **How the four baselines were run** (own code or re-implementation, which early-stopping set, how
  many runs). No baseline code is public. None of the first author's other public GitHub
  repositories (McMLP, METRIC and others; checked 2026-10-08) contains PRISM + NLIBD baseline code.
- **Whether the headline mNODE run is one of the 50 re-runs.** The match between the headline values
  and the upper outliers comes from reading the figure, not from data.
- **The size, and strictly even the sign, of the test-set early-stopping effect.** The rule stops
  after a run of declines and does not restore the best epoch, so it is not a simple max-over-epochs
  selection. Measuring the effect would mean re-running mNODE (Julia 1.6, Flux with Tracker).
- **The version of record** on nature.com is paywalled. The text was checked against the PMC author
  manuscript. Figures and SI were checked against the publisher's own files.
- **Figure-derived values** (baseline bars, ED Fig. 8 medians) come from pixel measurement
  calibrated on the axis ticks. On the journal figure, this reproduces the stated mNODE and MiMeNet
  values within 0.002 and 0.5 counts. The preprint figure is coarser (one pixel ≈ 0.005–0.008, or
  ≈ 0.9 counts), so version-to-version comparisons are good to about one pixel.

## 9. Sources

- mNODE preprint v1: https://www.biorxiv.org/content/10.1101/2022.06.23.497381v1. Full-text XML:
  https://www.biorxiv.org/content/early/2022/06/24/2022.06.23.497381.source.xml (bracketed symbols
  such as [ρ̄] are inline images there). SI:
  https://www.biorxiv.org/content/biorxiv/early/2022/06/24/2022.06.23.497381/DC1/embed/media-1.pdf
- mNODE journal: https://doi.org/10.1038/s42256-023-00627-3. Author manuscript:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10786629/. SI:
  https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-023-00627-3/MediaObjects/42256_2023_627_MOESM1_ESM.pdf
- mNODE figures:
  - Fig. 3: https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs42256-023-00627-3/MediaObjects/42256_2023_627_Fig3_HTML.png
  - ED Fig. 7: https://media.springernature.com/full/springer-static/esm/art%3A10.1038%2Fs42256-023-00627-3/MediaObjects/42256_2023_627_Fig12_ESM.jpg
  - ED Fig. 8: https://media.springernature.com/full/springer-static/esm/art%3A10.1038%2Fs42256-023-00627-3/MediaObjects/42256_2023_627_Fig13_ESM.jpg
- mNODE code:
  - v1.0.0 = `8677bf988d91ef6ec80b74f1f26e16c622bd6b5c` (https://doi.org/10.5281/zenodo.7602940):
    https://github.com/wt1005203/mNODE/blob/8677bf988d91ef6ec80b74f1f26e16c622bd6b5c/mNODE.jl and
    https://github.com/wt1005203/mNODE/blob/8677bf988d91ef6ec80b74f1f26e16c622bd6b5c/data_processing.py
  - Preprint era: https://github.com/wt1005203/mNODE/blob/07afef65d1687b84f45f9d3bf94e012692be7629/mNODE.jl
    and https://github.com/wt1005203/mNODE/blob/43bd25846832fa2ab245cf1603805d99551e7011/best_fitted_metabolites_corr.csv
- MiMeNet:
  - Paper: https://doi.org/10.1371/journal.pcbi.1009021. S3 Table:
    https://journals.plos.org/ploscompbiol/article/file?type=supplementary&id=10.1371/journal.pcbi.1009021.s012
  - Code `f9064b54d61c40d12207db896bc137341969aef4` (the D1 clone in `data/raw/MiMeNet`; `run.sh`,
    `MiMeNet_train.py`, `src/models/MiMeNet.py`).
- MelonnPan: https://doi.org/10.1038/s41467-019-10927-1 (PMC6637180). SI:
  https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-10927-1/MediaObjects/41467_2019_10927_MOESM1_ESM.pdf
- MLDataUtils/MLDataPattern `kfolds`: https://mldatapatternjl.readthedocs.io/en/latest/documentation/folds.html.
  For `kfolds(data, k)`, the form `mNODE.jl` calls: *"there is no random assignment of observations to
  parts"*. The index form `kfolds(n, k)` says *"there is no random placement of observations into
  subsets"*.
- Julia parses `nan` as `NaN` in `readdlm`. In Julia 1.6's stdlib DelimitedFiles, untyped cells go
  through `jl_try_substrtod`
  (https://github.com/JuliaLang/julia/blob/release-1.6/stdlib/DelimitedFiles/src/DelimitedFiles.jl,
  `colval`), and `parse_inf_or_nan` matches `nan` case-insensitively
  (https://github.com/JuliaLang/julia/blob/release-1.6/src/support/strtod.c).

## Appendix: how the checks were done

- **No NLIBD measurement file was opened.** That covers `*_external.csv`, mNODE's `X_test.csv` /
  `y_test.csv` and `true_metabolomic_profiles.csv`.
- **The only NLIBD-derived inputs were the authors' published per-metabolite correlation files.**
  These are results, not cohort data, and only aggregates of them are reported here.
- **Everything else used PRISM files, which are byte-identical to our D1 copy.**
- **No third-party code was executed.** All scripts were throwaway and kept outside the repository.
- **Spearman test.** For untied data with n samples, Spearman ρ = 1 − 6Σd²/(n(n² − 1)) with integer
  Σd², so (1 − ρ)·n(n² − 1)/6 must be an integer (45,760 for n = 65). The test is a tolerance check
  of 1e-6 on each published ρ.

### Fact-check pass (2026-10-08)

A second, independent pass re-fetched every source and re-ran every check under the same rule: no
NLIBD measurement file was opened. Sources it re-fetched: the PMC XML, the bioRxiv XML and SI, the
publisher's SI and figures, a fresh clone of both repositories, PLoS S3 Table and the MelonnPan SI.

**Confirmed unchanged:**

- every quoted sentence;
- every section and figure location;
- 0.2871 / 0.6221 / 69, and 0.261 / 0.713 / 1,237 over all features;
- the n = 65 Spearman fit;
- the June 2022 values 0.277 / 0.680 / 102;
- the md5 matches;
- the CLR(x + 1) reproduction (5e-15 / 3e-15) and r = 0.997;
- the counts 151 / 163 / 8,847 / 427 / 6,948 / 39;
- 0.290 / 0.619 / 66, 0.258 / 0.705 / 921 and 0.254;
- the raw-vs-CLR Spearman values 0.959 / 0.898 / 0.739 and 0.40;
- the code line references;
- the Fig. 3a and ED Fig. 8 readings (re-measured).

**Corrected:**

1. "The reported run is the upper outlier" overstated what the figure shows. It holds for mean ρ and
   N(ρ > 0.5) only; the published top-50 value is inside the whisker. Whether the published run is one
   of the 50 cannot be checked.
2. "Test-set early stopping biases upwards" was stated as fact. It is now "probably optimistic, not
   quantified": the rule has no best-epoch restore.
3. The verdict said "same 200 species". Our filter keeps 151 of the 200.
4. MiMeNet S3 Table was labelled "CLR input / RA input". Its blocks are CLR- vs RA-transformed data,
   and IBD microbes are RA in both blocks.
5. For the June 2022 file, "none" of the values fitting the Spearman test became "at most 1 of 8,848
   per n".
6. "The author's other repositories are unrelated" became "contain no PRISM + NLIBD baseline code".
7. The pixel precision is now stated per figure.
8. The `kfolds` quote now uses the data-container wording.
9. Inferred subsets are now marked as inferred: ED Fig. 8 and MiMeNet S4 Fig D.
