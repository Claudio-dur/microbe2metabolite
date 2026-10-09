# Provenance of commit IDs

Before publication (2026-10-09), the commit history was rewritten to change only the author and
committer identity. The commits had been made with a work git identity by mistake.

Files, commit messages and dates are unchanged. Every commit keeps its tree hash, so each
rewritten commit contains exactly the same files as the original.

## Why this matters

The results files were written before the rewrite, so they record the **original** commit IDs
in their `git_sha` column:

- `results/runs.csv`
- `results/summary.csv`
- `results/external_access_log.csv`

They were not edited, because they are the evidence of what ran. This table maps the original
IDs to the published ones.

| Original | Published | Tree (unchanged) | Commit |
|---|---|---|---|
| `067fc8b` | `ed235c0` | `4bfd471a801eb1` | Initial commit: MIT licence and ignore rules |
| `caab2dd` | `751f679` | `0e42af7a0609c4` | M1: data download, loaders, section-3 data checks, preprocessing, EDA notebook |
| `c13face` | `2f22f4f` | `c34da4c0b4dd06` | M2: evaluation harness and E0 baselines |
| `d38b655` | `54c17c3` | `5666527863194e` | M3: E1 MLP from scratch |
| `0145499` | `efb6016` | `cbb5eb7ed46467` | M4: E2/E3 with the MGM foundation model, and their from-scratch controls |
| `21b88fb` | `5156bb8` | `800c736526bfa1` | M5: E4 transfer, E5 learning curves, E6 interpretation, ablations, paired comparisons |
| `ce8fb15` | `f1d7f3c` | `d28f86aa98c440` | Freeze model selection before the external evaluation |
| `f0fbae7` | `fedab73` | `7c5bae1029ce85` | M6: results of the single external evaluation, README write-up |
| `79b1994` | `5ce8c35` | `e68bdfe1830328` | Docs: corrections from an independent fact-check |

## Reading the logged IDs

- **`ce8fb15` is the freeze commit, now `f1d7f3c`.**
  - Every row of the single external evaluation carries this ID: the 22 evaluation rows of the
    access log, and every row of `summary.csv`.
  - The 14 data-check rows of the access log have no ID, because they predate the first commit.
- **`21b88fb-dirty` is the M5 commit, now `5156bb8`.**
  - It marks the selection-phase rows of `runs.csv`.
  - "-dirty" means `results/selected/` and `results/splits/` held uncommitted changes while
    `make select` wrote its selections. The freeze commit then recorded those selections.

## Checking a row

Run:

```bash
git rev-parse <published>^{tree}
```

It prints the tree hash in the table; the table shows the first 14 characters of it.

To check the freeze itself, run `python -m m2m.freeze check`. It compares the committed
selections and configs with `results/selected/FROZEN.json`, and does not depend on commit IDs.
