# microbe2metabolite — `make all` reproduces every result from a fresh clone.
#
# The protocol runs in two phases (docs/DESIGN.md, decision #24), and the code enforces it:
#   make select    PRISM only: CV, model selection, pretraining, interpretation. Never opens the
#                  external cohort. Ends by freezing a manifest of every selection and config
#                  (python -m m2m.freeze write).
#   (commit)       results/selected and results/splits. In a fresh clone they are already
#                  committed, and re-deriving them on the same platform reproduces them exactly.
#   make evaluate  The single external evaluation of every selected configuration. Every opening
#                  of the external cohort refuses unless the freeze exists, nothing changed since,
#                  and it is committed.
# The one-experiment targets (baselines, mlp, ...) run that experiment's selection stages only.
# Run any subset of stages with
#   .venv/bin/python -m m2m.run configs/<experiment>.yaml --stage <stage>[,<stage>...]

PY  := .venv/bin/python
RUN := $(PY) -m m2m.run

.PHONY: all env mgm-env data check-data select evaluate train figures report \
        baselines mlp mgm transfer ablations curves interpret test test-mgm lint eda clean-results

all: env data mgm-env train eda

## Environments -----------------------------------------------------------------------------
env: .venv/.installed
.venv/.installed: pyproject.toml
	uv venv --allow-existing --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"
	touch $@

# The MGM environment installs MGM from the downloaded clone, so it needs `make data` first.
mgm-env: envs/mgm/.venv/.installed
envs/mgm/.venv/.installed: envs/mgm/requirements.txt scripts/setup_mgm_env.sh | data/raw/MGM
	bash scripts/setup_mgm_env.sh
	touch $@

data/raw/MGM:
	$(MAKE) data

## M1: data ----------------------------------------------------------------------------------
data: env
	bash scripts/download_data.sh
	$(MAKE) check-data

check-data: env
	$(PY) scripts/check_data.py

## The protocol -----------------------------------------------------------------------------
train: select evaluate

select: env mgm-env
	$(RUN) configs/e0_baselines.yaml --stage cv
	$(RUN) configs/e1_mlp.yaml --stage select
	$(RUN) configs/e2_mgm_frozen.yaml --stage prepare,select
	$(RUN) configs/e3_mgm_finetune.yaml --stage prepare-cv,run-cv,select
	$(RUN) configs/e4_transfer.yaml --stage sources,e4a,e4b,select
	$(RUN) configs/ablations.yaml --stage cv
	$(RUN) configs/e5_learning_curves.yaml --stage subsets
	$(RUN) configs/e6_interpret.yaml --stage importance
	$(PY) -m m2m.freeze write

evaluate: env mgm-env
	$(PY) -m m2m.freeze check
	$(RUN) configs/e0_baselines.yaml --stage null,external,curve
	$(RUN) configs/e1_mlp.yaml --stage external,curve
	$(RUN) configs/e2_mgm_frozen.yaml --stage embed-external,external,curve
	$(RUN) configs/e3_mgm_finetune.yaml --stage prepare-final,run-final,evaluate
	$(RUN) configs/e4_transfer.yaml --stage null,external,curve
	$(RUN) configs/ablations.yaml --stage external
	$(RUN) configs/compare.yaml
	$(RUN) configs/e5_learning_curves.yaml --stage check,figure
	$(RUN) configs/e6_interpret.yaml --stage classes
	$(MAKE) figures report

figures:
	$(RUN) configs/figures.yaml

report:
	$(PY) scripts/report_tables.py

## One experiment's selection stages (its evaluation stages run only in `make evaluate`) -------
baselines:
	$(RUN) configs/e0_baselines.yaml --stage cv

mlp:
	$(RUN) configs/e1_mlp.yaml --stage select

mgm: mgm-env
	$(RUN) configs/e2_mgm_frozen.yaml --stage prepare,select
	$(RUN) configs/e3_mgm_finetune.yaml --stage prepare-cv,run-cv,select

transfer:
	$(RUN) configs/e4_transfer.yaml --stage sources,e4a,e4b,select

ablations:
	$(RUN) configs/ablations.yaml --stage cv

curves:
	$(RUN) configs/e5_learning_curves.yaml --stage subsets

interpret:
	$(RUN) configs/e6_interpret.yaml --stage importance

## Quality -----------------------------------------------------------------------------------
test:
	$(PY) -m pytest

# Opt-in: runs envs/mgm on synthetic genus tables (reads MGM's weights from data/raw).
test-mgm:
	M2M_TEST_MGM_ENV=1 $(PY) -m pytest tests/test_exp_mgm_frozen.py -k envs_mgm

lint:
	$(PY) -m ruff check .

eda:
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

# Removes generated outputs but keeps what a rerun and an audit need: the saved splits, the
# frozen selections, the data checks and the external-access log (append-only evidence).
clean-results:
	find results -type f ! -path 'results/splits/*' ! -path 'results/selected/*' \
		! -name 'data_checks.md' ! -name 'external_access_log.csv' -delete
