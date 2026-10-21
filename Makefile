PYTHON ?= python3
RUFF ?= $(PYTHON) -m ruff
PIP_CHECK_PYTHON ?= $(PYTHON)

.PHONY: env-check-gpu env-check-local guard help lint m1-verify m2-finalize m2-verify m3-closure-verify m3-verify m4-verify m6-verify m7-preflight-verify m8-preflight-verify m9-closure-verify m9-preflight-verify m10-cleanroom m10-verify m11-verify smoke status test verify

help:
	@echo "make verify          Run durable-state guard and unit tests"
	@echo "make m1-verify       Run verify plus the local environment check"
	@echo "make m2-verify       Run verify plus lint and bytecode compilation"
	@echo "make m2-finalize     Finalize M2 manifests from frozen private evidence"
	@echo "make m3-verify       Verify Day 4A M3 contracts without model/data downloads"
	@echo "make m3-closure-verify  Re-audit the returned M3 attempt-003 evidence"
	@echo "make m4-verify       Verify M4 pre-cloud contracts without GPU/data/model access"
	@echo "make m6-verify       Verify M6 reward math and local safety contracts"
	@echo "make m7-preflight-verify  Verify the unapproved M7 E2 smoke contract"
	@echo "make m8-preflight-verify  Verify M8 evaluation and E3/E4 post-training contracts"
	@echo "make m9-preflight-verify  Verify the local M9 serving and benchmark contract"
	@echo "make m9-closure-verify  Rebuild and check the downloaded M9 serving summary"
	@echo "make m10-verify      Verify the CPU release packet and all public evidence"
	@echo "make m10-cleanroom   Install and verify in a newly created virtual environment"
	@echo "make m11-verify      Verify the public release surface and headline evidence"
	@echo "make env-check-gpu   Validate the pinned Linux/CUDA stack"
	@echo "make smoke           Run an approved GPU smoke via scripts/run_m1_smoke.sh"

guard:
	$(PYTHON) scripts/check_project_state.py

test:
	PYTHONPATH=src $(PYTHON) -m pytest

lint:
	$(RUFF) check src tests scripts

env-check-local:
	PYTHONPATH=src $(PYTHON) -m verireason.envcheck --profile local

env-check-gpu:
	PYTHONPATH=src $(PYTHON) -m verireason.envcheck --profile gpu

verify: guard test

.PHONY: e5-verify
e5-verify:
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_e5_credit.py
	PYTHONPATH=src $(PYTHON) scripts/diagnose_e5.py --output results/e5/e5-cpu-feasibility.json

.PHONY: e5-interface-verify
E5_TOKENIZER_JSON ?= artifacts/e5-interface/qwen3-tokenizer/tokenizer.json
E5_TRL_WHEEL ?= artifacts/e5-interface/trl-1.9.0-py3-none-any.whl
e5-interface-verify:
	PYTHONPATH=src $(PYTHON) scripts/diagnose_e5_interface.py \
	  --tokenizer-json "$(E5_TOKENIZER_JSON)" --trl-wheel "$(E5_TRL_WHEEL)" \
	  --output results/e5/e5-cpu-interface.json
	E5_TOKENIZER_JSON="$(E5_TOKENIZER_JSON)" PYTHONPATH=src $(PYTHON) -m pytest tests/test_e5_interface.py

.PHONY: e5-runner-verify
e5-runner-verify:
	$(PYTHON) -c "import torch, transformers, trl, jsonschema; from pathlib import Path; assert torch.__version__ == '2.11.0+cpu'; assert transformers.__version__ == '5.14.1'; assert trl.__version__ == '1.9.0'; assert Path('artifacts/e5-interface/qwen3-tokenizer/tokenizer.json').is_file()"
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 E5_CPU_RECEIPT=results/e5/e5-cpu-runner.json PYTHONPATH=src $(PYTHON) -m pytest tests/test_e5_credit.py tests/test_e5_interface.py tests/test_e5_runner.py

m1-verify: verify env-check-local

m2-verify: verify lint
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests

m3-verify: verify lint
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_m3_contract.py tests/test_m3_orchestrator.py tests/test_m3_closure.py

m3-closure-verify:
	PYTHONPATH=src $(PYTHON) scripts/close_m3_result.py \
	  --evidence-dir artifacts/m3-cloud-attempt-003 \
	  --output results/m3/m3-vertical-slice-summary.json \
	  --execution-commit 530a0f22f491882def5e13b8344d1310c1a7960d \
	  --pin-commit f1a50c463c011153fbe6ea098e5c8a72e53af150 \
	  --provider-charge 2026-08-24T04:00:17+08:00=1.43241388 \
	  --provider-charge 2026-08-24T04:56:46+08:00=2.45042777 \
	  --previous-cumulative-cost-cny 9.64484428 \
	  --provider-balance-cny 24.18879004 \
	  --check

m4-verify: verify lint
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests
	PYTHONPATH=src $(PYTHON) -m pytest -rs tests/test_m4_contract.py tests/test_m4_trainer_regression.py
	PYTHONPATH=src $(PYTHON) scripts/verify_m4_trainer_contract.py --allow-unavailable
	$(PIP_CHECK_PYTHON) -m pip check

m6-verify: verify lint
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_reward_contract.py tests/test_m6_sql_features.py tests/test_m6_audit.py

m7-preflight-verify: verify lint
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_m7_contract.py tests/test_m7_runtime.py tests/test_trl_reward_adapter.py tests/test_m6_component_scorer.py

m8-preflight-verify: verify lint
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_m8_contract.py tests/test_m8_rl_contract.py tests/test_posttraining_result.py tests/test_m4_contract.py tests/test_m7_contract.py tests/test_m7_runtime.py tests/test_trl_reward_adapter.py tests/test_m6_component_scorer.py

m9-preflight-verify: verify lint
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests
	PYTHONPATH=src $(PYTHON) -m pytest tests/test_m9_serving.py

m9-closure-verify: m9-preflight-verify
	PYTHONPATH=src $(PYTHON) scripts/summarize_m9_serving.py \
	  --config configs/m9_serving.toml \
	  --raw results/m9/raw/m9-serving-raw-private.json \
	  --gpu-samples results/m9/raw/gpu-samples.csv \
	  --output results/m9/m9-serving-summary.json \
	  --check

m10-verify: m2-verify env-check-local
	PYTHONPATH=src $(PYTHON) scripts/verify_m10_release.py \
	  --config configs/m10_public_artifacts.toml \
	  --manifest results/m10/release-manifest.json \
	  --check
	$(PIP_CHECK_PYTHON) -m pip check

m10-cleanroom:
	sh scripts/run_m10_cleanroom.sh

m11-verify: m10-verify
	PYTHONPATH=src $(PYTHON) scripts/verify_m11_release.py \
	  --config configs/m11_release.toml

m2-finalize:
	PYTHONPATH=src $(PYTHON) scripts/finalize_m2_data.py \
	  --materialization-partial "$(M2_PARTIAL)" \
	  --diagnostic-checkpoint "$(M2_DIAGNOSTIC_CHECKPOINT)" \
	  --training-labels "$(M2_TRAINING_LABELS)" \
	  --heldout-labels "$(M2_HELDOUT_LABELS)" \
	  --final-dir "$(M2_FINAL_DIR)"

smoke:
	sh scripts/run_m1_smoke.sh

status:
	@git status --short --branch
	@sed -n '/^## Current phase/,/^## /p' docs/STATE.md
	@sed -n '/^## Exactly one next action/,$$p' docs/STATE.md
