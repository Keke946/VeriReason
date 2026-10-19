#!/bin/sh
set -eu
umask 077

: "${M7_APPROVED_COMMIT:?set the control-approved full SHA}"
: "${M7_DATASET_ROOT:?set the retained M2 dataset root}"
: "${M7_TRAINING_MANIFEST:?set training-manifest.json}"
: "${M7_M4_OUTPUT_DIR:?set the retained M4 output directory}"
: "${M7_OUTPUT_DIR:?set a fresh persistent path outside Git}"
: "${M7_CONFIG:=configs/m7_e2_smoke.toml}"
export M7_CONFIG
exec env PYTHONPATH=src python scripts/orchestrate_m7_smoke.py
