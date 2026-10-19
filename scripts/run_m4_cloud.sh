#!/bin/sh
set -eu
umask 077

: "${M4_APPROVED_COMMIT:?set the control-approved full SHA}"
: "${M4_DATASET_ROOT:?set the retained M2 dataset root}"
: "${M4_MANIFEST_INDEX:?set manifest-index.json}"
: "${M4_TRAINING_MANIFEST:?set training-manifest.json}"
: "${M4_OUTPUT_DIR:?set a persistent path outside Git}"
exec env PYTHONPATH=src python scripts/orchestrate_m4.py
