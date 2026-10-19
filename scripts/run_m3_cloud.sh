#!/bin/sh
set -eu
umask 077

: "${M3_DATASET_ROOT:?set M3_DATASET_ROOT to the published M2 root}"
: "${M3_TRAINING_MANIFEST:?set M3_TRAINING_MANIFEST}"
: "${M3_MANIFEST_INDEX:?set M3_MANIFEST_INDEX}"
: "${M3_OUTPUT_DIR:?set M3_OUTPUT_DIR outside Git}"
: "${M3_APPROVED_COMMIT:?set M3_APPROVED_COMMIT to the control-approved full SHA}"

exec env PYTHONPATH=src python scripts/orchestrate_m3.py
