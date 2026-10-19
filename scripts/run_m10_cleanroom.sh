#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${M10_PYTHON:-python3.11}
VENV_DIR=${M10_VENV:-/tmp/verireason-m10-venv}

if [ -e "$VENV_DIR" ]; then
  echo "Refusing to reuse an existing clean-room path: $VENV_DIR" >&2
  echo "Set M10_VENV to a new empty path." >&2
  exit 2
fi

"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
  --requirement "$ROOT/requirements/m10-cpu-lock.txt"
"$VENV_DIR/bin/python" -m pip install --disable-pip-version-check \
  --no-build-isolation --no-deps "$ROOT"

make -C "$ROOT" m10-verify \
  PYTHON="$VENV_DIR/bin/python" \
  PIP_CHECK_PYTHON="$VENV_DIR/bin/python" \
  RUFF="$VENV_DIR/bin/python -m ruff"

echo "M10 clean-room verification passed: $VENV_DIR"
