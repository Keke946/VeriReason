# M10 Clean-room Reproduction Runbook

## Goal

M10 proves that the public repository is independently installable and internally verifiable
without a GPU, a private database, a model checkpoint, or cloud credentials. It does **not** claim
that CPU verification reruns the expensive SFT/RL experiments. Instead it verifies the code,
contracts, tests, and immutable SQL-free evidence produced by those experiments.

## Canonical environment

- Python: 3.11.15
- OS image: `python:3.11.15-slim-bookworm`
- Dependencies: exact CPU verification versions in `requirements/m10-cpu-lock.txt`
- GPU dependencies: excluded from M10; their measured environment remains documented by M1

## Fast local verification

In an already prepared Python 3.11 environment:

```bash
python -m pip install -r requirements/m10-cpu-lock.txt
python -m pip install --no-build-isolation --no-deps .
make m10-verify PYTHON=python PIP_CHECK_PYTHON=python
```

This runs the durable-state guard, complete unit suite, Ruff, bytecode compilation, the local
environment check, `pip check`, and the deterministic public-evidence manifest verifier.

## Fresh virtual environment

The script refuses to reuse an existing target so a stale environment cannot create a false pass:

```bash
M10_PYTHON=python3.11 \
M10_VENV=/tmp/verireason-m10-$(date +%s) \
make m10-cleanroom
```

## Container verification

```bash
docker build --tag verireason:m10 .
docker run --rm verireason:m10
```

The Docker build uses the exact Python patch version, installs no GPU stack, switches to an
unprivileged user, and runs `make m10-verify` during the image build and again by default at runtime.

## Public evidence boundary

`configs/m10_public_artifacts.toml` is the explicit allowlist. The verifier parses every listed JSON
file, rejects data-bearing SQL/question keys and SQL-like string values, and verifies exact byte
size and SHA-256 against `results/m10/release-manifest.json`. Raw generations, private databases,
heldout examples, adapters, and logs remain outside this manifest and outside Git.

Regenerate the manifest only after intentionally accepting a changed public artifact:

```bash
PYTHONPATH=src python scripts/verify_m10_release.py \
  --config configs/m10_public_artifacts.toml \
  --manifest results/m10/release-manifest.json \
  --write
```

Then inspect the diff and run `make m10-verify` before committing it.

## Verification receipt

Implementation commit `392b8a3155fac3c83b50745517b3c32dca3b6874` was exported with
`git archive` and installed into a new virtual environment using managed CPython 3.11.15. The
archive-only run passed 314 tests with four optional-runtime skips, Ruff, compileall, environment
validation, `pip check`, and reconstruction of all 19 public artifact identities. No ignored file
or prior virtual environment was present. Docker was not installed on the local Mac; its build and
runtime checks are therefore delegated to the checked-in Linux CI job and are not claimed as a
local measurement.
