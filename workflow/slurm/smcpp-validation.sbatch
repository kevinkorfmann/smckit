#!/usr/bin/env bash
#SBATCH --job-name=smckit-smcpp-validation
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

: "${SMCKIT_REPO:?Set SMCKIT_REPO to the persistent smckit checkout.}"
: "${SMCKIT_PYTHON:?Set SMCKIT_PYTHON to the controlled Python executable.}"
: "${SMCKIT_SMCPP_PYTHON:?Set SMCKIT_SMCPP_PYTHON to the pinned upstream SMC++ Python.}"
: "${SLURM_TMPDIR:?This script must run inside a Slurm allocation with SLURM_TMPDIR.}"

cd "$SMCKIT_REPO"

export PYTHONPATH="$SMCKIT_REPO/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$SLURM_TMPDIR/matplotlib"
mkdir -p "$MPLCONFIGDIR"

"$SMCKIT_PYTHON" -m pytest -q \
  tests/unit/test_smcpp.py \
  tests/unit/test_smcpp_cross_validation.py \
  tests/unit/test_smcpp_input.py \
  tests/unit/test_smcpp_model_io.py \
  tests/unit/test_smcpp_plotting.py \
  tests/unit/test_smcpp_preprocessing.py \
  tests/unit/test_publication_aggregate.py \
  tests/unit/test_publication_architecture.py \
  tests/unit/test_publication_benchmark.py \
  tests/unit/test_publication_protocol.py \
  tests/unit/test_validation_metrics.py \
  tests/integration/test_smcpp_parity_matrix.py \
  tests/integration/test_smcpp_split_validation.py \
  tests/integration/test_smcpp_upstream_validation.py \
  tests/integration/test_smcpp_validation.py \
  tests/integration/test_smcpp_vcf2smc_oracle.py

"$SMCKIT_PYTHON" -m build --outdir "$SLURM_TMPDIR/dist"
"$SMCKIT_PYTHON" -m sphinx -W --keep-going -b html docs "$SLURM_TMPDIR/docs-html"
