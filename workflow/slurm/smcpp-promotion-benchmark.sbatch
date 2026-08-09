#!/usr/bin/env bash
#SBATCH --job-name=smckit-smcpp-benchmark
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail

: "${SMCKIT_REPO:?Set SMCKIT_REPO to the persistent smckit checkout.}"
: "${SMCKIT_PYTHON:?Set SMCKIT_PYTHON to the controlled Python executable.}"
: "${SMCKIT_SMCPP_PYTHON:?Set SMCKIT_SMCPP_PYTHON to the pinned upstream SMC++ Python.}"
: "${SMCKIT_EVIDENCE_DIR:?Set SMCKIT_EVIDENCE_DIR to a persistent empty output directory.}"
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

if [[ -e "$SMCKIT_EVIDENCE_DIR" ]]; then
  echo "Refusing to overwrite existing evidence path: $SMCKIT_EVIDENCE_DIR" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "Refusing to benchmark a dirty worktree." >&2
  git status --short >&2
  exit 2
fi
mkdir -p "$SMCKIT_EVIDENCE_DIR"

protocol="$SMCKIT_EVIDENCE_DIR/protocol.json"
native_record="$SMCKIT_EVIDENCE_DIR/smcpp-split-native.json"
upstream_record="$SMCKIT_EVIDENCE_DIR/smcpp-split-upstream.json"
aggregate="$SMCKIT_EVIDENCE_DIR/smcpp-split-aggregate.json"

"$SMCKIT_PYTHON" workflow/publication/scripts/freeze_protocol.py \
  workflow/publication/config.yaml \
  "$protocol"

protocol_id="$("$SMCKIT_PYTHON" -c \
  'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_text())["protocol_id"])' \
  "$protocol")"

"$SMCKIT_PYTHON" workflow/publication/scripts/run_benchmark.py \
  --method smcpp \
  --implementation native \
  --dataset split-control-v1 \
  --measurement-component inference_api_excluding_input_preparation \
  --repetitions 5 \
  --threads 1 \
  --persistent-jsonl \
  --protocol-id "$protocol_id" \
  --output "$native_record" \
  -- "$SMCKIT_PYTHON" -u workflow/publication/scripts/benchmark_smcpp_split.py \
  --implementation native

"$SMCKIT_PYTHON" workflow/publication/scripts/run_benchmark.py \
  --method smcpp \
  --implementation upstream \
  --dataset split-control-v1 \
  --measurement-component inference_api_excluding_input_preparation \
  --repetitions 5 \
  --threads 1 \
  --persistent-jsonl \
  --protocol-id "$protocol_id" \
  --output "$upstream_record" \
  -- "$SMCKIT_PYTHON" -u workflow/publication/scripts/benchmark_smcpp_split.py \
  --implementation upstream

"$SMCKIT_PYTHON" workflow/publication/scripts/aggregate_results.py \
  --benchmark "$native_record" \
  --benchmark "$upstream_record" \
  --required-warm-repetitions 5 \
  --output "$aggregate"

"$SMCKIT_PYTHON" -m pip freeze --local > "$SMCKIT_EVIDENCE_DIR/python-environment.txt"
git rev-parse HEAD > "$SMCKIT_EVIDENCE_DIR/git-commit.txt"
uname -a > "$SMCKIT_EVIDENCE_DIR/uname.txt"
lscpu > "$SMCKIT_EVIDENCE_DIR/lscpu.txt"
"$SMCKIT_PYTHON" -c \
  'import hashlib, pathlib, sys; root = pathlib.Path(sys.argv[1]); paths = sorted(path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS"); (root / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in paths))' \
  "$SMCKIT_EVIDENCE_DIR"
