# Reproducible evidence workflow

This directory contains public workflow code and immutable evidence schemas.
It does not contain the private manuscript, supplement, or submission files.

## PSMC+ promotion capability matrix

Run the deterministic native-versus-upstream matrix from a clean checkout with
the PSMC+ optional dependencies installed:

```bash
PYTHONPATH=src python workflow/publication/scripts/validate_psmcplus_matrix.py \
  --output psmcplus-promotion-matrix.json
```

The retained clean-commit Linux x86-64 record is under
`evidence/psmcplus-promotion/sha256-5aa89ae3/`. It passes all twelve fit/decode
cases. This closes the expanded deterministic simulation surface, but does not
by itself authorize native `auto`; human/nonhuman empirical validation and the
macOS ARM64 matrix remain open gates.

The conceptual preservation/promotion schematic is code-generated and can be
rendered independently of benchmark evidence:

```bash
python workflow/publication/scripts/plot_architecture.py \
  --output-prefix /private/path/figure2_architecture
```

It writes PDF, SVG, and a 600-dpi LZW TIFF using a colorblind-safe palette and
redundant lane/shape encoding.

## SMC++ split promotion benchmark

The SMC++ split benchmark uses one persistent process per implementation so the
record separates startup, fixture preparation, the first cold inference call,
and five genuinely warmed calls. Run native and preserved-upstream records in
the same controlled environment with identical thread settings. Set
`SMCKIT_SMCPP_PYTHON` to the pinned upstream interpreter before the upstream
run.

```bash
export PYTHONPATH="$PWD/src"
export SMCKIT_SMCPP_PYTHON=/absolute/path/to/pinned-smcpp-python

python workflow/publication/scripts/run_benchmark.py \
  --method smcpp \
  --implementation native \
  --dataset split-control-v1 \
  --measurement-component inference_api_excluding_input_preparation \
  --repetitions 5 \
  --threads 1 \
  --persistent-jsonl \
  --protocol-id sha256:REPLACE_WITH_FROZEN_PROTOCOL_HASH \
  --output results/benchmarks/smcpp-split-native.json \
  -- python -u workflow/publication/scripts/benchmark_smcpp_split.py \
  --implementation native

python workflow/publication/scripts/run_benchmark.py \
  --method smcpp \
  --implementation upstream \
  --dataset split-control-v1 \
  --measurement-component inference_api_excluding_input_preparation \
  --repetitions 5 \
  --threads 1 \
  --persistent-jsonl \
  --protocol-id sha256:REPLACE_WITH_FROZEN_PROTOCOL_HASH \
  --output results/benchmarks/smcpp-split-upstream.json \
  -- python -u workflow/publication/scripts/benchmark_smcpp_split.py \
  --implementation upstream
```

Do not use a placeholder protocol ID for retained evidence. Freeze
`config.yaml` with `scripts/freeze_protocol.py`, then copy the resulting
protocol hash into both commands. Keep the two raw JSON records unchanged.

Aggregate the matched records only after correctness and build gates are green:

```bash
python workflow/publication/scripts/aggregate_results.py \
  --benchmark results/benchmarks/smcpp-split-native.json \
  --benchmark results/benchmarks/smcpp-split-upstream.json \
  --required-warm-repetitions 5 \
  --output results/aggregates/smcpp-split.json
```

The aggregate applies the frozen bootstrap speed-confidence gate and the
native peak-memory limit. A promotable performance record is necessary but not
sufficient: complete parity, regression, installation, and documentation gates
must also pass before changing `auto` to native.

The retained SMC++ split promotion run is in
`evidence/smcpp-split/sha256-c339dbb6/`. Its aggregate reports an 11.01x warmed
speedup (95% bootstrap CI 9.47-11.11x) and a 0.512 native/upstream peak-memory
ratio. `SHA256SUMS` covers every file emitted by the frozen runner; the adjacent
source-provenance record links the narrowed clean Sesame execution snapshot to
the canonical repository commits and source hashes.

## Betty Slurm execution

Never run the validation or benchmark commands directly on the Betty login
node. Transfer or update the checkout and inspect status there, then submit the
tracked Slurm entrypoints:

```bash
sbatch \
  --export=ALL,SMCKIT_REPO=/persistent/path/smckit,SMCKIT_PYTHON=/controlled/env/bin/python,SMCKIT_SMCPP_PYTHON=/pinned/smcpp/env/bin/python \
  workflow/slurm/smcpp-validation.sbatch

sbatch \
  --dependency=afterok:VALIDATION_JOB_ID \
  --export=ALL,SMCKIT_REPO=/persistent/path/smckit,SMCKIT_PYTHON=/controlled/env/bin/python,SMCKIT_SMCPP_PYTHON=/pinned/smcpp/env/bin/python,SMCKIT_EVIDENCE_DIR=/persistent/path/evidence/smcpp-split \
  workflow/slurm/smcpp-promotion-benchmark.sbatch
```

Replace every placeholder explicitly. The benchmark job freezes the protocol,
writes native and upstream raw records plus their aggregate, captures the
Python environment, and generates `SHA256SUMS`. Select a partition or account
with normal site-specific `sbatch` flags; those values are intentionally not
hard-coded in the repository.
