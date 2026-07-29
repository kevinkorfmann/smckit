# Testing

The acceptance suite has five explicit tiers. A test may carry additional
`slow` or `gpu` markers, but it must have exactly one primary tier based on its
path or explicit marker.

## Tiers and budgets

| Tier | Purpose | Routine budget |
|---|---|---:|
| `unit` | Pure Python/native kernels, no external runtime | under 5 minutes |
| `integration` | Packaged workflows and fixed fixtures | per-PR |
| `oracle` | Live native-versus-original comparisons | scheduled/on demand |
| `benchmark` | Runtime and peak-memory regression | scheduled |
| `publication` | Frozen end-to-end evidence workflow | release gate |

Run them with:

```bash
pixi run test-unit
pixi run test-integration
pixi run -e upstream test-oracle
uv run pytest -m benchmark
uv run pytest -m publication
```

Expensive optimization, compilation, and whole-genome cases must be marked
`slow`; they never belong in routine unit feedback. The scheduled jobs retain
coverage.

## Numerical acceptance

Every native method has a method-specific oracle specification. Identical
deterministic kernels normally use machine-precision tolerances. Unless the
specification justifies another scientific metric, final deterministic results
must meet 0.1% scalar relative error, 1% normalized trajectory error, and
`1e-6` per-site log-likelihood error.

Stochastic optimization is evaluated on at least 20 paired simulations. Native
results must lie within upstream seed-to-seed variation and show no material
paired bias.

Optimization starts only after correctness is locked. Parsing, compilation/JIT,
algorithmic execution, and plotting are timed separately. Native promotion
requires a warmed-runtime bootstrap confidence interval excluding parity and
peak memory at most 25% above upstream.

## Required edge cases

Acceptance covers every documented example and option, malformed and missing
inputs, masks, empty/short sequences, multiple chromosomes/files/records,
interrupted upstream processes, absent runtimes, unsupported native requests,
schema round trips, CPU/backend agreement, x86/ARM agreement, fixed-seed
determinism, and long-chromosome stability.

Package tests install wheels into a clean environment without the source tree.
Preservation tests also exercise source-checkout and container execution.

The machine-readable ledgers are in `docs/parity/feature-ledger.json` and
`docs/parity/oracle-specifications.json`.
