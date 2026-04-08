# smckit

Unified framework for Sequentially Markovian Coalescent methods.

PSMC, MSMC/MSMC2, SMC++ — one API, GPU-accelerated.

Preserve upstream algorithms now; grow native implementations over time.

## Philosophy

smckit is upstream-first by design.

- `vendor/` holds the original source or release artifacts that act as the oracle.
- `implementation="upstream"` means "run the original tool" when that bridge is wired and ready.
- `implementation="native"` means "run the in-repo implementation".
- `implementation="auto"` should prefer upstream fidelity whenever the upstream path is ready.

The repository now exposes `smckit.upstream.status()` and
`smckit.upstream.bootstrap()` so upstream readiness is inspectable rather than
implicit.
