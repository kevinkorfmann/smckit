# SMC++ Internals

This page holds the technical notes behind {func}`smckit.tl.smcpp`. The
user-facing overview lives on [the SMC++ method page](../methods/smcpp.md).

## Why this page exists

SMC++ is one of the most mathematically dense methods in the repo. The main
docs now focus on when to use it and how the dual implementation paths behave.
This page preserves the deeper implementation notes needed for parity work.

## Core ingredients

- Distinguished-lineage HMM rather than pairwise diploid windows
- Conditioned site frequency spectrum (CSFS) emissions
- Piecewise demographic parameters over a fixed interval grid
- Upstream one-pop preprocessing plus EM/coordinate-update optimization
- A legacy one-distinguished native path that is now compatibility-only

## Porting hotspots

- Hidden-state construction and time-grid selection
- CSFS approximation and emission layout
- Span-based forward/backward recurrences
- Transition structure for one- and two-distinguished variants
- Native-vs-upstream comparison on shared `N_e(t)` grids

## Current native state

- Headerless `.smc` input now defaults to the upstream one-pop interpretation
  with `n_distinguished=2`.
- The default native fit now applies the one-pop preprocessing stack
  (thinning, binning, monomorphic recoding, compression) before fitting.
- The default native optimizer for the one-pop path now runs an
  upstream-style EM loop with the global scale step plus coordinate-wise
  M-steps instead of the older direct penalized likelihood minimization.
- The native one-pop reduced-emission path now respects the upstream
  observation scale (`alpha = bin width`) after preprocessing, which removed
  the earlier order-of-magnitude scale drift on the tracked fixture.
- The default native one-pop path now expands short compressed preprocessed
  spans back into unit observations during the HMM/E-step. That fixed the last
  tracked shape-sign mismatch and brought the small parity fixture above the
  `0.99` log-correlation gate.
- The fixed-model one-pop internals already have strong oracle checks against
  the upstream runtime, but end-to-end parity on larger one-pop fixtures is
  still incomplete.

## Handoff notes

- Treat `vendor/smcpp` as the oracle source tree for future parity work.
- The remaining gap is now mostly optimizer trajectory fidelity and fixture
  coverage, not basic emission/transition mechanics.
- Small one-pop parity fixtures now pass in the test suite, but that should
  not be read as full parity.
- The current tracked fixture now has both near-perfect scale agreement and
  the correct trajectory direction.
- The expensive legacy one-distinguished truth tests still exist for explicit
  compatibility checks; they are no longer the default native ceremony.

## Upstream bridge notes

- `implementation="upstream"` runs the vendored upstream package through the
  controlled side environment and maps results back into `SmcData`.
- `implementation="native"` stays entirely in-repo and records the same
  provenance metadata.
- `implementation="auto"` currently resolves to upstream when that side
  environment exists.

## Where to read code

- `src/smckit/tl/_smcpp.py`
- `src/smckit/tl/_smcpp_upstream_runner.py`
- `vendor/smcpp/`

## What stayed out of the main docs

- Full CSFS derivation
- Rank-structure algebra for span transitions
- Exact observation-layout notation
- Optimization objective details for every native approximation path
