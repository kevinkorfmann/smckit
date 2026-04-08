# SMC++ Internals

This page holds the technical notes behind {func}`smckit.tl.smcpp`. The
user-facing overview lives on [the SMC++ method page](../methods/smcpp.md).
The step-by-step parity closure record lives on
[SMC++ parity closure notes](./smcpp-parity-closure.md).

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
- The default native one-pop preprocessing path now also mirrors the upstream
  `BreakLongSpans` leading missing-base offset. That one-base phase shift was
  the last small-control preprocessing difference.
- The native one-pop HMM now stays on the compressed upstream run-length
  observation stream rather than expanding spans back into unit rows.
- The strict small control and larger tracked one-pop `.smc` fixture now both
  clear the shared-grid parity gate, and fixed-model `gamma0`, `xisum`, and
  log-likelihood match upstream tightly on the same matrix.

## Handoff notes

- Treat `vendor/smcpp` as the oracle source tree for future parity work.
- The remaining work is fixture coverage, not the tracked one-pop HMM or
  optimizer semantics.
- Tracked one-pop parity now holds on both enforced fixtures, but that still
  does not automatically prove parity for future untracked SMC++ input shapes.
- The larger tracked `.smc` fixture remains in the docs gallery as the more
  realistic one-pop parity panel.
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
- `docs/developer/smcpp-parity-closure.md`

## What stayed out of the main docs

- Full CSFS derivation
- Rank-structure algebra for span transitions
- Exact observation-layout notation
- Optimization objective details for every native approximation path
