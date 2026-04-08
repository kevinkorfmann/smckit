# eSMC2 Internals

This page keeps the technical notes behind {func}`smckit.tl.esmc2`. The
practical overview lives on [the eSMC2 method page](../methods/esmc2.md).

## Why this page exists

eSMC2 adds ecological parameters on top of pairwise SMC, and the equations can
take over the page very quickly. The main docs now focus on interpretation and
workflow; this page is for implementation review and parity work.

## Core model additions over PSMC

- `beta` controls dormancy / germination
- `sigma` controls self-fertilization
- `Xi` tracks piecewise relative population size
- emission and transition quantities are rescaled by dormancy/selfing effects

## Practical parity notes

- The native HMM builder is a line-by-line translation target for the vendored
  R formulas.
- Final-state builder and sufficient-statistics comparisons match the upstream
  oracle on the tracked fixture matrix.
- Native fit parity is currently enforced for fixed-rho, rho-redo, beta,
  sigma, beta+sigma, and grouped `pop_vect=[3,3]` cases on the clean 800 bp
  fixture.
- Broader grouped layouts and multi-sequence coverage are still not fully
  gated, so the method remains dual-path and `auto` should prefer upstream
  when available.

## Upstream bridge

- `implementation="upstream"` runs the vendored R package through `Rscript`.
- `implementation="native"` runs the in-repo Python/Numba port.
- The upstream bridge currently supports the clean pairwise sequence path used
  by the parity fixtures.

## Where to read code

- `src/smckit/tl/_esmc2.py`
- `src/smckit/backends/_numba_esmc2.py`
- `vendor/eSMC2`

## What stayed out of the main docs

- Full time-boundary derivation
- Stationary-distribution equations
- Full transition-matrix formula cases
- Optimization-space transforms for every bounded parameter
