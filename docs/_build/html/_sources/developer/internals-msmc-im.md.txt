# MSMC-IM Internals

This page holds the implementation notes behind {func}`smckit.tl.msmc_im`.
The practical overview lives on [the MSMC-IM method page](../methods/msmc-im.md).

## Why this page exists

MSMC-IM is conceptually simpler for users than it is to implement: it takes
coalescence-rate curves and fits a continuous two-population IM model. The
main docs stay focused on interpretation; this page keeps the model mechanics.

## Internal model shape

- Input is the combined MSMC/MSMC2 output with `lambda_00`, `lambda_01`, and
  `lambda_11`.
- The fitted state space is a five-state continuous-time IM model.
- Optimization is done in log-space over piecewise `N1`, `N2`, and `m`.
- The main outputs are corrected `N1(t)`, `N2(t)`, `m(t)`, and cumulative
  migration `M(t)`.

## Review hotspots

- Parsing and validating the time-segment pattern
- Converting MSMC lambdas into TMRCA densities
- Propagating the five-state IM process through interval-specific parameters
- Numerical integration and Powell optimization behavior
- Post-fit migration correction and split-time quantile extraction

## Where to read code

- `src/smckit/tl/_msmc_im.py`
- `src/smckit/io/_multihetsep.py`
- `vendor/MSMC-IM`

## What stayed out of the main docs

- Full five-state rate-matrix derivation
- Hazard/TMRCA algebra
- Pattern-expansion details
- Objective-function regularization formulas
