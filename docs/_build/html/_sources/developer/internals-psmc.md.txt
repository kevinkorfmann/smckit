# PSMC Internals

This page captures the implementation-heavy material behind
{func}`smckit.tl.psmc`. The user-facing overview lives on
[the PSMC method page](../methods/psmc.md).

## Why this page exists

PSMC is both the historical foundation of the SMC family and the reference
shape for many smckit abstractions. The main method page stays practical; this
page keeps the details needed for porting and review.

## Source map

| Upstream file | Role in the original implementation |
|---|---|
| `main.c` | CLI entry point and orchestration |
| `cli.c` | argument parsing, pattern parsing, input/output |
| `core.c` | time grid and coalescent-to-HMM parameter mapping |
| `em.c` | EM loop and expectation maximization bookkeeping |
| `khmm.c` | forward, backward, Viterbi, expected counts |
| `kmin.c` | Hooke-Jeeves optimizer |

## Core model

- Hidden states are discretized coalescent time intervals.
- Observations are window-level homozygous, heterozygous, or missing calls.
- Free parameters are `theta`, `rho`, `max_t`, and grouped `lambda` values.
- The pattern string controls how neighboring intervals share `lambda`
  parameters.

## The parts that matter for parity

- Time intervals use the same exponential schedule as the C code.
- The HMM is built from the same survival-probability and interval-weighted
  quantities as upstream.
- EM uses forward/backward expected counts and a direct-search M-step.
- Output agreement is currently tracked on the bundled `NA12878_chr22` fixture.

## What stayed out of the main docs

- Full derivation of the transition matrix
- The exact `alpha`, `beta`, and auxiliary recurrence notation
- Detailed per-file C-to-Python mapping
- Porting notes for every low-level helper

When editing the implementation, check `src/smckit/tl/_psmc.py`,
`src/smckit/backends/_numba.py`, and the vendored `vendor/psmc` source
together.
