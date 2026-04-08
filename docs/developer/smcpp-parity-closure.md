# SMC++ Parity Closure Notes

This note records what actually closed the tracked native-vs-upstream SMC++
one-pop gap, why those changes mattered, and which earlier changes were
necessary but not sufficient.

## Scope

The closure described here is for the tracked one-pop SMC++ contract used by
smckit:

- one population
- `n_distinguished=2`
- upstream-style preprocessing
- the strict small control fixture
- the bundled larger `tests/data/smcpp_onepop_larger.smc` fixture

It is not a blanket claim about every possible future SMC++ input family.

## Final state

The tracked one-pop matrix is now green:

- small control: `log_corr=0.9991253179`, `scale_ratio=0.9999492985`,
  `median_rel=5.07e-05`
- larger tracked fixture: `log_corr=0.9999910567`,
  `scale_ratio=0.9999821749`, `median_rel=1.78e-05`

Fixed-model HMM statistics also now match upstream tightly on both fixtures:

- `gamma0_rel <= 4.4e-08`
- `xisum_rel <= 6.9e-06`
- log-likelihood absolute error `<= 2.3e-05`

## What was actually wrong

Two issues were the final blockers.

### 1. Native one-pop preprocessing was not phase-aligned with upstream

Upstream `BaseAnalysis` always runs one-pop contigs through
`BreakLongSpans` before the later one-pop thinning/binning stack. Even when
there are no long missing spans to split, that code path prepends a single
missing row at the start of each contig.

That one-base missing row changes the phase of:

- thinning
- 100 bp binning
- downstream compressed one-pop observation blocks

Native preprocessing had skipped that offset, so the small control fixture was
being fit on a slightly different compressed observation stream than upstream.

This was subtle because:

- emissions matched
- transition matrices matched
- larger-fixture end-to-end parity was already very strong

But the small control still differed at the fixed-model likelihood/stat level
because the actual run-length encoded observation rows were not identical.

### 2. Native one-pop HMM was leaving the upstream compressed-row ceremony

Native had been expanding compressed one-pop spans back into unit observations
for the E-step. That seemed attractive because it avoided spectral
approximation concerns, but it was not what upstream actually does.

Upstream runs the one-pop HMM on compressed run-length rows and uses the
span-aware eigensystem / `span_Qs` machinery directly on those rows.

So even after transitions and emissions were effectively exact, native was
still taking a different posterior-accounting path on repeated one-pop blocks.

## How the issue was found

The debugging sequence that finally mattered was:

1. Match the obvious ingredients first.
   - hidden states
   - Watterson scaling
   - one-pop emissions
   - one-pop transition matrix
2. Compare native vs upstream at a fixed model, not just at final curves.
   - `gamma0`
   - `xisum`
   - `log_likelihood`
3. Notice that the small control still drifted even when transition and
   emission probabilities were essentially exact.
4. Check the forward likelihood directly and confirm the mismatch existed
   before any native M-step explanation could be convincing.
5. Inspect the upstream preprocessing code paths and replay the upstream
   pipeline on the small control.
6. Compare the actual preprocessed row streams.

That comparison exposed the real issue:

- upstream preprocessed stream: 1299 rows, total span 6001
- native preprocessed stream: 1298 rows, total span 6000

The difference was the leading missing row injected by upstream
`BreakLongSpans`, which shifts the thinning/binning phase and changes the
compressed one-pop record.

## The fixes that closed parity

These are the changes that directly closed the tracked one-pop matrix.

### Upstream preprocessing offset in native one-pop preprocessing

In `src/smckit/tl/_smcpp.py`, native preprocessing now prepends the same
leading `(1, -1, 0, 0)` missing row before running the one-pop thinning /
binning / monomorphic recoding / compression stack.

Why this worked:

- it reproduces the upstream contig phase exactly
- thinning now lands on the same sites
- bin boundaries now line up with upstream
- compressed one-pop block keys and spans now match upstream

### Stay on the compressed run-length stream in the one-pop HMM

In `src/smckit/tl/_smcpp.py`, native one-pop E-step no longer expands short
compressed rows back into unit observations. It now keeps the compressed
upstream run-length rows and uses the same span-aware HMM semantics.

Why this worked:

- it matches upstream’s actual one-pop `HMM::Estep` contract
- repeated monomorphic blocks are handled through the same span-aware
  eigensystem path
- fixed-model `gamma0`, `xisum`, and log-likelihood now line up with upstream

## Earlier fixes that mattered

The final closure was not one patch. These earlier changes were important
preconditions.

### One-pop Watterson estimator

Native now uses the upstream sample-size convention in the one-pop Watterson
estimate.

Why it mattered:

- upstream initialization depends on that estimate
- wrong Watterson scaling pushes native into the wrong model scale early

### Monomorphic one-pop emission handling

Native no longer incorrectly splits the fully observed monomorphic `(0, 0)`
mass across the folded `(2, n)` mirror under polarization error.

Why it mattered:

- it corrected a real one-pop emission bug
- it removed an artificial emission-side scale distortion

### One-pop transition details

Several transition details were brought into line with upstream:

- exact diagonal completion with `1 - row_sum`
- upstream-style smoothing with `beta / (m + 1)`
- no row renormalization after smoothing

Why they mattered:

- larger-fixture scale mismatch was sensitive to transition details
- fixed-model transition agreement had to be exact before later debugging
  could be trusted

### Prefit / initialization parity

Native one-pop initialization was aligned more closely with upstream:

- prefit uses the upstream-style joint step (`single=False`)
- seeded randomization uses `RandomState`, not `default_rng`

Why it mattered:

- upstream and native started from the same basin
- remaining drift could then be attributed to real HMM/M-step mismatches

### Scale-step optimizer semantics

Native scale-step behavior was changed to match upstream’s actual
`ScaleOptimizer` behavior, including the aliased final assignment effect.

Why it mattered:

- native larger-fixture scale mismatch did not close until the scale-step
  semantics matched upstream more literally

## What did not close parity by itself

These were useful diagnostics or partial improvements, but they did not close
the last gap alone:

- tightening the scalar M-step tolerance
- unit-observation E-step rewrites
- hidden-state matching by itself
- exact transition matching by itself
- exact emission matching by itself

Those changes reduced the search space, but the remaining small-control gap
only disappeared after the preprocessing phase offset and compressed-row HMM
semantics were fixed.

## Practical lesson

For one-pop SMC++, end-to-end curve agreement was not a strong enough oracle.
The closure only became obvious after comparing:

- the exact preprocessed observation stream
- fixed-model `gamma0`
- fixed-model `xisum`
- fixed-model log-likelihood

That should be the first debugging path if tracked one-pop parity drifts again.

## Code pointers

- native implementation: `src/smckit/tl/_smcpp.py`
- upstream runner / oracle hooks: `src/smckit/tl/_smcpp_upstream_runner.py`
- vendored upstream source: `vendor/smcpp/`
- tracked parity gate: `tests/integration/test_smcpp_parity_matrix.py`
- quick metric report: `scripts/compare_smcpp_backends.py`
