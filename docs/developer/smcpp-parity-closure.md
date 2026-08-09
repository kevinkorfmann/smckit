# SMC++ Parity Closure Notes

This note records what closed the tracked native-vs-upstream SMC++ gaps, why
those changes mattered, and which earlier changes were necessary but not
sufficient.

## Scope

The closure described here is for the tracked one-pop SMC++ contract used by
smckit:

- one population
- `n_distinguished=2`
- upstream-style preprocessing
- the strict small control fixture
- the bundled larger `tests/data/smcpp_onepop_larger.smc` fixture

The tracked clean-split contract additionally covers:

- two populations with distinguished allocations `(2, 0)`, `(1, 1)`, and the
  public reversed-order form `(0, 2)`
- a deterministic expected joint SFS and resulting joint-CSFS emissions
- fully observed, downsampled, missing-distinguished, and reduced observations
- the shared marginal-history scale coordinate followed by split-time fitting
- Piecewise, CubicSpline, PChipSpline, AkimaSpline, and BSpline histories

These are enforced fixtures, not a blanket claim about every possible future
SMC++ input family.

### Production distinguished-lineage contract

Upstream one-population SMC++ requires exactly two distinguished lineages. An
older smckit-only one-distinguished surrogate was never part of the upstream
capability surface and produced a materially biased constant-size simulation
trajectory. The production `smcpp()` API now validates top-level and per-record
metadata and rejects that surrogate for `native`, `upstream`, and `auto` before
runtime selection. Its low-level kernels remain available for research and
historical diagnosis but are classified as obsolete in the feature ledger.

The replacement simulation gate uses two distinguished and ten
undistinguished haplotypes from a 10 Mb, 12-haplotype msprime simulation. It
checks the upstream scale convention, optimizer success and finite likelihood,
the result scaling identity, and that every inferred interval lies within tenfold
of the constant-size truth. The three assertions share one fit and passed in the
controlled one-thread Sesame environment in 998.14 seconds.

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

For the tracked clean-split fit, native and preserved upstream return the same
split (`5.526022037850897e-06`) and shared log scale
(`-0.9999959801362373`). Native raw joint-CSFS entries agree with the
preserved estimator to about `1.5e-3` or better; the residual is expected
because upstream averages Monte-Carlo histories while native evaluates the
expectation deterministically.

The expanded fixed-stat oracle also compares the emission probability actually
used by the upstream inference manager for `(2, 0)` and `(1, 1)` across full,
downsampled, missing-distinguished, reduced-monomorphic, and
reduced-heterozygous observations. The focused native suite and the complete
split-oracle file passed on Linux x86-64 in the controlled Sesame environment.
This method-specific closure was followed by the full 159-test SMC++ matrix,
the repository unit tier, build/install checks, and the frozen performance
gate described below.

### Performance promotion evidence

Protocol `sha256:c339dbb68e7ec26c721d909916edea5e388d77a60f03c04847e9daaa5cf560dd`
measured five warmed calls per implementation on one Linux x86-64 CPU thread.
Native took 0.245-0.249 seconds per warmed call and preserved upstream took
2.356-2.745 seconds. The aggregate speedup was 11.01x with a 95% bootstrap
confidence interval of 9.47-11.11x. Native peak RSS was 198,643,712 bytes,
0.512x the upstream peak of 387,686,400 bytes. Both the speed and memory gates
passed, so the tracked clean-split capability is promoted for `auto`.

The raw records, frozen protocol, environment, hardware description, aggregate,
and checksums are retained under
`workflow/publication/evidence/smcpp-split/sha256-c339dbb6/`. This result applies
to the frozen control workload and hardware; it is not a universal performance
claim for every chromosome or machine.

## Two-population clean-split closure

Upstream split inference uses hidden states `[0, infinity]`. Under that
contract, the joint conditioned SFS reduces to the expected two-population
joint SFS, followed by hypergeometric allocation to the two distinguished
lineages. Native computes that expectation directly from lineage-count
transition matrices on the two marginal histories and their common ancestral
history.

The implementation then:

1. converts each serialized marginal spline to the same 100 right-endpoint
   stepwise history used upstream;
2. applies one shared bounded log-scale coordinate to both marginal histories;
3. fits the clean split with a bounded scalar search; and
4. returns the reloadable joint model, normalized histories, split time,
   likelihood, and complete provenance.

The preserved SMC++ BSpline implementation performs an unsafe unequal-shape
NumPy comparison. A narrowly scoped upstream-runner compatibility shim retains
the original alignment algorithm on modern NumPy without changing vendored
source. Native spline evaluation is independent and has an oracle test for
every supported spline class.

### Missing, downsampled, and reduced observations

The upstream two-population emission path does not use a simple allele-count
lookup when observations omit lineages or distinguish fewer alleles than the
model. Native now independently reproduces its probability semantics in this
order:

1. expand compatible states for missing distinguished lineages;
2. condition compatible full-sample states with exact hypergeometric weights;
3. recode the globally fixed-derived state to the ancestral monomorphic state;
4. apply the polarization transform;
5. remove the final fully derived representation; and
6. normalize the remaining observation weights.

Changing that order produces a measurable per-site likelihood error. Applying
the full mapping closes the earlier missing/downsampled gap and is checked
against the upstream manager's final emission probabilities rather than a
separate Monte-Carlo joint-CSFS draw.

Reduced observations contain only the two distinguished lineages. Their
emission is computed from expected pairwise coalescence time: same-population
lineages follow the relevant marginal history, whereas separated lineages
cannot coalesce before the split and then follow the common ancestral history.
Fully missing reduced observations have probability one.

### Population-order canonicalization

Upstream data loading canonicalizes `(0, 2)` to `(2, 0)`, although direct
joint-CSFS construction does not accept `(0, 2)`. Native uses the same internal
canonical form for the numerical problem while swapping observations, sample
sizes, and marginal models together. It then maps models, populations, and
result metadata back to the caller's order. A full native split regression test
checks that reversed input order preserves the public result meaning.

### Transition, structural-floor, and optimizer details

Split likelihoods include upstream's one-state transition factor and `1e-5`
transition smoothing. Structural emission zeros use the upstream `1e-10`
floor after monomorphic mass is assigned; the tensor is not renormalized after
that floor, so its total can exceed one by a microscopic amount.

Upstream scale and split optimizer plugins are stored in a `weakref.WeakSet`,
so their coordinate order is not guaranteed. Preserved execution retains that
behavior. The oracle runner can request an explicit order for diagnosis, but
the ordinary upstream path must not silently substitute it. Native uses a
documented deterministic scale-then-split order.

The inference manager uses a `K=10` Monte-Carlo joint-CSFS realization. A
later direct diagnostic call consumes another realization, so those two raw
tensors cannot be treated as the same oracle value. Exact native mathematics
is compared with a high-`K` upstream diagnostic where raw joint-CSFS accuracy
is the question, and with the manager's own final emission probabilities where
runtime observation mapping is the question.

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
- clean-split parity gate: `tests/integration/test_smcpp_split_validation.py`
- quick metric report: `scripts/compare_smcpp_backends.py`
