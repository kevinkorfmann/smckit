# diCal2 Native Parity Notes

This note records the concrete changes that moved the native `dical2` path
closer to the vendored `diCal2.jar`, why those changes mattered, and what still
does not match.

## Why these changes were necessary

The hard lesson from the README fixtures was that "curve looks similar" was not
good enough. The native path needed to be compared to upstream at three
different levels:

- same public result payload
- same likelihood at the same parameter vector
- same optimizer endpoint from the same starting point

The fixes below were made to close those gaps in that order.

## What changed

### 1. Shared upstream/native result normalization

The upstream runner now normalizes into the same public fields as native,
including `best_params`, `ordered_params`, `time`, `time_years`, and the
demographic arrays.

Why: without a shared result shape, it was too easy to compare different
quantities and overestimate parity.

### 2. VCF reader now preserves upstream-style physical-locus metadata

`read_dical2_vcf()` was changed to keep:

- segregating-site-compressed haplotypes
- `seg_positions`
- `reference_length`
- `reference_alleles`

Why: upstream grouped-locus runs do not operate on the compressed segregating
matrix alone. They still need the physical reference layout so non-segregating
sites and block boundaries are counted correctly.

### 3. Grouped-locus counting now uses physical blocks, not just segregating columns

The native multilocus handler was changed to build allele-pair counts over the
full physical sequence length using the VCF/reference metadata above.

Why: the earlier native path grouped only the compressed segregating-site
matrix. That changed the effective observation model and was one reason the
native fixed-parameter likelihood did not line up with `diCal2.jar`.

### 4. Refined intervals now materialize interval-specific epochs

The native core and trunk builders no longer reuse one coarse growth-bearing
epoch across all refined HMM intervals. They now materialize a refined epoch per
interval, with the correct interval-specific end sizes.

Why: this was the biggest single hidden-state fix. Reusing the coarse epoch
distorted the time-varying state model inside growth-bearing intervals. After
this change, the tracked `exp` hidden-state marginal error dropped from about
`7.39e-4` to about `3.02e-8`, and the fixed-parameter `exp` likelihood gap
dropped to about `7.45e-4`.

### 5. Meta-start generation was tightened toward upstream semantics

The native meta-search now follows upstream more closely by using:

- Java-style RNG handling
- per-candidate spawned RNG streams
- an upstream-style cumulative initial simplex
- upstream-like zero-SD handling for new-point proposals
- resampling against demo validity instead of blindly clipping every proposal

Why: once the fixed-parameter core was closer, the remaining visible mismatch
came from the search ceremony. Explicit-start replay only became meaningful once
the native meta-start behavior consumed randomness and generated candidates in a
more upstream-like way.

### 6. Gallery rerun used a targeted diCal2-only render path

The full `scripts/generate_dual_gallery.py` run was blocked locally because it
imports `generate_method_gallery.py`, which hard-imports `msprime`, and that
dependency is not installed in the current environment.

Why: the user still asked for the gallery to reflect the current diCal2 state.
The practical solution was to render only the diCal2 native/upstream panels
directly from the current README `exp` run and then rebuild the Sphinx HTML so
the landing page and gallery page copied those refreshed images.

### 7. Meta-start replay tracing is now available on the native path

The native implementation now accepts `record_meta_trace=True` in
`native_options` and stores a generation-by-generation replay under
`results["dical2"]["meta_trace"]`.

Why: the remaining mismatch could no longer be understood from the final plot or
even from the final best-fit parameters alone. We needed to see:

- the starting points used in each meta generation
- the seeded EM result from each start point
- the retained parents
- the effective marginal SD
- the concrete offspring proposals for the next generation

### 8. Java `nextLong()` and coordinatewise shuffle semantics were still off

The native `_JavaRandom` implementation initially combined the second 32-bit
piece of `nextLong()` as an unsigned integer. Java sign-extends that lower
`int`, so the third spawned offspring seed on the tracked README `exp` run was
wrong by exactly `2^32`.

That mattered because the wrong offspring seed changed the initial simplex sign
pattern in the seeded M-step replay for the `exp` outlier path. After fixing
`nextLong()`, the full native README `exp` meta-start run began reaching the
same best-fit parameter vector as upstream.

The README `IM` fixture still differed after that because it uses the default
coordinatewise M-step. Upstream shuffles the coordinate order with
`Collections.shuffle(..., Random)`, which consumes `nextInt(i)` draws. The
native permutation code had been using `int(random() * i)`, which is not the
same RNG path. After switching to Java-style bounded `nextInt()` for the
coordinatewise shuffle, the full native README `IM` run also reached the same
best-fit parameter vector as upstream.

### 9. VCF-backed native configs needed to be compacted after filtering

The VCF reader already filtered the returned sequence matrix down to the
haplotypes selected by the `.config` file. However, the stored native
`DiCal2Config` still kept the original unfiltered haplotype row indexing.

That was harmless on the packaged README fixtures because their included rows
happened to be a prefix of the config, but it broke reduced pairwise debug
configs immediately. The fix was to compact the native `DiCal2Config` after VCF
filtering so `config.haplotype_populations`, `haplotype_multiplicities`, and
`sample_sizes` line up with the filtered sequence rows.

Why this mattered: once the config and sequence matrix shared the same compact
indexing, reduced two-haplotype and three-haplotype native/upstream probes
became valid tools for localizing the remaining `IM` evaluator mismatch.

### 10. The remaining `IM` mismatch is concentrated in cross-pop PCL cases

After the VCF config compaction fix, reduced README `IM` probes could be run on
small temporary configs while still using the same vendored VCF/demo/param
bundle.

What those probes showed:

- two-haplotype cross-pop configs such as `(0, 2)` and `(0, 3)` now run cleanly
  and show native-minus-upstream fixed-point log-likelihood deltas of about
  `-1.07e-2`
- three-haplotype mixed-pop configs such as `(0, 1, 2)` and `(0, 2, 3)` show
  deltas around `-2.1e-2`
- same-pop tiny configs on this VCF often collapse to no segregating sites, so
  they do not provide a useful upstream oracle on this bundle

Interpretation:

- the remaining README `IM` evaluator gap is not a general optimizer problem
- it is not primarily caused by the old config-index mismatch
- it is concentrated in the cross-pop PCL evaluator path
- varying the migration parameter alone on the two-haplotype cross-pop probe
  does not materially change the delta, so the remaining bug is more likely a
  fixed normalization/emission/state-space issue than a simple “wrong migration
  rate” bug

## What improved

On the README `exp` fixture:

- fixed-point likelihood at the upstream best-fit parameters is now within about
  `7.45e-4`
- replaying each explicit `exp.rand` start point now lands on the upstream
  endpoint to displayed precision, with log-likelihood deltas at or below about
  `2.21e-4`
- the full independent meta-start search now reaches the same best-fit
  parameters as upstream

On the README `IM` fixture:

- the native fixed-point likelihood at the upstream best-fit parameters is now
  within about `1.63e-3`
- the full independent meta-start search now reaches the same best-fit
  parameters as upstream
- the earlier large apparent `IM` gap turned out to be partly diagnostic noise:
  the old oracle comparison script was reporting the last stdout row instead of
  the best one, while the upstream bridge was already selecting the best fit
- reduced VCF-backed `IM` probes now work even when the included haplotypes are
  not a prefix of the original config rows

Those two results are the reason the current docs now describe `exp`
oracle-point, explicit-start, and full-search parameter parity as strong.

### 11. Structured no-growth native runs needed `ODECore`, not `EigenCore`

The reduced two-haplotype cross-pop README `IM` probes eventually isolated the
remaining evaluator gap to the emission side of the no-growth structured core.

The decisive checks were:

- the reduced cross-pop fixed-point delta stayed at about `-1.07e-2` even when
  `rho = 0`, which ruled recombination transitions out as the main blocker
- under `rho = 0`, that same delta grew rapidly with `theta`, which pointed to
  the emission path rather than the initial-state marginal
- swapping the reduced and full README `IM` fixed-point evaluations from the
  no-growth structured `EigenCore` to `ODECore` collapsed the gap by more than
  an order of magnitude

Why this mattered: the issue was not in grouped-locus pair counting itself. It
was in the exact no-growth structured emission path used by `EigenCore`.
Routing structured multi-deme native runs through `ODECore` preserved the
already-correct search path while bringing the README `IM` fixed-point value
much closer to the vendored oracle.

### 12. The ODE relative tolerance must match the Java oracle

The native solver used a relative tolerance of `1e-13`, while the pinned Java
implementation configures its Higham-Hall solver at `1e-14`. SciPy clamps
relative tolerances below 100 machine eps, so native now uses that explicit
floating-point floor: the closest representable setting.

This change reduced the README `IM` fixed-point native-minus-upstream
log-likelihood difference from about `1.63e-3` to `-2.88e-9`. The strict
structured IM oracle is therefore enforced at `1e-8`.

### 13. Preserve the Ethan trunk solver's asymmetric infinity check

The last exponential-growth difference was not in growth itself. It came from
an upstream numerical quirk in the `migratingEthan` trunk approximation. After
the first five-unit integration segment of an infinite interval, Java continues
only if at least one derivative is positive. A one-population Ethan trunk has
only negative derivatives, so upstream intentionally stops there rather than
integrating to equilibrium.

Reproducing that asymmetric first check reduced the README exponential fixture
difference from about `7.45e-4` to `5.38e-11`. A control run with the `simple`
trunk agrees to about `6.1e-13`, confirming that the core, leave-one-out
expansion, grouped emissions, and transition calculation were already aligned.

## Current closure

The tracked README exponential-growth and isolation-migration workflows now:

- reach the same best-fit parameter vectors in independent searches
- agree on fixed-point likelihood within `1e-8`
- reproduce upstream grouped-locus, LOL/PCL, ODE, migration, and growth
  semantics

Native promotion still requires independent structured/growth simulations,
remaining feature-ledger closure, original-compatible output artifacts, and the
performance gate.

### 14. PAC permutations and generated starts now follow the pinned Java control flow

The native PAC evaluator now treats each contig as a mixture over permutations.
For each permutation it visits haplotypes in reverse order, evaluates the
selected CSD trunk sizes, sums CSD log likelihoods, and combines permutation
likelihoods with `logsumexp`. Expected-count arrays used by the M-step are
weighted by the corresponding posterior permutation probability.

Generated permutations reuse one mutable ordering across consecutive
Java-compatible `Collections.shuffle` operations. File-backed permutations,
reduced CSD counts, and independent per-contig permutations are supported and
recorded with hashes and the resolved orders in result provenance. The typed
upstream bridge forwards the matching original CLI options.

Generated meta starts now cover both the upstream log-spaced grid and seeded
log-uniform random modes. Grid enumeration preserves the upstream odometer
order in which the first parameter changes fastest. The complete initial
candidate file is evaluated before `metaNumPoints` controls later generations.

The frozen live Java checks cover a two-permutation, reduced-CSD PAC
fixed-point likelihood and all 32 points of a five-parameter generated grid.
Both agree within `1e-8`. These features remain unpromoted until one-step EM,
file/per-contig permutations, random-start sequences, and broader independent
fixtures are covered. This slice closes the known implementation gap without
claiming full native equivalence.

### 15. Posterior-weighted PAC EM and both generated-start modes have live oracles

The next live Java checks moved beyond fixed-point likelihoods. With two
generated permutations and CSD trunk sizes 1 and 7, one native EM step reaches
the same five-parameter endpoint as the pinned jar: every parameter differs by
less than `2e-13`, and the final log-likelihood differs by about `2.23e-11`.
This directly validates that permutation posterior weights are applied to the
expected counts used by the M-step, not only to the reported likelihood.

A separate two-contig check supplies two permutation files with two distinct
orders each. Native and Java agree within `1e-8`, and native provenance records
both resolved order sets and both file hashes. This test also exposed and fixed
a typed-upstream bridge bug: optional per-contig paths represented as
`[None, None]` now mean the option is absent, while a genuinely partial list
still fails clearly.

Finally, the seeded random-start oracle freezes all three Java-generated points
for a five-parameter model as well as the selected winner. Together with the
32-point grid oracle, both generated initialization strategies now have exact
sequence evidence.

These checks close the originally identified PAC/search semantics gap, but do
not promote native diCal2. Structured PAC models, invalid-candidate resampling,
the pinned jar's empty-trunk failure, complete artifacts, and performance still
require independent evidence.

### 16. Independent structured fixtures exposed boundary and pulse semantics

Four deterministic msprime fixtures now exercise the pinned Java tool and the
native implementation with independently generated VCFs. Clean split,
migration window, and three-population fixed points pass: their largest total
log-likelihood difference is about `1.37e-6`, or less than `3e-10` per simulated
base. These checks use Java's ODE core and native `ODECore` on the same typed
inputs.

The clean-split fixture initially failed before likelihood evaluation because a
demographic boundary at `0.2` and a log-uniform HMM boundary a few ulps away
were treated as a zero-duration pulse. Refinement now merges numerically
coincident boundaries while retaining duplicates only for real pulse events.

The introgression fixture found a separate preservation defect: the demo reader
discarded the instantaneous-migration matrix attached to the start of a normal
epoch, including its `?1` placeholder. Native parsing now retains that matrix
and parameter identity, constructs the zero-duration pulse used by Java, and
fills its diagonal as one minus the outgoing pulse probability. Unit tests
freeze the resulting `0.03` migration and `0.97` stay probabilities.

The pinned tool requires `--useEigenCore` for pulse migration. Applying the
pulse transition to the native extended lineage matrix, no-recombination and
recombination distributions, mutation-event distribution, and marginal
lineage transition closes the independent introgression fixed point. The
native EigenCore endpoint differs from Java by about `4.32e-6` total
log-likelihood, inside the frozen `1e-5` tolerance and well inside the `1e-6`
per-base criterion.

The exact ancient-recombination calculation now builds interval tensors with
batched contractions and reuses equivalent core matrices across CSDs. This
removes the worst scalar Python nesting, but a one-step fitted introgression
diagnostic still exceeded two minutes versus roughly 1.3 seconds for Java.
Accordingly, fitted introgression and the performance gate remain explicit
promotion blockers even though fixed-point correctness is now enforced.
