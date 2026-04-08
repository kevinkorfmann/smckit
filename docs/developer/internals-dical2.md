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
  within about `3.15e-2`
- the full independent meta-start search now reaches the same best-fit
  parameters as upstream
- the earlier large apparent `IM` gap turned out to be partly diagnostic noise:
  the old oracle comparison script was reporting the last stdout row instead of
  the best one, while the upstream bridge was already selecting the best fit

Those two results are the reason the current docs now describe `exp`
oracle-point, explicit-start, and full-search parameter parity as strong.

## What still does not match

The full independent searches now agree on the best-fit parameters, but the
native and upstream fixed-point likelihood values still differ slightly at those
same points:

- README `exp`
  - upstream log-likelihood: about `-15.87706545`
  - native log-likelihood at the same best-fit point: about `-15.87632006`
- README `IM`
  - upstream log-likelihood: about `-70.16830865`
  - native log-likelihood at the same best-fit point: about `-70.19983301`

Interpretation:

- the optimizer-path mismatch is no longer the blocker on the tracked README
  fixtures
- the remaining blocker is the native fixed-point likelihood calculation itself,
  which is now close enough that it should be debugged as a numerical/core-CSD
  issue rather than as a search-ceremony issue

## Next steps

- close the remaining fixed-point `exp` likelihood delta now that the search
  winner is aligned
- bring `IM` fixed-point likelihood parity down from `3.15e-2` to the same
  tighter standard
- only then tighten the integration gate back to true native/upstream
  interchangeability
