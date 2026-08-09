# Changelog

This project follows Keep a Changelog and Semantic Versioning from 1.0 onward.

## Unreleased

### Added

- Pin the complete MIT-licensed PSMC+ source, expose its original inference and
  HMM-simulation entry points through the shell-free preservation runner, record
  NumPy 2 compatibility provenance, freeze a numeric upstream oracle, and add a
  hash-locked OCI/Apptainer runtime definition.
- Add a typed PSMC+ upstream adapter covering the complete inference option
  surface, normalized demographic and posterior/recombination results, physical
  scaling, persistent hashed artifacts, common result accessors, and live fit
  and decoding oracles while keeping the exact original path available.
- Add independently written native PSMC+ preprocessing and Numba HMM kernels:
  memory-efficient multihetsep masks and mutation/recombination maps, time
  discretization, local-rate transitions and emissions, forward/backward and EM
  evidence, posterior decoding, and marginal recombination.
- Complete public native PSMC+ fit/decode workflows with multi-file likelihoods,
  grouped free/fixed parameters, fixed or estimated recombination, optimizer and
  approximation controls, original-compatible artifacts, and an explicit
  local-mutation-corrected marginal result. Frozen homogeneous/mapped final
  oracles enforce parity, and checksum-locked Linux evidence records warmed-core
  speedups of 2.27x for fit and 1.25x for decode with bootstrap intervals above
  parity while retaining conservative upstream `auto` selection.
- Add a checksum-locked, clean-commit PSMC+ promotion matrix spanning twelve
  deterministic simulation workflows. Repair native compatibility for the
  upstream rate-map reduction order and its distinct final-time emission and
  transition grids; all matrix cases now pass strict fit/decode parity.
- Tighten the ASMC published-array oracle to the 0.1% scalar threshold and
  99.9% MAP agreement, record the observed 100% MAP agreement, and keep native
  sequence decoding unpromoted until its WGS oracle is closed.
- Add deterministic publication-protocol validation plus reusable trajectory,
  likelihood, posterior-coverage, bootstrap-speedup, and memory-promotion
  metrics.
- Add a three-panel, colorblind-safe MSMC-IM diagnostic with vector and
  publication-resolution export.
- Enforce MSMC-IM on an independent synthetic split family and persist hashed,
  original-compatible estimates from both native and upstream typed runs.

- Capability registry and capability-aware implementation selection.
- Versioned, JSON-serializable execution provenance.
- Unified command-line interface and exact raw upstream runner.
- Explicit unit, integration, oracle, benchmark, and publication test tiers.
- Multi-platform CI and scheduled upstream-oracle workflow.
- Immutable PSMC and MSMC2 source pins.
- Native PSMC consensus FASTA/FASTQ preprocessing with original quality,
  missingness, mutation-class, pseudoautosomal, and custom interval masks.
- Native PSMC posterior/full decoding, transition capping, divergence,
  sequence probabilities, HMM simulation, bootstrap replicates, and
  original-compatible result files.
- Selectable preserved PSMC helper entry points such as `fq2psmcfa` and
  `splitfa`.
- Native MSMC2 original-compatible final, loop, and log artifacts plus
  explicit mutation/lambda initialization.
- Native and upstream eSMC2 numeric artifact export.
- Validated PHLASH 1.0.6 PSMCFA, VCF/BCF, tree-sequence, and constructed-Contig
  workflows with deterministic JAX seeds, normalized posterior parameter
  summaries and credible intervals, hashed JSON/NPZ artifacts, and publication
  plotting.
- A provenance-recorded compatibility shim for the PHLASH 1.0.6 PSMCFA
  matrix-shape regression and a real-package integration smoke test.
- Native SMC++ VCF-to-SMC preparation with gzip, BED masks, long-gap
  missingness, distinguished-lineage selection, one/two-population headers,
  and an exact preserved-upstream conversion oracle.
- Typed preserved-upstream SMC++ split inference with two marginal model
  inputs, normalized population histories and split time, and hashed original
  plus normalized output artifacts.
- Parser-compatible diCal2 text output and versioned provenance JSON for both
  native and preserved-upstream runs.
- Native diCal2 multi-contig likelihoods with independent HMM resets, VCF
  coordinate offsets, zero-based half-open BED exclusions, VCF-header
  references, and lossless typed forwarding of those controls to the original
  Java CLI.
- Direct immutable-Java diCal2 VCF-reader oracles plus native support for
  haploid and allele-specific partial-missing genotypes, strict default
  unphased rejection, opt-in unphased-to-missing conversion, four-allele
  mapping, and duplicate-record repair with typed upstream provenance.
- Native diCal2 PAC permutation mixtures with Java-compatible generated or
  file-backed orders, selectable CSD counts, per-contig controls,
  posterior-weighted EM statistics, and generated log-grid/log-uniform
  multi-start search. The typed upstream bridge preserves the corresponding
  original CLI options, and live Java oracles freeze fixed-point and one-step
  PAC endpoints, file-backed per-contig permutations, and exact grid/random
  start sequences.
- Lossless one/two-population SMC++ I/O, contig-level regularization
  cross-validation, reloadable model initialization, upstream-readable model
  JSON, hashed result artifacts, and publication-ready demography/CV plots.
- Immutable SMC++ clean-split promotion evidence with matched persistent native
  and preserved-upstream timing, memory, environment, hardware, and checksums.

### Changed

- `implementation="auto"` now chooses native only for promoted capabilities.
- SMC++ two-population clean-split workflows are promoted for `auto` after the
  correctness matrix and frozen speed/memory gates passed; explicit upstream
  execution remains available.
- Match diCal2's Higham-Hall tolerance and Ethan-trunk infinity behavior,
  closing the tracked exponential-growth and isolation-migration fixed-point
  likelihood oracles to within `1e-8`.
- Forward typed diCal2 trunk, cake, ancient-deme-state, and additional-trunk
  controls to the preserved Java implementation instead of recording them
  without changing execution.
- Treat optional per-contig diCal2 path lists containing only `None` as absent
  in the typed Java bridge while still rejecting genuinely partial lists.
- Preserve parameterized instantaneous-migration matrices from diCal2 demo
  files, refine them into stochastic pulse epochs, and avoid false pulse
  intervals when demographic and HMM boundaries differ only by rounding.
- Apply diCal2 pulse migration consistently to lineage, recombination,
  mutation, marginal, and EigenCore transition surfaces. Independent simulated
  introgression now meets the pinned Java EigenCore fixed-point oracle, and
  exact ancient-recombination tensors use batched linear algebra with reusable
  core matrices. The shared optimizer contract now applies Java's implicit
  one-iteration M-step default to both implementations, fitted introgression
  has a strict one-step oracle, physical VCF blocks and objective values are
  cached, and constant-size zero-migration Ethan trunk epochs use their exact
  solution. A persistent worker records paired runtime and memory evidence.
- Frozen macOS ARM64 fitted-introgression evidence shows native diCal2 at
  `1.1381x` the preserved Java speed (paired-bootstrap 95% CI
  `1.1343-1.1433x`) with a `0.4909` peak-memory ratio across ten warmed
  repetitions. This is capability-specific evidence, not whole-method
  promotion.
- Add an independent fitted exponential-growth Java oracle and persistent
  benchmark worker. Finite constant-rate ODE epochs now use exact matrix
  exponentials while genuine growth retains high-accuracy integration, and
  explicit native diCal2 calls no longer probe the Java runtime. Frozen macOS
  ARM64 evidence records a `1.3052x` warmed speedup (95% CI
  `1.2835-1.3544x`), a `0.5959` peak-memory ratio, identical fitted
  parameters, and a maximum final-likelihood difference below `4e-12`.
- Add independent fitted clean-split, migration-window, and three-population
  Java oracles. Native diCal2 now retains transition-distance-specific EM
  statistics and matches the default Java objective's treatment of
  recombination back into the same lineage, closing all three one-step
  endpoints at `1e-12` parameter and `1e-5` likelihood tolerances.
- Add diCal2 transition-type-conditioned and marginal-KL objectives to the
  native and typed-upstream option surfaces. The transition-conditioned
  one-step fit has strict Java parity. Exact upstream marginal-KL preserves a
  pinned null-mutation-rate crash; native implements the intended source
  formula behind an explicit experimental warning and frozen endpoint.
- Close structured diCal2 PAC one-step parity across clean split, migration
  window, and three populations, plus fitted two-contig file-backed workflows.
  The native M-step now skips impossible zero-trunk states instead of producing
  `0 * NaN`. The pinned jar's separate null-trunk crash for the default all-CSD
  set is frozen while native's non-empty-CSD repair remains unpromoted.
- Add native diCal2 `oldCake`, `meanCake`, `multiCake`,
  `multiCakeUpdating`, `migratingMultiCake`, `migMultiCakeUpdating`, and
  `exactCake` trunks. All valid beginning/middle/end/average cake modes have
  fixed-point Java oracles, every new family has a fitted one-step oracle, and
  structured fixtures cover every non-exact family. The original `recursive`
  normal-CLI null-trunk failure is frozen and native fails explicitly.
- Implement diCal2 epoch/ancient-deme/present-deme hidden states and per-CSD
  mean-rate `addTrunkIntervals` refinement. Fixed-point and fitted Java oracles
  cover exponential and isolation-migration models, interval and ancient-deme
  state aggregation, and the combined controls. Moving demographic boundaries
  are now recomputed during each EM objective evaluation rather than frozen
  from the template model.
- Complete the normal diCal2 output catalogue. Native results now retain and
  persist every E-step row with original generation/step/particle identifiers
  and measured E-step milliseconds; preserved upstream output remains exact
  captured Java stdout. The original CLI creates no separate result files.
- Add a deterministic, GPL-separated repaired-source diCal2 marginal-KL oracle.
  Its one-method patch is applied only to a temporary, checksum-verified source
  copy; exact upstream still runs the immutable jar and preserves its crash.
  Native fitted endpoints match the repaired oracle for both LOL and
  posterior-weighted PAC.
- Apple-Silicon Pixi environments no longer request the unavailable D compiler.
- MSMC2 bootstrap now detects `dmd` or `ldc2` and Homebrew/Linux GSL layouts
  without modifying the pinned upstream source.
- Expensive diCal2 optimization cases are excluded from the routine unit tier.
