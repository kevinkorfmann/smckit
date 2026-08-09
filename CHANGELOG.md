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
  and decoding oracles while keeping native execution explicitly unavailable.
- Add independently written native PSMC+ preprocessing and Numba HMM kernels:
  memory-efficient multihetsep masks and mutation/recombination maps, time
  discretization, local-rate transitions and emissions, forward/backward and EM
  evidence, posterior decoding, and marginal recombination. Both layers are
  locked to frozen content-addressed upstream oracles while public native
  execution remains gated on full workflow parity.
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
- Apple-Silicon Pixi environments no longer request the unavailable D compiler.
- MSMC2 bootstrap now detects `dmd` or `ldc2` and Homebrew/Linux GSL layouts
  without modifying the pinned upstream source.
- Expensive diCal2 optimization cases are excluded from the routine unit tier.
