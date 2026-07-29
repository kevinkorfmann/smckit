# Changelog

This project follows Keep a Changelog and Semantic Versioning from 1.0 onward.

## Unreleased

### Added

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

### Changed

- `implementation="auto"` now chooses native only for promoted capabilities.
- Apple-Silicon Pixi environments no longer request the unavailable D compiler.
- MSMC2 bootstrap now detects `dmd` or `ldc2` and Homebrew/Linux GSL layouts
  without modifying the pinned upstream source.
- Expensive diCal2 optimization cases are excluded from the routine unit tier.
