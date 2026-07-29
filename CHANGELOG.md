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

### Changed

- `implementation="auto"` now chooses native only for promoted capabilities.
- Apple-Silicon Pixi environments no longer request the unavailable D compiler.
- Expensive diCal2 optimization cases are excluded from the routine unit tier.
