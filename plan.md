# smckit — Implementation Plan

## Vision

A unified, GPU-accelerated Python framework for SMC-based demographic inference.
What scanpy is to single-cell genomics, smckit is to coalescent-based population history reconstruction.

---

## Phase 0: Project Scaffolding

- [x] Write CLAUDE.md with project guidelines
- [x] Write plan.md (this file)
- [x] Clone PSMC into `vendor/psmc/`
- [x] Initialize Python package structure (`pyproject.toml`, `src/smckit/`)
- [x] Set up ruff, pytest
- [x] Define the core data container (`SmcData`)
- [x] Implement backend dispatch (`smckit.settings.backend`, auto-detection)
- [x] Stub out module namespaces: `pp`, `tl`, `pl`, `io`, `ext`

## Phase 1: PSMC — The Foundation

PSMC is the simplest and most widely used SMC method. It serves as the proving ground
for the entire architecture.

### 1a. Understand & Document PSMC Internals
- [x] Read and annotate the original C source (`vendor/psmc/`)
- [x] Document the HMM structure: states, transitions, emissions
- [x] Document the input format (psmcfa) and decoding pipeline
- [x] Map every C function to its mathematical operation
- [x] Write a technical reference doc: `docs/psmc_internals.md`

### 1b. NumPy Reference Implementation
- [x] `smckit.io.read_psmcfa()` — read PSMC input format
- [ ] `smckit.pp.psmcfa_from_vcf()` — generate input from VCF
- [x] `smckit.tl.psmc()` — full PSMC pipeline:
  - [x] Forward algorithm (NumPy)
  - [x] Backward algorithm (NumPy)
  - [x] Baum-Welch / EM parameter estimation (Nelder-Mead M-step)
  - [x] Viterbi decoding
  - [x] Time discretization and Ne(t) reconstruction
- [x] `smckit.pl.demographic_history()` — plot Ne(t) curve
- [x] Validation: compare output against original `psmc` on NA12878 chr22 (1000G)
      - Lambda correlation: 0.9999, theta diff: 0.18%, rho diff: 0.15%
- [x] Benchmarks: C PSMC ~6s vs smckit NumPy ~126s (chr22, 10 iterations)

### 1c. GPU Backends for PSMC
- [ ] CuPy forward/backward algorithms
- [ ] CuPy Baum-Welch
- [ ] CUDA kernel: forward algorithm (the bottleneck)
- [ ] CUDA kernel: transition matrix exponentiation
- [ ] Validate GPU outputs against NumPy reference (tolerance tests)
- [ ] Benchmark: GPU vs CPU speedup on realistic genome-scale data

### 1d. PSMC Polish
- [ ] Bootstrap support (`smckit.tl.psmc_bootstrap()`)
- [ ] Composite likelihood across chromosomes
- [ ] Parameter priors / regularization options
- [x] Unit tests (38 tests), integration tests (2 e2e tests), GPU tests pending

## Phase 2: MSMC / MSMC2

### 2a. Clone & Study
- [ ] Clone MSMC2 into `vendor/msmc2/`
- [ ] Clone MSMC into `vendor/msmc/` (for reference)
- [ ] Document the D source — state space, pair coalescent, cross-coalescent
- [ ] Understand the differences from PSMC: multiple haplotypes, pair-HMM structure

### 2b. MSMC2 NumPy Implementation
- [ ] `smckit.io.read_msmc_input()` — read MSMC input format
- [ ] `smckit.pp.msmc_input_from_vcf()` — VCF to MSMC input
- [ ] `smckit.tl.msmc2()` — full pipeline
  - [ ] Pair-HMM forward/backward
  - [ ] Cross-coalescence rate estimation
  - [ ] Population split time inference
- [ ] Validation against original MSMC2 binary

### 2c. MSMC2 GPU Backends
- [ ] CuPy pair-HMM
- [ ] CUDA kernels for multi-haplotype transition matrices
- [ ] Benchmarks

## Phase 3: SMC++

### 3a. Clone & Study
- [ ] Clone SMC++ into `vendor/smcpp/`
- [ ] Document the C++/Python architecture
- [ ] Understand the distinguished-lineage trick and variational inference

### 3b. SMC++ NumPy Implementation
- [ ] `smckit.io.read_smcpp_input()`
- [ ] `smckit.tl.smcpp()` — full pipeline
  - [ ] Conditioned SFS computation
  - [ ] Optimization (L-BFGS or similar)
  - [ ] Regularization / penalty terms
- [ ] Validation against original SMC++ binary

### 3c. SMC++ GPU Backends
- [ ] CuPy conditioned SFS
- [ ] CUDA optimization kernels
- [ ] Benchmarks

## Phase 4: Unified API & Cross-Method Features

- [ ] Unified `SmcData` container works seamlessly across all methods
- [ ] `smckit.tl.compare()` — run multiple methods, compare Ne(t) trajectories
- [ ] `smckit.pl.comparison_plot()` — multi-method overlay plot
- [ ] `smckit.io.to_tskit()` / `smckit.io.from_tskit()` — tree sequence interop
- [ ] `smckit.pp.simulate()` — wrappers for msprime/stdpopsim for testing
- [ ] Common parameter objects (mutation rate, recombination rate, generation time)
- [ ] CLI: `smckit run psmc input.psmcfa -o results/`

## Phase 5: SSM Extension Framework

- [ ] `smckit.ext.ssm.SmcStateSpace` — abstract base class
  - [ ] `.transition_matrix(params)` → T
  - [ ] `.emission_matrix(params)` → E
  - [ ] `.initial_distribution(params)` → π
- [ ] Implement PSMC as `PsmcStateSpace(SmcStateSpace)`
- [ ] Implement MSMC2 as `Msmc2StateSpace(SmcStateSpace)`
- [ ] Allow custom state-space definitions for new models
- [ ] Integration point for external SSM libraries (dynamax, ssm, etc.)
- [ ] Experimental: continuous-time extensions, neural emission models

## Phase 6: Community & Polish

- [ ] Documentation site (Sphinx + Read the Docs, or mkdocs)
- [ ] Tutorial notebooks (like scanpy tutorials)
  - [ ] Quick start: run PSMC on example data
  - [ ] Comparing methods on simulated data
  - [ ] Writing a custom SSM model
  - [ ] GPU acceleration guide
- [ ] Contributing guide
- [ ] Packaging on PyPI and conda-forge
- [ ] Preprint / JOSS paper

---

## Design Decisions Log

| Date       | Decision | Rationale |
|------------|----------|-----------|
| 2026-04-06 | Scanpy-style API (`pp`, `tl`, `pl`, `io`) | Familiar to bio-Python users, proven ergonomics |
| 2026-04-06 | Three-tier backend: CUDA > CuPy > NumPy | Maximum performance with guaranteed CPU fallback |
| 2026-04-06 | Vendor upstream repos read-only in `vendor/` | Preserve reference, track upstream changes via git |
| 2026-04-06 | PSMC first | Simplest algorithm, establishes all architectural patterns |
| 2026-04-06 | SSM as extension, not core | Keep core simple; SSM abstraction is for power users & researchers |

---

## Current Focus

**Phase 0 + Phase 1a** — Scaffold the project and deeply understand PSMC internals
before writing any algorithm code.

Next immediate steps:
1. Clone PSMC into vendor/
2. Set up package skeleton
3. Read and annotate the PSMC C source
