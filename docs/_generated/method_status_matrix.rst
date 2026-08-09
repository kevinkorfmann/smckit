.. list-table::
   :header-rows: 1

   * - Method
     - Upstream
     - Native
     - Native default eligible
     - Tracked agreement
     - Notes
   * - PSMC
     - ✓
     - ✓
     - ✓
     - `0.9999223 lambda corr`
     - Preserved upstream execution and the native port cover missing and multi-record inputs, explicit intervals and parameters, divergence, bootstrap, decoding, transition capping, and original-compatible result files.
   * - ASMC
     - ✓
     - ✓
     - ✓
     - `array: 100% MAP and 2.99e-4 max mean error; dense sequence: >=99.9% MAP and <=1e-3 max mean error`
     - Native parity covers array and dense sequence decoding, folded or ancestral coding, CSFS spacing/compression, explicit pairs and interval burn-in, job partitioning, complete posterior summaries, and original-compatible artifacts.
   * - MSMC2
     - ✓
     - ✓
     - ✓
     - `>= 0.999999865 lambda corr`
     - Native parity covers pair selection, ambiguous and missing data, multiple chromosomes/files, fixed recombination, Li-Durbin and quantile boundaries, explicit initialization, and original-compatible artifacts.
   * - MSMC-IM
     - ✓
     - ✓
     - ✓
     - `strict payload match on four Yoruba/French controls and an independent synthetic split family`
     - The native fitter and preserved upstream runner share the corrected/raw population-size, migration, split-quantile, chi-square, and estimates-artifact contract across two enforced oracle families.
   * - eSMC2
     - ✓
     - ✓
     - ✓
     - `<= 0.3605% max tracked Xi rel err`
     - Native and upstream are interchangeable on the tracked public input and model matrix; native output preserves the original numeric tables, while raw upstream R execution exposes the complete exported helper package.
   * - SMC++
     - ✓
     - ✓
     - ✓
     - `>= 0.999125 one-pop log-Ne corr; split dloglik 1.30e-05 and identical fitted split`
     - The native one-pop path clears the tracked inference matrix and includes upstream-matched VCF preparation, masks/compression, cross-validation, reloadable model artifacts, and plotting. Two-population clean-split inference is promoted across all distinguished-lineage allocations, missing/downsampled/reduced observations, and five spline classes. Frozen one-thread evidence measured an 11.01x warmed speedup (95% bootstrap CI 9.47-11.11x) and 0.512x peak memory versus preserved upstream.
   * - diCal2
     - ✓
     - ✓
     - ✗
     - `README searches match; default and transition-conditioned fitted endpoints match within 1e-12 parameters and 1e-5 log likelihood.`
     - Public upstream Java bridge parses and persists exact EM-path stdout. Native persistence records every original-grammar E-step row across particles plus normalized JSON. Native README searches and independent clean-split, migration-window, three-population, pulse-introgression, exponential-growth, and structured PAC one-step fits reproduce the pinned Java endpoints. All runnable original trunk families, valid cake modes, epoch-ancient-present hidden states, and per-CSD additional trunk intervals have native fixed-point and fitted coverage; the pinned recursive CLI null-trunk failure is preserved. Transition-distance statistics and both Java conditioning modes are preserved. Native marginal-KL matches a deterministic GPL repaired-source oracle for LOL and PAC while exact upstream preserves the immutable jar's crash. The repaired all-nonempty-CSD PAC default remains unpromoted pending empirical validation.
   * - PHLASH
     - ✓
     - n/a
     - ✗
     - `real-package PSMCFA smoke plus normalized posterior contract`
     - Reproducible adapter for the maintained PHLASH 1.0.6 Python package with PSMCFA, VCF/BCF, tree-sequence, posterior, credible-interval, artifact, and plotting support; no independent rewrite is planned for 1.0.
   * - PSMC+
     - ✓
     - ✓
     - ✗
     - `all 12 deterministic capability cases pass strict native/upstream parity; warmed native core is 2.27x faster for fit and 1.25x for decode`
     - Pinned MIT-licensed PSMC+ source with exact inference and HMM-simulation entry points plus a typed upstream adapter covering all scientifically meaningful inference controls, normalized fit/decode results, persistent hashed artifacts, and versioned provenance. The independent native engine covers multi-file fitting, local mutation/recombination maps, grouped and fixed parameters, fixed or estimated recombination, approximation controls, decoding, parallel likelihoods, and original-compatible artifacts. Frozen constant and mapped oracles enforce final-output parity. A clean-commit, one-thread Linux x86-64 matrix now passes all 12 deterministic simulation cases, including every approximation control independently and in combination; missing, multi-file, local-rate, estimated-rho, fit, and decode workflows are represented. A separate five-repetition benchmark found warmed-core speedups of 2.27x for fit (95% bootstrap CI 2.22-2.32) and 1.25x for decode (1.21-1.27), with end-to-end peak memory below upstream. Auto remains conservative until human/nonhuman empirical and macOS ARM64 validation close. The NumPy 2 compatibility shim does not modify upstream code.
   * - SSM
     - n/a
     - ✓
     - n/a
     - `—`
     - Novel in-repo extension framework rather than an upstream compatibility target.
