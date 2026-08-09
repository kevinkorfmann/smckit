# Parity and Oracle Status

This page expands on the compact landing-page status matrix. It tracks native
vs upstream agreement for each method, the decisions that guided the current
validation strategy, the context for why the method matters in the framework,
and the remaining work before declaring the oracle complete.

## PSMC

**Progress:** The NA12878\_chr22 C-reference run shows `lambda` correlation of `0.9999223`, `lambda` max relative error `9.54e-03`, `theta` rel error `1.83e-03`, and `rho` rel error `1.50e-03`.
**Decisions:** Use the vendored binary as the archetype, keep the core HMM in NumPy/Numba, and treat PSMC as the proving ground for the dispatcher.
**Context:** PSMC supplies the baseline `SmcData` serialization, plotting helpers, and is the simplest time-discretized state-space that all other methods extend.
**Remaining tasks:** Finish GPU backends (`Numba` → `CuPy` → `CUDA`), add bootstrap/composite support, and include multi-chromosome pipelines.
**References:** :doc:`methods/psmc`, ``tests/integration/test_psmc_validation.py``, ``tests/integration/test_psmc_e2e.py``

## PSMC+

**Progress:** The MIT-licensed source is pinned at commit `032168f2ceed3c0e46b7f214f890faf83dff41ae`, both original Python entry points are exposed through the shell-free raw runner, and typed upstream fit/decode workflows normalize results and provenance. The independent native engine now covers multi-file fitting, mutation/recombination maps and downsampling, grouped free/fixed parameters, fixed or estimated recombination, approximation and optimizer controls, parallel likelihood evaluation, posterior decoding, original-compatible artifacts, and a corrected local-mutation-aware marginal-recombination result. Frozen homogeneous and mapped oracles enforce preprocessing, intermediate, final-fit, posterior, likelihood, and original-output agreement at floating-point precision. On one-thread Linux x86-64, five warmed-core repetitions give a 2.27x fit speedup (95% bootstrap CI 2.22--2.32) and 1.25x decode speedup (1.21--1.27); isolated end-to-end measurements put native peak memory at no more than 0.395x upstream.
**Decisions:** Preserve every original argument and artifact before promoting native execution. Build the native method from published SMC and locus-rescaling equations, use the pinned source only to freeze external numeric oracles, expose public `native` fit/decode now, and keep `auto` on upstream until the broader empirical matrix closes. Treat the original simulator as a permanently preserved raw-CLI capability rather than claiming an independent rewrite.
**Context:** PSMC+ extends pairwise inference to local mutation, recombination, and coalescence-rate heterogeneity, which directly exercises smckit's preservation-first handling of maps, multi-file data, decoding, and original artifacts.
**Remaining tasks:** Expand the empirical and simulation matrix, test the pinned OCI/Apptainer runtime, validate additional optimizer and approximation combinations, and rerun the frozen benchmark on macOS ARM64 before changing the `auto` default.
**References:** :doc:`methods/psmcplus`, ``tests/unit/test_psmcplus_native_preprocessing.py``, ``tests/unit/test_psmcplus_native_kernels.py``, ``tests/unit/test_psmcplus_native_fit.py``, ``tests/integration/test_psmcplus_upstream_validation.py``, ``benchmarks/psmcplus/``

## ASMC

**Progress:** The vendored `n300` array fixture clears the strict gate: MAP indices agree at every tested site, median posterior-mean relative error is `9.12e-5`, and the maximum relative error is `2.99e-4`. A separate dense `n300` whole-genome sequence interval with CSFS emissions and 0.5 cM burn-in also clears the `1e-3` maximum posterior-mean error and 99.9% MAP-agreement gates.
**Decisions:** Canonical decoding quantities (`.decodingQuantities.gz`) anchor the comparison, and we replay the same haplotypes/genetic positions as the reference. Sequence mode preserves ASMC 1.4's batched intermediate-buffer semantics because those values affect the public posterior output.
**Context:** ASMC extends PSMC by decoding per-pair coalescence times; it depends on optimized transition table decomposition and dense decoding quantization.
**Remaining tasks:** Keep both array and dense-sequence oracles in the scheduled upstream matrix and add additional empirical sequence panels to the publication benchmark.
**References:** :doc:`methods/asmc`, ``tests/integration/test_asmc_validation.py``, ``tests/integration/test_asmc_sequence_oracle.py``

## MSMC2

**Progress:** Six fixed upstream fixtures show left boundary relative error `<=4.15e-06`, lambda relative error `<=2.45e-03`, lambda correlation `>=0.999999865`, and log-likelihood delta `<=4.75e-03`.
**Decisions:** Reuse the same `vendor/msmc2` fixtures for the integration test, and build `smckit.tl.msmc2` to match the D implementation before layering GPU/async optimizations.
**Context:** MSMC2 is the multi-haplotype extension that fuels MSMC-IM, SMC++, and eSMC2 fixtures; the multihetsep input parsers and combined output readers all trace back to this parity gate.
**Remaining tasks:** Track original MSMC parity, stress-test cross-count heatmaps (stdpopsim `z`), and broaden multi-population validations.
**References:** :doc:`methods/msmc2`, ``tests/integration/test_msmc_validation.py``

## MSMC-IM

**Progress:** The vendored Yoruba/French ceremony is enforced as a four-case upstream-backed oracle matrix, and an independent synthetic split-migration family now broadens the input surface. Native and upstream match the public payload fields for corrected/raw `N`, thresholded `m`, cumulative migration, split quantiles, chi-square diagnostics, and persisted estimates.
**Decisions:** The official `MSMC_IM.py` run remains the ceremony, the native fitter keeps the vendored TMRCA/objective semantics together with SciPy Powell, and the upstream bridge now always captures the vendored `.fittingdetails.txt` artifact so result normalization stays implementation-independent.
**Context:** MSMC-IM is a thin reparameterization of MSMC2’s coalescence rates, so parity here is a downstream check on MSMC2’s output, the chi-square objective in `smckit.tl._msmc_im`, and the public upstream bridge semantics.
**Remaining tasks:** Keep the helper-level oracle tests synchronized if the upstream script changes and add empirical nonhuman sensitivity runs in the frozen publication workflow.
**References:** :doc:`methods/msmc-im`, ``tests/integration/test_msmc_im_validation.py``

## eSMC2

**Progress:** The upstream R bridge is runnable locally, the native HMM builder matches upstream exactly on the tracked clean 800 bp fixture, and the native zipped E-step now matches upstream final-state sufficient statistics (`N`, `M`, `q_`, and log-likelihood) at numerical precision. Native end-to-end fit parity is enforced for the fixed-rho one-iteration fixture, the `estimate_rho=True` redo-extension fixture, the `estimate_beta` / `estimate_sigma` / combined beta-plus-sigma branches, the grouped `pop_vect=[3, 3]` beta-fitting fixture, and the harder all-singleton grouped-state matrix `pop_vect=[1, 1, 1, 1, 1, 1]` across beta, sigma, and beta-plus-sigma fits. The public input-family gate now also covers `.psmcfa` clean, missing-site, and multi-record runs plus `multihetsep` single-pair, multi-pair, multi-file, and `skip_ambiguous=True` cases.
**Decisions:** Keep `vendor/eSMC2` as the oracle, prefer the upstream bridge when users ask for oracle behavior, and only promote native parity where there is an explicit upstream-backed integration gate. Normalize public upstream payloads to the final builder state while preserving raw returned values in `results["esmc2"]["upstream"]`.
**Context:** eSMC2 adds dormancy and self-fertilization to the PSMC-style HMM, so it is the method that stress-tests provenance-aware result reporting, zipped Baum-Welch sufficient statistics, and parity between native and upstream control flow.
**Remaining tasks:** Keep `implementation="auto"` conservative until there is a reason to prefer native beyond the current public-surface matrix. The remaining oracle work is broader than input families now: additional grouped-`Xi` layouts, more diverse low-information cases around the vendor theta filter, and any vendor-only controls that smckit later decides to expose. Keep using the final-state oracle payload for fit-level likelihood checks because raw upstream `res$LH` can lag the final returned parameter point just as raw `res$Tc` can.
**References:** :doc:`methods/esmc2`, ``tests/integration/test_esmc2_upstream_validation.py``, ``tests/unit/test_esmc2.py``

## diCal2

**Progress:** The vendored README fixtures share upstream/native result normalization, and the native search path runs independently with Java-style RNG handling, exact Java `nextLong` spawning, and Java-style coordinatewise shuffle semantics. The native optimizer reaches the same best-fit parameter vector as upstream on both README `exp` and README `IM`. Matching the Java ODE tolerance and Ethan-trunk infinite-tail stopping rule closes the fixed-point likelihood differences to at most `5.38e-11`. Independent fitted clean-split, migration-window, three-population, pulse-introgression, and exponential-growth one-step simulations now reproduce the Java endpoint. Generated structured PAC fits and a two-contig file-backed structured PAC fit also reproduce Java endpoints; impossible zero-trunk states are excluded from the native M-step exactly where Java records zero expectation. All runnable original trunk families have native fixed-point oracles across their valid cake modes, and all newly implemented cake/exact families have fitted one-step coverage. Epoch/ancient-deme/present-deme hidden states and per-CSD mean-rate additional trunk intervals also match Java at fixed points and fitted endpoints, alone and in combination. The pinned recursive normal-CLI path returns a null trunk, which exact upstream preserves while native rejects explicitly. Transition sufficient statistics remain grouped by physical HMM distance, and the default M-step matches Java's conditioning that merges no-recombination with recombination back into the same lineage. The transition-type-conditioned objective also has a strict fitted oracle. Native marginal-KL implements the intended source formula, while exact upstream execution preserves the pinned jar's null-mutation-rate crash.
**Decisions:** Keep the `diCal2.jar` bundle as the reference, normalize upstream results into the same plot-ready `time`/`ne` fields as native, and preserve upstream numerical quirks when they affect results. The former full-search expected failure is now a strict scheduled oracle.
**Context:** diCal2 is the most experimental method shipped, so parity ensures we understand the limits of our tokenizer, stochastic optimizer replay, and Java-bridge validation path.
**Remaining tasks:** Add empirical PAC simulations, establish a repaired-source marginal-KL oracle, complete the original-compatible artifact catalogue, and gather Linux plus broader-workflow performance evidence before reconsidering the native trust warning.
**References:** :doc:`methods/dical2`, :doc:`internals-dical2`, ``tests/integration/test_dical2_upstream_validation.py``

## SMC++

**Progress:** The repo now vendors the upstream SMC++ source tree, the public upstream bridge still runs through the controlled side Python environment, and the default native path follows the upstream-style one-pop interpretation (two distinguished haplotypes, one-pop preprocessing, one-pop observation scaling, and an EM/coordinate-update optimizer with the upstream-style global scale step). The strict small one-pop control fixture and the larger tracked `.smc` one-pop fixture now both clear the shared-grid parity gate, and fixed-model `gamma0`, `xisum`, and log-likelihood also match upstream tightly on the same matrix.
**Decisions:** Treat the vendored upstream tree as the oracle, keep `implementation="upstream"` as the fidelity path when users want the original tool, and promote native one-pop SMC++ only where the tracked upstream-backed matrix is green.
**Context:** SMC++ is the conditioned SFS method that will stress-test the prepared `SmcData` container and cross-method comparison plots.
**Remaining tasks:** Expand the one-pop matrix beyond the current strict small control plus bundled larger `.smc` fixture, then decide whether additional SMC++ input families deserve separate parity contracts.
**References:** :doc:`methods/smcpp`, :doc:`smcpp-parity-closure`

## SSM

**Progress:** The extension framework is in design mode, so parity has not yet been assessed.
**Decisions:** Define `smckit.ext.ssm.SmcStateSpace` and core transition/emission contracts before writing any concrete method.
**Context:** SSM is the future-proofing surface that will allow new models (e.g., eSMC2 variations) to plug into the existing API.
**Remaining tasks:** Implement the base class, deliver the first concrete SSM (PSMC/MSMC wrapped) and then add parity checks against whichever methods it mirrors.
**References:** :doc:`methods/ssm`
