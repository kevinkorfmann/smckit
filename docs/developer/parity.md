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

## ASMC

**Progress:** The vendored `n300` fixture confirms MAP indices match ~99.84% of sites; posterior means still require a tighter `rtol=1e-3` match for a handful of sites.
**Decisions:** Canonical decoding quantities (`.decodingQuantities.gz`) anchor the comparison, and we replay the same haplotypes/genetic positions as the reference.
**Context:** ASMC extends PSMC by decoding per-pair coalescence times; it depends on optimized transition table decomposition and dense decoding quantization.
**Remaining tasks:** Eliminate the remaining posterior drift, expand the fixture catalog beyond the n300 array, and harden the `smckit.tl.asmc` regression suite.
**References:** :doc:`methods/asmc`, ``tests/integration/test_asmc_validation.py``

## MSMC2

**Progress:** Six fixed upstream fixtures show left boundary relative error `<=4.15e-06`, lambda relative error `<=2.45e-03`, lambda correlation `>=0.999999865`, and log-likelihood delta `<=4.75e-03`.
**Decisions:** Reuse the same `vendor/msmc2` fixtures for the integration test, and build `smckit.tl.msmc2` to match the D implementation before layering GPU/async optimizations.
**Context:** MSMC2 is the multi-haplotype extension that fuels MSMC-IM, SMC++, and eSMC2 fixtures; the multihetsep input parsers and combined output readers all trace back to this parity gate.
**Remaining tasks:** Track original MSMC parity, stress-test cross-count heatmaps (stdpopsim `z`), and broaden multi-population validations.
**References:** :doc:`methods/msmc2`, ``tests/integration/test_msmc_validation.py``

## MSMC-IM

**Progress:** The vendored Yoruba/French ceremony is now enforced as a small oracle matrix: the public upstream runner matches the vendored CLI artifact directly, and the native path matches the same `.MSMC_IM.estimates.txt` outputs across the default case plus two non-default pattern/regularization/init bundles with strict array comparisons.
**Decisions:** The official `MSMC_IM.py` run remains the ceremony, and the native fitter now reuses the vendored TMRCA/objective semantics together with SciPy Powell so the search stays on the same solution path outside the default bundle.
**Context:** MSMC-IM is a thin reparameterization of MSMC2’s coalescence rates, so parity here is a downstream check on MSMC2’s output, the chi-square objective in `smckit.tl._msmc_im`, and the public upstream bridge semantics.
**Remaining tasks:** Expand fixture diversity beyond the tracked Yoruba/French input, add heatmap diagnostics to monitor `m(t)` vs `M(t)`, and keep the helper-level oracle tests in sync if the upstream script changes.
**References:** :doc:`methods/msmc-im`, ``tests/integration/test_msmc_im_validation.py``

## eSMC2

**Progress:** The upstream R bridge is runnable locally, the native HMM builder matches upstream exactly on the tracked clean 800 bp fixture, and the native zipped E-step now matches upstream final-state sufficient statistics (`N`, `M`, `q_`, and log-likelihood) at numerical precision. Native end-to-end fit parity is enforced for the fixed-rho one-iteration fixture, the `estimate_rho=True` redo-extension fixture, the `estimate_beta` / `estimate_sigma` / combined beta-plus-sigma branches, and one non-default grouped `pop_vect=[3, 3]` beta-fitting fixture.
**Decisions:** Keep `vendor/eSMC2` as the oracle, prefer the upstream bridge when users ask for oracle behavior, and only promote native parity where there is an explicit upstream-backed integration gate. Normalize public upstream payloads to the final builder state while preserving raw returned values in `results["esmc2"]["upstream"]`.
**Context:** eSMC2 adds dormancy and self-fertilization to the PSMC-style HMM, so it is the method that stress-tests provenance-aware result reporting, zipped Baum-Welch sufficient statistics, and parity between native and upstream control flow.
**Remaining tasks:** Expand the gate matrix beyond the tracked single-sequence fixture, cover harder grouped-state layouts (for example all-singleton `pop_vect`), and only then reconsider whether `implementation="auto"` should ever prefer native for eSMC2. Keep using the final-state oracle payload for fit-level likelihood checks because raw upstream `res$LH` can lag the final returned parameter point just as raw `res$Tc` can.
**References:** :doc:`methods/esmc2`, ``tests/integration/test_esmc2_upstream_validation.py``, ``tests/unit/test_esmc2.py``

## diCal2

**Progress:** The vendored README fixtures now share upstream/native result normalization, and the native search path runs independently again with Java-style RNG handling and upstream-style meta-start machinery. On the tracked README `exp` fixture, the native fixed-point likelihood gap at the upstream best-fit parameters is now about `7.45e-4`, and explicit `exp.rand` start-point replays now match the upstream endpoint to displayed precision with log-likelihood deltas at or below about `2.21e-4`. The current full independent `exp` meta-start run still misses the oracle fit, however, landing around `-15.9743` versus the upstream `-15.8771`.
**Decisions:** Keep the `diCal2.jar` bundle as the reference, normalize upstream results into the same plot-ready `time`/`ne` fields as native, and use the README `exp` sequence bundle as the first hard parity gate.
**Context:** diCal2 is the most experimental method shipped, so parity ensures we understand the limits of our tokenizer and JADE bridging code.
**Remaining tasks:** Close the remaining README `exp` meta-start selection gap, then bring the README `IM` grouped-locus likelihood/search path up to the same standard. Only after those fit-level gates are restored should the native trust warning be reconsidered.
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
