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

**Progress:** The upstream R bridge is runnable locally, the native HMM builder matches upstream exactly on the tracked clean 800 bp fixture, and the native zipped E-step now matches upstream final-state sufficient statistics (`N`, `M`, `q_`, and log-likelihood) at numerical precision. Native end-to-end fit parity is enforced for the fixed-rho one-iteration fixture and the `estimate_rho=True`, redo-extension fixture.
**Decisions:** Keep `vendor/eSMC2` as the oracle, prefer the upstream bridge when users ask for oracle behavior, and only promote native parity where there is an explicit upstream-backed integration gate. Normalize public upstream payloads to the final builder state while preserving raw returned values in `results["esmc2"]["upstream"]`.
**Context:** eSMC2 adds dormancy and self-fertilization to the PSMC-style HMM, so it is the method that stress-tests provenance-aware result reporting, zipped Baum-Welch sufficient statistics, and parity between native and upstream control flow.
**Remaining tasks:** Port the remaining upstream-specific M-step behavior for `estimate_beta`, `estimate_sigma`, and broader grouped-`Xi` cases, add explicit parity fixtures for those branches, and only then reconsider whether `implementation="auto"` should ever prefer native for eSMC2.
**Deferred checks:** Revisit native fit parity for `estimate_beta=True`, `estimate_sigma=True`, combined beta-plus-sigma fitting, and non-default grouped `pop_vect` / grouped-`Xi` cases. When checking those branches, compare against the normalized upstream public payload (`Tc`/`t` from the final builder state), but keep `results["esmc2"]["upstream"]["Tc_returned"]` in view because the raw upstream `res$Tc` can lag behind the final returned parameter point.
**References:** :doc:`methods/esmc2`, ``tests/integration/test_esmc2_upstream_validation.py``, ``tests/unit/test_esmc2.py``

## diCal2

**Progress:** Vendored oracle runs (from the README examples) cap log-likelihood deltas at `0.36` absolute (`2.27%` relative) and `0.86` absolute (`1.22%` relative) for the `IM` variant.
**Decisions:** Keep the `diCal2.jar` bundle as the reference and target the documented experiment sequences.
**Context:** diCal2 is the most experimental method shipped, so parity ensures we understand the limits of our tokenizer and JADE bridging code.
**Remaining tasks:** Increase fixture coverage across `exp`/`IM` settings and document how `smckit.tl.dical2` is configured to match the original command lines.
**References:** :doc:`methods/dical2`

## SMC++

**Progress:** The repo now vendors the upstream SMC++ source tree, the public upstream bridge still runs through the controlled side Python environment, and the default native path has been moved onto the upstream-style one-pop interpretation (two distinguished haplotypes, one-pop preprocessing, one-pop observation scaling, and an EM/coordinate-update optimizer with the upstream-style global scale step). On the tracked small one-pop fixture, native now clears the shared-grid parity gate with `log_corr > 0.99`, scale ratio near `1.0`, and median relative `N_e(t)` error below `1%`.
**Decisions:** Treat the vendored upstream tree as the oracle, keep `implementation="upstream"` as the fidelity path, and use the default native path only as an improving port rather than a parity claim.
**Context:** SMC++ is the conditioned SFS method that will stress-test the prepared `SmcData` container and cross-method comparison plots.
**Remaining tasks:** Confirm the same behavior on a larger realistic one-pop fixture, add tracked `.smc` fixtures in `tests/data`, and promote that larger fixture to the same parity gate before claiming completion.
**References:** :doc:`methods/smcpp`

## SSM

**Progress:** The extension framework is in design mode, so parity has not yet been assessed.
**Decisions:** Define `smckit.ext.ssm.SmcStateSpace` and core transition/emission contracts before writing any concrete method.
**Context:** SSM is the future-proofing surface that will allow new models (e.g., eSMC2 variations) to plug into the existing API.
**Remaining tasks:** Implement the base class, deliver the first concrete SSM (PSMC/MSMC wrapped) and then add parity checks against whichever methods it mirrors.
**References:** :doc:`methods/ssm`
