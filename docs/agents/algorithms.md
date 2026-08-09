# Algorithm Map for Agents

## PSMC

- Purpose: infer `N_e(t)` from one diploid genome.
- Input: `.psmcfa`.
- Upstream status: vendored C source plus public upstream runner.
- Native status: working and validated against tracked reference output.
- Main outputs: `time`, `lambda_k`, `ne`, `theta`, `rho`.

## PSMC+

- Purpose: pairwise demographic inference with genomic-rate heterogeneity.
- Input: one or more multihetsep files plus optional matched mutation and
  recombination maps.
- Upstream status: immutable MIT-licensed Python source, exact inference and HMM
  simulation entry points, shell-free raw runner, frozen numeric oracle, and a
  pinned OCI/Apptainer runtime definition.
- Native status: public fit/decode engine with frozen constant-rate and mapped
  final-output parity. Warmed Linux x86-64 core benchmarks show 2.27x fit and
  1.25x decode speedups with bootstrap intervals excluding parity; `auto`
  remains upstream until broader empirical validation closes.
- Main outputs: time boundaries, inverse coalescence rates, theta, rho,
  likelihood history, posterior TMRCA decoding, and marginal recombination.

## MSMC2

- Purpose: infer piecewise coalescence rates from multiple haplotypes.
- Input: `multihetsep`.
- Upstream status: vendored D source plus public CLI-backed upstream runner.
- Native status: strong fixture parity is tracked.
- Main outputs: time boundaries, lambdas, mutation/recombination estimates.

## MSMC-IM

- Purpose: fit a continuous isolation-migration model on top of MSMC2 output.
- Input: MSMC2 combined output.
- Upstream status: vendored Python script plus public upstream runner.
- Native status: interchangeable with upstream on the enforced 4-case
  Yoruba/French oracle matrix, including the shared public payload fields.
- Main outputs: `N1`, `N2`, `m`, `M`.

## SMC++

- Purpose: infer demography from many unphased individuals.
- Input: `.smc` / `.smc.gz`.
- Upstream status: vendored upstream source tree plus public bridge; execution
  still depends on a controlled side Python environment for the compiled runtime.
- Native status: available; default native behavior now follows the upstream
  one-pop preprocessing, hidden-state, and compressed HMM path. The tracked
  one-pop matrix now clears strict parity on both the small control and the
  bundled larger `.smc` fixture, with fixed-model HMM statistics also matching
  upstream on that matrix.
- Main outputs: `ne`, `time`, `theta`, `rho`, hidden-state metadata.

## eSMC2

- Purpose: extend pairwise SMC inference to dormancy and selfing.
- Input: pairwise sequence / `.psmcfa`-style data.
- Upstream status: vendored R package plus public bridge.
- Native status: available; tracked fit parity now covers the public `.psmcfa`
  and `multihetsep` input families, with upstream still acting as the broader
  fidelity baseline for vendor-surface experiments outside that matrix.
- Main outputs: `Tc`, `Xi`, `ne`, `beta`, `sigma`, `rho`.

## ASMC

- Purpose: decode pairwise TMRCA along the genome at scale.
- Input: ASMC haplotypes, maps, decoding quantities.
- Upstream status: vendored source/data plus public executable-backed upstream runner.
- Native status: good tracked parity on the vendored regression fixture.
- Main outputs: posterior means, MAP states, aggregated posteriors.

## diCal2

- Purpose: fit structured demographic histories with migration and multiple populations.
- Input: `.param`, `.demo`, `.config`, phased sequence data.
- Upstream status: vendored `diCal2.jar` plus public stdout-parsing upstream bridge.
- Native status: README searches and independent fitted clean-split,
  migration-window, three-population, pulse-introgression, and growth oracles
  reproduce the Java endpoints; transition-type conditioning also has a strict
  fitted oracle. Generated and file-backed structured PAC one-step fits also
  agree. All runnable original trunk families now have native fixed-point
  coverage; cake and exact families also have fitted one-step coverage. The
  pinned recursive CLI path returns a null trunk and is preserved as a failure.
  Native marginal-KL and the all-CSD PAC default repair distinct pinned Java
  null-reference crashes; marginal-KL now matches a separately built repaired
  GPL oracle for LOL and PAC, while the repaired all-CSD default remains
  empirical-validation-only. Ancient-state aggregation
  and per-CSD additional trunk intervals now have fixed-point and fitted
  oracles. The original stdout artifact surface is catalogued and native
  persists every EM row plus normalized JSON. Empirical PAC validation and
  broader cross-platform performance remain unpromoted.
- Main outputs: best-fit parameters, refined demography, likelihood metadata.
