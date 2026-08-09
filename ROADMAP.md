# Roadmap

This is the canonical delivery plan. A milestone is complete only when its
acceptance checks pass on a clean checkout; dates are forecasts, not substitutes
for evidence.

## 0.1 — production and preservation contract

- [x] Versioned result provenance schema.
- [x] Capability-aware `auto`, strict `native`, and exact `upstream` selection.
- [x] Unified `run`, `upstream`, `methods`, and `status` CLI.
- [x] Five test tiers and routine unit-test time budget.
- [x] Linux/macOS Python 3.10–3.13 CI definition.
- [x] Immutable PSMC and MSMC2 source pins.
- [ ] Build and verify all upstream OCI images on CI.
- [ ] Complete license, checksum, example, and citation audit for every oracle.

## 0.5 — complete preservation and Waves A/B

- [ ] Every documented upstream example is executable offline.
- [ ] PSMC, MSMC2, and eSMC2 feature ledgers have no scientific gaps.
- [ ] ASMC and MSMC-IM feature ledgers have no scientific gaps.
- [ ] Native promotions meet frozen correctness, runtime, and memory gates.

## 0.8 — high-risk parity closure

- [x] SMC++ supports preparation, masks, compression, multi-population and split workflows.
- [ ] diCal2 implements exact configuration and likelihood semantics.
- [ ] PSMC+ has an immutable upstream preservation layer, complete feature
  ledger, normalized schema, and fixed oracle fixtures.
- [ ] Both method oracle specifications and ledgers are complete.

## 0.9 — publication release candidate

- [ ] PHLASH external integration and normalized posterior schema are stable.
- [ ] Native PSMC+ covers fitting, local-rate maps, decoding, binning,
  grouped intervals, initialization, convergence, and recombination controls,
  passes parity, and clears its performance promotion gate.
- [ ] All standard native workflows are promoted.
- [ ] The Snakemake bioRxiv protocol, datasets, hardware, and claims are frozen.

## 1.0 — publication release

- [ ] Clean wheels and conda packages pass Linux x86-64 and macOS x86-64/ARM64.
- [ ] CPU equivalence and optional NVIDIA/JAX validation are green.
- [ ] Simulation, human, *Arabidopsis*, and an accession-frozen goat analysis
  reproduce from frozen inputs.
- [ ] The goat case uses VarGoats only after satisfying its publication/data
  sharing terms; otherwise a fully public goat cohort is substituted and the
  choice is documented before analysis.
- [ ] Documentation, SBOM, containers, tutorial data, Zenodo DOI, and API freeze exist.
- [ ] The private local bioRxiv manuscript, supplement, and code-generated
  figures reproduce from immutable benchmark and empirical result files.
- [ ] Two independent clean-install reports and the final claim-to-evidence audit pass.
- [ ] No unresolved correctness, licensing, or reproducibility blocker remains.

See `preservation/upstreams.json`, `docs/developer/testing.md`, and
`docs/parity/` for the evidence behind these checkpoints.
