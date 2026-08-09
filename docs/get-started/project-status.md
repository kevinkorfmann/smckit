# Project status

Updated 9 August 2026.

smckit is substantially beyond a collection of prototypes: every preserved
method has an explicit execution contract, six native method families are
default-eligible for their promoted workflows, PSMC+ has passed strict
cross-platform parity and performance gates, diCal2 has broad native oracle
coverage, and PHLASH is available through a normalized external integration.

It is **not yet a 1.0 or preprint-ready release**. The remaining work is
concentrated in empirical validation, the last conservative promotion gates,
production-scale benchmarks, and the final reproducibility/figure package.

## Method-by-method snapshot

| Method | Current state | What remains before the final release claim |
|---|---|---|
| **PSMC** | Preserved upstream and promoted native implementation, including bootstrap, decoding, divergence, simulation, explicit intervals/parameters, and compatible output. | Refresh production-scale cross-platform runtime evidence and include retained empirical runs. |
| **ASMC** | Preserved upstream and promoted native array/sequence decoding with pair selection, interval jobs, posterior summaries, and compatible artifacts. | Freeze production-scale timing/memory across representative pair counts and empirical sequence data. |
| **MSMC2** | Preserved upstream and promoted native inference across pair selection, missing/ambiguous data, multiple files/chromosomes, boundary choices, fixed recombination, and output. | Add retained whole-genome cross-platform performance and empirical multi-haplotype evidence. |
| **MSMC-IM** | Preserved upstream and promoted native fitter on enforced Yoruba/French and independent synthetic split families. | Expand retained empirical families and freeze production timing/memory. |
| **eSMC2** | Preserved R implementation and promoted native path across the tracked public input/model matrix, with original numeric output. | Freeze production timing/memory and the retained nonhuman selfing/dormancy analysis. |
| **SMC++** | Preserved upstream and promoted native one-population and clean-split workflows, including VCF preparation, masks, compression, cross-validation, model I/O, and plotting. | Scale retained benchmarks and empirical analyses beyond the frozen split control. |
| **diCal2** | Exact Java execution is preserved. Native clean-split, migration, growth, introgression, multi-population, trunk, cake, conditioning, PAC, and detailed EM-output families have extensive oracle coverage. | Empirically validate the repaired all-CSD PAC default and close Linux/broader-workflow performance before native becomes the default. |
| **PSMC+** | Exact upstream source/CLI is preserved. All 12 native capability cases pass on Linux x86-64 and macOS ARM64; fit/decode are faster with lower peak memory on both platforms. The method explicitly supports local coalescence/mutation-rate variation used to correct background-selection bias. | Complete retained human and nonhuman empirical validation and container execution before `auto` switches to native. |
| **PHLASH** | Maintained external PHLASH execution is normalized for PSMCFA, indexed VCF/BCF, and tree-sequence inputs with posterior/credible-interval artifacts. | Execute and retain the frozen 80-fit accuracy/coverage workflow and GPU/CPU resource matrix. A native rewrite is not required for 1.0. |
| **SSM** | Native experimental state-space framework for new methods and differentiable PSMC research. | Define a stable production surface only if it develops into a release method; it is not an upstream parity target. |

The machine-readable source of truth is returned by
`smckit.capabilities()` and rendered in the documentation homepage. Detailed
feature and numerical gates live in
[`docs/parity/feature-ledger.json`](../parity/feature-ledger.json) and
[`docs/parity/oracle-specifications.json`](../parity/oracle-specifications.json).

## Production foundations already in place

- `implementation={"auto", "native", "upstream"}` has defined, provenance-rich
  semantics across public methods.
- Original upstream tools remain available; exact raw CLI execution is distinct
  from native code and never silently substituted.
- Linux/macOS and Python 3.10–3.13 CI exercise unit tests, packaging, strict
  documentation, preservation checks, and selected live upstream oracles.
- Results include implementation, versions, arguments, hashes, seed, platform,
  runtime, warnings, and artifacts.
- PSMC+, SMC++ split inference, and two diCal2 structured fits have immutable
  comparative performance evidence with memory measurements.
- The native VCF-to-multihetsep converter requires explicit invariant-site
  callability, streams chromosome-scale inputs, and is checked against a
  checksum-pinned `msmc-tools` oracle.
- A deterministic NA12878/1000 Genomes 30× GRCh38 source manifest records
  accessions, source checksums, callability rules, and remaining blockers.

## The critical path from here

1. Implement and independently validate CRAM-derived consensus/depth
   callability, then complete the NA12878 chromosome 22 smoke analysis.
2. Run matched native/upstream human autosome analyses with immutable derived
   inputs and evidence JSON.
3. Pin and execute an unambiguously reusable *Arabidopsis thaliana* dataset for
   the selfing/nonhuman case. VarGoats requires steering-committee clearance
   under its current data-use agreement, so it cannot silently become retained
   publication evidence.
4. Finish empirical PSMC+ and diCal2 promotion gates and the missing per-method
   production runtime matrix.
5. Execute the 80-fit PHLASH workflow and the complete frozen simulation suite.
6. Generate code-derived parity, runtime/memory, accuracy, uncertainty, and
   empirical figures from immutable evidence.
7. Complete clean-install, containers/Apptainer, packaging, SBOM, DOI, API, and
   claim-to-evidence audits for 1.0.

Drafting and submission-authoring files are deliberately kept outside the
public repository. Public documentation and workflow code may describe the
evidence contract, but unpublished manuscript prose is not a project artifact.

## How to read the current claims

“Native exists” means a callable in-repo implementation is present. “Parity”
means the named frozen capability passed its documented oracle; it does not
automatically cover every untested option combination. “Faster” is used only
where a repeated, matched benchmark confidence interval excludes parity.
“Default-eligible” means `auto` may choose native for the promoted capability;
explicit upstream execution remains permanently available.

See [Runtime and resource planning](../guide/runtime-estimates.md) for the
measured numbers and the benchmark gaps that remain.
