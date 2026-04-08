# I/O Formats

This page answers two practical questions:

1. what kind of file does each method expect?
2. where can I see a concrete example file?

## Quick index

| Function | File type | Used by |
|---|---|---|
| {func}`smckit.io.read_psmcfa` | `.psmcfa` | PSMC, eSMC2 |
| {func}`smckit.io.read_multihetsep` | `.multihetsep` | MSMC2 |
| {func}`smckit.io.read_msmc_combined_output` | `.combined.msmc2.final.txt` | MSMC-IM |
| {func}`smckit.io.read_asmc` | prefix for `.hap.gz`, `.samples`, `.map.gz` | ASMC |
| {func}`smckit.io.read_decoding_quantities` | `.decodingQuantities.gz` | ASMC |
| {func}`smckit.io.read_smcpp_input` | `.smc` / `.smc.gz` | SMC++ |
| {func}`smckit.io.read_dical2` | model-file bundle + sequence input | diCal2 |

## PSMCFA

Used by: **PSMC**, **eSMC2**

A `.psmcfa` file is a FASTA-like window sequence. Each character summarizes one
window as homozygous, heterozygous, or missing.

Example:

- repo file: `tests/data/NA12878_chr22.psmcfa`
- GitHub: `https://github.com/kevinkorfmann/smckit/blob/main/tests/data/NA12878_chr22.psmcfa`

## Multihetsep

Used by: **MSMC2**

A `.multihetsep` file is a variant-centric table. Each row records one
segregating site, the distance from the previous site, and the alleles for the
haplotypes in the analysis.

Example:

- repo file: `data/msmc2_test.multihetsep`
- GitHub: `https://github.com/kevinkorfmann/smckit/blob/main/data/msmc2_test.multihetsep`

## MSMC combined output

Used by: **MSMC-IM**

This is not raw input. It is the output of an MSMC/MSMC2 two-population run,
containing within- and cross-population coalescence-rate curves.

Example:

- repo file: `vendor/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt`
- GitHub: `https://github.com/kevinkorfmann/smckit/blob/main/vendor/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt`

## ASMC file bundle

Used by: **ASMC**

ASMC needs a file prefix that expands to:

- `.hap.gz` for phased haplotypes
- `.samples` for sample metadata
- `.map.gz` for genetic positions

It also needs a `.decodingQuantities.gz` file with precomputed HMM tables.

Examples:

- repo directory: `vendor/ASMC/ASMC_data/examples/asmc`
- GitHub directory: `https://github.com/kevinkorfmann/smckit/tree/main/vendor/ASMC/ASMC_data/examples/asmc`
- decoding quantities: `https://github.com/kevinkorfmann/smckit/blob/main/vendor/ASMC/ASMC_data/decoding_quantities/30-100-2000_CEU.decodingQuantities.gz`

## SMC++ span-encoded input

Used by: **SMC++**

An `.smc.gz` file stores the genome in span-encoded form rather than listing
every position. It is the standard SMC++ input produced by the upstream
tooling.

The repository does not currently ship a compact `.smc.gz` fixture. The best
starting point is the upstream project:

- `https://github.com/popgenmethods/smcpp`
- `https://github.com/popgenmethods/smcpp/tree/master/example`

## diCal2 model-file bundle

Used by: **diCal2**

diCal2 expects a small family of files rather than one file:

- `.param`
- `.demo`
- optional `.rates`
- `.config`
- sequence input such as VCF plus a reference

Examples:

- repo directory: `vendor/diCal2/examples/fromReadme`
- GitHub directory: `https://github.com/kevinkorfmann/smckit/tree/main/vendor/diCal2/examples/fromReadme`

## Output readers

smckit also reads upstream outputs for comparison and post-processing:

- {func}`smckit.io.read_psmc_output`
- {func}`smckit.io.read_msmc_output`
- {func}`smckit.io.read_msmc_im_output`

Use these when you want to compare native and upstream results directly.
