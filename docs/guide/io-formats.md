# I/O Formats

This page answers two practical questions:

1. what kind of file does each method expect?
2. where can I see a concrete example file?

## Quick index

| Function | File type | Used by |
|---|---|---|
| {func}`smckit.io.read_psmcfa` | `.psmcfa` | PSMC, eSMC2 |
| {func}`smckit.io.read_multihetsep` | `.multihetsep` | MSMC2, PSMC+ |
| {func}`smckit.pp.multihetsep_from_vcf` | called VCF(s) + callability BED mask(s) | MSMC2, PSMC+ preparation |
| {func}`smckit.io.read_msmc_combined_output` | `.combined.msmc2.final.txt` | MSMC-IM |
| {func}`smckit.io.read_asmc` | prefix for `.hap.gz`, `.samples`, `.map.gz` | ASMC |
| {func}`smckit.io.read_decoding_quantities` | `.decodingQuantities.gz` | ASMC |
| {func}`smckit.io.read_smcpp_input` | `.smc` / `.smc.gz` | SMC++ |
| {func}`smckit.pp.smcpp_from_vcf` | VCF/VCF.GZ + optional BED | SMC++ preparation |
| {func}`smckit.io.read_smcpp_model` | `.smcpp.model.json` | SMC++ reload/plot |
| {func}`smckit.io.read_dical2` | model-file bundle + sequence input | diCal2 |

## PSMCFA

Used by: **PSMC**, **eSMC2**

A `.psmcfa` file is a FASTA-like window sequence. Each character summarizes one
window as homozygous, heterozygous, or missing.

Example:

- repo file: `tests/data/NA12878_chr22.psmcfa`
- GitHub: `https://github.com/kevinkorfmann/smckit/blob/main/tests/data/NA12878_chr22.psmcfa`

## Multihetsep

Used by: **MSMC2**, **PSMC+**

A `.multihetsep` file is a variant-centric table. Each row records one
segregating site, the number of callable bases since the previous segregating
site, and the alleles for the haplotypes in the analysis.

Use {func}`smckit.pp.multihetsep_from_vcf` to stream-join one called VCF per
individual, intersect sample/mappability masks, subtract exclusion masks,
represent unphased configurations, and optionally resolve/remove trios. The
converter refuses variant-only input by default: sites absent from a VCF are
not scientifically equivalent to confidently called homozygous-reference
sites. Set `assume_all_sites_callable=True` only for synthetic or independently
validated all-sites inputs.

Three-column BED masks use standard zero-based, half-open coordinates. The
two-column one-based, inclusive mask convention from `msmc-tools` is also
accepted. Source/output hashes, samples, controls, counts, and the pinned
compatibility-oracle identity are stored in `data.uns["preprocessing"]`.

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
every position. smckit ships a compact packaged fixture at
`smckit.io.example_path("smcpp/example.smc.gz")`.

Use {func}`smckit.pp.smcpp_from_vcf` for one- or two-population VCF input,
explicit distinguished haplotypes, BED masks, gzip output, and conservative
missing-run handling. The reader/writer preserve all population triplets.
Native split inference remains under development, but the I/O layer never
silently discards population columns.

Fitted models use `.smcpp.model.json`. These files contain a stable normalized
section and an original-style `SMCModel` block, and can initialize a later
native fit.

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
