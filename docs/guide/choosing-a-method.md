# Choosing a Method

This page helps you pick a method by data shape, inference target, and current
implementation maturity.

## Read this table first

| Method | Best for | Input | Output focus | Implementation reality |
|---|---|---|---|---|
| **PSMC** | one diploid genome | `.psmcfa` | `N_e(t)` | strong native path |
| **PSMC+** | one diploid genome with background selection or other local rate heterogeneity | `.multihetsep` plus optional local coalescence/mutation/recombination maps | `N_e(t)`, posterior TMRCA, marginal recombination | native fit/decode passes cross-platform parity and speed gates; `auto` remains upstream pending empirical validation |
| **eSMC2** | one diploid genome with dormancy/selfing | `.psmcfa` or pairwise sequence input | `N_e(t)` + `beta` + `sigma` | native and upstream |
| **MSMC2** | 2-8 phased haplotypes | `.multihetsep` | coalescence rates and `N_e(t)` | promoted native path plus exact upstream bridge |
| **MSMC-IM** | two-population MSMC output | `.combined.msmc2.final.txt` | `N1(t)`, `N2(t)`, `m(t)`, `M(t)` | promoted native path plus exact upstream bridge |
| **SMC++** | many unphased genomes | `.smc.gz` | `N_e(t)` and clean population splits | promoted native/upstream workflows with frozen split performance evidence |
| **ASMC** | per-site pairwise ancestry | hap/map/samples + decoding quantities | TMRCA along the genome | promoted native path plus exact upstream bridge |
| **diCal2** | explicit structured demographic models | `.param`, `.demo`, `.config`, sequences | sizes, growth, migration parameters | broad native oracle coverage; `auto` remains upstream pending empirical and Linux performance gates |
| **PHLASH** | Bayesian history with uncertainty and larger samples | `.psmcfa`, VCF/BCF, tree sequence | posterior `N_e(t)` and credible intervals | maintained external package with normalized smckit results |
| **PSMC-SSM** | research and optimizer experimentation | `psmcfa`-style observations | differentiable PSMC | native-only framework |

## A simple decision path

### One diploid genome

- use **PSMC** for a standard demographic-history run
- use **PSMC+** when background selection or local coalescence/mutation-rate
  heterogeneity would bias the standard homogeneous model
- use **eSMC2** if dormancy or selfing is biologically important
- use **PSMC-SSM** only if you need the experimental differentiable framework

### Multiple phased haplotypes

- use **MSMC2** for coalescence-rate or recent-history work
- use **ASMC** if you need per-site pairwise TMRCA, not just one population-size curve
- use **MSMC-IM** after MSMC2 when the real target is a two-population migration summary

### Many unphased genomes

- use **SMC++** for deterministic composite-likelihood inference
- use **PHLASH** when posterior uncertainty or tree-sequence input is central

### Structured demographic model files already exist

- use **diCal2**

## How implementation choice affects method choice

smckit now exposes implementation provenance directly through
`implementation={"auto","native","upstream"}` on every public `smckit.tl`
algorithm.

- Choose `implementation="auto"` when you want the safest current behavior.
- Choose `implementation="native"` when you want the in-repo port explicitly.
- Choose `implementation="upstream"` when that bridge exists and you want the
  original algorithm ceremony.

All preserved method families now have a public upstream bridge. Use the
homepage status matrix and per-method pages to see which native capabilities
are promoted and which still make `auto` choose upstream.

## If you are unsure

Start with:

- **PSMC** for one diploid genome
- **SMC++** for many unphased genomes
- **PHLASH** for Bayesian uncertainty and large-sample inference
- **ASMC** for pairwise local ancestry / TMRCA along the genome
- **MSMC2** if another downstream step depends on MSMC-style output

## See also

- [I/O formats](io-formats.md)
- [Runtime and resource planning](runtime-estimates.md)
- [Project status](../get-started/project-status.md)
- [Interpreting results](interpreting-results.md)
- [Gallery](gallery.md)
