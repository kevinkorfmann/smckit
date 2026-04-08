# Choosing a Method

This page helps you pick a method by data shape, inference target, and current
implementation maturity.

## Read this table first

| Method | Best for | Input | Output focus | Implementation reality |
|---|---|---|---|---|
| **PSMC** | one diploid genome | `.psmcfa` | `N_e(t)` | strong native path |
| **eSMC2** | one diploid genome with dormancy/selfing | `.psmcfa` or pairwise sequence input | `N_e(t)` + `beta` + `sigma` | native and upstream |
| **MSMC2** | 2-8 phased haplotypes | `.multihetsep` | coalescence rates and `N_e(t)` | native public path, upstream bridge pending |
| **MSMC-IM** | two-population MSMC output | `.combined.msmc2.final.txt` | `N1(t)`, `N2(t)`, `m(t)`, `M(t)` | native and upstream, auto prefers upstream |
| **SMC++** | many unphased genomes | `.smc.gz` | `N_e(t)` | native and upstream, auto prefers upstream |
| **ASMC** | per-site pairwise ancestry | hap/map/samples + decoding quantities | TMRCA along the genome | native public path |
| **diCal2** | explicit structured demographic models | `.param`, `.demo`, `.config`, sequences | sizes, growth, migration parameters | native public path |
| **PSMC-SSM** | research and optimizer experimentation | `psmcfa`-style observations | differentiable PSMC | native-only framework |

## A simple decision path

### One diploid genome

- use **PSMC** for a standard demographic-history run
- use **eSMC2** if dormancy or selfing is biologically important
- use **PSMC-SSM** only if you need the experimental differentiable framework

### Multiple phased haplotypes

- use **MSMC2** for coalescence-rate or recent-history work
- use **ASMC** if you need per-site pairwise TMRCA, not just one population-size curve
- use **MSMC-IM** after MSMC2 when the real target is a two-population migration summary

### Many unphased genomes

- use **SMC++**

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

Today, only some methods have a public upstream bridge. Use the homepage status
matrix and per-method pages to see which paths exist.

## If you are unsure

Start with:

- **PSMC** for one diploid genome
- **SMC++** for many unphased genomes
- **ASMC** for pairwise local ancestry / TMRCA along the genome
- **MSMC2** if another downstream step depends on MSMC-style output

## See also

- [I/O formats](io-formats.md)
- [Interpreting results](interpreting-results.md)
- [Gallery](gallery.md)
