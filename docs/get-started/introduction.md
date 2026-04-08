# Introduction

**smckit** is a unified Python framework for Sequentially Markovian Coalescent
(SMC) demographic inference. It brings the major SMC tools — PSMC, MSMC2,
SMC++, ASMC, eSMC2, MSMC-IM, and diCal2 — under a single, consistent API.

The project philosophy is preservation first:

- `implementation="upstream"` keeps the original algorithm alive by running
  the upstream tool in its native language/runtime whenever that bridge exists.
- `implementation="native"` is the long-term in-repo port that gradually
  replaces fragmented codebases with maintainable Python/Numba implementations.
- `smckit.upstream.status()` and `smckit.upstream.bootstrap()` make upstream
  readiness explicit, so the repo can tell you whether the oracle path is
  actually runnable instead of implying it.

The project is inspired by what
[scanpy](https://scanpy.readthedocs.io) did for single-cell genomics: take a
fragmented ecosystem of tools and provide one coherent, well-documented,
modern Python interface that scales from teaching to production.

## What is the coalescent?

The **coalescent** is the standard model in population genetics for the
genealogical history of a sample of DNA sequences. Tracing backward in time,
ancestral lineages "coalesce" (merge) at common ancestors. The shape of this
genealogy — when and how often coalescences occur — is determined by the
demographic history of the population: a small population produces frequent,
recent coalescences; a large population produces rare, ancient ones. By
observing the patterns of mutations along a genome, we can statistically
recover the coalescent times and therefore reconstruct $N_e(t)$, the
effective population size as a function of time.

## What is the Sequentially Markovian Coalescent?

The full coalescent with recombination produces a different genealogy at every
position in the genome, and inference under this model is computationally
intractable for whole genomes. The **Sequentially Markovian Coalescent (SMC)**
is an approximation introduced by McVean & Cardin (2005) and refined by
Marjoram & Wall (2006) that makes inference tractable: as we move along the
genome, the genealogy changes only via local recombination events, and the
sequence of genealogies forms a Markov chain. This Markov property is what
allows SMC methods to be cast as Hidden Markov Models (HMMs), where the hidden
states are discretized coalescent times and the observations are heterozygosity
patterns. Standard HMM machinery — forward-backward, Viterbi, EM — then yields
efficient demographic inference algorithms.

## The landscape of SMC tools

Over the past decade, several SMC-based methods have become standard tools in
population genetics. Each addresses a different inference problem or data
configuration:

- **PSMC** (Li & Durbin, 2011) — the original. Infers $N_e(t)$ from a single
  diploid genome using a 2-state HMM along the sequence.
- **MSMC** and **MSMC2** (Schiffels & Durbin, 2014; Schiffels & Wang, 2020) —
  extends PSMC to multiple haplotypes, providing better resolution at recent
  times and enabling cross-population coalescence rate estimation.
- **SMC++** (Terhorst, Kamm & Song, 2017) — uses a "distinguished lineage"
  trick and the conditioned site frequency spectrum to incorporate data from
  many unphased individuals at once.
- **ASMC** (Palamara et al., 2018) — infers per-pair, per-site coalescence
  times with a linear-time HMM, scalable to thousands of haplotype pairs.
- **eSMC2** (Sellinger et al., 2020) — extends PSMC to organisms with seed
  banks and self-fertilization.
- **MSMC-IM** (Wang et al., 2020) — fits a continuous Isolation-Migration
  model to MSMC2 cross-population coalescence rates.
- **diCal2** (Steinrücken, Kamm & Song, 2019) — fits structured demographic
  models with population sizes, migration, and growth from phased haplotypes.

These tools are written in a mix of C, C++, D, R, and Python; they have
incompatible input formats, divergent CLIs, and uneven documentation. smckit
provides a unified Python wrapper, a single in-memory data container, and a
migration path where upstream compatibility can ship first and native ports can
catch up later.

## What smckit provides

- **One unified API** modeled on scanpy: `smckit.io` for reading data,
  `smckit.tl` for inference tools, `smckit.pl` for plotting.
- **Dual implementation paths** for methods over time: upstream for fidelity,
  native for long-term maintenance and acceleration.
- **A central data container** ({class}`smckit.SmcData`) that carries input
  sequences, parameters, and results through the analysis pipeline.
- **JIT-accelerated kernels** via Numba, with gradient-based variants via JAX
  for the SSM extensions. The Numba PSMC implementation is roughly 17% faster
  than the original C reference on real data.
- **Extensible state-space model framework** ({mod}`smckit.ext.ssm`) for
  building new SMC-flavored models by composing transition and emission
  components.
- **Reproducible by design**: validated against the original implementations
  with numerical correctness tests.

## Hello, smckit

A complete PSMC analysis is three lines of Python:

```python
import smckit

data = smckit.io.read_psmcfa("sample.psmcfa")
data = smckit.tl.psmc(data, pattern="4+5*3+4", n_iterations=25)
smckit.pl.demographic_history(data)
```

The pattern is the same for every method: read data into an `SmcData`
container, run an inference tool that mutates the container in place, then
plot or further analyze the results.

## Where to go next

- **[Installation](installation.md)** — install smckit and optional
  dependencies.
- **[Quickstart: PSMC](quickstart-psmc.md)** — run your first PSMC analysis
  end-to-end.
- **[Quickstart: ASMC](quickstart-asmc.md)** — decode pairwise coalescence
  times for phased haplotypes.
- **[Quickstart: MSMC2](quickstart-msmc2.md)** — run the in-progress MSMC2
  reimplementation on `multihetsep` input.
- **[Quickstart: MSMC-IM](quickstart-msmc-im.md)** — fit a two-population
  isolation-migration model from MSMC2 output.
- **[Quickstart: eSMC2](quickstart-esmc2.md)** — estimate demography with
  optional dormancy and selfing.
- **[Quickstart: SMC++](quickstart-smcpp.md)** — infer `N_e(t)` from span-encoded
  SMC++ input.
- **[Quickstart: diCal2](quickstart-dical2.md)** — run a diCal2-style analysis
  from native `.param` / `.demo` / `.config` files and a VCF.
- **[The SmcData container](../guide/smcdata.md)** — understand the central
  data structure.
- **[Choosing a method](../guide/choosing-a-method.md)** — pick the right
  SMC method for your data.
