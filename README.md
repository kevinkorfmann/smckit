# smckit

[![PyPI version](https://img.shields.io/pypi/v/smckit.svg)](https://pypi.org/project/smckit/)
[![Python versions](https://img.shields.io/pypi/pyversions/smckit.svg)](https://pypi.org/project/smckit/)
[![PyPI status](https://img.shields.io/pypi/status/smckit.svg)](https://pypi.org/project/smckit/)
[![Publish](https://github.com/kevinkorfmann/smckit/actions/workflows/publish.yml/badge.svg?branch=main)](https://github.com/kevinkorfmann/smckit/actions/workflows/publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Extras: all-models](https://img.shields.io/badge/extra-all--models-0A7BBB)](https://pypi.org/project/smckit/)
[![Extra: asmc](https://img.shields.io/badge/extra-asmc-0A7BBB)](https://pypi.org/project/smckit/)
[![Extra: jax](https://img.shields.io/badge/extra-jax-0A7BBB)](https://pypi.org/project/smckit/)
[![Extra: dev](https://img.shields.io/badge/extra-dev-0A7BBB)](https://pypi.org/project/smckit/)
[![Extra: docs](https://img.shields.io/badge/extra-docs-0A7BBB)](https://pypi.org/project/smckit/)

Unified framework for Sequentially Markovian Coalescent methods.

PSMC, MSMC/MSMC2, SMC++, diCal2, ASMC, and related workflows in one preservation-first toolkit.

smckit keeps upstream tools runnable from the same repository while building native implementations that do not lose contact with upstream truth.

## Why smckit

- Upstream-first: original tools in `vendor/` remain the oracle implementation.
- Provenance-aware: results can record which tool ran and how it was bootstrapped.
- One API surface: shared data structures and workflows across multiple SMC methods.
- Native path: performance-oriented in-repo implementations can grow without pretending parity before it is validated.

## Installation

```bash
pip install smckit
```

Install contract:

- `pip install smckit` guarantees the packaged example data and the documented
  native quickstarts.
- Full `implementation="upstream"` preservation workflows are a source-checkout
  feature unless the method page explicitly says otherwise.
- When an upstream runtime is missing, smckit reports platform-specific install
  commands and, when needed, tells you to switch to a source checkout.

Repo environment:

```bash
uv sync --extra dev --extra docs
pixi run test-fast
```

Use `uv` for the Python package environment and `pixi` for the repo-level
cross-language task environment.

Model extras:

```bash
pip install "smckit[all-models]"
pip install "smckit[asmc]"
pip install "smckit[psmc,msmc2,msmc_im,smcpp,esmc2,dical2]"
```

These extras describe intent. Today only `asmc` adds an extra Python runtime;
the other upstream-backed methods still need platform runtimes such as Java, R,
or a D/C toolchain. When an upstream runtime is missing, smckit now returns
platform-specific install commands for macOS, Linux, or Windows.

Development extras:

```bash
pip install "smckit[jax]"
pip install "smckit[dev]"
pip install "smckit[docs]"
```

## Philosophy

smckit is preservation-first by design.

- `vendor/` holds the original source or release artifacts that act as the oracle.
- `implementation="upstream"` means "run the original tool" when that bridge is wired and ready.
- `implementation="native"` means "run the in-repo implementation".
- `implementation="auto"` selects native only for capabilities that have passed
  their promotion gate; otherwise it falls back to upstream and records why.

The repository exposes `smckit.upstream.status()` and `smckit.upstream.bootstrap()` so upstream readiness is inspectable rather than implicit.

The command line exposes the same contract:

```bash
smckit methods
smckit status
smckit run psmc sample.psmcfa --implementation auto
smckit upstream psmc --output-dir results/psmc -- -N 25 sample.psmcfa
```

## Status

smckit is beta software with a mature preservation and validation foundation,
but it is not yet the 1.0 publication release.

- PSMC, ASMC, MSMC2, MSMC-IM, eSMC2, and SMC++ have promoted native
  capabilities while exact upstream execution remains available.
- PSMC+—the extension for background selection and local genomic-rate
  heterogeneity—passes its complete 12-case native parity matrix on Linux
  x86-64 and macOS ARM64. Frozen native fit/decode benchmarks are faster with
  lower peak memory on both platforms; empirical validation remains before
  `auto` switches to native.
- diCal2 has broad native structured-model oracle coverage, but its repaired
  PAC default, empirical validation, and broader cross-platform performance
  still block native-by-default promotion.
- PHLASH is integrated through its maintained external package with normalized
  posterior and credible-interval results; the frozen full validation run is
  pending.
- A callability-aware VCF-to-multihetsep converter and an accession-recorded
  NA12878/1000 Genomes source manifest establish the human empirical input
  contract. CRAM-derived callability and retained empirical runs are next.

See the [full project status](docs/get-started/project-status.md) and
[runtime/resource guide](docs/guide/runtime-estimates.md) for precise evidence,
caveats, and remaining gates.

## Repository Guide

- Method docs: `docs/agents/algorithms.md`
- Project status: `docs/get-started/project-status.md`
- Runtime estimates: `docs/guide/runtime-estimates.md`
- Usage notes: `docs/agents/using-smckit.md`
- Developer architecture: `docs/developer/architecture.md`

## License

MIT. See [LICENSE](LICENSE).
