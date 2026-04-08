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

smckit is upstream-first by design.

- `vendor/` holds the original source or release artifacts that act as the oracle.
- `implementation="upstream"` means "run the original tool" when that bridge is wired and ready.
- `implementation="native"` means "run the in-repo implementation".
- `implementation="auto"` should prefer upstream fidelity whenever the upstream path is ready.

The repository exposes `smckit.upstream.status()` and `smckit.upstream.bootstrap()` so upstream readiness is inspectable rather than implicit.

## Status

smckit is early-stage software. The priority is preserving access to upstream methods, making readiness explicit, and validating native implementations against upstream behavior before claiming equivalence.

Recent parity progress:

- eSMC2 native/upstream interchangeability is now tracked across the public
  `.psmcfa` and `multihetsep` input families, including missing-site,
  multi-record, multi-pair, multi-file, and `skip_ambiguous=True` cases.
- The docs landing page and gallery summarize which methods have fixture-only
  validation and which now have broader public-surface parity gates.

## Repository Guide

- Method docs: `docs/agents/algorithms.md`
- Usage notes: `docs/agents/using-smckit.md`
- Developer architecture: `docs/developer/architecture.md`

## License

MIT. See [LICENSE](LICENSE).
