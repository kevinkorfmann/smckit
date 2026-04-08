# diCal2

diCal2 is the structured-demography tool in smckit. Instead of fitting a
generic population-size curve, it works from an explicit demographic model
specification and optimizes named size, growth, and migration parameters.

```{admonition} Best for
:class: tip
Analyses that already live naturally in diCal2 `.param` / `.demo` / `.config`
style files and need explicit structured demographic parameters.
```

## What it gives you

- epoch-level population sizes and growth rates
- time grid and scaled demographic summaries
- EM/meta-start optimization diagnostics

## Implementations

| Selector | Status | Notes |
|---|---|---|
| `implementation="native"` | Available | Public path today. |
| `implementation="upstream"` | Not yet exposed | Public `diCal2.jar` execution path is not wired yet. |
| `implementation="auto"` | Available | Currently resolves to `native`. |

## Input

diCal2 does not start from a single magic file. A normal run uses:

- a mutation/recombination parameter file such as `.param`
- a demographic model file such as `.demo`
- sometimes a `.rates` file
- a `.config` file describing sample assignments
- sequence data, often from VCF plus a reference

Concrete example files in the repository:

- `vendor/diCal2/examples/fromReadme/test.param`
- `vendor/diCal2/examples/fromReadme/exp.demo`
- `vendor/diCal2/examples/fromReadme/exp.config`
- `vendor/diCal2/examples/fromReadme/test.vcf`

GitHub directory:

- `https://github.com/kevinkorfmann/smckit/tree/main/vendor/diCal2/examples/fromReadme`

If those files are unfamiliar, read [I/O formats](../guide/io-formats.md)
before trying to run the method.

## Minimal workflow

```python
import smckit

data = smckit.io.read_dical2(
    sequences="vendor/diCal2/examples/fromReadme/test.vcf",
    param_file="vendor/diCal2/examples/fromReadme/test.param",
    demo_file="vendor/diCal2/examples/fromReadme/exp.demo",
    config_file="vendor/diCal2/examples/fromReadme/exp.config",
    reference_file="vendor/diCal2/examples/fromReadme/test.fa",
    filter_pass_string=".",
)

data = smckit.tl.dical2(
    data,
    n_intervals=11,
    max_t=4.0,
    n_em_iterations=2,
    composite_mode="pac",
    implementation="native",
)
```

## Current parity snapshot

The native implementation is close on the bundled README-style fixtures but not
yet full upstream parity:

- `exp` example relative log-likelihood delta: `2.27%`
- best tested `IM` example relative log-likelihood delta: `1.22%`

## Learn more

- [Quickstart: diCal2](../get-started/quickstart-dical2.md)
- [Parity notes](../developer/parity.md)
- [I/O formats](../guide/io-formats.md)
