# PSMC

PSMC is the simplest and most established SMC method in smckit. It infers a
population-size history from one diploid genome by reading a windowed
heterozygosity sequence and fitting a hidden Markov model over discretized
coalescent times.

```{admonition} Best for
:class: tip
One diploid whole-genome sequence, a standard demographic-history workflow,
and a method that is already a strong native reference point inside smckit.
```

## What it gives you

- `N_e(t)` through time
- `theta`, `rho`, and grouped `lambda` parameters
- per-iteration EM diagnostics in `data.results["psmc"]["rounds"]`

## Implementations

| Selector | Status | Notes |
|---|---|---|
| `implementation="native"` | Available | Main in-repo implementation and current default-resolved path. |
| `implementation="upstream"` | Not yet exposed | Public upstream bridge is planned but not wired today. |
| `implementation="auto"` | Available | Currently resolves to `native` for PSMC. |

## Input

PSMC reads `.psmcfa` files through {func}`smckit.io.read_psmcfa`.

A `.psmcfa` file is a FASTA-like sequence where each character summarizes one
window of the genome:

- homozygous / no heterozygosity observed
- heterozygous / at least one variant observed
- missing / filtered out

Use the [I/O formats guide](../guide/io-formats.md) for the format details.
An example file in the project repository:

- `tests/data/NA12878_chr22.psmcfa`
- `https://github.com/kevinkorfmann/smckit/blob/main/tests/data/NA12878_chr22.psmcfa`

## Minimal workflow

```python
import smckit

data = smckit.io.read_psmcfa("tests/data/NA12878_chr22.psmcfa")
data = smckit.tl.psmc(
    data,
    pattern="4+5*3+4",
    n_iterations=25,
    implementation="native",
)

res = data.results["psmc"]
print(res["theta"], res["rho"])
smckit.pl.demographic_history(data)
```

## Interpreting the result

- The recent and ancient extremes of the curve are the least trustworthy.
- The `pattern` changes how strongly neighboring time intervals are tied
  together.
- `N_e(t)` is an effective-population-size summary, not census size.

For interpretation guidance, see [Interpreting results](../guide/interpreting-results.md).

## Current parity snapshot

The native PSMC port tracks the bundled C reference fixture closely:

- lambda correlation: `0.9999223`
- lambda max relative error: `9.54e-03`
- theta relative error: `1.83e-03`
- rho relative error: `1.50e-03`

See [parity notes](../developer/parity.md) for the fixture details.

## Learn more

- [Quickstart: PSMC](../get-started/quickstart-psmc.md)
- [I/O formats](../guide/io-formats.md)
- [PSMC internals](../developer/internals-psmc.md)
