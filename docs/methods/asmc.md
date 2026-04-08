# ASMC

ASMC infers pairwise coalescence times along the genome for phased haplotype
pairs. Instead of producing one population-size curve, it gives you per-pair
and per-site TMRCA summaries that are useful for recent ancestry and selection
work.

```{admonition} Best for
:class: tip
Large phased-haplotype datasets where you want per-site pairwise coalescence
information rather than a single population-level history.
```

## What it gives you

- posterior state probabilities across sites
- per-pair posterior mean TMRCA
- optional per-pair MAP state calls
- shared decoding-quantity metadata

## Implementations

| Selector | Status | Notes |
|---|---|---|
| `implementation="native"` | Available | Main public path today. |
| `implementation="upstream"` | Not yet exposed | Public upstream bridge is planned but not wired. |
| `implementation="auto"` | Available | Currently resolves to `native`. |

## Input

ASMC needs two things:

1. a haplotype dataset prefix, which expands to `.hap.gz`, `.samples`, and
   `.map.gz`
2. a `.decodingQuantities.gz` file containing precomputed transition and
   emission tables

Concrete example files already vendored in the repository:

- `vendor/ASMC/ASMC_data/examples/asmc/exampleFile.n300.array.hap.gz`
- `vendor/ASMC/ASMC_data/examples/asmc/exampleFile.n300.array.samples`
- `vendor/ASMC/ASMC_data/examples/asmc/exampleFile.n300.array.map.gz`
- `vendor/ASMC/ASMC_data/decoding_quantities/30-100-2000_CEU.decodingQuantities.gz`

GitHub links:

- `https://github.com/kevinkorfmann/smckit/tree/main/vendor/ASMC/ASMC_data/examples/asmc`
- `https://github.com/kevinkorfmann/smckit/blob/main/vendor/ASMC/ASMC_data/decoding_quantities/30-100-2000_CEU.decodingQuantities.gz`

If these file roles are unfamiliar, read the [I/O formats guide](../guide/io-formats.md)
before running the method.

## Minimal workflow

```python
import smckit

data = smckit.io.read_asmc(
    "vendor/ASMC/ASMC_data/examples/asmc/exampleFile.n300.array",
    "vendor/ASMC/ASMC_data/decoding_quantities/30-100-2000_CEU.decodingQuantities.gz",
)

data = smckit.tl.asmc(
    data,
    pairs=[(0, 1), (2, 3)],
    mode="array",
    store_per_pair_map=True,
    implementation="native",
)
```

## Current parity snapshot

- MAP state agreement on the vendored fixture is about `99.84%`.
- Posterior means are close but not yet full promotion-level parity against
  upstream C++ output.

## Learn more

- [Quickstart: ASMC](../get-started/quickstart-asmc.md)
- [I/O formats](../guide/io-formats.md)
- [Parity notes](../developer/parity.md)
