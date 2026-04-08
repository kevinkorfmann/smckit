# MSMC2

MSMC2 extends the pairwise SMC idea to multiple haplotypes. In practice, it is
the bridge between simple one-genome history inference and cross-population
coalescence analyses such as MSMC-IM.

```{admonition} Best for
:class: tip
Two to eight phased haplotypes when you need more recent-time resolution than
PSMC or you want the coalescence-rate outputs that feed MSMC-IM.
```

```{admonition} Current state
:class: warning
The native implementation is validated on strong fixed fixtures, but the public
upstream bridge is not exposed yet and the full original feature surface is still
growing.
```

## What it gives you

- piecewise coalescence-rate estimates
- population-size summaries derived from those rates
- per-iteration optimization diagnostics

## Implementations

| Selector | Status | Notes |
|---|---|---|
| `implementation="native"` | Available | Public path today. |
| `implementation="upstream"` | Not yet exposed | Public runner still needs wiring. |
| `implementation="auto"` | Available | Currently resolves to `native`. |

## Input

MSMC2 reads `.multihetsep` files through {func}`smckit.io.read_multihetsep`.

A `.multihetsep` file is a variant-centric format: each line describes one
segregating site, the distance from the previous site, and the observed alleles
for each haplotype.

Example file in this repository:

- `data/msmc2_test.multihetsep`
- `https://github.com/kevinkorfmann/smckit/blob/main/data/msmc2_test.multihetsep`

## Minimal workflow

```python
import smckit

data = smckit.io.read_multihetsep("data/msmc2_test.multihetsep")
data = smckit.tl.msmc2(
    data,
    time_pattern="1*2+25*1+1*2+1*3",
    n_iterations=20,
    mu=1.25e-8,
    generation_time=25.0,
    implementation="native",
)
```

## Current parity snapshot

Across the current fixed upstream-reference scenarios, the native implementation
stays very close to upstream:

- left-boundary max relative error: `4.15e-06`
- lambda max relative error: `2.45e-03`
- minimum lambda correlation: `0.999999865`
- maximum absolute log-likelihood delta: `4.75e-03`

## Learn more

- [Quickstart: MSMC2](../get-started/quickstart-msmc2.md)
- [MSMC-IM](msmc-im.md)
- [Parity notes](../developer/parity.md)
