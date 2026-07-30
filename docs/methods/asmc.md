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
| `implementation="native"` | Available | Implements promoted array and dense sequence decoding. |
| `implementation="upstream"` | Available | Executes the preserved ASMC binary or binding. |
| `implementation="auto"` | Capability-aware | Uses native for promoted array and sequence workflows. |

Install contract:

- wheel install: native ASMC quickstart is supported with packaged example data
- optional extra: `pip install "smckit[asmc]"` installs the published ASMC runtime on macOS/Linux
- source checkout: still required for vendored-upstream ASMC preservation workflows

## Input

ASMC needs two things:

1. a haplotype dataset prefix, which expands to `.hap.gz`, `.samples`, and
   `.map.gz`
2. a `.decodingQuantities.gz` file containing precomputed transition and
   emission tables

Packaged quickstart files:

- `smckit.io.example_prefix("asmc/exampleFile.n300.array", (".hap.gz", ".samples", ".map.gz"))`
- `smckit.io.example_path("asmc/30-100-2000_CEU.decodingQuantities.gz")`

If these file roles are unfamiliar, read the [I/O formats guide](../guide/io-formats.md)
before running the method.

## Recommended starting call

```python
import smckit

root = smckit.io.example_prefix(
    "asmc/exampleFile.n300.array",
    (".hap.gz", ".samples", ".map.gz"),
)
dq = smckit.io.example_path("asmc/30-100-2000_CEU.decodingQuantities.gz")

data = smckit.io.read_asmc(root, dq)
data = smckit.tl.asmc(
    data,
    pairs=[(0, 1), (2, 3)],
    mode="array",
    store_per_pair_map=True,
    implementation="native",
)
```

## How to think about the arguments

### Common workflow controls

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `implementation` | Choose `native`, `upstream`, or `auto`. | Use an explicit selector for oracle work. | `auto` is capability-aware. |
| `upstream_options` | Extra original CLI controls for the preserved bridge. | Use only for options not represented by typed arguments. | Leave as `None`. |
| `native_options` | Extra controls for the native decoder. | Not part of routine use today. | Leave as `None`. |

### What to decode

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `pairs` | Which haplotype pairs to decode. | Set it when you want a subset rather than all unique pairs. | Important for controlling runtime and output size. |
| `mode` | `"array"` uses the array-oriented compressed model, `"sequence"` uses the sequence-style model. | Change when your data type demands it. | `"array"` is the normal starting point for the packaged example. |

### Decoding behavior and output size

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `fold_data` | Use folded CSFS tables. | Change only if your upstream workflow expects otherwise. | Usually leave it at the default. |
| `skip_csfs_distance` | Minimum distance between CSFS sites. | Change when you are intentionally tuning how densely CSFS information is used. | Leave at `0.0` unless you know why to alter it. |
| `scaling_skip` | Apply scaling every this many positions. | Change only if you are tuning performance details. | Advanced. |
| `store_per_pair_posterior_mean` | Keep per-pair posterior mean TMRCA outputs. | Turn off only if you are trimming memory or output size. | Common and usually worth keeping. |
| `store_per_pair_map` | Keep per-pair MAP state calls. | Turn on when you need discrete pairwise TMRCA tracks. | Optional; can increase output size. |

## What comes back in `data.results["asmc"]`

Common fields to inspect:

- `implementation`
- `n_pairs_decoded`
- `sum_of_posteriors`
- `per_pair_posterior_mean` when stored
- `per_pair_map` when stored

ASMC is less about one summary curve and more about local pairwise ancestry
structure along the genome.

## How to tell if the run behaved sensibly

- Make sure the pair count and output array shapes match what you expected.
- Confirm you are using the right `mode` for the input and decoding quantities.
- Be deliberate about whether you need posterior means, MAP calls, or both.

## Common confusion points

- ASMC outputs pairwise TMRCA information, not a single global demographic
  history like PSMC or SMC++.
- `pairs=None` can explode output size on large haplotype panels because it
  decodes all unique pairs.
- `mode="array"` versus `mode="sequence"` is a data-model choice, not a style
  preference.

## Current parity snapshot

- On the vendored `n300` array oracle, MAP state agreement is 100% and the
  maximum posterior-mean relative error is `2.99e-4`.
- On the dense `n300` sequence oracle with CSFS emissions, interval decoding,
  and 0.5 cM burn-in, the maximum posterior-mean relative error is at most
  `1e-3` and MAP agreement is at least 99.9%; `auto` therefore uses native.

## Learn more

- [Quickstart: ASMC](../get-started/quickstart-asmc.md)
- [I/O formats](../guide/io-formats.md)
- [Interpreting results](../guide/interpreting-results.md)
- [Parity notes](../developer/parity.md)
