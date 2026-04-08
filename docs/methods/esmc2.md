# eSMC2

eSMC2 extends pairwise SMC inference to species where dormancy and
self-fertilization matter. It keeps the PSMC-style one-genome workflow but
adds ecological parameters on top of demographic history.

```{admonition} Best for
:class: tip
One diploid genome from a species where seed banking or selfing is biologically
important and you need more than a plain PSMC history.
```

## What it gives you

- `N_e(t)`
- germination / dormancy parameter `beta`
- selfing parameter `sigma`
- fitted `Xi`, `rho`, and per-iteration diagnostics

## Implementations

| Selector | Status | Notes |
|---|---|---|
| `implementation="native"` | Available | In-repo Python/Numba implementation with tracked fit parity on the current oracle fixture matrix. |
| `implementation="upstream"` | Available | Runs the vendored R implementation through `Rscript`. |
| `implementation="auto"` | Available | Currently prefers upstream when the R environment is configured. |

## Input

eSMC2 can read the same pairwise-style data you use for PSMC:

- `.psmcfa` via {func}`smckit.io.read_psmcfa`
- multihetsep-derived pairwise sequences via {func}`smckit.io.read_multihetsep`

Example pairwise input in this repository:

- `tests/data/NA12878_chr22.psmcfa`
- `https://github.com/kevinkorfmann/smckit/blob/main/tests/data/NA12878_chr22.psmcfa`

If you are not sure what a `.psmcfa` file represents, start with the
[I/O formats guide](../guide/io-formats.md).

## Minimal workflow

```python
import smckit

data = smckit.io.read_psmcfa("tests/data/NA12878_chr22.psmcfa")
data = smckit.tl.esmc2(
    data,
    n_states=20,
    n_iterations=20,
    estimate_beta=True,
    beta=0.8,
    mu=1.25e-8,
    generation_time=1.0,
    implementation="auto",
)

res = data.results["esmc2"]
print(res["implementation"], res["beta"], res["sigma"])
```

## Practical notes

- Dormancy and selfing can be hard to separate from sequence data alone.
- `implementation="auto"` is preferred when you have the upstream R
  environment available.
- Native/upstream fit parity is enforced on the current oracle fixture matrix
  for fixed-rho, rho-redo, beta, sigma, beta+sigma, and grouped
  `pop_vect=[3,3]` runs.
- Upstream remains the safer choice outside that tracked matrix, especially for
  multi-sequence inputs and broader grouped-`Xi` layouts.

## Learn more

- [Quickstart: eSMC2](../get-started/quickstart-esmc2.md)
- [Interpreting results](../guide/interpreting-results.md)
- [eSMC2 internals](../developer/internals-esmc2.md)
