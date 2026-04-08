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
| `implementation="native"` | Available | In-repo Python/Numba implementation with tracked fit parity across the public `.psmcfa` and `multihetsep` eSMC2 input families. |
| `implementation="upstream"` | Available | Runs the vendored R implementation through `Rscript`. |
| `implementation="auto"` | Available | Currently prefers upstream when the R environment is configured. |

Install contract:

- wheel install: native eSMC2 quickstart is supported
- source checkout: required for vendored-upstream eSMC2 R workflow
- additional R runtime: still required whenever you explicitly ask for `implementation="upstream"`

## Input

eSMC2 can read the same pairwise-style data you use for PSMC:

- `.psmcfa` via {func}`smckit.io.read_psmcfa`
- multihetsep-derived pairwise sequences via {func}`smckit.io.read_multihetsep`

Packaged quickstart fixture:

- `smckit.io.example_path("psmc/NA12878_chr22.psmcfa")`

If you are not sure what a `.psmcfa` file represents, start with the
[I/O formats guide](../guide/io-formats.md).

## Public parity matrix

The tracked native/upstream gate now covers the public eSMC2 input surface
rather than only one clean fixture.

- `.psmcfa`: one clean record, one record with missing sites, and multi-record inputs
- `multihetsep`: one pair, multiple pairs from one file, multiple files / chromosomes, and `skip_ambiguous=True` runs where ambiguous sites become missing
- grouped `pop_vect` fitting on a non-default multi-record `.psmcfa` case

The upstream bridge records which public family ran in
`data.results["esmc2"]["upstream"]`, together with `n_sequences`,
`sequence_lengths`, and `used_sequence_indices`. That provenance is important
because the vendored eSMC2 wrapper still drops very poor pairwise sequences
before fitting; smckit now records exactly which sequences survived that
vendor-side filter.

## Recommended starting call

```python
import smckit

data = smckit.io.read_psmcfa(smckit.io.example_path("psmc/NA12878_chr22.psmcfa"))
data = smckit.tl.esmc2(
    data,
    n_states=20,
    n_iterations=20,
    estimate_beta=True,
    beta=0.8,
    mu=1.25e-8,
    generation_time=1.0,
    implementation="native",
)
```

Use this as a starting point, not as a claim that `beta=0.8` is right for your
organism. The quickstart turns on `estimate_beta` so you can see the ecological
part of the model in action on the packaged example.

If the result matters more than convenience, prefer `implementation="upstream"`
when the vendored R workflow is available.

## What the function is fitting

eSMC2 fits the same broad demographic story as PSMC, then layers ecological
rates on top of it.

- `Xi` is the relative demographic history. It is the piecewise population-size
  shape before conversion into absolute `N_e(t)`.
- `rho` controls recombination. It changes how quickly the hidden genealogy is
  allowed to switch along the genome.
- `beta` controls germination versus dormancy. Smaller values imply stronger
  seed-bank style effects.
- `sigma` controls self-fertilization. Larger values imply more selfing.

These parameters can trade off against each other. A run that changes `beta`,
`sigma`, and `rho` all at once is asking the data to separate several similar
signals, so unstable ecological estimates are common on weak inputs.

## How to think about the arguments

### Common workflow controls

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `implementation` | Choose `native`, `upstream`, or `auto`. | Force `upstream` for fidelity-sensitive work or parity checks. | Use `native` for wheel-installed quickstarts. Use `auto` or `upstream` once your R environment is ready. |
| `backend` | Deprecated alias for `implementation`. | Only for old code you have not cleaned up yet. | Prefer `implementation`. |
| `upstream_options` | Extra controls for the upstream bridge. | Only when you are matching a specific upstream workflow. | Leave alone for normal use. |
| `native_options` | Extra controls for the native implementation. | Not part of the normal public surface today. | Leave as `None`. |

### Model size and run length

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `n_states` | Number of hidden time states in the HMM. | Increase when you want a finer time grid. | `20` is a good practical default. More states do not automatically mean more real resolution. |
| `n_iterations` | Number of optimization / EM-style refinement rounds. | Increase if the log-likelihood and fitted parameters are still moving. | Start with `20`. Use fewer only for smoke tests. |

### Scaling and baseline rates

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `mu` | Per-base per-generation mutation rate used for absolute scaling. | Set this to an organism-appropriate value. | This strongly affects absolute time and absolute `N_e`, not just presentation. |
| `generation_time` | Generations-to-years conversion. | Change when you want calendar years rather than generations. | If you do not know it well, be explicit that the time axis is uncertain. |
| `rho_over_theta` | Starting recombination-to-mutation ratio. | Change when your organism has a clearly different baseline ratio or when fits are unstable. | Usually leave at the default first. It is a start value, not the main scientific result. |

### Ecological parameters

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `estimate_beta` | Allow the model to fit `beta` instead of holding it fixed. | Turn on when dormancy / germination is biologically relevant and you want that estimated. | One of the first ecological knobs to try. |
| `beta` | Starting or fixed germination parameter. | Change when you have a plausible prior range or you are fixing `beta` on purpose. | If `estimate_beta=False`, this is treated as fixed. |
| `estimate_sigma` | Allow the model to fit `sigma`. | Turn on when selfing is biologically plausible and identifiable from your data. | More fragile than a plain demographic fit; inspect convergence carefully. |
| `sigma` | Starting or fixed selfing parameter. | Change when you want to encode a prior expectation or hold selfing fixed. | Leave at the default unless you have a reason. |
| `mu_b` | Mutation-rate modifier used in the dormancy-aware scaling formulas. | Change only if you know you need a non-default upstream-style ecological assumption. | Most users should leave this alone. Treat it as advanced. |

### Recombination fitting

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `estimate_rho` | Fit `rho` instead of keeping it at the starting ratio. | Turn off when you want a fixed-rho comparison or are debugging identifiability. | Leave on for normal runs. |
| `rho_penalty` | Extra penalty that discourages large movement in `rho`. | Use only when `rho` is clearly wandering and you are deliberately regularizing the fit. | Advanced. Leave at `0.0` unless you are tuning difficult fits. |

### Advanced grouping and search bounds

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `pop_vect` | Groups neighboring `Xi` states so they share fitted values. | Change when you intentionally want a grouped demographic history. | Advanced. Leave as `None` for the default layout. |
| `box_b` | Search range for fitted `beta`. | Only if you are deliberately constraining the optimizer. | Advanced. Most users should not touch it. |
| `box_s` | Search range for fitted `sigma`. | Same as `box_b`, but for selfing. | Advanced. Most users should not touch it. |
| `box_p` | Search range for demographic `Xi` parameters in transformed space. | Only when reproducing or stress-testing optimizer behavior. | Advanced. Leave alone. |
| `box_r` | Search range for `rho` in transformed space. | Only when diagnosing or constraining difficult fits. | Advanced. Leave alone. |
| `rp` | Penalty applied to the `Xi` parameters. | Only when you intentionally want extra smoothing or shrinkage. | Advanced. Leave at the default unless you are matching an upstream experiment. |

## What comes back in `data.results["esmc2"]`

After fitting, the main place to inspect is `data.results["esmc2"]`.

Common fields to look at first:

- `implementation`: which path actually ran
- `beta`: fitted or fixed dormancy / germination parameter
- `sigma`: fitted or fixed selfing parameter
- `rho`: fitted recombination parameter
- `Xi`: relative demographic parameters
- `ne`: absolute effective population size curve
- `time` and `time_years`: time grid in model units and calendar units
- `rounds`: per-iteration diagnostics
- `upstream`: raw upstream metadata and provenance when the vendored path ran

The quickest sanity check is usually:

```python
res = data.results["esmc2"]
print(res["beta"], res["sigma"], res["rho"])
print(res["ne"][:5])
print(res["rounds"][-1])
```

## How to tell if the run behaved sensibly

- Check whether the log-likelihood and major parameters stabilize over the last
  few rounds.
- If `beta`, `sigma`, and `rho` keep moving together, the ecological part of
  the fit may not be well identified.
- Treat the earliest and latest time bins as the weakest part of the curve,
  just as in PSMC-like methods.
- Remember that `mu` and `generation_time` control the absolute scaling. A
  biologically implausible time axis can be a scaling problem rather than a
  model problem.

## Common confusion points

- `beta` is not “the amount of dormancy” in a simple everyday sense. It is a
  fitted model parameter whose interpretation depends on the eSMC2 model.
- `estimate_beta=True` means “fit `beta`”; it does not mean your starting
  `beta` is ignored completely.
- `mu_b`, `box_*`, `rp`, and `rho_penalty` are advanced controls. Most users
  should not touch them on a first pass.
- Native/upstream fit parity is enforced across the public `.psmcfa` and
  `multihetsep` input families, plus the tracked fixed-rho, rho-redo, beta,
  sigma, beta+sigma, and grouped-state validation cases.
- The vendored path is still the fidelity baseline for broader vendor-surface
  experiments, especially unusual grouped-`Xi` layouts that are not yet in the
  tracked gate matrix.

## Learn more

- [Quickstart: eSMC2](../get-started/quickstart-esmc2.md)
- [Interpreting results](../guide/interpreting-results.md)
- [eSMC2 internals](../developer/internals-esmc2.md)
