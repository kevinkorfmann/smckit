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
| `implementation="native"` | Available | Public path today, but upstream remains the fidelity baseline for README-style fixtures. |
| `implementation="upstream"` | Available | Runs the vendored `diCal2.jar` through the Java-backed upstream bridge. |
| `implementation="auto"` | Available | Prefers `upstream` when path-backed inputs and the Java runtime are ready. |

Install contract:

- wheel install: native diCal2 quickstart is supported with the packaged example bundle
- source checkout: required for vendored `diCal2.jar` upstream workflow
- additional Java runtime: still required whenever you explicitly ask for `implementation="upstream"`

## Input

diCal2 does not start from a single file. A normal run uses:

- a mutation/recombination parameter file such as `.param`
- a demographic model file such as `.demo`
- sometimes a `.rates` file
- a `.config` file describing sample assignments
- sequence data, often from VCF plus a reference

Packaged quickstart files:

- `smckit.io.example_path("dical2/test.param")`
- `smckit.io.example_path("dical2/exp.demo")`
- `smckit.io.example_path("dical2/exp.config")`
- `smckit.io.example_path("dical2/test.vcf")`

If those files are unfamiliar, read [I/O formats](../guide/io-formats.md)
before trying to run the method.

## Recommended starting call

```python
import smckit

data = smckit.io.read_dical2(
    sequences=smckit.io.example_path("dical2/test.vcf"),
    param_file=smckit.io.example_path("dical2/test.param"),
    demo_file=smckit.io.example_path("dical2/exp.demo"),
    config_file=smckit.io.example_path("dical2/exp.config"),
    reference_file=smckit.io.example_path("dical2/test.fa"),
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

diCal2 has the densest public signature in the current smckit surface. For
most users, the safest rule is: change only the common controls first, and
leave the meta-start and bounds machinery alone until you know why you need it.

## How to think about the arguments

### Common workflow controls

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `implementation` | Choose `native`, `upstream`, or `auto`. | Force `upstream` when you want the vendored Java tool. | `auto` prefers upstream when all required path-backed inputs are present and Java is ready. |
| `upstream_options` | Extra bridge controls for the upstream CLI path. | Use only when reproducing a specific upstream command. | Leave as `None` first. |
| `native_options` | Extra controls for the native implementation. | Use only when you intentionally need diCal2-specific advanced controls. | Not a first-pass knob. |

### Core demographic grid and fit controls

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `n_intervals` | Number of refined coalescent time intervals. | Change when you want a different time-grid resolution. | Common control. |
| `max_t` | Maximum time depth in model units. | Change when you want deeper or shallower ancient coverage. | Common control. |
| `alpha` | Controls time resolution near the present. | Change only when you intentionally want a different interval layout. | Usually leave alone first. |
| `n_em_iterations` | Number of EM iterations per start point. | Increase for harder fits. | Common control. |
| `composite_mode` | Composite-likelihood scheme such as `pac`, `pcl`, or `lol`. | Change when your intended diCal2 workflow depends on a specific scheme. | `pac` is the standard starting point. |
| `loci_per_hmm_step` | Groups loci into one HMM step. | Change when matching an upstream run or trading detail for speed. | Usually leave at `1` first. |

### Biological scaling

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `mu` | Mutation rate for absolute scaling. | Set for your organism. | Affects absolute times and sizes. |
| `generation_time` | Converts generations to years. | Change for your organism. | Same scaling caveat as other methods. |
| `n_ref` | Reference effective population size for scaling. | Set only when you want to override the value derived from `theta` and `mu`. | Most users should leave it as `None`. |

### Start-point and meta-start controls

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `start_point` | Explicit initial values for placeholder parameters. | Use when you already know the parameter ordering and want one controlled start. | Advanced. |
| `meta_start_file` | File containing multiple candidate starts. | Use when reproducing a multi-start diCal2 search. | Advanced. |
| `meta_num_iterations` | Number of meta-start generations to run. | Increase only if you intentionally want iterative multi-start search. | Advanced. |
| `meta_keep_best` | Number of best start points retained between generations. | Change when tuning the breadth of multi-start search. | Advanced. |
| `meta_num_points` | Number of points evaluated per meta-start generation. | Change when tuning multi-start breadth. | Advanced. |

### Search constraints and reproducibility

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `bounds` | Parameter bounds in diCal2 placeholder order. | Use when the demographic model requires explicit constraints. | Advanced and easy to misuse. |
| `seed` | Random seed. | Set for reproducibility. | Good practice, though not the main science knob. |

## What comes back in `data.results["dical2"]`

Common fields to inspect:

- `implementation`
- `log_likelihood`
- `best_params`
- `ordered_params`
- demographic arrays such as `pop_sizes`
- `time`
- EM or meta-start diagnostics

With diCal2, the fitted named parameters are often more important than any one
plotted summary curve.

## How to tell if the run behaved sensibly

- Confirm the file bundle is internally consistent before blaming the optimizer.
- Check whether repeated starts land in similar regions of parameter space.
- Be skeptical of one apparently “best” run if the meta-start search is narrow.
- Prefer the upstream path when the result is important and you need the
  preservation-first baseline.

## Common confusion points

- diCal2 is model-driven. The `.demo` and `.param` setup matters as much as the
  numeric optimizer settings.
- `start_point`, `meta_start_file`, and `bounds` are powerful but advanced.
  They are not routine first-pass arguments.
- The native and upstream paths now expose aligned public result fields, but
  they are not yet interchangeable on every search path.

## Current parity snapshot

The upstream and native paths now share the same normalized public result
fields, including `best_params`, `ordered_params`, `time`, and demographic
arrays.

On the tracked README fixtures, parity is materially tighter than before:

- at the upstream best-fit parameter vector, the native fixed-point log-likelihood
  delta is now about `7.45e-4`
- replaying each explicit `exp.rand` start point now lands on the same endpoint
  to displayed precision, with log-likelihood deltas at or below about `2.21e-4`
- the full independent native searches now land on the same best-fit parameter
  vectors as upstream on both README `exp` and README `IM`
- at the upstream best-fit parameter vector on README `IM`, the native
  fixed-point log-likelihood delta is now about `1.63e-3`

The remaining gap is the objective value, not the search winner. Native and
upstream now agree on the README `exp` and `IM` best-fit parameter vectors, but
the native reported log-likelihood is still slightly offset at those same
points, so the method is closer to interchangeable than before without yet
fully reaching it.

## Learn more

- [Quickstart: diCal2](../get-started/quickstart-dical2.md)
- [I/O formats](../guide/io-formats.md)
- [Interpreting results](../guide/interpreting-results.md)
- [Parity notes](../developer/parity.md)
- [Developer parity notes](../developer/internals-dical2.md)
