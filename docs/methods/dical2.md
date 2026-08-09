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

Pass a list of VCF paths to treat chromosomes or contigs as independent HMM
contributions. `reference_file`, `bed_files`, and `vcf_offsets` accept either
one value reused for every VCF or one value per VCF. BED intervals follow the
upstream zero-based, half-open exclusion convention. If `reference_file` is
omitted, each VCF must provide a `##reference=file://` header. These controls
are preserved by the typed upstream bridge as comma-separated original CLI
arguments.

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
| `output_prefix` | Prefix for the result files written after a successful run. | Set when you want durable, hashed artifacts. | Writes `<prefix>.dical2.txt` and `<prefix>.dical2.json`. |
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

Generated starts and PAC-specific controls use `native_options` or
`upstream_options`. Both snake-case names and the original Java spellings are
accepted:

- `meta_num_start_points` / `metaNumStartPoints` generates multiple starts;
  add `meta_grid_start=True` / `metaGridStart=True` for a log-spaced grid,
  otherwise starts are sampled log-uniformly with the Java-compatible RNG
- `num_permutations` / `numPermutations` generates PAC permutations, while
  `permutation_files` / `permutationsFile` reads exact permutation rows
- `num_csds_per_permutation` / `numCsdsPerPerm` selects how many CSD trunk
  sizes contribute per permutation
- `different_permutations_per_contig` / `diffPermsPerChunk` uses independent
  generated permutations, or one supplied permutation file, for each contig

Native results record the exact resolved permutations and initialization mode
in `results["dical2"]["permutations"]` and
`results["dical2"]["initialization"]`.

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
- `artifacts` with paths and SHA-256 hashes when `output_prefix` is set

With diCal2, the fitted named parameters are often more important than any one
plotted summary curve.

The text artifact uses the original parser-compatible layout: log-likelihood,
elapsed seconds, ordered parameter values, and run identifier. Preserved
upstream runs retain the exact captured Java stdout; the JSON artifact records
the normalized result and provenance.

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

- at the upstream best-fit parameter vector, the native exponential-growth
  fixed-point log-likelihood delta is about `5.38e-11`
- replaying each explicit `exp.rand` start point now lands on the same endpoint
  to displayed precision, with log-likelihood deltas at or below about `2.21e-4`
- the full independent native searches now land on the same best-fit parameter
  vectors as upstream on both README `exp` and README `IM`
- at the upstream best-fit parameter vector on README `IM`, the native
  fixed-point log-likelihood delta is about `2.88e-9`
- independent simulated clean-split, migration-window, and three-population
  fixed points agree within `1.4e-6` total log-likelihood and within the frozen
  `1e-6` per-base criterion

The tracked objective values and search winners now agree, and both execution
paths produce original-parser-compatible text plus normalized JSON artifacts.
Native promotion still requires independent simulation families, remaining
feature-ledger closure, and the performance gate.

Parameterized instantaneous migration is preserved by the native demo reader
and refined into a stochastic pulse epoch. Pulse transitions are applied to
all native lineage-state surfaces, and the independent introgression
fixed-point now agrees with the Java EigenCore oracle within the frozen
`1e-5` total-likelihood tolerance. This closes fixed-point correctness only;
fitted introgression remains unpromoted because the native M-step is still
materially slower than Java.

The authoritative option-by-option status is recorded in
[the diCal2 feature ledger](../parity/dical2-feature-ledger.json). PAC
permutation controls and generated grid/random starts are implemented but
remain unpromoted. Direct Java checks now cover generated fixed-point and
one-step PAC EM, two file-backed per-contig permutation sets, and exact grid and
seeded-random start sequences, plus four passing independent structured fixed
points, including pulse introgression. Independent growth inference, fitted
structured/empirical breadth, and the performance gate are still missing.
Native parallel execution remains upstream-only; use exact upstream execution
when those workflows are scientifically consequential.

## Learn more

- [Quickstart: diCal2](../get-started/quickstart-dical2.md)
- [I/O formats](../guide/io-formats.md)
- [Interpreting results](../guide/interpreting-results.md)
- [Parity notes](../developer/parity.md)
- [Developer parity notes](../developer/internals-dical2.md)
