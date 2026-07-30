# SMC++

SMC++ is the many-sample member of the smckit family. It is designed for
datasets with many unphased genomes and uses a distinguished-lineage HMM plus
site-frequency information to recover `N_e(t)` with better recent-time
resolution than pairwise methods.

```{admonition} Best for
:class: tip
Dozens to hundreds of unphased diploid genomes from one population, especially
when PSMC is too sample-limited at recent times.
```

## What it gives you

- population-size history `N_e(t)`
- fitted `theta`, `rho`, and `n0`
- optimization diagnostics in `data.results["smcpp"]["optimization"]`

## Implementations

| Selector | Status | Notes |
|---|---|---|
| `implementation="native"` | Available | Covers the upstream-style one-pop path and exact two-population clean-split fitting. The split path is available but remains unpromoted while its broader oracle/performance matrix is completed. |
| `implementation="upstream"` | Available | Runs the vendored upstream source through the controlled side environment, including two-population split inference. |
| `implementation="auto"` | Available | Uses native for promoted one-population workflows and routes two-population split requests to upstream. |

Install contract:

- wheel install: native SMC++ quickstart is supported on the packaged tiny `.smc.gz` example
- source checkout plus side environment: required for full upstream SMC++ workflow

## Input

SMC++ uses `.smc` or `.smc.gz` span-encoded inputs through
{func}`smckit.io.read_smcpp_input`.

That file stores long monomorphic stretches compactly rather than one genomic
site per row. It is the right format when you have many unphased individuals
and want SMC++ rather than a pairwise SMC method.

smckit now ships a tiny packaged quickstart fixture:

- `smckit.io.example_path("smcpp/example.smc.gz")`

If the file does not contain an SMC++ header, smckit now defaults to the
upstream one-pop assumption of two distinguished haplotypes rather than the
older one-distinguished surrogate path.

## Recommended starting call

```python
import smckit

data = smckit.io.read_smcpp_input(smckit.io.example_path("smcpp/example.smc.gz"))
data = smckit.tl.smcpp(
    data,
    n_intervals=4,
    max_iterations=1,
    regularization=10.0,
    mu=1.25e-8,
    generation_time=25.0,
    implementation="native",
)
```

SMC++ is more data-hungry than the pairwise methods. The packaged example is a
smoke-test-scale fixture, not a realistic data volume.

## How to think about the arguments

### Common workflow controls

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `implementation` | Choose `native`, `upstream`, or `auto`. | Force `upstream` when you want the vendored tool in its side environment. | `auto` uses promoted native one-population workflows. |
| `backend` | Deprecated alias for `implementation`. | Only for legacy code. | Prefer `implementation`. |
| `upstream_options` | Extra bridge controls for the upstream path. | Only when reproducing an upstream workflow. | Leave alone first. |
| `native_options` | Extra controls for the native path. | Not part of routine use today. | Leave as `None`. |

### Model size and fit duration

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `n_intervals` | Number of demographic intervals in the fitted history. | Increase for a finer history; decrease for a simpler one. | More intervals can overfit weak data. |
| `max_iterations` | Maximum optimization iterations. | Increase if the optimizer is still making progress. | Small values are fine for smoke tests, not for real analyses. |

### Time-grid and scaling controls

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `max_t` | Controls how far back the time grid extends in model units. | Change if you want to emphasize deeper or shallower history. | Leave at the default first. |
| `alpha` | Controls time-grid spacing. | Change only when you intentionally want a different grid shape. | Usually leave alone. |
| `mu` | Mutation rate used for absolute scaling. | Set for your organism. | Strongly affects absolute time and size. |
| `recombination_rate` | Recombination rate assumption used by the model. | Change for your organism or experiment. | Important biological input, not just a technical knob. |
| `generation_time` | Converts generations to years. | Change for species-specific scaling. | Same logic as other demographic methods. |

### Optimization controls

| Argument | What it means | When to change it | Default guidance |
|---|---|---|---|
| `regularization` | Smoothness penalty on the inferred history. | Increase to discourage noisy curves; decrease to allow more flexibility. | This is one of the first tuning knobs to learn. |
| `seed` | Random seed for reproducibility. | Set for repeatable runs. | Good practice for examples and comparisons. |
| `initial_model` | Reloadable SMC++ model path, mapping, or `SmcData`. | Resume from or compare a frozen history. | Omit for data-driven initialization. |
| `split_models` | Two fitted marginal histories for a joint two-population input. | Fit a clean split after fitting each population. | Required only for split analysis; choose `native` explicitly until the split path is promoted. |
| `output_prefix` | Prefix for normalized result and model JSON. | Use for reproducible analyses. | Artifacts are SHA-256 recorded in provenance. |

## VCF preparation, masks, and multi-population files

{func}`smckit.pp.smcpp_from_vcf` converts plain or gzip-compressed VCF input
to `.smc`/`.smc.gz`. It supports one or two disjoint population sample lists,
explicit distinguished haplotypes, 0-based half-open BED masks, conservative
long-gap missingness, and deterministic headers. A frozen indexed-VCF oracle
test requires the native observation stream to match preserved upstream
`vcf2smc` exactly.

{func}`smckit.io.read_smcpp_input` and
{func}`smckit.io.write_smcpp_input` preserve every population triplet. Native
two-population clean-split inference now uses an independently implemented,
deterministic joint-SFS calculation. It fits the same shared marginal-history
scale coordinate followed by split time used by the preserved workflow. The
tracked oracle covers the five serialized upstream spline classes: Piecewise,
CubicSpline, PChipSpline, AkimaSpline, and BSpline.

Use `implementation="native"` to exercise that unpromoted path explicitly, or
`implementation="auto"` to retain the preserved upstream fallback:

```python
joint = smckit.io.read_smcpp_input("joint-populations.smc.gz")
joint = smckit.tl.smcpp(
    joint,
    implementation="native",
    split_models=("population-a.model.json", "population-b.model.json"),
    output_prefix="results/joint",
)
print(joint.results["smcpp"]["split_years"])
```

This writes `results/joint.smcpp.split.model.json` with the reloadable
two-population model and `results/joint.smcpp.split.json` with normalized
population histories, split time, provenance, and hashes. Select
`implementation="upstream"` to execute the original split implementation
without native substitution.

## Cross-validation

{func}`smckit.tl.smcpp_cross_validate` mirrors upstream contig-level
cross-validation. Whole records are assigned to folds, candidate
regularization penalties are fitted on the remaining records, held-out HMM log
likelihood selects the penalty, and the winner is refitted to all records.

## Models and figures

{func}`smckit.io.write_smcpp_model` writes a versioned normalized model with an
upstream-readable `SMCModel` block. The same file can be passed back through
`initial_model`. `output_prefix` writes both model and complete result JSON.

Use {func}`smckit.pl.smcpp_demographic_history` for the inferred history and
{func}`smckit.pl.smcpp_cross_validation_scores` for fold-level model-selection
evidence. Vector PDF/SVG/EPS and 600-dpi PNG/TIFF export are supported.

## What comes back in `data.results["smcpp"]`

Common fields to inspect:

- `implementation`
- `theta`
- `rho`
- `n0`
- `ne`
- `time` and `time_years`
- `log_likelihood`
- `optimization`

The `optimization` payload is especially useful because SMC++ is not just a
simple EM loop; it uses a heavier optimization stack than PSMC.

## How to tell if the run behaved sensibly

- Check whether the optimizer is still improving at the final iteration.
- Watch for very jagged histories if `regularization` is too small.
- Confirm the input interpretation is the one you intended, especially for
  headerless `.smc` files.

## Common confusion points

- SMC++ is not just “PSMC with more samples.” The input representation and
  likelihood are genuinely different.
- `regularization` is a meaningful scientific tuning knob because it controls
  how wiggly the fitted history can be.
- The bundled example is tiny; do not use its runtime or apparent stability as
  a proxy for real data behavior.

## Practical notes

- `implementation="auto"` is the safest choice when the upstream side
  environment is configured.
- The default native path now uses upstream-style one-pop preprocessing,
  hidden-state construction, upstream observation scaling for binned data, and
  an EM/coordinate-update optimizer with the upstream-style global scale step.
- The tracked one-pop parity matrix now includes both the strict small control
  fixture and the bundled larger `.smc` fixture, and native clears both at
  `log_corr >= 0.999` with near-unity scale ratio.
- Fixed-model one-pop `gamma0`, `xisum`, and log-likelihood also now match the
  upstream HMM on that same tracked matrix, so the native and upstream
  one-pop paths are interchangeable for the enforced fixtures shown in docs.
- The native clean-split path matches the preserved optimizer's split time and
  shared scale on its tracked end-to-end fixture. Its exact deterministic
  joint-CSFS agrees with the preserved raw tensor within the upstream
  Monte-Carlo estimator's sampling error.
- Upstream remains the fidelity baseline for broader validation and for
  untracked fixtures; the tracked matrix should not be read as a blanket claim
  of parity for every possible future SMC++ input family.

## Learn more

- [Quickstart: SMC++](../get-started/quickstart-smcpp.md)
- [Choosing a method](../guide/choosing-a-method.md)
- [Interpreting results](../guide/interpreting-results.md)
- [SMC++ internals](../developer/internals-smcpp.md)
- [SMC++ parity closure notes](../developer/smcpp-parity-closure.md)
