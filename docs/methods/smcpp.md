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
| `implementation="native"` | Available | Defaults to the upstream-style one-pop interpretation and preprocessing path, and now clears the tracked one-pop parity matrix against upstream. |
| `implementation="upstream"` | Available | Runs the vendored upstream source through the controlled side environment. |
| `implementation="auto"` | Available | Currently prefers upstream when the side environment exists. |

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

Upstream example directory:

- `https://github.com/popgenmethods/smcpp/tree/master/example`

## Minimal workflow

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

res = data.results["smcpp"]
print(res["implementation"], res["log_likelihood"])
smckit.pl.demographic_history(data, method="smcpp")
```

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
- Upstream remains the fidelity baseline for broader validation and for
  untracked fixtures; the tracked matrix should not be read as a blanket claim
  of parity for every possible future SMC++ input family.
- The old one-distinguished native path still exists for explicit compatibility
  cases, but it is no longer the default interpretation of headerless input.
- The method is more data-hungry and computationally heavier than PSMC.

## Learn more

- [Quickstart: SMC++](../get-started/quickstart-smcpp.md)
- [Choosing a method](../guide/choosing-a-method.md)
- [SMC++ internals](../developer/internals-smcpp.md)
- [SMC++ parity closure notes](../developer/smcpp-parity-closure.md)
