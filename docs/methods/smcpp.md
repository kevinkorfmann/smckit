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
| `implementation="native"` | Available | Now defaults to the upstream-style one-pop interpretation and preprocessing path, but still falls short of promotion-level parity on larger fixtures. |
| `implementation="upstream"` | Available | Runs the vendored upstream source through the controlled side environment. |
| `implementation="auto"` | Available | Currently prefers upstream when the side environment exists. |

## Input

SMC++ uses `.smc` or `.smc.gz` span-encoded inputs through
{func}`smckit.io.read_smcpp_input`.

That file stores long monomorphic stretches compactly rather than one genomic
site per row. It is the right format when you have many unphased individuals
and want SMC++ rather than a pairwise SMC method.

The repository does not currently ship a small standalone `.smc.gz` fixture in
`tests/data`, so the format explanation lives in the [I/O guide](../guide/io-formats.md)
and the upstream-backed implementation remains the best reference behavior.

If the file does not contain an SMC++ header, smckit now defaults to the
upstream one-pop assumption of two distinguished haplotypes rather than the
older one-distinguished surrogate path.

Upstream example directory:

- `https://github.com/popgenmethods/smcpp/tree/master/example`

## Minimal workflow

```python
import smckit

data = smckit.io.read_smcpp_input("data.smc.gz")
data = smckit.tl.smcpp(
    data,
    n_intervals=8,
    mu=1.25e-8,
    generation_time=25.0,
    implementation="auto",
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
- On the tracked small one-pop fixture, the default native path now also
  matches the upstream trajectory closely enough to clear the `0.99`
  log-correlation gate.
- Native and upstream paths still do not have promotion-level parity on larger
  one-pop fixtures; use `implementation="upstream"` for fidelity-sensitive work.
- The old one-distinguished native path still exists for explicit compatibility
  cases, but it is no longer the default interpretation of headerless input.
- The method is more data-hungry and computationally heavier than PSMC.

## Learn more

- [Quickstart: SMC++](../get-started/quickstart-smcpp.md)
- [Choosing a method](../guide/choosing-a-method.md)
- [SMC++ internals](../developer/internals-smcpp.md)
