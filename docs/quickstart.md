# Quick Start

## Running PSMC

```python
import smckit

# Read input data (PSMCFA format from samtools/bcftools)
data = smckit.io.read_psmcfa("NA12878_chr22.psmcfa")

# Run PSMC inference
data = smckit.tl.psmc(
    data,
    pattern="4+5*3+4",    # parameter grouping pattern
    n_iterations=25,       # EM iterations
    max_t=15.0,           # max coalescent time (2N0 units)
    tr_ratio=4.0,         # initial theta/rho ratio
    mu=1.25e-8,           # per-base mutation rate
    generation_time=25.0,  # years per generation
)

# Plot demographic history
smckit.pl.demographic_history(data)
```

## Accessing results

```python
res = data.results["psmc"]

res["theta"]       # estimated theta
res["rho"]         # estimated rho
res["ne"]          # N_e(t) array
res["time_years"]  # time points in years
res["lambda"]      # relative population sizes per state
res["rounds"]      # per-iteration EM diagnostics
```

## Comparing with C PSMC output

```python
# Load output from the original C implementation
rounds = smckit.io.read_psmc_output("sample.psmc")
final = rounds[-1]

print(f"theta: {final['theta']:.6f}")
print(f"rho:   {final['rho']:.6f}")
print(f"lambda: {final['lambda']}")
```
