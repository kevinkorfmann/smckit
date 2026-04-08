# Plotting

smckit ships with a small plotting module ({mod}`smckit.pl`) for visualizing
SMC inference results. The current focus is the
{func}`~smckit.pl.demographic_history` function, which plots $N_e(t)$ as a
step function — the standard PSMC-style plot.

## Basic usage

```python
import smckit

data = smckit.io.read_psmcfa("sample.psmcfa")
data = smckit.tl.psmc(data, pattern="4+5*3+4", n_iterations=25)

ax = smckit.pl.demographic_history(data)
```

This produces a log-time x-axis, linear $N_e$ y-axis, with title
"Demographic History" and standard axis labels.

## Selecting a method

If your `SmcData` contains results from multiple methods, pick which one to
plot with the `method` argument:

```python
smckit.pl.demographic_history(data, method="psmc")
smckit.pl.demographic_history(data, method="esmc2")
smckit.pl.demographic_history(data, method="smcpp")
```

## Overlaying multiple methods

To compare methods on a single plot, pass the same `ax` repeatedly:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

smckit.pl.demographic_history(data, method="psmc",
                               ax=ax, label="PSMC", color="C0")
smckit.pl.demographic_history(data, method="esmc2",
                               ax=ax, label="eSMC2", color="C1")

ax.legend()
plt.show()
```

## Unit conversion

Time is converted from coalescent units ($2 N_0$ generations) to years using
the mutation rate `mu` and `generation_time`. By default these are read from
`data.params`, but you can override them per-call:

```python
smckit.pl.demographic_history(
    data,
    mu=2.5e-8,            # higher per-base rate
    generation_time=29.0, # human-like
)
```

If `data.results[method]["time_years"]` is already populated by the tool, it
takes precedence over the conversion.

## Axis scaling

Both axes can be log- or linear-scaled:

```python
# Log time, log Ne (the most common scientific view)
smckit.pl.demographic_history(data, log_x=True, log_y=True)

# Linear time (useful for very recent histories)
smckit.pl.demographic_history(data, log_x=False)
```

## Bootstrap confidence intervals

If you ran the inference with bootstrap replicates and stored them in
`data.results[method]`, set `show_bootstraps=True` to overlay thin bootstrap
lines:

```python
smckit.pl.demographic_history(data, show_bootstraps=True)
```

## Customizing with matplotlib kwargs

Any extra keyword arguments are passed through to
`matplotlib.axes.Axes.step`, so all standard matplotlib styling works:

```python
smckit.pl.demographic_history(
    data,
    label="NA12878",
    color="navy",
    linewidth=3,
    alpha=0.8,
    linestyle="--",
)
```

## See also

- The **[Gallery](gallery.md)** for example plots produced with this
  function.
- **[Interpreting results](interpreting-results.md)** for what the curves
  actually mean.
