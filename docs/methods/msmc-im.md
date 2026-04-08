# MSMC-IM

MSMC-IM takes MSMC or MSMC2 two-population output and fits a continuous
isolation-migration model. It is a post-processing step, not a direct raw-read
or VCF inference method.

```{admonition} Best for
:class: tip
Two-population MSMC/MSMC2 analyses where you want split-time and migration
summaries instead of just raw coalescence-rate curves.
```

## What it gives you

- corrected `N1(t)` and `N2(t)`
- symmetric migration rate `m(t)`
- cumulative migration summary `M(t)`
- split-time quantiles from `M(t)`

## Implementations

| Selector | Status | Notes |
|---|---|---|
| `implementation="native"` | Available | Validated against a 3-case vendored oracle matrix on the tracked Yoruba/French input. |
| `implementation="upstream"` | Available | Runs the vendored `MSMC_IM.py` script directly. |
| `implementation="auto"` | Available | Prefers `upstream` when the bridge is ready. |

## Input

MSMC-IM expects a combined MSMC/MSMC2 output file with within- and
cross-population coalescence rates. This is the
`.combined.msmc2.final.txt` style file read by
{func}`smckit.io.read_msmc_combined_output`.

Example file already vendored in the repository:

- `vendor/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt`
- `https://github.com/kevinkorfmann/smckit/blob/main/vendor/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt`

## Minimal workflow

```python
import smckit

data = smckit.tl.msmc_im(
    "vendor/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt",
    pattern="1*2+25*1+1*2+1*3",
    mu=1.25e-8,
    beta=(1e-8, 1e-6),
    implementation="native",
)

res = data.results["msmc_im"]
print(res["split_time_quantiles"])
```

## Practical notes

- The `pattern` must match the MSMC/MSMC2 run that produced the input file.
- This is a fitted demographic summary of MSMC output, not a separate raw-data
  model.
- The native implementation is regression-tested against the vendored
  Yoruba/French oracle output with a small 3-case matrix rather than tracked
  with loose correlation bounds.

## Learn more

- [Quickstart: MSMC-IM](../get-started/quickstart-msmc-im.md)
- [I/O formats](../guide/io-formats.md)
- [MSMC-IM internals](../developer/internals-msmc-im.md)
