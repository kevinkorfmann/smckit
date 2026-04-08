# Quickstart: MSMC-IM

This quickstart fits MSMC-IM on the vendored Yoruba-French combined MSMC2
output shipped with the repository.

## What the input file is

MSMC-IM does not read raw genomes directly. It expects a combined MSMC/MSMC2
output file with:

- left and right time boundaries
- within-population coalescence rates
- cross-population coalescence rates

Example file:

- repository path: `vendor/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt`
- GitHub: `https://github.com/kevinkorfmann/smckit/blob/main/vendor/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt`

## Run MSMC-IM

```python
import smckit

data = smckit.tl.msmc_im(
    "vendor/MSMC-IM/example/Yoruba_French.8haps.combined.msmc2.final.txt",
    pattern="1*2+25*1+1*2+1*3",
    mu=1.25e-8,
    beta=(1e-8, 1e-6),
    implementation="native",
)
```

## Inspect the result

```python
res = data.results["msmc_im"]

print(res["implementation"])
print(res["split_time_quantiles"])
print(res["M"][:5])
```

## Next steps

- [MSMC-IM method page](../methods/msmc-im.md)
- [I/O formats](../guide/io-formats.md)
- [MSMC-IM internals](../developer/internals-msmc-im.md)
