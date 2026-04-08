# Quickstart: MSMC2

This quickstart runs the current native MSMC2 implementation on the bundled
fixture-style `.multihetsep` file.

## What the input file is

`read_multihetsep()` expects a `.multihetsep` file. Each row represents one
segregating site plus its distance from the previous site and the observed
alleles for each haplotype.

Example file:

- repository path: `data/msmc2_test.multihetsep`
- GitHub: `https://github.com/kevinkorfmann/smckit/blob/main/data/msmc2_test.multihetsep`

## Run MSMC2

```python
import smckit

data = smckit.io.read_multihetsep("data/msmc2_test.multihetsep")

data = smckit.tl.msmc2(
    data,
    time_pattern="1*2+25*1+1*2+1*3",
    n_iterations=20,
    mu=1.25e-8,
    generation_time=25.0,
    implementation="native",
)
```

## Inspect the result

```python
res = data.results["msmc2"]

print(res["implementation"])
print(res["lambda"][:5])
print(res["time_years"][:5])
```

## Next steps

- [MSMC2 method page](../methods/msmc2.md)
- [MSMC-IM quickstart](quickstart-msmc-im.md)
- [Parity notes](../developer/parity.md)
