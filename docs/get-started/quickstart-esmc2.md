# Quickstart: eSMC2

This quickstart runs eSMC2 on the same style of pairwise input used by PSMC.

## What the input file is

The example below uses a `.psmcfa` file. This is a windowed representation of a
diploid genome where each character marks a window as homozygous,
heterozygous, or missing.

Example file:

- repository path: `tests/data/NA12878_chr22.psmcfa`
- GitHub: `https://github.com/kevinkorfmann/smckit/blob/main/tests/data/NA12878_chr22.psmcfa`

## Run eSMC2

```python
import smckit

data = smckit.io.read_psmcfa("tests/data/NA12878_chr22.psmcfa")

data = smckit.tl.esmc2(
    data,
    n_states=20,
    n_iterations=20,
    estimate_beta=True,
    beta=0.8,
    mu=1.25e-8,
    generation_time=1.0,
    implementation="auto",
)
```

## Inspect the result

```python
res = data.results["esmc2"]

print(res["implementation"])
print(res["beta"], res["sigma"])
print(res["ne"][:5])
```

## Next steps

- [eSMC2 method page](../methods/esmc2.md)
- [I/O formats](../guide/io-formats.md)
- [eSMC2 internals](../developer/internals-esmc2.md)
