# Quickstart: PSMC

This quickstart runs the native PSMC implementation on a real example
`.psmcfa` file bundled with the repository.

## What the input file is

`read_psmcfa()` expects a `.psmcfa` file. This is a FASTA-like sequence where
each character summarizes one genome window as homozygous, heterozygous, or
missing.

Example file:

- repository path: `tests/data/NA12878_chr22.psmcfa`
- GitHub: `https://github.com/kevinkorfmann/smckit/blob/main/tests/data/NA12878_chr22.psmcfa`

If you want the exact file semantics first, read [I/O formats](../guide/io-formats.md).

## Run PSMC

```python
import smckit

data = smckit.io.read_psmcfa("tests/data/NA12878_chr22.psmcfa")

data = smckit.tl.psmc(
    data,
    pattern="4+5*3+4",
    n_iterations=25,
    max_t=15.0,
    tr_ratio=4.0,
    mu=1.25e-8,
    generation_time=25.0,
    implementation="native",
)

smckit.pl.demographic_history(data)
```

## Inspect the result

```python
res = data.results["psmc"]

print(res["implementation"])
print(res["theta"], res["rho"])
print(res["ne"][:5])
print(res["time_years"][:5])
```

## Next steps

- [PSMC method page](../methods/psmc.md)
- [Interpreting results](../guide/interpreting-results.md)
- [PSMC internals](../developer/internals-psmc.md)
