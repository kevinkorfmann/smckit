# Quickstart: SMC++

This quickstart shows the SMC++ workflow and, importantly, what kind of input
file the method expects.

## What the input file is

SMC++ reads `.smc` or `.smc.gz` files through {func}`smckit.io.read_smcpp_input`.
These are span-encoded files: long monomorphic stretches are compressed, and
variant records store the distinguished lineage state plus undistinguished
allele counts.

The repository does not currently ship a standalone `.smc.gz` example in
`tests/data`, so treat the [I/O formats guide](../guide/io-formats.md) as the
canonical explanation and use the upstream SMC++ tooling to generate the file.

When the input file lacks an SMC++ header, smckit now assumes the upstream
one-pop layout with two distinguished haplotypes. If you need the legacy
one-distinguished native path for comparison work, set
`data.uns["n_distinguished"] = 1` explicitly before calling
{func}`smckit.tl.smcpp`.

The upstream project repository and example directory are:

- `https://github.com/popgenmethods/smcpp`
- `https://github.com/popgenmethods/smcpp/tree/master/example`

## Run SMC++

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
```

## Inspect the result

```python
res = data.results["smcpp"]

print(res["implementation"])
print(res["log_likelihood"])
print(res["ne"][:5])
print(res["optimization"])
```

## Next steps

- [SMC++ method page](../methods/smcpp.md)
- [I/O formats](../guide/io-formats.md)
- [SMC++ internals](../developer/internals-smcpp.md)
