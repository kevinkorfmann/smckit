# Quickstart: SMC++

This quickstart shows the SMC++ workflow on a tiny packaged `.smc.gz`
fixture that ships with the Python package.

## What the input file is

SMC++ reads `.smc` or `.smc.gz` files through {func}`smckit.io.read_smcpp_input`.
These are span-encoded files: long monomorphic stretches are compressed, and
variant records store the distinguished lineage state plus undistinguished
allele counts.

Packaged example:

- installed path: `smckit.io.example_path("smcpp/example.smc.gz")`

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

example = smckit.io.example_path("smcpp/example.smc.gz")

data = smckit.io.read_smcpp_input(example)

data = smckit.tl.smcpp(
    data,
    n_intervals=4,
    max_iterations=1,
    regularization=10.0,
    mu=1.25e-8,
    generation_time=25.0,
    implementation="native",
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

For fidelity-sensitive work on real data, `implementation="upstream"` still
requires a separate SMC++ environment and vendored upstream source.

On the current tracked one-pop matrix, the native path now matches upstream on
both the strict small control and the bundled larger `.smc` fixture, so the
two paths are interchangeable for those enforced docs fixtures.

## Next steps

- [SMC++ method page](../methods/smcpp.md)
- [I/O formats](../guide/io-formats.md)
- [SMC++ internals](../developer/internals-smcpp.md)
