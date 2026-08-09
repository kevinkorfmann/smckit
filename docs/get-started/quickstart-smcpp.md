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

When the input file lacks an SMC++ header, smckit assumes the upstream one-pop
layout with two distinguished haplotypes. Production inference rejects the
historical one-distinguished native surrogate because it is not an upstream
SMC++ capability and did not pass simulation validation.

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
    output_prefix="results/example",
)
```

The small `n_intervals=4` and `max_iterations=1` settings keep this example
fast. `regularization=10.0` is there so the tiny fixture still produces a
stable-looking curve instead of a noisy one.

## Inspect the result

```python
res = data.results["smcpp"]

print(res["implementation"])
print(res["log_likelihood"])
print(res["ne"][:5])
print(res["optimization"])
```

The output prefix writes a normalized result and a reloadable model:

- `results/example.smcpp.json`
- `results/example.smcpp.model.json`

Both paths and hashes are stored in `res["provenance"]["artifacts"]`.

## Start from VCF

```python
data = smckit.pp.smcpp_from_vcf(
    "cohort.vcf.gz",
    "results/chr22.smc.gz",
    contig="chr22",
    populations={"EUR": ["sample-a", "sample-b", "sample-c"]},
    distinguished=[("sample-a", 0), ("sample-a", 1)],
    mask_path="chr22-callable-mask.bed",
)
```

BED masks use 0-based half-open coordinates. Use `missing_cutoff` instead of a
mask to mark sufficiently long unobserved stretches as missing.

## Select regularization by held-out contigs

Combine multiple records in `data.uns["records"]`, then run:

```python
data = smckit.tl.smcpp_cross_validate(
    data,
    regularization_candidates=[2, 3, 4, 5, 6, 7, 8, 9],
    folds=2,
    seed=1729,
    n_intervals=32,
    max_iterations=100,
)

ax = smckit.pl.smcpp_cross_validation_scores(data)
smckit.pl.save_smcpp_figure(ax.figure, "results/smcpp-cv.pdf")
```

Cross-validation splits whole records/contigs, matching the original SMC++
procedure.

For a fuller explanation of `regularization`, input interpretation, and the
optimization payload, see the [SMC++ method page](../methods/smcpp.md).

## Next steps

- [SMC++ method page](../methods/smcpp.md)
- [I/O formats](../guide/io-formats.md)
- [SMC++ internals](../developer/internals-smcpp.md)
