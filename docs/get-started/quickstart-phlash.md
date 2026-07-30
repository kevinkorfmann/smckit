# Quickstart: PHLASH

PHLASH is the Bayesian option in smckit. It returns posterior demographic
trajectories and credible intervals instead of only a point estimate.

## PSMCFA input

```python
import smckit

data = smckit.tl.phlash(
    ["sample.psmcfa"],
    implementation="auto",
    random_seed=1729,
    credible_level=0.95,
    output_prefix="results/sample",
)

result = data.results["phlash"]
print(result["n_posterior_samples"])
print(result["provenance"]["upstream"])

ax = smckit.pl.phlash_demographic_history(
    data,
    posterior_samples=20,
)
smckit.pl.save_phlash_figure(ax.figure, "results/sample-phlash.pdf")
```

The output prefix creates:

- `results/sample.phlash.json`
- `results/sample.phlash.posterior.npz`

Both files are recorded with SHA-256 hashes in result provenance.

## VCF/BCF input

```python
data = smckit.tl.phlash(
    ["cohort.bcf"],
    samples=["sample-a", "sample-b", "sample-c"],
    region="chr22:1-50000000",
    random_seed=1729,
)
```

Use the same `samples` and `region` for an optional `test_input` holdout. For
tree-sequence input, provide sample-node pairs and omit `region`.

## Reproducibility controls

Record `random_seed`, all fitting options, exact input paths, and the package
environment. smckit converts `random_seed` to PHLASH's JAX key and stores the
seed, PHLASH version, platform, input hashes, runtime, and any compatibility
warning in the standard provenance envelope.
