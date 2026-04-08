# Quickstart: ASMC

This quickstart decodes phased haplotype pairs with ASMC using the vendored
example files already present in the repository.

## What the input files are

ASMC needs:

- a haplotype prefix, which expands to `.hap.gz`, `.samples`, and `.map.gz`
- a `.decodingQuantities.gz` file with precomputed HMM tables

Example files in this repository:

- `vendor/ASMC/ASMC_data/examples/asmc/exampleFile.n300.array.hap.gz`
- `vendor/ASMC/ASMC_data/examples/asmc/exampleFile.n300.array.samples`
- `vendor/ASMC/ASMC_data/examples/asmc/exampleFile.n300.array.map.gz`
- `vendor/ASMC/ASMC_data/decoding_quantities/30-100-2000_CEU.decodingQuantities.gz`

GitHub directory:

- `https://github.com/kevinkorfmann/smckit/tree/main/vendor/ASMC/ASMC_data/examples/asmc`

## Run ASMC

```python
import smckit

data = smckit.io.read_asmc(
    "vendor/ASMC/ASMC_data/examples/asmc/exampleFile.n300.array",
    "vendor/ASMC/ASMC_data/decoding_quantities/30-100-2000_CEU.decodingQuantities.gz",
)

data = smckit.tl.asmc(
    data,
    pairs=[(0, 1), (2, 3)],
    mode="array",
    store_per_pair_posterior_mean=True,
    store_per_pair_map=True,
    implementation="native",
)
```

## Inspect the result

```python
res = data.results["asmc"]

print(res["implementation"])
print(res["n_pairs_decoded"])
print(res["sum_of_posteriors"].shape)
```

## Next steps

- [ASMC method page](../methods/asmc.md)
- [I/O formats](../guide/io-formats.md)
- [Gallery](../guide/gallery.md)
