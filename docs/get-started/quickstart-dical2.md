# Quickstart: diCal2

This quickstart uses the vendored README-style diCal2 example files already
tracked in the repository.

## What the input files are

diCal2 works from a small file bundle rather than one file:

- `.param` for mutation/recombination parameters
- `.demo` for the demographic model
- `.config` for sample assignments
- sometimes `.rates`
- sequence data, often VCF plus a reference

Repository examples:

- `vendor/diCal2/examples/fromReadme/test.param`
- `vendor/diCal2/examples/fromReadme/exp.demo`
- `vendor/diCal2/examples/fromReadme/exp.config`
- `vendor/diCal2/examples/fromReadme/test.vcf`

GitHub directory:

- `https://github.com/kevinkorfmann/smckit/tree/main/vendor/diCal2/examples/fromReadme`

## Run diCal2

```python
import smckit

data = smckit.io.read_dical2(
    sequences="vendor/diCal2/examples/fromReadme/test.vcf",
    param_file="vendor/diCal2/examples/fromReadme/test.param",
    demo_file="vendor/diCal2/examples/fromReadme/exp.demo",
    config_file="vendor/diCal2/examples/fromReadme/exp.config",
    reference_file="vendor/diCal2/examples/fromReadme/test.fa",
    filter_pass_string=".",
)

data = smckit.tl.dical2(
    data,
    n_intervals=11,
    max_t=4.0,
    n_em_iterations=2,
    composite_mode="pac",
    implementation="native",
)
```

## Inspect the result

```python
res = data.results["dical2"]

print(res["implementation"])
print(res["log_likelihood"])
print(res["pop_sizes"])
```

## Next steps

- [diCal2 method page](../methods/dical2.md)
- [I/O formats](../guide/io-formats.md)
- [Parity notes](../developer/parity.md)
