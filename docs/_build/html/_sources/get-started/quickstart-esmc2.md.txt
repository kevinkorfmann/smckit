# Quickstart: eSMC2

This quickstart runs eSMC2 on the same style of pairwise input used by PSMC.

## What the input file is

The example below uses a packaged `.psmcfa` file. This is a windowed representation of a
diploid genome where each character marks a window as homozygous,
heterozygous, or missing.

Packaged example:

- installed path: `smckit.io.example_path("psmc/NA12878_chr22.psmcfa")`
- packaged source directory: `https://github.com/kevinkorfmann/smckit/tree/main/src/smckit/data/examples/psmc`

## Run eSMC2

```python
import smckit

example = smckit.io.example_path("psmc/NA12878_chr22.psmcfa")

data = smckit.io.read_psmcfa(example)

data = smckit.tl.esmc2(
    data,
    n_states=20,
    n_iterations=20,
    estimate_beta=True,
    beta=0.8,
    mu=1.25e-8,
    generation_time=1.0,
    output_dir="analysis/esmc2",
    implementation="native",
)
```

This example turns on `estimate_beta` so the ecological part of the model is
active. `beta=0.8` is just a starting assumption for that optimizer, not a
trusted species-specific value by itself.

The output directory contains the original numeric result tables (`Tc.txt`,
`Xi.txt`, `rho.txt`, `beta.txt`, `sigma.txt`, `mu.txt`, and `LH.txt`) plus
normalized time and effective-size tables. The complete preserved R package
is available to custom scripts with:

```bash
smckit upstream esmc2 -- analysis.R argument1 argument2
```

The runner injects the bootstrapped repository-local R library, so every
exported upstream helper remains available without changing the user library.

## Inspect the result

```python
res = data.results["esmc2"]

print(res["implementation"])
print(res["beta"], res["sigma"])
print(res["ne"][:5])
```

For a fuller explanation of `beta`, `sigma`, `mu_b`, `pop_vect`, and the
advanced tuning knobs, see the [eSMC2 method page](../methods/esmc2.md). That
page also summarizes the tracked public input-family parity gate for `.psmcfa`
and `multihetsep`.

## Next steps

- [eSMC2 method page](../methods/esmc2.md)
- [Gallery](../guide/gallery.md)
- [I/O formats](../guide/io-formats.md)
- [eSMC2 internals](../developer/internals-esmc2.md)
