# Quickstart: PSMC

This quickstart runs the native PSMC implementation on a real example
`.psmcfa` file bundled with the Python package.

## What the input file is

`read_psmcfa()` expects a `.psmcfa` file. This is a FASTA-like sequence where
each character summarizes one genome window as homozygous, heterozygous, or
missing.

Packaged example:

- installed path: `smckit.io.example_path("psmc/NA12878_chr22.psmcfa")`
- packaged source directory: `https://github.com/kevinkorfmann/smckit/tree/main/src/smckit/data/examples/psmc`

If you want the exact file semantics first, read [I/O formats](../guide/io-formats.md).

## Run PSMC

```python
import smckit

example = smckit.io.example_path("psmc/NA12878_chr22.psmcfa")

data = smckit.io.read_psmcfa(example)

data = smckit.tl.psmc(
    data,
    pattern="4+5*3+4",
    n_iterations=25,
    max_t=15.0,
    tr_ratio=4.0,
    mu=1.25e-8,
    generation_time=25.0,
    decode="posterior",
    output_path="sample.psmc",
    implementation="native",
)

smckit.pl.demographic_history(data)
```

`pattern="4+5*3+4"` is the classic starting choice because it gives a smooth,
widely used time grouping. `tr_ratio=4.0` is a normal starting value for the
mutation-to-recombination scale ratio, not a claim about your organism.

If you want to validate against the original upstream `psmc` binary, use a
source checkout instead of a wheel install so the vendored upstream source is
present.

For a diploid consensus FASTA or FASTQ, create the windowed input natively:

```python
data = smckit.pp.psmcfa_from_consensus(
    "diploid.fq.gz",
    output_path="diploid.psmcfa.gz",
    min_quality=20,
    min_good_bases=10_000,
    block_size=100,
    masks={"chr1": [(10_000, 20_000)]},
)
```

The mask coordinates are 0-based and half-open. The converter also supports
the original transition, transversion, CpG-only, and CpG-exclusion filters.
For bootstrap confidence curves, use `smckit.tl.psmc_bootstrap`; it applies
the original `splitfa` boundary and length-matched resampling rules.

Every original PSMC entry point remains selectable. For example:

```bash
smckit upstream psmc --entrypoint fq2psmcfa -- -q20 diploid.fq.gz
smckit upstream psmc -- -N 25 -t 15 -r 5 -p 4+25*2+4+6 diploid.psmcfa
```

## Inspect the result

```python
res = data.results["psmc"]

print(res["implementation"])
print(res["theta"], res["rho"])
print(res["ne"][:5])
print(res["time_years"][:5])
```

For a fuller explanation of `pattern`, `max_t`, and the result fields, see the
[PSMC method page](../methods/psmc.md).

## Next steps

- [PSMC method page](../methods/psmc.md)
- [Interpreting results](../guide/interpreting-results.md)
- [PSMC internals](../developer/internals-psmc.md)
