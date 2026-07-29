# PHLASH

smckit integrates the maintained PHLASH 1.0.6 package as an exact external
method. It does not independently rewrite PHLASH. The adapter preserves its
posterior samples and adds common time, median effective population size, 95%
credible intervals, hashes, versions, runtime, and platform provenance.

Current PHLASH requires Python 3.12 or newer:

```bash
pip install "smckit[phlash]"
```

For PSMCFA inputs:

```python
from smckit.tl import phlash

data = phlash(["sample1.psmcfa", "sample2.psmcfa"], mutation_rate=1.29e-8)
posterior = data.results["phlash"]
times = data.times("phlash")
median_ne = data.effective_population_size("phlash")
interval = posterior["credible_interval"]
```

For VCF/BCF or tree-sequence inputs, pass the arguments needed by
`phlash.contig`:

```python
data = phlash(
    ["chromosome22.vcf.gz"],
    input_kind="contig",
    samples=["NA12878"],
    region="22:5000000-30000000",
    mutation_rate=1.29e-8,
)
```

Import `phlash` directly when using parts of its original Python interface that
are not normalized by smckit. PHLASH intentionally has no original CLI, so
`smckit upstream phlash` reports that fact instead of inventing one.
