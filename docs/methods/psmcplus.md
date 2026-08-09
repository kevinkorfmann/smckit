# PSMC+

PSMC+ is the PSMC extension designed to correct demographic inference in the
presence of background selection. It models background selection through
locus-specific coalescence-rate scaling and can also account for local mutation
and recombination-rate heterogeneity. smckit preserves the complete original
implementation and adds typed upstream and independent native fit/decoding
engines with a shared normalized result schema. See the [PSMC+
paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10838404/) for the biological
motivation and validation.

## Preservation and implementation contract

The upstream source is pinned to commit
`032168f2ceed3c0e46b7f214f890faf83dff41ae` and remains read-only under
`vendor/PSMCplus`. The project has no tagged release, and its associated article
is currently a bioRxiv preprint. The source and smckit's native code are both
MIT-licensed, but native work will still be implemented independently and
validated against the pinned source.

Install the Python dependency stack used by the raw preservation runner:

```bash
uv pip install "smckit[psmcplus]"
```

`implementation="upstream"` executes the original tool and
`implementation="native"` executes the independent in-repo engine. The
conservative `implementation="auto"` policy still chooses upstream while the
empirical validation matrix is expanded. Exact raw simulation remains an
upstream-only capability and is permanently available.

## Typed inference and normalized results

Read one or more original multihetsep files and fit the preserved model through
the typed API:

```python
import smckit

data = smckit.io.read_multihetsep("chromosome.multihetsep")
smckit.tl.psmcplus(
    data,
    options=smckit.tl.PSMCPlusOptions(
        number_time_windows=32,
        bin_size=100,
        iterations=20,
        cores=1,
    ),
    mutation_rate=1.25e-8,
    generation_time=25,
    output_prefix="results/psmcplus_",
    implementation="native",
)
```

For empirical data, construct multihetsep from a sample-level called VCF and
explicit callability masks rather than treating every site absent from a
variant VCF as callable:

```python
data = smckit.pp.multihetsep_from_vcf(
    "sample.chr22.called.vcf.gz",
    "sample.chr22.multihetsep",
    mask_paths=["sample.chr22.depth-callable.bed.gz"],
    negative_mask_paths=["chr22.exclusions.bed.gz"],
)
```

The native converter streams arbitrarily distant sites without chromosome-sized
arrays, supports multiple individuals and trio phasing/removal, and records a
complete preprocessing provenance block. Its shared behavior is executed
against checksum-pinned `msmc-tools` `generate_multihetsep.py` in CI. The
native negative-mask implementation deliberately repairs the helper's
end-of-file behavior by treating positions after the last excluded interval as
eligible; exact historical behavior remains available by running the pinned
helper itself.

The result is stored at `data.results["psmcplus"]`. Fit results expose common
fields including `time`, `ne`, `lambda`, `theta`, `rho`, and
`log_likelihood`, alongside the original scaled boundaries, hashed artifacts,
the exact executed command, input hashes, runtime, environment, and recorded
compatibility adjustments. Supplying `mutation_rate` converts population size
to individuals; supplying both `mutation_rate` and `generation_time` reports
time in years.

Posterior decoding and optional marginal recombination output use the same
interface:

```python
smckit.tl.psmcplus(
    data,
    options=smckit.tl.PSMCPlusOptions(
        mode="decode",
        number_time_windows=32,
        lambda_initial=[1.0] * 32,
        decode_downsample=10,
        cores=1,
    ),
    output_prefix="results/posterior.txt",
    marginal_recombination_path="results/recombination.txt",
    implementation="native",
)
```

Decoding results include genomic positions, normalized state posteriors, time
boundaries, posterior mean coalescence time, likelihood, and—when requested—the
marginal probabilities of recombination and no recombination. The typed option
object forwards all scientifically meaningful inference controls from the
original CLI, including matched mutation/recombination maps, interval grouping,
fixed or estimated recombination, optimizer and convergence controls,
approximation switches, and iteration artifacts.

For mapped mutation rates, the native result preserves the original marginal
recombination calculation under `marginal_recombination` and also exposes a
`corrected_marginal_recombination` field that applies the local mutation factor
to the marginal calculation. This makes the compatibility behavior explicit
without silently inheriting the upstream inconsistency.

## Exact original interface

Run the original inference program with all of its arguments unchanged:

```bash
smckit upstream psmcplus --output-dir results -- \
  -in chromosome.multihetsep -D 32 -b 100 -its 20 -o psmcplus_
```

Run the original HMM simulator by selecting its second entry point:

```bash
smckit upstream psmcplus --entrypoint simulate_HMM.py \
  --output-dir simulations -- \
  -D 10 -theta 0.001 -rho 0.0005 -L 1000000 \
  -o_mhs variants.mhs -o_coal coal.txt.gz
```

The runner uses argument vectors without shell interpolation, executes in an
isolated output directory, preserves exit status/stdout/stderr, and hashes all
generated files. Both entry points remain available even after a native method
is introduced.

## Preserved upstream surface

The inference entry point supports multi-file multihetsep input, matching local
mutation and recombination maps, genomic binning, fixed or estimated
recombination, grouped/fixed demographic intervals, alternative transition and
emission approximations, Powell tolerances and other optimizer choices,
convergence controls, iteration artifacts, posterior decoding, marginal
recombination output, and controlled parallelism. The simulation entry point
generates variant and latent coalescence histories under a supplied PSMC'
history.

The feature ledger distinguishes exact upstream preservation and normalized
typed coverage from future native coverage. The HMM simulator remains available
through the exact raw CLI because it has a distinct command surface. No
scientifically meaningful original option is silently substituted or dropped.

## Runtime compatibility and containers

The pinned source uses `numpy.math`, which NumPy 2 removed. smckit's raw launcher
restores only that alias inside the child process and records the adjustment in
execution provenance; it does not edit the vendored files. A separate
content-addressed OCI definition under
`preservation/containers/psmcplus/` uses the dependency versions documented by
upstream and can be converted for Apptainer-based HPC use.

## Current validation

The frozen constant-population oracle runs one EM iteration with four time
windows and compares final likelihood, likelihood change, theta, rho, time
boundaries, and demographic parameters numerically. The same oracle now runs
through the typed adapter and validates normalized physical scaling, persistent
artifacts, common accessors, input hashes, and runtime provenance. A live
decoding oracle independently validates posterior normalization and marginal
recombination probabilities. On the one-thread Sesame validation environment,
both typed fit and decode completed successfully against the pinned source.
The independent native implementation is public and locked to frozen,
content-addressed oracles generated from the pinned source. Constant-rate and
mapped workflows validate preprocessing, time grids, interval expectations,
local-rate transition matrices, mutation-adjusted emissions, forward/backward
arrays, EM evidence, fitted parameters, likelihood, state posteriors, and
original marginal-recombination probabilities within floating-point precision.
Multi-file likelihoods, parameter grouping/fixing, optimizer bounds and
tolerances, alternative time/transition approximations, original-compatible
iteration/final artifacts, and fixed/estimated recombination are exercised by
the native suite.

A separate clean-commit Linux x86-64 capability matrix covers twelve
deterministic constant, bottleneck, and expansion cases. It adds missing data,
multi-file input, grouped/fixed parameters, estimated recombination, local
maps, every approximation control independently and in combination, the
original final-time-grid behavior, and fit/decode outputs. All cases pass the
strict gate; the frozen JSON and checksum are under
`workflow/publication/evidence/psmcplus-promotion/sha256-5aa89ae3/`.

The identical twelve-case protocol also passes from a clean macOS ARM64
checkout. Maximum lambda relative error was `1.73e-8`, maximum fit
log-likelihood absolute error was `7.02e-10`, maximum posterior absolute error
was `2.01e-12`, and decoded positions were exact. The checksum-addressed record
is under
`workflow/publication/evidence/psmcplus-macos-arm64/sha256-73ea05e5/`.

The current cross-platform benchmark uses five counterbalanced warmed pairs
and 20,000 paired-bootstrap samples per capability. On Linux x86-64, native fit
is 2.16x faster (95% CI 2.07--2.19) and decoding is 1.10x faster (1.06--1.14).
On macOS ARM64, fit is 2.32x faster (2.29--2.35) and decoding is 1.11x faster
(1.11--1.12). Separate typed end-to-end measurements put native median peak
memory between 0.363x and 0.387x upstream across both capabilities and record
cold native cost independently. Subprocess-heavy end-to-end runtimes are
diagnostic and are not used for the speed claim. The immutable paired records
and raw measurements are under
`workflow/publication/evidence/psmcplus-performance-paired/sha256-c339dbb6/`;
the earlier Linux records remain under `benchmarks/psmcplus/` for audit
history. Human/nonhuman empirical validation remains required before changing
`auto`.
