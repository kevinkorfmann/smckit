# PSMC+

PSMC+ is a pairwise SMC method designed to account for genomic heterogeneity,
including local mutation, recombination, and coalescence-rate variation. smckit
preserves the complete original implementation and adds typed upstream and
independent native fit/decoding engines with a shared normalized result schema.

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

The frozen one-thread Linux x86-64 benchmark uses five repetitions and 20,000
bootstrap resamples. With both engines warmed in the same Python process, native
fit is 2.27x faster (95% CI 2.22--2.32) and decoding is 1.25x faster (95% CI
1.21--1.27). The separate end-to-end measurement reports native peak memory at
0.394x upstream or lower and records cold JIT cost independently. The immutable
records, raw timings, checksums, and reproduction scripts are under
`benchmarks/psmcplus/`. These results support the frozen fixture only; broader
simulation, empirical, and macOS ARM64 results remain required before changing
`auto`.
