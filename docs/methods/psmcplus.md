# PSMC+

PSMC+ is a pairwise SMC method designed to account for genomic heterogeneity,
including local mutation, recombination, and coalescence-rate variation. smckit
preserves the complete original implementation and adds a typed upstream
adapter with normalized fit and decoding results. An independent native
implementation remains the next stage and is not yet claimed.

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

`implementation="upstream"` executes the original tool,
`implementation="native"` is rejected clearly until the future in-repo
implementation exists, and `implementation="auto"` chooses upstream until a
native capability has passed its correctness and performance promotion gates.
There is no native parity or speed claim yet.

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
    implementation="auto",
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
    implementation="upstream",
)
```

Decoding results include genomic positions, normalized state posteriors, time
boundaries, posterior mean coalescence time, likelihood, and—when requested—the
marginal probabilities of recombination and no recombination. The typed option
object forwards all scientifically meaningful inference controls from the
original CLI, including matched mutation/recombination maps, interval grouping,
fixed or estimated recombination, optimizer and convergence controls,
approximation switches, and iteration artifacts.

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
Native/default eligibility and performance remain explicitly unclaimed.
