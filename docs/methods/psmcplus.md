# PSMC+

PSMC+ is a pairwise SMC method designed to account for genomic heterogeneity,
including local mutation, recombination, and coalescence-rate variation. smckit
currently preserves the complete original implementation; an independent
native implementation is planned but has not started.

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

`implementation="upstream"` will mean the original tool,
`implementation="native"` will mean the future in-repo implementation, and
`implementation="auto"` will continue to choose upstream until a native
capability has passed its correctness and performance promotion gates. There is
currently no typed PSMC+ inference function and no native parity claim.

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

The feature ledger distinguishes exact upstream preservation from future
normalized and native coverage. A scientifically meaningful preserved option
cannot be silently dropped from the eventual typed interface.

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
boundaries, and demographic parameters numerically. The fixture proves exact
preserved execution only. Native/default eligibility and performance remain
explicitly unclaimed.
