# PSMC+ benchmark evidence

The original immutable JSON records compare the independent native PSMC+ engine with
the source pinned at commit `032168f2ceed3c0e46b7f214f890faf83dff41ae` on the
upstream constant-population fixture. All measurements use one thread and five
timed repetitions on Linux x86-64.

- `warm-core.json` is the first Linux warmed-core record. Both implementations
  ran in the same warmed Python process and upstream preprocessing was excluded,
  but its schema-1 confidence interval independently resampled implementation
  timings. It is retained for audit history; new promotion evidence uses the
  schema-2 counterbalanced paired design described below.
- `aggregate.json` summarizes isolated end-to-end typed calls, including cold
  native startup, warmed native calls, subprocess startup for upstream, and
  peak process-tree RSS. It measures user-facing execution, not pure algorithm
  speed, so its larger speedups are not used alone for promotion.
- `*-fit.json` and `*-decode.json` are the raw end-to-end measurements consumed
  by `aggregate.json`.

Reproduce the records with `scripts/benchmark_psmcplus_warm_core.py`,
`scripts/benchmark_psmcplus_native.py`, and
`scripts/aggregate_psmcplus_benchmark.py`. The tracked `SHA256SUMS` file covers
every JSON record in this directory.

The current protocol warms both inference cores, alternates
native-then-upstream and upstream-then-native measurement order, and
bootstraps within-pair speedup ratios. Typed end-to-end runs are a separate
scope used for cold cost and peak process-tree memory; their subprocess-heavy
runtime is explicitly diagnostic and cannot satisfy the speed gate. Combine
the paired core and typed memory records with
`scripts/aggregate_psmcplus_promotion.py`. A capability passes only when the
paired speed confidence interval excludes parity and native peak memory is no
more than 1.25 times upstream.
