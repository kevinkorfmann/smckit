# PSMC+ benchmark evidence

These immutable JSON records compare the independent native PSMC+ engine with
the source pinned at commit `032168f2ceed3c0e46b7f214f890faf83dff41ae` on the
upstream constant-population fixture. All measurements use one thread and five
timed repetitions on Linux x86-64.

- `warm-core.json` is the promotion-relevant comparison. Both implementations
  run in the same warmed Python process, and upstream preprocessing is excluded.
  The script asserts numerical agreement before writing the record.
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
