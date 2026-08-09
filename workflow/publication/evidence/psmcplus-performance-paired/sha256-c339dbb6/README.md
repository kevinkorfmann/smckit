# PSMC+ paired cross-platform performance evidence

This immutable bundle applies the frozen PSMC+ promotion protocol to Linux
x86-64 and macOS ARM64 at clean smckit commit
`c9df632ee0220a1a55e0fb58d0211dc3d5284917`. Both platforms use the exact
upstream commit `032168f2ceed3c0e46b7f214f890faf83dff41ae`, Python 3.12,
NumPy 2.4.4, one numeric thread, five warmed timing pairs, and 20,000 paired
bootstrap samples.

The promotion-relevant runtime scope excludes input preprocessing and runs
both cores inside one warmed process. Measurement order alternates between
native-then-upstream and upstream-then-native, and confidence intervals
resample within-pair speedup ratios. Separate typed end-to-end records capture
cold native cost and peak process-tree RSS. Their subprocess-heavy runtimes
are diagnostic only and are explicitly ineligible for the speed claim.

| Platform | Capability | Speedup | Paired-bootstrap 95% CI | Memory ratio |
|---|---|---:|---:|---:|
| Linux x86-64 | fit | 2.1613x | 2.0676–2.1881x | 0.3868 |
| Linux x86-64 | decode | 1.0960x | 1.0586–1.1368x | 0.3855 |
| macOS ARM64 | fit | 2.3206x | 2.2879–2.3459x | 0.3635 |
| macOS ARM64 | decode | 1.1116x | 1.1079–1.1229x | 0.3664 |

Every speed interval excludes parity and every native/upstream median
peak-memory ratio is below the maximum allowed 1.25. Both fit and decode
therefore pass the frozen performance gate on both required CPU platforms.
The raw timings, source state, environment, input checksum, per-scope hashes,
cold calls, and combined decisions are retained for audit.
