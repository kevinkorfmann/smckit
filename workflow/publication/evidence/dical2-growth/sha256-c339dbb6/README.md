# diCal2 fitted-growth evidence

This directory freezes the capability-specific correctness and performance
gate for one native diCal2 exponential-growth workflow. The input is a
deterministic msprime simulation with five diploid samples, a recent
exponential-growth epoch, and two older constant-size epochs. Both
implementations use the vendored `expGrowth` demo, rates, configuration, and
mutation/recombination files.

Native and preserved Java runs receive the same VCF information, start point,
bounds, seed, log-uniform interval grid, PCL composite likelihood, physical
locus grouping, one EM iteration, and one M-step iteration. Fixture preparation
is measured separately from inference. All numerical-library and benchmark
threads are fixed to one.

Across ten warmed macOS ARM64 repetitions, native smckit was `1.3052290626x`
faster than preserved Java; the paired-bootstrap 95% confidence interval was
`1.2834588164-1.3543839272x`. Native peak process-tree memory was
`0.5959217645x` upstream memory. Every repetition returned identical fitted
parameters (`[1.2800000000000002, 2.00000003]`), and the maximum absolute
native-versus-Java final log-likelihood difference was
`3.7516656448e-12`.

Finite constant-rate refined epochs use exact matrix exponentials in the
native implementation. The genuinely time-varying growth epoch and the
Java-compatible infinity sentinel retain high-accuracy ODE integration. This
evidence closes fitted growth on this macOS platform only. It does not promote
all native diCal2 workflows or establish the Linux gate; broader structured
fitting, detailed artifacts, empirical validation, and parallel execution
remain open.

The benchmark source is commit
`95d983d10c4a2f8d9fb6345d133c7d3c4a714d8e`. No private manuscript,
continuation checkpoint, or excluded publication prose is included.
