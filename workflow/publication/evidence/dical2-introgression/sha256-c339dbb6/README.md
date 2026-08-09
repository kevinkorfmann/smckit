# diCal2 fitted-introgression evidence

This directory freezes the first capability-specific performance gate for
native diCal2: one EM step on a deterministic, independently simulated
three-population pulse-introgression dataset.

Both implementations used the same VCF information, model files, start point,
seed, interval grid, LOL composite likelihood, one EM iteration, and the
shared one-iteration M-step contract. The preserved Java run explicitly used
`--useEigenCore`, as required for pulse migration. Fixture preparation was
measured separately from each inference call. All numerical-library and
benchmark threads were fixed to one.

Across ten warmed macOS ARM64 repetitions, native smckit was `1.1381x` faster
than the preserved Java implementation; the paired-bootstrap 95% confidence
interval was `1.1343-1.1433x`. Native peak process-tree memory was `0.4909x`
upstream memory. Every repetition returned the same fitted parameters, and the
strict live oracle records native-versus-Java likelihood agreement separately
in `tests/integration/test_dical2_structured_oracles.py`.

This evidence closes only the fitted introgression capability on this
platform. It does not promote all native diCal2 workflows or establish the
Linux performance gate. Growth inference, broader fitted structured models,
parallel execution, detailed artifacts, and additional platforms remain open.

The benchmark source is commit `7e01824ef0d02caf865e12dea3a51923e341331f`.
The manuscript and private continuation checkpoint were not read, copied, or
included.
