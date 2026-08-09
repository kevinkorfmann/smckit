# diCal2 output and artifact catalogue

This catalogue defines the preserved and native output contract for the
normal diCal2 inference CLI (`StructureEstimationEM`). It is based on the
vendored manual's **Output** section and the pinned Java source.

## Authoritative upstream surface

The normal inference CLI writes result data to standard output and diagnostics
to standard error. It does not create result files. Users of the original tool
make an artifact by redirecting stdout.

The stable, machine-readable portion of stdout is one tab-separated row for
every E-step:

1. log likelihood;
2. E-step time in milliseconds;
3. parameters in `?0`, `?1`, ... order; and
4. `[GENERATION_STEP_PARTICLE]` identifier.

Comment lines beginning with `#` contain the command, parsed settings,
permutations, interval diagnostics, expectation/maximization markers, and total
timings. They are valuable diagnostics but are deliberately ignored by the
original post-processing convention. Standard error contains warnings and
failures. The raw preservation command remains the authority whenever exact
diagnostic text or an undocumented Java behavior is required.

## smckit preservation and native artifacts

With `output_prefix=PREFIX`, both implementations write:

- `PREFIX.dical2.txt`: exact captured Java stdout for `upstream`; for `native`,
  every E-step in the original tab-separated row grammar and original
  generation/step/particle ordering.
- `PREFIX.dical2.json`: the normalized, provenance-rich smckit result,
  including the full `em_path`, selected optimum, resolved options,
  initialization, permutations, demographic arrays, platform and version
  metadata, warnings, and artifact hashes.

The native `em_path` is recorded even when the optional detailed optimization
trace is disabled. Invalid meta-start particles retain their `-inf` E-step row,
matching the observable upstream particle sequence. Exact raw stdout and stderr
remain available in `results["dical2"]["upstream"]` for preserved executions.

## Compatibility evidence

- `TestDical2Output` round-trips single and multi-step native artifacts through
  the same parser used for Java stdout.
- `TestReadDical2.test_multiple_contigs_reset_the_native_hmm` verifies that a
  real native inference persists its full E-step path and artifact hashes.
- `test_dical2_native_random_start_sequence_matches_upstream` verifies native
  and Java generation/step/particle identifiers across multiple particles.
- Exact upstream output persistence is exercised by the same writer path and
  never reconstructed from normalized fields.

This closes the original result-artifact catalogue for the normal inference
CLI. Other upstream entry points remain accessible through
`smckit upstream dical2 -- <original arguments>` and are not silently mapped to
the typed inference result schema.
