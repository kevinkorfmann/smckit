# diCal2 repaired marginal-KL oracle

The pinned `vendor/diCal2/diCal2.jar` is the immutable compatibility oracle and
must never be changed. Its `--marginalKL` path crashes during the first E-step:
`MarginalKLDivergence.getLogLikelihood()` evaluates the marginal objective with
a null mutation-rate vector instead of returning the already-computed sequence
log likelihood inherited from `StructMigEmHMMObjectiveFunction`.

`marginal-kl-get-likelihood.patch` is a one-method GPL-3.0-or-later repair used
only to construct an additional scientific oracle. It is applied to a temporary
copy of the pinned source by `scripts/build_dical2_repaired_oracle.py`; neither
the vendored source nor the pinned jar is modified. The derived jar remains GPL
oracle material and must not be included in smckit's MIT wheel or copied into
the independent native implementation.

Build it locally with a JDK:

```bash
python scripts/build_dical2_repaired_oracle.py \
  --output /tmp/dical2-marginal-kl-oracle.jar
```

The builder verifies the pinned source and jar hashes, applies the patch with
zero fuzz, compiles only the repaired Java compilation unit as Java 8 bytecode
against the pinned jar, updates a copy with fixed ZIP timestamps, and prints
JSON provenance including the derived SHA-256.
Exact upstream execution continues to use the unmodified pinned jar.
