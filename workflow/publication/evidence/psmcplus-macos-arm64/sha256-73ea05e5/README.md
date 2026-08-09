# PSMC+ macOS ARM64 capability matrix

This immutable record repeats the frozen native-versus-upstream PSMC+
capability matrix on Apple Silicon. It was generated from clean detached
smckit commit `9f924bdd385fddd1f45cc214d9cbc80b2e761606`, with the exact upstream
at commit `032168f2ceed3c0e46b7f214f890faf83dff41ae`, Python 3.12.11,
NumPy 2.4.4, and one execution thread.

All twelve cases passed on `macOS-26.2-arm64-arm-64bit`. The matrix covers
constant, bottleneck, and expansion simulations; missing data; multi-file
fitting; grouped and fixed parameters; estimated recombination; local
mutation and recombination maps; rate-map downsampling; every approximation
control independently and in combination; the original final-time behavior;
posterior decoding; and marginal recombination.

The largest fitted lambda relative error was `1.7289537506e-08`, the largest
fit log-likelihood absolute error was `7.0218675319e-10`, the largest posterior
absolute error was `2.0059509609e-12`, and decoded and marginal positions were
exact. These satisfy the frozen capability gates.

Reproduce the protocol from a clean checkout with:

```console
PYTHONPATH=src python workflow/publication/scripts/validate_psmcplus_matrix.py \
  --output psmcplus-macos-arm64-matrix.json
```

The per-case runtimes in the JSON are diagnostic single observations. They
must not be used as performance-promotion evidence; the separate paired,
five-repetition benchmark remains authoritative for speed and memory claims.
