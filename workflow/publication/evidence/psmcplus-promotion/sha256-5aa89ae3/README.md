# PSMC+ promotion capability matrix

This frozen record compares native smckit with the exact PSMC+ upstream at
commit `032168f2ceed3c0e46b7f214f890faf83dff41ae`. It was generated from a clean,
detached smckit checkout at `ab3a52b033336483dd248acecb826eaeee3e8246` on
Linux x86-64 with one execution thread.

All twelve deterministic cases passed. The matrix covers constant,
bottleneck, and expansion simulations; missing data; multi-file fitting;
grouped and fixed parameters; estimated recombination; mutation and
recombination maps; rate-map downsampling; every approximation control both
independently and in combination; posterior decoding; marginal recombination;
and the preserved final-time-grid behavior.

Run the same protocol with:

```console
PYTHONPATH=src python workflow/publication/scripts/validate_psmcplus_matrix.py \
  --output psmcplus-promotion-matrix.json
```

The JSON records seeds, input checksums, source cleanliness, software and
platform versions, arguments, normalized scalar/vector results, hashes of
large decoded arrays, comparison errors, and end-to-end runtimes. These timing
observations are diagnostic because each case was run once; the separate
five-repetition benchmark remains the performance-promotion evidence.

