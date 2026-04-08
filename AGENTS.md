# AGENTS

smckit is a preservation-first library for Sequentially Markovian Coalescent
methods. The repository has two equally important responsibilities:

1. keep the original upstream tools available and runnable from this repo
2. build native smckit implementations without losing access to upstream truth

## Working rules

- Treat `vendor/` as the oracle source of truth. Do not edit vendored code.
- Prefer validating behavior against upstream before claiming a native method works.
- Use `smckit.upstream.status()` to inspect which upstream tools are source-ready,
  runtime-ready, and bootstrapped.
- Use `smckit.upstream.bootstrap("<tool>")` to build local upstream artifacts when
  that bootstrap path is implemented.
- `implementation="upstream"` means "run the original tool".
- `implementation="native"` means "run the in-repo implementation".
- `implementation="auto"` should prefer upstream when the bridge is ready.

## How agents should work in this repo

- Read method docs before changing algorithm code.
- Keep public API changes provenance-aware: if a method has upstream-specific
  controls, surface them through structured method-specific options rather than
  deleting them.
- Record and inspect provenance in `data.results[method]["upstream"]`.
- If upstream is not runnable, report that precisely instead of silently falling
  back and pretending parity is solved.

## Where to look next

- `docs/agents/using-smckit.md`
- `docs/agents/algorithms.md`
- `docs/developer/architecture.md`
