# Contributing

Use a focused branch and keep preservation, parity, and optimization changes
separate when possible. Install with `uv sync --extra dev --extra docs`.

Before opening a pull request:

1. Run `pixi run test-fast`.
2. Run `uv run ruff format --check src/smckit tests`.
3. Run the affected integration or oracle specification.
4. Update the method feature ledger, capability status, and changelog.
5. Include commands, platform, random seeds, runtime, memory, and numerical
   comparisons for any parity or performance claim.

Do not copy implementation code from GPL or otherwise incompatible upstream
projects into the MIT-native package. Upstream source is preserved as a
separate oracle and retains its own license.

Native promotion requires correctness first, then a warmed runtime confidence
interval excluding parity and peak memory no more than 25% above upstream.
