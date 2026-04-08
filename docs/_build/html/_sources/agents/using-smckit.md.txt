# Using smckit as an Agent

This page is for coding agents and contributors working inside the repository.
It describes the operational contract, not the user tutorial flow.

## Core contract

- Vendored upstream code in `vendor/` is the oracle.
- Repo-local caches and runtime installs are disposable.
- Native ports are important, but they do not replace the requirement that the
  upstream original be runnable and inspectable.

## Check upstream readiness

```python
import smckit

status = smckit.upstream.status()
print(status["psmc"]["ready"])
print(status["esmc2"]["missing"])
```

The status payload tells you:

- whether vendored source is present
- whether the needed runtime exists
- whether any cache/bootstrap artifact is ready
- whether the public upstream bridge is actually wired

## Bootstrap when needed

```python
import smckit

smckit.upstream.bootstrap("esmc2")
```

Use bootstrap only for tools whose registry entry supports it. If a tool is not
public yet, status should still tell you exactly what is missing.

## Choose implementation intentionally

- Use `implementation="upstream"` when validating behavior or reproducing the
  original tool.
- Use `implementation="native"` when working on the in-repo port.
- Use `implementation="auto"` only when you want the library to prefer upstream
  fidelity automatically.

## Method-specific controls

smckit keeps a unified API, but meaningful original-tool controls should remain
reachable through structured method-specific options:

```python
data = smckit.tl.esmc2(
    data,
    implementation="upstream",
    upstream_options={"note": "record original invocation semantics here"},
)
```

The resolved upstream arguments and provenance should be visible in
`data.results[method]["upstream"]`.

## Validation workflow

1. Check `smckit.upstream.status("<tool>")`.
2. Bootstrap if needed and supported.
3. Run the upstream path on a vendored or tracked fixture.
4. Run the native path on the same fixture.
5. Compare outputs explicitly. Do not describe parity as complete unless the
   tracked tests actually support that claim.
