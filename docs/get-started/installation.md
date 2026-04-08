# Installation

smckit has two installation layers:

1. install the Python package itself
2. enable the upstream tools you want to preserve and run

## Base install

```bash
git clone https://github.com/kevinkorfmann/smckit.git
cd smckit
pip install -e ".[dev]"
```

Core Python dependencies:

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.7
- Numba >= 0.59

Optional JAX support for the SSM extension:

```bash
pip install -e ".[jax]"
```

## Upstream tool readiness

Check what is currently source-ready, runtime-ready, and bootstrapped:

```python
import smckit

print(smckit.upstream.status())
```

## Runtime and bootstrap matrix

| Tool | Vendored source in repo | Runtime | Bootstrap contract | Current public bridge |
|---|---|---|---|---|
| PSMC | Yes | `make` + C compiler | build vendored source and cache `psmc` binary | Public |
| MSMC2 | Yes | `make` + D toolchain | build vendored source and cache `msmc2` binary | Public |
| MSMC-IM | Yes | Python | vendored script is the oracle entrypoint | Public |
| SMC++ | No | controlled Python env | still depends on side environment | Public, but not fully vendored |
| eSMC2 | Yes | `Rscript` / `R CMD INSTALL` | install vendored R package into `.r-lib/` | Public |
| ASMC | Yes | CMake/C++ toolchain | build vendored `ASMC_exe` and cache it | Public |
| diCal2 | Yes | Java | vendored `diCal2.jar` is the oracle artifact | Public |

## Bootstrapping examples

When a bootstrap path exists, use:

```python
import smckit

smckit.upstream.bootstrap("esmc2")
status = smckit.upstream.status("esmc2")
print(status["ready"])
```

For tools without a public bridge yet, `status()` should still tell you what is
missing so docs and code do not overclaim readiness.
