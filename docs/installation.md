# Installation

## From source

```bash
git clone https://github.com/kevinkorfmann/smckit.git
cd smckit
pip install -e ".[dev]"
```

## Dependencies

- Python >= 3.10
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.7
- Numba >= 0.59

Numba is required for the JIT-compiled compute kernels. The first call
to `smckit.tl.psmc()` triggers JIT compilation (~1s), which is cached
to disk for subsequent runs.
