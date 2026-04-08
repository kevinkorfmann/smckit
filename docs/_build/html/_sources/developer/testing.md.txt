# Testing

smckit uses `pytest` and a three-tier test organization. The goals are
fast feedback during development, confidence in numerical correctness, and
GPU coverage where applicable.

## Test organization

```
tests/
├── unit/          # fast, CPU-only, no external data
├── integration/   # validates against original tool outputs
└── gpu/           # GPU backend correctness
```

### Unit tests (`tests/unit/`)

Unit tests exercise the public API of individual modules with **synthetic,
in-memory data**. They should:

- Run in well under a second each.
- Require no files on disk other than what they create.
- Have no network or external-tool dependencies.
- Be safe to run in parallel.

```bash
pytest tests/unit/
```

CI runs the full unit suite on every push.

### Integration tests (`tests/integration/`)

Integration tests **validate the smckit reimplementation against the
original tool's output** on real example data. They are the source of truth
that the reimplementation is correct.

```bash
pytest tests/integration/
```

Each integration test typically:

1. Loads an example input from `vendor/<tool>/example_data/`.
2. Runs the smckit version with the same parameters as the original tool.
3. Loads the original tool's output from a checked-in golden file.
4. Asserts numerical equivalence within tolerance.

```python
def test_psmc_matches_c_reference():
    data = smckit.io.read_psmcfa("vendor/psmc/example.psmcfa")
    data = smckit.tl.psmc(data, pattern="4+5*3+4", n_iterations=25)

    c_rounds = smckit.io.read_psmc_output("vendor/psmc/example.psmc")
    expected = c_rounds[-1]

    np.testing.assert_allclose(
        data.results["psmc"]["lambda"],
        expected["lambda"],
        rtol=1e-3,
    )
```

Tolerances depend on the method. PSMC and MSMC-IM should match to machine
precision; ASMC has stochastic CSFS sampling so use ~1% tolerance.

### GPU tests (`tests/gpu/`)

GPU tests are gated behind a `pytest` marker so they only run on machines
with a CUDA device:

```python
import pytest

@pytest.mark.gpu
def test_psmc_gpu_matches_cpu():
    ...
```

```bash
pytest tests/gpu/ -m gpu
```

CI runs GPU tests on dedicated runners only.

## The three-tier validation principle

For every algorithm in smckit, three levels of correctness must be checked:

1. **Reference level.** A NumPy implementation reproduces the original
   tool's output bit-for-bit (or to machine precision). This is the math
   anchor.
2. **Backend level.** The Numba JIT (and future CuPy/CUDA) implementations
   are validated against the NumPy reference. This decouples math
   correctness from performance optimization.
3. **End-to-end level.** Integration tests in `tests/integration/` exercise
   the full pipeline through the public API.

This pattern is what allows the project to swap backends or optimize
kernels without introducing silent numerical regressions.

## Writing a new test

Pick the right tier:

- **Unit** — testing a single function or class with synthetic input.
- **Integration** — comparing smckit output to a reference tool.
- **GPU** — testing a GPU-only kernel or backend equivalence.

Naming convention: `tests/<tier>/test_<module>.py`. Test functions are
discovered automatically by pytest.

For numerical comparisons, prefer `np.testing.assert_allclose` over manual
`abs(...) < eps` checks — the failure messages are much more informative.

## Running everything

```bash
pytest tests/unit/ tests/integration/    # CPU-only, fast feedback
pytest tests/gpu/ -m gpu                  # GPU machines only
pytest                                    # everything (skips gpu without marker)
```

## Continuous integration

CI configuration runs:

- `ruff format --check` and `ruff check` on every push.
- `pytest tests/unit/ tests/integration/` on every push.
- `pytest tests/gpu/ -m gpu` on dedicated GPU runners (when configured).

A PR cannot be merged with red CI.

## See also

- **[Contributing](contributing.md)** — day-to-day workflow.
- **[Adding a Method](adding-a-method.md)** — where the test patterns
  shown here come from.
