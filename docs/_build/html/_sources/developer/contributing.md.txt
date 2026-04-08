# Contributing

smckit is an open project — contributions are welcome, whether they are bug
reports, documentation fixes, new SMC method implementations, or backend
optimizations. This page covers the day-to-day contributor workflow. For an
overview of the project's design philosophy, see
[Architecture](architecture.md). For a step-by-step guide to adding a new
method, see [Adding a Method](adding-a-method.md).

## Development setup

Clone the repository and install in editable mode with the development
extras:

```bash
git clone https://github.com/kevinkorfmann/smckit.git
cd smckit
pip install -e ".[dev]"
```

The `dev` extra pulls in `pytest`, `ruff`, and the documentation tooling.
For the JAX-based PSMC-SSM features, also install:

```bash
pip install -e ".[jax]"
```

## Code style

smckit uses [ruff](https://docs.astral.sh/ruff/) for both formatting and
linting:

```bash
ruff format src/ tests/
ruff check src/ tests/
```

Style rules:

- **Python 3.10+** features only.
- **Line length 99**.
- **Type hints** on all public-API functions.
- **NumPy-style docstrings** (compatible with `sphinx.ext.napoleon`).

## Running tests

Tests are split into three tiers (see [Testing](testing.md) for the
philosophy):

```bash
pytest tests/unit/         # fast, CPU-only, no external data
pytest tests/integration/  # validates against original tool outputs
pytest tests/gpu/ -m gpu   # GPU backend tests (require a CUDA device)
```

CI runs the unit and integration tiers on every push.

## Pull request workflow

1. Create a feature branch off `main`:
   ```bash
   git checkout -b feat/my-new-thing
   ```
2. Make focused commits with imperative-mood subjects under 72 characters:
   ```
   add MSMC2 reimplementation with Numba JIT
   fix MSMC2 theta estimation to match D code
   ```
3. Run formatting, linting, and tests locally before pushing.
4. Open a PR against `main`. Describe *why* the change exists, not just
   *what* it does — the diff already shows what.
5. CI must be green before merge.

## Reporting bugs

When filing an issue, please include:

- The smckit version (`pip show smckit`).
- A minimal reproducer (or pseudo-code if real data can't be shared).
- The exact error message and traceback.
- The compute backend (CPU/GPU, Numba/JAX).

## Asking questions

For usage questions that aren't bugs, use the GitHub Discussions tab. For
methodological questions about specific SMC methods, the corresponding
upstream tool repositories are often the best venue.
