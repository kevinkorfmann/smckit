# Adding a Method

This guide walks through adding a new SMC method to smckit. The pattern is
the same for any new tool: write an internals document, implement an I/O
reader, implement the inference function, store results in the standard
shape, validate against an original implementation, and add tests.

## Overview

A complete new method consists of:

1. An internals document at `docs/methods/<method>.md` (algorithm + math)
2. An I/O reader at `src/smckit/io/_<method>.py`
3. The inference tool at `src/smckit/tl/_<method>.py`
4. A unit test at `tests/unit/test_<method>.py`
5. A validation test at `tests/integration/test_<method>_validation.py`
6. (Optional) Plotting support for `pl.demographic_history`
7. (Optional) An SSM subclass at `src/smckit/ext/ssm/_<method>_ssm.py`

New methods should also decide their two implementation paths up front:

- `implementation="upstream"`: how smckit will call the original tool.
- `implementation="native"`: how the in-repo port will expose the same method.

It is acceptable to ship the upstream bridge first and leave the native path as
future work, but the public API contract should be designed for both from day
one.

New methods must also declare their upstream readiness contract in
`smckit.upstream`: vendored source path, runtime requirement, bootstrap
expectations, and how readiness is detected. If the project cannot say how the
upstream tool becomes runnable from the repo, the method is not integrated
cleanly yet.

## Step 1: Write the internals document

Before touching code, document the algorithm. The internals pages in
`docs/methods/` follow a consistent structure:

- **Source file map** — which files in the original implementation do what.
- **Input format** — the on-disk file format with examples.
- **HMM structure** — hidden states, observations, transition/emission.
- **Inference algorithm** — EM, gradient, or other optimization.
- **Output format** — what the original tool produces.
- **C-to-Python function mapping** — for porting.

Use any of the existing pages (`docs/methods/psmc.md`, `docs/methods/asmc.md`,
`docs/methods/smcpp.md`) as a template.

This step is not optional. The internals document is what allows reviewers
(and future-you) to verify the implementation against the math.

## Step 2: Implement the I/O reader

Add a module under `src/smckit/io/`:

```python
# src/smckit/io/_<method>.py
from smckit._core import SmcData

def read_<method>(path: str) -> SmcData:
    """Read <method> input format and return an SmcData container."""
    data = SmcData()
    # ... parse the file, populate data.sequences / data.uns / data.params
    return data
```

Then export it from `src/smckit/io/__init__.py`:

```python
from smckit.io._<method> import read_<method>

__all__ = [
    # ... existing entries
    "read_<method>",
]
```

For methods that wrap an upstream tool's output (rather than reading raw
sequence data), provide both a reader for the input format and a reader for
the output format, mirroring `read_psmc_output` and `read_msmc_im_output`.

## Step 3: Implement the inference tool

Add a module under `src/smckit/tl/`:

```text
# src/smckit/tl/_<method>.py
from dataclasses import dataclass
from smckit._core import SmcData

@dataclass
class <Method>Result:
    theta: float
    rho: float
    lambda_: np.ndarray
    ne: np.ndarray
    time_years: np.ndarray
    log_likelihood: float
    rounds: list

def <method>(data: SmcData, **kwargs) -> SmcData:
    """Run <method> inference on data, storing results in data.results.

    Parameters
    ----------
    data : SmcData
        Container holding input sequences from io.read_<method>.
    ...

    Returns
    -------
    SmcData
        The same container, with `results["<method>"]` populated.
    """
    # ... run the algorithm
    data.results["<method>"] = {
        "theta": ...,
        "rho": ...,
        "lambda": ...,
        "ne": ...,
        "time_years": ...,
        "log_likelihood": ...,
        "rounds": [...],
    }
    return data
```

Public tools should accept `implementation={"auto","native","upstream"}`.
For methods with an existing historical `backend=` selector for provenance,
keep it only as a compatibility alias and normalize internally to
`implementation`.

When the original tool exposes meaningful controls that do not fit the common
cross-method surface, prefer structured method-specific options such as
`upstream_options` rather than silently dropping that functionality.

Then export it from `src/smckit/tl/__init__.py`:

```python
from smckit.tl._<method> import <method>

__all__ = [
    # ... existing entries
    "<method>",
]
```

### Result key conventions

The {func}`smckit.pl.demographic_history` function expects these keys in
`data.results[method]`:

- `ne` — array of $N_e$ values per state
- `time_years` — array of state boundary times in years (or `time` + `n0`
  for fallback conversion)

If your method produces these, plotting will work out of the box. If your
method produces something else (e.g., MSMC-IM produces $N_1, N_2, m, M$),
add a custom plotter under `src/smckit/pl/`.

## Step 4: Numba JIT (optional but encouraged)

For per-site/per-state inner loops, write a Numba-compiled kernel under
`src/smckit/backends/`:

```python
# src/smckit/backends/_numba_<method>.py
import numba

@numba.njit(cache=True)
def forward_<method>(...):
    ...
```

This pattern is used by `_numba.py` (PSMC) and `_numba_esmc2.py`.

## Step 5: Validation against the original tool

Add an integration test that compares smckit output with the upstream
reference implementation:

```python
# tests/integration/test_<method>_validation.py
def test_matches_original_<method>():
    data = smckit.io.read_<method>("vendor/<method>/example_input")
    data = smckit.tl.<method>(data, **default_params)

    expected = parse_original_output("vendor/<method>/example_output")
    np.testing.assert_allclose(
        data.results["<method>"]["lambda"],
        expected["lambda"],
        rtol=1e-3,
    )
```

The `vendor/` directory contains reference clones of every upstream tool —
use them. **Never modify vendor files.**

If the upstream bridge does not exist yet, `implementation="upstream"` should
raise a precise `NotImplementedError` rather than silently falling back.

## Step 6: Unit tests

Add a unit test that exercises the public API on a tiny synthetic input:

```python
# tests/unit/test_<method>.py
def test_<method>_runs():
    data = make_tiny_smc_data()
    data = smckit.tl.<method>(data, n_iterations=2)
    assert "<method>" in data.results
```

Unit tests should run in well under a second and require no external data.

## Step 7: Add to method comparison

Update `docs/guide/choosing-a-method.md` to add your method to the
comparison table and decision tree.

## Step 8: SSM subclass (optional)

If your method fits into the state-space model framework, add a subclass
under `src/smckit/ext/ssm/`:

```text
# src/smckit/ext/ssm/_<method>_ssm.py
from smckit.ext.ssm._base import SmcStateSpace

class <Method>SSM(SmcStateSpace):
    def transition_matrix(self, params): ...
    def emission_matrix(self, params): ...
    def initial_distribution(self, params): ...
```

This gives you gradient-based fitting, log-likelihood evaluation, and
composability with the rest of the SSM framework.

## See also

- **[Architecture](architecture.md)** — design principles and module layout.
- **[Testing](testing.md)** — full testing conventions.
- The existing **[Method Reference](../methods/psmc.md)** pages — templates
  for new internals documents.
