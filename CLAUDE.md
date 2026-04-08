# CLAUDE.md — smckit

## Project Identity

**smckit** is a unified Python framework for Sequentially Markovian Coalescent (SMC) methods
in population genetics. It wraps, reimplements, and extends PSMC, MSMC/MSMC2, SMC++, and
future SMC-based demographic inference tools under a single, coherent API — inspired by
what Scanpy did for single-cell genomics.

The project exists to **maintain and modernize the legacy of SMC methods**: cleaning up
fragmented codebases, providing GPU-accelerated backends, and making these foundational
algorithms accessible to a broader community.

## Authorship & Commits

- **Claude and Codex must NOT appear as author or co-author in any commit.** No
  `Co-Authored-By`, no `Signed-off-by`, and no attribution to Claude or Codex in
  commit messages, PR text, or git metadata.
- Commits should be authored by the human contributor only.
- Commit messages: imperative mood, concise subject (<72 chars), optional body for why.

## Architecture Principles

### Compute Backend Strategy

**CuPy is the default and primary backend.** The code is written using
NumPy-compatible vectorized operations that run identically on CuPy (GPU)
and NumPy (CPU). CuPy is a drop-in replacement for NumPy — same API,
same broadcasting, same `@` matmul — but executes on GPU.

1. **CuPy (default/primary)** — all new algorithm implementations should
   target CuPy first. Since CuPy mirrors the NumPy API, this code also
   serves as the CPU fallback when run with NumPy arrays.
2. **NumPy (CPU fallback)** — the same vectorized code runs on CPU when CuPy
   is not installed. NumPy-only backends are provided for PSMC (the reference
   implementation), but future methods (MSMC2, SMC++) may not require a
   separate NumPy implementation — the CuPy code IS the NumPy code.
3. **CUDA (optional, performance-critical paths)** — raw CUDA kernels via
   PyCUDA or custom `.cu` extensions only where CuPy's overhead is the
   bottleneck (e.g., custom forward-algorithm kernels that fuse the matvec
   + scale + store in one kernel launch).

### Vectorization Requirements

All numerical code must be **fully vectorized** — no Python-level loops over
data dimensions (sequence length L, state count n) except where sequential
dependencies make it unavoidable:

- **Forward/backward algorithms:** The loop over L is inherently sequential
  (each position depends on the previous). The inner n-state operations MUST
  be matrix-vector products (`at @ f[u-1]`), not Python loops over states.
- **Expected counts:** Must use symbol-grouped matmuls (`f_sub.T @ b_sub`)
  instead of looping over L. This is the key operation that benefits from GPU.
- **Q-function, posterior decode, log-likelihood:** Fully vectorized, no loops.
- **compute_hmm_params:** Use `np.cumsum`, `np.cumprod`, broadcasting, and
  vectorized `np.where` for the per-state calculations. The only remaining
  loop is over n_states for the q-matrix construction (n ≈ 23, negligible).

Backend selection should be automatic (detect GPU availability) with explicit override:

```python
import smckit
smckit.settings.backend = "cupy"  # or "numpy", "cuda"
```

### API Design (Scanpy Philosophy)

- **Central data container**: like AnnData for scanpy, smckit uses a core data object
  that carries input sequences, decoded HMM states, demographic parameters, and results.
- **Functional API with side effects on the container**:
  ```python
  smckit.tl.psmc(data)          # runs PSMC, stores results in data
  smckit.tl.msmc2(data)         # runs MSMC2, stores results in data
  smckit.pl.demographic_history(data)  # plots Ne(t) from results
  ```
- **Module namespaces** mirror scanpy:
  - `smckit.pp` — preprocessing (sequence alignment to input format, masking, filtering)
  - `smckit.tl` — tools (the actual inference algorithms)
  - `smckit.pl` — plotting (demographic history, bootstrap CIs, model comparison)
  - `smckit.io` — I/O (read/write PSMC/MSMC/SMC++ native formats, VCF, tree sequences)
  - `smckit.ext` — extensions (SSM model integration, experimental methods)

### SSM (State-Space Model) Integration

The `smckit.ext.ssm` module defines SMC methods through a formal state-space model lens:

- Hidden states = coalescent time intervals (discretized)
- Observations = heterozygosity patterns along the genome
- Transition model = coalescent process with recombination
- Emission model = mutation process

This abstraction allows:
- Swapping transition/emission models for experimentation
- Connecting to broader SSM/HMM libraries (e.g., dynamax, ssm)
- Defining new SMC-flavored models without reimplementing the full pipeline

### Algorithms Roadmap

| Method   | Source Language | Status   | Priority |
|----------|---------------|----------|----------|
| PSMC     | C             | Done     | High     |
| ASMC     | C++           | Done     | High     |
| eSMC2    | R             | Done     | High     |
| MSMC2    | D             | Phase 2  | High     |
| SMC++    | C++/Python    | Done     | High     |
| MSMC     | D             | Phase 2  | Medium   |
| diCal    | Java          | Future   | Low      |

## Code Style & Standards

- Python 3.10+
- Type hints on all public API functions
- Docstrings: NumPy style (consistent with scanpy ecosystem)
- Formatting: `ruff format` (line length 99)
- Linting: `ruff check`
- Tests: `pytest`, separate directories for unit / integration / GPU tests
- GPU tests gated behind `@pytest.mark.gpu` marker

## File Organization

```
smckit/
├── CLAUDE.md
├── plan.md
├── pyproject.toml
├── src/
│   └── smckit/
│       ├── __init__.py
│       ├── _core.py          # central data container
│       ├── settings.py       # global config (backend, verbosity)
│       ├── pp/               # preprocessing
│       ├── tl/               # tools (algorithms)
│       │   ├── _psmc.py
│       │   ├── _msmc.py
│       │   ├── _smcpp.py
│       │   └── ...
│       ├── pl/               # plotting
│       ├── io/               # I/O adapters
│       ├── ext/              # extensions
│       │   └── ssm/          # state-space model framework
│       └── backends/
│           ├── _numpy.py     # CPU reference kernels
│           ├── _cupy.py      # CuPy GPU kernels
│           └── _cuda.py      # raw CUDA kernels
├── tests/
│   ├── unit/
│   ├── integration/
│   └── gpu/
├── vendor/                   # cloned upstream repos for reference
│   ├── psmc/
│   ├── msmc2/
│   └── smcpp/
└── docs/
```

## Contributing & Workflow

- Feature branches off `main`, PR-based workflow
- Every algorithm reimplementation must include:
  1. A NumPy reference implementation that matches the original tool's output
  2. Numerical validation tests comparing against the original tool
  3. GPU implementation(s) validated against the NumPy reference
- Keep vendor/ repos as read-only references — never modify them directly
- When porting from C/D/C++: first write a line-by-line Python translation,
  then refactor into idiomatic Python, then optimize with GPU backends

## Testing Conventions

- `pytest tests/unit/` — fast, CPU-only, no external data
- `pytest tests/integration/` — validates against original tool outputs
- `pytest tests/gpu/ -m gpu` — GPU backend correctness
- CI runs CPU tests on every push; GPU tests on dedicated runners

## Documentation Philosophy

The smckit docs follow the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
model: clear progressive disclosure from concepts to API, with separate
sections for new users, conceptual building blocks, contributors, and
exhaustive reference. The sidebar is organized into five sections:

1. **Get Started** — what is SMC, install, first PSMC/ASMC analyses.
2. **User Guide** — `SmcData`, I/O formats, choosing a method, interpreting
   results, plotting, gallery.
3. **Method Reference** — per-method internals (PSMC, ASMC, eSMC2, MSMC-IM,
   SMC++, SSM, MSMC2). Each page reframed from "internals reference" to
   "what it is, when to use it, quick example, then technical detail."
4. **Developer Guide** — contributing, architecture, adding a new method,
   testing.
5. **API Reference** — per-module autodoc pages (`core`, `io`, `tl`, `pl`,
   `pp`, `ssm`, `settings`).

Source layout: `docs/get-started/`, `docs/guide/`, `docs/methods/`,
`docs/developer/`, `docs/api/`. The toctree in `docs/index.rst` defines the
sidebar; never put documentation pages directly at the top level of `docs/`.

In-development features (e.g., MSMC2, `smckit.pp`) get a `{admonition} In
Development :class: warning` block at the top of their page rather than
being omitted. The status column in `docs/guide/choosing-a-method.md` is
the central place where method maturity is tracked.

## Glossary

- **SMC**: Sequentially Markovian Coalescent — approximation to the full coalescent
  that makes HMM-based inference tractable along genomes
- **Ne(t)**: Effective population size as a function of time
- **PSMC**: Pairwise SMC — infers Ne(t) from a single diploid genome
- **MSMC/MSMC2**: Multiple SMC — extends to multiple haplotypes
- **SMC++**: Extends SMC to many unphased individuals using a distinguished lineage
- **SSM**: State-Space Model — the mathematical framework underlying these HMMs
