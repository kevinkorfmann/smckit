# Architecture

This page describes how smckit is organized internally and the design
principles behind it. It is intended for contributors and for users who want
to understand the trade-offs of the framework.

## Project identity

smckit is a unified Python framework for Sequentially Markovian Coalescent
(SMC) methods in population genetics. It wraps, reimplements, and extends
PSMC, MSMC/MSMC2, SMC++, ASMC, eSMC2, MSMC-IM, and future SMC-based
demographic inference tools under a single, coherent API — inspired by what
[scanpy](https://scanpy.readthedocs.io) did for single-cell genomics.

The project exists to **maintain and modernize SMC methods**:
keep the original algorithms usable now, expose them through one coherent API,
and replace them with native implementations incrementally where that becomes
practical.

## API design (scanpy philosophy)

smckit borrows three structural ideas from scanpy:

1. **A central data container.** All input, intermediate state, and results
   live in a single {class}`smckit.SmcData` object — no juggling of arrays
   between functions.

2. **Functional API with side effects on the container.** Tools mutate the
   container and return it, enabling pipeline chaining:
   ```python
   data = smckit.io.read_psmcfa("sample.psmcfa")
   data = smckit.tl.psmc(data, pattern="4+5*3+4")
   smckit.pl.demographic_history(data)
   ```

3. **Module namespaces** that mirror scanpy's:

   | Namespace | Purpose |
   |---|---|
   | {mod}`smckit.pp` | Preprocessing — alignment to input format, masking, filtering |
   | {mod}`smckit.tl` | Tools — the actual inference algorithms |
   | {mod}`smckit.pl` | Plotting — demographic history, bootstrap CIs, comparisons |
   | {mod}`smckit.io` | I/O — read/write native formats, VCF, tree sequences |
   | {mod}`smckit.ext` | Extensions — SSM framework, experimental methods |
   | `smckit.backends` | Compute kernels (NumPy reference, Numba JIT, future GPU) |

## Implementation provenance

Method selection is distinct from compute backend selection.

- `implementation="upstream"` means "run the original algorithm" through a
  controlled bridge to the vendored upstream tool.
- `implementation="native"` means "run the in-repo implementation".
- `implementation="auto"` currently prefers upstream when available, because
  preserving full original behavior takes priority over promoting partial native
  ports too early.

This is separate from `smckit.settings.backend`, which controls how native code
executes (`numpy`, `numba`, and future GPU options).

## Upstream registry and bootstrap contract

Upstream availability is now explicit. The `smckit.upstream` package owns:

- the registry of vendored tools and their runtime requirements
- readiness checks (`source_present`, `runtime_ready`, `cache_ready`)
- local bootstrap hooks for tools that can be prepared from the repo alone
- provenance metadata that is attached to `data.results[method]["upstream"]`

The intended contract is:

- `vendor/` contains the oracle source or release artifact
- repo-local caches hold generated executables or runtime installs
- `smckit.upstream.status()` reports truth about readiness
- `smckit.upstream.bootstrap()` builds local artifacts where supported

This avoids documentation drift where `implementation="auto"` claims to prefer
upstream while the code silently relies on partial or method-specific logic.

## Compute backend strategy

The current production backend is **Numba JIT** for everything performance
sensitive. Numba compiles tight loops over states and sequence positions to
native code, achieving parity with — or better than — the original C
implementations.

The longer-term plan is to support **CuPy** and raw **CUDA** backends as
drop-in replacements for the NumPy/Numba paths. CuPy mirrors the NumPy API,
so most vectorized code runs unchanged on GPU. Custom CUDA kernels are
reserved for performance-critical inner loops where CuPy's launch overhead
becomes the bottleneck (e.g., fused matvec + scale + store in the forward
algorithm).

The selection model is:

```python
import smckit
smckit.settings.backend = "numba"  # or "numpy", "cupy", "cuda" (future)
```

## Vectorization requirements

All numerical code must be **fully vectorized**. The only Python-level loops
allowed are over dimensions that are *inherently sequential*:

- **Forward/backward algorithms.** The loop over sequence length L is
  unavoidable (each position depends on the previous). The inner n-state
  operations must be matvecs (`A @ f[t-1]`), not Python loops.
- **Expected counts.** Use symbol-grouped matmuls (`f_sub.T @ b_sub`) instead
  of looping over L. This is the operation that benefits most from GPU.
- **HMM parameter construction.** Use `cumsum`, `cumprod`, broadcasting, and
  vectorized `where` for per-state calculations.

## SSM (state-space model) integration

The {mod}`smckit.ext.ssm` module defines SMC methods through a formal
state-space model lens:

- **Hidden states** = coalescent time intervals (discretized)
- **Observations** = heterozygosity patterns along the genome
- **Transition model** = coalescent process with recombination
- **Emission model** = mutation process

This abstraction allows:

- Swapping transition/emission models for experimentation
- Connecting to broader SSM/HMM libraries (`dynamax`, `ssm`)
- Defining new SMC-flavored models without reimplementing the full pipeline
- **Differentiable inference** via JAX, enabling gradient-based optimization
  as an alternative to EM

## Algorithms roadmap

| Method   | Source language | Status            | Priority |
|----------|-----------------|-------------------|----------|
| PSMC     | C               | Done              | High     |
| ASMC     | C++             | Done              | High     |
| eSMC2    | R               | Done              | High     |
| MSMC2    | D               | In development    | High     |
| SMC++    | C++/Python      | Done              | High     |
| MSMC-IM  | Python          | Done              | High     |
| MSMC     | D               | Phase 2           | Medium   |
| diCal    | Java            | Future            | Low      |

## File organization

```
smckit/
├── CLAUDE.md
├── pyproject.toml
├── src/
│   └── smckit/
│       ├── __init__.py
│       ├── _core.py          # SmcData
│       ├── settings.py       # global config
│       ├── pp/               # preprocessing
│       ├── tl/               # tools (algorithms)
│       │   ├── _psmc.py
│       │   ├── _msmc.py      # MSMC2
│       │   ├── _msmc_im.py
│       │   ├── _esmc2.py
│       │   ├── _asmc.py
│       │   └── _smcpp.py
│       ├── pl/               # plotting
│       ├── io/               # I/O adapters
│       ├── ext/              # extensions
│       │   └── ssm/          # state-space model framework
│       └── backends/
│           ├── _numpy.py     # CPU reference kernels
│           ├── _numba.py     # Numba JIT kernels
│           └── _numba_esmc2.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── gpu/
├── vendor/                   # cloned upstream repos for reference
└── docs/
```

The `vendor/` directory contains read-only clones of the upstream tool
implementations (psmc, msmc2, smcpp, ASMC, eSMC2, MSMC-IM). They are used
as the source of truth when porting algorithms but never modified.

## Three-tier validation

Every algorithm reimplementation must pass three layers of validation:

1. **Numerical equivalence with the original tool.** A NumPy reference
   implementation reproduces the original C/D/R output bit-for-bit (or to
   machine precision).
2. **Unit and integration tests.** Tests in `tests/unit/` and
   `tests/integration/` lock in the expected behavior.
3. **Backend equivalence.** Numba and (future) GPU backends are validated
   against the NumPy reference.

This is what allows the project to confidently swap backends without
introducing silent regressions.

## See also

- **[Contributing](contributing.md)** — day-to-day developer workflow.
- **[Adding a Method](adding-a-method.md)** — step-by-step guide to
  porting a new SMC method.
- **[Testing](testing.md)** — test organization and validation patterns.
