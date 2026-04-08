# PSMC-SSM

`PSMC-SSM` is smckit's explicit state-space-model formulation of PSMC.
It exists for two reasons:

1. to make the standard PSMC HMM easier to inspect, test, and reuse
2. to make the coalescent-to-HMM map differentiable, so we can optimize it
   with gradients instead of only with classical EM

If you only want a standard demographic-history estimate from one diploid
genome, use {func}`smckit.tl.psmc`. If you want to treat PSMC as a reusable
HMM object, experiment with optimization strategies, or build new SMC-flavored
models on the same interface, use {class}`smckit.ext.ssm.PsmcSSM`.

## Why We Built It

The original `psmc` implementation is algorithmically elegant, but it mixes
several concerns together:

- the biological parameterization
- the hidden Markov model
- the EM optimizer
- the implementation details of a specific backend

That is fine for running PSMC itself, but it becomes limiting if you want to:

- inspect the transition, emission, and initial distributions directly
- swap EM for gradient-based optimization
- compare optimizer behavior on the exact same model
- reuse the same SMC model inside a broader state-space framework
- prototype new variants without rewriting the whole inference stack

`PsmcSSM` separates those concerns. It makes the HMM an explicit Python object
with a small contract:

- `transition_matrix(params)`
- `emission_matrix(params)`
- `initial_distribution(params)`
- `log_likelihood(params, observations)`
- `fit(...)`

That separation is the main point of the feature. The gradient support is the
most visible consequence, not the whole story.

## What It Is

`PsmcSSM` is still PSMC.

It uses the same hidden states, the same observation alphabet, and the same
coalescent parameterization as the standard implementation:

| Piece | PSMC meaning | `PsmcSSM` object |
|---|---|---|
| Hidden state | coalescent time interval | `z_t` |
| Observation | homozygous / heterozygous / missing window | `x_t` |
| Transition | recombination + re-coalescence | `A(theta, rho, lambda)` |
| Emission | mutation process | `E(theta)` |
| Initial distribution | stationary state weights | `pi(theta, rho, lambda)` |

So the SSM version is not a different demographic model. It is a different
software representation of the same one.

## What It Buys You

### 1. Explicit HMM access

You can ask for the matrices directly:

```python
from smckit.ext.ssm import PsmcSSM

model = PsmcSSM(pattern="4+5*3+4")
a = model.transition_matrix(params)
e = model.emission_matrix(params)
pi = model.initial_distribution(params)
```

That is useful for debugging, educational plots, regression tests, and custom
inference code.

### 2. A unified fitting interface

The same model object supports both:

- `method="em"` for the classical PSMC fitting path
- `method="gradient"` for JAX-based direct optimization of the likelihood

That makes optimizer comparisons much cleaner because the model stays fixed
while only the fitting strategy changes.

### 3. A clean extension point

Once PSMC is expressed as a `SmcStateSpace`, new SMC-style models can reuse the
same structure. The important abstraction is: define the HMM pieces once, then
plug them into shared likelihood and fitting machinery.

## When To Use It

Use `PsmcSSM` when:

- you want to experiment with PSMC as a model, not only as a CLI-style method
- you want gradient-based fitting with JAX
- you want direct access to the HMM components
- you want to implement a new SMC method on the `SmcStateSpace` interface

Do not use it when:

- you just want the standard single-genome `N_e(t)` workflow
- you do not need matrix-level access or optimizer experiments
- you want the simplest, most validated entry point

For that case, `smckit.tl.psmc()` remains the default.

## How To Use It

### Standard setup

```python
import smckit
from smckit.ext.ssm import PsmcSSM

data = smckit.io.read_psmcfa("sample.psmcfa")
observations = [rec["codes"] for rec in data.uns["records"]]

model = PsmcSSM(pattern="4+5*3+4")
params = model.make_initial_params(
    data.uns["sum_L"],
    data.uns["sum_n"],
    tr_ratio=5.0,
    seed=42,
)
```

### EM fitting

```python
result = model.fit(
    observations,
    params,
    method="em",
    n_iterations=25,
)
```

### Gradient fitting

```python
result = model.fit(
    observations,
    params,
    method="gradient",
    optimizer="adam",
    learning_rate=0.01,
    n_iterations=500,
)
```

### Convert back to demographic history

```python
phys = model.to_physical_units(
    result.params,
    mu=1.25e-8,
    generation_time=25.0,
    window_size=data.window_size,
)
```

## How It Is Implemented

The implementation is split into a small number of files with clear roles:

| File | Responsibility |
|---|---|
| `src/smckit/ext/ssm/_base.py` | abstract `SmcStateSpace` interface and `FitResult` |
| `src/smckit/ext/ssm/_psmc_ssm.py` | concrete PSMC state-space model |
| `src/smckit/ext/ssm/_numpy_backend.py` | EM fitting wrapper using the existing Numba kernels |
| `src/smckit/ext/ssm/_jax_backend.py` | differentiable JAX implementation and gradient optimizer |
| `src/smckit/tl/_psmc.py` | standard top-level PSMC workflow |

Conceptually, the stack looks like this:

```text
coalescent parameters
        ↓
  PsmcSSM._hmm_params()
        ↓
 A, E, pi matrices
        ↓
 log-likelihood / fit()
        ↓
 fitted params
        ↓
 demographic history in physical units
```

## How `method="em"` Relates To `smckit.tl.psmc()`

The EM path in `PsmcSSM` is meant to be the same model and the same fitting
logic as standard PSMC, but exposed through the SSM interface.

In the current realistic fixture coverage:

- `PsmcSSM.fit(..., method="em")` matches `smckit.tl.psmc()` exactly in
  statewise `lambda`
- `smckit.tl.psmc()` remains very close to the original C PSMC reference
- the remaining visible per-state error in the docs figure is therefore the
  smckit-vs-C port gap, not an SSM-vs-PSMC gap

This matters because it means the SSM layer is not changing the demographic
model. It is reorganizing the implementation.

## How `method="gradient"` Works

The gradient path replaces EM with direct optimization of the marginal
log-likelihood:

$$
\hat\phi = \arg\min_\phi -\log P(x \mid \phi)
$$

where:

- `phi = (theta, rho, max_t, lambda_0, ..., lambda_K)`
- `x` is the observed sequence of homozygous / heterozygous / missing windows

The important implementation move is that the HMM construction is rewritten in
JAX so the full likelihood remains differentiable. Once that exists, standard
optimizers like Adam become available.

This does not automatically make the gradient path better than EM. It makes it
possible to study that tradeoff directly.

## Benchmark Snapshot

On the bundled `NA12878_chr22.psmcfa` fixture in this local development
environment, the current wall-clock timings are:

| Run | Setting | Time |
|---|---|---:|
| `smckit.tl.psmc()` | 10 EM iterations | `2.02 s` |
| `PsmcSSM.fit(..., method="em")` | 10 EM iterations | `1.99 s` |
| `PsmcSSM.fit(..., method="gradient")` | Adam, 50 iterations, `lr=0.01` | `63.38 s` |

These numbers should be read as relative guidance, not fixed promises:

- the EM path in `PsmcSSM` is essentially the same cost as standard PSMC
- the gradient path is much more expensive on CPU
- GPU-backed JAX runs can change that tradeoff substantially
- gradient timing depends heavily on iteration count, optimizer, backend, and
  whether JAX compilation has already happened

The key practical takeaway is simple: use `method="em"` when you want standard
PSMC-like throughput, and use `method="gradient"` when you specifically need
the differentiable fitting path enough to pay for it.

## How To Implement A New SMC Model On This Pattern

If you want to build a new SMC-flavored model, `PsmcSSM` is the template.

### Step 1: define the hidden states and observations

Be precise about:

- what each hidden state means biologically
- what each observation symbol means in the input encoding
- which parameters control transition and emission behavior

### Step 2: subclass `SmcStateSpace`

```python
from smckit.ext.ssm import SmcStateSpace

class MyMethodSSM(SmcStateSpace):
    def transition_matrix(self, params):
        ...

    def emission_matrix(self, params):
        ...

    def initial_distribution(self, params):
        ...
```

That is the core contract. Once those methods are correct, the shared
likelihood and fitting interfaces become usable.

### Step 3: add a parameter initializer

In practice you almost always want a `make_initial_params(...)` helper, because
raw optimization from arbitrary scales is brittle.

### Step 4: add physical-unit conversion

Most SMC models are easier to optimize in internal units but easier to inspect
in physical units. A `to_physical_units(...)` method keeps that boundary clear.

### Step 5: validate against an oracle

At minimum, check:

- matrix-level agreement with the intended backend
- likelihood agreement on small synthetic examples
- regression behavior on a realistic fixture

### Step 6: only then add a gradient path

Do not start with autodiff. First get the HMM itself correct and testable.
The JAX version is much easier to trust once a stable non-JAX reference exists.

## Recommended Reading Order

If you are approaching this for the first time:

1. read {doc}`../methods/psmc` to understand the underlying model
2. use this page for the motivation and software structure
3. read {doc}`../methods/ssm` for the API and backend details
4. read {doc}`../developer/adding-a-method` if you want to add a new method

## See Also

- {doc}`../methods/ssm`
- {doc}`../methods/psmc`
- {doc}`choosing-a-method`
- {doc}`../developer/adding-a-method`
