# SSM Framework

The state-space model framework is the experimental side of smckit. It
re-expresses SMC methods as explicit transition, emission, and initial-state
objects so they can be fitted with alternative optimization strategies such as
JAX-based gradient descent.

```{admonition} Best for
:class: tip
Method development, experimentation, and gradient-based alternatives to the
classic EM pipelines.
```

## What it gives you

- `SmcStateSpace` as a shared abstraction
- `PsmcSSM` as the first concrete implementation
- gradient or EM fitting routes through the same model object

## Relationship to the dual implementation philosophy

This framework is native-only today. It is not an upstream compatibility layer
and does not have an `implementation="upstream"` target of its own.

## Minimal workflow

```python
from smckit.ext.ssm import PsmcSSM
import smckit

data = smckit.io.read_psmcfa("tests/data/NA12878_chr22.psmcfa")
observations = [rec["codes"] for rec in data.uns["records"]]

model = PsmcSSM(pattern="4+5*3+4")
params = model.make_initial_params(data.uns["sum_L"], data.uns["sum_n"])
result = model.fit(observations, params, method="gradient", n_iterations=500)
```

## Practical notes

- Use this when you are exploring models or optimizers, not when you need the
  most battle-tested public workflow.
- For the motivation and practical framing, start with
  [PSMC-SSM](../guide/psmc-ssm.md).
- For standard production-style one-genome inference, use
  [PSMC](psmc.md).

## Learn more

- [PSMC-SSM guide](../guide/psmc-ssm.md)
- [API reference](../api/ssm.rst)
