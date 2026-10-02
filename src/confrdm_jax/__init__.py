"""Neural likelihood estimation for racing diffusion models of conflict tasks.

The models, priors and simulators live here; the accumulator densities, the race
likelihood, the parameterization layer, the neural flow, the hierarchical prior family and
the samplers all come from `eamax <https://github.com/maltelueken/eamax>`_.

What this package still owns is what the three `eamax` consumers genuinely disagree about:

* :mod:`confrdm_jax.specs` — the two parameter vectors (``rdm``, ``crdm``), their links, and
  how they map onto accumulator drifts, noises and thresholds.
* :mod:`confrdm_jax.priors` — the ``distrax.Joint`` training and recovery priors, and the
  hierarchical prior's hyperparameters.
* :mod:`confrdm_jax.likelihoods` — the hybrid CRDM race, where the flow covers the pulsed
  accumulator and the inverse Gaussian the rest.
* :mod:`confrdm_jax.simulators` — the samplers the training loop and the recovery scripts
  call, in the shapes those callers expect.

:mod:`confrdm_jax.runtime` sets 64-bit precision and the compute device; call
:func:`~confrdm_jax.runtime.configure_jax` before anything else touches JAX.
"""

from .runtime import configure_jax

__all__ = ["configure_jax"]
