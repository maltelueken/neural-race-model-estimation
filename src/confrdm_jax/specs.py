"""Parameterizations: the two models' parameter vectors, as `eamax` specs.

A :class:`eamax.design.Parameterization` says three things at once — what the parameter
vector holds and in what order, what link each entry is on, and how those entries become
per-accumulator drifts, noises and thresholds. Both the likelihood
(:mod:`confrdm_jax.likelihoods`) and the simulators (:mod:`confrdm_jax.simulators`) go
through the same spec object, so the model that generates the data and the model that
scores it cannot drift apart.

**Accumulator indexing**, shared by both models: index 0 is the "false" / non-target
accumulator, index 1 is the "true" / target one. Responses are coded ``0``/``1`` to match,
hence ``first_response=0``. The design's ``target`` column is therefore all ones — the
target accumulator is index 1 on every trial — and is what the ``target()`` /
``nontarget()`` contrasts read.

**Drift and noise.** The non-target accumulator drifts at ``v_intercept`` and the target at
``v_intercept + v_slope``; the non-target's within-trial noise is pinned to ``noise_scale``
and the target's is the free ``s_true``. Fixing one noise coefficient is what makes the rest
identifiable — only drift-to-noise ratios are — so ``s_true`` is a relative quantity.

**Conflict.** The CRDM adds a gamma-shaped pulse of peak height ``amp`` and time scale
``tau`` to exactly one accumulator, selected by the design's ``distractor`` column. Under
the two-condition design that column *is* the congruency indicator: a congruent trial's
distracting feature points at the target (accumulator 1), an incongruent one's at the
non-target (accumulator 0). That replaces the sign-of-``amp`` routing the old simulators
used; the pulse is now zero on every accumulator but the distractor's, by construction.

**``t0`` is last, and log-linked**, in both specs. :class:`eamax.inference.init.T0Support`
locates it by name and raises if either stops holding, so the convention is checked rather
than assumed.
"""

import jax.numpy as jnp
from eamax.design import (
    TrialDesign,
    coef,
    constant,
    distractor,
    intercept,
    log,
    nontarget,
    parameterization,
    quantity,
    target,
    term,
)

#: Parameter names in vector order, per model. Also the coordinate labels in the stored
#: ``.nc`` files, so they are part of the output format.
RDM_PARAM_NAMES = ("v_intercept", "v_slope", "s_true", "b", "t0")
CRDM_PARAM_NAMES = ("v_c_intercept", "v_c_slope", "amp", "tau", "s_true", "b", "t0")

#: Conditioning sets of the trained flows, in the order they were trained on. Written to the
#: checkpoint sidecar by ``scripts/train_conditioner.py`` and checked on load — a reordered
#: context of the same width is otherwise a silent wrong-density bug.
WALD_CONTEXT_NAMES = ("v", "s", "b")
CRDM_CONTEXT_NAMES = ("v", "amp", "tau", "s", "b")

#: Data columns, in the order both simulators emit them. The CRDM has the third.
DATA_COL_NAMES = ("rt", "choice", "condition")

#: How many *trailing* parameters a hierarchical prior treats as centered: ``b`` and ``t0``,
#: the two the data pins down hardest per subject.
NUM_CENTERED = 2


def rdm_spec(noise_scale=1.0, hyperparameters=None):
    """The two-accumulator RDM: ``[v_intercept, v_slope, s_true, b, t0]``.

    Parameters
    ----------
    noise_scale : float, optional
        The non-target accumulator's within-trial noise, held fixed for identification.
    hyperparameters : dict, optional
        Hierarchical prior arrays (``inverse_gamma_scale``, ``mu_loc``, ``mu_scale``),
        carried on the spec so :meth:`eamax.hierarchical.HierarchicalLKJMVNPrior.from_spec`
        can read them.

    Returns
    -------
    eamax.design.Parameterization
    """
    v_intercept = coef("v_intercept", log())
    v_slope = coef("v_slope", log())
    s_true = coef("s_true", log())
    b = coef("b", log())
    t0 = coef("t0", log())
    return parameterization(
        order=[v_intercept, v_slope, s_true, b, t0],
        quantities=[
            quantity("v", term(v_intercept, intercept()), term(v_slope, target())),
            quantity("s", term(constant(noise_scale), nontarget()), term(s_true, target())),
            quantity("b", term(b, intercept())),
            quantity("t0", term(t0, intercept())),
        ],
        num_responses=2,
        num_centered=NUM_CENTERED,
        first_response=0,
        hyperparameters=hyperparameters,
    )


def crdm_spec(noise_scale=1.0, hyperparameters=None):
    """The conflict RDM: ``[v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]``.

    :func:`rdm_spec` plus the conflict pulse. ``amp`` rides the distractor accumulator only
    and ``tau`` is shared across accumulators, so a pulsed accumulator sees ``amp = 0``
    everywhere the pulse does not apply and reduces to a plain Wald there.

    Parameters
    ----------
    noise_scale : float, optional
        The non-target accumulator's within-trial noise, held fixed for identification.
    hyperparameters : dict, optional
        See :func:`rdm_spec`.

    Returns
    -------
    eamax.design.Parameterization
    """
    v_intercept = coef("v_c_intercept", log())
    v_slope = coef("v_c_slope", log())
    amp = coef("amp", log())
    tau = coef("tau", log())
    s_true = coef("s_true", log())
    b = coef("b", log())
    t0 = coef("t0", log())
    return parameterization(
        order=[v_intercept, v_slope, amp, tau, s_true, b, t0],
        quantities=[
            quantity("v", term(v_intercept, intercept()), term(v_slope, target())),
            quantity("s", term(constant(noise_scale), nontarget()), term(s_true, target())),
            quantity("b", term(b, intercept())),
            quantity("amp", term(amp, distractor())),
            quantity("tau", term(tau, intercept())),
            quantity("t0", term(t0, intercept())),
        ],
        num_responses=2,
        num_centered=NUM_CENTERED,
        first_response=0,
        hyperparameters=hyperparameters,
    )


#: Spec builders by model name, for config-driven lookup.
SPECS = {"rdm": rdm_spec, "crdm": crdm_spec}


def spec_for(name, **kwargs):
    """The spec named `name` (``"rdm"`` or ``"crdm"``)."""
    try:
        builder = SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown model spec {name!r}; expected one of {sorted(SPECS)}.") from None
    return builder(**kwargs)


def make_design(data, mask=None):
    """A :class:`eamax.design.TrialDesign` from one dataset's ``[rt, choice, (condition)]``.

    The two columns the spec needs but the data do not carry are filled in here, because
    both are constants of this experiment rather than observations:

    * ``target`` is the scalar 1 — the target accumulator is index 1 on every trial. A scalar
      rather than a ``(T,)`` column of ones, because that is how `eamax` learns the RDM's
      accumulator parameters are the same on every trial: a
      :class:`~eamax.flows.FlowAccumulator` then runs its conditioner once per accumulator
      instead of once per trial, which is the difference between a hierarchical SMC cloud
      fitting on the GPU and not. The densities are identical either way.
    * ``distractor`` is the congruency column, which names the accumulator the conflict
      pulse rides. A two-column (RDM) dataset has no conflict and gets none.

    Parameters
    ----------
    data : array
        Shape ``(T, 2)`` for the RDM or ``(T, 3)`` for the CRDM.
    mask : array, optional
        Boolean, shape ``(T,)``. False trials contribute exactly zero to the likelihood.

    Returns
    -------
    TrialDesign
    """
    data = jnp.asarray(data)
    rt = data[:, 0]
    condition = data[:, 2] if data.shape[-1] > 2 else None
    return TrialDesign(
        rt=rt,
        response=data[:, 1],
        target=1,
        condition=condition,
        distractor=condition,
        mask=mask,
        first_response=0,
    )
