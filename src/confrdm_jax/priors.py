"""Priors — the part of the model `eamax` deliberately does not own.

Three repositories share the racing diffusion model's densities and simulators but disagree
scientifically about what a plausible parameter is, so `eamax` stops at the likelihood
boundary and the priors stay here. Two kinds live in this module:

* **Single-subject** ``distrax.Joint`` factories over the natural scale. Each is used twice
  — to generate test data, and (with the log-transform Jacobian added by the recovery
  script) as the MCMC log-prior — so the two can never drift apart. The live
  hyperparameters are in ``conf_jax/prior/*.yaml``; the defaults here only document the
  intended magnitudes.
* **Hierarchical** wrappers over :class:`eamax.hierarchical.HierarchicalLKJMVNPrior`, whose
  hyperparameters live in ``conf_jax/model/*.yaml`` under ``hierarchical.prior``.

A *training* prior (``wald_uniform``, ``crdm_single_uniform``) is not an inference prior: it
is the box the flow is fitted over, and it has to cover everything a sampler can plausibly
visit under the recovery prior, because the flow interpolates and does not extrapolate.

``distrax.Joint`` priors are passed to jitted samplers as static arguments and hashed by
identity, so reuse one object rather than rebuilding it per call — a fresh object
recompiles.
"""

from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
from distrax._src.utils import conversion
from eamax.hierarchical import HierarchicalLKJMVNPrior

from .specs import crdm_spec, rdm_spec


class TruncatedNormal(distrax.Distribution):
    """Normal distribution restricted to ``[lower, upper]``.

    Used for every strictly-positive parameter whose prior is naturally described by a mean
    and a spread — drifts, pulse amplitude and time scale, non-decision time — with
    ``upper = jnp.inf``. Hand-rolled because distrax has no native truncated normal and the
    TFP substrate's does not always compose with ``distrax.Joint``.

    Sampling is by inverse-CDF on the truncated interval, which is exact and branch-free (so
    it survives ``jit`` and ``vmap``), unlike rejection sampling.

    Args:
        loc: Location of the untruncated normal, *not* the mean of the truncated one.
        scale: Scale of the untruncated normal.
        lower: Lower truncation point.
        upper: Upper truncation point; ``jnp.inf`` for one-sided truncation.
    """

    def __init__(
        self,
        loc: jnp.ndarray,
        scale: jnp.ndarray,
        lower: jnp.ndarray,
        upper: jnp.ndarray,
    ):
        super().__init__()
        self._loc = conversion.as_float_array(loc)
        self._scale = conversion.as_float_array(scale)
        self._lower = conversion.as_float_array(lower)
        self._upper = conversion.as_float_array(upper)

        self._lower_z = (self._lower - self._loc) / self._scale
        self._upper_z = (self._upper - self._loc) / self._scale

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        # jax.scipy uses z-standardized bounds
        return jax.scipy.stats.truncnorm.logpdf(
            value, self._lower_z, self._upper_z, self._loc, self._scale
        )

    def _sample_n(self, key: jnp.ndarray, n: int) -> jnp.ndarray:
        out_shape = (n,) + self.batch_shape
        dtype = jnp.result_type(self._loc, self._scale)

        q_lower = jax.scipy.stats.norm.cdf(self._lower, loc=self._loc, scale=self._scale)
        q_upper = jax.scipy.stats.norm.cdf(self._upper, loc=self._loc, scale=self._scale)

        u = jax.random.uniform(
            key=key, shape=out_shape, dtype=dtype, minval=q_lower, maxval=q_upper
        )

        return jax.scipy.stats.norm.ppf(u, self._loc, self._scale)

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def batch_shape(self) -> Tuple:
        return jax.lax.broadcast_shapes(
            self._loc.shape, self._scale.shape, self._lower.shape, self._upper.shape
        )

    def mode(self) -> jnp.ndarray:
        """Mode of the truncated density: `loc`, clipped into the interval.

        Warning:
            The comparisons are Python-level, so this only works for scalar, *concrete*
            parameters — it raises under ``jit``/``vmap`` or with array-valued `loc`.
            Nothing in the sampling path calls it.
        """
        if self._loc < self._lower:
            return self._lower
        elif self._loc > self._upper:
            return self._upper
        return self._loc


# --------------------------------------------------------------------------------------- #
# Single-subject training priors: the box each flow is fitted over.
# --------------------------------------------------------------------------------------- #


def create_wald_prior_uniform(
    v_min: float = 0.0,
    v_max: float = 8.0,
    s_min: float = 0.0,
    s_max: float = 2.0,
    b_min: float = 0.0,
    b_max: float = 2.0,
) -> distrax.Joint:
    """Wide uniform training prior over ``[v, s, b]`` — the RDM conditioner's box.

    It must cover the region the RDM recovery prior reaches. The binding constraint is the
    *target* accumulator, whose drift is ``v_intercept + v_slope``, so `v_max` has to bound
    the sum rather than either term. Live values in ``conf_jax/prior/wald_uniform.yaml``.
    """
    return distrax.Joint([
        distrax.Uniform(v_min, v_max),
        distrax.Uniform(s_min, s_max),
        distrax.Uniform(b_min, b_max),
    ])


def create_wald_prior_informed(
    v_loc: float = 4.0,
    v_scale: float = 0.5,
    s_loc: float = 1.0,
    s_scale: float = 0.5,
    b_loc: float = 1.0,
    b_scale: float = 0.5,
) -> distrax.Joint:
    """Narrow informed alternative to :func:`create_wald_prior_uniform`.

    Not referenced by any config; training on it would leave the flow unreliable wherever
    the recovery prior strays outside these bounds.
    """
    return distrax.Joint([
        TruncatedNormal(v_loc, v_scale, 0.0, jnp.inf),
        TruncatedNormal(s_loc, s_scale, 0.0, jnp.inf),
        TruncatedNormal(b_loc, b_scale, 0.0, jnp.inf),
    ])


def create_crdm_single_prior_uniform(
    v_c_min: float = 0.0,
    v_c_max: float = 8.0,
    amp_min: float = 0.0,
    amp_max: float = 0.5,
    tau_min: float = 0.0,
    tau_max: float = 0.5,
    s_min: float = 0.0,
    s_max: float = 2.0,
    b_min: float = 0.0,
    b_max: float = 2.0,
) -> distrax.Joint:
    """Wide uniform training prior over ``[v_c, amp, tau, s, b]`` — the CRDM conditioner's box.

    Deliberately much wider than the recovery prior, for the reason in
    :func:`create_wald_prior_uniform`. Live values in
    ``conf_jax/prior/crdm_single_uniform.yaml``.

    The low-drift / high-boundary corner of this box is where trials fail to cross within
    ``t_max`` — about 3% of trials overall, heavily concentrated there. Those are handled as
    right-censored observations by ``eamax.flows.loss_fn`` rather than dropped.
    """
    return distrax.Joint([
        distrax.Uniform(v_c_min, v_c_max),
        distrax.Uniform(amp_min, amp_max),
        distrax.Uniform(tau_min, tau_max),
        distrax.Uniform(s_min, s_max),
        distrax.Uniform(b_min, b_max),
    ])


def create_crdm_single_prior_informed(
    v_c_loc: float = 4.0,
    v_c_scale: float = 0.5,
    amp_loc: float = 0.3,
    amp_scale: float = 0.05,
    tau_loc: float = 0.1,
    tau_scale: float = 0.05,
    s_loc: float = 0.8,
    s_scale: float = 0.25,
    b_loc: float = 0.7,
    b_scale: float = 0.25,
) -> distrax.Joint:
    """Narrow informed alternative to :func:`create_crdm_single_prior_uniform`.

    Not referenced by any config — kept for focused experiments where training the flow only
    over the plausible region is enough. Training on it would make the flow unusable for
    MCMC under a wider recovery prior.
    """
    return distrax.Joint([
        TruncatedNormal(v_c_loc, v_c_scale, 0.0, jnp.inf),
        TruncatedNormal(amp_loc, amp_scale, 0.0, jnp.inf),
        TruncatedNormal(tau_loc, tau_scale, 0.0, jnp.inf),
        TruncatedNormal(s_loc, s_scale, 0.0, jnp.inf),
        TruncatedNormal(b_loc, b_scale, 0.0, jnp.inf),
    ])


# --------------------------------------------------------------------------------------- #
# Single-subject recovery priors: they generate the test data and, with the log-transform
# Jacobian, serve as the MCMC log-prior.
# --------------------------------------------------------------------------------------- #


def create_rdm_prior_informed(
    v_intercept_loc: float = 1.0,
    v_intercept_scale: float = 0.25,
    v_scale_loc: float = 1.5,
    v_scale_scale: float = 0.5,
    s_true_shape: float = 12.0,
    s_true_scale: float = 0.1,
    b_shape: float = 8.0,
    b_scale: float = 0.15,
    t0_loc: float = 0.3,
    t0_scale: float = 0.2,
) -> distrax.Joint:
    """Informed RDM prior over ``[v_intercept, v_slope, s_true, b, t0]``.

    Its support has to sit inside the flow's training box
    (:func:`create_wald_prior_uniform`), and the binding constraint is on the *target*
    accumulator: its drift is ``v_intercept + v_slope``, i.e. ~2.5 under these defaults
    against a training range of ``[0, 8]``.

    `s_true` and `b` are gammas parameterised as ``Gamma(shape, rate = 1 / scale)``, so
    ``s_true_shape=12``, ``s_true_scale=0.1`` means a mean of 1.2, not 12. Live values in
    ``conf_jax/prior/rdm_informed.yaml``.
    """
    return distrax.Joint([
        TruncatedNormal(v_intercept_loc, v_intercept_scale, 0.0, jnp.inf),
        TruncatedNormal(v_scale_loc, v_scale_scale, 0.0, jnp.inf),
        distrax.Gamma(s_true_shape, 1.0 / s_true_scale),
        distrax.Gamma(b_shape, 1.0 / b_scale),
        TruncatedNormal(t0_loc, t0_scale, 0.0, jnp.inf),
    ])


def create_crdm_prior_informed(
    v_c_intercept_loc: float = 1.0,
    v_c_intercept_scale: float = 0.25,
    v_c_slope_loc: float = 4.0,
    v_c_slope_scale: float = 0.5,
    amp_loc: float = 0.3,
    amp_scale: float = 0.05,
    tau_loc: float = 0.1,
    tau_scale: float = 0.05,
    s_true_shape: float = 8.0,
    s_true_scale: float = 0.1,
    b_shape: float = 7.0,
    b_scale: float = 0.1,
    t0_loc: float = 0.3,
    t0_scale: float = 0.2,
) -> distrax.Joint:
    """Informed CRDM prior over ``[v_c_intercept, v_c_slope, amp, tau, s_true, b, t0]``.

    Narrow, centred on plausible values, and used both to generate test data and — after
    adding the log-transform Jacobian — as the MCMC log-prior. It must stay inside the
    support of the wide uniform prior the flow was trained over
    (:func:`create_crdm_single_prior_uniform`).

    Drifts, pulse parameters and the non-decision time are truncated normals on the positive
    half-line; `s_true` and `b` are gammas parameterised as
    ``Gamma(shape, rate = 1 / scale)``. Live values in ``conf_jax/prior/crdm_informed.yaml``.
    """
    return distrax.Joint([
        TruncatedNormal(v_c_intercept_loc, v_c_intercept_scale, 0.0, jnp.inf),
        TruncatedNormal(v_c_slope_loc, v_c_slope_scale, 0.0, jnp.inf),
        TruncatedNormal(amp_loc, amp_scale, 0.0, jnp.inf),
        TruncatedNormal(tau_loc, tau_scale, 0.0, jnp.inf),
        distrax.Gamma(s_true_shape, 1.0 / s_true_scale),
        distrax.Gamma(b_shape, 1.0 / b_scale),
        TruncatedNormal(t0_loc, t0_scale, 0.0, jnp.inf),
    ])


def log_prior_fn(prior):
    """The MCMC log-prior for a single-subject `distrax.Joint`, on log parameters.

    Sampling happens in log space while the prior is defined on the natural scale, so the
    change of variables belongs here: ``d(exp(x))/dx = exp(x)`` contributes ``sum(x)``. The
    prior object is the same one that generated the data, which is what keeps the generative
    and inferential models identical.

    Parameters
    ----------
    prior : distrax.Joint
        Over the natural-scale parameters, in spec order.

    Returns
    -------
    callable
        ``f(log_theta: (P,)) -> scalar``.
    """

    @jax.jit
    def log_prior(x):
        params = jnp.exp(x)
        return prior.log_prob([params[i] for i in range(params.shape[0])]) + jnp.sum(x)

    return log_prior


# --------------------------------------------------------------------------------------- #
# Hierarchical priors: semi-centered LKJ-MVN over subject-level log-parameters.
# --------------------------------------------------------------------------------------- #


def _hierarchical_prior(spec, num_subjects, *, inverse_gamma_scale, mu_loc, mu_scale,
                        inverse_gamma_concentration=4.0, lkj_concentration=2.0):
    """Build :class:`eamax.hierarchical.HierarchicalLKJMVNPrior` for `spec`."""
    return HierarchicalLKJMVNPrior(
        num_subjects,
        num_params=spec.num_params,
        num_centered=spec.num_centered,
        inverse_gamma_scale=inverse_gamma_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
        inverse_gamma_concentration=inverse_gamma_concentration,
        lkj_concentration=lkj_concentration,
    )


def create_hierarchical_rdm_prior_lkj_mvn(
    num_subjects,
    *,
    inverse_gamma_scale,
    mu_loc,
    mu_scale,
    inverse_gamma_concentration=4.0,
    lkj_concentration=2.0,
):
    """The 5-parameter hierarchical RDM prior.

    The three hyperparameter arrays are required and keyword-only: there is no defensible
    default shared by the RDM (P=5) and CRDM (P=7) layouts, and the live values are declared
    once in ``conf_jax/model/rdm.yaml`` under ``hierarchical.prior``. Parameter order is
    :data:`confrdm_jax.specs.RDM_PARAM_NAMES`, with ``b`` and ``t0`` centered.

    ``inverse_gamma_concentration`` is the *tail index* of the between-subject standard
    deviations: ``P(s > x) ~ x**-concentration``. At the historical value of 4 a rare
    population draws ``s ~ 1`` in log space and so subject parameters spanning orders of
    magnitude — far outside the box the flow was trained on, where its log-density has
    gradient spikes and dead-flat plateaus that collapse NUTS step-size adaptation. Lowering
    `inverse_gamma_scale` cannot fix that: it shifts ``s`` down but leaves the tail index
    unchanged.
    """
    return _hierarchical_prior(
        rdm_spec(),
        num_subjects,
        inverse_gamma_scale=inverse_gamma_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
        inverse_gamma_concentration=inverse_gamma_concentration,
        lkj_concentration=lkj_concentration,
    )


def create_hierarchical_crdm_prior_lkj_mvn(
    num_subjects,
    *,
    inverse_gamma_scale,
    mu_loc,
    mu_scale,
    inverse_gamma_concentration=4.0,
    lkj_concentration=2.0,
):
    """The 7-parameter hierarchical CRDM prior.

    As :func:`create_hierarchical_rdm_prior_lkj_mvn`, over
    :data:`confrdm_jax.specs.CRDM_PARAM_NAMES`. Live values in ``conf_jax/model/crdm.yaml``
    under ``hierarchical.prior``.
    """
    return _hierarchical_prior(
        crdm_spec(),
        num_subjects,
        inverse_gamma_scale=inverse_gamma_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
        inverse_gamma_concentration=inverse_gamma_concentration,
        lkj_concentration=lkj_concentration,
    )
