from typing import Tuple

import numpy as np
import distrax
import jax
import jax.numpy as jnp
from distrax._src.utils import conversion
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb

_Z_95 = 1.96  # z-score for 97.5th percentile (95% CI spans P2.5 to P97.5)


def interval_to_mu_loc_scale(percentile_interval, halfnormal_scale):
    """Convert 95% CI intervals on the original scale to Normal hyperparameters in log-space.

    For a LogNormal(mu, sigma_total) marginal prior, the 95% CI is:
        [exp(mu - 1.96 * sigma_total),  exp(mu + 1.96 * sigma_total)]
    Solving for mu and sigma_total given [lo, hi]:
        mu_loc      = (ln(lo) + ln(hi)) / 2
        sigma_total = ln(hi / lo) / (2 * 1.96)
    The total variance splits across two independent sources:
        sigma_total^2 = mu_scale^2 + halfnormal_scale^2
    so:
        mu_scale = sqrt(sigma_total^2 - halfnormal_scale^2)

    Args:
        percentile_interval: array-like, shape (P, 2). Each row is [lo, hi] —
            the 2.5th and 97.5th percentile of the marginal prior on that
            parameter (on the original, non-log scale).
        halfnormal_scale: array-like, shape (P,). Scale of the HalfNormal prior
            on between-subject std devs. Must be smaller than sigma_total for
            every parameter.

    Returns:
        mu_loc: ndarray, shape (P,).
        mu_scale: ndarray, shape (P,).
    """
    interval = np.asarray(percentile_interval, dtype=float)
    hn_scale = np.asarray(halfnormal_scale, dtype=float)
    lo, hi = interval[:, 0], interval[:, 1]

    mu_loc = (np.log(lo) + np.log(hi)) / 2.0
    sigma_total = np.log(hi / lo) / (2.0 * _Z_95)
    sigma_sq_remaining = sigma_total**2 - hn_scale**2
    if np.any(sigma_sq_remaining <= 0):
        bad = np.where(sigma_sq_remaining <= 0)[0]
        raise ValueError(
            f"halfnormal_scale is too large relative to the requested interval "
            f"for parameter indices {bad.tolist()}. Reduce halfnormal_scale or widen the interval."
        )
    mu_scale = np.sqrt(sigma_sq_remaining)
    return mu_loc, mu_scale


class TruncatedNormal(distrax.Distribution):
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

        u = jax.random.uniform(key=key, shape=out_shape, dtype=dtype, minval=q_lower, maxval=q_upper)

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
        if self._loc < self._lower:
            return self._lower
        elif self._loc > self._upper:
            return self._upper
        return self._loc
    

class HierarchicalRDMPriorLKJMVN:
    """Hierarchical LKJ-MVN prior parameterized by (L, mu, log_theta).

    Internally uses CholeskyLKJ, HalfNormal, Normal, and MVN components.
    The ``log_prob`` method accepts the Cholesky factor of the covariance
    matrix L (not the separate rho_chol and s), and includes the Jacobian
    for the L -> (rho_chol, s) decomposition where L = diag(s) @ rho_chol.

    Parameters
    ----------
    num_subjects : int
        Number of subjects.
    lkj_concentration : float
        Concentration parameter for CholeskyLKJ.
    halfnormal_scale : array-like, shape (P,)
        Scale for HalfNormal prior on standard deviations.
    mu_loc : array-like, shape (P,)
        Mean of Normal prior on population mean.
    mu_scale : array-like, shape (P,)
        Std dev of Normal prior on population mean.
    """

    def __init__(
        self,
        num_subjects,
        num_params,
        lkj_concentration=2.0,
        halfnormal_scale=None,
        mu_loc=None,
        mu_scale=None,
    ):
        self.num_params = num_params
        halfnormal_scale = jnp.asarray(halfnormal_scale)
        mu_loc = jnp.asarray(mu_loc)
        mu_scale = jnp.asarray(mu_scale)

        if mu_loc.shape[0] != num_params or mu_scale.shape[0] != num_params or halfnormal_scale.shape[0] != num_params:
            raise ValueError("Length of location and scale parameters must be equal to 'num_params'")

        self._halfnormal_scale = halfnormal_scale
        self._lkj_concentration = lkj_concentration
        self._mu_loc = mu_loc
        self._mu_scale = mu_scale

        self._joint = tfd.JointDistributionNamed({
            # Each parameter in P gets its own HalfNormal scale
            "s": tfd.Independent(tfd.HalfNormal(scale=halfnormal_scale), reinterpreted_batch_ndims=1),
            
            # Each parameter in P gets its own Normal mean
            "mu": tfd.Independent(tfd.Normal(loc=mu_loc, scale=mu_scale), reinterpreted_batch_ndims=1),
            
            "psi_raw": tfd.CholeskyLKJ(num_params, lkj_concentration),
            
            # CENTERED PARAMETERIZATION: 
            # We replace `z` with `theta`, directly sampling the subject parameters.
            # The lambda arguments must strictly match the dictionary keys defined above.
            "theta": lambda psi_raw, s, mu: tfd.Sample(
                tfd.MultivariateNormalTriL(
                    loc=mu,
                    # Broadcasting: s[..., jnp.newaxis] * psi_raw efficiently 
                    # computes the matrix multiplication diag(s) @ psi_raw
                    scale_tril=s[..., jnp.newaxis] * psi_raw
                ),
                sample_shape=[num_subjects]
            )
        })

    def sample(self, seed):
        """Sample from the prior, returning (s, L, mu, log_theta)."""
        return self._joint.sample(seed=seed)

    def log_prob(self, params):
        """Evaluate log-prior density in the (L, mu, log_theta) parameterization.

        Derives s = sqrt(diag(L @ L.T)) from L and evaluates the joint
        distribution at (s, L, mu, log_theta).  The TransformedDistribution
        component (ScaleMatvecDiag bijector) handles the L <-> rho_chol
        Jacobian automatically.

        Parameters
        ----------
        L : array, shape (P, P)
            Lower-triangular Cholesky factor of the covariance matrix Sigma.
        mu : array, shape (P,)
            Population-level mean.
        log_theta : array, shape (S, P)
            Subject-level parameters in log space.

        Returns
        -------
        Scalar log-density.
        """
        return self._joint.log_prob(params)
