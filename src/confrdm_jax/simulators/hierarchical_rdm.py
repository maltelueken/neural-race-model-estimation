import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb

from .rdm import simulate_rdm


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
        lkj_concentration=2.0,
        halfnormal_scale=None,
        mu_loc=None,
        mu_scale=None,
    ):
        self.P = 5
        P = self.P
        if halfnormal_scale is None:
            halfnormal_scale = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])
        else:
            halfnormal_scale = jnp.asarray(halfnormal_scale)
        if mu_loc is None:
            mu_loc = jnp.array([-0.2, 0.6, 0.3, 0.5, -1.8])
        else:
            mu_loc = jnp.asarray(mu_loc)
        if mu_scale is None:
            mu_scale = jnp.array([0.5, 0.5, 0.5, 0.5, 2.0])
        else:
            mu_scale = jnp.asarray(mu_scale)

        self._halfnormal_scale = halfnormal_scale
        self._lkj_concentration = lkj_concentration
        self._mu_loc = mu_loc
        self._mu_scale = mu_scale

        self._joint = tfd.JointDistributionSequential([
            tfd.Independent(tfd.HalfNormal(halfnormal_scale), 1),
            lambda s: tfd.TransformedDistribution(tfd.CholeskyLKJ(P, jnp.asarray(lkj_concentration, dtype=s.dtype)), tfb.ScaleMatvecDiag(s)),
            tfd.Independent(tfd.Normal(mu_loc, mu_scale), 1),
            lambda mu, psi: tfd.Sample(
                tfd.MultivariateNormalTriL(
                    mu,
                    scale_tril=psi,
                ),
                num_subjects,
            )
        ])

    def sample(self, seed):
        """Sample from the prior, returning (s, L, mu, log_theta)."""
        return self._joint.sample(seed=seed)

    def log_prob(self, L, mu, log_theta):
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
        s = jnp.sqrt(jnp.diag(L @ L.T))
        return self._joint.log_prob((s, L, mu, log_theta))


def create_hierarchical_rdm_prior_lkj_mvn(
    num_subjects,
    lkj_concentration=2.0,
    halfnormal_scale=None,
    mu_loc=None,
    mu_scale=None,
):
    return HierarchicalRDMPriorLKJMVN(
        num_subjects,
        lkj_concentration=lkj_concentration,
        halfnormal_scale=halfnormal_scale,
        mu_loc=mu_loc,
        mu_scale=mu_scale,
    )


def sample_conditional_rdm_hierarchical_lkj_mvn(
    key, num_trials, num_subjects,
    lkj_concentration=2.0, halfnormal_scale=None, mu_loc=None, mu_scale=None,
):
    """Sample data from LKJ-MVN hierarchical RDM prior.

    Args:
        key: JAX random key.
        num_trials: Number of trials per subject.
        num_subjects: Number of subjects.
        lkj_concentration: LKJ concentration parameter.
        halfnormal_scale: Scale for HalfNormal prior on std devs (P,).
        mu_loc: Mean of Normal prior on population mean (P,).
        mu_scale: Std dev of Normal prior on population mean (P,).

    Returns:
        Tuple of (data, context):
            data: Simulated data array (S, num_trials, 2) with columns [RT, choice].
            context: Dictionary containing hierarchical prior sample.
    """
    key_context, key_data = jax.random.split(key)

    prior = create_hierarchical_rdm_prior_lkj_mvn(
        num_subjects, lkj_concentration=lkj_concentration,
        halfnormal_scale=halfnormal_scale, mu_loc=mu_loc, mu_scale=mu_scale,
    )

    # Joint returns (s, L, mu, log_theta) where:
    #   s: HalfNormal output (P,) - standard deviations
    #   L: TransformedDist output (P,P) - Cholesky factor L = diag(s) @ rho_chol
    #   mu: Normal output (P,) - population mean
    #   log_theta: MVN output (S,P) - subject params in log space
    s, L, mu, log_theta = prior.sample(seed=key_context)

    Sigma = L @ L.T
    rho = jnp.diag(1.0 / s) @ Sigma @ jnp.diag(1.0 / s)
    theta = jnp.exp(log_theta)

    context = {
        'rho': rho,
        's': s,
        'mu': mu,
        'L': L,
        'Sigma': Sigma,
        'log_theta': log_theta,
        'theta': theta,
    }

    v_intercept = theta[:, 0]
    v_slope = theta[:, 1]
    s_true = theta[:, 2]
    b = theta[:, 3]
    t0 = theta[:, 4]

    keys = jax.random.split(key_data, num_subjects)

    def _simulate_subject(v_int, v_sl, s, b_, t0_, k):
        return simulate_rdm(v_int, v_sl, s, b_, t0_, num_trials, k)

    data = jax.vmap(_simulate_subject)(v_intercept, v_slope, s_true, b, t0, keys)

    return data, context
