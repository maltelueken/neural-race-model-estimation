import jax
import jax.numpy as jnp
from flax import nnx
from jax.scipy import stats

from confrdm_jax.flows import evaluate_pdf_sf
from .rdm import inv_gauss_log_pdf_sf


def create_rdm_hierarchical_likelihood(data, mask, num_params=5, num_pop_params=25):
    """Hierarchical analytical likelihood for multi-subject RDM.

    Args:
        data: Padded trial data, shape (S, max_trials, 2) with columns [RT, choice].
        mask: Boolean mask, shape (S, max_trials). True for real trials.
        num_params: Number of per-subject RDM parameters (default 5).
        num_pop_params: Number of population-level parameters in the flat vector (default 8).

    Returns:
        A JIT-compiled function `likelihood_fun(x)` that takes a flat parameter vector
        and returns a scalar total log-likelihood across all subjects and trials.
    """
    num_subjects = data.shape[0]

    def _single_subject_ll(subject_data, subject_mask, subject_params):
        """Compute per-trial log-likelihoods for one subject."""
        rt = subject_data[:, 0]
        choice = subject_data[:, 1]

        x = jnp.exp(subject_params)
        v_c_true = x[0] + x[1]
        v_c_false = x[0]
        s_true = x[2]
        s_false = 1.0
        b = x[3]
        t0 = x[4]

        log_pdf_true, log_sf_true = inv_gauss_log_pdf_sf(rt, v_c_true, s_true, b, t0)
        log_pdf_false, log_sf_false = inv_gauss_log_pdf_sf(rt, v_c_false, s_false, b, t0)

        dens_true = log_pdf_true.squeeze() + log_sf_false.squeeze()
        dens_false = log_pdf_false.squeeze() + log_sf_true.squeeze()

        ll = jnp.where(choice == 1, dens_true, dens_false)
        ll = jnp.where(jnp.isfinite(ll), ll, jnp.log(1e-12))

        return jnp.sum(jnp.where(subject_mask, ll, 0.0))

    _vmapped_ll = jax.vmap(_single_subject_ll, in_axes=(0, 0, 0))

    @jax.jit
    def likelihood_fun(x):
        subj_params = x # x[num_pop_params:].reshape(num_subjects, num_params)
        return jnp.sum(_vmapped_ll(data, mask, subj_params))

    return likelihood_fun


def create_rdm_hierarchical_likelihood_factory_approx(conditioner, num_params=5, num_pop_params=25):
    """Factory for hierarchical neural-approximate likelihood for multi-subject RDM.

    Args:
        conditioner: Trained neural network conditioner for density estimation.
        num_params: Number of per-subject RDM parameters (default 5).
        num_pop_params: Number of population-level parameters in the flat vector (default 8).

    Returns:
        A function `create_likelihood(data, mask)` that returns a likelihood function.
    """
    def inv_gauss_log_pdf_sf_approx(rt, v, s, b, t0):
        rt = rt - t0
        rt = jnp.maximum(1e-10, rt)
        return evaluate_pdf_sf(conditioner, rt, jnp.array([v, s, b]))

    def create_likelihood(data, mask):
        num_subjects = data.shape[0]

        def _single_subject_ll(subject_data, subject_mask, subject_params):
            """Compute per-trial log-likelihoods for one subject."""
            rt = subject_data[:, 0]
            choice = subject_data[:, 1]

            x = jnp.exp(subject_params)
            v_true = x[0] + x[1]
            v_false = x[0]
            s_true = x[2]
            s_false = 1.0
            b = x[3]
            t0 = x[4]

            log_pdf_true, log_sf_true = inv_gauss_log_pdf_sf_approx(rt, v_true, s_true, b, t0)
            log_pdf_false, log_sf_false = inv_gauss_log_pdf_sf_approx(rt, v_false, s_false, b, t0)

            dens_true = log_pdf_true.squeeze() + log_sf_false.squeeze()
            dens_false = log_pdf_false.squeeze() + log_sf_true.squeeze()

            ll = jnp.where(choice == 1, dens_true, dens_false)
            ll = jnp.where(jnp.isfinite(ll), ll, jnp.log(1e-12))

            return jnp.sum(jnp.where(subject_mask, ll, 0.0))

        _vmapped_ll = jax.vmap(_single_subject_ll, in_axes=(0, 0, 0))

        @nnx.jit
        def likelihood_fun(x):
            subj_params = x # x[num_pop_params:].reshape(num_subjects, num_params)
            return jnp.sum(_vmapped_ll(data, mask, subj_params))

        return likelihood_fun

    return create_likelihood