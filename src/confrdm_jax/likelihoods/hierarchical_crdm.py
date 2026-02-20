import jax
import jax.numpy as jnp
from flax import nnx

from confrdm_jax.flows import evaluate_pdf_sf
from .rdm import _clamp_log
from .rdm import inv_gauss_log_pdf_sf


_FLOOR = 1e-10


def create_crdm_hierarchical_likelihood_factory_approx(conditioner, num_params=7, num_pop_params=35):
    """Factory for hierarchical neural-approximate likelihood for multi-subject CRDM.

    Args:
        conditioner: Trained neural network conditioner for density estimation.
        num_params: Number of per-subject CRDM parameters (default 7).
        num_pop_params: Number of population-level parameters in the flat vector.

    Returns:
        A function `create_likelihood(data, mask)` that returns a likelihood function.
    """
    def crdm_log_pdf_sf(rt, v_c, amp, tau, s, b, t0):
        rt = rt - t0
        rt = jnp.maximum(rt, _FLOOR)

        v_c = jnp.maximum(v_c, _FLOOR)
        amp = jnp.maximum(amp, _FLOOR)
        tau = jnp.maximum(tau, _FLOOR)
        s = jnp.maximum(s, _FLOOR)
        b = jnp.maximum(b, _FLOOR)

        context = jnp.transpose(jnp.array([v_c, jnp.abs(amp), tau, s, b]))
        return evaluate_pdf_sf(conditioner, rt, context)

    def create_likelihood(data, mask):
        def _single_subject_ll(subject_data, subject_mask, subject_params):
            """Compute per-trial log-likelihoods for one subject."""
            rt = subject_data[:, 0]
            choice = subject_data[:, 1]
            condition = subject_data[:, 2]  # 1 = Congruent, 0 = Incongruent

            x = jnp.exp(subject_params)
            v_c_true = x[0] + x[1]
            v_c_false = x[0]
            amp = x[2]
            tau = x[3]
            s_true = x[4]
            s_false = 1.0
            b = x[5]
            t0 = x[6]

            # Input routing based on condition
            nn_v_c = jnp.where(condition == 1, v_c_true, v_c_false)
            nn_s = jnp.where(condition == 1, s_true, s_false)
            ig_v_c = jnp.where(condition == 1, v_c_false, v_c_true)
            ig_s = jnp.where(condition == 1, s_false, s_true)

            amp_arr = jnp.tile(amp, nn_v_c.shape)
            tau_arr = jnp.tile(tau, nn_v_c.shape)
            b_arr = jnp.tile(b, nn_v_c.shape)

            # Neural network for the accumulator with conflict modulation
            nn_log_pdf, nn_log_sf = crdm_log_pdf_sf(rt, nn_v_c, amp_arr, tau_arr, nn_s, b_arr, t0)

            # Analytic inverse Gaussian for the other accumulator
            ig_log_pdf, ig_log_sf = inv_gauss_log_pdf_sf(rt, ig_v_c, ig_s, b, t0)

            # Clamp all log-density outputs upstream to avoid NaN gradient poisoning
            nn_log_pdf_c = _clamp_log(nn_log_pdf.squeeze())
            nn_log_sf_c = _clamp_log(nn_log_sf.squeeze())
            ig_log_pdf_c = _clamp_log(ig_log_pdf.squeeze())
            ig_log_sf_c = _clamp_log(ig_log_sf.squeeze())

            # Result routing
            log_pdf_true = jnp.where(condition == 1, nn_log_pdf_c, ig_log_pdf_c)
            log_sf_false = jnp.where(condition == 1, ig_log_sf_c, nn_log_sf_c)
            log_pdf_false = jnp.where(condition == 1, ig_log_pdf_c, nn_log_pdf_c)
            log_sf_true = jnp.where(condition == 1, nn_log_sf_c, ig_log_sf_c)

            # Final choice logic
            dens_choice_1 = log_pdf_true + log_sf_false
            dens_choice_0 = log_pdf_false + log_sf_true

            ll = jnp.where(choice == 1, dens_choice_1, dens_choice_0)

            return jnp.sum(jnp.where(subject_mask, ll, 0.0))

        _vmapped_ll = jax.vmap(_single_subject_ll, in_axes=(0, 0, 0))

        @nnx.jit
        def likelihood_fun(x):
            subj_params = x
            return jnp.sum(_vmapped_ll(data, mask, subj_params))

        return likelihood_fun

    return create_likelihood
