import jax
import jax.numpy as jnp
from flax import nnx

from confrdm_jax.flows import evaluate_pdf_sf
from .rdm import inv_gauss_log_pdf_sf


def create_crdm_likelihood_factory_approx(conditioner):
    """Factory that returns a likelihood creator for CRDM using neural approximation.

    Args:
        conditioner: Trained neural network conditioner for density estimation.

    Returns:
        A function that takes data and returns a likelihood function.
    """
    def crdm_log_pdf_sf(rt, v_c, amp, tau, s, b, t0):
        rt = rt - t0
        rt = jnp.maximum(0.0, rt)
        context = jnp.transpose(jnp.array([v_c, jnp.abs(amp), tau, s, b]))
        return evaluate_pdf_sf(conditioner, rt, context)

    def create_likelihood(data):
        rt = data[:, 0]
        choice = data[:, 1]
        condition = data[:, 2]  # 1 = Congruent, 0 = Incongruent

        def likelihood_fun(x, min_ll=1e-12):
            x = jnp.exp(x)

            # Define parameters
            v_c_true = x[0] + x[1]
            v_c_false = x[0]
            amp = x[2]
            tau = x[3]
            s_true = x[4]
            s_false = 1.0
            b = x[5]
            t0 = x[6]

            # Input routing based on condition
            # If Congruent: CRDM gets (v_c_true, s_true)
            # If Incongruent: CRDM gets (v_c_false, s_false)
            nn_v_c = jnp.where(condition == 1, v_c_true, v_c_false)
            nn_s = jnp.where(condition == 1, s_true, s_false)

            # Conversely for InvGauss:
            # If Congruent: InvGauss gets (v_c_false, s_false)
            # If Incongruent: InvGauss gets (v_c_true, s_true)
            ig_v_c = jnp.where(condition == 1, v_c_false, v_c_true)
            ig_s = jnp.where(condition == 1, s_false, s_true)

            amp = jnp.tile(amp, nn_v_c.shape)
            tau = jnp.tile(tau, nn_v_c.shape)
            b = jnp.tile(b, nn_v_c.shape)

            # Run Neural Network once per trial with correctly routed params
            nn_log_pdf, nn_log_sf = crdm_log_pdf_sf(rt, nn_v_c, amp, tau, nn_s, b, t0)

            # Run Analytic Function once per trial
            ig_log_pdf, ig_log_sf = inv_gauss_log_pdf_sf(rt, ig_v_c, ig_s, b, t0)

            # Result routing
            # Target Accumulator: If Congruent -> CRDM, If Incongruent -> InvGauss
            log_pdf_true = jnp.where(condition == 1, nn_log_pdf.squeeze(), ig_log_pdf.squeeze())

            # Non-Target Accumulator: If Congruent -> InvGauss, If Incongruent -> CRDM
            log_sf_false = jnp.where(condition == 1, ig_log_sf.squeeze(), nn_log_sf.squeeze())

            log_pdf_false = jnp.where(condition == 1, ig_log_pdf.squeeze(), nn_log_pdf.squeeze())
            log_sf_true = jnp.where(condition == 1, nn_log_sf.squeeze(), ig_log_sf.squeeze())

            # Final choice logic
            dens_choice_1 = log_pdf_true + log_sf_false
            dens_choice_0 = log_pdf_false + log_sf_true

            ll = jnp.where(choice == 1, dens_choice_1, dens_choice_0)

            return jnp.where(jnp.isfinite(ll), ll, jnp.log(min_ll))

        return likelihood_fun

    return create_likelihood
