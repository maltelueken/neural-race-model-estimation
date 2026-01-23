
import jax
import jax.numpy as jnp

from flax import nnx
from tensorflow_probability.substrates.jax import distributions
from confrdm_jax.flows import evaluate_pdf_sf


MIN_LL = 1e-12


# @nnx.jit
def crdm_log_pdf_sf(rt, v_c, amp, tau, s, b, t0, conditioner):
    rt = rt - t0
    rt = jnp.maximum(0.0, rt)

    return evaluate_pdf_sf(conditioner, rt, jnp.array([v_c, amp, tau, s, b]))

# @jax.jit
def inv_gauss_log_pdf_sf(rt, v, s, b, t0):
    rt = rt - t0
    rt = jnp.maximum(0.0, rt)
    # mu_winner = b/drift_winner
    mu = b / v
    # lam_winner = (b/s_winner)**2
    lam = (b / s)**2

    dist = distributions.InverseGaussian(mu, lam)

    return dist.log_prob(rt), dist.log_survival_function(rt)


def _create_crdm_two_accumulators_likelihood(data):
    rt = data[:, 0]
    choice = data[:, 1]

    # @nnx.jit
    def likelihood_fun(x, conditioner):

        v_c_true = x[0] + x[1]
        v_c_false = x[0]
        amp = x[2]
        tau = x[3]
        s_true = x[4]
        s_false = 1.0
        b = x[5]
        t0 = x[6]

        log_pdf_true, log_sf_true = crdm_log_pdf_sf(rt, v_c_true, amp, tau, s_true, b, t0, conditioner)
        log_pdf_false, log_sf_false = inv_gauss_log_pdf_sf(rt, v_c_false, s_false, b, t0)

        dens_true = log_pdf_true.squeeze() + log_sf_false.squeeze()
        dens_false = log_pdf_false.squeeze() + log_sf_true.squeeze()

        ll = jnp.where(choice == 1, dens_true, dens_false)

        return jnp.where(jnp.isfinite(ll), ll, jnp.log(1e-12))

    return likelihood_fun


def create_crdm_two_accumulators_likelihood(data):
    _likelihood_fun = _create_crdm_two_accumulators_likelihood(data)

    # We accept graphdef and state separately to keep it JAX-pure
    @nnx.jit
    def likelihood_fun_pure(x, conditioner_graphdef, conditioner_state):
        # Reconstruct the model locally for this call
        conditioner = nnx.merge(conditioner_graphdef, conditioner_state)
        
        # Calculate likelihood
        return _likelihood_fun(jnp.exp(x), conditioner)
    
    return likelihood_fun_pure


def create_crdm_two_accumulators_likelihood_conditions(data):
    condition = data[:, 2]

    # Create the internal likelihood helpers for each condition
    _likelihood_fun_con = _create_crdm_two_accumulators_likelihood(data[condition == 1, :2])
    _likelihood_fun_inc = _create_crdm_two_accumulators_likelihood(data[condition == 0, :2])

    @nnx.jit
    def likelihood_fun(x, conditioner_graphdef, conditioner_state):
        # 1. Reconstruct the conditioner from the pure state
        conditioner = nnx.merge(conditioner_graphdef, conditioner_state)
        
        # 2. Transform parameters (keeping x differentiable)
        x_val = jnp.exp(x)
        # Assuming index 2 is the parameter that flips for incongruent
        x_inc = x_val.at[2].multiply(-1.0)

        # 3. Compute LL for both conditions
        ll_con = _likelihood_fun_con(x_val, conditioner)
        ll_inc = _likelihood_fun_inc(x_inc, conditioner)

        # 4. Return as a stack (for vector-valued operations)
        # Note: Use jnp.concatenate if you want a flat array of all trials
        return jnp.concatenate([ll_con, ll_inc])

    return likelihood_fun
