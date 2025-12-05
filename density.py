from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.stats import norm
from jax.scipy.special import gammainc, gammaln

# Enable 64-bit precision (crucial for numerical stability in accumulators)
jax.config.update("jax_enable_x64", True)

@jit
def get_gamma_stats(t, alpha, tau):
    """
    Computes Gamma PDF and CDF manually for JAX stability.
    params: alpha (shape), tau (scale)
    """
    # PDF: (x^(alpha-1) * exp(-x/tau)) / (tau^alpha * Gamma(alpha))
    # Log-space calculation for stability
    log_pdf = (alpha - 1) * jnp.log(t) - (t / tau) - (alpha * jnp.log(tau)) - gammaln(alpha)
    pdf = jnp.exp(log_pdf)
    
    # CDF: Regularized lower incomplete gamma function P(a, x)
    # JAX's gammainc is P(a, x) where x is the integration limit.
    # Note: Scipy's scale notation vs mathematical notation requires checking inputs.
    # gammainc(a, x) in JAX matches the standard definition.
    cdf = gammainc(alpha, t / tau) 
    return pdf, cdf

@jit
def compute_boundary(t_grid, mu_c, A, alpha, tau, b, sigma):
    """
    Computes the effective boundary B(t) for standard Brownian motion.
    B(t) = (b - integrated_drift(t)) / sigma
    """
    # _, gamma_cdf = get_gamma_stats(t_grid, alpha, tau)
    gamma_cdf = gammainc(alpha, t_grid / tau)
    
    # Cumulative drift M(t)
    M_t = (mu_c * t_grid) + (A * gamma_cdf)
    
    # Transform boundary
    B_t = (b - M_t) / sigma
    return B_t

@jit
def volterra_step(carry, i, t_grid, boundary, dt):
    """
    One step of the Smith (2000) numerical solver.
    This function is designed to be passed to jax.lax.scan.
    """
    g_grid = carry # The history of g values
    
    t_curr = t_grid[i]
    b_curr = boundary[i]

    # --- 1. Calculate LHS Term ---
    # LHS = 1 - Phi(B(t) / sqrt(t))
    # This represents the probability the unconstrained process is above B(t)
    lhs = 1.0 - norm.cdf(b_curr / jnp.sqrt(t_curr))
    
    # --- 2. Calculate Convolution Integral (Interaction with past) ---
    # We need to compute sum(g[j] * K(t_i, t_j)) for j < i
    
    # Vectorized calculation of differences for ALL t (masked later)
    t_diff = t_curr - t_grid
    b_diff = b_curr - boundary
    
    # Handle singularity at t_diff = 0 to avoid NaNs
    # We replace 0 with 1.0 temporarily; the mask will zero it out anyway.
    safe_t_diff = jnp.where(t_diff > 0, t_diff, 1.0)
    
    # Kernel: 1 - Phi((B(t) - B(s)) / sqrt(t - s))
    kernel_vals = 1.0 - norm.cdf(b_diff / jnp.sqrt(safe_t_diff))
    
    # Create Causal Mask: only sum where index j < i
    mask = jnp.arange(len(t_grid)) < i
    
    # The integral term: sum(g * kernel * dt)
    integral_term = jnp.sum(g_grid * kernel_vals * mask) * dt
    
    # --- 3. Solve for g[i] ---
    # Smith's method weight for current step is 0.5 * dt
    g_val = (lhs - integral_term) / (0.5 * dt)
    
    # Enforce non-negativity (numerical stability)
    g_val = jnp.maximum(g_val, 0.0)
    
    # Update history grid
    g_new = g_grid.at[i].set(g_val)
    
    return g_new, g_val

@partial(jit, static_argnames=("t_max", "dt"))
def get_pdf(params, t_max=1.5, dt=0.005):
    """
    Solves the Volterra equation to return the full PDF array.
    """
    # Create grid (exclude 0 to avoid division by zero)
    t_grid = jnp.arange(dt, t_max + dt, dt)
    
    # 1. Get Boundary
    B_t = compute_boundary(t_grid, 
                           params['mu_c'], params['A'], 
                           params['alpha'], params['tau'], 
                           params['b'], params['sigma'])
    
    # 2. Run Volterra Loop via lax.scan
    # We initialize g_grid with zeros
    init_g = jnp.zeros_like(t_grid)
    indices = jnp.arange(len(t_grid))
    
    # wrapper to freeze static args for scan
    def scan_fn(carry, i):
        return volterra_step(carry, i, t_grid, B_t, dt)

    # Scan returns (final_carry, stacked_outputs)
    # The stacked_output is exactly our PDF g(t)
    _, pdf = jax.lax.scan(scan_fn, init_g, indices)
    
    return t_grid, pdf

# --- BATCHING & LIKELIHOOD ---

@partial(jit, static_argnames=("t_max", "dt"))
def race_model_likelihood(rt, choice, params_1, params_2, t0, t_max, dt):
    """
    Computes log likelihood for a single trial.
    Differentiable!
    """
    # 1. Adjust RT
    decision_time = rt - t0
    
    # 2. Get PDFs for both accumulators
    t_grid, pdf1 = get_pdf(params_1, t_max, dt)
    _, pdf2 = get_pdf(params_2, t_max, dt)
    
    # 3. Compute CDFs (Cumulative Sum * dt)
    cdf1 = jnp.cumsum(pdf1) * dt
    cdf2 = jnp.cumsum(pdf2) * dt
    
    # 4. Find index corresponding to decision time
    # We use soft interpolation or nearest index. For simplicity, nearest:
    idx = jnp.round(decision_time / dt).astype(int) - 1
    
    # Safety clipping for indices
    idx = jnp.clip(idx, 0, len(t_grid)-1)
    
    # 5. Calculate Likelihood
    # L = pdf_winner * (1 - cdf_loser)
    # We use jnp.where to select based on choice (avoiding python 'if')
    
    # Likelihood if choice == 0
    L1 = pdf1[idx] * (1.0 - cdf2[idx])
    
    # Likelihood if choice == 0
    L2 = pdf2[idx] * (1.0 - cdf1[idx])
    
    likelihood = jnp.where(choice == 0, L1, L2)
    
    # Avoid log(0)
    likelihood = jnp.maximum(likelihood, 1e-10)
    
    return jnp.log(likelihood)

# Vectorize the likelihood function to handle a full dataset at once


def rdmc_approx_likelihood(rt, choice, v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape=2.0, t_max=1500.0, dt=5.0):
    batch_likelihood = vmap(race_model_likelihood, in_axes=(0, 0, None, None, None, None, None))

    params_1 = {'mu_c': v_c_intercept, 'A': 0.0, 'alpha': a_shape, 'tau': tau, 'b': b, 'sigma': s_false}
    params_2 = {'mu_c': v_c_intercept + v_c_slope, 'A': amp, 'alpha': a_shape, 'tau': tau, 'b': b, 'sigma': s_true}

    return batch_likelihood(rt, choice, params_1, params_2, t0, t_max, dt)
