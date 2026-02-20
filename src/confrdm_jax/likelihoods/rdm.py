import jax
import jax.numpy as jnp
from flax import nnx
from jax.scipy import stats

from confrdm_jax.flows import evaluate_pdf_sf


_LOG_FLOOR = jnp.log(1e-12)
_FLOOR = 1e-10


def _clamp_log(x):
    """Clamp log-densities to a finite floor, avoiding NaN in the computation graph.

    Uses nan_to_num first because jnp.clip propagates NaN (IEEE 754).
    The gradient of nan_to_num is 0 for NaN inputs, so this is safe for
    reverse-mode differentiation (HMC / NUTS).
    """
    return jnp.maximum(jnp.nan_to_num(x, nan=_LOG_FLOOR), _LOG_FLOOR)


def _penalize_invalid_rt(rt_shifted, log_pdf, log_sf):
    """Replace log-densities with a steep penalty when rt <= t0.

    For trials where rt - t0 <= 0, the observation is impossible under
    the model.  Instead of returning a flat floor (zero gradient), we
    return a linear penalty whose gradient pushes t0 below the minimum
    observed RT.

    Args:
        rt_shifted: rt - t0 (before clamping).
        log_pdf: Log-PDF computed on clamped rt.
        log_sf: Log-SF computed on clamped rt.

    Returns:
        Corrected (log_pdf, log_sf) with steep penalty for invalid trials.
    """
    valid = rt_shifted > _FLOOR
    penalty = _LOG_FLOOR + 1e3 * jnp.minimum(rt_shifted - _FLOOR, 0.0)
    log_pdf = jnp.where(valid, log_pdf, penalty)
    log_sf = jnp.where(valid, log_sf, 0.0)
    return log_pdf, log_sf

@jax.jit
def inv_gauss_logpdf(t, mu, lam):

    e = -(lam / (2 * t)) * (t**2 / mu**2 - 2 * t / mu  + 1)

    x = e + 0.5 * jnp.log(lam) - 0.5 * jnp.log(2 * t**3 * jnp.pi)

    return x

@jax.jit
def inv_gauss_logsf(t, mu, lam):
    """https://journal.r-project.org/archive/2016-1/giner-smyth.pdf"""
    # Clamp inputs to avoid NaN from sqrt/division on non-positive values.
    mu = mu / lam
    t = t / lam
    r = 1.0 / jnp.sqrt(t)
    a = stats.norm.logcdf(-r * ((t / mu) - 1.0))
    b = 2.0 / mu + stats.norm.logcdf(-r * (t + mu) / mu)
    # Clamp b - a <= 0 to prevent log1p argument < -1, which produces NaN.
    # Mathematically b <= a always, but floating-point arithmetic can violate this.
    result = a + jnp.log1p(-jnp.exp(jnp.minimum(b - a, 0.0)))
    return jnp.where(t > 0.0, result, 0.0)

@jax.jit
def inv_gauss_log_pdf_sf(rt, v, s, b, t0):
    rt_shifted = rt - t0
    rt_safe = jnp.maximum(rt_shifted, _FLOOR)

    v = jnp.maximum(v, _FLOOR)
    s = jnp.maximum(s, _FLOOR)
    b = jnp.maximum(b, _FLOOR)

    # mu_winner = b/drift_winner
    mu = b/v
    # lam_winner = (b/s_winner)**2
    lam = (b/s)**2

    log_pdf = inv_gauss_logpdf(rt_safe, mu, lam)
    log_sf = inv_gauss_logsf(rt_safe, mu, lam)
    return _penalize_invalid_rt(rt_shifted, log_pdf, log_sf)


def create_rdm_two_accumulators_likelihood(data):
    """Reference likelihood using analytical inverse Gaussian."""
    rt = data[:, 0]
    choice = data[:, 1]

    @jax.jit
    def likelihood_fun(x):
        x = jnp.exp(x)
        v_c_true = x[0] + x[1]
        v_c_false = x[0]
        s_true = x[2]
        s_false = 1.0
        b = x[3]
        t0 = x[4]

        log_pdf_true, log_sf_true = inv_gauss_log_pdf_sf(rt, v_c_true, s_true, b, t0)
        log_pdf_false, log_sf_false = inv_gauss_log_pdf_sf(rt, v_c_false, s_false, b, t0)

        dens_true = _clamp_log(log_pdf_true.squeeze()) + _clamp_log(log_sf_false.squeeze())
        dens_false = _clamp_log(log_pdf_false.squeeze()) + _clamp_log(log_sf_true.squeeze())

        return jnp.where(choice == 1, dens_true, dens_false)

    return likelihood_fun


def create_rdm_likelihood_factory_approx(conditioner):
    """Factory that returns a likelihood creator using neural approximation.

    Args:
        conditioner: Trained neural network conditioner for density estimation.

    Returns:
        A function that takes data and returns a likelihood function.
    """
    def inv_gauss_log_pdf_sf_approx(rt, v, s, b, t0):
        rt_shifted = rt - t0
        rt_safe = jnp.maximum(rt_shifted, _FLOOR)

        v = jnp.maximum(v, _FLOOR)
        s = jnp.maximum(s, _FLOOR)
        b = jnp.maximum(b, _FLOOR)

        log_pdf, log_sf = evaluate_pdf_sf(conditioner, rt_safe, jnp.array([v, s, b]))
        return _penalize_invalid_rt(rt_shifted, log_pdf, log_sf)

    def create_likelihood(data):
        rt = data[:, 0]
        choice = data[:, 1]

        @nnx.jit
        def likelihood_fun(x):
            x = jnp.exp(x)
            v_true = x[0] + x[1]
            v_false = x[0]
            s_true = x[2]
            s_false = 1.0
            b = x[3]
            t0 = x[4]

            log_pdf_true, log_sf_true = inv_gauss_log_pdf_sf_approx(rt, v_true, s_true, b, t0)
            log_pdf_false, log_sf_false = inv_gauss_log_pdf_sf_approx(rt, v_false, s_false, b, t0)

            dens_true = _clamp_log(log_pdf_true.squeeze()) + _clamp_log(log_sf_false.squeeze())
            dens_false = _clamp_log(log_pdf_false.squeeze()) + _clamp_log(log_sf_true.squeeze())

            return jnp.where(choice == 1, dens_true, dens_false)

        return likelihood_fun

    return create_likelihood
