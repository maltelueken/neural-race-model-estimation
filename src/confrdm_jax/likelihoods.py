
import jax
import jax.numpy as jnp
from jax.scipy import stats

from tensorflow_probability.substrates.jax import distributions


@jax.jit
def inv_gauss_logpdf(t, mu, lam):

    e = -(lam / (2 * t)) * (t**2 / mu**2 - 2 * t / mu  + 1)

    x = e + 0.5 * jnp.log(lam) - 0.5 * jnp.log(2 * t**3 * jnp.pi)

    return x

@jax.jit
def inv_gauss_logsf(t, mu, lam):
    """https://journal.r-project.org/archive/2016-1/giner-smyth.pdf"""
    mu = mu / lam
    t = t / lam
    r = 1.0 / jnp.sqrt(t)
    a = stats.norm.logcdf(-r * ((t / mu) - 1.0))
    b = 2.0 / mu + stats.norm.logcdf(-r * (t + mu) / mu)
    return jnp.where(jnp.isposinf(t), -jnp.inf, jnp.where(t > 0.0, a + jnp.log1p(-jnp.exp(b - a)), 0.0))

@jax.jit
def inv_gauss_log_pdf_sf(rt, v, s, b, t0):
    rt = rt - t0
    rt = jnp.maximum(0.0, rt)

    # mu_winner = b/drift_winner
    mu = b/v
    # lam_winner = (b/s_winner)**2
    lam = (b/s)**2

    return inv_gauss_logpdf(rt, mu, lam), inv_gauss_logsf(rt, mu, lam)


def create_rdm_two_accumulators_likelihood(data):
    rt = data[:, 0]
    choice = data[:, 1]

    @jax.jit
    def likelihood_fun(x, min_ll=1e-12):
        x = jnp.exp(x)
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

        return jnp.where(jnp.isfinite(ll), ll, jnp.log(min_ll))

    return likelihood_fun
