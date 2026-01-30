
import jax
import jax.numpy as jnp
from jax.scipy import stats


def scale_logtruncnorm(x, loc, scale, a):
    return stats.truncnorm.logpdf(x, loc=loc, scale=scale, a=(a - loc) / scale, b=jnp.inf)

@jax.jit
def rdm_prior(x):
    x = jnp.exp(x)
    v_intercept = scale_logtruncnorm(x[0], 1.0, 0.5, 0.0)
    v_slope = scale_logtruncnorm(x[1], 4.0, 0.5, 0.0)
    s_true = stats.gamma.logpdf(x[2], a=12.0, scale=0.1)
    b = stats.gamma.logpdf(x[3], a=8.0, scale=0.15)
    t0 = scale_logtruncnorm(x[4], 0.3, 0.2, 0.0)

    return v_intercept + v_slope + s_true + b + t0