"""Helper functions for prior distributions.
"""

import distrax
import jax
import jax.numpy as jnp

from distrax._src.utils import conversion


class TruncatedNormal(distrax.Distribution):
    def __init__(self, loc, scale, lower, upper):
        super().__init__()
        self._loc = conversion.as_float_array(loc)
        self._scale = conversion.as_float_array(scale)
        self._lower = conversion.as_float_array(lower)
        self._upper = conversion.as_float_array(upper)

        self._lower_z = (self._lower - self._loc) / self._scale
        self._upper_z = (self._upper - self._loc) / self._scale 


    def log_prob(self, value):
        # jax.scipy uses z-standardized bounds
        return jax.scipy.stats.truncnorm.logpdf(value, self._lower_z, self._upper_z, self._loc, self._scale)
    

    def _sample_n(self, key, n):
        out_shape = (n,) + self.batch_shape
        dtype = jnp.result_type(self._loc, self._scale)

        q_lower = jax.scipy.stats.norm.cdf(self._lower, loc=self._loc, scale=self._scale)
        q_upper = jax.scipy.stats.norm.cdf(self._upper, loc=self._loc, scale=self._scale)

        u = jax.random.uniform(key=key, shape=out_shape, dtype=dtype, minval=q_lower, maxval=q_upper)

        return jax.scipy.stats.norm.ppf(u, self._loc, self._scale)
    
    @property
    def event_shape(self):
        return ()
    
    @property
    def batch_shape(self):
        return jax.lax.broadcast_shapes(self._loc.shape, self._scale.shape, self._lower.shape, self._upper.shape)


def create_rdmc_prior(
    drift_c_intercept_loc=0.05,
    drift_c_intercept_scale=0.05,
    drift_c_slope_loc=0.5,
    drift_c_slope_scale=0.1,
    amp_shape=10,
    amp_scale=2,
    tau_shape=8,
    tau_scale=10,
    sd_true_shape=80,
    sd_true_scale=0.05,
    threshold_shape=100,
    threshold_scale=0.7,
    t0_loc=300,
    t0_scale=50,
):
    return distrax.Joint([
        TruncatedNormal(drift_c_intercept_loc, drift_c_intercept_scale, 0.0, jnp.inf), # only lower bound
        TruncatedNormal(drift_c_slope_loc, drift_c_slope_scale, 0.0, jnp.inf),
        distrax.Gamma(amp_shape, 1 / amp_scale),
        distrax.Gamma(tau_shape, 1 / tau_scale),
        distrax.Gamma(sd_true_shape, 1 / sd_true_scale),
        distrax.Gamma(threshold_shape, 1 / threshold_scale),
        TruncatedNormal(t0_loc, t0_scale, 0.0, jnp.inf),
    ])
