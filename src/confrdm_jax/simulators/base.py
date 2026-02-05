from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
from distrax._src.utils import conversion


class TruncatedNormal(distrax.Distribution):
    def __init__(
        self,
        loc: jnp.ndarray,
        scale: jnp.ndarray,
        lower: jnp.ndarray,
        upper: jnp.ndarray,
    ):
        super().__init__()
        self._loc = conversion.as_float_array(loc)
        self._scale = conversion.as_float_array(scale)
        self._lower = conversion.as_float_array(lower)
        self._upper = conversion.as_float_array(upper)

        self._lower_z = (self._lower - self._loc) / self._scale
        self._upper_z = (self._upper - self._loc) / self._scale

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        # jax.scipy uses z-standardized bounds
        return jax.scipy.stats.truncnorm.logpdf(
            value, self._lower_z, self._upper_z, self._loc, self._scale
        )

    def _sample_n(self, key: jnp.ndarray, n: int) -> jnp.ndarray:
        out_shape = (n,) + self.batch_shape
        dtype = jnp.result_type(self._loc, self._scale)

        q_lower = jax.scipy.stats.norm.cdf(self._lower, loc=self._loc, scale=self._scale)
        q_upper = jax.scipy.stats.norm.cdf(self._upper, loc=self._loc, scale=self._scale)

        u = jax.random.uniform(key=key, shape=out_shape, dtype=dtype, minval=q_lower, maxval=q_upper)

        return jax.scipy.stats.norm.ppf(u, self._loc, self._scale)

    @property
    def event_shape(self) -> Tuple:
        return ()

    @property
    def batch_shape(self) -> Tuple:
        return jax.lax.broadcast_shapes(
            self._loc.shape, self._scale.shape, self._lower.shape, self._upper.shape
        )

    def mode(self) -> jnp.ndarray:
        if self._loc < self._lower:
            return self._lower
        elif self._loc > self._upper:
            return self._upper
        return self._loc
