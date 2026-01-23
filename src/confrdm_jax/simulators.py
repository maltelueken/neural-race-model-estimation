from functools import partial
from typing import Tuple
import distrax
import jax
import jax.numpy as jnp
from distrax._src.utils import conversion
from tensorflow_probability.substrates.jax import distributions


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


def create_wald_prior_informed(
    v_loc: float = 4.0,
    v_scale: float = 0.5,
    s_loc: float = 1.0,
    s_scale: float = 0.5,
    b_loc: float = 1.0,
    b_scale: float = 0.5,
) -> distrax.Joint:
    return distrax.Joint([
        TruncatedNormal(v_loc, v_scale, 0.0, jnp.inf),
        TruncatedNormal(s_loc, s_scale, 0.0, jnp.inf),
        TruncatedNormal(b_loc, b_scale, 0.0, jnp.inf),
    ])


@partial(jax.jit, static_argnames=("batch_shape", "prior"))
def sample_conditional_wald(
    key: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    prior: distrax.Joint,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key_context, key_data = jax.random.split(key, 2)

    prior_shape = batch_shape[:-1]

    context = prior.sample(seed=key_context, sample_shape=prior_shape)

    # mu = b/drift
    mu = context[2] / context[0]
    # lam = (b/s)**2
    lam = (context[2] / context[1]) ** 2

    data = distributions.InverseGaussian(mu, lam).sample(batch_shape[-1], key_data)

    context = jnp.expand_dims(jnp.array(context), axis=-1)
    context = jnp.moveaxis(jnp.array(context), 0, -1)

    return jnp.expand_dims(jnp.moveaxis(data, 0, -1), -1), context


def simulate_rdm(
    v_intercept: jnp.ndarray,
    v_slope: jnp.ndarray,
    s_true: jnp.ndarray,
    b: jnp.ndarray,
    t0: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    key: jnp.ndarray,
) -> jnp.ndarray:
    v = jnp.hstack([v_intercept, v_intercept + v_slope])
    s = jnp.hstack([1.0, s_true])

    mu = b / v
    lam = (b / s) ** 2

    fpt = distributions.InverseGaussian(mu, lam).sample(batch_shape, key)

    print(fpt.shape)

    resp = jnp.argmin(fpt, axis=-1)
    rt = jnp.min(fpt, axis=-1) + t0

    return jnp.c_[rt, resp]


def create_crdm_single_prior_informed(
    v_c_loc: float = 4.0,
    v_c_scale: float = 0.5,
    amp_loc: float = 0.3,
    amp_scale: float = 0.1,
    tau_loc: float = 0.1,
    tau_scale: float = 0.1,
    s_loc: float = 0.9,
    s_scale: float = 0.25,
    b_loc: float = 0.5,
    b_scale: float = 0.25,
) -> distrax.Joint:
    return distrax.Joint([
        TruncatedNormal(v_c_loc, v_c_scale, 0.0, jnp.inf),
        TruncatedNormal(amp_loc, amp_scale, 0.0, jnp.inf),
        TruncatedNormal(tau_loc, tau_scale, 0.0, jnp.inf),
        TruncatedNormal(s_loc, s_scale, 0.0, jnp.inf),
        TruncatedNormal(b_loc, b_scale, 0.0, jnp.inf),
    ])


@jax.jit
def simulate_crdm_single_trial(
    mu: jnp.ndarray,
    b: jnp.ndarray,
    s: jnp.ndarray,
    t0: float,
    dt: float,
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Generate noise for all accumulators and time steps at once
    noise = jax.random.normal(key, shape=mu.shape)

    # 1. FIX: Scale drift by dt (Euler-Maruyama integration)
    # dx = v * dt + s * dW
    increments = (mu * dt) + (s * noise * jnp.sqrt(dt))

    # Calculate trajectories
    x = jnp.cumsum(increments, axis=-1)

    # Determine crossings
    crossing = x >= b

    # Find the index of the first crossing
    # argmax returns the first index where True occurs; if all False, returns 0
    first_idx = jnp.argmax(crossing, axis=-1)

    # Calculate the time of crossing
    # We use (idx + 1) because cumsum at index 0 is actually step 1
    first_time = (first_idx + 1) * dt

    # Check which accumulators actually crossed the boundary
    did_cross = jnp.any(crossing, axis=-1)

    # 2. FIX: Handle non-crossers
    # If an accumulator didn't cross, set its FPT to infinity.
    # Your original code set this to 'dt', which made non-crossers win the race immediately.
    fpt = jnp.where(did_cross, first_time, jnp.inf)

    # 3. Race Logic
    # Find the fastest FPT among all accumulators
    min_fpt = jnp.min(fpt)     # The winning time
    winner_idx = jnp.argmin(fpt) # The winning accumulator

    # If NO accumulator crossed, return -1.0/-1
    any_one_crossed = jnp.any(did_cross)

    rt = jnp.where(any_one_crossed, min_fpt + t0, -1.0)
    resp = jnp.where(any_one_crossed, winner_idx, -1)

    return rt, resp


def gamma_pulse(x, amp, tau, a_shape):
    return (amp
        * jnp.exp(-x / tau)
        * (jnp.exp(1) * x / (a_shape - 1) / tau) ** (a_shape - 1)
    ) * ((a_shape - 1) / x - 1 / tau)


# @partial(jax.jit, static_argnames=["dt", "t_max"])
def simulate_crdm_dataset(
    keys: jnp.ndarray,
    v_c_intercept: jnp.ndarray,
    v_c_slope: jnp.ndarray,
    amp: jnp.ndarray,
    tau: jnp.ndarray,
    s_true: jnp.ndarray,
    b: jnp.ndarray,
    t0: jnp.ndarray,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    mu_c = jnp.hstack([v_c_intercept, v_c_intercept + v_c_slope])
    s = jnp.expand_dims(jnp.hstack([1.0, s_true]), -1)
    a_shape = 2.0

    t = jnp.arange(dt, t_max + dt, dt)

    v_a = gamma_pulse(t, jnp.abs(amp), tau, a_shape)

    mu = jnp.tile(mu_c, (t.shape[0], 1)).T

    dim_con = jnp.where(amp > 0.0, 1, 0)
    # dim_inc = jnp.where(amp > 0.0, 0, 1)

    mu = mu.at[dim_con, :].add(v_a)
    # mu = mu.at[dim_inc, :].subtract(eq4)

    rt, resp = jax.vmap(simulate_crdm_single_trial, in_axes=(None, None, None, None, None, 0))(
        mu, b, s, t0, dt, keys
    )

    return jnp.c_[rt, resp]


@partial(jax.jit, static_argnames=["dt", "t_max"])
def simulate_crdm_batch(
    keys: jnp.ndarray,
    v_c_intercept: jnp.ndarray,
    v_c_slope: jnp.ndarray,
    amp: jnp.ndarray,
    tau: jnp.ndarray,
    s_true: jnp.ndarray,
    b: jnp.ndarray,
    t0: jnp.ndarray,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    in_axes = [0] * 8 + [None] * 2
    x = jax.vmap(simulate_crdm_dataset, in_axes=in_axes)(
        keys, v_c_intercept, v_c_slope, amp, tau, s_true, b, t0, dt, t_max,
    )

    return x


@partial(jax.jit, static_argnames=["dt", "t_max"])
def simulate_crdm_single_dataset(
    keys: jnp.ndarray,
    v_c: jnp.ndarray,
    amp: jnp.ndarray,
    tau: jnp.ndarray,
    s: jnp.ndarray,
    b: jnp.ndarray,
    t0: jnp.ndarray,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    a_shape = 2.0

    t = jnp.arange(dt, t_max + dt, dt)

    v_a = gamma_pulse(t, amp, tau, a_shape)

    mu = jnp.expand_dims(v_a + v_c, axis=0)
    s = jnp.expand_dims(s, axis=(0, 1))

    rt, _ = jax.vmap(simulate_crdm_single_trial, in_axes=(None, None, None, None, None, 0))(
        mu, b, s, t0, dt, keys
    )

    return rt


@partial(jax.jit, static_argnames=["dt", "t_max"])
def simulate_crdm_single_batch(
    keys: jnp.ndarray,
    v_c: jnp.ndarray,
    amp: jnp.ndarray,
    tau: jnp.ndarray,
    s: jnp.ndarray,
    b: jnp.ndarray,
    t0: jnp.ndarray,
    dt: float,
    t_max: float,
) -> jnp.ndarray:
    in_axes = [0] * 7 + [None] * 2
    rt = jax.vmap(simulate_crdm_single_dataset, in_axes=in_axes)(
        keys, v_c, amp, tau, s, b, t0, dt, t_max,
    )

    return rt


def sample_conditional_crdm_single(
    key: jnp.ndarray,
    batch_shape: Tuple[int, ...],
    prior: distrax.Joint,
    dt: float,
    t_max: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key_context, key_data = jax.random.split(key, 2)

    prior_shape = batch_shape[:-1]

    context = prior.sample(seed=key_context, sample_shape=prior_shape)

    keys = jax.random.split(key_data, batch_shape)

    x = simulate_crdm_single_batch(
        keys,
        context[0],
        context[1],
        context[3],
        context[3],
        context[4],
        jnp.zeros_like(context[0]),
        dt,
        t_max,
    )

    context = jnp.expand_dims(jnp.array(context), axis=-1)
    context = jnp.moveaxis(jnp.array(context), 0, -1)

    return jnp.expand_dims(x, -1), context
