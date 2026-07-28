
import distrax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from jax.scipy import stats


class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = nnx.gelu(self.linear1(x))
    return self.linear2(x)


def make_mlp_conditioner(rngs, num_in, num_mid=64, num_bins=4):
    return MLP(din=num_in, dmid=num_mid, dout=num_bins * 3 + 1, rngs=rngs)

# @nnx.jit
def spline_flow(data, context, conditioner):
    spline_params = conditioner(context)

    spline_layer = distrax.RationalQuadraticSpline(
        spline_params,
        range_min=-5.0,  # This needs to be low to capture low RTs - otherwise t0 is biased
        range_max=5.0,
        boundary_slopes="identity",
        min_bin_size=1e-4,
    )

    exp_layer = distrax.Lambda(
        forward=jnp.exp,
        inverse=jnp.log,
        forward_log_det_jacobian=lambda z: z,
        inverse_log_det_jacobian=lambda x: -jnp.log(x),
        event_ndims_in=0,
        event_ndims_out=0,
    )

    bijector = distrax.Chain([exp_layer, spline_layer])

    base_dist = distrax.Normal(loc=0.0, scale=1.0)

    flow = distrax.Transformed(base_dist, bijector)

    return flow.log_prob(data), flow


def loss_fn(conditioner, data, context, t_max: float | None = None):
  r"""Right-censored negative log-likelihood of the first-passage time.

  A trial that never crosses carries the sentinel ``rt = -1.0`` (see
  ``simulate_crdm_single_trial``).  It is *not* missing data: it is the
  observation :math:`T > t_{\max}`, whose likelihood is the survival function.

  Dropping those trials — or, equivalently, replacing their log-density with a
  constant, which zeroes their gradient — makes the flow fit the conditional
  density :math:`p(t \mid T < t_{\max}) = p(t)/(1-q)` instead of :math:`p(t)`,
  with :math:`q = P(T > t_{\max})`.  The fitted survival function then decays
  to 0 rather than to ``q``, and the density is inflated by ``-log(1-q)``
  uniformly in ``t``.  Both errors are :math:`\theta`-dependent, so they do not
  cancel out of the likelihood ratio and they bias the posterior.  See
  ``CONFRDM_JAX.md`` §7 for the measured magnitudes.

  Scoring the censored trials by ``log S(t_max)`` instead is the standard
  right-censored MLE and removes the bias at no extra cost — the flow's survival
  function is exact given the flow (``CONFRDM_JAX.md`` §3), so it is directly
  differentiable.

  ``t_max`` may be left ``None`` for samplers that cannot censor
  (``sample_conditional_wald`` draws exactly from the inverse Gaussian).  With
  no sentinels present the survival branch is never selected and the returned
  value is bit-identical to the uncensored loss.
  """
  data_flat = data.squeeze()

  is_valid = jnp.isfinite(data_flat) & (data_flat > 0.0)

  # Both branches are evaluated, so the substituted value must stay inside the
  # support or the unused branch contributes NaN to the gradient.
  safe_data = jnp.where(is_valid, data_flat, 1.0 if t_max is None else t_max)

  log_pdf, flow = spline_flow(safe_data, context, conditioner)
  log_sf = stats.norm.logsf(flow.bijector.inverse(safe_data))

  return -jnp.mean(jnp.where(is_valid, log_pdf, log_sf))

@nnx.jit(static_argnames="t_max")
def train_step(conditioner, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, data, context, t_max=None):
  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(conditioner, data, context, t_max)
  metrics.update(loss=loss)  # In-place updates.
  optimizer.update(conditioner, grads)  # In-place updates.


def eval_step(conditioner, metrics: nnx.MultiMetric, data, context, t_max=None):
  loss = loss_fn(conditioner, data, context, t_max)
  metrics.update(loss=loss)  # In-place updates.


def evaluate_pdf_sf(conditioner, data, context):
    conditioner.eval()

    _, flow = spline_flow(data.squeeze(), context, conditioner)

    z = flow.bijector.inverse(data)

    return flow.log_prob(data), stats.norm.logsf(z)


def save_conditioner(conditioner, path, step=0):
    _, state = nnx.split(conditioner)

    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)

    with ocp.CheckpointManager(ocp.test_utils.erase_and_create_empty(path), options=options) as mngr:
        mngr.save(step, args=ocp.args.StandardSave(state))
        mngr.wait_until_finished()


def load_conditioner(conditioner, path, step=0):
    abstract_model = nnx.eval_shape(lambda: conditioner)
    graphdef, abstract_state = nnx.split(abstract_model)

    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ('x',)),
        jax.sharding.PartitionSpec(),
    )
    def set_sharding(x: jax.ShapeDtypeStruct) -> jax.ShapeDtypeStruct:
      return x.update(sharding=sharding)

    change_sharding_abstract_state = jax.tree_util.tree_map(
        set_sharding, abstract_state
    )

    options = ocp.CheckpointManagerOptions()
    with ocp.CheckpointManager(path, options=options) as mngr:
        state_restored = mngr.restore(step, args=ocp.args.StandardRestore(change_sharding_abstract_state))
        mngr.wait_until_finished()

    return nnx.merge(graphdef, state_restored)
