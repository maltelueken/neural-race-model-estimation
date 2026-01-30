
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
        range_min=-3.,
        range_max=3.,
        boundary_slopes="identity",
        min_bin_size=1e-3,
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


def loss_fn(conditioner, data, context, min_log_prob: float = 1e-8):
  log_probs, _ = spline_flow(data.squeeze(), context, conditioner)
  return -jnp.mean(jnp.where(jnp.isfinite(log_probs), log_probs, jnp.log(min_log_prob)))


def loss_fn(conditioner, data, context, min_log_prob: float = 1e-8):
  # 1. Identify invalid data (simulated data that causes NaNs).
  #    Since your flow includes a Log (Inverse Exp), data <= 0 is invalid.
  #    We also check for Infs or existing NaNs.
  data_flat = data.squeeze()
  is_valid = jnp.isfinite(data_flat) & (data_flat > 1e-6)

  # 2. Create "Safe Data".
  #    Replace invalid data with a dummy safe value (e.g., 1.0) BEFORE the flow.
  #    This ensures the flow acts on valid numbers, preventing NaNs in the graph.
  safe_data = jnp.where(is_valid, data_flat, 1.0)

  # 3. Run the flow on safe data.
  #    Since safe_data is always valid, safe_log_probs will not be NaN.
  safe_log_probs, _ = spline_flow(safe_data, context, conditioner)

  # 4. Mask the loss.
  #    We use the mask to zero out the contribution of the invalid data.
  #    Since safe_log_probs is finite, 0 * Finite = 0, so gradients are safe.
  final_log_probs = jnp.where(is_valid, safe_log_probs, jnp.log(min_log_prob))
  
  return -jnp.mean(final_log_probs)

@nnx.jit
def train_step(conditioner, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, data, context):
  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(conditioner, data, context)
  metrics.update(loss=loss)  # In-place updates.
  optimizer.update(conditioner, grads)  # In-place updates.


def eval_step(conditioner, metrics: nnx.MultiMetric, data, context):
  loss = loss_fn(conditioner, data, context)
  metrics.update(loss=loss)  # In-place updates.


def evaluate_pdf_sf(conditioner, data, context):
    conditioner.eval()

    _, flow = spline_flow(data.squeeze(), context, conditioner)

    z = flow.bijector.inverse(data)

    return flow.log_prob(data), stats.norm.logsf(z)


def save_conditioner(conditioner, path, step=0):
    # 1. Split to get the state (weights)
    _, state = nnx.split(conditioner)

    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)

    # Remove 'erase_and_create_empty' for production use; 'create=True' handles creation.
    # passing the path directly allows appending to existing checkpoints.
    with ocp.CheckpointManager(ocp.test_utils.erase_and_create_empty(path), options=options) as mngr:
        mngr.save(step, args=ocp.args.StandardSave(state))
        mngr.wait_until_finished()


def load_conditioner(conditioner, path, step=0):
    # 1. Create an abstract version of the state to tell Orbax what to look for.
    # Using the instance `conditioner` here is okay if its structure matches the saved one.
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
        # 2. Restore the state. StandardRestore uses abstract_state as the schema.
        state_restored = mngr.restore(step, args=ocp.args.StandardRestore(change_sharding_abstract_state))
        mngr.wait_until_finished()

    # 3. CRITICAL: Update the existing instance in-place.
    # This ensures 'conditioner' now contains the loaded weights.
    # nnx.merge(conditioner, state_restored)

    # If you prefer returning a new model, you would use:
    return nnx.merge(graphdef, state_restored)
