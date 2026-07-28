"""Conditional normalising flow used as the neural likelihood estimator.

The flow is amortised over *parameters*, not over data sets: it learns the
first-passage-time density of a **single** accumulator conditioned on that
accumulator's parameters, and the two-accumulator race is assembled
analytically afterwards (see ``confrdm_jax.likelihoods``).

The generative direction is::

    Z ~ N(0, 1)  --rational-quadratic spline-->  --exp-->  T

so the support is ``(0, inf)`` by construction and the transform is strictly
increasing.  That monotonicity is what makes the race likelihood possible: the
events ``{T > t}`` and ``{Z > z}`` coincide, so the survival function is
``S(t) = sf_{N(0,1)}(z)`` with ``z = bijector.inverse(t)`` — exact given the
flow, not a second approximation.  ``proof.md`` has the argument in full.

Conditioning sets, one column per parameter, in **natural** (non-log) space:

============  ================================  =========
Model         Context                           ``num_in``
============  ================================  =========
Wald / RDM    ``(v, s, b)``                     3
CRDM          ``(v_c, |amp|, tau, s, b)``       5
============  ================================  =========

``t0`` is deliberately *not* a conditioning variable.  Training data is
generated with ``t0 = 0``, so the flow models decision times only and the
likelihood evaluates it at ``rt - t0``; this drops a dimension from the
conditioning set and makes ``t0`` exactly a location shift.
"""

import distrax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from jax.scipy import stats


class MLP(nnx.Module):
  """Two-layer GELU perceptron mapping a context vector to spline knots."""

  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = nnx.gelu(self.linear1(x))
    return self.linear2(x)


def make_mlp_conditioner(rngs, num_in, num_mid=64, num_bins=4):
    """Build the conditioner MLP for a ``num_bins``-knot spline.

    The output width is ``3 * num_bins + 1``: one width, one height and one
    derivative per bin, plus the final knot derivative — the parameterisation
    ``distrax.RationalQuadraticSpline`` expects.

    Args:
        rngs: ``nnx.Rngs`` supplying the ``default`` stream for initialisation.
        num_in: Number of conditioning variables (3 for Wald/RDM, 5 for CRDM).
        num_mid: Hidden width.
        num_bins: Number of spline bins.

    Returns:
        An :class:`MLP`.
    """
    return MLP(din=num_in, dmid=num_mid, dout=num_bins * 3 + 1, rngs=rngs)

# @nnx.jit
def spline_flow(data, context, conditioner):
    """Build the conditional flow for `context` and score `data` under it.

    Args:
        data: Decision times at which to evaluate the log-density. Must
            broadcast against the batch shape implied by `context`.
        context: Conditioning parameters, either ``(num_in,)`` for a single
            parameter vector shared by every element of `data`, or
            ``(N, num_in)`` for one parameter vector per observation.
        conditioner: The MLP from :func:`make_mlp_conditioner`.

    Returns:
        ``(log_prob, flow)``. The `flow` is returned so callers can reach the
        bijector for the survival function; `log_prob` is often discarded and
        eliminated by JIT.

    Note:
        The spline's active range is ``[-5, 5]`` in the *latent* Gaussian
        coordinate, outside which the bijector is the identity. Five standard
        deviations covers the base distribution to ~6e-7 in each tail, so the
        identity region is only ever reached by extreme RTs; narrowing it
        truncates the left tail and biases ``t0`` upward.
    """
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
  cancel out of the likelihood ratio and they bias the posterior.

  How much censoring there is, measured over the training prior
  (``crdm_single_uniform``, ``dt = 0.004``, ``t_max = 4.0``, 2000 draws x 500
  trials): 3.16% of trials overall, 25.5% of draws losing at least one trial,
  1.5% losing more than half.  It is concentrated rather than diffuse — draws
  with ``q > 0.5`` have mean ``v_c = 0.22`` against 4.95 for clean draws, i.e.
  the slow-drift, high-boundary corner where the flow has the least data and
  the most distorted target.

  Scoring the censored trials by ``log S(t_max)`` instead is the standard
  right-censored MLE and removes the bias at no extra cost — the flow's survival
  function is exact given the flow (see the module docstring), so it is
  directly differentiable.

  ``t_max`` may be left ``None`` for samplers that cannot censor
  (``sample_conditional_wald`` draws exactly from the inverse Gaussian).  With
  no sentinels present the survival branch is never selected and the returned
  value is bit-identical to the uncensored loss.

  Args:
      conditioner: The MLP from :func:`make_mlp_conditioner`.  Differentiated
          against, so it has to stay the first positional argument.
      data: Decision times; any shape that squeezes to the batch shape implied
          by `context`.  Non-crossing trials carry the sentinel ``-1.0``.
      context: Conditioning parameters, natural scale.
      t_max: Integration horizon of the simulator that produced `data`, or
          ``None`` if that simulator cannot censor.

  Returns:
      Scalar mean negative log-likelihood.
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
  """Take one optimiser step on `loss_fn`, updating `conditioner` in place.

  `metrics` accumulates the loss but is never reset here — the caller owns the
  averaging window and must call ``metrics.reset()`` after each
  ``metrics.compute()`` if it wants a windowed mean rather than a running one.
  """
  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(conditioner, data, context, t_max)
  metrics.update(loss=loss)  # In-place updates.
  optimizer.update(conditioner, grads)  # In-place updates.


def eval_step(conditioner, metrics: nnx.MultiMetric, data, context, t_max=None):
  """Score a batch without updating the conditioner.

  Provided for held-out evaluation; the online training loop in
  ``scripts/train_conditioner.py`` simulates fresh data every step and so has
  no separate validation set to call this on.
  """
  loss = loss_fn(conditioner, data, context, t_max)
  metrics.update(loss=loss)  # In-place updates.


def evaluate_pdf_sf(conditioner, data, context):
    """Evaluate the trained flow's log-density and log survival function.

    This is the inference-time entry point: every likelihood in
    ``confrdm_jax.likelihoods`` reaches the flow through it.

    Because both the spline and ``exp`` are strictly increasing, ``{T > t}``
    and ``{Z > z}`` are the same event, so the survival function is the
    standard-normal one evaluated at the flow inverse.  It is therefore exact
    given the flow rather than a second approximation on top of it — which is
    what lets the racing likelihood multiply a flow density by a flow survival
    function without compounding error.

    Args:
        conditioner: A trained MLP conditioner.
        data: Decision times ``rt - t0``, shape ``(N,)``. Callers are expected
            to have floored this away from zero already; values at or below 0
            leave the flow's support.
        context: ``(num_in,)`` for one parameter vector shared across all of
            `data`, or ``(N, num_in)`` for one per observation.

    Returns:
        ``(log_pdf, log_sf)``, both broadcast to the shape of `data`.
    """
    conditioner.eval()

    _, flow = spline_flow(data.squeeze(), context, conditioner)

    z = flow.bijector.inverse(data)

    return flow.log_prob(data), stats.norm.logsf(z)


def save_conditioner(conditioner, path, step=0):
    """Write the conditioner's NNX state to an Orbax checkpoint at `path`.

    The directory is erased and recreated first, and only one step is kept:
    checkpoints are keyed by Hydra output directory, so a rerun of the same
    configuration is meant to replace its predecessor rather than accumulate.
    Only the state is stored — restoring needs a conditioner built with the
    same ``num_in`` / ``num_mid`` / ``num_bins``, which is why every consumer
    reconstructs it from the same config before calling
    :func:`load_conditioner`.
    """
    _, state = nnx.split(conditioner)

    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)

    with ocp.CheckpointManager(ocp.test_utils.erase_and_create_empty(path), options=options) as mngr:
        mngr.save(step, args=ocp.args.StandardSave(state))
        mngr.wait_until_finished()


def load_conditioner(conditioner, path, step=0):
    """Restore checkpointed weights into a freshly built `conditioner`.

    `conditioner` is used only for its structure — it supplies the abstract
    state Orbax restores into — so it must have been built with the same
    ``num_in`` / ``num_mid`` / ``num_bins`` as the checkpoint.  Nothing on disk
    records those, so a mismatch surfaces as an Orbax shape error rather than
    as silently wrong densities.

    Returns:
        A new merged module; `conditioner` itself is not modified.
    """
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
