"""Training-loop plumbing: the clipped optimizer config and the gradient-norm metrics.

The objective itself is tested in `test_flows_affine.py`. What is tested here is that
`optimizer=adam_cosine_decay_clip` really clips at `grad_clip_norm` and is otherwise the
unclipped optimizer, and that `train_step` reports the raw gradient norm the training log
prints.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from confrdm_jax.flows_affine import Maximum, make_mlp_conditioner, train_step

CONFIG_DIR = str(Path(__file__).parents[1] / "conf_jax")


def _optimizer(name, *overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose("config", overrides=["model=crdm", f"optimizer={name}", "train_steps=1000", *overrides])
    return instantiate(cfg.optimizer), cfg


def _first_update(tx, grads):
    params = {"w": jnp.zeros_like(grads["w"])}
    updates, _ = tx.update(grads, tx.init(params), params)
    return np.asarray(updates["w"])


def test_clipped_optimizer_clips_large_gradients_to_the_threshold():
    clipped, cfg = _optimizer("adam_cosine_decay_clip")
    plain, _ = _optimizer("adam_cosine_decay")
    assert cfg.grad_clip_norm == 5.0

    big = {"w": jnp.array([30.0, -40.0, 0.0, 10.0])}  # global norm ~51
    scaled = {"w": big["w"] * (5.0 / float(optax.global_norm(big)))}

    # A clipped step on a large gradient is the unclipped step on that gradient rescaled to
    # the threshold -- so the clip acts on the gradient, before AdamW's moment estimates.
    np.testing.assert_allclose(_first_update(clipped, big), _first_update(plain, scaled), rtol=1e-12)
    # AdamW's first step is sign-like, so the rescaled and unscaled gradients would give the same
    # update; the second-step comparison below is what shows the moment estimates differ.
    params = {"w": jnp.zeros(4)}
    state_c, state_p = clipped.init(params), plain.init(params)
    small = {"w": jnp.array([0.1, 0.2, -0.1, 0.05])}
    _, state_c = clipped.update(big, state_c, params)
    _, state_p = plain.update(big, state_p, params)
    after_c, _ = clipped.update(small, state_c, params)
    after_p, _ = plain.update(small, state_p, params)
    assert not np.allclose(np.asarray(after_c["w"]), np.asarray(after_p["w"]))


def test_clipped_optimizer_leaves_normal_gradients_alone():
    clipped, _ = _optimizer("adam_cosine_decay_clip")
    plain, _ = _optimizer("adam_cosine_decay")
    normal = {"w": jnp.array([0.3, -0.4, 0.5, 0.1])}  # global norm ~0.7, the trained-flow regime
    np.testing.assert_array_equal(_first_update(clipped, normal), _first_update(plain, normal))


def test_clip_threshold_follows_its_override():
    clipped, cfg = _optimizer("adam_cosine_decay_clip", "grad_clip_norm=1.0")
    plain, _ = _optimizer("adam_cosine_decay")
    grads = {"w": jnp.array([3.0, 4.0])}  # norm 5
    assert cfg.grad_clip_norm == 1.0
    np.testing.assert_allclose(
        _first_update(clipped, grads), _first_update(plain, {"w": grads["w"] / 5.0}), rtol=1e-12,
    )


def test_maximum_metric_tracks_and_resets():
    metric = Maximum("grad_norm")
    assert float(metric.compute()) == -np.inf
    for value in (1.0, 7.5, 3.0):
        metric.update(grad_norm=jnp.array(value))
    assert float(metric.compute()) == pytest.approx(7.5)
    metric.reset()
    metric.update(grad_norm=jnp.array(2.0))
    assert float(metric.compute()) == pytest.approx(2.0)
    with pytest.raises(TypeError):
        metric.update(loss=1.0)


def test_train_step_reports_the_raw_gradient_norm():
    conditioner = make_mlp_conditioner(nnx.Rngs(0), num_in=3, num_mid=16, num_bins=4, affine=True)
    optimizer = nnx.Optimizer(conditioner, optax.adamw(1e-3), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        grad_norm=nnx.metrics.Average("grad_norm"),
        grad_norm_max=Maximum("grad_norm"),
    )
    context = jnp.tile(jnp.array([[[2.0, 1.0, 1.0]]]), (4, 1, 1))
    data = jnp.tile(jnp.linspace(0.1, 2.0, 16), (4, 1))

    from confrdm_jax.flows_affine import loss_fn

    expected = float(optax.global_norm(nnx.state(nnx.grad(loss_fn)(conditioner, data, context, 4.0), nnx.Param)))
    train_step(conditioner, optimizer, metrics, data, context, 4.0)
    computed = metrics.compute()
    assert float(computed["grad_norm"]) == pytest.approx(expected, rel=1e-5)
    assert float(computed["grad_norm_max"]) == pytest.approx(expected, rel=1e-5)
    assert np.isfinite(float(computed["loss"]))

    # A metric set that only averages the loss still works: the extra keyword is ignored.
    loss_only = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    train_step(conditioner, optimizer, loss_only, data, context, 4.0)
    assert np.isfinite(float(loss_only.compute()["loss"]))
    assert jax.tree_util.tree_leaves(nnx.state(conditioner, nnx.Param))
