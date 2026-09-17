"""Train the conditional spline flow that serves as the neural likelihood.

Online training loop: a fresh batch is simulated from the training prior at every step and
discarded afterwards, so there is no stored dataset and no epochs — `train_steps` is the
whole budget. Data are therefore never reused, which is why there is no held-out validation
split either.

The objective, the optimiser step and the checkpoint format come from `eamax.flows`; what
stays here is the schedule, the logging window and the checkpoint policy.

Which model is trained comes from the Hydra config:

    python scripts/train_conditioner.py model=rdm
    python scripts/train_conditioner.py model=crdm num_bins=12 num_mid=128

`model=rdm` trains on ``sample_conditional_wald`` (one accumulator, constant drift, exact
inverse Gaussian draws); `model=crdm` on ``sample_conditional_crdm_single`` (one accumulator
plus conflict pulse, simulated on a grid). In both cases the flow learns a *single*
accumulator's first-passage density; the race is reassembled at inference time.

The checkpoint is written to ``<hydra output dir>/conditioner``, together with a JSON
sidecar recording what the flow was conditioned on. The same architecture overrides must be
repeated when loading it — the sidecar records the context, not the weights' shapes.
"""

import logging
from pathlib import Path

import hydra
import jax
from flax import nnx
from hydra.utils import instantiate
from tqdm import tqdm

from confrdm_jax import configure_jax
# Local patch over eamax.flows: identical for a plain conditioner, plus the optional
# per-context affine stage selected by `model.flow_affine`.
from confrdm_jax.flows_affine import (
    Maximum,
    flow_options,
    make_mlp_conditioner,
    save_conditioner,
    train_step,
)

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):
    configure_jax(cfg["device"], require_device=cfg["require_device"])

    prior = instantiate(cfg["model"]["training_prior"])
    sampler = instantiate(cfg["model"]["sampler"], _partial_=True)

    train_steps = cfg["train_steps"]

    # Ten logged points over the run. Clamped to >= 1 so that short configurations (smoke
    # tests, sweep trials) do not divide by zero below.
    eval_every = max(1, int(train_steps // 10))

    batch_shape = (cfg["train_batch_size"], cfg["train_num_obs"])

    # Simulators that integrate over a finite window censor the trials that never cross;
    # those must be scored by the survival function, not dropped. Samplers with no `t_max`
    # (the exact-inverse-Gaussian Wald sampler) cannot censor and pass None.
    t_max = cfg["model"]["sampler"].get("t_max", None)

    # The order the flow's conditioning columns are trained in. Recorded in the checkpoint
    # sidecar, because nothing in an Orbax checkpoint says what its context columns mean and
    # a *reordered* context of the same width would load without complaint and produce
    # silently wrong densities.
    context_names = list(cfg["model"]["context_names"])

    metrics_history = {"train_loss": [], "train_grad_norm": [], "train_grad_norm_max": []}

    train_key = jax.random.key(cfg["train_seed"])
    conditioner_key, sampling_key = jax.random.split(train_key, 2)

    rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

    conditioner = make_mlp_conditioner(
        num_in=len(context_names),
        num_bins=cfg["model"]["num_bins"],
        num_mid=cfg["model"]["num_mid"],
        **flow_options(cfg["model"]),
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(conditioner, instantiate(cfg["optimizer"]), wrt=nnx.Param)
    # The gradient norm is the raw one, before any clipping in the optimizer: its window mean
    # says where training normally sits, its window max whether anything spiked -- which an
    # average over 100k steps would otherwise hide.
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        grad_norm=nnx.metrics.Average("grad_norm"),
        grad_norm_max=Maximum("grad_norm"),
    )

    for step in tqdm(range(train_steps)):
        data, context = sampler(rngs.sampling(), batch_shape, prior)

        conditioner.train()

        train_step(conditioner, optimizer, metrics, data, context, t_max)

        # Always record the final step, so `metrics_history` is non-empty whatever
        # `train_steps` is; skip step 0, whose average is one batch.
        if step == train_steps - 1 or (step > 0 and step % eval_every == 0):
            computed = metrics.compute()
            for metric, value in computed.items():
                metrics_history[f"train_{metric}"].append(value)

            # `nnx.metrics.Average` accumulates from wherever it was last reset, so without
            # this every logged value would be the running mean since step 0 rather than the
            # mean over this window. That matters beyond the log line: the return value below
            # is the Optuna objective, and a whole-run mean scores a configuration on how
            # fast it started as much as on where it converged — which, under a cosine-decay
            # schedule over ~1e6 steps, is dominated by the early high-loss phase.
            metrics.reset()

            # "Loss: <value>" stays first on the line, as downstream log parsing expects it.
            logger.info(
                "Loss: %s | grad norm mean %.4g max %.4g",
                computed["loss"], computed["grad_norm"], computed["grad_norm_max"],
            )

    if cfg["save"]:
        save_conditioner(
            conditioner,
            Path("conditioner").absolute(),
            context_names=context_names,
            num_mid=cfg["model"]["num_mid"],
            num_bins=cfg["model"]["num_bins"],
        )

    # Mean loss over the final window, i.e. the last ~10% of training.
    return metrics_history["train_loss"][-1]


if __name__ == "__main__":
    main()
