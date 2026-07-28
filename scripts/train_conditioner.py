"""Train the conditional spline flow that serves as the neural likelihood.

Online training loop: a fresh batch is simulated from the training prior at
every step and discarded afterwards, so there is no stored dataset and no
epochs — `train_steps` is the whole budget.  Data are therefore never reused,
which is why there is no held-out validation split either.

Which model is trained comes from the Hydra config:

    python scripts/train_conditioner.py model=rdm
    python scripts/train_conditioner.py model=crdm num_bins=12 num_mid=128

`model=rdm` trains on ``sample_conditional_wald`` (one accumulator, constant
drift, exact inverse Gaussian draws); `model=crdm` on
``sample_conditional_crdm_single`` (one accumulator plus conflict pulse,
Euler-Maruyama).  In both cases the flow learns a *single* accumulator's
first-passage density; the race is reassembled analytically at inference time.

The checkpoint is written to ``<hydra output dir>/conditioner``, and the same
overrides must be repeated when loading it — nothing on disk records the
architecture, so a mismatch surfaces as an Orbax shape error.
"""

import logging
from pathlib import Path
import hydra
import jax
from flax import nnx
from hydra.utils import instantiate
from tqdm import tqdm
from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.flows import save_conditioner
from confrdm_jax.flows import train_step

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):
    prior = instantiate(cfg["model"]["training_prior"])
    sampler = instantiate(cfg["model"]["sampler"], _partial_=True)

    train_steps = cfg["train_steps"]

    # Ten logged points over the run. Clamped to >= 1 so that short
    # configurations (smoke tests, sweep trials) do not divide by zero below.
    eval_every = max(1, int(train_steps // 10))

    batch_shape = (cfg["train_batch_size"], cfg["train_num_obs"])

    # Simulators that integrate over a finite window censor the trials that
    # never cross; those must be scored by the survival function, not dropped.
    # Samplers with no `t_max` (e.g. the exact-inverse-Gaussian Wald sampler)
    # cannot censor and pass None.
    t_max = cfg["model"]["sampler"].get("t_max", None)

    metrics_history = {
        "train_loss": [],
    }

    train_key = jax.random.key(cfg["train_seed"])
    conditioner_key, sampling_key = jax.random.split(train_key, 2)

    rngs = nnx.Rngs(default=conditioner_key, sampling=sampling_key)

    conditioner = make_mlp_conditioner(
        num_in=cfg["model"]["num_params"],
        num_bins=cfg["model"]["num_bins"],
        num_mid=cfg["model"]["num_mid"],
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(
        conditioner, instantiate(cfg["optimizer"]), wrt=nnx.Param,
    )
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    for step in tqdm(range(train_steps)):
        data, context = sampler(rngs.sampling(), batch_shape, prior)

        conditioner.train()

        train_step(conditioner, optimizer, metrics, data, context, t_max)

        # Always record the final step, so `metrics_history` is non-empty
        # whatever `train_steps` is; skip step 0, whose average is one batch.
        if step == train_steps - 1 or (step > 0 and step % eval_every == 0):
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)

            # `nnx.metrics.Average` accumulates from wherever it was last
            # reset, so without this every logged value would be the running
            # mean since step 0 rather than the mean over this window. That
            # matters beyond the log line: the return value below is the
            # Optuna objective, and a whole-run mean scores a configuration on
            # how fast it started as much as on where it converged — which,
            # under a cosine-decay schedule over ~1e6 steps, is dominated by
            # the early high-loss phase.
            metrics.reset()

            logger.info("Loss: %s", value)

    if cfg["save"]:
        save_conditioner(conditioner, Path("conditioner").absolute())

    # Mean loss over the final window, i.e. the last ~10% of training.
    return metrics_history["train_loss"][-1]

if __name__ == "__main__":
    main()
