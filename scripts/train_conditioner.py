
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

    eval_every = int(train_steps // 10)

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

        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)

            logger.info("Loss: %s", value)

    if cfg["save"]:
        save_conditioner(conditioner, Path("conditioner").absolute())

    return metrics.compute()["loss"]

if __name__ == "__main__":
    main()
