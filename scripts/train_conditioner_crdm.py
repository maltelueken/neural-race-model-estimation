
import logging
from pathlib import Path
import hydra
import jax
import optax
from flax import nnx
from tqdm import tqdm
from confrdm_jax.flows import make_mlp_conditioner
from confrdm_jax.flows import save_conditioner
from confrdm_jax.flows import train_step
from confrdm_jax.simulators import create_crdm_single_prior_uniform
from confrdm_jax.simulators import sample_conditional_crdm_single

logger = logging.getLogger(__name__)

logging.getLogger("absl").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="../conf_jax", config_name="config")
def main(cfg):
    prior = create_crdm_single_prior_uniform()

    train_steps = cfg["train_steps"]

    eval_every = int(train_steps // 10)

    batch_shape = (cfg["train_batch_size"], cfg["train_num_obs"])

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

    learning_rate = cfg["learning_rate"] # optax.schedules.cosine_decay_schedule(0.005, train_steps, 1e-6)

    optimizer = nnx.Optimizer(
        conditioner, optax.adamw(learning_rate), wrt=nnx.Param,
    )
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    for step in tqdm(range(train_steps)):
        data, context = sample_conditional_crdm_single(rngs.sampling(), batch_shape, prior, cfg["model"]["dt"], cfg["model"]["t_max"])

        conditioner.train()

        train_step(conditioner, optimizer, metrics, data, context)

        if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
            for metric, value in metrics.compute().items():
                metrics_history[f"train_{metric}"].append(value)

            logger.info("Loss: %s", value)

    if cfg["save"]:
        save_conditioner(conditioner, Path("conditioner").absolute())

    return metrics.compute()["loss"]

if __name__ == "__main__":
    main()
