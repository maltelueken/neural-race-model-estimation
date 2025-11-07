import os

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"


import bayesflow as bf
import keras
import numpy as np

from priors import truncated_normal_rvs
from rdmc import rdmc_experiment_simple


def prior_fun_train(
    drift_c_intercept_loc=0.05,
    drift_c_intercept_scale=0.05,
    drift_c_slope_loc=0.5,
    drift_c_slope_scale=0.1,
    amp_shape=10,
    amp_scale=2,
    tau_shape=8,
    tau_scale=10,
    sd_true_shape=80,
    sd_true_scale=0.05,
    threshold_shape=100,
    threshold_scale=0.7,
    t0_loc=300,
    t0_scale=50,
    rng=np.random.default_rng(2025),
):
    drift_c_intercept = truncated_normal_rvs(drift_c_intercept_loc, drift_c_intercept_scale, random_state=rng)
    drift_c_slope = truncated_normal_rvs(drift_c_slope_loc, drift_c_slope_scale, random_state=rng)
    amp = rng.gamma(shape=amp_shape, scale=amp_scale)
    tau = rng.gamma(shape=tau_shape, scale=tau_scale)
    s_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale)
    b = rng.gamma(shape=threshold_shape, scale=threshold_scale)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, random_state=rng)

    return {
        "v_c_intercept": drift_c_intercept,
        "v_c_slope": drift_c_slope,
        "amp": amp,
        "tau": tau,
        "s_true": s_true,
        "b": b,
        "t0": t0
    }

def simulator_fun(**kwargs):
    return rdmc_experiment_simple(**kwargs, t_max=5000, seed=2025, a_shape=2, s_false=4)


def meta(batch_size):
    num_obs = np.random.default_rng(2025).integers(250, 750)

    return dict(num_obs=num_obs)

simulator = bf.simulators.make_simulator([prior_fun_train, simulator_fun], meta_fn=meta)

param_names = ["v_c_intercept", "v_c_slope", "tau", "amp", "s_true", "b", "t0"]

adapter = (
    bf.Adapter()
    .to_array()
    .convert_dtype("float64", "float32")
    .broadcast(param_names, to="x", expand=1)
    .drop("num_obs")
    .as_set(["x"])
    .log(param_names)
    .concatenate(param_names, into="inference_conditions")
    .rename("x", "inference_variables")
)

inference_network = bf.networks.CouplingFlow(
    base_distribution=bf.distributions.DiagonalStudentT(df=50),
    subnet_kwargs=dict(
        dropout=0.0,
        depth=12,
        transform="spline",
    )
)

approximator = bf.ContinuousApproximator(
    inference_network=inference_network,
    adapter=adapter,
    standardize="inference_conditions"
)

epochs = 100
num_batches = 500
batch_size = 64
learning_rate = keras.optimizers.schedules.CosineDecay(5e-4, decay_steps=epochs*num_batches, alpha=1e-6)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
approximator.compile(optimizer=optimizer)

history = approximator.fit(
    epochs=epochs,
    num_batches=num_batches,
    batch_size=batch_size,
    simulator=simulator,
    callbacks=[keras.callbacks.ModelCheckpoint("rdmc_nle_spline_student_deep.keras", monitor="loss", mode="min", save_best_only=True)]
)
