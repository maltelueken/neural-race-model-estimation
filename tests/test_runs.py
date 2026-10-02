"""The final run's definition: the experiment config and confrdm_jax.runs, which reads it.

The experiment is the one place the run tag and flow recipe are written, and every path to a
result is rebuilt from it by `runs`. These tests pin the two together -- in particular that
`runs.run_dir` reproduces the directory Hydra actually writes to -- and check that the final
checkpoints, when present, still match what the configs say they were trained on.
"""

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from confrdm_jax import runs
from confrdm_jax.flows_affine import flow_options
from eamax.flows.checkpoint import read_metadata


def compose_final(model, *overrides, hydra_config=False):
    with initialize_config_dir(config_dir=str(runs.CONF_DIR), version_base=None):
        return compose(
            "config",
            overrides=[f"model={model}", "+experiment=final", *overrides],
            return_hydra_config=hydra_config,
        )


def test_run_tag_is_the_experiments():
    assert runs.run_tag() == OmegaConf.load(runs.FINAL_EXPERIMENT)["run_tag"]
    assert compose_final("rdm")["run_tag"] == runs.run_tag()


@pytest.mark.parametrize("model", ["rdm", "crdm"])
def test_final_experiment_sets_the_recipe(model):
    cfg = compose_final(model)
    m = cfg["model"]
    assert (m["flow_affine"], m["flow_log_inputs"], m["flow_num_hidden"]) == (True, True, 2)
    assert (m["num_bins"], m["num_mid"]) == (12, 128)
    assert cfg["train_steps"] == 100000
    assert cfg["optimizer"]["_args_"][0]["_target_"] == "optax.clip_by_global_norm"


def test_final_training_boxes_and_step():
    # Moved from command-line overrides into the model defaults; the checkpoints were
    # trained on these, so a change here must come with retraining.
    rdm = compose_final("rdm")["model"]["training_prior"]
    assert (rdm["s_min"], rdm["s_max"], rdm["b_min"], rdm["b_max"]) == (0.25, 3.5, 0.25, 3.5)
    crdm = compose_final("crdm")["model"]
    assert crdm["sampler"]["dt"] == 0.0005
    assert crdm["recovery_prior"]["v_c_slope_loc"] == 2.5


@pytest.mark.parametrize(
    "model, overrides",
    [
        ("rdm", []),
        ("crdm", ["model.sampler.dt=0.005"]),
        ("rdm", ["test_num_obs=50"]),
        ("crdm", ["test_num_obs=1000", "model.sampler.dt=0.05"]),
    ],
)
def test_run_dir_matches_hydra(model, overrides):
    hydra = compose_final(model, *overrides, "device=cpu", hydra_config=True)["hydra"]
    dirname = hydra["job"]["override_dirname"]
    expected = runs.ROOT / "outputs" / model / runs.run_tag()
    if dirname:
        expected = expected / dirname
    assert runs.run_dir(model, *overrides) == expected


def test_conditioner_dir_maps_dt_to_subdirectory():
    base = runs.run_dir("crdm") / "conditioner"
    assert runs.conditioner_dir("crdm") == base
    assert runs.conditioner_dir("crdm", dt=0.0005) == base
    assert runs.conditioner_dir("crdm", dt=0.005) == runs.run_dir("crdm", "model.sampler.dt=0.005") / "conditioner"
    with pytest.raises(ValueError):
        runs.conditioner_dir("rdm", dt=0.005)


def test_run_dir_resolver():
    cfg = OmegaConf.create({"path": "${run_dir:crdm,'model.sampler.dt=0.05'}/conditioner"})
    assert cfg["path"] == str(runs.conditioner_dir("crdm", dt=0.05))


def test_figure_settings():
    settings = runs.figure_settings()
    assert settings["num_obs"] == [50, 250, 500, 1000]
    assert set(settings["convergence"]) == {"single", "hierarchical", "posterior_correlation"}


@pytest.mark.parametrize("model, dt", [("rdm", None), ("crdm", None), ("crdm", 0.005), ("crdm", 0.05)])
def test_final_checkpoint_matches_config(model, dt):
    """The checkpoint's recorded log-input constants are the ones the config derives.

    They follow from the training box, so this is what catches a box that drifted from the
    one the checkpoint was trained on. Skipped where the results are not downloaded.
    """
    path = runs.conditioner_dir(model, dt=dt)
    metadata = read_metadata(path) if path.exists() else None
    if metadata is None:
        pytest.skip(f"no checkpoint at {path}")
    overrides = [] if dt is None else [f"model.sampler.dt={dt}"]
    options = flow_options(compose_final(model, *overrides)["model"])
    recorded = metadata["input_scaling"]
    assert recorded is not None
    for key in ("eps", "loc", "scale"):
        assert recorded[key] == pytest.approx(list(options["input_scaling"][key]), rel=1e-12)
    assert metadata["num_hidden"] == options["num_hidden"]
    assert metadata["affine"] is options["affine"]
