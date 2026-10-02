# Neural race model estimation

This is the code repository for the paper "Neural racing accumulator model estimation with
lightweight monotonic flows" ([LINK](https://doi.org/10.6084/m9.figshare.32935031)).

It trains monotonic normalizing flows as neural likelihoods for two race models: the Racing
Diffusion Model (RDM) and the Conflict Racing Diffusion Model (CRDM). It evaluates them by
comparing densities against reference solutions and by single-subject and hierarchical
parameter recovery.

## Layout

- `conf_jax/`: Hydra configs. `experiment/final.yaml` defines the run reported in the paper.
- `figures/`: Figures written by the figure scripts; `figures/final/` holds the paper versions.
- `scripts/`: Training, experiments, figure scripts and helpers.
- `slurm/`: SLURM jobs for the final run.
- `src/confrdm_jax/`: This study's models: the two parameterizations, the priors, the hybrid
  CRDM likelihood, the samplers. Also contains `runs.py`, which locates a run's results.
- `tests/`: Tests for the Python package.

Generic model code — accumulator densities, the race likelihood, the parameterization
layer, simulation, the neural flow, the hierarchical prior family and the samplers — lives in
[`eamax`](https://github.com/maltelueken/eamax).

## Installation

The experiments were run with Python 3.12.3. With [uv](https://docs.astral.sh/uv/), which
installs `eamax` from its pinned tag:

```bash
uv sync --extra gpu   # drop --extra gpu on a machine without a CUDA GPU
```

Every entry point takes `device=gpu`, which fails loudly if JAX cannot see an accelerator
rather than falling back to CPU. A fallback produces the same numbers two orders of magnitude
slower, which is easy to miss in a log. `device=auto` takes whatever is available.

## Runs

A run is selected with a Hydra experiment config. The run reported in the paper is
`conf_jax/experiment/final.yaml`. That file is the one place its tag
(`affine_log_deep_clip_100k`) and its flow recipe are written down: a per-context affine stage
on log decision time, log-scaled conditioner inputs, two hidden layers, gradient clipping and
100k training steps. The training boxes, the CRDM's simulator step (dt = 0.0005) and the
recovery priors are the model defaults in `conf_jax/model/` and `conf_jax/prior/`.

```bash
python scripts/train_conditioner.py model=rdm +experiment=final
```

Results go to `outputs/<model>/<run_tag>/`, and parameter sweeps to
`multirun/<model>/<run_tag>/`. Any further override becomes a subdirectory, for example
`outputs/crdm/<run_tag>/model.sampler.dt=0.005/` for the CRDM flow trained at a coarser step.
Jobs that load a checkpoint must select the same experiment and repeat those further
overrides.

The figure scripts, the SLURM jobs and `conf_jax/compare_densities.yaml` find the run's
results through `confrdm_jax.runs`, which reads the tag from the experiment file:

```bash
python -m confrdm_jax.runs run-tag
python -m confrdm_jax.runs run-dir crdm model.sampler.dt=0.005
```

To start a new run, copy `conf_jax/experiment/final.yaml`, give it a new `run_tag`, and select
the copy instead.

## Running the experiments

All jobs are in `slurm/` and run the final run. Train the conditioners first:

```bash
sbatch slurm/train_conditioner_rdm.sh
sbatch slurm/train_conditioner_crdm.sh          # dt = 0.0005, used everywhere below
sbatch slurm/train_conditioner_crdm.sh 0.005    # coarser flows, for the density comparison
sbatch slurm/train_conditioner_crdm.sh 0.05
```

**Experiment 1: density comparison.** Each flow is compared against the analytic inverse
Gaussian (RDM) and against a Volterra first-passage-time solver (CRDM):

```bash
sbatch slurm/compare_densities.sh
```

**Experiment 2: parameter recovery.** Single-subject recovery sweeps 50, 250, 500 and 1000
trials (set in `conf_jax/figures.yaml`); hierarchical recovery uses five simulated populations:

```bash
sbatch slurm/parameter_recovery_rdm_single.sh
sbatch slurm/parameter_recovery_crdm_single.sh
sbatch slurm/parameter_recovery_rdm_hierarchical.sh
sbatch slurm/parameter_recovery_crdm_hierarchical.sh
```

For the RDM, the recovery jobs also fit the analytic likelihood. A classifier two-sample test
(C2ST) then compares the neural and analytic posteriors:

```bash
sbatch slurm/c2st_recovery_single.sh
sbatch slurm/c2st_recovery_hierarchical.sh
```

## Figures

Each `scripts/create_figure*.py` script writes to `figures/`. Figures that depend on a run
carry its tag in the file name. Once they look right, copy them to `figures/final/` without the
tag:

```bash
python scripts/create_figures_parameter_recovery_single.py
bash scripts/finalize_figures.sh
```

Convergence thresholds and other settings shared by the figure scripts are in
`conf_jax/figures.yaml`.

## ONNX export

`scripts/export_conditioner_onnx.py` exports a trained flow to ONNX, with the log-input
scaling inside the graph, and checks the ONNX output against JAX. It needs jax2onnx 0.17 or
later.

## Tests

```bash
pytest tests/
```

## Generative AI usage

Claude Opus 4.5 - 5.5 were used for partial code generation/improvement. All code
was double-checked and verified by the authors.
