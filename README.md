# Neural race model estimation

This is the code repository for the paper "Neural racing accumulator model estimation with
lightweight monotonic flows" (LINK).

The structure is as follows:

- `conf_jax`: Config files for running scripts.
- `notebooks`: Jupyter otebooks for visualizing results.
- `scripts`: Scripts to run experiments.
- `slurm`: Slurm scripts.
- `src/confrdm_jax`: This study's models — the two parameterizations, the priors, the hybrid CRDM likelihood and the samplers.
- `tests`: Tests for Python package.

## Requirements

The experiments were run with Python 3.11.15.

The shared model code — accumulator densities, the race likelihood, the parameterization
layer, simulation, the neural flow, the hierarchical prior family and the samplers — lives in
[`eamax`](https://github.com/maltelueken/eamax), which is also used by `eam-abi-robustness`
and `cognitive-control-comparison`. What stays here is what those three disagree about: the
parameterizations, the priors, and the conflict model's hybrid race.

## Installation

`eamax` is not on PyPI yet, so install it from a sibling checkout first:

```bash
pip install -e ../eamax
pip install -e .
```

The experiments run on GPU. Install the matching JAX plugin with:

```bash
pip install -U "jax[cuda12]"
```

Every entry point takes `device=gpu`, which fails loudly if JAX cannot see an accelerator
rather than falling back to CPU — a fallback produces the same numbers two orders of
magnitude slower, which is easy to miss in a log. `device=auto` takes whatever is available.

## Running experiments

Before running the experiments, the conditioner networks for the monotonic flows must be trained for the Racing Diffusion Model (RDM) and Conflict Racing Diffusion Model (CRDM).

The RDM conditioner network was trained using the specs in `slurm/train_conditioner_rdm.sh`. The three CRDM conditioner networks were trained using the specs in `slurm/train_conditioner_crdm_{dt}.sh` with $dt = 0.05, 0.005, 0.0005$.

After training the conditioner networks, the first experiment can be run with:

```bash
python scripts/compare_neural_densities.py device=gpu
```

The second experiment can be run with the specs in:

- Single-subject RDM `slurm/parameter_recovery_rdm_multirun.sh` (runs script across different trial numbers)
- Hierarchical RDM `slurm/parameter_recovery_rdm_hierarchical.sh`
- Single-subject CRDM `slurm/parameter_recovery_crdm_multirun.sh` (runs script across different trial numbers)
- Hierarchical CRDM `slurm/parameter_recovery_crdm_hierarchical.sh`

## Generative AI usage

Claude Opus 4.6, 4.7, and 4.8 were used for partial code generation/improvement. All code was double-checked and verified by the authors.
