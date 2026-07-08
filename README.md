# Neural race model estimation

This is the code repository for the paper "Neural racing accumulator model estimation with
lightweight monotonic flows" (LINK).

The structure is as follows:

- `conf_jax`: Config files for running scripts.
- `notebooks`: Jupyter otebooks for visualizing results.
- `scripts`: Scripts to run experiments.
- `slurm`: Slurm scripts.
- `src/confrdm_jax`: Experimental Python package. Will be refactored into a standalone repository soon.
- `tests`: Tests for Python package.

## Requirements

The experiments were run with Python 3.11.15.

## Installation

The confrdm_jax Python package can be installed with:

```bash
pip install -e .
```

## Running experiments

Before running the experiments, the conditioner networks for the monotonic flows must be trained for the Racing Diffusion Model (RDM) and Conflict Racing Diffusion Model (CRDM).

The RDM conditioner network was trained using the specs in `slurm/train_conditioner_rdm.sh`. The three CRDM conditioner networks were trained using the specs in `slurm/train_conditioner_crdm_{dt}.sh` with $dt = 0.05, 0.005, 0.0005$.

After trainin the conditioner networks, the first experiment can be run with:

```bash
python scripts/compare_neural_densities.py
```

The second experiment can be run with the specs in:

- Single-subject RDM `slurm/parameter_recovery_rdm_multirun.sh` (runs script across different trial numbers)
- Hierarchical RDM `slurm/parameter_recovery_rdm_hierarchical.sh`
- Single-subject CRDM `slurm/parameter_recovery_crdm_multirun.sh` (runs script across different trial numbers)
- Hierarchical CRDM `slurm/parameter_recovery_crdm_hierarchical.sh`

## Generative AI usage

Claude Opus 4.6, 4.7, and 4.8 were used for partial code generation/improvement. All code was double-checked and verified by the authors.
