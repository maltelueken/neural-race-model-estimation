#!/bin/bash
#SBATCH --job-name=rec_single_crdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.err

# Single-subject recovery with the final CRDM conditioner, one sweep job per trial count
# (conf_jax/figures.yaml, num_obs). Writes multirun/crdm/<run_tag>/test_num_obs=<N>/. A sweep
# runs in its own directory, so the conditioner is named explicitly. About 3% of the recovery
# prior's data sets fall below the training box; see conf_jax/prior/crdm_informed.yaml.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

num_obs=$(uv run --frozen --extra gpu python -m confrdm_jax.runs num-obs)
conditioner_dir=$(uv run --frozen --extra gpu python -m confrdm_jax.runs run-dir crdm)

uv run --frozen --extra gpu python scripts/parameter_recovery.py --multirun \
    device=gpu \
    model=crdm \
    +experiment=final \
    test_num_obs="${num_obs}" \
    conditioner_dir="${conditioner_dir}"
