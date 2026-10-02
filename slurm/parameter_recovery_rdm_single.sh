#!/bin/bash
#SBATCH --job-name=rec_single_rdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.err

# Single-subject recovery with the final RDM conditioner, one sweep job per trial count
# (conf_jax/figures.yaml, num_obs). Writes multirun/rdm/<run_tag>/test_num_obs=<N>/, with the
# analytic reference fit alongside (run_reference_recovery). A sweep runs in its own directory,
# so the conditioner is named explicitly.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

num_obs=$(uv run --frozen --extra gpu python -m confrdm_jax.runs num-obs)
conditioner_dir=$(uv run --frozen --extra gpu python -m confrdm_jax.runs run-dir rdm)

uv run --frozen --extra gpu python scripts/parameter_recovery.py --multirun \
    device=gpu \
    model=rdm \
    +experiment=final \
    test_num_obs="${num_obs}" \
    conditioner_dir="${conditioner_dir}"
