#!/bin/bash
#SBATCH --job-name=comp_dens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.err

# Neural vs. reference densities for the final run's RDM flow and its three CRDM flows
# (conf_jax/compare_densities.yaml). Writes outputs/compare_densities/density_comparison.nc.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

uv run --frozen --extra gpu python scripts/compare_neural_densities.py device=gpu
