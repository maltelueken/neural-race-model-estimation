#!/bin/bash
#SBATCH --job-name=train_crdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.err

# Train a final CRDM conditioner (conf_jax/experiment/final.yaml) on the box in
# conf_jax/prior/crdm_single_uniform.yaml.
#
#   sbatch slurm/train_conditioner_crdm.sh          # dt = 0.0005, outputs/crdm/<run_tag>/
#   sbatch slurm/train_conditioner_crdm.sh 0.005    # outputs/crdm/<run_tag>/model.sampler.dt=0.005/
#   sbatch slurm/train_conditioner_crdm.sh 0.05
#
# The default step is conf_jax/model/crdm.yaml's sampler.dt; the coarser flows exist for the
# density comparison (scripts/compare_neural_densities.py).

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

dt=${1:-}
overrides=()
if [[ -n ${dt} ]]; then
    overrides+=("model.sampler.dt=${dt}")
fi

uv run --frozen --extra gpu python scripts/train_conditioner.py \
    device=gpu \
    model=crdm \
    +experiment=final \
    "${overrides[@]}"
