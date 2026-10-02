#!/bin/bash
#SBATCH --job-name=rec_hier_rdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=05:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.err

# Hierarchical recovery with the final RDM conditioner. Runs in the conditioner's own
# directory, outputs/rdm/<run_tag>/, and writes hierarchical_recovery_pop{N}_{approx,ref}.nc
# there: model=rdm also fits the analytic reference (run_reference_recovery), so the neural
# posterior can be checked against the exact one.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

uv run --frozen --extra gpu python scripts/parameter_recovery_hierarchical.py \
    device=gpu \
    model=rdm \
    +experiment=final
