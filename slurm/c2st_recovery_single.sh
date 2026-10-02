#!/bin/bash
#SBATCH --job-name=c2st_single_rdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.err

# Single-subject C2ST, neural vs. analytic posterior, on the recoveries from
# parameter_recovery_rdm_single.sh. Runs one C2ST per test_num_obs=* directory and writes
# multirun/rdm/<run_tag>/c2st.csv next to them.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

recovery_dir=$(uv run --frozen --extra gpu python -m confrdm_jax.runs run-dir rdm --multirun)

uv run --frozen --extra gpu python scripts/c2st_recovery.py \
    c2st.mode=single \
    c2st.recovery_dir="${recovery_dir}"
