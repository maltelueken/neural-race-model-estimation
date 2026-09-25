#!/bin/bash
#SBATCH --job-name=c2st_hrdm_affine_log_deep_clip_box
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/c2st_recovery_hierarchical_rdm_affine_log_deep_clip_box_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/c2st_recovery_hierarchical_rdm_affine_log_deep_clip_box_%j.err

# Hierarchical C2ST for the RDM flow trained by train_conditioner_rdm_affine_log_deep_clip_box.sh,
# on the recoveries from parameter_recovery_rdm_hierarchical_affine_log_deep_clip_box.sh.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

base_dir=/projects/0/prjs1372/racing-diffusion-conflict

cd ${base_dir}

uv run --frozen --extra gpu python scripts/c2st_recovery.py c2st.mode=hierarchical \
    c2st.recovery_dir="outputs/rdm/model.flow_affine\=true/model.flow_log_inputs\=true/model.flow_num_hidden\=2/model.num_bins\=12/model.num_mid\=128/model.training_prior.b_max\=3.5/model.training_prior.b_min\=0.25/model.training_prior.s_max\=3.5/model.training_prior.s_min\=0.25/optimizer\=adam_cosine_decay_clip/train_steps\=100000"
