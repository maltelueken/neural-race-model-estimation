#!/bin/bash
#SBATCH --job-name=c2st_rdm_affine_log_deep_clip_box
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/c2st_recovery_rdm_affine_log_deep_clip_box_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/c2st_recovery_rdm_affine_log_deep_clip_box_%j.err

# Single-subject C2ST for the RDM flow trained by train_conditioner_rdm_affine_log_deep_clip_box.sh,
# on the recoveries from parameter_recovery_rdm_multirun_affine_log_deep_clip_box.sh. Those
# recoveries were launched with model.flow_affine=True / model.flow_log_inputs=True, which is what
# Hydra wrote into the directory names -- unlike the conditioner directory, which says "true".
# The CSV goes to c2st.recovery_dir, so it is pointed at the parent of the test_num_obs=* runs
# (the default is the launch directory).

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

base_dir=/projects/0/prjs1372/racing-diffusion-conflict

cd ${base_dir}

run_dir="multirun/rdm/model.flow_affine=True/model.flow_log_inputs=True/model.flow_num_hidden=2/model.num_bins=12/model.num_mid=128/model.training_prior.b_max=3.5/model.training_prior.b_min=0.25/model.training_prior.s_max=3.5/model.training_prior.s_min=0.25/optimizer=adam_cosine_decay_clip"

uv run --frozen --extra gpu python scripts/c2st_recovery.py c2st.mode=single \
    "c2st.recovery_dir='${run_dir}'" \
    "c2st.single_recovery_dirs=[ \
    '${run_dir}/test_num_obs=50/train_steps=100000', \
    '${run_dir}/test_num_obs=250/train_steps=100000', \
    '${run_dir}/test_num_obs=500/train_steps=100000', \
    '${run_dir}/test_num_obs=1000/train_steps=100000']"
