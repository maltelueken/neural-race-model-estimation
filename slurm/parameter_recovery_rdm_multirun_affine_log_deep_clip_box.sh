#!/bin/bash
#SBATCH --job-name=rec_rdm_affine_log_deep_clip_box
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_multirun_rdm_affine_log_deep_clip_box_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_multirun_rdm_affine_log_deep_clip_box_%j.err

# Single-subject recovery (50, 250, 500 and 1000 trials) with the RDM flow trained by
# train_conditioner_rdm_affine_log_deep_clip_box.sh. The single-subject recovery prior
# (conf_jax/prior/rdm_informed.yaml) puts fewer than 0.05% of draws outside the box.
#
# Training box passed on the command line rather than edited into conf_jax/prior/wald_uniform.yaml,
# so the existing plain RDM checkpoint keeps the clamp it was trained with: s and b are held in
# [0.25, 3.5], v stays in [0, 8]. The hierarchical RDM prior has s_true and b 0.01% quantiles
# near 0.69 and a median near 1.49, so the lower limit sits well below it and the upper one
# excludes only ~0.1% of initial SMC particles (clamped at inference). The overrides also name
# the output directory, and the log-input constants are derived from them, so every job that
# uses this checkpoint must pass exactly the same ones.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

base_dir=/projects/0/prjs1372/racing-diffusion-conflict

cd ${base_dir}

uv run --frozen --extra gpu python scripts/parameter_recovery.py --multirun \
    device=gpu \
    model=rdm \
    model.flow_affine=true \
    model.flow_log_inputs=true \
    model.flow_num_hidden=2 \
    model.num_bins=12 \
    model.num_mid=128 \
    model.training_prior.s_min=0.25 \
    model.training_prior.s_max=3.5 \
    model.training_prior.b_min=0.25 \
    model.training_prior.b_max=3.5 \
    train_steps=100000 \
    optimizer=adam_cosine_decay_clip \
    test_num_obs=50,250,500,1000 \
    conditioner_dir="${base_dir}/outputs/rdm/model.flow_affine\=true/model.flow_log_inputs\=true/model.flow_num_hidden\=2/model.num_bins\=12/model.num_mid\=128/model.training_prior.b_max\=3.5/model.training_prior.b_min\=0.25/model.training_prior.s_max\=3.5/model.training_prior.s_min\=0.25/optimizer\=adam_cosine_decay_clip/train_steps\=100000"
