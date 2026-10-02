#!/bin/bash
#SBATCH --job-name=rec_crdm_affine_log_deep_clip_box
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_multirun_crdm_affine_log_deep_clip_box_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/parameter_recovery_multirun_crdm_affine_log_deep_clip_box_%j.err

# Single-subject recovery (50, 250, 500 and 1000 trials) with the affine, log-input,
# two-hidden-layer CRDM flow trained for 100k steps with gradient clipping on the
# raised-minimum training box (s, b >= 0.25, tau >= 0.01).
#
# The single-subject recovery prior (conf_jax/prior/crdm_informed.yaml, v_c_slope_loc=2.5)
# puts about 3% of datasets below the box: tau < 0.01 in 1.4%, b < 0.25 in 1.4%, s_true < 0.25
# in 0.4%. The flow's inputs are clamped to the box there, so recovery for those few datasets
# is not exact; the hierarchical prior does not reach that region.

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

base_dir=/projects/0/prjs1372/racing-diffusion-conflict

cd ${base_dir}

uv run --frozen --extra gpu python scripts/parameter_recovery.py --multirun \
    device=gpu \
    model=crdm \
    model.sampler.dt=0.0005 \
    model.flow_affine=true \
    model.flow_log_inputs=true \
    model.flow_num_hidden=2 \
    model.num_bins=12 \
    model.num_mid=128 \
    train_steps=100000 \
    optimizer=adam_cosine_decay_clip \
    test_num_obs=50,250,500,1000 \
    model.recovery_prior.v_c_slope_loc=2.5 \
    conditioner_dir="${base_dir}/outputs/crdm/model.flow_affine\=true/model.flow_log_inputs\=true/model.flow_num_hidden\=2/model.num_bins\=12/model.num_mid\=128/model.sampler.dt\=0.0005/optimizer\=adam_cosine_decay_clip/train_steps\=100000"
