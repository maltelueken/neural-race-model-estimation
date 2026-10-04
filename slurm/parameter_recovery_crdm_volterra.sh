#!/bin/bash
#SBATCH --job-name=rec_volterra_crdm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --exclusive
#SBATCH --partition=genoa
#SBATCH --time=120:00:00
#SBATCH --output=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.out
#SBATCH --error=/projects/0/prjs1372/racing-diffusion-conflict/slurm/logs/%x_%j.err

# Single-subject CRDM recovery with the Volterra likelihood, on the data sets of the flow
# recovery (slurm/parameter_recovery_crdm_single.sh, which must have run). One job per trial
# count:
#
#     sbatch slurm/parameter_recovery_crdm_volterra.sh 500 [DT]
#
# Every (data set, chain) is its own single-threaded process, run as many at a time as the
# cores and the node's memory allow, longest solver grid first; see the `pool` command of
# scripts/parameter_recovery_volterra.py. Finished chains are kept, so
# a job that hits the time limit is resumed by resubmitting the same command. The merge at the
# end writes multirun/crdm/<run_tag>/test_num_obs=<N>/parameter_recovery_volterra_dt<DT>.nc
# with whatever data sets are complete; it can also be run on its own.
#
# Estimated at dt = 0.0005 from single-thread solve timings on a laptop, assuming ~30
# leapfrog steps per warm-up iteration and ~20 per draw: 2-4k core-hours per trial count, a
# median chain of 4-7 h and a slowest of 22 / 50 / 58 / 75 h for 50 / 250 / 500 / 1000
# trials. The slowest chain, not the total, sets the wall time.

set -euo pipefail

num_obs="${1:?usage: sbatch slurm/parameter_recovery_crdm_volterra.sh NUM_OBS [DT]}"
dt="${2:-0.0005}"

module purge
module load 2025

# Make sure that uv is on path
export PATH="$HOME/.local/bin:$PATH"

cd /projects/0/prjs1372/racing-diffusion-conflict

# One thread per process: the parallelism is across chains, and XLA's intra-op threads buy
# little on a sequential recursion while 192 processes each spawning a full pool would thrash.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1

# Resolve the interpreter once: 192 concurrent `uv run`s would each re-check the environment.
# `--extra gpu` keeps the shared environment as the GPU jobs left it; the run itself is on CPU.
python=$(uv run --frozen --extra gpu python -c 'import sys; print(sys.executable)')
script=scripts/parameter_recovery_volterra.py

# 192 crashing processes each writing a multi-GB core file would fill the project quota.
ulimit -c 0

# One process per chain, as many at a time as the cores and the node's memory allow: at this
# dt memory runs out long before cores do (a 3000-step grid peaks at ~3 GB per chain, a
# 6400-step one at ~12 GB), and overcommitting it showed up as bus errors and segfaults.
# Failed chains are logged and leave no file, so the next submission retries them.
"${python}" "${script}" pool --num-obs "${num_obs}" --dt "${dt}" \
    --processes "${SLURM_CPUS_PER_TASK:-192}" \
    --log-dir "slurm/logs/volterra_n${num_obs}_dt${dt}"

"${python}" "${script}" merge --num-obs "${num_obs}" --dt "${dt}"
