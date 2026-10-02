#!/bin/bash
# One-off: move the final run's results from the old override-named directories to the
# run-tag layout introduced with conf_jax/experiment/final.yaml:
#
#   outputs/<model>/<long override dirname>  ->  outputs/<model>/<run_tag>[/<deviation>]
#   multirun/<model>/<long override dirname>/test_num_obs=N/train_steps=100000
#                                            ->  multirun/<model>/<run_tag>/test_num_obs=N
#
# Only the final run moves; every other directory stays where it is. Run it from the
# repository root -- locally and on Snellius. Without --apply it only prints what it would
# do. It never overwrites: a destination that already exists is reported and skipped.
#
# Usage:
#   bash scripts/migrate_to_run_tag.sh            # dry run
#   bash scripts/migrate_to_run_tag.sh --apply

set -euo pipefail

apply=false
if [[ ${1:-} == "--apply" ]]; then
    apply=true
elif [[ $# -gt 0 ]]; then
    echo "usage: $0 [--apply]" >&2
    exit 1
fi

tag=affine_log_deep_clip_100k

flow_true="model.flow_affine=true/model.flow_log_inputs=true/model.flow_num_hidden=2/model.num_bins=12/model.num_mid=128"
flow_True="model.flow_affine=True/model.flow_log_inputs=True/model.flow_num_hidden=2/model.num_bins=12/model.num_mid=128"
rdm_box="model.training_prior.b_max=3.5/model.training_prior.b_min=0.25/model.training_prior.s_max=3.5/model.training_prior.s_min=0.25"
opt="optimizer=adam_cosine_decay_clip"

move() {
    local src=$1 dest=$2
    if [[ ! -e ${src} ]]; then
        echo "missing  ${src}"
    elif [[ -e ${dest} ]]; then
        echo "EXISTS   ${dest} (not moved: ${src})"
    elif ${apply}; then
        mkdir -p "$(dirname "${dest}")"
        mv "${src}" "${dest}"
        # Remove the override-named parents this left empty; stops at the first non-empty one.
        rmdir -p --ignore-fail-on-non-empty "$(dirname "${src}")"
        echo "moved    ${src} -> ${dest}"
    else
        echo "would mv ${src} -> ${dest}"
    fi
}

# Conditioners and hierarchical recoveries. The CRDM dt = 0.0005 run moves first: the
# coarser flows become subdirectories of it.
move "outputs/rdm/${flow_true}/${rdm_box}/${opt}/train_steps=100000" "outputs/rdm/${tag}"
move "outputs/crdm/${flow_true}/model.sampler.dt=0.0005/${opt}/train_steps=100000" "outputs/crdm/${tag}"
for dt in 0.005 0.05; do
    move "outputs/crdm/${flow_true}/model.sampler.dt=${dt}/${opt}/train_steps=100000" \
        "outputs/crdm/${tag}/model.sampler.dt=${dt}"
done

# Single-subject recoveries. Hydra spelled these sweeps' booleans "True", and the training
# steps were a trailing directory level.
rdm_single="multirun/rdm/${flow_True}/${rdm_box}/${opt}"
crdm_single="multirun/crdm/${flow_True}/model.recovery_prior.v_c_slope_loc=2.5/model.sampler.dt=0.0005/${opt}"
for n in 50 250 500 1000; do
    move "${rdm_single}/test_num_obs=${n}/train_steps=100000" "multirun/rdm/${tag}/test_num_obs=${n}"
    move "${crdm_single}/test_num_obs=${n}/train_steps=100000" "multirun/crdm/${tag}/test_num_obs=${n}"
done
move "${rdm_single}/c2st.csv" "multirun/rdm/${tag}/c2st.csv"
