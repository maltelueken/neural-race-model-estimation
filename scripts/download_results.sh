#!/bin/bash
# Download the final run's results from Snellius into the same paths here:
#
#   outputs/{rdm,crdm}/<run_tag>/       conditioners, hierarchical recoveries, C2ST
#   multirun/{rdm,crdm}/<run_tag>/      single-subject recoveries
#   outputs/compare_densities/          density comparison
#
# RUN_TAG defaults to the final run's, read from conf_jax/experiment/final.yaml. rsync only
# transfers what changed, so rerunning after a new job is cheap.
#
# Usage:
#   bash scripts/download_results.sh [RUN_TAG]

set -euo pipefail

remote=mluken@snellius:/projects/prjs1372/racing-diffusion-conflict

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

run_tag=${1:-$(sed -n 's/^run_tag: *//p' conf_jax/experiment/final.yaml)}
if [[ -z ${run_tag} ]]; then
    echo "no run_tag found in conf_jax/experiment/final.yaml" >&2
    exit 1
fi

for dir in outputs/rdm/"${run_tag}" outputs/crdm/"${run_tag}" \
    multirun/rdm/"${run_tag}" multirun/crdm/"${run_tag}" outputs/compare_densities; do
    mkdir -p "${dir}"
    rsync -av "${remote}/${dir}/" "${dir}/"
done
