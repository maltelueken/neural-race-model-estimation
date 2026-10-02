#!/bin/bash
# Copy the figures of one run into figures/final/, dropping the run tag from the name:
#   figures/<name>_<RUN_TAG>.<ext>  ->  figures/final/<name>.<ext>
#
# RUN_TAG defaults to the final run's, read from conf_jax/experiment/final.yaml. Existing
# files in figures/final/ with the same name are overwritten. Figures that do not depend on
# a run (crdm_phenomena, rdm_crdm_illustration, hierarchical_prior_*) carry no tag and are not
# copied.
#
# Usage:
#   bash scripts/finalize_figures.sh [RUN_TAG]

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -gt 1 ]]; then
    echo "usage: $0 [RUN_TAG]" >&2
    exit 1
fi
run_tag=${1:-$(sed -n 's/^run_tag: *//p' "${root}/conf_jax/experiment/final.yaml")}
if [[ -z ${run_tag} ]]; then
    echo "no run_tag found in conf_jax/experiment/final.yaml" >&2
    exit 1
fi

figures_dir="${root}/figures"
final_dir="${figures_dir}/final"

shopt -s nullglob
sources=("${figures_dir}"/*_"${run_tag}".*)
if [[ ${#sources[@]} -eq 0 ]]; then
    echo "no figures tagged '${run_tag}' in ${figures_dir}" >&2
    exit 1
fi

mkdir -p "${final_dir}"
for src in "${sources[@]}"; do
    file=$(basename "${src}")
    ext=${file##*.}
    stem=${file%.*}
    dest="${final_dir}/${stem%_"${run_tag}"}.${ext}"
    cp "${src}" "${dest}"
    echo "${file} -> final/$(basename "${dest}")"
done
