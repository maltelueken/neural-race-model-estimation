#!/bin/bash
# Copy the figures of one run into figures/final/, dropping the run tag from the name:
#   figures/<name>_<RUN_TAG>.<ext>  ->  figures/final/<name>.<ext>
#
# Existing files in figures/final/ with the same name are overwritten.
#
# Usage:
#   bash scripts/finalize_figures.sh affine_log_deep_clip_100k

set -euo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
    echo "usage: $0 RUN_TAG" >&2
    exit 1
fi
run_tag=$1

figures_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/figures"
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
