#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: bash collect_data.sh TASK_NAME TASK_CONFIG GPU_ID" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_name=$1
task_config=$2
gpu_id=$3

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_id"
python script/collect_data.py "$task_name" "$task_config"
