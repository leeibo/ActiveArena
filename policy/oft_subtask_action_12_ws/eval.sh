#!/usr/bin/env bash
set -euo pipefail

policy_name="$(basename "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
task_name=${1:?task_name}
gpu_id=${2:-5}
port=${3:-19001}
task_config=${4:-info_gathering_demo}
ckpt_setting=${5:-${policy_name}}
seed=${6:-0}
test_num=${7:-${ACTIVEARENA_EVAL_TEST_NUM:-${ROBOTWIN_EVAL_TEST_NUM:-50}}}

python_bin=${ACTIVEARENA_PYTHON:-${ROBOTWIN_PYTHON:-${ROBOTWIN_PY:-python}}}

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (env side): ${gpu_id}\033[0m"
echo -e "\033[33mActiveArena policy server port: ${port}\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVEARENA_ROOT="${ACTIVEARENA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ACTIVEARENA_ROOT}"

PYTHONWARNINGS=ignore::UserWarning "${python_bin}" script/eval_policy.py \
    --config "policy/${policy_name}/deploy_policy.yml" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --port "${port}" \
    --test_num "${test_num}"
