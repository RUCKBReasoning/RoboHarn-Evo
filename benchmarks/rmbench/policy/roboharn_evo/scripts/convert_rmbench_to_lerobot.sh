#!/usr/bin/env bash
set -euo pipefail

RAW_DIR=${1:?usage: convert_rmbench_to_lerobot.sh <raw_rmbench_dir> <repo_id>}
REPO_ID=${2:?usage: convert_rmbench_to_lerobot.sh <raw_rmbench_dir> <repo_id>}
TASK_CONFIG=${3:-demo_clean}
INSTRUCTION_TYPE=${4:-seen}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

python policy/roboharn_evo/scripts/convert_rmbench_to_lerobot.py \
  --raw-dir "${RAW_DIR}" \
  --repo-id "${REPO_ID}" \
  --task-config "${TASK_CONFIG}" \
  --instruction-type "${INSTRUCTION_TYPE}"
