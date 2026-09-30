#!/usr/bin/env bash
set -euo pipefail

REPO_ID=${1:?usage: train_pi05_robotwin.sh <lerobot_repo_id> <exp_name> [max_frames]}
EXP_NAME=${2:?usage: train_pi05_robotwin.sh <lerobot_repo_id> <exp_name> [max_frames]}
MAX_FRAMES=${3:-}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${MAX_FRAMES}" ]]; then
  python policy/roboharn_evo/scripts/compute_pi05_robotwin_norm_stats.py \
    --repo-id "${REPO_ID}" \
    --exp-name "${EXP_NAME}" \
    --max-frames "${MAX_FRAMES}"
else
  python policy/roboharn_evo/scripts/compute_pi05_robotwin_norm_stats.py \
    --repo-id "${REPO_ID}" \
    --exp-name "${EXP_NAME}"
fi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python policy/roboharn_evo/scripts/train_pi05_robotwin.py \
  --repo-id "${REPO_ID}" \
  --exp-name "${EXP_NAME}" \
  --overwrite
