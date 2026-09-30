#!/usr/bin/env bash
set -Eeuo pipefail

# Transition from the four-slot smoke batch to four independent task batches.
# Each task stays on one eval GPU and one SAM3 GPU, while its benchmark-listed
# seeds run sequentially.  The four task batches run in parallel.  Defaults
# preserve the original 4x20 launch; SEED_COUNT and INSTRUCTION_SET may select a
# separately labelled evaluation condition.  Shutdown is Ctrl+C/SIGINT-only;
# this script never sends TERM/KILL.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CANONICAL_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
REPO_ROOT="${REPO_ROOT:-${CANONICAL_REPO_ROOT}}"
REPO_ROOT="$(realpath -m -- "${REPO_ROOT}")"
if [[ "${REPO_ROOT}" != "${CANONICAL_REPO_ROOT}" ]]; then
  echo "REPO_ROOT must be the copied RoboHarn-Evo benchmark: ${CANONICAL_REPO_ROOT}" >&2
  exit 2
fi
ROBOHARN_EVO_PROJECT_ROOT="$(cd -- "${REPO_ROOT}/../.." && pwd -P)"
FORMAL_PYTHONPATH="${ROBOHARN_EVO_PROJECT_ROOT}:${REPO_ROOT}"
RMBENCH_OUTPUT_ROOT="${RMBENCH_OUTPUT_ROOT:-${ROBOHARN_EVO_PROJECT_ROOT}/eval_result/rmbench}"
RMBENCH_OUTPUT_ROOT="$(realpath -m -- "${RMBENCH_OUTPUT_ROOT}")"
ALLOWED_OUTPUT_ROOT="$(realpath -m -- "${ROBOHARN_EVO_PROJECT_ROOT}/eval_result/rmbench")"
if [[ "${RMBENCH_OUTPUT_ROOT}" != "${ALLOWED_OUTPUT_ROOT}" \
      && "${RMBENCH_OUTPUT_ROOT}" != "${ALLOWED_OUTPUT_ROOT}/"* ]]; then
  echo "RMBENCH_OUTPUT_ROOT must stay under ${ALLOWED_OUTPUT_ROOT}: ${RMBENCH_OUTPUT_ROOT}" >&2
  exit 2
fi
RMBENCH_ASSETS_ROOT="${RMBENCH_ASSETS_ROOT:-}"
ROBOHARN_EVO_SAM3_REPO="${ROBOHARN_EVO_SAM3_REPO:-}"
ROBOHARN_EVO_SAM3_CHECKPOINT="${ROBOHARN_EVO_SAM3_CHECKPOINT:-}"
ROBOHARN_EVO_SAM3_BPE_PATH="${ROBOHARN_EVO_SAM3_BPE_PATH:-}"
CAUSALWAM_ROOT="${CAUSALWAM_ROOT:-}"
GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-${RMBENCH_OUTPUT_ROOT}/runtime_locks/gpus}"
PYTHON="${PYTHON:-python}"
SCHEDULER="${REPO_ROOT}/policy/roboharn_evo/scripts/run_gpt55_parallel4_isolated.py"
export PYTHONPATH="${FORMAL_PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
SMOKE_SESSION="${SMOKE_SESSION:-gpt55_parallel4_smoke_20260812_172256}"
FORMAL_STAMP="${FORMAL_STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
SEED_COUNT="${SEED_COUNT:-20}"
INSTRUCTION_SET="${INSTRUCTION_SET:-rmbench_original}"

if [[ ! "${SEED_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEED_COUNT must be a positive integer: ${SEED_COUNT}" >&2
  exit 2
fi
if [[ ! "${INSTRUCTION_SET}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "INSTRUCTION_SET must be a catalog identifier: ${INSTRUCTION_SET}" >&2
  exit 2
fi

BATCH_SHAPE="4x${SEED_COUNT}"
FORMAL_ROOT="${FORMAL_ROOT:-${RMBENCH_OUTPUT_ROOT}/formal_runs/rmbench_gpt55_formal${BATCH_SHAPE}_${FORMAL_STAMP}}"
FORMAL_ROOT="$(realpath -m -- "${FORMAL_ROOT}")"
GPU_LOCK_ROOT="$(realpath -m -- "${GPU_LOCK_ROOT}")"
for runtime_path in "${FORMAL_ROOT}" "${GPU_LOCK_ROOT}"; do
  if [[ "${runtime_path}" != "${ALLOWED_OUTPUT_ROOT}" \
        && "${runtime_path}" != "${ALLOWED_OUTPUT_ROOT}/"* ]]; then
    echo "Formal runtime paths must stay under ${ALLOWED_OUTPUT_ROOT}: ${runtime_path}" >&2
    exit 2
  fi
done
if [[ "${INSTRUCTION_SET}" == "rmbench_original" ]]; then
  DEFAULT_RUN_LABEL="formalv3_parallel${BATCH_SHAPE}_seedmanifest_9104_${FORMAL_STAMP}"
else
  DEFAULT_RUN_LABEL="formalv3_parallel${BATCH_SHAPE}_seedmanifest_${INSTRUCTION_SET}_9104_${FORMAL_STAMP}"
fi
RUN_LABEL="${RUN_LABEL:-${DEFAULT_RUN_LABEL}}"

TASKS=(rearrange_blocks swap_blocks swap_T battery_try)
EVAL_GPUS=(4 5 6 7)
SAM3_GPUS=(0 1 2 3)
SAM3_PORTS=(9311 9312 9313 9314)

mkdir -p "${FORMAL_ROOT}"
exec > >(tee -a "${FORMAL_ROOT}/transition.log") 2>&1

echo "Formal transition started: $(date --iso-8601=seconds)"
echo "Formal root: ${FORMAL_ROOT}"
echo "Run label: ${RUN_LABEL}"
echo "Seed count per task: ${SEED_COUNT}"
echo "Instruction set: ${INSTRUCTION_SET}"

if tmux has-session -t "=${SMOKE_SESSION}" 2>/dev/null; then
  echo "Sending Ctrl+C to smoke session: ${SMOKE_SESSION}"
  tmux send-keys -t "=${SMOKE_SESSION}" C-c
  while tmux has-session -t "=${SMOKE_SESSION}" 2>/dev/null; do
    sleep 2
  done
  echo "Smoke session exited cleanly after Ctrl+C."
else
  echo "Smoke session is not present; continuing: ${SMOKE_SESSION}"
fi

printf 'task\teval_gpu\tsam3_gpu\tsam3_port\tseed_count\tseeds\ttmux_session\tbatch_log_root\n' \
  > "${FORMAL_ROOT}/formal_plan.tsv"

for index in "${!TASKS[@]}"; do
  task="${TASKS[index]}"
  seed_file="${REPO_ROOT}/data/data/${task}/demo_clean/seed.txt"
  if [[ ! -f "${seed_file}" ]]; then
    echo "Missing benchmark seed manifest: ${seed_file}" >&2
    exit 2
  fi

  seeds="$(${PYTHON} - "${seed_file}" "${SEED_COUNT}" <<'PY'
from pathlib import Path
import sys

values = Path(sys.argv[1]).read_text(encoding="utf-8").split()
count = int(sys.argv[2])
if len(values) < count:
    raise SystemExit(f"seed manifest contains only {len(values)} entries")
chosen = [str(int(value)) for value in values[:count]]
if len(set(chosen)) != count:
    raise SystemExit(f"first {count} seed-manifest entries are not unique")
print(",".join(chosen))
PY
)"

  session="gpt55_formal${BATCH_SHAPE}_${FORMAL_STAMP}_${task}"
  batch_root="${FORMAL_ROOT}/${task}"
  if tmux has-session -t "=${session}" 2>/dev/null; then
    echo "Refusing an existing formal tmux session: ${session}" >&2
    exit 2
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${task}" "${EVAL_GPUS[index]}" "${SAM3_GPUS[index]}" \
    "${SAM3_PORTS[index]}" "${SEED_COUNT}" "${seeds}" "${session}" "${batch_root}" \
    >> "${FORMAL_ROOT}/formal_plan.tsv"

  echo "Launching task=${task} eval_gpu=${EVAL_GPUS[index]} sam3_gpu=${SAM3_GPUS[index]} seeds=${seeds}"
  env \
    REPO_ROOT="${REPO_ROOT}" \
    ROBOHARN_EVO_PROJECT_ROOT="${ROBOHARN_EVO_PROJECT_ROOT}" \
    PYTHONPATH="${FORMAL_PYTHONPATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    RMBENCH_ROOT="${REPO_ROOT}" \
    RMBENCH_OUTPUT_ROOT="${RMBENCH_OUTPUT_ROOT}" \
    ROBOHARN_EVO_OUTPUT_ROOT="${RMBENCH_OUTPUT_ROOT}" \
    RMBENCH_ASSETS_ROOT="${RMBENCH_ASSETS_ROOT}" \
    ROBOHARN_EVO_SAM3_REPO="${ROBOHARN_EVO_SAM3_REPO}" \
    ROBOHARN_EVO_SAM3_CHECKPOINT="${ROBOHARN_EVO_SAM3_CHECKPOINT}" \
    ROBOHARN_EVO_SAM3_BPE_PATH="${ROBOHARN_EVO_SAM3_BPE_PATH}" \
    CAUSALWAM_ROOT="${CAUSALWAM_ROOT}" \
    GPU_LOCK_ROOT="${GPU_LOCK_ROOT}" \
    PARALLEL_SLOTS=1 \
    GPU_IDS="${EVAL_GPUS[index]}" \
    SAM3_GPU_IDS="${SAM3_GPUS[index]}" \
    SAM3_BASE_PORT="${SAM3_PORTS[index]}" \
    TASKS_CSV="${task}" \
    EVAL_START_SEEDS_CSV="${seeds}" \
    ALLOW_SHARED_GPU=0 \
    ALLOW_OCCUPIED_GPU=0 \
    ALLOW_VERIFIED_CAUSALWAM_GPU_OCCUPANCY=1 \
    RMBENCH_RENDER_DEVICE=pci:auto \
    TASK_CONFIG=demo_clean \
    INSTRUCTION_SET="${INSTRUCTION_SET}" \
    POLICY_NAME=policy.roboharn_evo.deploy_policy \
    PERCEPTION_CONDITION=no_oracle \
    AGENT_API_BASE_URL=http://127.0.0.1:9104 \
    MAX_OBJECTS=8 \
    EVAL_STEP_LIMIT=150 \
    MAX_ROUNDS=10 \
    MAX_CONTROL_TURNS=64 \
    MAX_NO_PROGRESS_CONTROL_TURNS=10 \
    CONTINUE_ON_ERROR=0 \
    SAM3_REAL_PROBE=1 \
    BATCH_STAMP="${FORMAL_STAMP}_${task}" \
    RUN_LABEL="${RUN_LABEL}" \
    BATCH_LOG_ROOT="${batch_root}" \
    BATCH_SESSION="${session}" \
    "${PYTHON}" "${SCHEDULER}" --detach
done

echo "All four formal task batches were submitted: $(date --iso-8601=seconds)"
echo "Plan: ${FORMAL_ROOT}/formal_plan.tsv"
echo "Transition log: ${FORMAL_ROOT}/transition.log"
