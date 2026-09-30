#!/usr/bin/env bash
set -euo pipefail

unalias python 2>/dev/null || true
unalias pip 2>/dev/null || true
hash -r

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to benchmarks/rmbench}"
RUNNER="${RUNNER:-${REPO_ROOT}/policy/roboharn_evo/scripts/run_gpt55_pure_tool_control_8way.sh}"
CONDA_SH="${CONDA_SH:-/path/to/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-RMBench}"
RUN_MODE="${RUN_MODE:-both}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_PARENT="${LOG_PARENT:-/tmp/rmbench_gpt55_object_info_ablation_${RUN_STAMP}}"
DETACH="${DETACH:-0}"
DETACH_SESSION="${DETACH_SESSION:-rmbench_gpt55_ablation_${RUN_STAMP}}"
RMBENCH_ABLATION_DETACHED_CHILD="${RMBENCH_ABLATION_DETACHED_CHILD:-0}"
DETACHED_LAUNCHER_LOG="${DETACHED_LAUNCHER_LOG:-${LOG_PARENT}/launcher.log}"
EXTRA_ARGS=("$@")

if [[ ! -x "${RUNNER}" ]]; then
  echo "Runner is not executable: ${RUNNER}" >&2
  echo "Run: chmod +x ${RUNNER}" >&2
  exit 2
fi

mkdir -p "${LOG_PARENT}"

if [[ "${RMBENCH_ABLATION_DETACHED_CHILD}" == "1" ]]; then
  exec >>"${DETACHED_LAUNCHER_LOG}" 2>&1
  echo "Detached ablation runner started at $(date --iso-8601=seconds)"
fi

if [[ "${DETACH}" == "1" && "${RMBENCH_ABLATION_DETACHED_CHILD}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "DETACH=1 requires tmux." >&2
    exit 2
  fi
  DETACH_SESSION="${DETACH_SESSION//[^a-zA-Z0-9_-]/_}"
  if [[ -z "${DETACH_SESSION}" ]]; then
    echo "DETACH_SESSION must contain at least one alphanumeric character." >&2
    exit 2
  fi
  if tmux has-session -t "=${DETACH_SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${DETACH_SESSION}" >&2
    exit 2
  fi

  wrapper_path="${BASH_SOURCE[0]}"
  if [[ "${wrapper_path}" != /* ]]; then
    wrapper_path="${REPO_ROOT}/${wrapper_path#./}"
  fi
  detached_command=(env "RMBENCH_ABLATION_DETACHED_CHILD=1" "DETACH=0")
  detached_env_names=(
    REPO_ROOT RUNNER CONDA_SH CONDA_ENV RUN_MODE RUN_STAMP LOG_PARENT
    NUM_WORKERS GPU_IDS TASK_NAME TASK_CONFIG INSTRUCTION_SET POLICY_NAME PERCEPTION_CONDITION BASE_CKPT_PREFIX N_PER_WORKER
    SEED_OFFSET EVAL_START_SEEDS REQUIRE_EXPLICIT_EVAL_START_SEEDS NON_FORMAL_DIAGNOSTIC
    AGENT_API_BASE_URL SAM3_SERVICE_URL
    SAM3_BASE_PORT PLANNER_TIMEOUT_SEC
    RECOVERY_TIMEOUT_SEC OOD_TIMEOUT_SEC QUERY_TIMEOUT_SEC PREFLIGHT_TIMEOUT_SEC SKIP_PREFLIGHT
    PREFLIGHT_ONLY SKIP_AGENT_IDENTITY_CHECK REQUIRE_AGENT_INFERENCE_PREFLIGHT
    AGENT_INFERENCE_PREFLIGHT_TIMEOUT_SEC EXPECTED_AGENT_MODEL EXPECTED_AGENT_API_MODE
    EXPECTED_REASONING_EFFORT EXPECTED_RESPONSE_STORAGE MIN_AGENT_TIMEOUT_SEC
    EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS WORKER_START_DELAY_SEC MAX_OBJECTS MAX_WAIT_STEPS
    RETRY_BUDGET MAX_ROUNDS MAX_CONTROL_TURNS MAX_NO_PROGRESS_CONTROL_TURNS BACKEND_ERROR_BUDGET EMPTY_PLAN_REPLAN_THRESHOLD
    ORACLE_MAX_OBJECTS SHUTDOWN_GRACE_SEC HEARTBEAT_SEC DETACHED_LAUNCHER_LOG
    RECORD_RUNTIME_PROVENANCE RUNTIME_PROVENANCE_PATH
  )
  for name in "${detached_env_names[@]}"; do
    if [[ -v "${name}" ]]; then
      detached_command+=("${name}=${!name}")
    fi
  done
  detached_command+=("${wrapper_path}" "${EXTRA_ARGS[@]}")
  printf -v detached_command_string '%q ' "${detached_command[@]}"

  tmux new-session -d -s "${DETACH_SESSION}" -c "${REPO_ROOT}" "${detached_command_string}"
  echo "Detached ablation session started: ${DETACH_SESSION}"
  echo "Attach: tmux attach -t ${DETACH_SESSION}"
  echo "Launcher log: ${DETACHED_LAUNCHER_LOG}"
  echo "Worker logs: ${LOG_PARENT}"
  exit 0
fi

run_condition() {
  local condition="$1"
  local require_sam3_preflight="1"
  shift
  if [[ "${condition}" == "oracle" ]]; then
    require_sam3_preflight="0"
  fi
  echo "=== ${condition} ==="
  REPO_ROOT="${REPO_ROOT}" \
  BASE_CKPT="${BASE_CKPT_PREFIX:-gpt55_pure_tool_control}_${condition}" \
  LOG_ROOT="${LOG_PARENT}/${condition}" \
  PERCEPTION_CONDITION="${condition}" \
  REQUIRE_SAM3_PREFLIGHT="${require_sam3_preflight}" \
  "${RUNNER}" "$@" "${EXTRA_ARGS[@]}"
}

case "${RUN_MODE}" in
  both)
    run_condition "no_oracle" \
      --agent.observation_preprocess.oracle_objects.enabled False
    run_condition "oracle" \
      --agent.observation_preprocess.oracle_objects.enabled True \
      --agent.observation_preprocess.oracle_objects.include_all True \
      --agent.observation_preprocess.oracle_objects.max_objects "${ORACLE_MAX_OBJECTS:-12}"
    ;;
  no_oracle)
    run_condition "no_oracle" \
      --agent.observation_preprocess.oracle_objects.enabled False
    ;;
  oracle)
    run_condition "oracle" \
      --agent.observation_preprocess.oracle_objects.enabled True \
      --agent.observation_preprocess.oracle_objects.include_all True \
      --agent.observation_preprocess.oracle_objects.max_objects "${ORACLE_MAX_OBJECTS:-12}"
    ;;
  *)
    echo "RUN_MODE must be one of: both, no_oracle, oracle" >&2
    exit 2
    ;;
esac

if [[ -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
  set +u
  conda activate "${CONDA_ENV}"
  set -u
fi

python "${REPO_ROOT}/policy/roboharn_evo/scripts/summarize_object_info_ablation.py" --input "${LOG_PARENT}"
echo "Object-info ablation logs: ${LOG_PARENT}"
