#!/usr/bin/env bash
set -Eeuo pipefail

# Run a configurable RMBench task/seed grid strictly sequentially.  Every
# child invocation uses exactly one worker and one episode, and the next
# invocation starts only after the formal trace checker has examined the
# previous one.  The historical default remains five tasks x seeds 0--5;
# TASKS_CSV and EVAL_START_SEEDS_CSV select smaller batches without changing
# the formal child launcher.  Each rollout is capped at 150 environment steps
# by default.

unalias python 2>/dev/null || true
unalias pip 2>/dev/null || true
hash -r

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
COPIED_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
REPO_ROOT="${REPO_ROOT:-${COPIED_REPO_ROOT}}"
ROBOHARN_EVO_PROJECT_ROOT="${ROBOHARN_EVO_PROJECT_ROOT:-$(cd -- "${COPIED_REPO_ROOT}/../.." && pwd -P)}"
RMBENCH_ROOT="${REPO_ROOT}"
RMBENCH_ASSETS_ROOT="${RMBENCH_ASSETS_ROOT:-}"
RMBENCH_OUTPUT_ROOT="${RMBENCH_OUTPUT_ROOT:-${ROBOHARN_EVO_PROJECT_ROOT}/eval_result/rmbench}"
EVAL_RESULT_ROOT="${RMBENCH_OUTPUT_ROOT}"
ROBOHARN_EVO_OUTPUT_ROOT="${RMBENCH_OUTPUT_ROOT}"
FORMAL_PYTHONPATH="${ROBOHARN_EVO_PROJECT_ROOT}:${REPO_ROOT}"
PYTHONDONTWRITEBYTECODE=1
RUNNER="${RUNNER:-${REPO_ROOT}/policy/roboharn_evo/scripts/run_gpt55_pure_tool_control_8way.sh}"
CHECKER="${CHECKER:-${REPO_ROOT}/policy/roboharn_evo/scripts/check_pure_tool_control_early_stop.py}"
CHECKER_PYTHON="${CHECKER_PYTHON:-python}"

GPU_ID="${GPU_ID:-4}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
INSTRUCTION_SET="${INSTRUCTION_SET:-rmbench_original}"
POLICY_NAME="${POLICY_NAME:-policy.roboharn_evo.deploy_policy}"
PERCEPTION_CONDITION="${PERCEPTION_CONDITION:-no_oracle}"
AGENT_API_BASE_URL="${AGENT_API_BASE_URL:-http://127.0.0.1:9104}"
SAM3_SERVICE_URL="${SAM3_SERVICE_URL:-http://127.0.0.1:9301}"
MAX_OBJECTS="${MAX_OBJECTS:-8}"
EVAL_STEP_LIMIT="${EVAL_STEP_LIMIT:-150}"

# Current formal Protocol v3 budgets.  The child launcher independently
# enforces these values and records them in every runtime provenance file.
MAX_ROUNDS="${MAX_ROUNDS:-10}"
MAX_CONTROL_TURNS="${MAX_CONTROL_TURNS:-64}"
MAX_NO_PROGRESS_CONTROL_TURNS="${MAX_NO_PROGRESS_CONTROL_TURNS:-10}"

# Optional simulator-infrastructure contract.  Parallel launchers set these
# explicitly; empty values preserve the historical SAPIEN auto-selection path.
RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN="${RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN:-}"
RMBENCH_RENDER_DEVICE="${RMBENCH_RENDER_DEVICE:-}"
RMBENCH_RENDER_DEVICE_STRICT="${RMBENCH_RENDER_DEVICE_STRICT:-}"
RMBENCH_EXPECTED_RENDER_CUDA_ID="${RMBENCH_EXPECTED_RENDER_CUDA_ID:-}"
RMBENCH_EXPECTED_RENDER_PCI_BUS_ID="${RMBENCH_EXPECTED_RENDER_PCI_BUS_ID:-}"
RMBENCH_EXPECTED_PHYSICAL_GPU="${RMBENCH_EXPECTED_PHYSICAL_GPU:-}"
RMBENCH_RENDER_DEVICE_PROVENANCE_PATH="${RMBENCH_RENDER_DEVICE_PROVENANCE_PATH:-}"

BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-formalv3_seq5_step${EVAL_STEP_LIMIT}_9104_${BATCH_STAMP}}"
BATCH_LOG_ROOT="${BATCH_LOG_ROOT:-${RMBENCH_OUTPUT_ROOT}/formal_runs/rmbench_gpt55_five_tasks_sequential_${BATCH_STAMP}}"
SEGMENTATION_ARTIFACT_ROOT="${SEGMENTATION_ARTIFACT_ROOT:-${BATCH_LOG_ROOT}/segmentation_artifacts}"
RMBENCH_RUNTIME_CONFIG_ROOT="${RMBENCH_RUNTIME_CONFIG_ROOT:-${BATCH_LOG_ROOT}/runtime_configs}"
ROBOHARN_EVO_WORKSPACE_ROOT="${ROBOHARN_EVO_WORKSPACE_ROOT:-${BATCH_LOG_ROOT}/runtime_workspace}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${BATCH_LOG_ROOT}/runtime_caches}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUNTIME_CACHE_ROOT}/xdg}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${RUNTIME_CACHE_ROOT}/cuda}"
MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/mesa}"
TORCH_HOME="${TORCH_HOME:-${RUNTIME_CACHE_ROOT}/torch}"
HF_HOME="${HF_HOME:-${RUNTIME_CACHE_ROOT}/huggingface}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/numba}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${RUNTIME_CACHE_ROOT}/matplotlib}"
TMPDIR="${TMPDIR:-${RUNTIME_CACHE_ROOT}/tmp}"
TMP="${TMP:-${RUNTIME_CACHE_ROOT}/tmp}"
TEMP="${TEMP:-${RUNTIME_CACHE_ROOT}/tmp}"
RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS="${RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS:-}"
BATCH_SESSION="${BATCH_SESSION:-gpt55_five_tasks_${BATCH_STAMP}}"
BATCH_DETACH="${BATCH_DETACH:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"
RMBENCH_SEQUENTIAL_CHILD="${RMBENCH_SEQUENTIAL_CHILD:-0}"
TASKS_CSV="${TASKS_CSV:-put_back_block,swap_T,rearrange_blocks,swap_blocks,battery_try}"
EVAL_START_SEEDS_CSV="${EVAL_START_SEEDS_CSV:-0,1,2,3,4,5}"
RUN_JOBS_CSV="${RUN_JOBS_CSV:-}"
CONTROL_ACTION=""

current_runner_pid=""
current_runner_starttime=""
current_run_token=""
current_run_index=""
current_run_task=""
current_run_seed=""
batch_token=""
batch_start_ticks=""
control_boot_id=""
batch_shutdown_started=0

IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
IFS=',' read -r -a SEEDS <<< "${EVAL_START_SEEDS_CSV}"
JOB_TASKS=()
JOB_SEEDS=()

usage() {
  cat <<'EOF'
Usage:
  bash policy/roboharn_evo/scripts/run_gpt55_six_tasks_sequential.sh [option]

Options:
  --dry-run      Print all 30 task/seed invocations without launching them.
  --detach       Run the complete sequential batch in one tmux session.
  --foreground   Run in the current terminal (the default).
  --skip-current Gracefully stop only the current rollout, record it as
                 skipped, and continue with the next planned rollout.
  --batch-log-root DIR
                 Batch log/control directory used with --skip-current.
  -h, --help     Show this help.

Useful environment overrides:
  GPU_ID=4
  BATCH_DETACH=1
  CONTINUE_ON_ERROR=1
  PERCEPTION_CONDITION=no_oracle|oracle
  AGENT_API_BASE_URL=http://127.0.0.1:9104
  SAM3_SERVICE_URL=http://127.0.0.1:9301
  MAX_OBJECTS=8
  EVAL_STEP_LIMIT=150
  TASKS_CSV=put_back_block,swap_T,rearrange_blocks,swap_blocks,battery_try
  EVAL_START_SEEDS_CSV=0,1,2,3,4,5
  RUN_JOBS_CSV=put_back_block:0,swap_T:1  # optional ordered subset
  SEGMENTATION_ARTIFACT_ROOT=/absolute/per-batch/artifact/root

The default condition is formal Protocol v3, rmbench_original instructions,
GPT-5.5/xhigh responses_compat on port 9104, no-oracle perception, and SAM3 on
port 9301.  Every rollout is capped at 150 environment steps by default.  A
valid task failure continues to the next rollout; an
infrastructure/trace failure stops the batch unless CONTINUE_ON_ERROR=1.

To intentionally skip one active rollout without weakening error handling:
  bash policy/roboharn_evo/scripts/run_gpt55_six_tasks_sequential.sh \
    --skip-current --batch-log-root \
    <ROBOHARN_EVO_PROJECT_ROOT>/eval_result/rmbench/formal_runs/<batch-log-directory>

Ctrl+C retains its normal meaning and stops the entire sequential batch.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --detach)
      BATCH_DETACH=1
      ;;
    --foreground)
      BATCH_DETACH=0
      ;;
    --skip-current)
      CONTROL_ACTION="skip-current"
      ;;
    --batch-log-root)
      shift
      if (( $# == 0 )); then
        echo "--batch-log-root requires a directory argument." >&2
        exit 2
      fi
      BATCH_LOG_ROOT="$1"
      ;;
    --batch-log-root=*)
      BATCH_LOG_ROOT="${1#*=}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

read_proc_identity() {
  local pid="$1"
  local -n state_ref="$2"
  local -n ppid_ref="$3"
  local -n start_ticks_ref="$4"
  local stat_line stat_tail
  local -a stat_fields=()

  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  IFS= read -r stat_line < "/proc/${pid}/stat" || return 1
  stat_tail="${stat_line##*) }"
  read -r -a stat_fields <<< "${stat_tail}"
  (( ${#stat_fields[@]} >= 20 )) || return 1
  state_ref="${stat_fields[0]}"
  ppid_ref="${stat_fields[1]}"
  start_ticks_ref="${stat_fields[19]}"
}

read_proc_identity_snapshot() {
  local pid="$1"
  local -n state_ref="$2"
  local -n ppid_ref="$3"
  local -n start_ticks_ref="$4"
  local stat_file="/proc/${pid}/stat"
  local stat_line stat_tail
  local -a stat_fields=()

  [[ "${pid}" =~ ^[1-9][0-9]*$ && -r "${stat_file}" ]] || return 1
  IFS= read -r stat_line < "${stat_file}" || return 1
  stat_tail="${stat_line##*) }"
  read -r -a stat_fields <<< "${stat_tail}"
  (( ${#stat_fields[@]} >= 20 )) || return 1
  state_ref="${stat_fields[0]}"
  ppid_ref="${stat_fields[1]}"
  start_ticks_ref="${stat_fields[19]}"
}

pid_namespace_id() {
  local pid="$1"
  local -n namespace_ref="$2"
  local namespace_link
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  namespace_link="$(readlink "/proc/${pid}/ns/pid" 2>/dev/null)" || return 1
  [[ "${namespace_link}" =~ ^pid:\[([0-9]+)\]$ ]] || return 1
  namespace_ref="${BASH_REMATCH[1]}"
}

local_pid_to_host_pid() {
  local local_pid="$1"
  local -n host_pid_ref="$2"
  local self_stat self_host_pid parent_host_pid children_file
  local candidate_host_pid candidate_local_pid
  local -a host_children=()

  IFS= read -r self_stat < /proc/self/stat || return 1
  self_host_pid="${self_stat%% *}"
  if [[ "${local_pid}" == "$$" ]]; then
    local self_tail
    local -a self_fields=()
    self_tail="${self_stat##*) }"
    read -r -a self_fields <<< "${self_tail}"
    parent_host_pid="${self_fields[1]}"
    if [[ "${parent_host_pid}" =~ ^[1-9][0-9]*$ ]] \
        && [[ -r "/proc/${parent_host_pid}/cmdline" ]] \
        && grep -zq -- "${script_path:-${BASH_SOURCE[0]}}" \
          "/proc/${parent_host_pid}/cmdline" 2>/dev/null; then
      host_pid_ref="${parent_host_pid}"
    else
      host_pid_ref="${self_host_pid}"
    fi
    [[ "${host_pid_ref}" =~ ^[1-9][0-9]*$ ]]
    return
  fi
  children_file="/proc/${self_host_pid}/task/${self_host_pid}/children"
  [[ "${local_pid}" =~ ^[1-9][0-9]*$ && -r "${children_file}" ]] || return 1
  read -r -a host_children < "${children_file}"
  for candidate_host_pid in "${host_children[@]}"; do
    candidate_local_pid="$(awk '/^NSpid:/ {print $NF}' "/proc/${candidate_host_pid}/status" 2>/dev/null || true)"
    if [[ "${candidate_local_pid}" == "${local_pid}" ]]; then
      host_pid_ref="${candidate_host_pid}"
      return 0
    fi
  done
  return 1
}

control_value() {
  local path="$1"
  local wanted_key="$2"
  local key value

  [[ -s "${path}" ]] || return 1
  while IFS=$'\t' read -r key value; do
    if [[ "${key}" == "${wanted_key}" ]]; then
      printf '%s' "${value}"
      return 0
    fi
  done < "${path}"
  return 1
}

read_current_run() {
  local current_file="$1"
  local -n schema_ref="$2"
  local -n batch_token_ref="$3"
  local -n token_ref="$4"
  local -n index_ref="$5"
  local -n task_ref="$6"
  local -n seed_ref="$7"
  local -n pid_ref="$8"
  local -n starttime_ref="$9"
  local -n state_ref="${10}"

  [[ -s "${current_file}" ]] || return 1
  IFS=$'\t' read -r \
    schema_ref batch_token_ref token_ref index_ref task_ref seed_ref \
    pid_ref starttime_ref state_ref \
    < "${current_file}"
  [[ "${schema_ref}" == "1" \
        && -n "${batch_token_ref}" \
        && -n "${token_ref}" \
        && -n "${state_ref}" ]]
}

skip_current_rollout() {
  local control_root="${BATCH_LOG_ROOT}/control"
  local batch_file="${control_root}/batch.tsv"
  local current_file="${control_root}/current.tsv"
  local schema current_batch_token token index task seed runner_pid runner_starttime state
  local now_schema now_batch_token now_token now_index now_task now_seed
  local now_pid now_starttime now_state
  local recorded_batch_token recorded_boot_id batch_pid recorded_batch_start batch_state
  local recorded_pid_namespace actual_boot_id proc_state proc_ppid actual_starttime batch_pid_namespace
  local request_dir request_tmp claim_dir
  local accepted_file rejected_file
  local ack_wait

  recorded_batch_token="$(control_value "${batch_file}" batch_token 2>/dev/null || true)"
  recorded_boot_id="$(control_value "${batch_file}" boot_id 2>/dev/null || true)"
  batch_pid="$(control_value "${batch_file}" batch_pid 2>/dev/null || true)"
  recorded_batch_start="$(control_value "${batch_file}" batch_start_ticks 2>/dev/null || true)"
  recorded_pid_namespace="$(control_value "${batch_file}" pid_namespace_id 2>/dev/null || true)"
  batch_state="$(control_value "${batch_file}" state 2>/dev/null || true)"
  actual_boot_id="$(< /proc/sys/kernel/random/boot_id)"
  if [[ -z "${recorded_batch_token}" \
        || "${recorded_boot_id}" != "${actual_boot_id}" \
        || "${batch_state}" != "running" \
        || ! "${batch_pid}" =~ ^[1-9][0-9]*$ \
        || ! "${recorded_batch_start}" =~ ^[1-9][0-9]*$ \
        ]] \
      || ! read_proc_identity \
          "${batch_pid}" proc_state proc_ppid actual_starttime \
      || [[ "${proc_state}" == "Z" \
            || "${actual_starttime}" != "${recorded_batch_start}" ]]; then
    echo "No live compatible sequential batch owns ${control_root}; nothing was signalled." >&2
    return 4
  fi
  if ! pid_namespace_id "${batch_pid}" batch_pid_namespace; then
    echo "Could not verify the live batch PID namespace; nothing was signalled." >&2
    return 4
  fi
  if [[ "${recorded_pid_namespace}" != "${batch_pid_namespace}" ]]; then
    echo "The batch PID namespace changed; nothing was signalled." >&2
    return 4
  fi

  if ! read_current_run "${current_file}" \
      schema current_batch_token token index task seed \
      runner_pid runner_starttime state; then
    echo "No active rollout control record found: ${current_file}" >&2
    echo "The batch may be idle, finished, or predates skip-current support." >&2
    return 3
  fi
  if [[ "${current_batch_token}" != "${recorded_batch_token}" ]]; then
    echo "The current-run record belongs to a different batch; nothing was signalled." >&2
    return 4
  fi
  if [[ "${state}" == "skip_accepted" ]]; then
    echo "Skip is already being handled for [${index}] task=${task} seed=${seed}."
    return 0
  fi
  if [[ "${state}" == "starting" ]]; then
    echo "The rollout is still preparing its eval worker; try skip-current again when state=ready." >&2
    return 3
  fi
  if [[ "${state}" != "ready" ]]; then
    echo "No rollout is currently skippable (state=${state}, task=${task}, seed=${seed})." >&2
    return 3
  fi
  if [[ ! "${runner_pid}" =~ ^[1-9][0-9]*$ \
        || ! "${runner_starttime}" =~ ^[1-9][0-9]*$ ]] \
      || ! read_proc_identity \
          "${runner_pid}" proc_state proc_ppid actual_starttime \
      || [[ "${proc_state}" == "Z" \
            || "${proc_ppid}" != "${batch_pid}" \
            || "${actual_starttime}" != "${runner_starttime}" ]]; then
    echo "The current runner is stale or is not owned by this batch; nothing was signalled." >&2
    return 4
  fi

  request_dir="${control_root}/requests/${token}"
  mkdir -p "${request_dir}"
  claim_dir="${request_dir}/skip.claim"
  accepted_file="${request_dir}/skip_accepted.tsv"
  rejected_file="${request_dir}/skip_rejected.tsv"
  if ! mkdir "${claim_dir}" 2>/dev/null; then
    echo "Skip was already requested for [${index}] task=${task} seed=${seed}."
    return 0
  fi
  request_tmp="${request_dir}/skip_requested.tsv.tmp.$$"
  {
    printf 'schema_version\t1\n'
    printf 'batch_token\t%s\n' "${recorded_batch_token}"
    printf 'run_token\t%s\n' "${token}"
    printf 'requester_pid\t%d\n' "$$"
    printf 'requested_at\t%s\n' "$(date --iso-8601=seconds)"
  } > "${request_tmp}"
  mv -f "${request_tmp}" "${request_dir}/skip_requested.tsv"

  # Re-read after publishing the token-scoped request.  The controller never
  # signals a PID; only the live batch that owns the runner may accept it.
  if ! read_current_run "${current_file}" \
      now_schema now_batch_token now_token now_index now_task now_seed \
      now_pid now_starttime now_state \
      || [[ "${now_batch_token}" != "${recorded_batch_token}" \
            || "${now_token}" != "${token}" \
            || "${now_pid}" != "${runner_pid}" \
            || "${now_starttime}" != "${runner_starttime}" \
            || ( "${now_state}" != "starting" \
                 && "${now_state}" != "ready" \
                 && "${now_state}" != "skip_accepted" ) ]]; then
    echo "The batch advanced before the request was accepted; the new rollout was left untouched." >&2
    return 5
  fi

  for (( ack_wait = 0; ack_wait < 50; ack_wait++ )); do
    if [[ -f "${accepted_file}" ]]; then
      echo "Skip accepted for [${index}] task=${task} seed=${seed}; Ctrl+C is being delivered to its eval worker."
      return 0
    fi
    if [[ -f "${rejected_file}" ]]; then
      echo "The skip request was too late or could not be accepted; no later rollout was signalled." >&2
      return 5
    fi
    sleep 0.1
  done
  echo "Skip request queued for [${index}] task=${task} seed=${seed}; it will be accepted only after the eval worker is safely registered."
}

if [[ -n "${CONTROL_ACTION}" ]]; then
  if [[ "${CONTROL_ACTION}" != "skip-current" ]]; then
    echo "Unknown control action: ${CONTROL_ACTION}" >&2
    exit 2
  fi
  skip_current_rollout
  exit $?
fi

canonical_path() {
  realpath -m -- "$1"
}

path_is_within() {
  local candidate boundary
  candidate="$(canonical_path "$1")"
  boundary="$(canonical_path "$2")"
  [[ "${candidate}" == "${boundary}" || "${candidate}" == "${boundary}/"* ]]
}

resolved_repo_root="$(canonical_path "${REPO_ROOT}")"
if [[ "${resolved_repo_root}" != "${COPIED_REPO_ROOT}" ]]; then
  echo "REPO_ROOT must be this copied benchmark and may not select a donor checkout: expected ${COPIED_REPO_ROOT}, got ${resolved_repo_root}" >&2
  exit 2
fi
expected_project_root="$(canonical_path "${resolved_repo_root}/../..")"
resolved_project_root="$(canonical_path "${ROBOHARN_EVO_PROJECT_ROOT}")"
if [[ "${resolved_project_root}" != "${expected_project_root}" ]]; then
  echo "ROBOHARN_EVO_PROJECT_ROOT must contain the copied benchmark: expected ${expected_project_root}, got ${resolved_project_root}" >&2
  exit 2
fi
REPO_ROOT="${resolved_repo_root}"
ROBOHARN_EVO_PROJECT_ROOT="${resolved_project_root}"
RMBENCH_ROOT="${REPO_ROOT}"
FORMAL_PYTHONPATH="${ROBOHARN_EVO_PROJECT_ROOT}:${REPO_ROOT}"
allowed_runtime_root="$(canonical_path "${ROBOHARN_EVO_PROJECT_ROOT}/eval_result/rmbench")"
runtime_path_names=(
  RMBENCH_OUTPUT_ROOT BATCH_LOG_ROOT SEGMENTATION_ARTIFACT_ROOT
  RMBENCH_RUNTIME_CONFIG_ROOT ROBOHARN_EVO_WORKSPACE_ROOT RUNTIME_CACHE_ROOT
  XDG_CACHE_HOME CUDA_CACHE_PATH MESA_SHADER_CACHE_DIR TORCH_HOME HF_HOME
  TRITON_CACHE_DIR NUMBA_CACHE_DIR MPLCONFIGDIR TMPDIR TMP TEMP
)
for runtime_path_name in "${runtime_path_names[@]}"; do
  runtime_path_value="${!runtime_path_name}"
  if ! path_is_within "${runtime_path_value}" "${allowed_runtime_root}"; then
    echo "${runtime_path_name} must stay under ${allowed_runtime_root}: ${runtime_path_value}" >&2
    exit 2
  fi
done
if path_is_within "${RMBENCH_OUTPUT_ROOT}" "${REPO_ROOT}" \
    || path_is_within "${REPO_ROOT}" "${RMBENCH_OUTPUT_ROOT}"; then
  echo "RMBENCH_OUTPUT_ROOT must not overlap copied benchmark source ${REPO_ROOT}" >&2
  exit 2
fi
if [[ -n "${RMBENCH_ASSETS_ROOT}" ]] \
    && { path_is_within "${RMBENCH_OUTPUT_ROOT}" "${RMBENCH_ASSETS_ROOT}" \
         || path_is_within "${RMBENCH_ASSETS_ROOT}" "${RMBENCH_OUTPUT_ROOT}"; }; then
  echo "RMBENCH_OUTPUT_ROOT must not overlap RMBENCH_ASSETS_ROOT" >&2
  exit 2
fi
if [[ -n "${RMBENCH_RENDER_DEVICE_PROVENANCE_PATH}" ]] \
    && ! path_is_within "${RMBENCH_RENDER_DEVICE_PROVENANCE_PATH}" "${allowed_runtime_root}"; then
  echo "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH must stay under ${allowed_runtime_root}" >&2
  exit 2
fi

export \
  PYTHONPATH="${FORMAL_PYTHONPATH}" \
  PYTHONDONTWRITEBYTECODE \
  RMBENCH_ROOT RMBENCH_ASSETS_ROOT RMBENCH_OUTPUT_ROOT ROBOHARN_EVO_OUTPUT_ROOT \
  RMBENCH_RUNTIME_CONFIG_ROOT ROBOHARN_EVO_WORKSPACE_ROOT \
  XDG_CACHE_HOME CUDA_CACHE_PATH MESA_SHADER_CACHE_DIR TORCH_HOME HF_HOME \
  TRITON_CACHE_DIR NUMBA_CACHE_DIR MPLCONFIGDIR TMPDIR TMP TEMP

require_boolean() {
  local name="$1"
  local value="$2"
  case "${value}" in
    0|1) ;;
    *)
      echo "${name} must be 0 or 1, got: ${value}" >&2
      exit 2
      ;;
  esac
}

require_boolean BATCH_DETACH "${BATCH_DETACH}"
require_boolean CONTINUE_ON_ERROR "${CONTINUE_ON_ERROR}"
require_boolean DRY_RUN "${DRY_RUN}"
require_boolean RMBENCH_SEQUENTIAL_CHILD "${RMBENCH_SEQUENTIAL_CHILD}"

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer, got: ${GPU_ID}" >&2
  exit 2
fi
if [[ ! "${MAX_OBJECTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_OBJECTS must be a positive integer, got: ${MAX_OBJECTS}" >&2
  exit 2
fi
if [[ ! "${EVAL_STEP_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_STEP_LIMIT must be a positive integer, got: ${EVAL_STEP_LIMIT}" >&2
  exit 2
fi
if [[ ! "${RUN_LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "RUN_LABEL may contain only letters, digits, '.', '_', and '-': ${RUN_LABEL}" >&2
  exit 2
fi
if [[ "${SEGMENTATION_ARTIFACT_ROOT}" != /* ]]; then
  echo "SEGMENTATION_ARTIFACT_ROOT must be an absolute path: ${SEGMENTATION_ARTIFACT_ROOT}" >&2
  exit 2
fi
if (( ${#TASKS[@]} == 0 )); then
  echo "TASKS_CSV must contain at least one task." >&2
  exit 2
fi
declare -A seen_tasks=()
for index in "${!TASKS[@]}"; do
  task="${TASKS[index]//[[:space:]]/}"
  if [[ -z "${task}" || ! "${task}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "TASKS_CSV contains an invalid task name: ${TASKS[index]}" >&2
    exit 2
  fi
  if [[ -n "${seen_tasks[${task}]+x}" ]]; then
    echo "TASKS_CSV contains a duplicate task: ${task}" >&2
    exit 2
  fi
  seen_tasks["${task}"]=1
  TASKS[index]="${task}"
done
if (( ${#SEEDS[@]} == 0 )); then
  echo "EVAL_START_SEEDS_CSV must contain at least one seed." >&2
  exit 2
fi
declare -A seen_seeds=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[index]//[[:space:]]/}"
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "EVAL_START_SEEDS_CSV contains a non-negative integer seed requirement violation: ${SEEDS[index]}" >&2
    exit 2
  fi
  canonical_seed="$((10#${seed}))"
  if [[ -n "${seen_seeds[${canonical_seed}]+x}" ]]; then
    echo "EVAL_START_SEEDS_CSV contains a duplicate canonical seed: ${seed}" >&2
    exit 2
  fi
  seen_seeds["${canonical_seed}"]=1
  SEEDS[index]="${canonical_seed}"
done

# RUN_JOBS_CSV lets an outer scheduler give this single-GPU sequential
# launcher an ordered, non-overlapping subset of the declared task/seed grid.
# An empty value deliberately reconstructs the historical task-major Cartesian
# product byte-for-byte in the same order.
if [[ -n "${RUN_JOBS_CSV//[[:space:]]/}" ]]; then
  compact_run_jobs="${RUN_JOBS_CSV//[[:space:]]/}"
  if [[ "${compact_run_jobs}" == ,* \
        || "${compact_run_jobs}" == *, \
        || "${compact_run_jobs}" == *,,* ]]; then
    echo "RUN_JOBS_CSV contains an empty job: ${RUN_JOBS_CSV}" >&2
    exit 2
  fi
  IFS=',' read -r -a run_job_specs <<< "${compact_run_jobs}"
  declare -A seen_run_jobs=()
  for run_job_spec in "${run_job_specs[@]}"; do
    if [[ ! "${run_job_spec}" =~ ^([A-Za-z0-9_]+):([0-9]+)$ ]]; then
      echo "RUN_JOBS_CSV entries must use task:seed, got: ${run_job_spec}" >&2
      exit 2
    fi
    job_task="${BASH_REMATCH[1]}"
    job_seed="$((10#${BASH_REMATCH[2]}))"
    if [[ -z "${seen_tasks[${job_task}]+x}" ]]; then
      echo "RUN_JOBS_CSV task is not declared in TASKS_CSV: ${job_task}" >&2
      exit 2
    fi
    if [[ -z "${seen_seeds[${job_seed}]+x}" ]]; then
      echo "RUN_JOBS_CSV seed is not declared in EVAL_START_SEEDS_CSV: ${job_seed}" >&2
      exit 2
    fi
    job_key="${job_task}:${job_seed}"
    if [[ -n "${seen_run_jobs[${job_key}]+x}" ]]; then
      echo "RUN_JOBS_CSV contains a duplicate canonical job: ${run_job_spec}" >&2
      exit 2
    fi
    seen_run_jobs["${job_key}"]=1
    JOB_TASKS+=("${job_task}")
    JOB_SEEDS+=("${job_seed}")
  done
else
  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      JOB_TASKS+=("${task}")
      JOB_SEEDS+=("${seed}")
    done
  done
fi
if (( ${#JOB_TASKS[@]} == 0 )); then
  echo "The rollout plan must contain at least one task/seed job." >&2
  exit 2
fi
if [[ "${MAX_ROUNDS}" != "10" || "${MAX_CONTROL_TURNS}" != "64" || "${MAX_NO_PROGRESS_CONTROL_TURNS}" != "10" ]]; then
  echo "This script is formal Protocol v3 and requires MAX_ROUNDS/MAX_CONTROL_TURNS/MAX_NO_PROGRESS_CONTROL_TURNS=10/64/10." >&2
  exit 2
fi

case "${PERCEPTION_CONDITION}" in
  no_oracle)
    REQUIRE_SAM3_PREFLIGHT=1
    ;;
  oracle)
    REQUIRE_SAM3_PREFLIGHT=0
    ;;
  *)
    echo "PERCEPTION_CONDITION must be no_oracle or oracle, got: ${PERCEPTION_CONDITION}" >&2
    exit 2
    ;;
esac

for required_file in \
    "${RUNNER}" "${CHECKER}" "${CHECKER_PYTHON}" \
    "${REPO_ROOT}/task_config/${TASK_CONFIG}.yml"; do
  if [[ ! -e "${required_file}" ]]; then
    echo "Required path is missing: ${required_file}" >&2
    exit 2
  fi
done

for task in "${TASKS[@]}"; do
  required_paths=(
    "${REPO_ROOT}/envs/${task}.py"
    "${REPO_ROOT}/data/data/${task}/${TASK_CONFIG}"
  )
  if [[ "${INSTRUCTION_SET}" == "rmbench_original" ]]; then
    required_paths+=("${REPO_ROOT}/description/task_instruction/${task}.json")
  else
    required_paths+=("${REPO_ROOT}/description/task_instruction_sets/${INSTRUCTION_SET}/${task}.json")
  fi
  for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Task ${task} is missing required path: ${required_path}" >&2
      exit 2
    fi
  done
done

total_runs=${#JOB_TASKS[@]}

print_plan() {
  local index=0
  local task seed seed_tag base_ckpt expected_ckpt
  if [[ -n "${RUN_JOBS_CSV//[[:space:]]/}" ]]; then
    echo "Sequential rollout plan: ${total_runs} explicitly assigned task/seed jobs"
  else
    echo "Sequential rollout plan: ${#TASKS[@]} tasks x ${#SEEDS[@]} seeds = ${total_runs} runs"
  fi
  echo "Condition: ${PERCEPTION_CONDITION}; instruction_set=${INSTRUCTION_SET}; GPU=${GPU_ID}; max_objects=${MAX_OBJECTS}; env_step_limit=${EVAL_STEP_LIMIT}"
  echo "Protocol: formal-v3 max_rounds/control_turns/no_progress=${MAX_ROUNDS}/${MAX_CONTROL_TURNS}/${MAX_NO_PROGRESS_CONTROL_TURNS}"
  for job_index in "${!JOB_TASKS[@]}"; do
    task="${JOB_TASKS[job_index]}"
    seed="${JOB_SEEDS[job_index]}"
    base_ckpt="gpt55_pure_tool_control_${task}_${RUN_LABEL}_${PERCEPTION_CONDITION}"
    index=$((index + 1))
    printf -v seed_tag '%06d' "${seed}"
    expected_ckpt="${base_ckpt}_w0_g${GPU_ID}_s0_e${seed}"
    printf '[%02d/%02d] task=%-22s episode=%s eval_start_seed=%d ckpt=%s\n' \
      "${index}" "${total_runs}" "${task}" "${seed_tag}" "${seed}" "${expected_ckpt}"
  done
}

if [[ "${DRY_RUN}" == "1" ]]; then
  print_plan
  exit 0
fi

if [[ -z "${RMBENCH_ASSETS_ROOT//[[:space:]]/}" ]]; then
  echo "RMBENCH_ASSETS_ROOT must explicitly name read-only licensed assets for a live rollout; there is no donor fallback." >&2
  exit 2
fi
for required_asset_dir in \
    "${RMBENCH_ASSETS_ROOT}/embodiments" \
    "${RMBENCH_ASSETS_ROOT}/objects"; do
  if [[ ! -d "${required_asset_dir}" ]]; then
    echo "Required read-only asset directory is missing: ${required_asset_dir}" >&2
    exit 2
  fi
done

mkdir -p \
  "${BATCH_LOG_ROOT}" \
  "${SEGMENTATION_ARTIFACT_ROOT}" \
  "${RMBENCH_RUNTIME_CONFIG_ROOT}" \
  "${ROBOHARN_EVO_WORKSPACE_ROOT}" \
  "${XDG_CACHE_HOME}" \
  "${CUDA_CACHE_PATH}" \
  "${MESA_SHADER_CACHE_DIR}" \
  "${TORCH_HOME}" \
  "${HF_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${NUMBA_CACHE_DIR}" \
  "${MPLCONFIGDIR}" \
  "${TMPDIR}"

script_path="${BASH_SOURCE[0]}"
if [[ "${script_path}" != /* ]]; then
  script_path="${REPO_ROOT}/${script_path#./}"
fi

if [[ "${BATCH_DETACH}" == "1" && "${RMBENCH_SEQUENTIAL_CHILD}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "--detach requires tmux." >&2
    exit 2
  fi
  session_name="${BATCH_SESSION//[^A-Za-z0-9_-]/_}"
  if [[ -z "${session_name}" ]]; then
    echo "BATCH_SESSION must contain at least one alphanumeric character." >&2
    exit 2
  fi
  if tmux has-session -t "=${session_name}" 2>/dev/null; then
    echo "tmux session already exists: ${session_name}" >&2
    exit 2
  fi
  mkdir -p "${BATCH_LOG_ROOT}"
  detached_command=(
    /usr/bin/env
    "RMBENCH_SEQUENTIAL_CHILD=1"
    "BATCH_DETACH=0"
    "REPO_ROOT=${REPO_ROOT}"
    "ROBOHARN_EVO_PROJECT_ROOT=${ROBOHARN_EVO_PROJECT_ROOT}"
    "PYTHONPATH=${FORMAL_PYTHONPATH}"
    "PYTHONDONTWRITEBYTECODE=${PYTHONDONTWRITEBYTECODE}"
    "RMBENCH_ROOT=${RMBENCH_ROOT}"
    "RMBENCH_ASSETS_ROOT=${RMBENCH_ASSETS_ROOT}"
    "RMBENCH_OUTPUT_ROOT=${RMBENCH_OUTPUT_ROOT}"
    "ROBOHARN_EVO_OUTPUT_ROOT=${ROBOHARN_EVO_OUTPUT_ROOT}"
    "RMBENCH_RUNTIME_CONFIG_ROOT=${RMBENCH_RUNTIME_CONFIG_ROOT}"
    "ROBOHARN_EVO_WORKSPACE_ROOT=${ROBOHARN_EVO_WORKSPACE_ROOT}"
    "RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT}"
    "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
    "CUDA_CACHE_PATH=${CUDA_CACHE_PATH}"
    "MESA_SHADER_CACHE_DIR=${MESA_SHADER_CACHE_DIR}"
    "TORCH_HOME=${TORCH_HOME}"
    "HF_HOME=${HF_HOME}"
    "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
    "NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR}"
    "MPLCONFIGDIR=${MPLCONFIGDIR}"
    "TMPDIR=${TMPDIR}"
    "TMP=${TMP}"
    "TEMP=${TEMP}"
    "RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS=${RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS}"
    "RUNNER=${RUNNER}"
    "CHECKER=${CHECKER}"
    "CHECKER_PYTHON=${CHECKER_PYTHON}"
    "GPU_ID=${GPU_ID}"
    "TASK_CONFIG=${TASK_CONFIG}"
    "INSTRUCTION_SET=${INSTRUCTION_SET}"
    "POLICY_NAME=${POLICY_NAME}"
    "PERCEPTION_CONDITION=${PERCEPTION_CONDITION}"
    "AGENT_API_BASE_URL=${AGENT_API_BASE_URL}"
    "SAM3_SERVICE_URL=${SAM3_SERVICE_URL}"
    "MAX_OBJECTS=${MAX_OBJECTS}"
    "EVAL_STEP_LIMIT=${EVAL_STEP_LIMIT}"
    "MAX_ROUNDS=${MAX_ROUNDS}"
    "MAX_CONTROL_TURNS=${MAX_CONTROL_TURNS}"
    "MAX_NO_PROGRESS_CONTROL_TURNS=${MAX_NO_PROGRESS_CONTROL_TURNS}"
    "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN=${RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN}"
    "RMBENCH_RENDER_DEVICE=${RMBENCH_RENDER_DEVICE}"
    "RMBENCH_RENDER_DEVICE_STRICT=${RMBENCH_RENDER_DEVICE_STRICT}"
    "RMBENCH_EXPECTED_RENDER_CUDA_ID=${RMBENCH_EXPECTED_RENDER_CUDA_ID}"
    "RMBENCH_EXPECTED_RENDER_PCI_BUS_ID=${RMBENCH_EXPECTED_RENDER_PCI_BUS_ID}"
    "RMBENCH_EXPECTED_PHYSICAL_GPU=${RMBENCH_EXPECTED_PHYSICAL_GPU}"
    "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH=${RMBENCH_RENDER_DEVICE_PROVENANCE_PATH}"
    "BATCH_STAMP=${BATCH_STAMP}"
    "RUN_LABEL=${RUN_LABEL}"
    "BATCH_LOG_ROOT=${BATCH_LOG_ROOT}"
    "SEGMENTATION_ARTIFACT_ROOT=${SEGMENTATION_ARTIFACT_ROOT}"
    "BATCH_SESSION=${session_name}"
    "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
    "TASKS_CSV=${TASKS_CSV}"
    "EVAL_START_SEEDS_CSV=${EVAL_START_SEEDS_CSV}"
    "RUN_JOBS_CSV=${RUN_JOBS_CSV}"
  )
  if [[ -n "${CUDA_DEVICE_ORDER:-}" ]]; then
    # tmux servers may retain an older environment than their client.
    detached_command+=("CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER}")
  fi
  detached_command+=(/bin/bash "${script_path}" --foreground)
  printf -v detached_command_string '%q ' "${detached_command[@]}"
  tmux new-session -d -s "${session_name}" -c "${REPO_ROOT}" "${detached_command_string}"
  echo "Sequential batch started in tmux: ${session_name}"
  echo "Attach: tmux attach -t ${session_name}"
  echo "Batch log: ${BATCH_LOG_ROOT}/batch.log"
  echo "Status ledger: ${BATCH_LOG_ROOT}/batch_status.tsv"
  exit 0
fi

mkdir -p "${BATCH_LOG_ROOT}"
exec > >(tee -a "${BATCH_LOG_ROOT}/batch.log") 2>&1

plan_file="${BATCH_LOG_ROOT}/batch_plan.tsv"
status_file="${BATCH_LOG_ROOT}/batch_status.tsv"
control_root="${BATCH_LOG_ROOT}/control"
current_file="${control_root}/current.tsv"
mkdir -p "${control_root}/requests"
printf 'index\ttask\tepisode\teval_start_seed\tcheckpoint\n' > "${plan_file}"
printf 'index\ttask\tepisode\teval_start_seed\tlauncher_rc\tstrict_rc\tstatus\tresult_dir\toutcome\n' > "${status_file}"

publish_current_run() {
  local token="$1"
  local run_index="$2"
  local task="$3"
  local seed="$4"
  local runner_pid="$5"
  local runner_starttime="$6"
  local state="$7"
  local tmp="${current_file}.tmp.$$"
  printf '1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${batch_token}" "${token}" "${run_index}" "${task}" "${seed}" \
    "${runner_pid}" "${runner_starttime}" "${state}" > "${tmp}"
  mv -f "${tmp}" "${current_file}"
}

publish_batch_state() {
  local state="$1"
  local tmp="${control_root}/batch.tsv.tmp.$$"
  {
    printf 'schema_version\t1\n'
    printf 'batch_token\t%s\n' "${batch_token}"
    printf 'boot_id\t%s\n' "${control_boot_id}"
    printf 'batch_pid\t%s\n' "${batch_host_pid}"
    printf 'batch_start_ticks\t%s\n' "${batch_start_ticks}"
    printf 'pid_namespace_id\t%s\n' "${control_pid_namespace_id}"
    printf 'script_realpath\t%s\n' "${script_path}"
    printf 'state\t%s\n' "${state}"
    printf 'updated_at\t%s\n' "$(date --iso-8601=seconds)"
  } > "${tmp}"
  mv -f "${tmp}" "${control_root}/batch.tsv"
}

write_request_marker() {
  local marker_path="$1"
  local status="$2"
  local reason="$3"
  local tmp="${marker_path}.tmp.$$"
  {
    printf 'schema_version\t1\n'
    printf 'batch_token\t%s\n' "${batch_token}"
    printf 'run_token\t%s\n' "${current_run_token}"
    printf 'status\t%s\n' "${status}"
    printf 'reason\t%s\n' "${reason}"
    printf 'timestamp\t%s\n' "$(date --iso-8601=seconds)"
  } > "${tmp}"
  mv -f "${tmp}" "${marker_path}"
}

ready_file_is_owned() {
  local ready_file="$1"
  local ready_batch ready_run ready_boot ready_owner ready_owner_start
  local ready_runner ready_runner_start ready_runner_ppid ready_policy
  local worker_count worker_rows=0
  local key worker_pid worker_start worker_ppid extra
  local worker_state actual_worker_ppid actual_worker_start
  local runner_state actual_runner_ppid actual_runner_start

  ready_batch="$(control_value "${ready_file}" batch_token 2>/dev/null || true)"
  ready_run="$(control_value "${ready_file}" run_token 2>/dev/null || true)"
  ready_boot="$(control_value "${ready_file}" boot_id 2>/dev/null || true)"
  ready_owner="$(control_value "${ready_file}" owner_pid 2>/dev/null || true)"
  ready_owner_start="$(control_value "${ready_file}" owner_start_ticks 2>/dev/null || true)"
  ready_runner="$(control_value "${ready_file}" runner_pid 2>/dev/null || true)"
  ready_runner_start="$(control_value "${ready_file}" runner_start_ticks 2>/dev/null || true)"
  ready_runner_ppid="$(control_value "${ready_file}" runner_ppid 2>/dev/null || true)"
  ready_policy="$(control_value "${ready_file}" interrupt_policy 2>/dev/null || true)"
  worker_count="$(control_value "${ready_file}" worker_count 2>/dev/null || true)"
  if [[ "${ready_batch}" != "${batch_token}" ]]; then echo "Skip readiness batch token mismatch." >&2; return 1; fi
  if [[ "${ready_run}" != "${current_run_token}" ]]; then echo "Skip readiness run token mismatch." >&2; return 1; fi
  if [[ "${ready_boot}" != "${control_boot_id}" ]]; then echo "Skip readiness boot ID mismatch." >&2; return 1; fi
  if [[ "${ready_owner}" != "${batch_host_pid}" ]]; then echo "Skip readiness owner PID mismatch: ${ready_owner} != ${batch_host_pid}." >&2; return 1; fi
  if [[ "${ready_owner_start}" != "${batch_start_ticks}" ]]; then echo "Skip readiness owner start mismatch." >&2; return 1; fi
  if [[ "${ready_runner}" != "${current_runner_host_pid}" ]]; then echo "Skip readiness runner PID mismatch: ${ready_runner} != ${current_runner_host_pid}." >&2; return 1; fi
  if [[ "${ready_runner_start}" != "${current_runner_starttime}" ]]; then echo "Skip readiness runner start mismatch: ${ready_runner_start} != ${current_runner_starttime}." >&2; return 1; fi
  if [[ "${ready_runner_ppid}" != "${batch_host_pid}" ]]; then echo "Skip readiness runner owner mismatch." >&2; return 1; fi
  if [[ "${ready_policy}" != "ctrl_c_only" ]]; then echo "Skip readiness interrupt policy mismatch." >&2; return 1; fi
  if [[ ! "${worker_count}" =~ ^[1-9][0-9]*$ ]]; then echo "Skip readiness worker count is invalid." >&2; return 1; fi
  if ! read_proc_identity_snapshot \
      "${ready_runner}" runner_state actual_runner_ppid actual_runner_start \
      || [[ "${runner_state}" == "Z" \
            || "${actual_runner_ppid}" != "${batch_host_pid}" \
            || "${actual_runner_start}" != "${ready_runner_start}" ]]; then
    echo "Skip readiness runner identity is no longer live or owned." >&2
    return 1
  fi

  while IFS=$'\t' read -r key worker_pid worker_start worker_ppid extra; do
    [[ "${key}" == "worker" ]] || continue
    [[ -z "${extra}" \
          && "${worker_pid}" =~ ^[1-9][0-9]*$ \
          && "${worker_start}" =~ ^[1-9][0-9]*$ \
          && "${worker_ppid}" == "${current_runner_host_pid}" ]] || {
      echo "Skip readiness contains malformed worker identity metadata." >&2
      return 1
    }
    if ! read_proc_identity_snapshot \
        "${worker_pid}" worker_state actual_worker_ppid actual_worker_start \
        || [[ "${worker_state}" == "Z" \
              || "${actual_worker_ppid}" != "${worker_ppid}" \
              || "${actual_worker_start}" != "${worker_start}" ]]; then
      echo "Skip readiness worker identity is no longer live or owned." >&2
      return 1
    fi
    worker_rows=$((worker_rows + 1))
  done < "${ready_file}"
  if [[ "${worker_rows}" != "${worker_count}" ]]; then
    echo "Skip readiness worker count mismatch: rows=${worker_rows}, expected=${worker_count}." >&2
    return 1
  fi
}

cleanup_marker_is_owned() {
  local marker_file="$1"
  local expected_status="$2"
  local marker_batch marker_run marker_runner marker_status
  marker_batch="$(control_value "${marker_file}" batch_token 2>/dev/null || true)"
  marker_run="$(control_value "${marker_file}" run_token 2>/dev/null || true)"
  marker_runner="$(control_value "${marker_file}" runner_pid 2>/dev/null || true)"
  marker_status="$(control_value "${marker_file}" status 2>/dev/null || true)"
  [[ "${marker_batch}" == "${batch_token}" \
        && "${marker_run}" == "${current_run_token}" \
        && "${marker_runner}" == "${current_runner_host_pid}" \
        && "${marker_status}" == "${expected_status}" ]]
}

ready_workers_are_gone() {
  local ready_file="$1"
  local key worker_pid worker_start worker_ppid extra
  local worker_state actual_worker_ppid actual_worker_start
  while IFS=$'\t' read -r key worker_pid worker_start worker_ppid extra; do
    [[ "${key}" == "worker" ]] || continue
    if read_proc_identity_snapshot \
        "${worker_pid}" worker_state actual_worker_ppid actual_worker_start \
        && [[ "${actual_worker_start}" == "${worker_start}" ]]; then
      echo "Owned eval worker ${worker_pid} still exists after cleanup completion." >&2
      return 1
    fi
  done < "${ready_file}"
}

skip_request_is_current() {
  local request_file="$1"
  local request_batch request_run
  request_batch="$(control_value "${request_file}" batch_token 2>/dev/null || true)"
  request_run="$(control_value "${request_file}" run_token 2>/dev/null || true)"
  [[ "${request_batch}" == "${batch_token}" \
        && "${request_run}" == "${current_run_token}" ]]
}

shutdown_sequential_batch() {
  local signal_name="$1"
  local exit_code="$2"
  if [[ "${batch_shutdown_started}" == "1" ]]; then
    exit "${exit_code}"
  fi
  batch_shutdown_started=1
  trap '' INT TERM
  publish_batch_state batch_interrupted
  echo "Received ${signal_name}; stopping the current rollout and the sequential batch." >&2
  if [[ -n "${current_runner_pid}" \
        && "${current_runner_pid}" =~ ^[1-9][0-9]*$ ]]; then
    publish_current_run \
      "${current_run_token:--}" "${current_run_index:-0}" \
      "${current_run_task:--}" "${current_run_seed:-0}" \
      "${current_runner_pid}" "${current_runner_starttime:-0}" batch_interrupted
    /bin/kill -INT "${current_runner_pid}" 2>/dev/null || true
    wait "${current_runner_pid}" 2>/dev/null || true
  fi
  publish_current_run '-' 0 '-' 0 0 0 batch_interrupted
  echo "Sequential batch interrupted; no later rollout was started." >&2
  exit "${exit_code}"
}

trap 'shutdown_sequential_batch SIGINT 130' INT
trap 'shutdown_sequential_batch SIGTERM 143' TERM

control_boot_id="$(< /proc/sys/kernel/random/boot_id)"
if ! local_pid_to_host_pid "$$" batch_host_pid \
    || ! read_proc_identity \
      "${batch_host_pid}" _batch_proc_state _batch_proc_ppid batch_start_ticks \
    || ! pid_namespace_id "${batch_host_pid}" control_pid_namespace_id; then
  echo "Could not read sequential batch process identity." >&2
  exit 2
fi
batch_token="${BATCH_STAMP}_${batch_host_pid}_${batch_start_ticks}_$(date +%s%N)"
publish_batch_state running
publish_current_run '-' 0 '-' 0 0 0 idle

index=0
for job_index in "${!JOB_TASKS[@]}"; do
  task="${JOB_TASKS[job_index]}"
  seed="${JOB_SEEDS[job_index]}"
  base_ckpt="gpt55_pure_tool_control_${task}_${RUN_LABEL}_${PERCEPTION_CONDITION}"
  index=$((index + 1))
  printf -v seed_tag '%06d' "${seed}"
  expected_ckpt="${base_ckpt}_w0_g${GPU_ID}_s0_e${seed}"
  printf '%d\t%s\t%s\t%d\t%s\n' \
    "${index}" "${task}" "${seed_tag}" "${seed}" "${expected_ckpt}" >> "${plan_file}"
done

echo "Batch started at $(date --iso-8601=seconds)"
echo "Repository: ${REPO_ROOT}"
echo "Batch logs: ${BATCH_LOG_ROOT}"
print_plan

failures=0
skipped=0
index=0
for job_index in "${!JOB_TASKS[@]}"; do
  task="${JOB_TASKS[job_index]}"
  seed="${JOB_SEEDS[job_index]}"
  base_ckpt="gpt55_pure_tool_control_${task}_${RUN_LABEL}_${PERCEPTION_CONDITION}"
  index=$((index + 1))
  printf -v seed_tag '%06d' "${seed}"
  expected_ckpt="${base_ckpt}_w0_g${GPU_ID}_s0_e${seed}"
  result_dir="${EVAL_RESULT_ROOT}/${task}/${POLICY_NAME}/${TASK_CONFIG}/${expected_ckpt}"
  run_log_root="${BATCH_LOG_ROOT}/${task}/episode_${seed_tag}"
  segmentation_artifact_dir="${SEGMENTATION_ARTIFACT_ROOT}/${task}/episode_${seed_tag}"
  mkdir -p "${run_log_root}" "${segmentation_artifact_dir}"

    echo
    echo "======================================================================"
    echo "[${index}/${total_runs}] START task=${task} episode=${seed_tag} eval_start_seed=${seed}"
    echo "Expected result: ${result_dir}"
    echo "Started: $(date --iso-8601=seconds)"
    echo "======================================================================"

    current_run_token="${BATCH_STAMP}_${index}_${task}_${seed}_$$_$(date +%s%N)"
    current_run_index="${index}"
    current_run_task="${task}"
    current_run_seed="${seed}"
    skip_request_dir="${control_root}/requests/${current_run_token}"
    skip_ready_file="${skip_request_dir}/runner_ready.tsv"
    skip_accepted_file="${skip_request_dir}/skip_accepted.tsv"
    skip_cleanup_complete_file="${skip_request_dir}/cleanup_complete.tsv"
    skip_cleanup_failed_file="${skip_request_dir}/cleanup_failed.tsv"
    mkdir -p "${skip_request_dir}"

    set +e
    /usr/bin/env \
      -u ALL_PROXY -u all_proxy \
      -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
      -u SAM3_BASE_PORT \
      NO_PROXY=127.0.0.1,localhost \
      no_proxy=127.0.0.1,localhost \
      REPO_ROOT="${REPO_ROOT}" \
      ROBOHARN_EVO_PROJECT_ROOT="${ROBOHARN_EVO_PROJECT_ROOT}" \
      PYTHONPATH="${FORMAL_PYTHONPATH}" \
      PYTHONDONTWRITEBYTECODE=1 \
      RMBENCH_ROOT="${RMBENCH_ROOT}" \
      RMBENCH_ASSETS_ROOT="${RMBENCH_ASSETS_ROOT}" \
      RMBENCH_OUTPUT_ROOT="${run_log_root}" \
      ROBOHARN_EVO_OUTPUT_ROOT="${run_log_root}" \
      RMBENCH_RUNTIME_CONFIG_ROOT="${RMBENCH_RUNTIME_CONFIG_ROOT}" \
      ROBOHARN_EVO_WORKSPACE_ROOT="${ROBOHARN_EVO_WORKSPACE_ROOT}" \
      XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
      CUDA_CACHE_PATH="${CUDA_CACHE_PATH}" \
      MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR}" \
      TORCH_HOME="${TORCH_HOME}" \
      HF_HOME="${HF_HOME}" \
      TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
      NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}" \
      MPLCONFIGDIR="${MPLCONFIGDIR}" \
      TMPDIR="${TMPDIR}" \
      TMP="${TMP}" \
      TEMP="${TEMP}" \
      RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS="${RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS}" \
      NUM_WORKERS=1 \
      GPU_IDS="${GPU_ID}" \
      RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN="${RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN}" \
      TASK_NAME="${task}" \
      TASK_CONFIG="${TASK_CONFIG}" \
      INSTRUCTION_SET="${INSTRUCTION_SET}" \
      POLICY_NAME="${POLICY_NAME}" \
      PERCEPTION_CONDITION="${PERCEPTION_CONDITION}" \
      REQUIRE_SAM3_PREFLIGHT="${REQUIRE_SAM3_PREFLIGHT}" \
      BASE_CKPT="${base_ckpt}" \
      N_PER_WORKER=1 \
      SEED_OFFSET=0 \
      EVAL_START_SEEDS="${seed}" \
      REQUIRE_EXPLICIT_EVAL_START_SEEDS=1 \
      NON_FORMAL_DIAGNOSTIC=0 \
      AGENT_API_BASE_URL="${AGENT_API_BASE_URL}" \
      SAM3_SERVICE_URL="${SAM3_SERVICE_URL}" \
      EXPECTED_AGENT_MODEL=gpt-5.5 \
      EXPECTED_AGENT_API_MODE=responses_compat \
      EXPECTED_REASONING_EFFORT=xhigh \
      EXPECTED_RESPONSE_STORAGE=account_default \
      EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS=4 \
      REQUIRE_AGENT_INFERENCE_PREFLIGHT=1 \
      RECORD_RUNTIME_PROVENANCE=1 \
      MAX_OBJECTS="${MAX_OBJECTS}" \
      MAX_ROUNDS="${MAX_ROUNDS}" \
      MAX_CONTROL_TURNS="${MAX_CONTROL_TURNS}" \
      MAX_NO_PROGRESS_CONTROL_TURNS="${MAX_NO_PROGRESS_CONTROL_TURNS}" \
      RMBENCH_RENDER_DEVICE="${RMBENCH_RENDER_DEVICE}" \
      RMBENCH_RENDER_DEVICE_STRICT="${RMBENCH_RENDER_DEVICE_STRICT}" \
      RMBENCH_EXPECTED_RENDER_CUDA_ID="${RMBENCH_EXPECTED_RENDER_CUDA_ID}" \
      RMBENCH_EXPECTED_RENDER_PCI_BUS_ID="${RMBENCH_EXPECTED_RENDER_PCI_BUS_ID}" \
      RMBENCH_EXPECTED_PHYSICAL_GPU="${RMBENCH_EXPECTED_PHYSICAL_GPU}" \
      RMBENCH_RENDER_DEVICE_PROVENANCE_PATH="${run_log_root}/renderer_device_provenance.json" \
      INTERRUPT_ESCALATION_POLICY=ctrl_c_only \
      ROBOHARN_EVO_SKIP_READY_FILE="${skip_ready_file}" \
      ROBOHARN_EVO_SKIP_ACCEPTED_FILE="${skip_accepted_file}" \
      ROBOHARN_EVO_SKIP_CLEANUP_COMPLETE_FILE="${skip_cleanup_complete_file}" \
      ROBOHARN_EVO_SKIP_CLEANUP_FAILED_FILE="${skip_cleanup_failed_file}" \
      ROBOHARN_EVO_CONTROL_BATCH_TOKEN="${batch_token}" \
      ROBOHARN_EVO_CONTROL_RUN_TOKEN="${current_run_token}" \
      ROBOHARN_EVO_CONTROL_OWNER_PID="${batch_host_pid}" \
      ROBOHARN_EVO_CONTROL_OWNER_START_TICKS="${batch_start_ticks}" \
      ROBOHARN_EVO_CONTROL_BOOT_ID="${control_boot_id}" \
      WORKER_START_DELAY_SEC=0 \
      RUN_STAMP="${BATCH_STAMP}_${task}_${seed_tag}" \
      LOG_ROOT="${run_log_root}" \
      SEGMENTATION_ARTIFACT_ROOT="${segmentation_artifact_dir}" \
      DETACH=0 \
      /usr/bin/env --default-signal=INT --default-signal=QUIT \
      /bin/bash "${RUNNER}" \
      --eval.step_limit "${EVAL_STEP_LIMIT}" \
      --output_root "${EVAL_RESULT_ROOT}" &
    current_runner_pid=$!
    current_runner_host_pid=0
    current_runner_starttime=0
    publish_current_run \
      "${current_run_token}" "${index}" "${task}" "${seed}" \
      "${current_runner_host_pid}" "${current_runner_starttime}" starting
    echo "Skip current rollout: bash ${script_path} --skip-current --batch-log-root ${BATCH_LOG_ROOT}"

    skip_was_accepted=0
    while /bin/kill -0 "${current_runner_pid}" 2>/dev/null; do
      if [[ -f "${skip_ready_file}" ]]; then
        current_runner_host_pid="$(control_value "${skip_ready_file}" runner_pid 2>/dev/null || true)"
        current_runner_starttime="$(control_value "${skip_ready_file}" runner_start_ticks 2>/dev/null || true)"
      fi
      if [[ "${current_runner_host_pid}" =~ ^[1-9][0-9]*$ \
            && "${current_runner_starttime}" =~ ^[1-9][0-9]*$ \
            && -f "${skip_ready_file}" ]] \
          && ready_file_is_owned "${skip_ready_file}"; then
        if [[ "${skip_was_accepted}" != "1" ]]; then
          publish_current_run \
            "${current_run_token}" "${index}" "${task}" "${seed}" \
            "${current_runner_host_pid}" "${current_runner_starttime}" ready
        fi
        if [[ "${skip_was_accepted}" != "1" \
              && -f "${skip_request_dir}/skip_requested.tsv" \
              && ! -f "${skip_accepted_file}" ]]; then
          if ! skip_request_is_current \
              "${skip_request_dir}/skip_requested.tsv"; then
            write_request_marker \
              "${skip_request_dir}/skip_rejected.tsv" rejected request_token_mismatch
          else
            write_request_marker "${skip_accepted_file}" accepted operator_skip_current
            publish_current_run \
              "${current_run_token}" "${index}" "${task}" "${seed}" \
              "${current_runner_host_pid}" "${current_runner_starttime}" skip_accepted
            skip_was_accepted=1
            if ! /bin/kill -INT "${current_runner_pid}" 2>/dev/null; then
              write_request_marker \
                "${skip_request_dir}/skip_rejected.tsv" rejected runner_exited_before_signal
            fi
          fi
        fi
      fi
      sleep 0.1
    done
    if wait "${current_runner_pid}"; then
      launcher_rc=0
    else
      launcher_rc=$?
    fi
    # wait has reaped the direct child.  Do not leave its local PID armed
    # while strict result checking runs: the kernel may reuse that PID, and a
    # later batch-level Ctrl+C must never signal an unrelated process.
    current_runner_pid=""
    publish_current_run \
      "${current_run_token}" "${index}" "${task}" "${seed}" \
      "${current_runner_host_pid}" "${current_runner_starttime}" finishing
    set -e

    strict_rc=99
    skip_requested=0
    if [[ "${skip_was_accepted}" == "1" \
          && "${launcher_rc}" =~ ^(130|143)$ \
          && -f "${skip_cleanup_complete_file}" \
          && ! -f "${skip_cleanup_failed_file}" ]] \
        && cleanup_marker_is_owned \
          "${skip_cleanup_complete_file}" cleanup_complete \
        && ready_workers_are_gone "${skip_ready_file}"; then
      skip_requested=1
    fi

    if [[ -f "${skip_request_dir}/skip_requested.tsv" \
          && "${skip_was_accepted}" != "1" \
          && ! -f "${skip_request_dir}/skip_rejected.tsv" ]]; then
      write_request_marker \
        "${skip_request_dir}/skip_rejected.tsv" rejected runner_exited_before_ready
    fi

    # Once a skip is accepted, cleanup evidence is a hard safety gate.  A
    # malformed/missing completion marker or a missed grace deadline must
    # stop this batch even when CONTINUE_ON_ERROR=1, because the old worker
    # may still own GPU or simulator resources.
    if [[ "${skip_was_accepted}" == "1" \
          && "${skip_requested}" != "1" ]]; then
      if [[ ! -f "${skip_cleanup_failed_file}" ]]; then
        write_request_marker \
          "${skip_cleanup_failed_file}" cleanup_failed \
          missing_valid_cleanup_completion
      fi
      echo "Accepted skip did not complete safely; refusing to start any later rollout." >&2
      cleanup_outcome="{\"reason\":\"operator_skip_cleanup_failed\",\"token\":\"${current_run_token}\"}"
      printf '%d\t%s\t%s\t%d\t%d\t%d\t%s\t%s\t%s\n' \
        "${index}" "${task}" "${seed_tag}" "${seed}" \
        "${launcher_rc}" 99 invalid "${result_dir}" "${cleanup_outcome}" \
        >> "${status_file}"
      publish_current_run \
        "${current_run_token}" "${index}" "${task}" "${seed}" \
        0 0 cleanup_failed
      publish_batch_state cleanup_failed
      exit 1
    fi

    if [[ "${skip_requested}" == "1" ]]; then
      strict_rc=-1
    elif [[ -d "${result_dir}" ]]; then
      set +e
      "${CHECKER_PYTHON}" "${CHECKER}" \
        --input "${result_dir}" \
        --strict \
        --json "${run_log_root}/strict_check.json" \
        > "${run_log_root}/strict_check.log" 2>&1
      strict_rc=$?
      set -e
      cat "${run_log_root}/strict_check.log"
    else
      echo "Expected result directory was not created: ${result_dir}" >&2
    fi

    outcome='{}'
    if [[ "${strict_rc}" == "0" ]]; then
      outcome="$("${CHECKER_PYTHON}" - "${result_dir}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
traces = sorted(
    root.rglob("episode_*_agent_trace.jsonl"),
    key=lambda path: path.stat().st_mtime_ns,
)
payload = {"trace": "", "success": None, "termination_reason": ""}
if traces:
    trace = traces[-1]
    payload["trace"] = str(trace)
    for raw in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "episode_end":
            payload.update(
                success=record.get("success"),
                task_finished=record.get("task_finished"),
                reward=record.get("reward"),
                termination_reason=record.get("termination_reason", record.get("reason", "")),
            )
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY
)"
    fi

    if [[ "${skip_requested}" == "1" ]]; then
      run_status=skipped
      skipped=$((skipped + 1))
      outcome="{\"reason\":\"operator_skip_current\",\"token\":\"${current_run_token}\"}"
      echo "[${index}/${total_runs}] SKIPPED task=${task} episode=${seed_tag}; continuing with the next rollout."
    elif [[ "${launcher_rc}" == "0" && "${strict_rc}" == "0" ]]; then
      run_status=complete
      echo "[${index}/${total_runs}] COMPLETE task=${task} episode=${seed_tag} outcome=${outcome}"
    else
      run_status=invalid
      failures=$((failures + 1))
      echo "[${index}/${total_runs}] INVALID task=${task} episode=${seed_tag} launcher_rc=${launcher_rc} strict_rc=${strict_rc}" >&2
    fi

    printf '%d\t%s\t%s\t%d\t%d\t%d\t%s\t%s\t%s\n' \
      "${index}" "${task}" "${seed_tag}" "${seed}" \
      "${launcher_rc}" "${strict_rc}" "${run_status}" "${result_dir}" "${outcome}" \
      >> "${status_file}"

    publish_current_run \
      "${current_run_token}" "${index}" "${task}" "${seed}" 0 0 "${run_status}"
    current_runner_pid=""

    echo "Finished: $(date --iso-8601=seconds)"
    if [[ "${run_status}" == "invalid" && "${CONTINUE_ON_ERROR}" != "1" ]]; then
      echo "Stopping after the first infrastructure/trace failure. Set CONTINUE_ON_ERROR=1 to continue instead." >&2
      echo "Status ledger: ${status_file}" >&2
      exit 1
    fi
done

publish_current_run '-' 0 '-' 0 0 0 finished
publish_batch_state finished

echo
echo "Batch finished at $(date --iso-8601=seconds)"
echo "Completed runs: $((total_runs - failures - skipped))/${total_runs}"
echo "Skipped runs: ${skipped}"
echo "Infrastructure/trace failures: ${failures}"
echo "Status ledger: ${status_file}"
echo "Batch log: ${BATCH_LOG_ROOT}/batch.log"

if (( failures > 0 )); then
  exit 1
fi
