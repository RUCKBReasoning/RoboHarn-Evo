#!/usr/bin/env bash
set -euo pipefail

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
ROBOHARN_EVO_OUTPUT_ROOT="${RMBENCH_OUTPUT_ROOT}"
FORMAL_PYTHONPATH="${ROBOHARN_EVO_PROJECT_ROOT}:${REPO_ROOT}"
PYTHONDONTWRITEBYTECODE=1
CONDA_SH="${CONDA_SH:-/path/to/conda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-RMBench}"

NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TASK_NAME="${TASK_NAME:-}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
INSTRUCTION_SET="${INSTRUCTION_SET:-rmbench_original}"
POLICY_NAME="${POLICY_NAME:-policy.roboharn_evo.deploy_policy}"
PERCEPTION_CONDITION="${PERCEPTION_CONDITION:-}"
BASE_CKPT="${BASE_CKPT:-gpt55_pure_tool_control_auto_8way}"
N_PER_WORKER="${N_PER_WORKER:-1}"
SEED_OFFSET="${SEED_OFFSET:-0}"
EVAL_START_SEEDS="${EVAL_START_SEEDS:-}"
REQUIRE_EXPLICIT_EVAL_START_SEEDS="${REQUIRE_EXPLICIT_EVAL_START_SEEDS:-1}"
NON_FORMAL_DIAGNOSTIC="${NON_FORMAL_DIAGNOSTIC:-0}"
FORMAL_PROTOCOL_VERSION=3

AGENT_API_BASE_URL="${AGENT_API_BASE_URL:-http://127.0.0.1:9104}"
SAM3_SERVICE_URL="${SAM3_SERVICE_URL:-http://127.0.0.1:9301}"
SAM3_BASE_PORT="${SAM3_BASE_PORT:-}"
REQUIRE_SAM3_PREFLIGHT="${REQUIRE_SAM3_PREFLIGHT:-1}"
PLANNER_TIMEOUT_SEC="${PLANNER_TIMEOUT_SEC:-1800}"
RECOVERY_TIMEOUT_SEC="${RECOVERY_TIMEOUT_SEC:-1800}"
OOD_TIMEOUT_SEC="${OOD_TIMEOUT_SEC:-300}"
QUERY_TIMEOUT_SEC="${QUERY_TIMEOUT_SEC:-900}"
PREFLIGHT_TIMEOUT_SEC="${PREFLIGHT_TIMEOUT_SEC:-10}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
SKIP_AGENT_IDENTITY_CHECK="${SKIP_AGENT_IDENTITY_CHECK:-0}"
REQUIRE_AGENT_INFERENCE_PREFLIGHT="${REQUIRE_AGENT_INFERENCE_PREFLIGHT:-1}"
AGENT_INFERENCE_PREFLIGHT_TIMEOUT_SEC="${AGENT_INFERENCE_PREFLIGHT_TIMEOUT_SEC:-650}"
EXPECTED_AGENT_MODEL="${EXPECTED_AGENT_MODEL-gpt-5.5}"
EXPECTED_AGENT_API_MODE="${EXPECTED_AGENT_API_MODE-responses_compat}"
EXPECTED_REASONING_EFFORT="${EXPECTED_REASONING_EFFORT-xhigh}"
EXPECTED_RESPONSE_STORAGE="${EXPECTED_RESPONSE_STORAGE-account_default}"
EXPECTED_AGENT_IDENTITY_JSON="${EXPECTED_AGENT_IDENTITY_JSON:-}"
MIN_AGENT_TIMEOUT_SEC="${MIN_AGENT_TIMEOUT_SEC:-600}"
EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS="${EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS-4}"
WORKER_START_DELAY_SEC="${WORKER_START_DELAY_SEC:-5}"
MAX_OBJECTS="${MAX_OBJECTS:-3}"
MAX_WAIT_STEPS="${MAX_WAIT_STEPS:-2}"
RETRY_BUDGET="${RETRY_BUDGET:-1}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
MAX_CONTROL_TURNS="${MAX_CONTROL_TURNS:-64}"
MAX_NO_PROGRESS_CONTROL_TURNS="${MAX_NO_PROGRESS_CONTROL_TURNS:-10}"
BACKEND_ERROR_BUDGET="${BACKEND_ERROR_BUDGET:-5}"
EMPTY_PLAN_REPLAN_THRESHOLD="${EMPTY_PLAN_REPLAN_THRESHOLD:-2}"

# Optional simulator renderer-device contract.  Keep empty by default so
# existing single-run launchers retain SAPIEN's historical auto-selection.
RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN="${RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN:-}"
RMBENCH_RENDER_DEVICE="${RMBENCH_RENDER_DEVICE:-}"
RMBENCH_RENDER_DEVICE_STRICT="${RMBENCH_RENDER_DEVICE_STRICT:-}"
RMBENCH_EXPECTED_RENDER_CUDA_ID="${RMBENCH_EXPECTED_RENDER_CUDA_ID:-}"
RMBENCH_EXPECTED_RENDER_PCI_BUS_ID="${RMBENCH_EXPECTED_RENDER_PCI_BUS_ID:-}"
RMBENCH_EXPECTED_PHYSICAL_GPU="${RMBENCH_EXPECTED_PHYSICAL_GPU:-}"
RMBENCH_RENDER_DEVICE_PROVENANCE_PATH="${RMBENCH_RENDER_DEVICE_PROVENANCE_PATH:-}"
RMBENCH_RAY_TRACING_DENOISER="${RMBENCH_RAY_TRACING_DENOISER:-oidn}"
RMBENCH_SLOW_ACTION_TRACE_SEC="${RMBENCH_SLOW_ACTION_TRACE_SEC:-0}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${RMBENCH_OUTPUT_ROOT}/formal_runs/rmbench_${BASE_CKPT}_${RUN_STAMP}}"
SEGMENTATION_ARTIFACT_ROOT="${SEGMENTATION_ARTIFACT_ROOT:-${LOG_ROOT}/segmentation_artifacts}"
RMBENCH_RUNTIME_CONFIG_ROOT="${RMBENCH_RUNTIME_CONFIG_ROOT:-${LOG_ROOT}/runtime_configs}"
ROBOHARN_EVO_WORKSPACE_ROOT="${ROBOHARN_EVO_WORKSPACE_ROOT:-${LOG_ROOT}/runtime_workspace}"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${LOG_ROOT}/runtime_caches}"
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
DETACH="${DETACH:-0}"
DETACH_SESSION="${DETACH_SESSION:-rmbench_${BASE_CKPT}_${RUN_STAMP}}"
RMBENCH_DETACHED_CHILD="${RMBENCH_DETACHED_CHILD:-0}"
SHUTDOWN_GRACE_SEC="${SHUTDOWN_GRACE_SEC:-90}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
INTERRUPT_ESCALATION_POLICY="${INTERRUPT_ESCALATION_POLICY:-term_then_kill}"
ROBOHARN_EVO_SKIP_READY_FILE="${ROBOHARN_EVO_SKIP_READY_FILE:-}"
ROBOHARN_EVO_SKIP_ACCEPTED_FILE="${ROBOHARN_EVO_SKIP_ACCEPTED_FILE:-}"
ROBOHARN_EVO_SKIP_CLEANUP_COMPLETE_FILE="${ROBOHARN_EVO_SKIP_CLEANUP_COMPLETE_FILE:-}"
ROBOHARN_EVO_SKIP_CLEANUP_FAILED_FILE="${ROBOHARN_EVO_SKIP_CLEANUP_FAILED_FILE:-}"
ROBOHARN_EVO_CONTROL_BATCH_TOKEN="${ROBOHARN_EVO_CONTROL_BATCH_TOKEN:-}"
ROBOHARN_EVO_CONTROL_RUN_TOKEN="${ROBOHARN_EVO_CONTROL_RUN_TOKEN:-}"
ROBOHARN_EVO_CONTROL_OWNER_PID="${ROBOHARN_EVO_CONTROL_OWNER_PID:-}"
ROBOHARN_EVO_CONTROL_OWNER_START_TICKS="${ROBOHARN_EVO_CONTROL_OWNER_START_TICKS:-}"
ROBOHARN_EVO_CONTROL_BOOT_ID="${ROBOHARN_EVO_CONTROL_BOOT_ID:-}"
DETACHED_LAUNCHER_LOG="${DETACHED_LAUNCHER_LOG:-${LOG_ROOT}/launcher.log}"
RECORD_RUNTIME_PROVENANCE="${RECORD_RUNTIME_PROVENANCE:-1}"
RUNTIME_PROVENANCE_PATH="${RUNTIME_PROVENANCE_PATH:-${LOG_ROOT}/runtime_provenance.json}"
AGENT_SERVICE_IDENTITY_PATH="${AGENT_SERVICE_IDENTITY_PATH:-${LOG_ROOT}/agent_service_identity.json}"
RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS="${RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS:-}"
EXPECTED_AGENT_SERVICE_IDENTITY_SHA256="${EXPECTED_AGENT_SERVICE_IDENTITY_SHA256:-}"
EXTRA_ARGS=("$@")

require_boolean_flag() {
  local name="$1"
  local value="$2"
  case "${value}" in
    0|1)
      ;;
    *)
      echo "${name} must be 0 or 1." >&2
      exit 2
      ;;
  esac
}

normalized_override_key() {
  local token="$1"
  while [[ "${token}" == -* || "${token}" == +* ]]; do
    token="${token:1}"
  done
  printf '%s' "${token%%=*}"
}

is_formal_protected_override_key() {
  local key="$1"
  case "${key}" in
    agent.observation_preprocess.oracle_objects.enabled|\
    agent.observation_preprocess.oracle_objects.include_all|\
    agent.observation_preprocess.oracle_objects.max_objects)
      return 1
      ;;
    config|ckpt_setting|instruction_set|instruction_type|task|task_name|task_config|seed|\
    eval|eval.start_seed|eval_start_seed|start_seed)
      return 0
      ;;
    eval.test_num|eval_test_num|test_num|policy|policy_name)
      return 0
      ;;
    planner|planner.*|ood|ood.*|recovery|recovery.*)
      return 0
      ;;
    agent|agent.*|agent.enabled|agent.pure_tool_control|agent.pure_tool_control.*|\
    agent.observation_preprocess|agent.observation_preprocess.*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

canonical_path() {
  realpath -m -- "$1"
}

path_is_within() {
  local candidate boundary
  candidate="$(canonical_path "$1")"
  boundary="$(canonical_path "$2")"
  [[ "${candidate}" == "${boundary}" || "${candidate}" == "${boundary}/"* ]]
}

require_boolean_flag "NON_FORMAL_DIAGNOSTIC" "${NON_FORMAL_DIAGNOSTIC}"
require_boolean_flag "REQUIRE_EXPLICIT_EVAL_START_SEEDS" "${REQUIRE_EXPLICIT_EVAL_START_SEEDS}"
require_boolean_flag "SKIP_PREFLIGHT" "${SKIP_PREFLIGHT}"
require_boolean_flag "PREFLIGHT_ONLY" "${PREFLIGHT_ONLY}"
require_boolean_flag "SKIP_AGENT_IDENTITY_CHECK" "${SKIP_AGENT_IDENTITY_CHECK}"
require_boolean_flag "REQUIRE_AGENT_INFERENCE_PREFLIGHT" "${REQUIRE_AGENT_INFERENCE_PREFLIGHT}"
require_boolean_flag "RECORD_RUNTIME_PROVENANCE" "${RECORD_RUNTIME_PROVENANCE}"
require_boolean_flag "REQUIRE_SAM3_PREFLIGHT" "${REQUIRE_SAM3_PREFLIGHT}"
case "${INTERRUPT_ESCALATION_POLICY}" in
  term_then_kill|ctrl_c_only)
    ;;
  *)
    echo "INTERRUPT_ESCALATION_POLICY must be term_then_kill or ctrl_c_only." >&2
    exit 2
    ;;
esac

identity_field_names=(
  EXPECTED_AGENT_MODEL
  EXPECTED_AGENT_API_MODE
  EXPECTED_RESPONSE_STORAGE
  EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS
)
if [[ "${EXPECTED_AGENT_API_MODE}" != "chat" ]]; then
  identity_field_names+=(EXPECTED_REASONING_EFFORT)
fi
for identity_field_name in "${identity_field_names[@]}"; do
  identity_field_value="${!identity_field_name}"
  if [[ -z "${identity_field_value//[[:space:]]/}" ]]; then
    echo "${identity_field_name} must be non-empty." >&2
    exit 2
  fi
done
if [[ ! "${EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS must be a positive integer." >&2
  exit 2
fi
if [[ "${SEGMENTATION_ARTIFACT_ROOT}" != /* ]]; then
  echo "SEGMENTATION_ARTIFACT_ROOT must be an absolute path: ${SEGMENTATION_ARTIFACT_ROOT}" >&2
  exit 2
fi
if [[ -n "${EXPECTED_AGENT_SERVICE_IDENTITY_SHA256}" && ! "${EXPECTED_AGENT_SERVICE_IDENTITY_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "EXPECTED_AGENT_SERVICE_IDENTITY_SHA256 must be empty or a lowercase SHA256." >&2
  exit 2
fi
if [[ ! "${MAX_CONTROL_TURNS}" =~ ^[0-9]+$ || ! "${MAX_NO_PROGRESS_CONTROL_TURNS}" =~ ^[0-9]+$ ]]; then
  echo "MAX_CONTROL_TURNS and MAX_NO_PROGRESS_CONTROL_TURNS must be non-negative integers." >&2
  exit 2
fi

FORMAL_PERCEPTION_OVERRIDE_ARGS=()
if [[ "${NON_FORMAL_DIAGNOSTIC}" == "0" ]]; then
  ROBOHARN_EVO_FORMAL_PROTOCOL=1
  ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION="${FORMAL_PROTOCOL_VERSION}"
  if [[ "${POLICY_NAME}" != "policy.roboharn_evo.deploy_policy" ]]; then
    echo "Formal protocol requires POLICY_NAME=policy.roboharn_evo.deploy_policy." >&2
    exit 2
  fi
  if [[ "${MAX_ROUNDS}" != "10" || "${MAX_CONTROL_TURNS}" != "64" || "${MAX_NO_PROGRESS_CONTROL_TURNS}" != "10" ]]; then
    echo "Formal protocol requires MAX_ROUNDS=10, MAX_CONTROL_TURNS=64, and MAX_NO_PROGRESS_CONTROL_TURNS=10; set NON_FORMAL_DIAGNOSTIC=1 only for tests or diagnostics." >&2
    exit 2
  fi
  if [[ "${REQUIRE_EXPLICIT_EVAL_START_SEEDS}" != "1" ]]; then
    echo "Formal protocol requires REQUIRE_EXPLICIT_EVAL_START_SEEDS=1." >&2
    exit 2
  fi
  if [[ "${SKIP_PREFLIGHT}" != "0" ]]; then
    echo "Formal protocol forbids SKIP_PREFLIGHT=1; set NON_FORMAL_DIAGNOSTIC=1 only for tests or diagnostics." >&2
    exit 2
  fi
  if [[ "${SKIP_AGENT_IDENTITY_CHECK}" != "0" ]]; then
    echo "Formal protocol forbids SKIP_AGENT_IDENTITY_CHECK=1; set NON_FORMAL_DIAGNOSTIC=1 only for tests or diagnostics." >&2
    exit 2
  fi
  if [[ "${REQUIRE_AGENT_INFERENCE_PREFLIGHT}" != "1" ]]; then
    echo "Formal protocol requires REQUIRE_AGENT_INFERENCE_PREFLIGHT=1; set NON_FORMAL_DIAGNOSTIC=1 only for tests or diagnostics." >&2
    exit 2
  fi
  if [[ "${RECORD_RUNTIME_PROVENANCE}" != "1" ]]; then
    echo "Formal protocol requires RECORD_RUNTIME_PROVENANCE=1; set NON_FORMAL_DIAGNOSTIC=1 only for tests or diagnostics." >&2
    exit 2
  fi
  case "${PERCEPTION_CONDITION}" in
    no_oracle)
      if [[ "${REQUIRE_SAM3_PREFLIGHT}" != "1" ]]; then
        echo "Formal no_oracle protocol requires REQUIRE_SAM3_PREFLIGHT=1." >&2
        exit 2
      fi
      FORMAL_PERCEPTION_OVERRIDE_ARGS=(
        --agent.observation_preprocess.oracle_objects.enabled False
      )
      ;;
    oracle)
      if [[ "${REQUIRE_SAM3_PREFLIGHT}" != "0" ]]; then
        echo "Formal oracle protocol requires REQUIRE_SAM3_PREFLIGHT=0." >&2
        exit 2
      fi
      FORMAL_PERCEPTION_OVERRIDE_ARGS=(
        --agent.observation_preprocess.oracle_objects.enabled True
      )
      ;;
    *)
      echo "Formal protocol requires PERCEPTION_CONDITION=oracle or no_oracle." >&2
      exit 2
      ;;
  esac
  if (( ${#EXTRA_ARGS[@]} % 2 != 0 )); then
    echo "Formal protocol requires key-value EXTRA_ARGS pairs." >&2
    exit 2
  fi
  for (( extra_index = 0; extra_index < ${#EXTRA_ARGS[@]}; extra_index += 2 )); do
    arg="${EXTRA_ARGS[extra_index]}"
    override_value="${EXTRA_ARGS[extra_index + 1]}"
    override_key="$(normalized_override_key "${arg}")"
    case "${override_key}" in
      eval.exact_seed_fail_closed)
        if [[ "${override_value}" != "True" ]]; then
          echo "Formal protocol requires eval.exact_seed_fail_closed=True." >&2
          exit 2
        fi
        ;;
      eval.step_limit)
        if [[ "${override_value}" != "150" ]]; then
          echo "Formal Q1 requires eval.step_limit=150." >&2
          exit 2
        fi
        ;;
      *)
        if is_formal_protected_override_key "${override_key}"; then
          echo "Formal protocol forbids EXTRA_ARGS override of ${override_key}: ${arg}" >&2
          exit 2
        fi
        ;;
    esac
  done
  echo "[protocol-mode] FORMAL perception_condition=${PERCEPTION_CONDITION} protocol_version=${FORMAL_PROTOCOL_VERSION} policy=${POLICY_NAME} max_rounds=10 max_control_turns=64 max_no_progress_control_turns=10 identity_check=required inference_preflight=required runtime_provenance=required"
else
  ROBOHARN_EVO_FORMAL_PROTOCOL=0
  ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION=0
  echo "[protocol-mode] NON-FORMAL DIAGNOSTIC protocol_version=0 opt_out=NON_FORMAL_DIAGNOSTIC=1 perception_condition=${PERCEPTION_CONDITION:-unspecified} policy=${POLICY_NAME} max_rounds=${MAX_ROUNDS} max_control_turns=${MAX_CONTROL_TURNS} max_no_progress_control_turns=${MAX_NO_PROGRESS_CONTROL_TURNS} skip_preflight=${SKIP_PREFLIGHT} inference_preflight=${REQUIRE_AGENT_INFERENCE_PREFLIGHT} runtime_provenance=${RECORD_RUNTIME_PROVENANCE}"
fi

ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH=""
if [[ "${RECORD_RUNTIME_PROVENANCE}" == "1" ]]; then
  ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH="${RUNTIME_PROVENANCE_PATH}"
fi

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
if [[ "${NON_FORMAL_DIAGNOSTIC}" == "0" ]]; then
  formal_runtime_path_names=(
    RMBENCH_OUTPUT_ROOT LOG_ROOT SEGMENTATION_ARTIFACT_ROOT
    RMBENCH_RUNTIME_CONFIG_ROOT ROBOHARN_EVO_WORKSPACE_ROOT RUNTIME_CACHE_ROOT
    XDG_CACHE_HOME CUDA_CACHE_PATH MESA_SHADER_CACHE_DIR TORCH_HOME HF_HOME
    TRITON_CACHE_DIR NUMBA_CACHE_DIR MPLCONFIGDIR TMPDIR TMP TEMP
    DETACHED_LAUNCHER_LOG RUNTIME_PROVENANCE_PATH AGENT_SERVICE_IDENTITY_PATH
  )
  for formal_runtime_path_name in "${formal_runtime_path_names[@]}"; do
    formal_runtime_path_value="${!formal_runtime_path_name}"
    if ! path_is_within "${formal_runtime_path_value}" "${allowed_runtime_root}"; then
      echo "${formal_runtime_path_name} must stay under ${allowed_runtime_root}: ${formal_runtime_path_value}" >&2
      exit 2
    fi
  done
  if [[ -n "${RMBENCH_RENDER_DEVICE_PROVENANCE_PATH}" ]] \
      && ! path_is_within "${RMBENCH_RENDER_DEVICE_PROVENANCE_PATH}" "${allowed_runtime_root}"; then
    echo "RMBENCH_RENDER_DEVICE_PROVENANCE_PATH must stay under ${allowed_runtime_root}" >&2
    exit 2
  fi
  if path_is_within "${RMBENCH_OUTPUT_ROOT}" "${REPO_ROOT}" \
      || path_is_within "${REPO_ROOT}" "${RMBENCH_OUTPUT_ROOT}"; then
    echo "RMBENCH_OUTPUT_ROOT must not overlap copied benchmark source ${REPO_ROOT}" >&2
    exit 2
  fi
  if [[ -z "${RMBENCH_ASSETS_ROOT//[[:space:]]/}" ]]; then
    echo "RMBENCH_ASSETS_ROOT must explicitly name read-only licensed assets for a formal rollout; there is no donor fallback." >&2
    exit 2
  fi
  if path_is_within "${RMBENCH_OUTPUT_ROOT}" "${RMBENCH_ASSETS_ROOT}" \
      || path_is_within "${RMBENCH_ASSETS_ROOT}" "${RMBENCH_OUTPUT_ROOT}"; then
    echo "RMBENCH_OUTPUT_ROOT must not overlap RMBENCH_ASSETS_ROOT" >&2
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
fi

export \
  PYTHONPATH="${FORMAL_PYTHONPATH}" \
  PYTHONDONTWRITEBYTECODE \
  RMBENCH_ROOT RMBENCH_ASSETS_ROOT RMBENCH_OUTPUT_ROOT ROBOHARN_EVO_OUTPUT_ROOT \
  RMBENCH_RUNTIME_CONFIG_ROOT ROBOHARN_EVO_WORKSPACE_ROOT \
  XDG_CACHE_HOME CUDA_CACHE_PATH MESA_SHADER_CACHE_DIR TORCH_HOME HF_HOME \
  TRITON_CACHE_DIR NUMBA_CACHE_DIR MPLCONFIGDIR TMPDIR TMP TEMP

if [[ -z "${TASK_NAME//[[:space:]]/}" ]]; then
  echo "TASK_NAME must be supplied explicitly; the formal launcher has no task-specific default." >&2
  exit 2
fi
if [[ ! "${INSTRUCTION_SET}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "INSTRUCTION_SET must be a non-empty catalog identifier containing only letters, digits, '_', '-', or '.'." >&2
  exit 2
fi
case "${REQUIRE_EXPLICIT_EVAL_START_SEEDS}" in
  1)
    if [[ -z "${EVAL_START_SEEDS}" ]]; then
      echo "EVAL_START_SEEDS must be supplied explicitly for formal runs." >&2
      exit 2
    fi
    ;;
  0)
    ;;
  *)
    echo "REQUIRE_EXPLICIT_EVAL_START_SEEDS must be 0 or 1." >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} < NUM_WORKERS )); then
  echo "GPU_IDS has ${#GPU_ARRAY[@]} entries, but NUM_WORKERS=${NUM_WORKERS}" >&2
  exit 2
fi
if [[ -n "${RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN}" ]]; then
  if (( NUM_WORKERS != 1 )); then
    echo "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN requires NUM_WORKERS=1." >&2
    exit 2
  fi
  if [[ "${RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN}" =~ [[:space:],] ]]; then
    echo "RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN must contain exactly one CUDA device token." >&2
    exit 2
  fi
fi

EVAL_START_SEED_ARRAY=()
if [[ -n "${EVAL_START_SEEDS}" ]]; then
  IFS=',' read -r -a EVAL_START_SEED_ARRAY <<< "${EVAL_START_SEEDS}"
  if (( ${#EVAL_START_SEED_ARRAY[@]} != NUM_WORKERS )); then
    echo "EVAL_START_SEEDS has ${#EVAL_START_SEED_ARRAY[@]} entries, but NUM_WORKERS=${NUM_WORKERS}" >&2
    exit 2
  fi
  if (( N_PER_WORKER != 1 )); then
    echo "EVAL_START_SEEDS requires N_PER_WORKER=1 to prevent overlapping seed ranges." >&2
    exit 2
  fi
  declare -A seen_eval_start_seeds=()
  for eval_start_seed_index in "${!EVAL_START_SEED_ARRAY[@]}"; do
    eval_start_seed="${EVAL_START_SEED_ARRAY[eval_start_seed_index]}"
    if [[ ! "${eval_start_seed}" =~ ^[0-9]+$ ]]; then
      echo "EVAL_START_SEEDS entries must be non-negative canonicalizable integers: ${eval_start_seed}" >&2
      exit 2
    fi
    canonical_eval_start_seed="${eval_start_seed}"
    while [[ "${#canonical_eval_start_seed}" -gt 1 && "${canonical_eval_start_seed:0:1}" == "0" ]]; do
      canonical_eval_start_seed="${canonical_eval_start_seed:1}"
    done
    if [[ -n "${seen_eval_start_seeds[${canonical_eval_start_seed}]:-}" ]]; then
      echo "EVAL_START_SEEDS contains a duplicate canonical seed: ${eval_start_seed} conflicts with ${seen_eval_start_seeds[${canonical_eval_start_seed}]} (canonical ${canonical_eval_start_seed})" >&2
      exit 2
    fi
    seen_eval_start_seeds["${canonical_eval_start_seed}"]="${eval_start_seed}"
    EVAL_START_SEED_ARRAY[eval_start_seed_index]="${canonical_eval_start_seed}"
  done
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Cannot find conda setup script: ${CONDA_SH}" >&2
  exit 2
fi

mkdir -p \
  "${LOG_ROOT}" \
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

if [[ "${RMBENCH_DETACHED_CHILD}" == "1" ]]; then
  exec >>"${DETACHED_LAUNCHER_LOG}" 2>&1
  echo "Detached runner started at $(date --iso-8601=seconds)"
fi

if [[ "${DETACH}" == "1" && "${RMBENCH_DETACHED_CHILD}" != "1" ]]; then
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

  runner_path="${BASH_SOURCE[0]}"
  if [[ "${runner_path}" != /* ]]; then
    runner_path="${REPO_ROOT}/${runner_path#./}"
  fi
  detached_command=(env "RMBENCH_DETACHED_CHILD=1" "DETACH=0")
  detached_env_names=(
    REPO_ROOT ROBOHARN_EVO_PROJECT_ROOT PYTHONPATH PYTHONDONTWRITEBYTECODE
    RMBENCH_ROOT RMBENCH_ASSETS_ROOT RMBENCH_OUTPUT_ROOT ROBOHARN_EVO_OUTPUT_ROOT
    RMBENCH_RUNTIME_CONFIG_ROOT ROBOHARN_EVO_WORKSPACE_ROOT RUNTIME_CACHE_ROOT
    XDG_CACHE_HOME CUDA_CACHE_PATH MESA_SHADER_CACHE_DIR TORCH_HOME HF_HOME
    TRITON_CACHE_DIR NUMBA_CACHE_DIR MPLCONFIGDIR TMPDIR TMP TEMP
    CONDA_SH CONDA_ENV NUM_WORKERS GPU_IDS TASK_NAME TASK_CONFIG INSTRUCTION_SET POLICY_NAME
    PERCEPTION_CONDITION
    BASE_CKPT N_PER_WORKER SEED_OFFSET EVAL_START_SEEDS REQUIRE_EXPLICIT_EVAL_START_SEEDS
    NON_FORMAL_DIAGNOSTIC ROBOHARN_EVO_FORMAL_PROTOCOL ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH
    AGENT_API_BASE_URL SAM3_SERVICE_URL SAM3_BASE_PORT
    REQUIRE_SAM3_PREFLIGHT
    PLANNER_TIMEOUT_SEC RECOVERY_TIMEOUT_SEC OOD_TIMEOUT_SEC QUERY_TIMEOUT_SEC
    PREFLIGHT_TIMEOUT_SEC SKIP_PREFLIGHT PREFLIGHT_ONLY SKIP_AGENT_IDENTITY_CHECK
    REQUIRE_AGENT_INFERENCE_PREFLIGHT AGENT_INFERENCE_PREFLIGHT_TIMEOUT_SEC
    EXPECTED_AGENT_MODEL EXPECTED_AGENT_API_MODE EXPECTED_REASONING_EFFORT
    EXPECTED_RESPONSE_STORAGE EXPECTED_AGENT_IDENTITY_JSON MIN_AGENT_TIMEOUT_SEC
    EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS AGENT_SERVICE_IDENTITY_PATH
    EXPECTED_AGENT_SERVICE_IDENTITY_SHA256 RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS
    WORKER_START_DELAY_SEC MAX_OBJECTS MAX_WAIT_STEPS RETRY_BUDGET MAX_ROUNDS MAX_CONTROL_TURNS
    MAX_NO_PROGRESS_CONTROL_TURNS
    BACKEND_ERROR_BUDGET EMPTY_PLAN_REPLAN_THRESHOLD
    RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN
    RMBENCH_RENDER_DEVICE RMBENCH_RENDER_DEVICE_STRICT
    RMBENCH_EXPECTED_RENDER_CUDA_ID RMBENCH_EXPECTED_RENDER_PCI_BUS_ID
    RMBENCH_EXPECTED_PHYSICAL_GPU RMBENCH_RENDER_DEVICE_PROVENANCE_PATH
    RMBENCH_RAY_TRACING_DENOISER RMBENCH_SLOW_ACTION_TRACE_SEC
    RUN_STAMP LOG_ROOT SEGMENTATION_ARTIFACT_ROOT SHUTDOWN_GRACE_SEC HEARTBEAT_SEC INTERRUPT_ESCALATION_POLICY
    DETACHED_LAUNCHER_LOG
    RECORD_RUNTIME_PROVENANCE RUNTIME_PROVENANCE_PATH
  )
  if [[ -n "${CUDA_DEVICE_ORDER:-}" ]]; then
    detached_env_names+=(CUDA_DEVICE_ORDER)
  fi
  for name in "${detached_env_names[@]}"; do
    detached_command+=("${name}=${!name}")
  done
  detached_command+=("${runner_path}" "${EXTRA_ARGS[@]}")
  printf -v detached_command_string '%q ' "${detached_command[@]}"

  tmux new-session -d -s "${DETACH_SESSION}" -c "${REPO_ROOT}" "${detached_command_string}"
  echo "Detached rollout session started: ${DETACH_SESSION}"
  echo "Attach: tmux attach -t ${DETACH_SESSION}"
  echo "Launcher log: ${DETACHED_LAUNCHER_LOG}"
  echo "Worker logs: ${LOG_ROOT}"
  exit 0
fi

source "${CONDA_SH}"
set +u
conda activate "${CONDA_ENV}"
set -u
cd "${REPO_ROOT}"

check_health() {
  local name="$1"
  local url="$2"
  if ! curl --noproxy '*' --fail --silent --show-error --max-time "${PREFLIGHT_TIMEOUT_SEC}" "${url%/}/health" >/dev/null; then
    echo "Preflight failed: ${name} is not healthy at ${url%/}/health" >&2
    return 1
  fi
}

check_agent_health() {
  local url="$1"
  local health
  if ! health="$(curl --noproxy '*' --fail --silent --show-error --max-time "${PREFLIGHT_TIMEOUT_SEC}" "${url%/}/health")"; then
    echo "Preflight failed: agent API is not healthy at ${url%/}/health" >&2
    return 1
  fi
  echo "Agent API health: ${health}"
  AGENT_HEALTH_JSON="${health}" \
  AGENT_SERVICE_IDENTITY_PATH="${AGENT_SERVICE_IDENTITY_PATH}" \
  EXPECTED_AGENT_SERVICE_IDENTITY_SHA256="${EXPECTED_AGENT_SERVICE_IDENTITY_SHA256}" \
  python -c '
import hashlib, json, os
payload = json.loads(os.environ["AGENT_HEALTH_JSON"])
path = os.environ["AGENT_SERVICE_IDENTITY_PATH"]
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
expected_sha256 = os.environ.get("EXPECTED_AGENT_SERVICE_IDENTITY_SHA256", "")
if expected_sha256:
    actual_sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            "Preflight failed: current agent service identity SHA256 changed: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
'
  if [[ "${SKIP_AGENT_IDENTITY_CHECK}" == "1" ]]; then
    return 0
  fi
  AGENT_HEALTH_JSON="${health}" \
  EXPECTED_AGENT_MODEL="${EXPECTED_AGENT_MODEL}" \
  EXPECTED_AGENT_API_MODE="${EXPECTED_AGENT_API_MODE}" \
  EXPECTED_REASONING_EFFORT="${EXPECTED_REASONING_EFFORT}" \
  EXPECTED_RESPONSE_STORAGE="${EXPECTED_RESPONSE_STORAGE}" \
  EXPECTED_AGENT_IDENTITY_JSON="${EXPECTED_AGENT_IDENTITY_JSON}" \
  MIN_AGENT_TIMEOUT_SEC="${MIN_AGENT_TIMEOUT_SEC}" \
  EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS="${EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS}" \
  ROBOHARN_EVO_FORMAL_PROTOCOL="${ROBOHARN_EVO_FORMAL_PROTOCOL}" \
  python -c '
import json, os, sys
payload = json.loads(os.environ["AGENT_HEALTH_JSON"])
if not isinstance(payload, dict):
    print("Preflight failed: agent API health payload must be a JSON object.", file=sys.stderr)
    raise SystemExit(1)
expected = {
    "model": os.environ["EXPECTED_AGENT_MODEL"],
}
optional_expected = {
    "api_mode": os.environ["EXPECTED_AGENT_API_MODE"],
    "reasoning_effort": os.environ["EXPECTED_REASONING_EFFORT"],
    "response_storage": os.environ["EXPECTED_RESPONSE_STORAGE"],
}
expected.update({key: value for key, value in optional_expected.items() if value})
errors = [f"{key}={payload.get(key)!r}, expected {value!r}" for key, value in expected.items() if payload.get(key) != value]
extra_expected_text = os.environ.get("EXPECTED_AGENT_IDENTITY_JSON", "").strip()
if extra_expected_text:
    try:
        extra_expected = json.loads(extra_expected_text)
    except json.JSONDecodeError as exc:
        print(f"Preflight failed: EXPECTED_AGENT_IDENTITY_JSON is invalid JSON: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if not isinstance(extra_expected, dict):
        print("Preflight failed: EXPECTED_AGENT_IDENTITY_JSON must be a JSON object.", file=sys.stderr)
        raise SystemExit(1)

    def compare_subset(actual, expected_value, path):
        if isinstance(expected_value, dict):
            if not isinstance(actual, dict):
                errors.append(f"{path}={actual!r}, expected object")
                return
            for child_key, child_value in expected_value.items():
                compare_subset(actual.get(child_key), child_value, f"{path}.{child_key}")
            return
        if actual != expected_value:
            errors.append(f"{path}={actual!r}, expected {expected_value!r}")

    for extra_key, extra_value in extra_expected.items():
        compare_subset(payload.get(extra_key), extra_value, extra_key)
if os.environ["ROBOHARN_EVO_FORMAL_PROTOCOL"] == "1":
    health_status = str(payload.get("status", "")).strip().lower()
    if health_status not in {"ok", "healthy"}:
        errors.append("status=%r, expected ok or healthy" % (payload.get("status"),))
    if (
        payload.get("backend") != "openai"
        and "app_server_running" in payload
        and payload.get("app_server_running") is not True
    ):
        errors.append(
            "app_server_running=%r, expected true when present"
            % (payload.get("app_server_running"),)
        )
minimum_timeout = int(os.environ["MIN_AGENT_TIMEOUT_SEC"])
if minimum_timeout > 0:
    try:
        timeout_sec = int(payload.get("timeout_sec"))
    except (TypeError, ValueError):
        errors.append("timeout_sec missing or invalid")
    else:
        if timeout_sec < minimum_timeout:
            errors.append(f"timeout_sec={timeout_sec}, expected >= {minimum_timeout}")
expected_concurrency = int(os.environ["EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS"])
if expected_concurrency > 0:
    try:
        max_concurrency = int(payload.get("max_concurrent_requests"))
    except (TypeError, ValueError):
        errors.append("max_concurrent_requests missing or invalid")
    else:
        if max_concurrency != expected_concurrency:
            errors.append(f"max_concurrent_requests={max_concurrency}, expected {expected_concurrency}")
if errors:
    print("Preflight failed: wrong agent API identity: " + "; ".join(errors), file=sys.stderr)
    raise SystemExit(1)
'
}

check_agent_inference() {
  local url="$1"
  local response
  local query_count
  local request_started_ns
  local request_finished_ns
  local latency_ms
  local curl_status

  if [[ ! "${AGENT_INFERENCE_PREFLIGHT_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "AGENT_INFERENCE_PREFLIGHT_TIMEOUT_SEC must be a positive integer." >&2
    return 2
  fi
  request_started_ns="$(date +%s%N)"
  if response="$(
    curl --noproxy '*' \
      --fail \
      --silent \
      --show-error \
      --max-time "${AGENT_INFERENCE_PREFLIGHT_TIMEOUT_SEC}" \
      -H 'Content-Type: application/json' \
      --data '{
        "global_task": "Select one generic manipulable item from the synthetic observation.",
        "current_subtask": "",
        "committed_memory": "",
        "observation_summary": "A synthetic scene contains one generic manipulable item.",
        "robot_state": {},
        "active_skill": "",
        "max_queries": 1,
        "camera": "",
        "cameras": []
      }' \
      "${url%/}/perception_queries"
  )"; then
    curl_status=0
  else
    curl_status=$?
  fi
  request_finished_ns="$(date +%s%N)"
  latency_ms=$(( (request_finished_ns - request_started_ns) / 1000000 ))
  if (( curl_status != 0 )); then
    echo "[agent-inference-preflight-failed] timestamp=$(date --iso-8601=seconds) endpoint=${url%/}/perception_queries outer_calls=1 latency_ms=${latency_ms} stage=request curl_exit=${curl_status}" >&2
    echo "Preflight failed: agent API could not complete a real structured inference at ${url%/}/perception_queries" >&2
    return 1
  fi

  if ! query_count="$(
    AGENT_INFERENCE_JSON="${response}" python -c '
import json
import os
import sys

try:
    payload = json.loads(os.environ["AGENT_INFERENCE_JSON"])
except (KeyError, json.JSONDecodeError) as exc:
    print(f"Preflight failed: agent inference returned invalid JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)

queries = payload.get("queries") if isinstance(payload, dict) else None
if not isinstance(queries, list) or not queries:
    print("Preflight failed: agent inference did not return a non-empty queries array.", file=sys.stderr)
    raise SystemExit(1)
if not any(
    isinstance(item, dict)
    and str(item.get("object_id", "")).strip()
    and str(item.get("text_prompt", "")).strip()
    for item in queries
):
    print("Preflight failed: agent inference returned no structured object query.", file=sys.stderr)
    raise SystemExit(1)
print(len(queries))
'
  )"; then
    echo "[agent-inference-preflight-failed] timestamp=$(date --iso-8601=seconds) endpoint=${url%/}/perception_queries outer_calls=1 latency_ms=${latency_ms} stage=response_validation" >&2
    return 1
  fi

  echo "[agent-inference-preflight] timestamp=$(date --iso-8601=seconds) endpoint=${url%/}/perception_queries outer_calls=1 latency_ms=${latency_ms} queries=${query_count}"
}

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  check_agent_health "${AGENT_API_BASE_URL}"
  case "${REQUIRE_AGENT_INFERENCE_PREFLIGHT}" in
    1)
      check_agent_inference "${AGENT_API_BASE_URL}"
      ;;
    0)
      echo "Agent inference preflight explicitly disabled."
      ;;
    *)
      echo "REQUIRE_AGENT_INFERENCE_PREFLIGHT must be 0 or 1." >&2
      exit 2
      ;;
  esac
  if [[ "${REQUIRE_SAM3_PREFLIGHT}" == "1" ]]; then
    if [[ -n "${SAM3_BASE_PORT}" ]]; then
      for (( worker = 0; worker < NUM_WORKERS; worker++ )); do
        check_health "SAM3 worker ${worker}" "http://127.0.0.1:$((SAM3_BASE_PORT + worker))"
      done
    else
      check_health "SAM3" "${SAM3_SERVICE_URL}"
    fi
  else
    echo "SAM3 preflight skipped for this condition."
  fi
fi

case "${RECORD_RUNTIME_PROVENANCE}" in
  1)
    runtime_provenance_external_args=()
    if [[ -n "${RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS}" ]]; then
      while IFS= read -r external_evidence_spec; do
        if [[ -z "${external_evidence_spec}" ]]; then
          echo "RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS must not contain blank lines." >&2
          exit 2
        fi
        runtime_provenance_external_args+=(--external-evidence "${external_evidence_spec}")
      done <<<"${RUNTIME_PROVENANCE_EXTERNAL_EVIDENCE_SPECS}"
    fi
    python "${REPO_ROOT}/policy/roboharn_evo/scripts/record_runtime_provenance.py" \
      --repo-root "${ROBOHARN_EVO_PROJECT_ROOT}" \
      --output "${RUNTIME_PROVENANCE_PATH}" \
      --service-identity "${AGENT_SERVICE_IDENTITY_PATH}" \
      "${runtime_provenance_external_args[@]}"
    ;;
  0)
    echo "Runtime provenance recording explicitly disabled."
    ;;
  *)
    echo "RECORD_RUNTIME_PROVENANCE must be 0 or 1." >&2
    exit 2
    ;;
esac

echo "[agent-plan] base_url=${AGENT_API_BASE_URL} api_mode=${EXPECTED_AGENT_API_MODE:-unchecked} response_storage=${EXPECTED_RESPONSE_STORAGE:-unchecked} max_concurrent_requests=${EXPECTED_AGENT_MAX_CONCURRENT_REQUESTS}"
echo "[reasoning-plan] reasoning_effort=${EXPECTED_REASONING_EFFORT:-not_applicable}"
echo "[budget-plan] max_rounds=${MAX_ROUNDS} max_control_turns=${MAX_CONTROL_TURNS} max_no_progress_control_turns=${MAX_NO_PROGRESS_CONTROL_TURNS}"
echo "[interrupt-plan] escalation_policy=${INTERRUPT_ESCALATION_POLICY} shutdown_grace_sec=${SHUTDOWN_GRACE_SEC}"
echo "[instruction-plan] instruction_set=${INSTRUCTION_SET} instruction_type=unseen task=${TASK_NAME}"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  if [[ -n "${EVAL_START_SEEDS}" ]]; then
    for (( worker = 0; worker < NUM_WORKERS; worker++ )); do
      echo "[seed-plan worker ${worker}] gpu=${GPU_ARRAY[worker]} eval_start_seed=${EVAL_START_SEED_ARRAY[worker]}"
    done
  fi
  echo "Preflight complete; no eval workers launched."
  exit 0
fi

pids=()
worker_names=()
heartbeat_pid=""

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
        && grep -zq -- "${BASH_SOURCE[0]}" \
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

write_skip_marker() {
  local marker_path="$1"
  local status="$2"
  local reason="$3"
  local marker_dir marker_tmp

  [[ -n "${marker_path}" ]] || return 0
  marker_dir="$(dirname "${marker_path}")"
  mkdir -p "${marker_dir}"
  marker_tmp="${marker_path}.tmp.$$"
  {
    printf 'schema_version\t1\n'
    printf 'batch_token\t%s\n' "${ROBOHARN_EVO_CONTROL_BATCH_TOKEN}"
    printf 'run_token\t%s\n' "${ROBOHARN_EVO_CONTROL_RUN_TOKEN}"
    printf 'runner_pid\t%s\n' "${runner_host_pid:-$$}"
    printf 'status\t%s\n' "${status}"
    printf 'reason\t%s\n' "${reason}"
    printf 'timestamp\t%s\n' "$(date --iso-8601=seconds)"
  } > "${marker_tmp}"
  mv -f "${marker_tmp}" "${marker_path}"
}

publish_skip_ready() {
  local runner_state runner_ppid runner_start_ticks
  local worker_pid worker_host_pid worker_state worker_ppid worker_start_ticks
  local ready_dir ready_tmp

  [[ -n "${ROBOHARN_EVO_SKIP_READY_FILE}" ]] || return 0
  if [[ -z "${ROBOHARN_EVO_CONTROL_BATCH_TOKEN}" \
        || -z "${ROBOHARN_EVO_CONTROL_RUN_TOKEN}" \
        || -z "${ROBOHARN_EVO_CONTROL_OWNER_PID}" \
        || -z "${ROBOHARN_EVO_CONTROL_OWNER_START_TICKS}" \
        || -z "${ROBOHARN_EVO_CONTROL_BOOT_ID}" ]]; then
    echo "Skip control metadata is incomplete; refusing to publish readiness." >&2
    return 1
  fi
  if ! local_pid_to_host_pid "$$" runner_host_pid \
      || ! read_proc_identity \
        "${runner_host_pid}" runner_state runner_ppid runner_start_ticks; then
    echo "Could not read runner process identity for skip control." >&2
    return 1
  fi
  if [[ "${runner_ppid}" != "${ROBOHARN_EVO_CONTROL_OWNER_PID}" ]]; then
    echo "Skip control owner mismatch: runner_ppid=${runner_ppid}, expected=${ROBOHARN_EVO_CONTROL_OWNER_PID}." >&2
    return 1
  fi

  ready_dir="$(dirname "${ROBOHARN_EVO_SKIP_READY_FILE}")"
  mkdir -p "${ready_dir}"
  ready_tmp="${ROBOHARN_EVO_SKIP_READY_FILE}.tmp.$$"
  {
    printf 'schema_version\t1\n'
    printf 'batch_token\t%s\n' "${ROBOHARN_EVO_CONTROL_BATCH_TOKEN}"
    printf 'run_token\t%s\n' "${ROBOHARN_EVO_CONTROL_RUN_TOKEN}"
    printf 'boot_id\t%s\n' "${ROBOHARN_EVO_CONTROL_BOOT_ID}"
    printf 'owner_pid\t%s\n' "${ROBOHARN_EVO_CONTROL_OWNER_PID}"
    printf 'owner_start_ticks\t%s\n' "${ROBOHARN_EVO_CONTROL_OWNER_START_TICKS}"
    printf 'runner_pid\t%s\n' "${runner_host_pid}"
    printf 'runner_start_ticks\t%s\n' "${runner_start_ticks}"
    printf 'runner_ppid\t%s\n' "${runner_ppid}"
    printf 'interrupt_policy\t%s\n' "${INTERRUPT_ESCALATION_POLICY}"
    printf 'worker_count\t%d\n' "${#pids[@]}"
    for worker_pid in "${pids[@]}"; do
      if ! local_pid_to_host_pid "${worker_pid}" worker_host_pid; then
        echo "Could not resolve worker ${worker_pid} host PID for skip readiness." >&2
        return 1
      fi
      if ! read_proc_identity \
          "${worker_host_pid}" worker_state worker_ppid worker_start_ticks; then
        echo "Worker ${worker_host_pid} exited before skip readiness was published." >&2
        return 1
      fi
      printf 'worker\t%s\t%s\t%s\n' \
        "${worker_host_pid}" "${worker_start_ticks}" "${worker_ppid}"
    done
    printf 'ready_at\t%s\n' "$(date --iso-8601=seconds)"
  } > "${ready_tmp}"
  mv -f "${ready_tmp}" "${ROBOHARN_EVO_SKIP_READY_FILE}"
}

stop_heartbeat() {
  if [[ -n "${heartbeat_pid}" ]] && kill -0 "${heartbeat_pid}" 2>/dev/null; then
    if [[ "${INTERRUPT_ESCALATION_POLICY}" == "ctrl_c_only" ]]; then
      # In ctrl_c_only mode the callers invoke this only after all workers
      # have been reaped.  The helper then observes workers_alive=0 and exits
      # naturally; no TERM/KILL is sent even to bookkeeping processes.
      wait "${heartbeat_pid}" 2>/dev/null || true
    else
      kill "${heartbeat_pid}" 2>/dev/null || true
      wait "${heartbeat_pid}" 2>/dev/null || true
    fi
  fi
  heartbeat_pid=""
}

heartbeat_workers() {
  while true; do
    sleep "${HEARTBEAT_SEC}"
    local alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    echo "[heartbeat] $(date --iso-8601=seconds) workers_alive=${alive}/${#pids[@]}"
    if (( alive == 0 )); then
      return
    fi
  done
}

shutdown_workers() {
  local signal_name="$1"
  local exit_code="$2"
  local watchdog_pid=""
  # Ignore duplicate interrupts while cleanup is already in progress.  In
  # ctrl_c_only mode the launcher must remain alive until every owned worker
  # has exited; otherwise a second signal could orphan an eval worker.
  trap '' INT TERM
  if [[ "${INTERRUPT_ESCALATION_POLICY}" == "ctrl_c_only" ]]; then
    echo "Received ${signal_name}; forwarding SIGINT to owned workers (grace=${SHUTDOWN_GRACE_SEC}s, escalation=disabled)." >&2
    if (( ${#pids[@]} > 0 )); then
      for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          kill -INT "${pid}" 2>/dev/null || true
        fi
      done
      local interrupt_started
      local alive_count
      local cleanup_deadline_missed=0
      interrupt_started="$(date +%s)"
      while true; do
        alive_count=0
        for pid in "${pids[@]}"; do
          if kill -0 "${pid}" 2>/dev/null; then
            alive_count=$((alive_count + 1))
          fi
        done
        if (( alive_count == 0 )); then
          for pid in "${pids[@]}"; do
            wait "${pid}" 2>/dev/null || true
          done
          break
        fi
        if (( $(date +%s) - interrupt_started >= SHUTDOWN_GRACE_SEC )); then
          if [[ "${cleanup_deadline_missed}" == "0" ]]; then
            cleanup_deadline_missed=1
            echo "${alive_count} owned worker(s) remain after SIGINT grace; no SIGTERM or SIGKILL was sent, and this batch will not advance." >&2
            if [[ -f "${ROBOHARN_EVO_SKIP_ACCEPTED_FILE}" ]]; then
              write_skip_marker \
                "${ROBOHARN_EVO_SKIP_CLEANUP_FAILED_FILE}" cleanup_failed \
                workers_remained_after_sigint_grace
            fi
          fi
        fi
        sleep 1
      done
      if [[ "${cleanup_deadline_missed}" == "1" ]]; then
        if [[ -f "${ROBOHARN_EVO_SKIP_ACCEPTED_FILE}" ]]; then
          echo "Owned workers eventually exited, but skip cleanup missed its grace deadline; stopping the batch." >&2
          exit 75
        fi
        echo "Owned workers eventually exited after the grace advisory; honoring the original interrupt result." >&2
      fi
    fi
    stop_heartbeat
    if [[ -f "${ROBOHARN_EVO_SKIP_ACCEPTED_FILE}" ]]; then
      write_skip_marker \
        "${ROBOHARN_EVO_SKIP_CLEANUP_COMPLETE_FILE}" cleanup_complete operator_skip_current
    fi
  else
    stop_heartbeat
    echo "Received ${signal_name}; requesting graceful worker shutdown (grace=${SHUTDOWN_GRACE_SEC}s)." >&2
    if (( ${#pids[@]} > 0 )); then
      for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
          kill -TERM "${pid}" 2>/dev/null || true
        fi
      done
      if (( SHUTDOWN_GRACE_SEC > 0 )); then
        (
          sleep "${SHUTDOWN_GRACE_SEC}"
          for pid in "${pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
              echo "Worker ${pid} did not stop within grace period; forcing shutdown." >&2
              kill -KILL "${pid}" 2>/dev/null || true
            fi
          done
        ) &
        watchdog_pid="$!"
      fi
      for pid in "${pids[@]}"; do
        wait "${pid}" 2>/dev/null || true
      done
      if [[ -n "${watchdog_pid}" ]] && kill -0 "${watchdog_pid}" 2>/dev/null; then
        kill "${watchdog_pid}" 2>/dev/null || true
        wait "${watchdog_pid}" 2>/dev/null || true
      fi
    fi
  fi
  echo "Interrupted run logs: ${LOG_ROOT}" >&2
  exit "${exit_code}"
}
trap 'shutdown_workers SIGINT 130' INT
trap 'shutdown_workers SIGTERM 143' TERM

echo "Launching ${NUM_WORKERS} pure tool-control eval workers"
echo "Logs: ${LOG_ROOT}"

for (( worker = 0; worker < NUM_WORKERS; worker++ )); do
  gpu="${GPU_ARRAY[worker]}"
  seed=$((SEED_OFFSET + worker))
  eval_start_seed_args=()
  eval_seed_suffix=""
  if [[ -n "${EVAL_START_SEEDS}" ]]; then
    eval_start_seed="${EVAL_START_SEED_ARRAY[worker]}"
    eval_start_seed_args=(--eval.start_seed "${eval_start_seed}")
    eval_seed_suffix="_e${eval_start_seed}"
  fi
  ckpt_setting="${BASE_CKPT}_w${worker}_g${gpu}_s${seed}${eval_seed_suffix}"
  log_file="${LOG_ROOT}/worker_${worker}_gpu_${gpu}_seed_${seed}${eval_seed_suffix}.log"
  segmentation_artifact_dir="${SEGMENTATION_ARTIFACT_ROOT}/worker_${worker}_gpu_${gpu}_seed_${seed}${eval_seed_suffix}"
  mkdir -p "${segmentation_artifact_dir}"

  if [[ -n "${SAM3_BASE_PORT}" ]]; then
    segmentation_url="http://127.0.0.1:$((SAM3_BASE_PORT + worker))"
  else
    segmentation_url="${SAM3_SERVICE_URL}"
  fi

  cuda_visible_device_token="${RMBENCH_CUDA_VISIBLE_DEVICE_TOKEN:-${gpu}}"

  echo "[worker ${worker}] gpu=${gpu} cuda_visible_device=${cuda_visible_device_token} seed=${seed} eval_start_seed=${eval_start_seed:-auto} ckpt=${ckpt_setting} sam3=${segmentation_url}"

  /usr/bin/env \
  --default-signal=INT \
  --default-signal=QUIT \
  CUDA_VISIBLE_DEVICES="${cuda_visible_device_token}" \
  PYTHONPATH="${FORMAL_PYTHONPATH}" \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONWARNINGS=ignore::UserWarning \
  RMBENCH_ROOT="${RMBENCH_ROOT}" \
  RMBENCH_ASSETS_ROOT="${RMBENCH_ASSETS_ROOT}" \
  RMBENCH_OUTPUT_ROOT="${RMBENCH_OUTPUT_ROOT}" \
  ROBOHARN_EVO_OUTPUT_ROOT="${ROBOHARN_EVO_OUTPUT_ROOT}" \
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
  ROBOHARN_EVO_FORMAL_PROTOCOL="${ROBOHARN_EVO_FORMAL_PROTOCOL}" \
  ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION="${ROBOHARN_EVO_FORMAL_PROTOCOL_VERSION}" \
  ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH="${ROBOHARN_EVO_RUNTIME_PROVENANCE_PATH}" \
  ROBOHARN_EVO_SAM3_SERVICE_URL="${segmentation_url}" \
  ROBOHARN_EVO_SEGMENTATION_ARTIFACT_DIR="${segmentation_artifact_dir}" \
  RMBENCH_RENDER_DEVICE="${RMBENCH_RENDER_DEVICE}" \
  RMBENCH_RENDER_DEVICE_STRICT="${RMBENCH_RENDER_DEVICE_STRICT}" \
  RMBENCH_EXPECTED_RENDER_CUDA_ID="${RMBENCH_EXPECTED_RENDER_CUDA_ID}" \
  RMBENCH_EXPECTED_RENDER_PCI_BUS_ID="${RMBENCH_EXPECTED_RENDER_PCI_BUS_ID}" \
  RMBENCH_EXPECTED_PHYSICAL_GPU="${RMBENCH_EXPECTED_PHYSICAL_GPU}" \
  RMBENCH_RENDER_DEVICE_PROVENANCE_PATH="${RMBENCH_RENDER_DEVICE_PROVENANCE_PATH}" \
  RMBENCH_RAY_TRACING_DENOISER="${RMBENCH_RAY_TRACING_DENOISER}" \
  RMBENCH_SLOW_ACTION_TRACE_SEC="${RMBENCH_SLOW_ACTION_TRACE_SEC}" \
  python -u script/eval_policy.py \
    --config policy/roboharn_evo/deploy_policy.yml \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --instruction_set "${INSTRUCTION_SET}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${POLICY_NAME}" \
    --eval.test_num "${N_PER_WORKER}" \
    "${eval_start_seed_args[@]}" \
    --planner.agent_api.server_url "${AGENT_API_BASE_URL}/plan" \
    --planner.agent_api.timeout_sec "${PLANNER_TIMEOUT_SEC}" \
    --ood.agent_api.server_url "${AGENT_API_BASE_URL}/ood" \
    --ood.agent_api.timeout_sec "${OOD_TIMEOUT_SEC}" \
    --recovery.agent_api.server_url "${AGENT_API_BASE_URL}/recover" \
    --recovery.agent_api.timeout_sec "${RECOVERY_TIMEOUT_SEC}" \
    --agent.observation_preprocess.enabled True \
    --agent.observation_preprocess.every_n_steps 1 \
    --agent.observation_preprocess.auto_objects True \
    --agent.observation_preprocess.query_url "${AGENT_API_BASE_URL}/perception_queries" \
    --agent.observation_preprocess.normalization_url "${AGENT_API_BASE_URL}/normalize_perception_queries" \
    --agent.observation_preprocess.query_timeout_sec "${QUERY_TIMEOUT_SEC}" \
    --agent.observation_preprocess.max_objects "${MAX_OBJECTS}" \
    --agent.observation_preprocess.segmentation.cameras "['head']" \
    --agent.observation_preprocess.segmentation.service_url "${segmentation_url}" \
    --agent.observation_preprocess.grounding.cameras "['head']" \
    --agent.pure_tool_control.enabled True \
    --agent.pure_tool_control.trigger_step 0 \
    --agent.pure_tool_control.signal task_level_recovery_control \
    --agent.pure_tool_control.reason "paper pure tool-control baseline" \
    --agent.pure_tool_control.wait_for_scene_memory True \
    --agent.pure_tool_control.max_wait_steps "${MAX_WAIT_STEPS}" \
    --agent.pure_tool_control.retry_budget "${RETRY_BUDGET}" \
    --agent.pure_tool_control.max_rounds "${MAX_ROUNDS}" \
    --agent.pure_tool_control.max_control_turns "${MAX_CONTROL_TURNS}" \
    --agent.pure_tool_control.max_no_progress_control_turns "${MAX_NO_PROGRESS_CONTROL_TURNS}" \
    --agent.pure_tool_control.backend_error_budget "${BACKEND_ERROR_BUDGET}" \
    --agent.pure_tool_control.empty_plan_replan_threshold "${EMPTY_PLAN_REPLAN_THRESHOLD}" \
    --agent.pure_tool_control.bootstrap_with_planner True \
    --agent.pure_tool_control.skip_vla_rollout True \
    "${EXTRA_ARGS[@]}" \
    "${FORMAL_PERCEPTION_OVERRIDE_ARGS[@]}" \
    > "${log_file}" 2>&1 &

  pids+=("$!")
  worker_names+=("worker_${worker}_gpu_${gpu}_seed_${seed}${eval_seed_suffix}")
  if (( worker + 1 < NUM_WORKERS )) && [[ "${WORKER_START_DELAY_SEC}" != "0" ]]; then
    sleep "${WORKER_START_DELAY_SEC}"
  fi
done

if ! publish_skip_ready; then
  echo "Could not establish safe skip-control readiness; stopping owned workers." >&2
  shutdown_workers INTERNAL_ERROR 75
fi

if (( HEARTBEAT_SEC > 0 )); then
  heartbeat_workers &
  heartbeat_pid="$!"
fi

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[idx]}"
  name="${worker_names[idx]}"
  if wait "${pid}"; then
    echo "[done] ${name}"
  else
    rc=$?
    echo "[failed] ${name} exit=${rc}" >&2
    status=1
  fi
done

stop_heartbeat

echo "Finished. Logs: ${LOG_ROOT}"
exit "${status}"
