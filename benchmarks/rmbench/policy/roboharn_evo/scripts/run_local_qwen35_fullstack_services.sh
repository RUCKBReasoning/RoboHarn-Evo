#!/usr/bin/env bash
set -euo pipefail

unalias python 2>/dev/null || true
unalias pip 2>/dev/null || true
hash -r

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to benchmarks/rmbench}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
MODEL_ID="Qwen/Qwen3.5-397B-A17B-FP8"
MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to the Qwen checkpoint directory}"
VLLM_ENV="${VLLM_ENV:?Set VLLM_ENV to the vLLM environment directory}"
VLLM_PYTHON="${VLLM_PYTHON:-${VLLM_ENV}/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ENV}/bin/vllm}"
RMBENCH_PYTHON="${RMBENCH_PYTHON:-python}"
RUNTIME_IDENTITY_RECORDER="${REPO_ROOT}/policy/roboharn_evo/scripts/record_local_serving_runtime_identity.py"
MODEL_IDENTITY_MANIFEST="${MODEL_IDENTITY_MANIFEST:-${REPO_ROOT}/policy/roboharn_evo/docs/experiments/qwen35_397b_a17b_fp8_local_model_identity.json}"
AGENT_CONTRACT_MANIFEST="${AGENT_CONTRACT_MANIFEST:-${REPO_ROOT}/policy/roboharn_evo/docs/experiments/qwen35_local_shared_agent_contract.json}"

ENGINE_HOST="127.0.0.1"
ENGINE_PORT="${QWEN_ENGINE_PORT:-8000}"
GATEWAY_HOST="127.0.0.1"
GATEWAY_PORT="${QWEN_GATEWAY_PORT:-9105}"
ENGINE_BASE_URL="http://${ENGINE_HOST}:${ENGINE_PORT}"
GATEWAY_BASE_URL="http://${GATEWAY_HOST}:${GATEWAY_PORT}"
ENGINE_SESSION="${QWEN_ENGINE_SESSION:-qwen35_vllm_8000}"
GATEWAY_SESSION="${QWEN_GATEWAY_SESSION:-qwen35_gateway_9105}"
SESSION_OWNER_OPTION="@rmbench_qwen35_service_owner"

SERVICE_PROFILE="${QWEN_SERVICE_PROFILE:-smoke}"
case "${SERVICE_PROFILE}" in
  smoke)
    PROFILE_CONTEXT_LIMIT=32768
    PROFILE_OUTPUT_LIMIT=8192
    ;;
  development)
    PROFILE_CONTEXT_LIMIT=65536
    PROFILE_OUTPUT_LIMIT=32768
    ;;
  *)
    echo "QWEN_SERVICE_PROFILE must be smoke or development, got ${SERVICE_PROFILE}." >&2
    exit 2
    ;;
esac
CONTEXT_LIMIT="${QWEN_CONTEXT_LIMIT:-${PROFILE_CONTEXT_LIMIT}}"
OUTPUT_LIMIT="${QWEN_OUTPUT_LIMIT:-${PROFILE_OUTPUT_LIMIT}}"
PLANNER_OUTPUT_LIMIT="${QWEN_PLANNER_OUTPUT_LIMIT:-8192}"
MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.80}"
GATEWAY_CONCURRENCY="${QWEN_GATEWAY_CONCURRENCY:-1}"
ENGINE_START_TIMEOUT_SEC="${QWEN_ENGINE_START_TIMEOUT_SEC:-1800}"
GATEWAY_START_TIMEOUT_SEC="${QWEN_GATEWAY_START_TIMEOUT_SEC:-60}"
RUNTIME_PYTHON_PROBE_TIMEOUT_SEC="${QWEN_RUNTIME_PYTHON_PROBE_TIMEOUT_SEC:-300}"
RUNTIME_VLLM_VERSION_TIMEOUT_SEC="${QWEN_RUNTIME_VLLM_VERSION_TIMEOUT_SEC:-300}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${QWEN_SERVICE_LOG_ROOT:-/tmp/qwen35_local_fullstack_${RUN_STAMP}}"
ENGINE_LOG="${QWEN_ENGINE_LOG:-${LOG_ROOT}/vllm_8000.log}"
GATEWAY_LOG="${QWEN_GATEWAY_LOG:-${LOG_ROOT}/gateway_9105.log}"
SERVING_RUNTIME_IDENTITY="${LOG_ROOT}/local_serving_runtime_identity.json"
SERVING_RUNTIME_IDENTITY_SHA256="${SERVING_RUNTIME_IDENTITY_SHA256:-}"

# FlashInfer's sampling JIT expects a conventional system CUDA toolkit layout.
# This runtime instead ships CUDA inside the isolated Python environment.  vLLM
# provides an equivalent native PyTorch/Triton sampler, selected explicitly so
# startup never depends on an unrecorded JIT toolchain discovery side effect.
VLLM_USE_FLASHINFER_SAMPLER_VALUE="0"

# This is the single source of truth for both the audited argument vector and
# the actual vLLM invocation. Keep ordering intact for exact reproducibility.
VLLM_SERVER_ARGS=(
  serve "${MODEL_DIR}"
  --served-model-name "${MODEL_ID}"
  --host "${ENGINE_HOST}"
  --port "${ENGINE_PORT}"
  --tensor-parallel-size 8
  --max-model-len "${CONTEXT_LIMIT}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --reasoning-parser qwen3
  --limit-mm-per-prompt '{"image":2}'
  --enforce-eager
)

usage() {
  echo "Usage: $0 {start|status|stop}"
  echo "  start: launch local vLLM on 8000 and the RoboHarn-Evo gateway on 9105 in separate tmux sessions"
  echo "  status: print session and health status without changing processes"
  echo "  stop: send Ctrl+C only to sessions carrying this script's ownership marker; never sends SIGKILL"
  echo "  QWEN_SERVICE_PROFILE=smoke (32768/8192, default) or development (65536/32768)"
  echo "  QWEN_PLANNER_OUTPUT_LIMIT=8192 by default; applies only to /plan"
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got ${value}." >&2
    exit 2
  fi
}

require_probe_timeout() {
  local name="$1"
  local value="$2"
  require_positive_integer "${name}" "${value}"
  if (( value > 1800 )); then
    echo "${name} must not exceed 1800 seconds, got ${value}." >&2
    exit 2
  fi
}

validate_configuration() {
  require_positive_integer QWEN_CONTEXT_LIMIT "${CONTEXT_LIMIT}"
  require_positive_integer QWEN_OUTPUT_LIMIT "${OUTPUT_LIMIT}"
  require_positive_integer QWEN_PLANNER_OUTPUT_LIMIT "${PLANNER_OUTPUT_LIMIT}"
  require_positive_integer QWEN_MAX_NUM_SEQS "${MAX_NUM_SEQS}"
  require_positive_integer QWEN_GATEWAY_CONCURRENCY "${GATEWAY_CONCURRENCY}"
  require_positive_integer QWEN_ENGINE_START_TIMEOUT_SEC "${ENGINE_START_TIMEOUT_SEC}"
  require_positive_integer QWEN_GATEWAY_START_TIMEOUT_SEC "${GATEWAY_START_TIMEOUT_SEC}"
  require_probe_timeout QWEN_RUNTIME_PYTHON_PROBE_TIMEOUT_SEC "${RUNTIME_PYTHON_PROBE_TIMEOUT_SEC}"
  require_probe_timeout QWEN_RUNTIME_VLLM_VERSION_TIMEOUT_SEC "${RUNTIME_VLLM_VERSION_TIMEOUT_SEC}"
  if (( OUTPUT_LIMIT >= CONTEXT_LIMIT )); then
    echo "QWEN_OUTPUT_LIMIT must be smaller than QWEN_CONTEXT_LIMIT." >&2
    exit 2
  fi
  if (( PLANNER_OUTPUT_LIMIT > OUTPUT_LIMIT )); then
    echo "QWEN_PLANNER_OUTPUT_LIMIT must not exceed QWEN_OUTPUT_LIMIT." >&2
    exit 2
  fi
  if [[ ! "${GPU_MEMORY_UTILIZATION}" =~ ^0\.[0-9]+$ ]]; then
    echo "QWEN_GPU_MEMORY_UTILIZATION must be a fraction such as 0.80." >&2
    exit 2
  fi
  if [[ "${LOG_ROOT}" != /* || "$(readlink -m -- "${LOG_ROOT}")" != "${LOG_ROOT}" ]]; then
    echo "QWEN_SERVICE_LOG_ROOT must be an absolute normalized path: ${LOG_ROOT}" >&2
    exit 2
  fi
}

record_serving_runtime_identity() {
  local command=(
    "${RMBENCH_PYTHON}"
    "${RUNTIME_IDENTITY_RECORDER}"
    --python "${VLLM_PYTHON}"
    --output "${SERVING_RUNTIME_IDENTITY}"
    --python-probe-timeout-sec "${RUNTIME_PYTHON_PROBE_TIMEOUT_SEC}"
    --vllm-version-timeout-sec "${RUNTIME_VLLM_VERSION_TIMEOUT_SEC}"
  )
  local argument
  for argument in "${VLLM_SERVER_ARGS[@]}"; do
    command+=("--server-arg=${argument}")
  done
  command+=(
    "--runtime-setting=VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER_VALUE}"
  )
  env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    PYTHONDONTWRITEBYTECODE=1 \
    "${command[@]}"
}

read_recorded_vllm_version() {
  "${RMBENCH_PYTHON}" -c '
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["vllm"]["version_output"])
' "${SERVING_RUNTIME_IDENTITY}"
}

validate_recorded_vllm_binary() {
  RUNTIME_IDENTITY_PATH="${SERVING_RUNTIME_IDENTITY}" \
  ACTUAL_VLLM_BIN="${VLLM_BIN}" \
  "${RMBENCH_PYTHON}" -c '
import hashlib
import json
import os
from pathlib import Path

identity_path = Path(os.environ["RUNTIME_IDENTITY_PATH"]).expanduser().resolve(strict=True)
requested = Path(os.environ["ACTUAL_VLLM_BIN"]).expanduser()
if not requested.is_absolute():
    raise SystemExit("VLLM_BIN must be an absolute path for runtime identity validation")
try:
    actual = requested.resolve(strict=True)
except OSError as exc:
    raise SystemExit(f"cannot resolve the actual VLLM_BIN: {exc}") from exc
if not actual.is_file() or not os.access(actual, os.X_OK):
    raise SystemExit("the resolved VLLM_BIN is not an executable regular file")

payload = json.loads(identity_path.read_text(encoding="utf-8"))
vllm = payload.get("vllm") if isinstance(payload, dict) else None
if not isinstance(vllm, dict):
    raise SystemExit("serving runtime identity omitted vllm identity")
recorded_text = vllm.get("executable")
if not isinstance(recorded_text, str) or not recorded_text:
    raise SystemExit("serving runtime identity omitted vllm.executable")
try:
    recorded = Path(recorded_text).resolve(strict=True)
except OSError as exc:
    raise SystemExit(f"cannot resolve recorded vllm.executable: {exc}") from exc
if recorded != actual:
    raise SystemExit(
        "serving runtime mismatch: recorded vllm.executable is not the actual VLLM_BIN"
    )

expected_sha256 = vllm.get("executable_sha256")
if (
    not isinstance(expected_sha256, str)
    or len(expected_sha256) != 64
    or any(character not in "0123456789abcdef" for character in expected_sha256)
):
    raise SystemExit("serving runtime identity has invalid vllm.executable_sha256")
digest = hashlib.sha256()
with actual.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != expected_sha256:
    raise SystemExit("actual VLLM_BIN SHA256 differs from the serving runtime identity")
'
}

validate_model_identity_manifest() {
  MODEL_DIR="${MODEL_DIR}" MODEL_ID="${MODEL_ID}" \
  MODEL_IDENTITY_MANIFEST="${MODEL_IDENTITY_MANIFEST}" \
  "${RMBENCH_PYTHON}" -c '
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_DIR"]).resolve()
manifest_path = Path(os.environ["MODEL_IDENTITY_MANIFEST"]).resolve()
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
errors = []
if payload.get("model_directory") != str(root):
    errors.append("model_directory")
source = payload.get("source", {})
if source.get("repo_id") != os.environ["MODEL_ID"]:
    errors.append("source.repo_id")
artifact = payload.get("artifact", {})
expected_sizes = artifact.get("file_sizes_bytes")
if not isinstance(expected_sizes, dict) or not expected_sizes:
    errors.append("artifact.file_sizes_bytes")
    expected_sizes = {}

actual_sizes = {}
unsafe_entries = []
for path in root.rglob("*"):
    relative = path.relative_to(root).as_posix()
    if path.is_symlink():
        unsafe_entries.append(relative)
    elif path.is_file():
        actual_sizes[relative] = path.stat().st_size
if unsafe_entries:
    errors.append("symbolic_links=" + ",".join(sorted(unsafe_entries)[:3]))
if actual_sizes != expected_sizes:
    missing = sorted(set(expected_sizes) - set(actual_sizes))
    extra = sorted(set(actual_sizes) - set(expected_sizes))
    wrong_size = sorted(
        name for name in set(actual_sizes) & set(expected_sizes)
        if actual_sizes[name] != expected_sizes[name]
    )
    errors.append(
        "artifact_files(missing=%r,extra=%r,wrong_size=%r)"
        % (missing[:3], extra[:3], wrong_size[:3])
    )

digest = hashlib.sha256()
for relative, size in sorted(actual_sizes.items()):
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(size).encode("ascii"))
    digest.update(b"\n")
if digest.hexdigest() != artifact.get("manifest_sha256"):
    errors.append("artifact.manifest_sha256")

for relative, expected_hash in artifact.get("evidence_file_sha256", {}).items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        errors.append("evidence_hash:" + relative)

quantization = payload.get("model_metadata", {}).get("quantization", {})
if quantization.get("quant_method") != "fp8" or quantization.get("weight_block_size") != [128, 128]:
    errors.append("model_metadata.quantization")
if errors:
    raise SystemExit("local model identity validation failed: " + "; ".join(errors))
'
}

validate_agent_contract_manifest() {
  REPO_ROOT="${REPO_ROOT}" AGENT_CONTRACT_MANIFEST="${AGENT_CONTRACT_MANIFEST}" \
  PYTHONPATH="${REPO_ROOT}" "${RMBENCH_PYTHON}" -c '
import json
import os
from pathlib import Path

from policy.roboharn_evo.scripts.record_agent_contract_manifest import build_agent_contract_manifest

root = Path(os.environ["REPO_ROOT"]).resolve()
stored = json.loads(Path(os.environ["AGENT_CONTRACT_MANIFEST"]).read_text(encoding="utf-8"))
current = build_agent_contract_manifest(
    repo_root=root,
    perception_condition="no_oracle",
    primary_cameras=["head"],
    verification_cameras=["third"],
    max_objects=8,
    instruction_set="rmbench_original",
)
if stored.get("contract_composite_sha256") != current.get("contract_composite_sha256"):
    raise SystemExit(
        "Agent contract manifest is stale: stored=%r current=%r"
        % (
            stored.get("contract_composite_sha256"),
            current.get("contract_composite_sha256"),
        )
    )
'
}

require_port_free() {
  local name="$1"
  local host="$2"
  local port="$3"
  if ! "${RMBENCH_PYTHON}" -c '
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    if sock.connect_ex((host, port)) == 0:
        raise SystemExit(1)
' "${host}" "${port}"; then
    echo "Refusing to start: ${name} port is already listening at ${host}:${port}." >&2
    return 1
  fi
}

port_is_listening() {
  local host="$1"
  local port="$2"
  "${RMBENCH_PYTHON}" -c '
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.5)
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
' "${host}" "${port}"
}

verify_service_ports_closed() {
  local status=0
  if port_is_listening "${GATEWAY_HOST}" "${GATEWAY_PORT}"; then
    echo "Gateway port remains open after Ctrl+C: ${GATEWAY_HOST}:${GATEWAY_PORT}; no stronger signal was sent." >&2
    status=1
  fi
  if port_is_listening "${ENGINE_HOST}" "${ENGINE_PORT}"; then
    echo "vLLM port remains open after Ctrl+C: ${ENGINE_HOST}:${ENGINE_PORT}; no stronger signal was sent." >&2
    status=1
  fi
  return "${status}"
}

health_to_file() {
  local url="$1"
  local output="$2"
  curl --noproxy '*' --fail --silent --show-error --max-time 10 "${url}" --output "${output}"
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local timeout_sec="$3"
  local session_name="$4"
  local service_kind="$5"
  local started
  started="$(date +%s)"
  while true; do
    if curl --noproxy '*' --fail --silent --max-time 5 "${url}" >/dev/null 2>&1; then
      if ! session_exists_exact "${session_name}"; then
        echo "${name} URL responded, but the newly created session has exited: ${session_name}" >&2
        return 1
      fi
      if ! session_is_owned "${session_name}" "${service_kind}"; then
        echo "${name} URL responded, but session ownership is invalid: ${session_name}" >&2
        return 1
      fi
      return 0
    fi
    if ! session_exists_exact "${session_name}"; then
      echo "${name} tmux session exited before becoming healthy: ${session_name}" >&2
      return 1
    fi
    if (( $(date +%s) - started >= timeout_sec )); then
      echo "Timed out waiting ${timeout_sec}s for ${name}: ${url}" >&2
      return 1
    fi
    sleep 5
  done
}

capture_gpt9104_identity() {
  local health_path="$1"
  local identity_path="$2"
  health_to_file "http://127.0.0.1:9104/health" "${health_path}"
  HEALTH_PATH="${health_path}" IDENTITY_PATH="${identity_path}" EXPECTED_PORT=9104 \
  "${RMBENCH_PYTHON}" -c '
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def json_object_without_duplicates(raw):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in 9104 health")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=hook)


def listener_inodes(port):
    result = set()
    for table_name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(table_name).read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                local_port = int(fields[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                result.add(fields[9])
    if not result:
        raise RuntimeError(f"no listening socket inode found for port {port}")
    return result


def listener_owner(inodes):
    owners = {}
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            descriptors = list((process_dir / "fd").iterdir())
        except OSError:
            continue
        owned = set()
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
            if match and match.group(1) in inodes:
                owned.add(match.group(1))
        if owned:
            owners[int(process_dir.name)] = sorted(owned, key=int)
    if len(owners) != 1:
        raise RuntimeError(
            "expected exactly one process owning the 9104 listener, found "
            + str(len(owners))
        )
    return next(iter(owners.items()))


def sanitized_cmdline(pid):
    raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    arguments = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    if not arguments:
        raise RuntimeError("9104 listener has an empty command line")
    sensitive = re.compile(
        r"(?i)(?:api[-_]?key|auth[-_]?file|authorization|bearer|cookie|credential|"
        r"password|secret|access[-_]?token|refresh[-_]?token|hf[-_]?token)"
    )
    obvious_value = re.compile(r"(?i)(?:bearer\s+\S+|sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,})")
    redacted = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            redacted.append("[REDACTED]")
            redact_next = False
            continue
        if "=" in argument:
            name, value = argument.split("=", 1)
            if sensitive.search(name) or sensitive.search(value) or obvious_value.search(value):
                redacted.append(name + "=[REDACTED]")
                continue
        if argument.startswith("-") and sensitive.search(argument):
            normalized_flag = argument.lstrip("-").replace("_", "-")
            if sensitive.fullmatch(normalized_flag):
                redacted.append(argument)
                redact_next = True
            else:
                redacted.append("[REDACTED_ARGUMENT]")
            continue
        if sensitive.search(argument) or obvious_value.search(argument):
            redacted.append("[REDACTED]")
            continue
        redacted.append(argument)
    encoded = json.dumps(redacted, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return redacted, hashlib.sha256(encoded).hexdigest()


health_path = Path(os.environ["HEALTH_PATH"]).resolve(strict=True)
identity_path = Path(os.environ["IDENTITY_PATH"]).resolve()
payload = json_object_without_duplicates(health_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit("9104 health must be a JSON object")
expected = {
    "model": "gpt-5.5",
    "api_mode": "responses_compat",
    "reasoning_effort": "xhigh",
}
errors = [key for key, value in expected.items() if payload.get(key) != value]
if errors:
    raise SystemExit("9104 identity mismatch in fields: " + ", ".join(errors))

inodes = listener_inodes(int(os.environ["EXPECTED_PORT"]))
pid, owned_inodes = listener_owner(inodes)
process_dir = Path("/proc") / str(pid)
stat_text = (process_dir / "stat").read_text(encoding="utf-8")
try:
    stat_fields = stat_text.rsplit(") ", 1)[1].split()
    starttime_ticks = int(stat_fields[19])
except (IndexError, ValueError) as exc:
    raise RuntimeError("cannot parse 9104 listener starttime") from exc
redacted_argv, redacted_sha256 = sanitized_cmdline(pid)
try:
    executable = str((process_dir / "exe").resolve(strict=True))
except OSError as exc:
    raise RuntimeError("cannot resolve 9104 listener executable") from exc

canonical_health = json.dumps(
    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
identity = {
    "schema_version": 1,
    "health_canonical_sha256": hashlib.sha256(canonical_health).hexdigest(),
    "listener": {
        "pid": pid,
        "starttime_ticks": starttime_ticks,
        "executable": executable,
        "socket_inodes": owned_inodes,
        "cmdline_redacted": redacted_argv,
        "cmdline_redacted_sha256": redacted_sha256,
    },
    "secrets_recorded": False,
}
identity_path.parent.mkdir(parents=True, exist_ok=True)
temporary_name = None
try:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + identity_path.name + ".", suffix=".tmp", dir=identity_path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(identity, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, identity_path)
    temporary_name = None
finally:
    if temporary_name is not None:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass
'
}

compare_gpt9104_identities() {
  local before_path="$1"
  local after_path="$2"
  "${RMBENCH_PYTHON}" -c '
import json
from pathlib import Path
import sys

before = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
after = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if before != after:
    changed = sorted(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    )
    listener_before = before.get("listener", {})
    listener_after = after.get("listener", {})
    listener_changed = sorted(
        key
        for key in set(listener_before) | set(listener_after)
        if listener_before.get(key) != listener_after.get(key)
    )
    details = []
    if changed:
        details.append("top_level=" + ",".join(changed))
    if listener_changed:
        details.append("listener=" + ",".join(listener_changed))
    raise SystemExit("GPT 9104 changed during Qwen startup: " + "; ".join(details))
' "${before_path}" "${after_path}"
}

validate_qwen_gateway_health() {
  local path="$1"
  HEALTH_PATH="${path}" \
  EXPECTED_MODEL="${MODEL_ID}" \
  EXPECTED_SERVICE_URL="${GATEWAY_BASE_URL}" \
  EXPECTED_BASE_URL="${ENGINE_BASE_URL}/v1" \
  EXPECTED_SOURCE="ModelScope official Qwen/Qwen3.5-397B-A17B-FP8" \
  EXPECTED_REVISION="master (immutable revision not locally recorded)" \
  EXPECTED_FRAMEWORK="${VLLM_VERSION}" \
  EXPECTED_CONTEXT_LIMIT="${CONTEXT_LIMIT}" \
  EXPECTED_OUTPUT_LIMIT="${OUTPUT_LIMIT}" \
  EXPECTED_PLANNER_OUTPUT_LIMIT="${PLANNER_OUTPUT_LIMIT}" \
  EXPECTED_CONCURRENCY="${GATEWAY_CONCURRENCY}" \
  EXPECTED_MODEL_DIR="${MODEL_DIR}" \
  EXPECTED_MODEL_MANIFEST_PATH="${MODEL_IDENTITY_MANIFEST}" \
  EXPECTED_MANIFEST_SHA256="${MODEL_MANIFEST_SHA256}" \
  EXPECTED_AGENT_CONTRACT_PATH="${AGENT_CONTRACT_MANIFEST}" \
  EXPECTED_AGENT_CONTRACT_SHA256="${AGENT_CONTRACT_MANIFEST_SHA256}" \
  EXPECTED_SERVING_RUNTIME_IDENTITY_PATH="${SERVING_RUNTIME_IDENTITY}" \
  EXPECTED_SERVING_RUNTIME_IDENTITY_SHA256="${SERVING_RUNTIME_IDENTITY_SHA256}" \
  "${RMBENCH_PYTHON}" -c '
import hashlib
import json
import os
from pathlib import Path


def json_object_without_duplicates(raw):
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in 9105 health: " + str(key))
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=hook)


try:
    payload = json_object_without_duplicates(
        Path(os.environ["HEALTH_PATH"]).read_text(encoding="utf-8")
    )
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
    raise SystemExit("9105 health must be unambiguous UTF-8 JSON: " + str(exc)) from exc
if not isinstance(payload, dict):
    raise SystemExit("9105 health must be a JSON object")
expected = {
    "status": "ok",
    "backend": "openai",
    "local_only": True,
    "service_url": os.environ["EXPECTED_SERVICE_URL"],
    "model": os.environ["EXPECTED_MODEL"],
    "api_mode": "chat",
    "base_url": os.environ["EXPECTED_BASE_URL"],
    "provider": "local-vllm",
    "reasoning_effort": "",
    "thinking_mode": "enabled",
    "model_source": os.environ["EXPECTED_SOURCE"],
    "model_revision": os.environ["EXPECTED_REVISION"],
    "quantization": "official_fp8_block128",
    "serving_framework": os.environ["EXPECTED_FRAMEWORK"],
    "context_limit": int(os.environ["EXPECTED_CONTEXT_LIMIT"]),
    "max_output_tokens": int(os.environ["EXPECTED_OUTPUT_LIMIT"]),
    "planner_prompt_mode": "rendered_system_once",
    "planner_max_output_tokens": int(os.environ["EXPECTED_PLANNER_OUTPUT_LIMIT"]),
    "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "model_artifact": {
        "path": os.environ["EXPECTED_MODEL_DIR"],
        "manifest_path": os.environ["EXPECTED_MODEL_MANIFEST_PATH"],
        "manifest_sha256": os.environ["EXPECTED_MANIFEST_SHA256"],
    },
    "agent_contract": {
        "path": os.environ["EXPECTED_AGENT_CONTRACT_PATH"],
        "manifest_sha256": os.environ["EXPECTED_AGENT_CONTRACT_SHA256"],
    },
    "serving_runtime_identity": {
        "path": os.environ["EXPECTED_SERVING_RUNTIME_IDENTITY_PATH"],
        "sha256": os.environ["EXPECTED_SERVING_RUNTIME_IDENTITY_SHA256"],
    },
    "response_storage": "disabled",
    "timeout_sec": 600,
    "max_concurrent_requests": int(os.environ["EXPECTED_CONCURRENCY"]),
    "max_retries": 0,
    "auth_mode": "api_key",
    "responses_endpoint": None,
    "app_server_running": None,
    "fallback_enabled": False,
}
errors = []
for key, value in expected.items():
    if payload.get(key) != value:
        errors.append(f"{key}={payload.get(key)!r}, expected {value!r}")
runtime_identity_path = Path(os.environ["EXPECTED_SERVING_RUNTIME_IDENTITY_PATH"])
if not runtime_identity_path.is_file():
    errors.append("serving_runtime_identity.path is not a current regular file")
else:
    actual_runtime_sha256 = hashlib.sha256(runtime_identity_path.read_bytes()).hexdigest()
    if actual_runtime_sha256 != os.environ["EXPECTED_SERVING_RUNTIME_IDENTITY_SHA256"]:
        errors.append("serving_runtime_identity current file SHA256 mismatch")
upstream = payload.get("upstream_model_identity", {})
if upstream.get("verified") is not True:
    errors.append("upstream_model_identity.verified is not true")
if upstream.get("expected_model") != os.environ["EXPECTED_MODEL"]:
    errors.append("upstream_model_identity.expected_model mismatch")
models_sha = upstream.get("models_response_sha256")
if not isinstance(models_sha, str) or len(models_sha) != 64 or any(
    character not in "0123456789abcdef" for character in models_sha
):
    errors.append("upstream_model_identity.models_response_sha256 is not a SHA256")
models_identity_sha = upstream.get("models_identity_sha256")
if not isinstance(models_identity_sha, str) or len(models_identity_sha) != 64 or any(
    character not in "0123456789abcdef" for character in models_identity_sha
):
    errors.append("upstream_model_identity.models_identity_sha256 is not a SHA256")
if errors:
    raise SystemExit("9105 identity mismatch: " + "; ".join(errors))
'
}

session_owner_value() {
  local service_kind="$1"
  printf '%s:%s' "${SCRIPT_PATH}" "${service_kind}"
}

session_id_for_exact_name() {
  local expected_name="$1"
  local observed_name
  local observed_id
  local matched_id=""
  while IFS=$'\t' read -r observed_name observed_id; do
    [[ "${observed_name}" == "${expected_name}" ]] || continue
    if [[ -n "${matched_id}" ]]; then
      echo "Multiple tmux sessions reported the exact name ${expected_name}." >&2
      return 2
    fi
    matched_id="${observed_id}"
  done < <(tmux list-sessions -F $'#{session_name}\t#{session_id}' 2>/dev/null || true)
  [[ -n "${matched_id}" ]] || return 1
  printf '%s' "${matched_id}"
}

session_exists_exact() {
  session_id_for_exact_name "$1" >/dev/null
}

active_pane_id_for_exact_session() {
  local session_name="$1"
  local session_id
  local pane_id
  session_id="$(session_id_for_exact_name "${session_name}")" || return 1
  pane_id="$(tmux display-message -p -t "${session_id}" '#{pane_id}')" || return 1
  if [[ ! "${pane_id}" =~ ^%[0-9]+$ ]]; then
    echo "Invalid active pane id for tmux session ${session_name}: ${pane_id}" >&2
    return 1
  fi
  printf '%s' "${pane_id}"
}

send_ctrl_c_to_exact_session() {
  local session_name="$1"
  local pane_id
  pane_id="$(active_pane_id_for_exact_session "${session_name}")" || return 1
  tmux send-keys -t "${pane_id}" C-c
}

mark_session_owned() {
  local session_name="$1"
  local service_kind="$2"
  local session_id
  session_id="$(session_id_for_exact_name "${session_name}")" || return 1
  tmux set-option -t "${session_id}" "${SESSION_OWNER_OPTION}" \
    "$(session_owner_value "${service_kind}")"
}

session_is_owned() {
  local session_name="$1"
  local service_kind="$2"
  local actual
  local session_id
  session_id="$(session_id_for_exact_name "${session_name}")" || return 1
  actual="$(
    tmux show-options -v -t "${session_id}" "${SESSION_OWNER_OPTION}" 2>/dev/null || true
  )"
  [[ "${actual}" == "$(session_owner_value "${service_kind}")" ]]
}

stop_session_with_ctrl_c() {
  local session_name="$1"
  local service_kind="$2"
  local wait_sec="${3:-180}"
  if ! session_exists_exact "${session_name}"; then
    echo "Session not running: ${session_name}"
    return 0
  fi
  if ! session_is_owned "${session_name}" "${service_kind}"; then
    echo "Refusing to stop unowned tmux session: ${session_name}" >&2
    return 1
  fi
  echo "Sending Ctrl+C to ${session_name}"
  send_ctrl_c_to_exact_session "${session_name}"
  local started
  started="$(date +%s)"
  while session_exists_exact "${session_name}"; do
    if (( $(date +%s) - started >= wait_sec )); then
      echo "${session_name} did not exit after Ctrl+C within ${wait_sec}s; no kill signal was sent." >&2
      return 1
    fi
    sleep 2
  done
}

cleanup_owned_start_sessions_on_signal() {
  local signal_name="$1"
  local exit_code="$2"
  trap - INT TERM
  echo "Received ${signal_name} while starting Qwen services; requesting owned sessions to stop with Ctrl+C." >&2
  if [[ "${START_GATEWAY_SESSION_OWNED:-0}" == "1" ]]; then
    stop_session_with_ctrl_c "${GATEWAY_SESSION}" gateway 60 || true
  fi
  if [[ "${START_ENGINE_SESSION_OWNED:-0}" == "1" ]]; then
    stop_session_with_ctrl_c "${ENGINE_SESSION}" engine 180 || true
  fi
  echo "Startup interrupted; no SIGTERM or SIGKILL was sent." >&2
  exit "${exit_code}"
}

engine_process() {
  mkdir -p "${LOG_ROOT}"
  exec >>"${ENGINE_LOG}" 2>&1
  echo "[qwen35-vllm] started_at=$(date --iso-8601=seconds)"
  echo "[qwen35-vllm] profile=${SERVICE_PROFILE} model_dir=${MODEL_DIR} model_id=${MODEL_ID} host=${ENGINE_HOST} port=${ENGINE_PORT} tensor_parallel_size=8 context_limit=${CONTEXT_LIMIT} max_num_seqs=${MAX_NUM_SEQS} gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} reasoning_parser=qwen3 multimodal_image_limit=2 enforce_eager=true language_model_only=false VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER_VALUE}"
  exec env \
    -u VLLM_BIN \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER_VALUE}" \
    DO_NOT_TRACK=1 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "${VLLM_BIN}" "${VLLM_SERVER_ARGS[@]}"
}

gateway_process() {
  mkdir -p "${LOG_ROOT}"
  exec >>"${GATEWAY_LOG}" 2>&1
  echo "[qwen35-gateway] started_at=$(date --iso-8601=seconds) profile=${SERVICE_PROFILE} engine_base_url=${ENGINE_BASE_URL}/v1 gateway=${GATEWAY_BASE_URL} context_limit=${CONTEXT_LIMIT} output_limit=${OUTPUT_LIMIT} planner_output_limit=${PLANNER_OUTPUT_LIMIT} planner_prompt_mode=rendered_system_once concurrency=${GATEWAY_CONCURRENCY} retries=0 fallback=false"
  exec env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OPENAI_API_KEY=EMPTY \
    PYTHONPATH="${REPO_ROOT}" \
    "${RMBENCH_PYTHON}" "${REPO_ROOT}/policy/roboharn_evo/scripts/serve_openai_planner.py" \
      --host "${GATEWAY_HOST}" \
      --port "${GATEWAY_PORT}" \
      --backend openai \
      --local-only \
      --api-mode chat \
      --base-url "${ENGINE_BASE_URL}/v1" \
      --provider-name local-vllm \
      --config-file "" \
      --model "${MODEL_ID}" \
      --reasoning-effort "" \
      --thinking-mode enabled \
      --temperature 0.6 \
      --top-p 0.95 \
      --top-k 20 \
      --min-p 0.0 \
      --max-output-tokens "${OUTPUT_LIMIT}" \
      --planner-max-output-tokens "${PLANNER_OUTPUT_LIMIT}" \
      --planner-prompt-mode rendered_system_once \
      --timeout-sec 600 \
      --max-concurrent-requests "${GATEWAY_CONCURRENCY}" \
      --request-queue-timeout-sec 5 \
      --max-retries 0 \
      --disable-response-storage \
      --model-source "ModelScope official Qwen/Qwen3.5-397B-A17B-FP8" \
      --model-revision "master (immutable revision not locally recorded)" \
      --quantization official_fp8_block128 \
      --serving-framework "${VLLM_VERSION}" \
      --context-limit "${CONTEXT_LIMIT}" \
      --model-artifact-path "${MODEL_DIR}" \
      --model-artifact-manifest-path "${MODEL_IDENTITY_MANIFEST}" \
      --model-artifact-manifest-sha256 "${MODEL_MANIFEST_SHA256}" \
      --agent-contract-path "${AGENT_CONTRACT_MANIFEST}" \
      --agent-contract-manifest-sha256 "${AGENT_CONTRACT_MANIFEST_SHA256}" \
      --serving-runtime-identity-path "${SERVING_RUNTIME_IDENTITY}" \
      --serving-runtime-identity-sha256 "${SERVING_RUNTIME_IDENTITY_SHA256}"
}

start_services() {
  START_ENGINE_SESSION_OWNED=0
  START_GATEWAY_SESSION_OWNED=0
  trap 'cleanup_owned_start_sessions_on_signal SIGINT 130' INT
  trap 'cleanup_owned_start_sessions_on_signal SIGTERM 143' TERM
  validate_configuration
  for executable in tmux curl sha256sum "${RMBENCH_PYTHON}" "${VLLM_PYTHON}" "${VLLM_BIN}"; do
    if [[ "${executable}" == */* ]]; then
      [[ -x "${executable}" ]] || { echo "Missing executable: ${executable}" >&2; exit 2; }
    else
      command -v "${executable}" >/dev/null 2>&1 || { echo "Missing command: ${executable}" >&2; exit 2; }
    fi
  done
  [[ -d "${MODEL_DIR}" ]] || { echo "Missing model directory: ${MODEL_DIR}" >&2; exit 2; }
  [[ -f "${RUNTIME_IDENTITY_RECORDER}" ]] || { echo "Missing runtime identity recorder: ${RUNTIME_IDENTITY_RECORDER}" >&2; exit 2; }
  [[ -f "${MODEL_IDENTITY_MANIFEST}" ]] || { echo "Missing model identity manifest: ${MODEL_IDENTITY_MANIFEST}" >&2; exit 2; }
  [[ -f "${AGENT_CONTRACT_MANIFEST}" ]] || { echo "Missing Agent contract manifest: ${AGENT_CONTRACT_MANIFEST}" >&2; exit 2; }
  if session_exists_exact "${ENGINE_SESSION}" || session_exists_exact "${GATEWAY_SESSION}"; then
    echo "Refusing to overwrite an existing Qwen session; run status first." >&2
    exit 2
  fi
  require_port_free vLLM "${ENGINE_HOST}" "${ENGINE_PORT}"
  require_port_free gateway "${GATEWAY_HOST}" "${GATEWAY_PORT}"

  mkdir -p "${LOG_ROOT}"
  capture_gpt9104_identity \
    "${LOG_ROOT}/gpt9104_health_before.json" \
    "${LOG_ROOT}/gpt9104_identity_before.json"
  validate_model_identity_manifest
  validate_agent_contract_manifest
  MODEL_MANIFEST_SHA256="$(sha256sum "${MODEL_IDENTITY_MANIFEST}" | awk '{print $1}')"
  AGENT_CONTRACT_MANIFEST_SHA256="$(sha256sum "${AGENT_CONTRACT_MANIFEST}" | awk '{print $1}')"
  # Fail closed before creating either tmux session. This records the exact
  # argument array later consumed by engine_process; no model is started here.
  record_serving_runtime_identity
  validate_recorded_vllm_binary
  SERVING_RUNTIME_IDENTITY_SHA256="$(sha256sum "${SERVING_RUNTIME_IDENTITY}" | awk '{print $1}')"
  VLLM_VERSION="$(read_recorded_vllm_version)"
  export MODEL_MANIFEST_SHA256 AGENT_CONTRACT_MANIFEST_SHA256 VLLM_VERSION
  export SERVING_RUNTIME_IDENTITY_SHA256
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits >"${LOG_ROOT}/gpu_memory_before_start.csv" 2>&1 || true

  cleanup_start_failure() {
    stop_session_with_ctrl_c "${GATEWAY_SESSION}" gateway 60 || true
    stop_session_with_ctrl_c "${ENGINE_SESSION}" engine 180 || true
  }

  local command_string
  printf -v command_string '%q ' env \
    REPO_ROOT="${REPO_ROOT}" VLLM_BIN="${VLLM_BIN}" \
    QWEN_SERVICE_PROFILE="${SERVICE_PROFILE}" \
    QWEN_SERVICE_LOG_ROOT="${LOG_ROOT}" QWEN_ENGINE_LOG="${ENGINE_LOG}" \
    QWEN_ENGINE_PORT="${ENGINE_PORT}" QWEN_CONTEXT_LIMIT="${CONTEXT_LIMIT}" \
    QWEN_OUTPUT_LIMIT="${OUTPUT_LIMIT}" QWEN_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
    QWEN_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    "${SCRIPT_PATH}" _engine_process
  if ! tmux new-session -d -s "${ENGINE_SESSION}" -c "${REPO_ROOT}" "${command_string}"; then
    echo "Failed to create engine tmux session: ${ENGINE_SESSION}" >&2
    return 1
  fi
  if ! mark_session_owned "${ENGINE_SESSION}" engine; then
    echo "Failed to mark newly created engine session; requesting Ctrl+C." >&2
    send_ctrl_c_to_exact_session "${ENGINE_SESSION}" 2>/dev/null || true
    return 1
  fi
  START_ENGINE_SESSION_OWNED=1
  echo "Started engine session ${ENGINE_SESSION}; waiting for ${ENGINE_BASE_URL}/v1/models"
  if ! wait_for_url vLLM "${ENGINE_BASE_URL}/v1/models" "${ENGINE_START_TIMEOUT_SEC}" "${ENGINE_SESSION}" engine; then
    tail -n 80 "${ENGINE_LOG}" 2>/dev/null || true
    cleanup_start_failure
    return 1
  fi

  if ! health_to_file "${ENGINE_BASE_URL}/v1/models" "${LOG_ROOT}/vllm_models.json"; then
    cleanup_start_failure
    return 1
  fi
  if ! "${RMBENCH_PYTHON}" -c '
import json, pathlib, sys
payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
model_ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
if sys.argv[2] not in model_ids:
    raise SystemExit(f"expected model {sys.argv[2]!r} not in upstream ids {model_ids!r}")
' "${LOG_ROOT}/vllm_models.json" "${MODEL_ID}"; then
    cleanup_start_failure
    return 1
  fi
  UPSTREAM_MODELS_SHA256="$(sha256sum "${LOG_ROOT}/vllm_models.json" | awk '{print $1}')"
  export UPSTREAM_MODELS_SHA256

  printf -v command_string '%q ' env \
    REPO_ROOT="${REPO_ROOT}" RMBENCH_PYTHON="${RMBENCH_PYTHON}" \
    MODEL_IDENTITY_MANIFEST="${MODEL_IDENTITY_MANIFEST}" \
    AGENT_CONTRACT_MANIFEST="${AGENT_CONTRACT_MANIFEST}" \
    QWEN_SERVICE_PROFILE="${SERVICE_PROFILE}" \
    QWEN_SERVICE_LOG_ROOT="${LOG_ROOT}" QWEN_GATEWAY_LOG="${GATEWAY_LOG}" \
    QWEN_ENGINE_PORT="${ENGINE_PORT}" QWEN_GATEWAY_PORT="${GATEWAY_PORT}" \
    QWEN_CONTEXT_LIMIT="${CONTEXT_LIMIT}" QWEN_OUTPUT_LIMIT="${OUTPUT_LIMIT}" \
    QWEN_PLANNER_OUTPUT_LIMIT="${PLANNER_OUTPUT_LIMIT}" \
    QWEN_GATEWAY_CONCURRENCY="${GATEWAY_CONCURRENCY}" \
    MODEL_MANIFEST_SHA256="${MODEL_MANIFEST_SHA256}" \
    AGENT_CONTRACT_MANIFEST_SHA256="${AGENT_CONTRACT_MANIFEST_SHA256}" \
    SERVING_RUNTIME_IDENTITY_SHA256="${SERVING_RUNTIME_IDENTITY_SHA256}" \
    UPSTREAM_MODELS_SHA256="${UPSTREAM_MODELS_SHA256}" \
    VLLM_VERSION="${VLLM_VERSION}" \
    "${SCRIPT_PATH}" _gateway_process
  if ! tmux new-session -d -s "${GATEWAY_SESSION}" -c "${REPO_ROOT}" "${command_string}"; then
    echo "Failed to create gateway tmux session: ${GATEWAY_SESSION}" >&2
    cleanup_start_failure
    return 1
  fi
  if ! mark_session_owned "${GATEWAY_SESSION}" gateway; then
    echo "Failed to mark newly created gateway session; requesting Ctrl+C." >&2
    send_ctrl_c_to_exact_session "${GATEWAY_SESSION}" 2>/dev/null || true
    cleanup_start_failure
    return 1
  fi
  START_GATEWAY_SESSION_OWNED=1
  if ! wait_for_url gateway "${GATEWAY_BASE_URL}/health" "${GATEWAY_START_TIMEOUT_SEC}" "${GATEWAY_SESSION}" gateway; then
    tail -n 80 "${GATEWAY_LOG}" 2>/dev/null || true
    cleanup_start_failure
    return 1
  fi
  if ! health_to_file "${GATEWAY_BASE_URL}/health" "${LOG_ROOT}/gateway_health.json"; then
    cleanup_start_failure
    return 1
  fi
  if ! validate_qwen_gateway_health "${LOG_ROOT}/gateway_health.json"; then
    cleanup_start_failure
    return 1
  fi
  if ! capture_gpt9104_identity \
    "${LOG_ROOT}/gpt9104_health_after.json" \
    "${LOG_ROOT}/gpt9104_identity_after.json"; then
    cleanup_start_failure
    return 1
  fi
  if ! compare_gpt9104_identities \
    "${LOG_ROOT}/gpt9104_identity_before.json" \
    "${LOG_ROOT}/gpt9104_identity_after.json"; then
    cleanup_start_failure
    return 1
  fi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits >"${LOG_ROOT}/gpu_memory_after_start.csv" 2>&1 || true

  START_ENGINE_SESSION_OWNED=0
  START_GATEWAY_SESSION_OWNED=0
  trap - INT TERM

  echo "Qwen local full-stack services are healthy (profile=${SERVICE_PROFILE}, context=${CONTEXT_LIMIT}, output=${OUTPUT_LIMIT})."
  echo "vLLM: ${ENGINE_BASE_URL}/v1  tmux=${ENGINE_SESSION}"
  echo "Gateway: ${GATEWAY_BASE_URL}  tmux=${GATEWAY_SESSION}"
  echo "Logs and identity evidence: ${LOG_ROOT}"
  echo "GPT-5.5 9104 full canonical health and listener process identity matched before and after startup."
}

status_services() {
  local session_name
  local service_kind
  for service_kind in engine gateway; do
    if [[ "${service_kind}" == "engine" ]]; then
      session_name="${ENGINE_SESSION}"
    else
      session_name="${GATEWAY_SESSION}"
    fi
    if session_exists_exact "${session_name}"; then
      if session_is_owned "${session_name}" "${service_kind}"; then
        echo "tmux ${session_name}: running (owned by this launcher as ${service_kind})"
      else
        echo "tmux ${session_name}: running (NOT owned by this launcher)"
      fi
    else
      echo "tmux ${session_name}: not running"
    fi
  done
  curl --noproxy '*' --silent --show-error --max-time 5 "${ENGINE_BASE_URL}/v1/models" || true
  echo
  curl --noproxy '*' --silent --show-error --max-time 5 "${GATEWAY_BASE_URL}/health" || true
  echo
  curl --noproxy '*' --silent --show-error --max-time 5 "http://127.0.0.1:9104/health" || true
  echo
}

stop_services() {
  local status=0
  stop_session_with_ctrl_c "${GATEWAY_SESSION}" gateway 60 || status=1
  stop_session_with_ctrl_c "${ENGINE_SESSION}" engine 180 || status=1
  verify_service_ports_closed || status=1
  if (( status != 0 )); then
    echo "Qwen service stop was incomplete; preserved residual state for diagnosis." >&2
  fi
  return "${status}"
}

action="${1:-}"
case "${action}" in
  _engine_process)
    engine_process
    ;;
  _gateway_process)
    gateway_process
    ;;
  start)
    start_services
    ;;
  status)
    status_services
    ;;
  stop)
    stop_services
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
