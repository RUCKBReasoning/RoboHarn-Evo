#!/usr/bin/env python3
"""Record a secret-free identity for a local OpenAI-compatible serving runtime.

The target interpreter is inspected in an isolated subprocess.  This script
does not import the serving stack into its own process and never starts a model
server.  A manifest is only installed after every required probe succeeds.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


SCHEMA_VERSION = 2
REQUIRED_DISTRIBUTIONS = ("vllm", "torch", "transformers", "openai")
DEFAULT_PYTHON_PROBE_TIMEOUT_SECONDS = 300
DEFAULT_VLLM_VERSION_TIMEOUT_SECONDS = 300
NVIDIA_SMI_TIMEOUT_SECONDS = 30
MAX_PROBE_TIMEOUT_SECONDS = 1800


class RuntimeIdentityError(ValueError):
    """Raised when a serving runtime cannot be identified unambiguously."""


_PYTHON_PROBE = r"""
import importlib.metadata as metadata
import hashlib
import json
import os
import platform
import shutil
import sys
import sysconfig

required = ("vllm", "torch", "transformers", "openai")
installed = []
for lookup_name in required:
    distribution = metadata.distribution(lookup_name)
    exact_name = distribution.metadata.get("Name")
    exact_version = distribution.version
    if not isinstance(exact_name, str) or not exact_name.strip():
        raise RuntimeError("distribution has no Name metadata: " + lookup_name)
    if not isinstance(exact_version, str) or not exact_version.strip():
        raise RuntimeError("distribution has no version metadata: " + lookup_name)
    installed.append({
        "lookup_name": lookup_name,
        "name": exact_name.strip(),
        "version": exact_version.strip(),
    })

import torch

scripts_dir = sysconfig.get_path("scripts")
vllm_executable = shutil.which("vllm", path=scripts_dir)
if vllm_executable is None:
    raise RuntimeError("the vllm console executable is absent from " + str(scripts_dir))
vllm_executable = os.path.realpath(vllm_executable)
vllm_digest = hashlib.sha256()
with open(vllm_executable, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        vllm_digest.update(chunk)
payload = {
    "python": {
        "executable": os.path.abspath(sys.executable),
        "real_executable": os.path.realpath(sys.executable),
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:5]),
    },
    "distributions": installed,
    "vllm": {
        "executable": vllm_executable,
        "executable_sha256": vllm_digest.hexdigest(),
    },
    "torch": {
        "cuda_build_version": torch.version.cuda,
    },
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""


_FORBIDDEN_SERVER_ARGUMENT_NAMES = {
    "api-key",
    "apikey",
    "auth-token",
    "authorization",
    "bearer-token",
    "cookie",
    "env",
    "env-file",
    "env-var",
    "header",
    "header-file",
    "headers",
    "hf-token",
    "password",
    "refresh-token",
    "secret",
    "token",
    "access-token",
}
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(?:authorization|api[-_]?key|auth[-_]?token|access[-_]?token|"
    r"refresh[-_]?token|bearer[-_]?token|hf[-_]?token|password|secret|cookie)\s*[:=]"
)
_OBVIOUS_TOKEN_VALUE = re.compile(r"(?i)^(?:bearer\s+\S+|sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,})$")
_CANONICAL_DISTRIBUTION_SEPARATOR = re.compile(r"[-_.]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_FIELD_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "bearer_token",
}

# Runtime environment variables are not accepted generically: doing so would
# make it too easy to persist credentials or unrelated host state.  Only
# non-secret settings that materially select a serving implementation may be
# recorded here, with an explicit value allowlist.
_ALLOWED_RUNTIME_SETTINGS = {
    "VLLM_USE_FLASHINFER_SAMPLER": frozenset({"0", "1"}),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe a local serving Python environment and atomically record its "
            "package, CUDA, GPU, and server-argument identity without launching a model."
        )
    )
    parser.add_argument("--python", type=Path, required=True, help="Target Python executable.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python-probe-timeout-sec",
        type=int,
        default=DEFAULT_PYTHON_PROBE_TIMEOUT_SECONDS,
        help=(
            "Bound for package/Python/CUDA identity probing; range 1-1800 seconds "
            f"(default: {DEFAULT_PYTHON_PROBE_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--vllm-version-timeout-sec",
        type=int,
        default=DEFAULT_VLLM_VERSION_TIMEOUT_SECONDS,
        help=(
            "Bound for the separate vllm --version cold-start probe; range 1-1800 "
            f"seconds (default: {DEFAULT_VLLM_VERSION_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help=(
            "One non-secret model-server argument, retained in order. For values "
            "beginning with '-', use --server-arg=--argument. Repeat as needed."
        ),
    )
    parser.add_argument(
        "--runtime-setting",
        action="append",
        default=[],
        help=(
            "One allowlisted, non-secret serving runtime setting in NAME=VALUE "
            "form. Repeat as needed; arbitrary environment variables are rejected."
        ),
    )
    return parser.parse_args(argv)


def _canonical_distribution_name(value: str) -> str:
    return _CANONICAL_DISTRIBUTION_SEPARATOR.sub("-", value).lower()


def _resolve_executable(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise RuntimeIdentityError("--python must be an absolute executable path")
    # Preserve a virtual environment's Python symlink: resolving it before
    # execution would silently inspect the base interpreter instead.
    absolute = Path(os.path.abspath(candidate))
    try:
        absolute.stat()
    except OSError as exc:
        raise RuntimeIdentityError(f"target Python executable does not exist: {candidate}") from exc
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise RuntimeIdentityError(f"target Python is not an executable file: {absolute}")
    return absolute


def _bounded_timeout(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeIdentityError(f"{label} must be an integer")
    if value < 1 or value > MAX_PROBE_TIMEOUT_SECONDS:
        raise RuntimeIdentityError(
            f"{label} must be between 1 and {MAX_PROBE_TIMEOUT_SECONDS} seconds"
        )
    return value


def _run_checked(
    command: Sequence[str], *, label: str, timeout_seconds: int
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeIdentityError(f"{label} timed out after {timeout_seconds} seconds") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeIdentityError(f"{label} could not be executed") from exc
    if result.returncode != 0:
        # Deliberately do not echo stdout/stderr: third-party startup diagnostics
        # can include environment values which must never enter this manifest.
        raise RuntimeIdentityError(f"{label} failed with exit code {result.returncode}")
    return result


def _json_object_no_duplicates(raw: str, *, label: str) -> dict[str, Any]:
    def object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeIdentityError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=object_pairs_hook)
    except RuntimeIdentityError:
        raise
    except json.JSONDecodeError as exc:
        raise RuntimeIdentityError(f"{label} did not return one valid JSON value") from exc
    if not isinstance(value, dict):
        raise RuntimeIdentityError(f"{label} must return a JSON object")
    return value


def _required_string(value: object, *, label: str, max_length: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeIdentityError(f"{label} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > max_length or "\x00" in normalized:
        raise RuntimeIdentityError(f"{label} is invalid or unexpectedly long")
    return normalized


def _validate_python_probe(payload: dict[str, Any], requested_python: Path) -> dict[str, Any]:
    python = payload.get("python")
    if not isinstance(python, dict):
        raise RuntimeIdentityError("Python probe omitted python identity")
    executable = Path(_required_string(python.get("executable"), label="python.executable"))
    if not executable.is_absolute():
        raise RuntimeIdentityError("python.executable returned by the probe is not absolute")
    if executable != requested_python:
        raise RuntimeIdentityError("target Python reported a different executable identity")
    real_executable = Path(
        _required_string(python.get("real_executable"), label="python.real_executable")
    )
    if not real_executable.is_absolute() or real_executable != requested_python.resolve():
        raise RuntimeIdentityError("target Python reported an inconsistent real executable identity")
    implementation = _required_string(
        python.get("implementation"), label="python.implementation", max_length=100
    )
    version = _required_string(python.get("version"), label="python.version", max_length=100)
    version_info = python.get("version_info")
    if (
        not isinstance(version_info, list)
        or len(version_info) != 5
        or any(isinstance(item, bool) or not isinstance(item, (int, str)) for item in version_info)
    ):
        raise RuntimeIdentityError("python.version_info is malformed")

    distributions = payload.get("distributions")
    if not isinstance(distributions, list):
        raise RuntimeIdentityError("Python probe omitted distribution identities")
    installed: list[dict[str, str]] = []
    observed: set[str] = set()
    for index, item in enumerate(distributions):
        if not isinstance(item, dict):
            raise RuntimeIdentityError(f"distributions[{index}] is not an object")
        lookup_name = _required_string(
            item.get("lookup_name"), label=f"distributions[{index}].lookup_name", max_length=100
        )
        name = _required_string(item.get("name"), label=f"distributions[{index}].name", max_length=200)
        distribution_version = _required_string(
            item.get("version"), label=f"distributions[{index}].version", max_length=200
        )
        canonical_lookup = _canonical_distribution_name(lookup_name)
        if canonical_lookup in observed:
            raise RuntimeIdentityError(f"duplicate distribution identity: {lookup_name}")
        if _canonical_distribution_name(name) != canonical_lookup:
            raise RuntimeIdentityError(
                f"distribution metadata name does not match lookup name: {lookup_name!r} vs {name!r}"
            )
        observed.add(canonical_lookup)
        installed.append(
            {"lookup_name": lookup_name, "name": name, "version": distribution_version}
        )
    expected = {_canonical_distribution_name(name) for name in REQUIRED_DISTRIBUTIONS}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeIdentityError(
            f"distribution identity set mismatch; missing={missing}, unexpected={extra}"
        )

    vllm = payload.get("vllm")
    if not isinstance(vllm, dict):
        raise RuntimeIdentityError("Python probe omitted vLLM CLI identity")
    vllm_executable = Path(
        _required_string(vllm.get("executable"), label="vllm.executable")
    )
    if not vllm_executable.is_absolute():
        raise RuntimeIdentityError("vllm.executable returned by the probe is not absolute")
    vllm_executable_sha256 = _required_string(
        vllm.get("executable_sha256"),
        label="vllm.executable_sha256",
        max_length=64,
    )
    if not _SHA256.fullmatch(vllm_executable_sha256):
        raise RuntimeIdentityError("vllm.executable_sha256 must be a lowercase SHA256")

    torch = payload.get("torch")
    if not isinstance(torch, dict) or "cuda_build_version" not in torch:
        raise RuntimeIdentityError("Python probe omitted torch CUDA build identity")
    cuda_build_version = torch["cuda_build_version"]
    if cuda_build_version is not None:
        cuda_build_version = _required_string(
            cuda_build_version, label="torch.cuda_build_version", max_length=100
        )

    return {
        "python": {
            "requested_executable": str(requested_python),
            "executable": str(executable),
            "real_executable": str(real_executable),
            "implementation": implementation,
            "version": version,
            "version_info": version_info,
        },
        "distributions": {
            "required_lookup_names": list(REQUIRED_DISTRIBUTIONS),
            "installed": sorted(installed, key=lambda item: item["lookup_name"]),
        },
        "vllm": {
            "executable": str(vllm_executable),
            "executable_sha256": vllm_executable_sha256,
        },
        "torch": {"cuda_build_version": cuda_build_version},
    }


def _validate_server_args(server_args: Sequence[str]) -> list[str]:
    validated: list[str] = []
    for index, argument in enumerate(server_args):
        if not isinstance(argument, str) or not argument or len(argument) > 4096:
            raise RuntimeIdentityError(f"server argument {index} is empty or unexpectedly long")
        if any(character in argument for character in ("\x00", "\n", "\r")):
            raise RuntimeIdentityError(f"server argument {index} contains a forbidden control character")
        flag = argument.split("=", 1)[0]
        if flag.startswith("-"):
            normalized_flag = flag.lstrip("-").strip().lower().replace("_", "-")
            if normalized_flag in _FORBIDDEN_SERVER_ARGUMENT_NAMES:
                raise RuntimeIdentityError(
                    f"server argument {index} is credential-bearing and cannot be recorded: {flag}"
                )
        if _CREDENTIAL_ASSIGNMENT.search(argument) or _OBVIOUS_TOKEN_VALUE.fullmatch(argument):
            raise RuntimeIdentityError(
                f"server argument {index} appears to contain credential material"
            )
        validated.append(argument)
    return validated


def _validate_runtime_settings(runtime_settings: Sequence[str]) -> dict[str, str]:
    validated: dict[str, str] = {}
    for index, setting in enumerate(runtime_settings):
        if not isinstance(setting, str) or not setting or len(setting) > 256:
            raise RuntimeIdentityError(
                f"runtime setting {index} is empty or unexpectedly long"
            )
        if any(character in setting for character in ("\x00", "\n", "\r")):
            raise RuntimeIdentityError(
                f"runtime setting {index} contains a forbidden control character"
            )
        if setting.count("=") != 1:
            raise RuntimeIdentityError(
                f"runtime setting {index} must use exact NAME=VALUE form"
            )
        name, value = setting.split("=", 1)
        allowed_values = _ALLOWED_RUNTIME_SETTINGS.get(name)
        if allowed_values is None:
            raise RuntimeIdentityError(
                f"runtime setting {index} is not allowlisted: {name!r}"
            )
        if value not in allowed_values:
            raise RuntimeIdentityError(
                f"runtime setting {index} has an unsupported value for {name}"
            )
        if name in validated:
            raise RuntimeIdentityError(f"duplicate runtime setting: {name}")
        validated[name] = value
    return validated


def _assert_no_secret_fields(value: object, path: str = "identity") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_FIELD_NAMES or normalized.endswith(
                ("_api_key", "_password", "_secret", "_access_token", "_refresh_token")
            ):
                raise RuntimeIdentityError(f"forbidden credential-like field: {path}.{key}")
            _assert_no_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_secret_fields(child, f"{path}[{index}]")


def _vllm_version_output(raw: str) -> str:
    version = _required_string(raw, label="vllm.version_output", max_length=4096)
    if "\n" in version or "\r" in version:
        raise RuntimeIdentityError("vllm.version_output must be a single line")
    return version


def _probe_nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "executable_not_found"}
    executable_path = str(Path(executable).resolve())
    result = _run_checked(
        [
            executable_path,
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        label="nvidia-smi GPU identity probe",
        timeout_seconds=NVIDIA_SMI_TIMEOUT_SECONDS,
    )
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if not rows:
        raise RuntimeIdentityError("nvidia-smi GPU identity probe returned no GPUs")
    gpus: list[dict[str, Any]] = []
    indices: set[int] = set()
    uuids: set[str] = set()
    driver_versions: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 5:
            raise RuntimeIdentityError(f"malformed nvidia-smi row {row_number}")
        try:
            index = int(row[0].strip())
            memory_total_mib = int(row[3].strip())
        except ValueError as exc:
            raise RuntimeIdentityError(f"non-integer nvidia-smi field in row {row_number}") from exc
        uuid = _required_string(row[1], label=f"nvidia_smi.gpus[{row_number - 1}].uuid")
        name = _required_string(row[2], label=f"nvidia_smi.gpus[{row_number - 1}].name")
        driver = _required_string(
            row[4], label=f"nvidia_smi.gpus[{row_number - 1}].driver_version", max_length=100
        )
        if index < 0 or memory_total_mib <= 0 or index in indices or uuid in uuids:
            raise RuntimeIdentityError(f"invalid or duplicate nvidia-smi GPU row {row_number}")
        indices.add(index)
        uuids.add(uuid)
        driver_versions.add(driver)
        gpus.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_total_mib": memory_total_mib,
            }
        )
    if len(driver_versions) != 1:
        raise RuntimeIdentityError("nvidia-smi reported inconsistent driver versions")
    return {
        "available": True,
        "executable": executable_path,
        "driver_version": next(iter(driver_versions)),
        "gpus": sorted(gpus, key=lambda gpu: gpu["index"]),
    }


def build_local_serving_runtime_identity(
    *,
    python_executable: Path,
    server_args: Sequence[str],
    runtime_settings: Sequence[str] = (),
    python_probe_timeout_seconds: int = DEFAULT_PYTHON_PROBE_TIMEOUT_SECONDS,
    vllm_version_timeout_seconds: int = DEFAULT_VLLM_VERSION_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    python_timeout = _bounded_timeout(
        python_probe_timeout_seconds, label="python probe timeout"
    )
    vllm_timeout = _bounded_timeout(
        vllm_version_timeout_seconds, label="vLLM version timeout"
    )
    resolved_python = _resolve_executable(python_executable)
    validated_server_args = _validate_server_args(server_args)
    validated_runtime_settings = _validate_runtime_settings(runtime_settings)
    result = _run_checked(
        [str(resolved_python), "-I", "-c", _PYTHON_PROBE],
        label="target Python serving-runtime probe",
        timeout_seconds=python_timeout,
    )
    probe = _json_object_no_duplicates(result.stdout, label="target Python serving-runtime probe")
    identity = _validate_python_probe(probe, resolved_python)
    version_result = _run_checked(
        [identity["vllm"]["executable"], "--version"],
        label="vllm --version probe",
        timeout_seconds=vllm_timeout,
    )
    identity["vllm"]["version_output"] = _vllm_version_output(version_result.stdout)
    identity.update(
        {
            "schema_version": SCHEMA_VERSION,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "nvidia_smi": _probe_nvidia_smi(),
            "probe_timeouts_seconds": {
                "python_identity": python_timeout,
                "vllm_version": vllm_timeout,
                "nvidia_smi": NVIDIA_SMI_TIMEOUT_SECONDS,
            },
            "server_args": validated_server_args,
            "runtime_settings": validated_runtime_settings,
            "secrets_recorded": False,
            "model_server_started": False,
        }
    )
    return identity


def write_json_atomic(output: Path, payload: dict[str, Any]) -> None:
    _assert_no_secret_fields(payload)
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_local_serving_runtime_identity(
        python_executable=args.python,
        server_args=args.server_arg,
        runtime_settings=args.runtime_setting,
        python_probe_timeout_seconds=args.python_probe_timeout_sec,
        vllm_version_timeout_seconds=args.vllm_version_timeout_sec,
    )
    write_json_atomic(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
