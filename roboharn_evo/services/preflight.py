"""Static and HTTP preflight for the RMBench service contract."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import yaml

from roboharn_evo.services.http_safety import (
    LEGACY_RMBENCH_DONOR_ROOT,
    ROBOHARN_EVAL_RESULT_ROOT,
    ROBOHARN_PROJECT_ROOT,
    path_is_within,
    paths_overlap,
    resolved_path,
)


class ServiceContractError(ValueError):
    """Raised when a deploy config does not describe the expected services."""


DEFAULT_PI_OPENPI_REPOSITORY = (
    ROBOHARN_PROJECT_ROOT / "benchmarks" / "rmbench" / "policy" / "pi05"
).resolve()
DEFAULT_SEGMENTATION_ARTIFACT_DIR = (
    ROBOHARN_EVAL_RESULT_ROOT / "rmbench" / "runtime_workspace" / "roboharn_segmentation"
).resolve()
SEGMENTATION_ARTIFACT_DIR_ENV = "ROBOHARN_EVO_SEGMENTATION_ARTIFACT_DIR"


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    health_url: str
    business_urls: tuple[str, ...]
    health_equals: tuple[tuple[str, Any], ...] = ()
    health_nonempty: tuple[str, ...] = ()
    health_endpoint_set: tuple[str, ...] = ()


@dataclass(frozen=True)
class ServiceIdentityExpectations:
    """Exact runtime identities expected from the PI0.5 and SAM3 services."""

    pi_openpi_repo: Path
    pi_checkpoint_dir: Path | None
    segmentation_artifact_dir: Path

    @property
    def segmentation_input_root(self) -> Path:
        return self.segmentation_artifact_dir / "inputs"

    @property
    def segmentation_output_root(self) -> Path:
        return self.segmentation_artifact_dir / "masks"

    def as_report(self) -> dict[str, Any]:
        return {
            "pi05_executor": {
                "openpi_repo": str(self.pi_openpi_repo),
                "checkpoint_dir": (
                    str(self.pi_checkpoint_dir)
                    if self.pi_checkpoint_dir is not None
                    else None
                ),
            },
            "sam3": {
                "segmentation_artifact_dir": str(self.segmentation_artifact_dir),
                "allowed_input_roots": [str(self.segmentation_input_root)],
                "output_root": str(self.segmentation_output_root),
                "allowed_output_root": str(self.segmentation_output_root),
            },
        }


def _identity_expectations(
    *,
    segmentation_artifact_dir: str | Path | None = None,
    pi_openpi_repo: str | Path | None = None,
    pi_checkpoint_dir: str | Path | None = None,
) -> ServiceIdentityExpectations:
    """Resolve probe expectations without creating any filesystem entries."""

    selected_openpi_repo = resolved_path(pi_openpi_repo or DEFAULT_PI_OPENPI_REPOSITORY)
    if selected_openpi_repo != DEFAULT_PI_OPENPI_REPOSITORY:
        raise ServiceContractError(
            "pi_openpi_repo must identify the code-only OpenPI copy in this RoboHarn-Evo "
            f"checkout: {selected_openpi_repo}, expected "
            f"{DEFAULT_PI_OPENPI_REPOSITORY}"
        )

    raw_artifact_dir = segmentation_artifact_dir
    if raw_artifact_dir is None:
        raw_artifact_dir = os.environ.get(SEGMENTATION_ARTIFACT_DIR_ENV, "").strip()
    artifact_dir = resolved_path(raw_artifact_dir or DEFAULT_SEGMENTATION_ARTIFACT_DIR)
    if not path_is_within(artifact_dir, ROBOHARN_EVAL_RESULT_ROOT):
        raise ServiceContractError(
            "segmentation_artifact_dir must resolve inside the RoboHarn-Evo eval_result "
            f"boundary: {artifact_dir}, expected root {ROBOHARN_EVAL_RESULT_ROOT.resolve()}"
        )

    checkpoint_dir = (
        resolved_path(pi_checkpoint_dir) if pi_checkpoint_dir is not None else None
    )
    return ServiceIdentityExpectations(
        pi_openpi_repo=selected_openpi_repo,
        pi_checkpoint_dir=checkpoint_dir,
        segmentation_artifact_dir=artifact_dir,
    )


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ServiceContractError(f"missing config key: {'.'.join(keys)}")
        current = current[key]
    return current


def _parse_http_url(value: Any, *, label: str) -> SplitResult:
    raw = str(value or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ServiceContractError(f"{label} must be an absolute HTTP(S) URL: {raw!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ServiceContractError(f"{label} must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ServiceContractError(f"{label} must not contain a query or fragment")
    return parsed


def _origin(parsed: SplitResult) -> tuple[str, str, int | None]:
    return parsed.scheme, str(parsed.hostname), parsed.port


def _origin_url(parsed: SplitResult) -> str:
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")


def _require_path(parsed: SplitResult, expected: str, *, label: str) -> None:
    actual = parsed.path.rstrip("/") or "/"
    if actual != expected:
        raise ServiceContractError(
            f"{label} must use path {expected!r}, got {actual!r}"
        )


def load_rmbench_service_specs(config_path: str | Path) -> tuple[ServiceSpec, ...]:
    """Load and validate the service URLs required by ``deploy_policy.yml``."""

    path = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ServiceContractError(f"deploy config must contain a mapping: {path}")

    agent_entries = (
        (
            "perception query",
            _nested(payload, "agent", "observation_preprocess", "query_url"),
            "/perception_queries",
        ),
        ("planner", _nested(payload, "planner", "agent_api", "server_url"), "/plan"),
        ("OOD", _nested(payload, "ood", "agent_api", "server_url"), "/ood"),
        (
            "recovery",
            _nested(payload, "recovery", "agent_api", "server_url"),
            "/recover",
        ),
    )
    parsed_agent: list[SplitResult] = []
    agent_urls: list[str] = []
    for label, raw_url, expected_path in agent_entries:
        parsed = _parse_http_url(raw_url, label=label)
        _require_path(parsed, expected_path, label=label)
        parsed_agent.append(parsed)
        agent_urls.append(str(raw_url))
    origins = {_origin(parsed) for parsed in parsed_agent}
    if len(origins) != 1:
        raise ServiceContractError(
            "planner, OOD, recovery, and perception query URLs must share one Agent API origin"
        )
    agent_origin = _origin_url(parsed_agent[0])
    agent_urls.append(f"{agent_origin}/normalize_perception_queries")
    agent_urls.append(f"{agent_origin}/hpk_strategy_proposal")

    executor_url = _nested(payload, "executor", "pi05_api", "server_url")
    executor_config = _nested(payload, "executor", "pi05_api")
    parsed_executor = _parse_http_url(executor_url, label="PI0.5 executor")
    _require_path(parsed_executor, "/act", label="PI0.5 executor")

    sam3_url = _nested(
        payload,
        "agent",
        "observation_preprocess",
        "segmentation",
        "service_url",
    )
    parsed_sam3 = _parse_http_url(sam3_url, label="SAM3 service")
    _require_path(parsed_sam3, "/", label="SAM3 service")
    sam3_origin = _origin_url(parsed_sam3)

    return (
        ServiceSpec(
            name="agent_api",
            health_url=f"{agent_origin}/health",
            business_urls=tuple(agent_urls),
            health_nonempty=("backend", "provider", "model"),
            health_endpoint_set=tuple(urlsplit(url).path for url in agent_urls),
        ),
        ServiceSpec(
            name="pi05_executor",
            health_url=f"{_origin_url(parsed_executor)}/health",
            business_urls=(str(executor_url),),
            health_equals=(
                ("backend", "pi05"),
                ("action_dim", int(executor_config["action_dim"])),
                ("pi0_step", int(executor_config["max_chunk_steps"])),
                ("config_name", str(executor_config["config_name"])),
                ("prompt_mode", str(executor_config["prompt_mode"])),
                ("asset_id", str(executor_config["asset_id"])),
            ),
            health_nonempty=(
                "config_name",
                "checkpoint_dir",
                "openpi_repo",
                "project_root",
                "cache_root",
                "protected_read_roots",
            ),
        ),
        ServiceSpec(
            name="sam3",
            health_url=f"{sam3_origin}/health",
            business_urls=(f"{sam3_origin}/segment_image",),
            health_equals=(("backend", "sam3_image_text_prompt"),),
            health_nonempty=(
                "sam3_repo",
                "checkpoint",
                "bpe_path",
                "project_root",
                "runtime_root",
                "output_root",
                "allowed_output_root",
                "allowed_input_roots",
                "cache_root",
                "protected_read_roots",
            ),
        ),
    )


def _sam3_health_identity_errors(
    payload: Mapping[str, Any],
    *,
    expectations: ServiceIdentityExpectations,
) -> list[str]:
    """Validate SAM3's write roots independently from its startup process."""

    errors: list[str] = []
    expected_project_root = ROBOHARN_PROJECT_ROOT.resolve()
    expected_runtime_root = ROBOHARN_EVAL_RESULT_ROOT.resolve()
    reported_project_root = resolved_path(str(payload.get("project_root", "")))
    if reported_project_root != expected_project_root:
        errors.append(
            "project_root does not identify this RoboHarn-Evo checkout: "
            f"{reported_project_root}, expected {expected_project_root}"
        )
    reported_runtime_root = resolved_path(str(payload.get("runtime_root", "")))
    if reported_runtime_root != expected_runtime_root:
        errors.append(
            "runtime_root does not identify this RoboHarn-Evo eval_result boundary: "
            f"{reported_runtime_root}, expected {expected_runtime_root}"
        )

    root_values: dict[str, Path] = {}
    for key in ("output_root", "allowed_output_root", "cache_root"):
        raw_value = payload.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        path = resolved_path(raw_value)
        root_values[key] = path
        if not path_is_within(path, expected_runtime_root):
            errors.append(f"{key} is outside the RoboHarn-Evo eval_result boundary: {path}")

    output_root = root_values.get("output_root")
    allowed_output_root = root_values.get("allowed_output_root")
    cache_root = root_values.get("cache_root")
    expected_output_root = expectations.segmentation_output_root
    if output_root is not None and output_root != expected_output_root:
        errors.append(
            "output_root does not match AgentTools' segmentation mask root: "
            f"{output_root}, expected {expected_output_root}"
        )
    if allowed_output_root is not None and allowed_output_root != expected_output_root:
        errors.append(
            "allowed_output_root does not match AgentTools' segmentation mask "
            f"root: {allowed_output_root}, expected {expected_output_root}"
        )
    if (
        output_root is not None
        and allowed_output_root is not None
        and not path_is_within(output_root, allowed_output_root)
    ):
        errors.append("output_root is outside allowed_output_root")
    if (
        cache_root is not None
        and allowed_output_root is not None
        and paths_overlap(cache_root, allowed_output_root)
    ):
        errors.append("cache_root overlaps allowed_output_root")

    raw_inputs = payload.get("allowed_input_roots")
    input_roots = (
        [resolved_path(str(value)) for value in raw_inputs]
        if isinstance(raw_inputs, Sequence) and not isinstance(raw_inputs, (str, bytes))
        else []
    )
    if raw_inputs and not input_roots:
        errors.append("allowed_input_roots is malformed")
    expected_input_root = expectations.segmentation_input_root
    if input_roots != [expected_input_root]:
        if expected_input_root not in input_roots:
            errors.append(
                "allowed_input_roots omits AgentTools' segmentation input root: "
                f"{expected_input_root}"
            )
        else:
            errors.append(
                "allowed_input_roots must exactly match AgentTools' segmentation "
                f"input root: {[expected_input_root]}"
            )
    for input_root in input_roots:
        for key in ("output_root", "allowed_output_root", "cache_root"):
            write_root = root_values.get(key)
            if write_root is not None and paths_overlap(input_root, write_root):
                errors.append(f"allowed_input_root overlaps {key}: {input_root}")

    raw_protected = payload.get("protected_read_roots")
    protected_roots = (
        [resolved_path(str(value)) for value in raw_protected]
        if isinstance(raw_protected, Sequence)
        and not isinstance(raw_protected, (str, bytes))
        else []
    )
    if raw_protected and not protected_roots:
        errors.append("protected_read_roots is malformed")
    if LEGACY_RMBENCH_DONOR_ROOT not in protected_roots:
        errors.append("protected_read_roots omits the legacy RMBench donor")
    for protected_root in protected_roots:
        for key, write_root in root_values.items():
            if paths_overlap(protected_root, write_root):
                errors.append(f"{key} overlaps protected read root: {protected_root}")
    return errors


def _pi05_health_identity_errors(
    payload: Mapping[str, Any],
    *,
    expectations: ServiceIdentityExpectations,
) -> list[str]:
    """Validate PI0.5 source and cache boundaries from reported identity."""

    errors: list[str] = []
    expected_project_root = ROBOHARN_PROJECT_ROOT.resolve()
    expected_runtime_root = ROBOHARN_EVAL_RESULT_ROOT.resolve()
    reported_project_root = resolved_path(str(payload.get("project_root", "")))
    if reported_project_root != expected_project_root:
        errors.append(
            "project_root does not identify this RoboHarn-Evo checkout: "
            f"{reported_project_root}, expected {expected_project_root}"
        )

    cache_root = resolved_path(str(payload.get("cache_root", "")))
    if not path_is_within(cache_root, expected_runtime_root):
        errors.append(
            f"cache_root is outside the RoboHarn-Evo eval_result boundary: {cache_root}"
        )

    openpi_repo = resolved_path(str(payload.get("openpi_repo", "")))
    checkpoint_dir = resolved_path(str(payload.get("checkpoint_dir", "")))
    if openpi_repo != expectations.pi_openpi_repo:
        errors.append(
            "openpi_repo does not identify the code-only OpenPI copy in this RoboHarn-Evo "
            f"checkout: {openpi_repo}, expected {expectations.pi_openpi_repo}"
        )
    if (
        expectations.pi_checkpoint_dir is not None
        and checkpoint_dir != expectations.pi_checkpoint_dir
    ):
        errors.append(
            "checkpoint_dir does not match the operator-selected read-only "
            f"checkpoint: {checkpoint_dir}, expected "
            f"{expectations.pi_checkpoint_dir}"
        )
    if path_is_within(openpi_repo, LEGACY_RMBENCH_DONOR_ROOT):
        errors.append("openpi_repo uses the legacy RMBench donor as execution source")

    raw_protected = payload.get("protected_read_roots")
    protected_roots = (
        [resolved_path(str(value)) for value in raw_protected]
        if isinstance(raw_protected, Sequence)
        and not isinstance(raw_protected, (str, bytes))
        else []
    )
    if raw_protected and not protected_roots:
        errors.append("protected_read_roots is malformed")
    for required_label, required_root in (
        ("legacy RMBench donor", LEGACY_RMBENCH_DONOR_ROOT),
        ("OpenPI source", openpi_repo),
        ("checkpoint", checkpoint_dir),
    ):
        if required_root not in protected_roots:
            errors.append(
                f"protected_read_roots omits {required_label}: {required_root}"
            )

    protected_boundaries = [("copied OpenPI source", DEFAULT_PI_OPENPI_REPOSITORY)]
    protected_boundaries.extend(
        ("protected read root", root) for root in protected_roots
    )
    for label, protected_root in protected_boundaries:
        if paths_overlap(cache_root, protected_root):
            errors.append(f"cache_root overlaps {label}: {protected_root}")
    return errors


def static_service_report(
    config_path: str | Path,
    *,
    segmentation_artifact_dir: str | Path | None = None,
    pi_openpi_repo: str | Path | None = None,
    pi_checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate URLs and resolve exact identities without network or writes."""

    path = Path(config_path).expanduser().resolve()
    specs = load_rmbench_service_specs(path)
    expectations = _identity_expectations(
        segmentation_artifact_dir=segmentation_artifact_dir,
        pi_openpi_repo=pi_openpi_repo,
        pi_checkpoint_dir=pi_checkpoint_dir,
    )
    return {
        "config": str(path),
        "expected_identity": expectations.as_report(),
        "mode": "validate-only",
        "services": [asdict(spec) for spec in specs],
        "valid": True,
    }


def probe_rmbench_services(
    config_path: str | Path,
    *,
    timeout_sec: float = 3.0,
    opener: Callable[..., Any] = urlopen,
    segmentation_artifact_dir: str | Path | None = None,
    pi_openpi_repo: str | Path | None = None,
    pi_checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    """GET every ``/health`` endpoint without invoking business operations."""

    path = Path(config_path).expanduser().resolve()
    expectations = _identity_expectations(
        segmentation_artifact_dir=segmentation_artifact_dir,
        pi_openpi_repo=pi_openpi_repo,
        pi_checkpoint_dir=pi_checkpoint_dir,
    )
    results: list[dict[str, Any]] = []
    for spec in load_rmbench_service_specs(path):
        result: dict[str, Any] = {
            **asdict(spec),
            "healthy": False,
        }
        try:
            request = Request(spec.health_url, headers={"Accept": "application/json"})
            with opener(request, timeout=float(timeout_sec)) as response:
                response_status = getattr(response, "status", None)
                if response_status is None:
                    response_status = response.getcode()
                status_code = int(response_status)
                body = response.read()
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("health response is not a JSON object")
            service_status = str(payload.get("status", "")).strip().lower()
            identity_errors: list[str] = []
            for key, expected in spec.health_equals:
                if payload.get(key) != expected:
                    identity_errors.append(
                        f"{key}={payload.get(key)!r}, expected {expected!r}"
                    )
            for key in spec.health_nonempty:
                if not payload.get(key):
                    identity_errors.append(f"{key} is empty")
            if spec.name == "pi05_executor":
                identity_errors.extend(
                    _pi05_health_identity_errors(
                        payload,
                        expectations=expectations,
                    )
                )
            if spec.name == "sam3":
                identity_errors.extend(
                    _sam3_health_identity_errors(
                        payload,
                        expectations=expectations,
                    )
                )
            if spec.health_endpoint_set:
                actual_endpoints = payload.get("business_endpoints")
                if not isinstance(actual_endpoints, Sequence) or isinstance(
                    actual_endpoints, (str, bytes)
                ):
                    identity_errors.append("business_endpoints is missing")
                elif set(map(str, actual_endpoints)) != set(spec.health_endpoint_set):
                    identity_errors.append(
                        "business_endpoints do not match deploy contract"
                    )
            result.update(
                {
                    "http_status": status_code,
                    "service_status": service_status,
                    "healthy": (
                        status_code == 200
                        and service_status == "ok"
                        and not identity_errors
                    ),
                }
            )
            if not result["healthy"]:
                result["error"] = "; ".join(identity_errors) or (
                    "health endpoint did not report status=ok"
                )
        except Exception as exc:  # network and malformed responses are report data
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return {
        "config": str(path),
        "expected_identity": expectations.as_report(),
        "mode": "http-health",
        "healthy": all(bool(item["healthy"]) for item in results),
        "services": results,
    }


__all__ = [
    "DEFAULT_PI_OPENPI_REPOSITORY",
    "DEFAULT_SEGMENTATION_ARTIFACT_DIR",
    "SEGMENTATION_ARTIFACT_DIR_ENV",
    "ServiceContractError",
    "ServiceIdentityExpectations",
    "ServiceSpec",
    "load_rmbench_service_specs",
    "probe_rmbench_services",
    "static_service_report",
]
