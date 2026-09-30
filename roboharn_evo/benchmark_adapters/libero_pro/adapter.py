"""Real, simulator-free LIBERO-PRO environment adapter.

The adapter is intentionally duck typed.  Importing this module never imports
LIBERO, robosuite, MuJoCo, a renderer, or a benchmark task registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from roboharn_evo.benchmark_adapters.base import (
    ActionRequest,
    ActionResult,
    BenchmarkAdapter,
    EpisodeState,
    NeutralObservation,
)
from roboharn_evo.benchmark_adapters.libero_pro.contracts import (
    LiberoProActionContract,
    LiberoProCapabilities,
    LiberoProContractError,
    LiberoProObservationContract,
    RoboHarnCapabilityReport,
    assess_roboharn_agent_compatibility,
)


class LiberoProAdapter(BenchmarkAdapter[Mapping[str, Any]]):
    """Translate one canonical LIBERO environment without task policy.

    ``env.step(action)`` may return the canonical four-tuple
    ``(observation, reward, done, info)`` or the compatible five-tuple
    ``(observation, reward, terminated, truncated, info)``.  In both cases the
    public ``check_success`` method (or an explicitly supplied equivalent) is
    the sole authority for benchmark success.
    """

    def __init__(
        self,
        env: Any,
        *,
        instruction: str | None = None,
        instruction_provider: Callable[[], str] | None = None,
        success_checker: Callable[[], Any] | None = None,
        observation_contract: LiberoProObservationContract | None = None,
        action_contract: LiberoProActionContract | None = None,
        capabilities: LiberoProCapabilities | None = None,
        step_limit: int | None = None,
        initial_step_count: int = 0,
    ) -> None:
        step = getattr(env, "step", None)
        if not callable(step):
            raise LiberoProContractError(
                "LIBERO-PRO environment must expose step(action)"
            )
        if instruction is not None and not isinstance(instruction, str):
            raise LiberoProContractError("instruction must be a string when provided")
        if instruction_provider is not None and not callable(instruction_provider):
            raise LiberoProContractError("instruction_provider must be callable")

        checker = success_checker or getattr(env, "check_success", None)
        if not callable(checker):
            raise LiberoProContractError(
                "LIBERO-PRO requires the benchmark-authoritative check_success() "
                "method or an explicit success_checker"
            )

        self.env = env
        self._step = step
        self._success_checker = checker
        self._instruction = instruction
        self._instruction_provider = instruction_provider
        self.observation_contract = (
            observation_contract or LiberoProObservationContract()
        )

        if (
            capabilities is not None
            and action_contract is not None
            and capabilities.action != action_contract
        ):
            raise LiberoProContractError(
                "action_contract conflicts with capabilities.action"
            )
        resolved_action = (
            capabilities.action
            if capabilities is not None
            else action_contract or _action_contract_from_env(env)
        )
        self.capabilities = capabilities or LiberoProCapabilities(
            camera_keys=self.observation_contract.camera_keys,
            proprioception_keys=self.observation_contract.proprioception_keys,
            action=resolved_action,
        )

        if step_limit is None:
            step_limit = _optional_nonnegative_int(getattr(env, "horizon", 0))
        else:
            step_limit = _optional_nonnegative_int(step_limit)
        initial_step_count = _optional_nonnegative_int(initial_step_count)
        self._step_limit = step_limit
        self._step_count = initial_step_count
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._last_api_arity = 0
        self._last_capabilities = self.capabilities

    def reset(
        self,
        raw_observation: Mapping[str, Any] | None = None,
    ) -> NeutralObservation | None:
        """Reset adapter counters after the caller resets the environment.

        This method never calls ``env.reset``; environment ownership remains in
        the benchmark runner.
        """
        self._step_count = 0
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._last_api_arity = 0
        self._last_capabilities = self.capabilities
        if raw_observation is None:
            return None
        return self.to_observation(raw_observation)

    def begin_episode(
        self,
        raw_observation: Mapping[str, Any] | None = None,
    ) -> NeutralObservation | None:
        """Compatibility spelling for :meth:`reset`."""
        return self.reset(raw_observation)

    def to_observation(
        self,
        raw_observation: Mapping[str, Any],
    ) -> NeutralObservation:
        if not isinstance(raw_observation, Mapping):
            raise LiberoProContractError(
                "LIBERO-PRO observation must be a mapping; "
                f"got {type(raw_observation).__name__}"
            )

        camera_values = {
            key: deepcopy(raw_observation[key])
            for key in self.observation_contract.camera_keys
            if key in raw_observation
        }
        proprioception_values = {
            key: deepcopy(raw_observation[key])
            for key in self.observation_contract.proprioception_keys
            if key in raw_observation
        }
        observed_capabilities = self.capabilities.with_observed_fields(
            camera_keys=tuple(camera_values),
            proprioception_keys=tuple(proprioception_values),
        )
        self._last_capabilities = observed_capabilities
        return NeutralObservation(
            raw=self._copy_public_raw(raw_observation),
            instruction=self._read_instruction(raw_observation),
            cameras=camera_values,
            proprioception=proprioception_values,
            capabilities=observed_capabilities.to_metadata(),
        )

    def _copy_public_raw(
        self,
        raw_observation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Copy only explicitly public fields, excluding object-state GT.

        Canonical LIBERO observations may expose simulator-derived object
        positions, quaternions, and relative poses. An unknown-field copy
        would leak those oracle values to the Agent, so this boundary is an
        allowlist rather than a denylist.
        """
        contract = self.observation_contract
        allowed_keys = dict.fromkeys(
            (
                *contract.camera_keys,
                *contract.proprioception_keys,
                *contract.public_raw_keys,
            )
        )
        if contract.instruction_key is not None:
            allowed_keys[contract.instruction_key] = None
        return {
            key: deepcopy(raw_observation[key])
            for key in allowed_keys
            if key in raw_observation
        }

    def execute(self, request: ActionRequest) -> ActionResult:
        self._validate_action_request(request)
        step_before = self._step_count

        # Exactly one benchmark execution call.  No retry, chunking, clipping,
        # action conversion, or hidden recovery occurs in this adapter.
        raw_result = self._step(request.action)
        parsed = _parse_step_result(raw_result)
        self._step_count = step_before + 1

        benchmark_success = self._read_success()
        post_observation = self.to_observation(parsed.observation)
        self._last_reward = parsed.reward
        self._last_terminated = parsed.terminated
        self._last_api_arity = parsed.api_arity
        self._last_truncated = (
            parsed.truncated
            if parsed.api_arity == 5
            else self._four_tuple_horizon_truncated(parsed.terminated)
        )
        state = self._make_episode_state(benchmark_success)
        return ActionResult(
            request=request,
            raw_result=raw_result,
            post_observation=post_observation,
            episode_state=state,
            step_before=step_before,
            step_after=self._step_count,
        )

    def episode_state(self) -> EpisodeState:
        return self._make_episode_state(self._read_success())

    def roboharn_capability_report(
        self,
        observation: NeutralObservation | None = None,
    ) -> RoboHarnCapabilityReport:
        capabilities = self._last_capabilities
        if observation is not None:
            capabilities = self.capabilities.with_observed_fields(
                camera_keys=tuple(observation.cameras),
                proprioception_keys=tuple(observation.proprioception),
            )
        return assess_roboharn_agent_compatibility(capabilities)

    def require_roboharn_agent_compatibility(
        self,
        observation: NeutralObservation | None = None,
    ) -> None:
        self.roboharn_capability_report(observation).require()

    def _read_instruction(self, raw_observation: Mapping[str, Any]) -> str:
        contract = self.observation_contract
        if contract.instruction_key is not None:
            if contract.instruction_key not in raw_observation:
                raise LiberoProContractError(
                    "configured instruction field is missing from observation: "
                    f"{contract.instruction_key}"
                )
            value = raw_observation[contract.instruction_key]
        elif self._instruction is not None:
            value = self._instruction
        elif self._instruction_provider is not None:
            value = self._instruction_provider()
        elif hasattr(self.env, contract.instruction_attribute):
            value = getattr(self.env, contract.instruction_attribute)
        else:
            raise LiberoProContractError(
                "LIBERO-PRO instruction is unavailable; pass instruction, "
                "instruction_provider, or configure its public attribute"
            )
        if not isinstance(value, str):
            raise LiberoProContractError(
                "LIBERO-PRO instruction must be a string and is never coerced"
            )
        return value

    def _read_success(self) -> bool:
        return _scalar_bool(
            self._success_checker(),
            field_name="check_success",
        )

    def _make_episode_state(self, benchmark_success: bool) -> EpisodeState:
        truncated = self._last_truncated
        if self._last_api_arity == 4:
            truncated = self._four_tuple_horizon_truncated(self._last_terminated)
        return EpisodeState(
            step_count=self._step_count,
            step_limit=self._step_limit,
            benchmark_success=benchmark_success,
            check_success=benchmark_success,
            reward=self._last_reward,
            terminated=self._last_terminated,
            truncated=truncated,
        )

    def _four_tuple_horizon_truncated(self, terminated: bool) -> bool:
        return (
            self._step_limit > 0
            and self._step_count >= self._step_limit
            and not terminated
        )

    def _validate_action_request(self, request: ActionRequest) -> None:
        contract = self.capabilities.action
        if request.action_type != contract.action_type:
            raise LiberoProContractError(
                "LIBERO-PRO action_type mismatch: expected "
                f"{contract.action_type!r}, got {request.action_type!r}; "
                "the adapter never guesses a controller conversion"
            )
        if contract.shape is None:
            return

        array = np.asarray(request.action)
        if tuple(array.shape) != contract.shape:
            raise LiberoProContractError(
                f"LIBERO-PRO action shape must be {contract.shape}, got {array.shape}"
            )
        if contract.dtype is not None and str(array.dtype) != contract.dtype:
            raise LiberoProContractError(
                f"LIBERO-PRO action dtype must be {contract.dtype}, got {array.dtype}"
            )
        try:
            numeric = np.asarray(array, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise LiberoProContractError(
                "LIBERO-PRO action must contain numeric values"
            ) from exc
        if not np.all(np.isfinite(numeric)):
            raise LiberoProContractError("LIBERO-PRO action contains non-finite values")
        if (
            contract.bounds_handling == "adapter_validate"
            and contract.lower_bounds is not None
        ):
            lower = np.asarray(contract.lower_bounds, dtype=np.float64)
            if np.any(numeric < lower):
                raise LiberoProContractError(
                    "LIBERO-PRO action is below the configured lower bounds"
                )
        if (
            contract.bounds_handling == "adapter_validate"
            and contract.upper_bounds is not None
        ):
            upper = np.asarray(contract.upper_bounds, dtype=np.float64)
            if np.any(numeric > upper):
                raise LiberoProContractError(
                    "LIBERO-PRO action is above the configured upper bounds"
                )


class _ParsedStepResult:
    __slots__ = (
        "api_arity",
        "info",
        "observation",
        "reward",
        "terminated",
        "truncated",
    )

    def __init__(
        self,
        *,
        api_arity: int,
        observation: Mapping[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Any,
    ) -> None:
        self.api_arity = api_arity
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info


def _parse_step_result(value: Any) -> _ParsedStepResult:
    if not isinstance(value, tuple) or len(value) not in {4, 5}:
        raise LiberoProContractError(
            "LIBERO-PRO step must return a 4-tuple or compatible 5-tuple"
        )
    if len(value) == 4:
        observation, reward, done, info = value
        terminated = _scalar_bool(done, field_name="done")
        truncated = False
    else:
        observation, reward, terminated_raw, truncated_raw, info = value
        terminated = _scalar_bool(terminated_raw, field_name="terminated")
        truncated = _scalar_bool(truncated_raw, field_name="truncated")
    if not isinstance(observation, Mapping):
        raise LiberoProContractError(
            "LIBERO-PRO post-step observation must be a mapping"
        )
    return _ParsedStepResult(
        api_arity=len(value),
        observation=observation,
        reward=_scalar_float(reward, field_name="reward"),
        terminated=terminated,
        truncated=truncated,
        info=info,
    )


def _action_contract_from_env(env: Any) -> LiberoProActionContract:
    space = getattr(env, "action_space", None)
    if space is None:
        return LiberoProActionContract()
    shape_value = getattr(space, "shape", None)
    shape = None
    if shape_value is not None:
        try:
            shape = tuple(int(size) for size in shape_value)
        except (TypeError, ValueError):
            shape = None
    dtype_value = getattr(space, "dtype", None)
    dtype = None if dtype_value is None else str(np.dtype(dtype_value))
    lower = _optional_flat_bounds(getattr(space, "low", None), shape)
    upper = _optional_flat_bounds(getattr(space, "high", None), shape)
    return LiberoProActionContract(
        shape=shape,
        dtype=dtype,
        lower_bounds=lower,
        upper_bounds=upper,
    )


def _optional_flat_bounds(
    value: Any,
    shape: tuple[int, ...] | None,
) -> tuple[float, ...] | None:
    if value is None or shape is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if tuple(array.shape) != shape:
        return None
    return tuple(float(item) for item in array.reshape(-1))


def _scalar_bool(value: Any, *, field_name: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise LiberoProContractError(
            f"LIBERO-PRO {field_name} must be scalar for one environment"
        )
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (str, bytes, np.str_, np.bytes_)):
        raise LiberoProContractError(
            f"LIBERO-PRO {field_name} must be boolean-like, not text"
        )
    return bool(scalar)


def _scalar_float(value: Any, *, field_name: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise LiberoProContractError(
            f"LIBERO-PRO {field_name} must be scalar for one environment"
        )
    try:
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise LiberoProContractError(
            f"LIBERO-PRO {field_name} must be numeric"
        ) from exc


def _optional_nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        raise LiberoProContractError("step counters and limits must be integers")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise LiberoProContractError(
            f"step counters and limits must be integers, got {value!r}"
        ) from exc
    if result < 0:
        raise LiberoProContractError(
            f"step counters and limits must be non-negative, got {result}"
        )
    return result


__all__ = ["LiberoProAdapter"]
