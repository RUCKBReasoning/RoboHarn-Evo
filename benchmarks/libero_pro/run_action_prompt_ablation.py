"""Compare two LIBERO policy prompts with one observation and one noise draw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from roboharn_evo.benchmark_adapters.libero_pro.native_policy import (
    LiberoPi05PolicyBackend,
    LiberoPi05PolicyConfig,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one object")
    return value


def summarize_prompt_ablation(
    *,
    baseline_actions: np.ndarray,
    grounded_actions: np.ndarray,
) -> dict[str, Any]:
    baseline = np.asarray(baseline_actions, dtype=np.float32)
    grounded = np.asarray(grounded_actions, dtype=np.float32)
    if baseline.shape != grounded.shape or baseline.ndim != 2:
        raise ValueError("controlled action chunks must be same-shape matrices")
    deltas = np.max(np.abs(baseline - grounded), axis=1)
    return {
        "action_shape": list(baseline.shape),
        "changed_action_steps": int(np.count_nonzero(deltas > 0)),
        "max_absolute_delta_by_step": deltas.tolist(),
        "max_absolute_delta": float(deltas.max(initial=0.0)),
        "baseline_actions": baseline.tolist(),
        "grounded_actions": grounded.tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--policy-observation", type=Path, required=True)
    parser.add_argument("--grounding-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("output already exists; refusing to overwrite")
    provenance = _read_json(args.provenance.expanduser().resolve(strict=True))
    decision = _read_json(args.grounding_decision.expanduser().resolve(strict=True))
    if decision.get("knowledge_adopted") is not True:
        raise ValueError("grounding decision did not adopt Action Knowledge")
    baseline_prompt = str(decision.get("prompt_before", "")).strip()
    grounded_prompt = str(decision.get("prompt_after", "")).strip()
    if not baseline_prompt or not grounded_prompt or baseline_prompt == grounded_prompt:
        raise ValueError("controlled prompts must be non-empty and distinct")
    with np.load(args.policy_observation.expanduser().resolve(strict=True)) as archive:
        observation = {key: archive[key].copy() for key in archive.files}
    policy = provenance["policy"]
    config = LiberoPi05PolicyConfig(
        checkpoint_dir=Path(policy["checkpoint_dir"]),
        source_ref=str(policy["source_ref"]),
        license_id=str(policy["license_id"]),
        config_name=str(policy["config_name"]),
        device=str(policy["device"]),
    )
    from openpi.training import config as training_config

    model_config = training_config.get_config(config.config_name).model
    noise = (
        np.random.default_rng(args.noise_seed)
        .standard_normal(
            (int(model_config.action_horizon), int(model_config.action_dim))
        )
        .astype(np.float32)
    )
    backend = LiberoPi05PolicyBackend(config)
    baseline_actions = backend.predict_native_action_chunk(
        observation,
        prompt=baseline_prompt,
        sampling_noise=noise,
    )
    grounded_actions = backend.predict_native_action_chunk(
        observation,
        prompt=grounded_prompt,
        sampling_noise=noise,
    )
    result = {
        "schema": "roboharn_evo/libero_cross_domain_action_prompt_ablation/v1",
        "baseline_prompt": baseline_prompt,
        "grounded_prompt": grounded_prompt,
        "same_observation": True,
        "same_sampling_noise": True,
        "sampling_noise_seed": args.noise_seed,
        "sampling_noise_shape": list(noise.shape),
        "motion_executed": False,
        "grounding_audit": decision["audit"],
        **summarize_prompt_ablation(
            baseline_actions=baseline_actions,
            grounded_actions=grounded_actions,
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
