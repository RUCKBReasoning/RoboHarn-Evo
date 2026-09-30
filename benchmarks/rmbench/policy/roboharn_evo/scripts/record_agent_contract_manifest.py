#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.agent.operation_candidates import SUPPORTED_OPERATION_ACTION_MODES
from policy.roboharn_evo.agent.recovery.tool_specs import RECOVERY_TOOLS
from policy.roboharn_evo.scripts.serve_qwen_planner import (
    OOD_SYSTEM_PROMPT,
    PERCEPTION_QUERY_SYSTEM_PROMPT,
    RECOVERY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    load_perception_query_normalization_prompt,
)


RUNTIME_SOURCE_PATHS = (
    "policy/roboharn_evo/scripts/serve_qwen_planner.py",
    "policy/roboharn_evo/scripts/serve_openai_planner.py",
    "policy/roboharn_evo/agent/prompts.py",
    "policy/roboharn_evo/agent/operation_candidates.py",
    "policy/roboharn_evo/agent/recovery/tool_specs.py",
    "policy/roboharn_evo/agent/recovery/action_effect_verifier.py",
    "policy/roboharn_evo/agent/perception/query_normalization.py",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record the shared RoboHarn-Evo prompt, skill, schema, tool, and observation contract."
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perception-condition", choices=("no_oracle", "oracle"), required=True)
    parser.add_argument("--primary-camera", action="append", required=True)
    parser.add_argument("--verification-camera", action="append", required=True)
    parser.add_argument("--max-objects", type=int, required=True)
    parser.add_argument("--instruction-set", required=True)
    return parser.parse_args(argv)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _composite_hash(values: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_agent_contract_manifest(
    *,
    repo_root: Path,
    perception_condition: str,
    primary_cameras: list[str],
    verification_cameras: list[str],
    max_objects: int,
    instruction_set: str,
) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"repo root is not a directory: {root}")
    if max_objects <= 0:
        raise ValueError("max_objects must be positive")
    if perception_condition not in {"no_oracle", "oracle"}:
        raise ValueError("perception_condition must be no_oracle or oracle")
    primary = [str(item).strip() for item in primary_cameras if str(item).strip()]
    verification = [str(item).strip() for item in verification_cameras if str(item).strip()]
    if not primary or not verification:
        raise ValueError("primary and verification camera lists must be non-empty")
    normalized_instruction_set = str(instruction_set).strip()
    if not normalized_instruction_set:
        raise ValueError("instruction_set must be non-empty")

    source_hashes: dict[str, str] = {}
    for relative in RUNTIME_SOURCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"contract source is missing: {relative}")
        source_hashes[relative] = _file_hash(path)

    skills_root = root / "policy/roboharn_evo/skills"
    skill_hashes = {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in sorted(skills_root.rglob("SKILL.md"))
        if path.is_file()
    }
    if not skill_hashes:
        raise ValueError("no SKILL.md files were found")

    prompt_hashes = {
        "plan_system_prompt": _sha256_text(SYSTEM_PROMPT),
        "ood_system_prompt": _sha256_text(OOD_SYSTEM_PROMPT),
        "recovery_system_prompt": _sha256_text(RECOVERY_SYSTEM_PROMPT),
        "perception_query_system_prompt": _sha256_text(PERCEPTION_QUERY_SYSTEM_PROMPT),
        "perception_query_normalization_skill_prompt": _sha256_text(
            load_perception_query_normalization_prompt()
        ),
    }
    tool_contract = {
        "public_operation_modes": sorted(SUPPORTED_OPERATION_ACTION_MODES),
        "recovery_tools": sorted(RECOVERY_TOOLS),
    }
    tool_contract_sha256 = _sha256_text(
        json.dumps(tool_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    contract_components = {
        **{f"prompt:{key}": value for key, value in prompt_hashes.items()},
        **{f"skill:{key}": value for key, value in skill_hashes.items()},
        **{f"source:{key}": value for key, value in source_hashes.items()},
        "tool_contract": tool_contract_sha256,
    }
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repository_root": str(root),
        "shared_model_role_contract": {
            "gateway_handler": "OpenAIPlannerHandler(QwenPlannerHandler)",
            "required_endpoints": [
                "/plan",
                "/ood",
                "/recover",
                "/perception_queries",
                "/normalize_perception_queries",
            ],
            "recover_modes": ["recovery_planning", "action_effect_verification"],
            "same_prompt_and_normalizer_path_for_gpt55_and_local_qwen": True,
            "qwen_specific_task_semantics": False,
            "fallback_route": None,
        },
        "prompt_sha256": prompt_hashes,
        "skill_bundle": {
            "skill_file_count": len(skill_hashes),
            "skill_file_sha256": skill_hashes,
            "composite_sha256": _composite_hash(skill_hashes),
        },
        "tool_contract": {
            **tool_contract,
            "sha256": tool_contract_sha256,
            "private_runtime_candidate_ids_exposed_to_model": False,
        },
        "runtime_source_sha256": source_hashes,
        "observation_contract": {
            "perception_condition": perception_condition,
            "oracle_objects_enabled": perception_condition == "oracle",
            "rgbd_grounding": True,
            "segmentation_backend": "sam3",
            "primary_camera_order": primary,
            "verification_camera_order": verification,
            "planner_image_order": ["segment_start", "segment_end"],
            "perception_multiview_encoding": "image_b64_by_camera_insertion_order",
            "media_transport": "base64_data_url",
            "max_objects": int(max_objects),
        },
        "instruction_contract": {
            "instruction_set": normalized_instruction_set,
            "task_instruction_modified_for_qwen": False,
        },
        "contract_composite_sha256": _composite_hash(contract_components),
        "secrets_recorded": False,
    }


def atomic_write_json(output: Path, payload: dict[str, Any]) -> None:
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_agent_contract_manifest(
        repo_root=args.repo_root,
        perception_condition=args.perception_condition,
        primary_cameras=args.primary_camera,
        verification_cameras=args.verification_camera,
        max_objects=args.max_objects,
        instruction_set=args.instruction_set,
    )
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "contract_manifest": str(args.output.expanduser().resolve()),
                "contract_composite_sha256": payload["contract_composite_sha256"],
                "skill_file_count": payload["skill_bundle"]["skill_file_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
