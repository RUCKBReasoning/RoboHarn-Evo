#!/usr/bin/env python3
"""Record and validate a local sharded safetensors model artifact.

The manifest deliberately hashes only small metadata/evidence files.  Weight
identity is established from the upstream index, every shard header, and a
deterministic name-and-size manifest; the hundreds of gigabytes of tensor
payload are not read or copied.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = 1
REQUIRED_EVIDENCE_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "LICENSE",
    "README.md",
)
MODEL_FILE_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
INCOMPLETE_SUFFIXES = (
    ".incomplete",
    ".partial",
    ".part",
    ".tmp",
    ".temp",
    ".download",
    ".crdownload",
    ".lock",
)
INCOMPLETE_EXACT_NAMES = {
    "incomplete",
    ".incomplete",
    "partial",
    ".partial",
    "tmp",
    ".tmp",
    "temp",
    ".temp",
}


class ArtifactValidationError(ValueError):
    """Raised when a local model artifact is incomplete or inconsistent."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a local sharded safetensors model and atomically record "
            "its source, revision evidence, metadata, and artifact identity."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--revision-selector", required=True)
    parser.add_argument(
        "--immutable-revision",
        default=None,
        help="Registry commit/digest if independently known; otherwise omit it.",
    )
    parser.add_argument(
        "--revision-evidence",
        required=True,
        help=(
            "A concise, non-secret description or reference explaining how the "
            "selected revision was obtained."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object_no_duplicates(raw: bytes, *, label: str) -> dict[str, Any]:
    def object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ArtifactValidationError(f"duplicate JSON key in {label}: {key}")
            value[key] = child
        return value

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=object_pairs_hook)
    except ArtifactValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid UTF-8 JSON in {label}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"expected a JSON object in {label}")
    return value


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    return _json_object_no_duplicates(path.read_bytes(), label=label)


def _looks_incomplete(relative_path: Path) -> bool:
    for part in relative_path.parts:
        lowered = part.lower()
        if lowered in INCOMPLETE_EXACT_NAMES:
            return True
        if lowered.startswith("._____temp"):
            return True
        if any(lowered.endswith(suffix) for suffix in INCOMPLETE_SUFFIXES):
            return True
    return False


def _list_ordinary_files(model_dir: Path) -> list[Path]:
    files: list[Path] = []
    incomplete_entries: list[str] = []
    unsupported_entries: list[str] = []
    for path in model_dir.rglob("*"):
        relative = path.relative_to(model_dir)
        if _looks_incomplete(relative):
            incomplete_entries.append(relative.as_posix())
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if stat.S_ISREG(mode):
            files.append(path)
            continue
        entry_kind = "symbolic link" if stat.S_ISLNK(mode) else "non-regular entry"
        unsupported_entries.append(f"{relative.as_posix()} ({entry_kind})")

    if incomplete_entries:
        rendered = ", ".join(sorted(incomplete_entries)[:10])
        raise ArtifactValidationError(f"incomplete/temporary artifact entries found: {rendered}")
    if unsupported_entries:
        rendered = ", ".join(sorted(unsupported_entries)[:10])
        raise ArtifactValidationError(
            "model artifact must contain only directories and ordinary files; found: " + rendered
        )
    if not files:
        raise ArtifactValidationError("model directory contains no ordinary files")

    files.sort(key=lambda path: path.relative_to(model_dir).as_posix())
    for path in files:
        if path.suffix.lower() in MODEL_FILE_SUFFIXES and path.stat().st_size == 0:
            raise ArtifactValidationError(
                f"zero-byte model file: {path.relative_to(model_dir).as_posix()}"
            )
    return files


def _manifest_sha256(model_dir: Path, files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(model_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalize_index_shard_path(raw_path: object) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ArtifactValidationError("weight_map contains an empty or non-string shard path")
    if "\\" in raw_path:
        raise ArtifactValidationError(f"weight_map shard path uses a backslash: {raw_path!r}")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ArtifactValidationError(f"unsafe weight_map shard path: {raw_path!r}")
    return pure.as_posix()


def _inspect_safetensors(path: Path, relative: str) -> dict[str, Any]:
    size = path.stat().st_size
    if size < 8:
        raise ArtifactValidationError(f"safetensors file is shorter than its header prefix: {relative}")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length == 0 or header_length > size - 8:
            raise ArtifactValidationError(
                f"invalid safetensors header length in {relative}: {header_length}"
            )
        header_raw = handle.read(header_length)
    if len(header_raw) != header_length:
        raise ArtifactValidationError(f"truncated safetensors header in {relative}")
    header = _json_object_no_duplicates(header_raw, label=f"safetensors header {relative}")

    metadata = header.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise ArtifactValidationError(f"invalid __metadata__ in safetensors header {relative}")

    payload_bytes = size - 8 - header_length
    tensors: dict[str, dict[str, Any]] = {}
    ranges: list[tuple[int, int, str]] = []
    dtype_counts: dict[str, int] = defaultdict(int)
    dtype_bytes: dict[str, int] = defaultdict(int)
    for tensor_name, tensor in header.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ArtifactValidationError(f"invalid tensor name in safetensors header {relative}")
        if not isinstance(tensor, dict):
            raise ArtifactValidationError(
                f"invalid tensor metadata for {tensor_name!r} in {relative}"
            )
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise ArtifactValidationError(f"invalid dtype for {tensor_name!r} in {relative}")
        if not isinstance(shape, list) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        ):
            raise ArtifactValidationError(f"invalid shape for {tensor_name!r} in {relative}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise ArtifactValidationError(
                f"invalid data_offsets for {tensor_name!r} in {relative}"
            )
        start, end = offsets
        if start < 0 or end < start or end > payload_bytes:
            raise ArtifactValidationError(
                f"out-of-range data_offsets for {tensor_name!r} in {relative}"
            )
        tensor_bytes = end - start
        tensors[tensor_name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, end],
        }
        ranges.append((start, end, tensor_name))
        dtype_counts[dtype] += 1
        dtype_bytes[dtype] += tensor_bytes

    cursor = 0
    for start, end, tensor_name in sorted(ranges):
        if start != cursor:
            relation = "overlap" if start < cursor else "gap"
            raise ArtifactValidationError(
                f"safetensors payload has a {relation} before {tensor_name!r} in {relative}"
            )
        cursor = end
    if cursor != payload_bytes:
        raise ArtifactValidationError(
            f"safetensors payload has unreferenced bytes in {relative}: "
            f"referenced={cursor}, payload={payload_bytes}"
        )

    return {
        "path": relative,
        "file_bytes": size,
        "header_bytes": header_length,
        "header_sha256": hashlib.sha256(header_raw).hexdigest(),
        "payload_bytes": payload_bytes,
        "tensor_count": len(tensors),
        "tensors": tensors,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "dtype_payload_bytes": dict(sorted(dtype_bytes.items())),
    }


def _validate_index_and_shards(
    model_dir: Path,
    files: list[Path],
    index: dict[str, Any],
) -> dict[str, Any]:
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict):
        raise ArtifactValidationError("model.safetensors.index.json metadata must be an object")
    total_size = metadata.get("total_size")
    if isinstance(total_size, bool) or not isinstance(total_size, int) or total_size < 0:
        raise ArtifactValidationError(
            "model.safetensors.index.json metadata.total_size must be a non-negative integer"
        )
    if not isinstance(weight_map, dict) or not weight_map:
        raise ArtifactValidationError(
            "model.safetensors.index.json weight_map must be a non-empty object"
        )

    expected_by_shard: dict[str, set[str]] = defaultdict(set)
    for tensor_name, raw_shard in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ArtifactValidationError("weight_map contains an empty or non-string tensor name")
        shard = _normalize_index_shard_path(raw_shard)
        expected_by_shard[shard].add(tensor_name)

    ordinary_by_name = {
        path.relative_to(model_dir).as_posix(): path
        for path in files
    }
    missing = sorted(set(expected_by_shard) - set(ordinary_by_name))
    if missing:
        raise ArtifactValidationError("indexed safetensors shard is missing: " + ", ".join(missing))

    discovered_safetensors = {
        relative
        for relative in ordinary_by_name
        if PurePosixPath(relative).suffix.lower() == ".safetensors"
    }
    unexpected = sorted(discovered_safetensors - set(expected_by_shard))
    unexpected_model_shards = [
        relative for relative in unexpected if PurePosixPath(relative).name.startswith("model")
    ]
    if unexpected_model_shards:
        raise ArtifactValidationError(
            "unexpected model safetensors shard(s) not present in weight_map: "
            + ", ".join(unexpected_model_shards)
        )
    if unexpected:
        raise ArtifactValidationError(
            "unindexed safetensors file(s) cannot be validated against weight_map: "
            + ", ".join(unexpected)
        )

    shard_records: list[dict[str, Any]] = []
    total_payload_bytes = 0
    dtype_counts: dict[str, int] = defaultdict(int)
    dtype_bytes: dict[str, int] = defaultdict(int)
    for relative in sorted(expected_by_shard):
        record = _inspect_safetensors(ordinary_by_name[relative], relative)
        actual_tensors = set(record.pop("tensors"))
        expected_tensors = expected_by_shard[relative]
        missing_tensors = sorted(expected_tensors - actual_tensors)
        unexpected_tensors = sorted(actual_tensors - expected_tensors)
        if missing_tensors or unexpected_tensors:
            details: list[str] = []
            if missing_tensors:
                details.append("missing=" + ",".join(missing_tensors[:10]))
            if unexpected_tensors:
                details.append("unexpected=" + ",".join(unexpected_tensors[:10]))
            raise ArtifactValidationError(
                f"safetensors header/index tensor mismatch in {relative}: " + "; ".join(details)
            )
        total_payload_bytes += int(record["payload_bytes"])
        for dtype, count in record["dtype_tensor_counts"].items():
            dtype_counts[dtype] += int(count)
        for dtype, byte_count in record["dtype_payload_bytes"].items():
            dtype_bytes[dtype] += int(byte_count)
        shard_records.append(record)

    if total_payload_bytes != total_size:
        raise ArtifactValidationError(
            "safetensors payload total does not match index metadata.total_size: "
            f"payload={total_payload_bytes}, index={total_size}"
        )
    if sum(dtype_counts.values()) != len(weight_map):
        raise ArtifactValidationError(
            "safetensors tensor count does not match index weight_map tensor count"
        )

    return {
        "shard_count": len(shard_records),
        "index_tensor_count": len(weight_map),
        "index_metadata_total_size": total_size,
        "validated_payload_bytes": total_payload_bytes,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "dtype_payload_bytes": dict(sorted(dtype_bytes.items())),
        "shards": shard_records,
    }


def _architecture_metadata(config: dict[str, Any]) -> dict[str, Any]:
    architectures = config.get("architectures")
    if isinstance(architectures, str):
        normalized_architectures = [architectures]
    elif isinstance(architectures, list) and all(isinstance(item, str) for item in architectures):
        normalized_architectures = list(architectures)
    elif architectures is None:
        normalized_architectures = []
    else:
        raise ArtifactValidationError("config.json architectures must be a string or string list")

    vision_config = config.get("vision_config")
    if vision_config is not None and not isinstance(vision_config, dict):
        raise ArtifactValidationError("config.json vision_config must be an object when present")
    quantization_config = config.get("quantization_config")
    if quantization_config is not None and not isinstance(quantization_config, dict):
        raise ArtifactValidationError(
            "config.json quantization_config must be an object when present"
        )
    text_config = config.get("text_config")
    if text_config is not None and not isinstance(text_config, dict):
        raise ArtifactValidationError("config.json text_config must be an object when present")

    task_config_fields = {
        key: config[key]
        for key in ("task", "task_type", "pipeline_tag", "finetuning_task", "problem_type")
        if key in config
    }
    inferred_task = None
    lowered_architectures = " ".join(normalized_architectures).lower()
    if "conditionalgeneration" in lowered_architectures:
        inferred_task = "conditional_generation"

    vision_summary: dict[str, Any] = {"present": vision_config is not None}
    if vision_config is not None:
        for key in (
            "model_type",
            "depth",
            "hidden_size",
            "in_channels",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
        ):
            if key in vision_config:
                vision_summary[key] = vision_config[key]
        vision_summary["image_token_id"] = config.get("image_token_id")
        vision_summary["video_token_id"] = config.get("video_token_id")

    quantization_summary: dict[str, Any] = {"present": quantization_config is not None}
    if quantization_config is not None:
        for key in (
            "quant_method",
            "activation_scheme",
            "weight_block_size",
            "weight_per_tensor",
            "act_per_tensor",
        ):
            if key in quantization_config:
                quantization_summary[key] = quantization_config[key]
        modules = quantization_config.get("modules_to_not_convert")
        if isinstance(modules, list):
            quantization_summary["modules_to_not_convert_count"] = len(modules)

    return {
        "architectures": normalized_architectures,
        "model_type": config.get("model_type"),
        "text_model_type": text_config.get("model_type") if text_config else None,
        "declared_dtype": (
            text_config.get("dtype") if text_config and "dtype" in text_config else config.get("dtype")
        ),
        "vision": vision_summary,
        "task": {
            "config_fields": task_config_fields,
            "architecture_name_inference": inferred_task,
            "inference_is_not_registry_metadata": inferred_task is not None,
        },
        "quantization": quantization_summary,
    }


def _nonempty(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ArtifactValidationError(f"{label} must be non-empty")
    return normalized


def build_local_model_identity(
    *,
    model_dir: Path,
    repo_id: str,
    source_registry: str,
    revision_selector: str,
    immutable_revision: str | None,
    revision_evidence: str,
) -> dict[str, Any]:
    original_model_dir = model_dir.expanduser()
    if not original_model_dir.is_absolute():
        raise ArtifactValidationError("model directory must be an absolute path")
    root = original_model_dir.resolve()
    if not root.exists():
        raise ArtifactValidationError(f"model directory does not exist: {root}")
    if not root.is_dir():
        raise ArtifactValidationError(f"model directory is not a directory: {root}")

    files = _list_ordinary_files(root)
    ordinary_by_name = {path.relative_to(root).as_posix(): path for path in files}
    missing_evidence = [name for name in REQUIRED_EVIDENCE_FILES if name not in ordinary_by_name]
    if missing_evidence:
        raise ArtifactValidationError(
            "required model evidence file(s) missing: " + ", ".join(missing_evidence)
        )

    config_path = ordinary_by_name["config.json"]
    index_path = ordinary_by_name["model.safetensors.index.json"]
    license_path = ordinary_by_name["LICENSE"]
    readme_path = ordinary_by_name["README.md"]
    config = _load_json_object(config_path, label="config.json")
    index = _load_json_object(index_path, label="model.safetensors.index.json")
    safetensors = _validate_index_and_shards(root, files, index)

    license_text = license_path.read_text(encoding="utf-8", errors="replace")
    apache_markers = {
        "contains_apache_license_title": "Apache License" in license_text,
        "contains_version_2_0": "Version 2.0" in license_text,
        "contains_apache_license_url": "www.apache.org/licenses" in license_text,
    }
    file_hashes = {
        "config.json": sha256_file(config_path),
        "model.safetensors.index.json": sha256_file(index_path),
        "LICENSE": sha256_file(license_path),
        "README.md": sha256_file(readme_path),
    }
    file_sizes = {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in files
    }
    normalized_immutable_revision = (
        _nonempty(immutable_revision, "immutable revision")
        if immutable_revision is not None
        else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_directory": str(root),
        "source": {
            "registry": _nonempty(source_registry, "source registry"),
            "repo_id": _nonempty(repo_id, "repo id"),
            "revision_selector": _nonempty(revision_selector, "revision selector"),
            "immutable_revision": normalized_immutable_revision,
            "revision_evidence": _nonempty(revision_evidence, "revision evidence"),
            "immutable_revision_status": (
                "recorded_from_cli" if normalized_immutable_revision is not None else "not_available"
            ),
        },
        "artifact": {
            "file_count": len(files),
            "total_file_bytes": sum(file_sizes.values()),
            "manifest_sha256": _manifest_sha256(root, files),
            "manifest_algorithm": "sha256(sorted(relative_posix_path + NUL + decimal_size + LF))",
            "file_sizes_bytes": file_sizes,
            "evidence_file_sha256": file_hashes,
            "validation_scope": (
                "All ordinary file names and sizes, all indexed safetensors headers and tensor "
                "offsets, and selected metadata file contents; tensor payload contents are not hashed."
            ),
        },
        "model_metadata": _architecture_metadata(config),
        "safetensors": safetensors,
        "license_evidence": {
            "file": "LICENSE",
            "sha256": file_hashes["LICENSE"],
            "apache_2_0_file_markers": apache_markers,
        },
        "license_review_status": "file_present_not_legal_review",
        "secrets_recorded": False,
    }


def atomic_write_json(output: Path, payload: dict[str, Any]) -> None:
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_dir = args.model_dir.expanduser()
    output = args.output.expanduser().resolve()
    if model_dir.is_absolute():
        resolved_model_dir = model_dir.resolve()
        try:
            output.relative_to(resolved_model_dir)
        except ValueError:
            pass
        else:
            raise ArtifactValidationError(
                "output must be outside model directory so the artifact manifest is not self-referential"
            )
    payload = build_local_model_identity(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        source_registry=args.source_registry,
        revision_selector=args.revision_selector,
        immutable_revision=args.immutable_revision,
        revision_evidence=args.revision_evidence,
    )
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "model_identity": str(output),
                "repo_id": payload["source"]["repo_id"],
                "immutable_revision": payload["source"]["immutable_revision"],
                "manifest_sha256": payload["artifact"]["manifest_sha256"],
                "file_count": payload["artifact"]["file_count"],
                "shard_count": payload["safetensors"]["shard_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
