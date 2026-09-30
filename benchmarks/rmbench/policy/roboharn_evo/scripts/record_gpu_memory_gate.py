#!/usr/bin/env python3
"""Record and enforce a read-only residual GPU-memory launch gate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


QUERY_FIELDS = ("index", "uuid", "name", "memory.total", "memory.used", "memory.free")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed unless one selected GPU has enough currently free memory."
    )
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--minimum-free-mib", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nvidia-smi-bin", default="nvidia-smi")
    args = parser.parse_args(argv)
    if args.gpu_index < 0:
        parser.error("--gpu-index must be non-negative")
    if args.minimum_free_mib <= 0:
        parser.error("--minimum-free-mib must be positive")
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    return args


def parse_inventory(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not raw_row:
            continue
        if len(raw_row) != len(QUERY_FIELDS):
            raise ValueError(
                f"nvidia-smi returned {len(raw_row)} columns, expected {len(QUERY_FIELDS)}"
            )
        values = [value.strip() for value in raw_row]
        try:
            index = int(values[0])
            total_mib = int(values[3])
            used_mib = int(values[4])
            free_mib = int(values[5])
        except ValueError as exc:
            raise ValueError("nvidia-smi returned a non-integer GPU memory value") from exc
        rows.append(
            {
                "index": index,
                "uuid": values[1],
                "name": values[2],
                "memory_total_mib": total_mib,
                "memory_used_mib": used_mib,
                "memory_free_mib": free_mib,
            }
        )
    if not rows:
        raise ValueError("nvidia-smi returned no GPU rows")
    return rows


def build_report(
    *,
    gpu_index: int,
    minimum_free_mib: int,
    nvidia_smi_bin: str,
) -> dict[str, Any]:
    query = ",".join(QUERY_FIELDS)
    result = subprocess.run(
        [
            nvidia_smi_bin,
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    inventory = parse_inventory(result.stdout)
    matching = [item for item in inventory if item["index"] == gpu_index]
    if len(matching) != 1:
        raise ValueError(
            f"expected exactly one nvidia-smi row for GPU {gpu_index}, found {len(matching)}"
        )
    gpu = matching[0]
    passed = gpu["memory_free_mib"] >= minimum_free_mib
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gate": "minimum_residual_gpu_memory",
        "status": "passed" if passed else "failed",
        "gpu": gpu,
        "minimum_free_mib": minimum_free_mib,
        "observed_free_mib": gpu["memory_free_mib"],
        "headroom_mib": gpu["memory_free_mib"] - minimum_free_mib,
        "query_stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "read_only_check": True,
        "processes_stopped": False,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        gpu_index=args.gpu_index,
        minimum_free_mib=args.minimum_free_mib,
        nvidia_smi_bin=args.nvidia_smi_bin,
    )
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "gpu_index": args.gpu_index,
                "observed_free_mib": report["observed_free_mib"],
                "minimum_free_mib": args.minimum_free_mib,
                "report": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
