from __future__ import annotations

import csv
import json
from pathlib import Path

from policy.roboharn_evo.scripts.audit_iclr_trace_evidence import analyze_trace
from policy.roboharn_evo.scripts.export_iclr_trace_metrics import aggregate_rows, row_from_report, write_csv


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_trace_metrics_row_exports_core_counts(tmp_path: Path) -> None:
    trace = tmp_path / "episode_0000_agent_trace.jsonl"
    write_jsonl(
        trace,
        [
            {"event": "episode_start", "task_name": "cover_blocks", "seed": 7},
            {
                "event": "observation_preprocess",
                "latency_sec": 0.1,
                "segmentation": [
                    {
                        "success": True,
                        "object_id": "lid",
                        "camera": "head",
                        "detections": [{"camera": "head", "grounding_3d": {"success": True}}],
                    }
                ],
            },
            {
                "event": "scene_memory_update",
                "scene_memory": {
                    "instances": [{"instance_id": "lid_01", "track_id": "track_0001"}],
                    "temporal_memory": {"tracks": []},
                    "task_focus": {"tool_instances": ["lid_01"]},
                },
            },
            {"event": "recovery_router", "workflow": "task-level-recovery-control", "tools": ["reobserve_scene"]},
            {"event": "recovery_dispatch"},
            {"event": "action_effect_verification", "result": {"effect_verified": "true", "confidence": 0.9}},
            {"event": "recovery_result", "results": [{"tool_name": "reobserve_scene", "success": True}]},
            {"event": "episode_end", "success": True, "result": "Success", "total_steps": 3},
        ],
    )

    row = row_from_report(analyze_trace(trace))

    assert row["task_name"] == "cover_blocks"
    assert row["seed"] == 7
    assert row["success"] is True
    assert row["has_complete_recovery_chain"] is True
    assert row["segmentation_success"] == 1
    assert row["grounding_success_entries"] == 1
    assert row["unique_track_ids"] == 1
    assert row["complete_recovery_chains"] == 1
    assert row["action_effect_true"] == 1


def test_trace_metrics_csv_and_aggregate(tmp_path: Path) -> None:
    rows = [
        {"trace": "a", "success": True, "has_complete_recovery_chain": True, "segmentation_errors": 0},
        {"trace": "b", "success": False, "has_complete_recovery_chain": False, "segmentation_errors": 2, "warnings": "segmentation_errors_present"},
    ]
    csv_path = tmp_path / "metrics.csv"

    write_csv(csv_path, rows)
    aggregate = aggregate_rows(rows)

    with csv_path.open("r", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))

    assert loaded[0]["trace"] == "a"
    assert aggregate["trace_count"] == 2
    assert aggregate["success_count"] == 1
    assert aggregate["numeric_sums"]["segmentation_errors"] == 2
    assert aggregate["flag_counts"]["has_complete_recovery_chain"] == 1
    assert aggregate["warnings"] == ["segmentation_errors_present"]
