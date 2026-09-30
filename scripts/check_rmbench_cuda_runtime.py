from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.rmbench.integration import _runtime_process_boundary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    args = parser.parse_args()
    state_path = args.state_file.resolve(strict=True)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    with _runtime_process_boundary(asset_dir=args.assets_root.resolve(strict=True), output_dir=directory, runtime_configs=directory / "runtime_configs", workspace_dir=directory / "runtime_workspace", cache_dir=directory / "runtime_caches"):
        import torch
        from curobo.types.math import Pose

        with np.load(state_path) as states:
            values = states["endpose/left_endpose"][0].tolist()
        pose = Pose.from_list(values)
        inverse = pose.inverse()
        restored = inverse.inverse()
        torch.cuda.synchronize()
        torch.testing.assert_close(restored.position, pose.position, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(restored.quaternion, pose.quaternion, atol=1e-5, rtol=1e-5)
        report = {"status": "passed", "source_state": str(state_path), "gpu": torch.cuda.get_device_name(), "temporary_directory": os.environ["TMPDIR"], "temporary_directory_length": len(os.environ["TMPDIR"]), "curobo_pose_inverse_completed": True, "robot_motion_executed": False}
        (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
