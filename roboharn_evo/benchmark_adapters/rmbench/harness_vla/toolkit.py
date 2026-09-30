"""Keep RPent's tool-calling and memory interface; bind only robot operations."""
from __future__ import annotations

from functools import partial
from pathlib import Path

from rpent.tools.toolkit import Toolkit


ARM = {"type": "string", "enum": ["left", "right", "both"]}
SINGLE_ARM = {"type": "string", "enum": ["left", "right"]}
XYZ = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
QUAT = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
CAMERA = {"type": "string", "enum": ["head", "left", "right"]}
NUM = {"type": "number"}
INT = {"type": "integer"}
STR = {"type": "string"}
GRIPPER_STEPS = {**INT, "minimum": 1, "maximum": 50}
MOVE_ARGUMENTS = {"arm": SINGLE_ARM, "xyz": XYZ, "quat_wxyz": QUAT,
                  "tol": {**NUM, "exclusiveMinimum": 0, "maximum": 0.05},
                  "max_steps": {**INT, "minimum": 1, "maximum": 100},
                  "step_clip": {**NUM, "exclusiveMinimum": 0, "maximum": 0.15}}


def spec(name, description, properties, required=()):
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": properties, "required": list(required),
                             "additionalProperties": False}}


PRIMITIVES = [
    spec("view_driver_state", "Observe all three RGB views and public robot state. Does not move the robot.", {}),
    spec("back_project", "Convert visible [u,v] pixels in the selected RGB-D image to measured world meters.",
         {"camera": CAMERA, "pixels": {"type": "array", "items": {"type": "array", "items": NUM,
                                                                       "minItems": 2, "maxItems": 2}}}, ["camera", "pixels"]),
    spec("segment", "SAM3 text segmentation of the current selected image; not a simulator object lookup.",
         {"camera": CAMERA, "prompt": STR}, ["camera", "prompt"]),
    spec("move_to", "Move one arm's action-frame EE toward observed world xyz, preserving its gripper and the other arm. See action_frame_to_tcp in state; xyz is NOT automatically a TCP/contact point.",
         MOVE_ARGUMENTS, ["arm", "xyz"]),
    spec("move_pose", "Move one EE to observed world xyz and wxyz orientation; other arm remains held.",
         MOVE_ARGUMENTS, ["arm", "xyz", "quat_wxyz"]),
    spec("rotate_wrist", "Relative yaw rotation in the selected EE frame, holding position.",
         {"arm": SINGLE_ARM, "delta_yaw": NUM}, ["arm", "delta_yaw"]),
    spec("rotate_pitch", "Relative pitch rotation in the selected EE frame, holding position.",
         {"arm": SINGLE_ARM, "delta_pitch": NUM}, ["arm", "delta_pitch"]),
    spec("set_gripper", "Set gripper open or closed. Not proof of a successful grasp.",
         {"arm": ARM, "gripper": {"type": "string", "enum": ["open", "closed"]}, "steps": GRIPPER_STEPS}, ["arm", "gripper"]),
    spec("release", "Open the selected gripper(s); inspect the resulting placement.",
         {"arm": ARM, "steps": GRIPPER_STEPS}, ["arm"]),
    spec("vla_act", "Invoke the frozen Pi0.5 with current head/left/right RGB and joint state for contact-rich interaction. The selected arm(s) receive real policy actions. A lift-and-grip return is only a primitive heuristic, not attachment or task-success evidence.",
         {"prompt": STR, "arm": ARM, "max_chunks": {**INT, "minimum": 1, "maximum": 100},
          "stop": {"type": "string", "enum": ["chunk_budget", "lift_and_grip", "benchmark_success"]},
          "lift_thresh": NUM, "gripper_closed_thresh": NUM}, ["prompt", "arm"]),
]


class RMBenchToolkit(Toolkit):
    def __init__(self, client, *, output_dir: Path, memory_dir: Path, bootstrap=False):
        super().__init__()
        self.client = client
        self.output_dir, self.memory_dir = Path(output_dir).resolve(), Path(memory_dir).resolve()
        self.bootstrap = bootstrap
        for definition in PRIMITIVES:
            self.add_tool(definition["name"], definition, partial(self.robot_tool, definition["name"]))
        if bootstrap:
            definition = spec("reset", "Exploration only: restart the SAME reference seed; no new seed search.", {})
            self.add_tool("reset", definition, partial(self.robot_tool, "reset"))
        # Use RPent's same file/IO vocabulary, with evaluation memory read-only.
        for name, handler in [("read_text_file", self.read_text_file), ("write_text_file", self.write_text_file),
                              ("list_dir", self.list_dir)]:
            definition = dict(self._tools[name][0])
            definition["description"] = definition["description"].replace("resources/libero/results_*_pert/", str(self.memory_dir))
            self.add_tool(name, definition, handler)

    def _path(self, value, *, write=False):
        path = Path(value)
        if not path.is_absolute():
            path = self.output_dir / path
        path = path.resolve()
        roots = [self.output_dir]
        if not write or self.bootstrap:
            roots.append(self.memory_dir)
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError("path outside this run's artifacts or permitted memory")
        return path

    def read_text_file(self, path, max_chars=40000):
        p = self._path(path)
        return {"path": str(p), "content": p.read_text(encoding="utf-8")[:max_chars]}

    def write_text_file(self, path, content):
        p = self._path(path, write=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": str(p), "bytes_written": len(content.encode())}

    def list_dir(self, path=""):
        p = self._path(path or str(self.output_dir))
        return {"path": str(p), "files": sorted(x.name for x in p.iterdir())}

    def robot_tool(self, name, **args):
        result = self.client.call("execute", kwargs={"name": name, "args": args}, timeout_s=3600)
        views = result.get("state", {}).get("views", {})
        # RPent's stock ToolResult and MCP bridge already support three images.
        for camera, key in zip(("head", "left", "right"), ("_image_bytes", "_image_cam_bytes", "_image_wrist_bytes")):
            if camera in views and name not in {"back_project", "segment"}:
                result[key] = self._path(views[camera]["rgb_path"]).read_bytes()
        if result.get("result", {}).get("error"):
            result["error"] = result["result"]["error"]
        return result
