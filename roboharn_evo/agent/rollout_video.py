from __future__ import annotations

import json
import os
import re
import select
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from roboharn_evo.agent.paths import resolve_writable_path


_STEP_FRAME_NAME = re.compile(r"^step_(\d+)\.png$")
_CAMERA_KEYS = {
    "head": "head_camera",
    "left": "left_camera",
    "right": "right_camera",
}


def write_video_frame(
    writer: subprocess.Popen[bytes],
    frame: bytes,
    *,
    timeout_sec: float = 5.0,
) -> None:
    """Feed one whole frame without letting a stalled encoder block control.

    Writers must be opened with ``bufsize=0`` so closing stdin cannot flush
    buffered bytes into the same stalled pipe. A partial write invalidates that
    video stream; callers must stop feeding it and report the recording error.
    """

    if writer.stdin is None:
        raise RuntimeError("ffmpeg stdin is unavailable")
    descriptor = writer.stdin.fileno()
    os.set_blocking(descriptor, False)
    remaining = memoryview(frame)
    deadline = time.monotonic() + timeout_sec
    while remaining:
        timeout = deadline - time.monotonic()
        if timeout <= 0 or not select.select([], [descriptor], [], timeout)[1]:
            raise TimeoutError(
                f"video encoder did not consume a frame within {timeout_sec:g}s"
            )
        try:
            written = os.write(descriptor, remaining)
        except BlockingIOError:
            continue
        remaining = remaining[written:]


def rollout_frame_paths(camera_dir: Path) -> list[Path]:
    """Return rollout PNGs in numeric environment-step order.

    Environment steps are intentionally sparse: a rollout may contain frames such
    as ``step_000000.png``, ``step_000008.png``, and ``step_000015.png``.  Treating
    those names as an ffmpeg ``%06d`` image sequence stops at the first gap.
    """

    frames: list[tuple[int, str, Path]] = []
    for path in Path(camera_dir).glob("step_*.png"):
        match = _STEP_FRAME_NAME.fullmatch(path.name)
        if match is not None and path.is_file():
            frames.append((int(match.group(1)), path.name, path))
    return [path for _step, _name, path in sorted(frames)]


def _ffconcat_path(path: Path) -> str:
    value = str(path.resolve())
    if "\n" in value or "\r" in value:
        raise ValueError("rollout frame paths may not contain newlines")
    # ffconcat uses backslash escaping outside single-quoted regions.  Closing the
    # quote around an apostrophe keeps arbitrary ordinary filesystem names safe.
    return "'" + value.replace("'", "'\\''") + "'"


def encode_rollout_video(
    camera_dir: Path,
    output_file: Path,
    *,
    fps: int = 10,
) -> subprocess.CompletedProcess[bytes]:
    """Encode every sparse rollout frame into a fixed-rate MP4.

    A concat manifest enumerates the actual files, while ``setpts`` assigns one
    output timestamp per listed frame.  The environment-step gaps therefore do
    not become missing inputs or long pauses in the visualization.
    """

    frame_rate = int(fps)
    if frame_rate <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")

    # Validate the final sink before creating its parent or the temporary
    # ffconcat manifest.  Public callers cannot redirect an encode into source,
    # donor、资源或 RoboHarn-Evo 专用 eval_result 目录以外的路径。
    output_path = resolve_writable_path(output_file)

    frames = rollout_frame_paths(Path(camera_dir))
    if not frames:
        raise FileNotFoundError(f"no rollout step PNGs found in {camera_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.stem}_",
            suffix=".ffconcat",
            dir=output_path.parent,
            delete=False,
        ) as manifest:
            manifest_path = Path(manifest.name)
            manifest.write("ffconcat version 1.0\n")
            for frame in frames:
                manifest.write(f"file {_ffconcat_path(frame)}\n")

        return subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-vf",
                f"setpts=N/({frame_rate}*TB)",
                "-r",
                str(frame_rate),
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=True,
        )
    finally:
        if manifest_path is not None:
            manifest_path.unlink(missing_ok=True)


class ContinuousRolloutVideoRecorder:
    """Record calibrated RMBench cameras at a fixed simulated-time rate.

    The Agent's PNG dump contains only decision-time observations.  This
    recorder instead receives one callback after every simulator control tick
    and samples those ticks at ``fps`` before writing raw RGB frames to ffmpeg.
    It therefore records actual intermediate motion rather than duplicating or
    interpolating sparse Agent observations.
    """

    def __init__(
        self,
        rollout_dir: Path,
        *,
        fps: int = 30,
        simulation_hz: int = 250,
    ) -> None:
        self.fps = int(fps)
        self.simulation_hz = int(simulation_hz)
        if self.fps <= 0:
            raise ValueError("continuous rollout video fps must be positive")
        if self.simulation_hz <= 0:
            raise ValueError("simulation_hz must be positive")
        if self.fps > self.simulation_hz:
            raise ValueError("video fps cannot exceed simulation_hz")

        self.video_dir = resolve_writable_path(Path(rollout_dir) / "video")
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, subprocess.Popen[bytes]] = {}
        self._stderr_paths: dict[str, Path] = {}
        self._frame_counts: dict[str, int] = {}
        self._sample_phase = 0
        self._task_env: Any | None = None
        self._closed = False
        self._errors: list[str] = []

    @property
    def frame_counts(self) -> dict[str, int]:
        return dict(self._frame_counts)

    def attach(self, task_env: Any) -> None:
        setter = getattr(task_env, "_set_eval_video_frame_callback", None)
        if not callable(setter):
            raise TypeError(
                "task environment does not expose a simulator-frame callback"
            )
        self._task_env = task_env
        setter(self.on_simulation_step)
        self._capture(task_env, refresh_render=True)

    def on_simulation_step(self, task_env: Any) -> None:
        if self._closed:
            return
        self._sample_phase += self.fps
        if self._sample_phase < self.simulation_hz:
            return
        self._sample_phase -= self.simulation_hz
        self._capture(task_env, refresh_render=False)

    @staticmethod
    def _rgb24(frame: Any, *, camera_name: str) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError(
                f"{camera_name} frame must have HxWx3 channels, got {array.shape}"
            )
        array = array[:, :, :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def _camera_frames(
        self, task_env: Any, *, refresh_render: bool
    ) -> dict[str, np.ndarray]:
        if refresh_render:
            task_env._update_render()
        cameras = task_env.cameras
        cameras.update_picture()
        rgb = cameras.get_rgb()
        frames = {
            output_name: self._rgb24(
                rgb[camera_key]["rgb"],
                camera_name=output_name,
            )
            for output_name, camera_key in _CAMERA_KEYS.items()
            if camera_key in rgb and "rgb" in rgb[camera_key]
        }
        third = cameras.get_third_view_observation(include_depth=False)
        if "rgb" in third:
            frames["third"] = self._rgb24(
                third["rgb"],
                camera_name="third",
            )
        if not frames:
            raise RuntimeError("RMBench exposed no RGB cameras for video")
        return frames

    def _start_writer(
        self,
        camera_name: str,
        frame: np.ndarray,
    ) -> subprocess.Popen[bytes]:
        height, width = frame.shape[:2]
        output_path = self.video_dir / f"{camera_name}.mp4"
        stderr_path = self.video_dir / f"{camera_name}.ffmpeg.log"
        # An unread stderr pipe can fill and stop ffmpeg reading its RGB input.
        # Keep diagnostics in a file instead of coupling two blocking pipes.
        with stderr_path.open("wb") as stderr:
            writer = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "-",
                    "-an",
                    "-vcodec",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-threads",
                    "2",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                stdin=subprocess.PIPE,
                stderr=stderr,
                bufsize=0,
                start_new_session=True,
            )
        self._writers[camera_name] = writer
        self._stderr_paths[camera_name] = stderr_path
        self._frame_counts[camera_name] = 0
        return writer

    def _capture(self, task_env: Any, *, refresh_render: bool) -> None:
        try:
            frames = self._camera_frames(
                task_env,
                refresh_render=refresh_render,
            )
            for camera_name, frame in frames.items():
                writer = self._writers.get(camera_name)
                if writer is None:
                    writer = self._start_writer(camera_name, frame)
                write_video_frame(writer, frame.tobytes())
                self._frame_counts[camera_name] += 1
        except Exception as exc:
            self._errors.append(f"capture={type(exc).__name__}: {exc}")
            setter = getattr(task_env, "_set_eval_video_frame_callback", None)
            if callable(setter):
                setter(None)
            for writer in self._writers.values():
                if writer.stdin is not None and not writer.stdin.closed:
                    writer.stdin.close()

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self._summary()
        self._closed = True
        if self._task_env is not None:
            setter = getattr(
                self._task_env,
                "_set_eval_video_frame_callback",
                None,
            )
            if callable(setter):
                setter(None)

        for camera_name, writer in self._writers.items():
            try:
                if writer.stdin is not None and not writer.stdin.closed:
                    writer.stdin.close()
                writer.wait(timeout=30)
                if writer.returncode != 0:
                    self._errors.append(
                        f"{camera_name}=ffmpeg_exit_{writer.returncode}: "
                        f"see {self._stderr_paths.get(camera_name)}"
                    )
            except subprocess.TimeoutExpired:
                writer.kill()
                writer.wait(timeout=5)
                self._errors.append(f"{camera_name}=ffmpeg_close_timeout")
            except Exception as exc:
                self._errors.append(f"{camera_name}=close_{type(exc).__name__}: {exc}")

        summary = self._summary()
        (self.video_dir / "continuous_recording.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def _summary(self) -> dict[str, Any]:
        expected = {"head", "left", "right", "third"}
        recorded = {name for name, count in self._frame_counts.items() if count > 0}
        return {
            "schema": "roboharn_evo/rollout_video/continuous_v1",
            "fps": self.fps,
            "simulation_hz": self.simulation_hz,
            "sampling": "fixed_simulated_time",
            "frame_counts": dict(sorted(self._frame_counts.items())),
            "complete": recorded == expected and not self._errors,
            "errors": list(self._errors),
        }
