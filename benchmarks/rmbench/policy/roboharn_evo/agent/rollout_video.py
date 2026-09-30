from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path


_STEP_FRAME_NAME = re.compile(r"^step_(\d+)\.png$")


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

    frames = rollout_frame_paths(Path(camera_dir))
    if not frames:
        raise FileNotFoundError(f"no rollout step PNGs found in {camera_dir}")

    output_path = Path(output_file)
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
