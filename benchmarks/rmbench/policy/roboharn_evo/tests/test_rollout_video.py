from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from PIL import Image

from policy.roboharn_evo.agent.core.img_agent import ImgAgent
from policy.roboharn_evo.agent.rollout_video import encode_rollout_video, rollout_frame_paths
import script.eval_policy as eval_policy


class RolloutVideoTest(unittest.TestCase):
    def test_sparse_step_frames_are_manifested_in_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_rollout_video_") as temp_dir:
            root = Path(temp_dir)
            camera_dir = root / "camera's head"
            camera_dir.mkdir()
            for name in (
                "step_000015.png",
                "step_000000.png",
                "step_000008.png",
                "step_invalid.png",
            ):
                (camera_dir / name).touch()

            captured: dict[str, object] = {}

            def capture_manifest(command, **kwargs):
                manifest_path = Path(command[command.index("-i") + 1])
                captured["command"] = command
                captured["manifest_path"] = manifest_path
                captured["manifest"] = manifest_path.read_text(encoding="utf-8")
                captured["kwargs"] = kwargs
                return subprocess.CompletedProcess(command, 0)

            with mock.patch(
                "policy.roboharn_evo.agent.rollout_video.subprocess.run",
                side_effect=capture_manifest,
            ) as run:
                encode_rollout_video(
                    camera_dir,
                    root / "video" / "head.mp4",
                    fps=12,
                )

            self.assertEqual(run.call_count, 1)
            self.assertEqual(
                [path.name for path in rollout_frame_paths(camera_dir)],
                ["step_000000.png", "step_000008.png", "step_000015.png"],
            )
            manifest = str(captured["manifest"])
            self.assertLess(manifest.index("step_000000.png"), manifest.index("step_000008.png"))
            self.assertLess(manifest.index("step_000008.png"), manifest.index("step_000015.png"))
            self.assertNotIn("step_invalid.png", manifest)
            command = list(captured["command"])
            self.assertEqual(command[command.index("-f") + 1], "concat")
            self.assertEqual(command[command.index("-vf") + 1], "setpts=N/(12*TB)")
            self.assertFalse(any("%06d" in item for item in command))
            self.assertTrue(dict(captured["kwargs"])["check"])
            self.assertFalse(Path(captured["manifest_path"]).exists())

    def test_missing_frames_are_reported_as_finalize_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_rollout_video_empty_") as temp_dir:
            root = Path(temp_dir)
            camera_dir = root / "head"
            camera_dir.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "no rollout step PNGs"):
                encode_rollout_video(camera_dir, root / "video" / "head.mp4")

    def test_sparse_frames_encode_as_three_video_frames(self) -> None:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg and ffprobe are required for the integration assertion")

        with tempfile.TemporaryDirectory(prefix="tcm_rollout_video_ffmpeg_") as temp_dir:
            root = Path(temp_dir)
            camera_dir = root / "head's camera"
            camera_dir.mkdir()
            for step, color in (
                (0, (255, 0, 0)),
                (8, (0, 255, 0)),
                (15, (0, 0, 255)),
            ):
                Image.new("RGB", (32, 32), color).save(camera_dir / f"step_{step:06d}.png")

            output_file = root / "video" / "head.mp4"
            result = encode_rollout_video(camera_dir, output_file, fps=10)

            self.assertIsNotNone(result)
            self.assertEqual(result.returncode, 0)
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=nw=1:nk=1",
                    str(output_file),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.stdout.strip(), "3")

    def test_img_agent_finalizes_all_calibrated_camera_outputs_with_shared_encoder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_rollout_video_agent_") as temp_dir:
            rollout_dir = Path(temp_dir)
            agent = object.__new__(ImgAgent)
            agent._rollout_dir = rollout_dir

            with mock.patch("policy.roboharn_evo.agent.core.img_agent.encode_rollout_video") as encode:
                agent.finalize_rollout_videos(fps=7)

            self.assertEqual(
                encode.call_args_list,
                [
                    mock.call(rollout_dir / "head", rollout_dir / "video" / "head.mp4", fps=7),
                    mock.call(rollout_dir / "left", rollout_dir / "video" / "left.mp4", fps=7),
                    mock.call(rollout_dir / "right", rollout_dir / "video" / "right.mp4", fps=7),
                    mock.call(rollout_dir / "third", rollout_dir / "video" / "third.mp4", fps=7),
                ],
            )

    def test_eval_fallback_uses_shared_encoder(self) -> None:
        rollout_dir = Path("/tmp/test_rollout")

        with mock.patch("policy.roboharn_evo.agent.rollout_video.encode_rollout_video") as encode:
            eval_policy._make_rollout_video(rollout_dir, "head", fps=9)

        encode.assert_called_once_with(
            rollout_dir / "head",
            rollout_dir / "video" / "head.mp4",
            fps=9,
        )


if __name__ == "__main__":
    unittest.main()
