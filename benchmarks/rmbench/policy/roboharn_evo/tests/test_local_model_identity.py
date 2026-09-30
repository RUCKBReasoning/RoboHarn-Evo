from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from policy.roboharn_evo.scripts.record_local_model_identity import (
    ArtifactValidationError,
    build_local_model_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "policy" / "roboharn_evo" / "scripts" / "record_local_model_identity.py"


def _write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> int:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, dtype, shape, tensor_bytes in tensors:
        start = len(payload)
        payload.extend(tensor_bytes)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    return len(payload)


class SyntheticModel:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model_dir = root / "model"
        self.model_dir.mkdir()
        (self.model_dir / "metadata").mkdir()
        (self.model_dir / "metadata" / "download-note.txt").write_text(
            "downloaded without credentials\n", encoding="utf-8"
        )
        (self.model_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["TinyVisionForConditionalGeneration"],
                    "model_type": "tiny_vision",
                    "text_config": {"model_type": "tiny_text", "dtype": "bfloat16"},
                    "vision_config": {
                        "model_type": "tiny_vision",
                        "depth": 2,
                        "hidden_size": 8,
                        "patch_size": 2,
                    },
                    "image_token_id": 17,
                    "quantization_config": {
                        "quant_method": "fp8",
                        "activation_scheme": "dynamic",
                        "weight_block_size": [128, 128],
                        "modules_to_not_convert": ["lm_head"],
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.model_dir / "LICENSE").write_text(
            "Apache License\nVersion 2.0, January 2004\n"
            "http://www.apache.org/licenses/\n",
            encoding="utf-8",
        )
        (self.model_dir / "README.md").write_text("# Tiny fixture\n", encoding="utf-8")
        first = self.model_dir / "model-00001-of-00002.safetensors"
        second = self.model_dir / "model-00002-of-00002.safetensors"
        first_payload = _write_safetensors(
            first,
            [("vision.weight", "F8_E4M3", [4], b"\x01\x02\x03\x04")],
        )
        second_payload = _write_safetensors(
            second,
            [("text.weight", "BF16", [2], b"\x05\x06\x07\x08")],
        )
        (self.model_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": first_payload + second_payload},
                    "weight_map": {
                        "vision.weight": first.name,
                        "text.weight": second.name,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def build(self, immutable_revision: str | None = None) -> dict:
        return build_local_model_identity(
            model_dir=self.model_dir,
            repo_id="Qwen/tiny-fixture",
            source_registry="ModelScope",
            revision_selector="master",
            immutable_revision=immutable_revision,
            revision_evidence="synthetic test fixture",
        )


class LocalModelIdentityTest(unittest.TestCase):
    def test_valid_synthetic_shards_record_complete_secret_free_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_model_identity_") as directory:
            fixture = SyntheticModel(Path(directory))
            with mock.patch.dict(os.environ, {"HF_TOKEN": "must-not-be-recorded"}):
                payload = fixture.build()

            expected_files = sorted(
                path for path in fixture.model_dir.rglob("*") if path.is_file()
            )
            expected_total = sum(path.stat().st_size for path in expected_files)
            config_sha256 = hashlib.sha256(
                (fixture.model_dir / "config.json").read_bytes()
            ).hexdigest()

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["source"]["registry"], "ModelScope")
        self.assertEqual(payload["source"]["revision_selector"], "master")
        self.assertIsNone(payload["source"]["immutable_revision"])
        self.assertEqual(payload["source"]["immutable_revision_status"], "not_available")
        self.assertEqual(payload["artifact"]["file_count"], len(expected_files))
        self.assertEqual(payload["artifact"]["total_file_bytes"], expected_total)
        self.assertEqual(len(payload["artifact"]["manifest_sha256"]), 64)
        self.assertIn("metadata/download-note.txt", payload["artifact"]["file_sizes_bytes"])
        self.assertEqual(
            payload["artifact"]["evidence_file_sha256"]["config.json"], config_sha256
        )
        self.assertEqual(payload["safetensors"]["shard_count"], 2)
        self.assertEqual(payload["safetensors"]["index_tensor_count"], 2)
        self.assertEqual(payload["safetensors"]["validated_payload_bytes"], 8)
        self.assertEqual(
            payload["safetensors"]["dtype_tensor_counts"], {"BF16": 1, "F8_E4M3": 1}
        )
        self.assertEqual(
            payload["safetensors"]["dtype_payload_bytes"], {"BF16": 4, "F8_E4M3": 4}
        )
        self.assertEqual(
            payload["model_metadata"]["architectures"],
            ["TinyVisionForConditionalGeneration"],
        )
        self.assertTrue(payload["model_metadata"]["vision"]["present"])
        self.assertEqual(payload["model_metadata"]["quantization"]["quant_method"], "fp8")
        self.assertEqual(payload["model_metadata"]["task"]["architecture_name_inference"],
                         "conditional_generation")
        self.assertTrue(
            all(payload["license_evidence"]["apache_2_0_file_markers"].values())
        )
        self.assertEqual(
            payload["license_review_status"], "file_present_not_legal_review"
        )
        self.assertFalse(payload["secrets_recorded"])
        self.assertNotIn("must-not-be-recorded", json.dumps(payload))

    def test_records_optional_immutable_revision_without_claiming_more(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_model_revision_") as directory:
            payload = SyntheticModel(Path(directory)).build("commit-0123456789abcdef")

        self.assertEqual(
            payload["source"]["immutable_revision"], "commit-0123456789abcdef"
        )
        self.assertEqual(payload["source"]["immutable_revision_status"], "recorded_from_cli")

    def test_rejects_relative_model_directory(self) -> None:
        with self.assertRaisesRegex(ArtifactValidationError, "absolute path"):
            build_local_model_identity(
                model_dir=Path("relative/model"),
                repo_id="Qwen/example",
                source_registry="ModelScope",
                revision_selector="master",
                immutable_revision=None,
                revision_evidence="not reached",
            )

    def test_rejects_incomplete_artifact_entry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_model_incomplete_") as directory:
            fixture = SyntheticModel(Path(directory))
            (fixture.model_dir / "model-00003.safetensors.part").write_bytes(b"unfinished")

            with self.assertRaisesRegex(ArtifactValidationError, "incomplete/temporary"):
                fixture.build()

    def test_rejects_missing_indexed_shard(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_model_missing_") as directory:
            fixture = SyntheticModel(Path(directory))
            (fixture.model_dir / "model-00002-of-00002.safetensors").unlink()

            with self.assertRaisesRegex(ArtifactValidationError, "shard is missing"):
                fixture.build()

    def test_rejects_unexpected_model_shard(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_model_unexpected_") as directory:
            fixture = SyntheticModel(Path(directory))
            _write_safetensors(
                fixture.model_dir / "model-00003-of-00003.safetensors",
                [("unexpected", "F32", [1], b"\x00\x00\x00\x00")],
            )

            with self.assertRaisesRegex(ArtifactValidationError, "unexpected model"):
                fixture.build()

    def test_rejects_payload_total_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_model_total_") as directory:
            fixture = SyntheticModel(Path(directory))
            index_path = fixture.model_dir / "model.safetensors.index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["metadata"]["total_size"] += 1
            index_path.write_text(json.dumps(index), encoding="utf-8")

            with self.assertRaisesRegex(ArtifactValidationError, "payload total"):
                fixture.build()

    def test_rejects_invalid_safetensors_header_length(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_model_header_") as directory:
            fixture = SyntheticModel(Path(directory))
            shard = fixture.model_dir / "model-00001-of-00002.safetensors"
            shard.write_bytes(struct.pack("<Q", 9999) + b"{}")

            with self.assertRaisesRegex(ArtifactValidationError, "header length"):
                fixture.build()

    def test_cli_writes_json_atomically_outside_model_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_model_cli_") as directory:
            root = Path(directory)
            fixture = SyntheticModel(root)
            output = root / "identity.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--model-dir",
                    str(fixture.model_dir),
                    "--repo-id",
                    "Qwen/tiny-fixture",
                    "--source-registry",
                    "ModelScope",
                    "--revision-selector",
                    "master",
                    "--revision-evidence",
                    "synthetic CLI fixture",
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            leftover_temps = list(root.glob(".identity.json.*.tmp"))

        self.assertEqual(payload["source"]["repo_id"], "Qwen/tiny-fixture")
        self.assertEqual(leftover_temps, [])

    def test_cli_rejects_self_referential_output_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tcm_model_selfref_") as directory:
            fixture = SyntheticModel(Path(directory))
            output = fixture.model_dir / "identity.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--model-dir",
                    str(fixture.model_dir),
                    "--repo-id",
                    "Qwen/tiny-fixture",
                    "--source-registry",
                    "ModelScope",
                    "--revision-selector",
                    "master",
                    "--revision-evidence",
                    "synthetic CLI fixture",
                    "--output",
                    str(output),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("self-referential", result.stderr)


if __name__ == "__main__":
    unittest.main()
