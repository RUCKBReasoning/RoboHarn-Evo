from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch

from policy.roboharn_evo.scripts.serve_sam3_segmentation import SAM3SegmentationHandler


def _bare_handler() -> SAM3SegmentationHandler:
    return object.__new__(SAM3SegmentationHandler)


def test_health_reports_orchestration_identity_and_artifact_root(tmp_path: Path) -> None:
    handler = _bare_handler()
    handler.path = "/health"
    handler.checkpoint = "/models/sam3.pt"
    handler.device = "cuda"
    handler.amp_dtype = "bfloat16"
    handler.confidence_threshold = 0.5
    handler.instance_id = "parallel4-slot-2"
    handler.output_root = tmp_path / "slot_2"
    handler.allowed_output_root = tmp_path / "slot_2"

    with mock.patch.object(handler, "_send_json") as send_json, mock.patch.object(
        torch.cuda, "is_available", return_value=True
    ), mock.patch.object(torch.cuda, "current_device", return_value=0), mock.patch.dict(
        os.environ,
        {
            "CUDA_VISIBLE_DEVICES": "GPU-abcd",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        },
    ):
        handler.do_GET()

    payload = send_json.call_args.args[0]
    assert payload["status"] == "ok"
    assert payload["backend"] == "sam3_image_text_prompt"
    assert payload["instance_id"] == "parallel4-slot-2"
    assert payload["output_root"] == str(tmp_path / "slot_2")
    assert payload["allowed_output_root"] == str(tmp_path / "slot_2")
    assert payload["device"] == "cuda"
    assert payload["cuda_available"] is True
    assert payload["cuda_current_device"] == 0
    assert payload["cuda_visible_devices"] == "GPU-abcd"
    assert payload["cuda_device_order"] == "PCI_BUS_ID"
    assert isinstance(payload["pid"], int) and payload["pid"] > 0


def test_handler_serializes_mutable_processor_state_within_one_instance(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(image_path)

    class Processor:
        def __init__(self) -> None:
            self.confidence_threshold = 0.5
            self.active = 0
            self.max_active = 0
            self.guard = threading.Lock()

        def set_image(self, _image: Image.Image) -> dict[str, object]:
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            return {}

        def set_text_prompt(
            self, *, state: dict[str, object], prompt: str
        ) -> dict[str, object]:
            del state, prompt
            time.sleep(0.05)
            with self.guard:
                self.active -= 1
            return {
                "masks": np.ones((1, 1, 4, 4), dtype=bool),
                "boxes": np.asarray([[0, 0, 3, 3]], dtype=np.float32),
                "scores": np.asarray([0.9], dtype=np.float32),
            }

    processor = Processor()
    old_processor = SAM3SegmentationHandler.processor
    old_output_root = SAM3SegmentationHandler.output_root
    old_device = SAM3SegmentationHandler.device
    old_amp_dtype = SAM3SegmentationHandler.amp_dtype
    try:
        SAM3SegmentationHandler.processor = processor
        SAM3SegmentationHandler.output_root = tmp_path / "output"
        SAM3SegmentationHandler.device = "cpu"
        SAM3SegmentationHandler.amp_dtype = "none"
        errors: list[BaseException] = []

        def invoke(index: int) -> None:
            try:
                _bare_handler()._segment_image(
                    {
                        "image_path": str(image_path),
                        "text_prompt": "block",
                        "object_id": f"block_{index}",
                        "confidence_threshold": 0.4 + (index * 0.1),
                    }
                )
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=invoke, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        assert not errors
        assert all(not thread.is_alive() for thread in threads)
        assert processor.max_active == 1
    finally:
        SAM3SegmentationHandler.processor = old_processor
        SAM3SegmentationHandler.output_root = old_output_root
        SAM3SegmentationHandler.device = old_device
        SAM3SegmentationHandler.amp_dtype = old_amp_dtype


def test_inference_lock_is_shared_across_threaded_request_handlers() -> None:
    first = _bare_handler()
    second = _bare_handler()

    assert first.inference_lock is second.inference_lock


def test_handler_rejects_output_outside_instance_root(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(image_path)
    handler = _bare_handler()
    handler.output_root = tmp_path / "slot_0"
    handler.allowed_output_root = tmp_path / "slot_0"
    handler.processor = object()

    with pytest.raises(ValueError, match="within this SAM3 instance allowed_output_root"):
        handler._segment_image(
            {
                "image_path": str(image_path),
                "text_prompt": "block",
                "output_dir": str(tmp_path / "slot_1"),
            }
        )


def test_request_threshold_is_reset_to_instance_default(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(image_path)

    class Processor:
        confidence_threshold = 0.5

        def set_image(self, _image: Image.Image) -> dict[str, object]:
            assert self.confidence_threshold == 0.25
            return {}

        def set_text_prompt(
            self, *, state: dict[str, object], prompt: str
        ) -> dict[str, object]:
            del state, prompt
            return {
                "masks": np.ones((1, 1, 4, 4), dtype=bool),
                "boxes": np.asarray([[0, 0, 3, 3]], dtype=np.float32),
                "scores": np.asarray([0.9], dtype=np.float32),
            }

    handler = _bare_handler()
    handler.processor = Processor()
    handler.output_root = tmp_path / "slot_0"
    handler.allowed_output_root = tmp_path / "slot_0"
    handler.device = "cpu"
    handler.amp_dtype = "none"
    handler.confidence_threshold = 0.5

    handler._segment_image(
        {
            "image_path": str(image_path),
            "text_prompt": "block",
            "output_dir": str(handler.output_root / "episode_0"),
            "confidence_threshold": 0.25,
        }
    )

    assert handler.processor.confidence_threshold == 0.5


def test_legacy_single_service_allows_explicit_external_output_dir(
    tmp_path: Path,
) -> None:
    """No allowed root keeps old hand-started service/client paths working."""
    image_path = tmp_path / "input.png"
    Image.new("RGB", (4, 4), color=(0, 0, 0)).save(image_path)

    class Processor:
        confidence_threshold = 0.5

        def set_image(self, _image: Image.Image) -> dict[str, object]:
            return {}

        def set_text_prompt(
            self, *, state: dict[str, object], prompt: str
        ) -> dict[str, object]:
            del state, prompt
            return {
                "masks": np.ones((1, 1, 4, 4), dtype=bool),
                "boxes": np.asarray([[0, 0, 3, 3]], dtype=np.float32),
                "scores": np.asarray([0.9], dtype=np.float32),
            }

    handler = _bare_handler()
    handler.processor = Processor()
    handler.output_root = tmp_path / "legacy_default"
    handler.allowed_output_root = None
    handler.device = "cpu"
    handler.amp_dtype = "none"
    handler.confidence_threshold = 0.5
    external = tmp_path / "legacy_client_artifacts" / "masks"

    result = handler._segment_image(
        {
            "image_path": str(image_path),
            "text_prompt": "block",
            "output_dir": str(external),
        }
    )

    assert result["success"] is True
    assert Path(result["mask_path"]).parent == external.resolve()
