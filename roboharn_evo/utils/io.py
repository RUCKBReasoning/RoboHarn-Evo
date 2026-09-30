from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict), f"YAML must deserialize to dict: {yaml_path}"
    return payload


def load_jsonl(paths: list[str] | list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like)
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                assert isinstance(row, dict), f"Invalid JSONL row in {path}"
                rows.append(row)
    return rows


def parse_episode_id(episode_id: str) -> tuple[str, int]:
    task_name, episode_part = episode_id.split("/", 1)
    episode_index = int(episode_part.replace("episode", ""))
    return task_name, episode_index


def decode_rgb_frame(encoded_frame: bytes | np.bytes_) -> np.ndarray:
    image = Image.open(BytesIO(bytes(encoded_frame)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def resize_rgb(image: np.ndarray, image_size: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def image_to_tensor(image: np.ndarray, image_size: int) -> Any:
    import torch

    resized = resize_rgb(image=image, image_size=image_size)
    image_float = resized.astype(np.float32) / 255.0
    # image_float: (H, W, 3)
    image_chw = np.transpose(image_float, (2, 0, 1))
    return torch.from_numpy(image_chw)
