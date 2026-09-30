from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp


def ensure_openpi_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    openpi_root = repo_root / "policy" / "roboharn_evo" / "third_party" / "openpi"
    openpi_src = openpi_root / "src"
    openpi_client_src = openpi_root / "packages" / "openpi-client" / "src"
    for path in (repo_root, openpi_root, openpi_src, openpi_client_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def create_trained_policy(
    train_config: Any,
    checkpoint_dir: Path | str,
    *,
    robotwin_repo_id: str | None = None,
    repack_transforms: Any | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, Any] | None = None,
    pytorch_device: str | None = None,
) -> Any:
    ensure_openpi_import_paths()

    import openpi.models.model as _model
    import openpi.policies.policy as _policy
    import openpi.shared.download as download
    from openpi.training import checkpoints as _checkpoints
    import openpi.transforms as transforms

    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading RoboHarn-Evo openpi policy.")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(Path(checkpoint_dir) / "params", dtype=jnp.bfloat16))

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if robotwin_repo_id is not None:
        data_config.asset_id = robotwin_repo_id
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(Path(checkpoint_dir) / "assets", data_config.asset_id)

    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


def build_pi05_robotwin_train_config(
    *,
    repo_id: str,
    exp_name: str,
    assets_base_dir: str,
    checkpoint_base_dir: str,
    overwrite: bool = False,
    resume: bool = False,
    wandb_enabled: bool = True,
    batch_size: int = 64,
    num_train_steps: int = 20_000,
    num_workers: int = 2,
    save_interval: int = 1000,
    keep_period: int = 5000,
    fsdp_devices: int = 1,
) -> Any:
    ensure_openpi_import_paths()
    repo_root = Path(__file__).resolve().parents[3]

    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as openpi_config
    import openpi.training.weight_loaders as weight_loaders
    import openpi.transforms as _transforms

    base_pi05_params = repo_root / "policy" / "roboharn_evo" / "checkpoints" / "pi05_base" / "params"

    return openpi_config.TrainConfig(
        name="pi05_aloha_robotwin_full",
        exp_name=exp_name,
        model=pi0_config.Pi0Config(pi05=True),
        data=openpi_config.LeRobotAlohaDataConfig(
            repo_id=repo_id,
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            base_config=openpi_config.DataConfig(
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(base_pi05_params)),
        assets_base_dir=assets_base_dir,
        checkpoint_base_dir=checkpoint_base_dir,
        batch_size=batch_size,
        num_train_steps=num_train_steps,
        num_workers=num_workers,
        save_interval=save_interval,
        keep_period=keep_period,
        fsdp_devices=fsdp_devices,
        overwrite=overwrite,
        resume=resume,
        wandb_enabled=wandb_enabled,
    )


def build_robotwin_runtime_config(
    *,
    repo_id: str,
    checkpoint_base_dir: str,
    assets_base_dir: str,
) -> Any:
    # Runtime still needs a TrainConfig object because openpi policy creation is config-driven.
    return build_pi05_robotwin_train_config(
        repo_id=repo_id,
        exp_name="runtime",
        checkpoint_base_dir=checkpoint_base_dir,
        assets_base_dir=assets_base_dir,
        overwrite=False,
        resume=False,
        wandb_enabled=False,
    )
