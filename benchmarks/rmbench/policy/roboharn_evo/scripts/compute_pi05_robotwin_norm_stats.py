from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.utils.openpi_utils import build_pi05_robotwin_train_config, ensure_openpi_import_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute norm stats for RoboHarn-Evo-local pi05 RobotWin training.")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--exp-name", type=str, default="norm_stats_only")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    ensure_openpi_import_paths()
    import openpi.shared.normalize as normalize
    import openpi.training.data_loader as _data_loader

    config = build_pi05_robotwin_train_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        assets_base_dir=str(repo_root / "policy" / "roboharn_evo" / "assets" / "pi05"),
        checkpoint_base_dir=str(repo_root / "policy" / "roboharn_evo" / "checkpoints" / "pi05"),
        overwrite=False,
        resume=False,
        wandb_enabled=False,
        batch_size=args.batch_size,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ],
    )

    num_frames = len(dataset) if args.max_frames is None else min(int(args.max_frames), len(dataset))
    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for index in range(num_frames):
        batch = dataset[index]
        for key in keys:
            stats[key].update(batch[key][None, ...])

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_path = config.assets_dirs / args.repo_id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    main()
