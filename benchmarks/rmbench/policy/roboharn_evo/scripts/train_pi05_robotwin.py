from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy.roboharn_evo.utils.openpi_utils import build_pi05_robotwin_train_config, ensure_openpi_import_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RoboHarn-Evo-local pi05 on RobotWin/LeRobot data.")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-train-steps", type=int, default=20000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-period", type=int, default=5000)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    ensure_openpi_import_paths()
    import scripts.train as openpi_train

    config = build_pi05_robotwin_train_config(
        repo_id=args.repo_id,
        exp_name=args.exp_name,
        assets_base_dir=str(repo_root / "policy" / "roboharn_evo" / "assets" / "pi05"),
        checkpoint_base_dir=str(repo_root / "policy" / "roboharn_evo" / "checkpoints" / "pi05"),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        wandb_enabled=not bool(args.disable_wandb),
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        num_workers=args.num_workers,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        fsdp_devices=args.fsdp_devices,
    )
    openpi_train.main(config)


if __name__ == "__main__":
    main()
