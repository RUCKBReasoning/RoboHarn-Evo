from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download RoboBrain checkpoint into policy/roboharn_evo/checkpoints.")
    parser.add_argument("--repo-id", type=str, default="BAAI/RoboBrain2.5-8B-NV")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional local output directory. Defaults to policy/roboharn_evo/checkpoints/robobrain/<repo_name>.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    from huggingface_hub import snapshot_download

    args = parse_args()
    repo_name = args.repo_id.split("/")[-1]
    target_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "policy" / "roboharn_evo" / "checkpoints" / "robobrain" / repo_name
    )
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and args.force:
        shutil.rmtree(target_dir)
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=not args.force,
    )
    print(f"local_path={target_dir}")


if __name__ == "__main__":
    main()
