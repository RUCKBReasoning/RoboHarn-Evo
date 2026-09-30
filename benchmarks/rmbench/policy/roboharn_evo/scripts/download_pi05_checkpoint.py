from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CHECKPOINT_URIS = {
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
    "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
    "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
}

TARGET_DIRS = {
    "pi05_base": REPO_ROOT / "policy" / "roboharn_evo" / "checkpoints" / "pi05_base",
    "pi05_droid": REPO_ROOT / "policy" / "roboharn_evo" / "checkpoints" / "pi05_droid",
    "pi05_libero": REPO_ROOT / "policy" / "roboharn_evo" / "checkpoints" / "pi05_libero",
}


def _ensure_openpi_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    openpi_src_dir = repo_root / "policy" / "roboharn_evo" / "third_party" / "openpi" / "src"
    openpi_client_src = repo_root / "policy" / "roboharn_evo" / "third_party" / "openpi" / "packages" / "openpi-client" / "src"
    for path in (repo_root, openpi_src_dir, openpi_client_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download an official pi0.5 checkpoint into the local openpi cache.")
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(CHECKPOINT_URIS.keys()),
        default="pi05_base",
        help="Official checkpoint alias to download.",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=None,
        help="Optional custom checkpoint URI. Overrides --model when set.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional local output directory. Defaults to policy/roboharn_evo/checkpoints/<model>.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even when the checkpoint is already cached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_openpi_import_path()
    from openpi.shared import download

    uri = args.uri or CHECKPOINT_URIS[args.model]
    cached_path = download.maybe_download(uri, force_download=bool(args.force))
    target_dir = Path(args.output_dir) if args.output_dir else TARGET_DIRS[args.model]
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists() and args.force:
        shutil.rmtree(target_dir)
    if not target_dir.exists():
        shutil.copytree(cached_path, target_dir)
    print(f"remote_uri={uri}")
    print(f"cached_path={cached_path}")
    print(f"local_path={target_dir}")


if __name__ == "__main__":
    main()
