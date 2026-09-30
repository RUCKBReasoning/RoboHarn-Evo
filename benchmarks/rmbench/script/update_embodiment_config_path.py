#!/usr/bin/env python3
import argparse
import os
import glob

from benchmarks.rmbench.paths import ASSETS_ROOT_ENV, assets_root
from benchmarks.rmbench.runtime_assets import (
    RUNTIME_CONFIG_ROOT_ENV,
    materialize_curobo_config,
    runtime_config_root,
)

def print_color(message, color_code):
    NC = '\033[0m'
    print(f"{color_code}{message}{NC}")

BLUE = '\033[0;34m'
YELLOW = '\033[0;33m'
GREEN = '\033[0;32m'


def main(asset_root_arg=None, runtime_config_root_arg=None):
    assets_path = assets_root(asset_root_arg)
    destination_root = runtime_config_root(runtime_config_root_arg)
    embodiments_path = assets_path / "embodiments"
    print_color(f"Assets root: {assets_path}", BLUE)
    print_color(f"Runtime config root: {destination_root}", BLUE)
    if not embodiments_path.is_dir():
        print_color(f"Error: embodiments directory not found: {embodiments_path}", YELLOW)
        print(f"Pass --assets-root or set {ASSETS_ROOT_ENV} to the RMBench assets directory.")
        return 1

    # Export environment variable
    os.environ['ASSETS_PATH'] = os.fspath(assets_path)
    print_color(f"Setting environment variable: ASSETS_PATH={assets_path}", BLUE)

    # Counters
    count_total = count_updated = count_error = 0

    # Find *_tmp.yml files
    print_color("Searching for configuration template files...", BLUE)
    pattern = os.path.join(embodiments_path, '**', '*_tmp.yml')
    config_files = glob.glob(pattern, recursive=True)

    if not config_files:
        print_color("No *_tmp.yml files found", YELLOW)
        return 1

    print_color("Starting to process configuration files...", BLUE)
    for tmp_file in config_files:
        count_total += 1
        source_file = tmp_file.replace('_tmp.yml', '.yml')
        print(f"Processing [{count_total}]: {tmp_file}")

        try:
            target_file = materialize_curobo_config(
                source_file,
                asset_root=assets_path,
                destination_root=destination_root,
            )
            print_color(
                f"  ✓ Materialized read-only asset template -> {target_file}",
                GREEN,
            )
            count_updated += 1

        except Exception as e:
            print_color(f"  ✗ Replacement failed: {e}", YELLOW)
            count_error += 1

    # Summary
    print()
    print_color("Processing complete!", BLUE)
    print(f"Total processed: {count_total} files")
    print_color(f"Successfully updated: {count_updated} files", GREEN)
    if count_error > 0:
        print_color(f"Failed to process: {count_error} files", YELLOW)

    print()
    print_color("All template files have been processed!", GREEN)
    print(f"External assets were not modified; configs are under {destination_root}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Materialize RMBench embodiment configs for an external assets root."
    )
    parser.add_argument(
        "--assets-root",
        help=f"Assets directory (defaults to ${ASSETS_ROOT_ENV}).",
    )
    parser.add_argument(
        "--runtime-config-root",
        help=(
            "Writable destination for materialized YAML files "
            f"(defaults to ${RUNTIME_CONFIG_ROOT_ENV} or RMBench output/runtime_configs)."
        ),
    )
    cli_args = parser.parse_args()
    raise SystemExit(main(cli_args.assets_root, cli_args.runtime_config_root))
