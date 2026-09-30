import os
import numpy as np
import argparse
import traceback
import hashlib
from pathlib import Path

from benchmarks.rmbench.paths import assets_root, output_root
from benchmarks.rmbench.runtime_assets import (
    readonly_asset_boundaries,
    require_writable_root_outside,
)

try:
    import trimesh
except ImportError:  # pragma: no cover - conversion-only optional dependency
    trimesh = None


def converted_glb_path(source_dir, output_path=None):
    source = Path(source_dir).expanduser().resolve()
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
    else:
        digest = hashlib.sha256(os.fsencode(source)).hexdigest()[:16]
        destination = (
            output_root()
            / "runtime_assets"
            / "converted_glb"
            / f"{source.name}-{digest}"
            / "base0.glb"
        ).resolve()
    if destination == source or source in destination.parents:
        raise ValueError(
            "converted GLB destination must be outside the read-only source directory"
        )
    require_writable_root_outside(
        destination.parent,
        readonly_roots=readonly_asset_boundaries(),
        label="converted GLB output",
    )
    return destination


def convert_obj_glb(source_dir, output_path=None):
    """
    Convert all OBJ files in the given source directory to a single GLB file.

    Args:
        source_dir: Directory containing OBJ files
        output_path: Explicit writable GLB destination. Defaults under the
            configured RMBench output root, never inside source_dir.

    Returns:
        bool: True if successful, False if an error occurs
    """
    try:
        if trimesh is None:
            raise RuntimeError(
                "trimesh is required to convert OBJ assets; install the RMBench "
                "simulator dependencies before running this utility"
            )
        source_path = Path(source_dir).expanduser().resolve()
        texture_dir = source_path / "textured_objs"
        destination = converted_glb_path(source_path, output_path)
        if destination.exists():
            print(f"File {destination} already exists")
            return True
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Create a scene to hold all meshes
        scene = trimesh.Scene()

        # Find all .obj files in the directory
        obj_files = [f for f in os.listdir(texture_dir) if f.endswith(".obj")]

        # Load each OBJ file and add it to the scene
        for obj_file in obj_files:
            file_path = texture_dir / obj_file
            try:
                with open(file_path, "rb") as file_obj:
                    mesh = trimesh.load(file_obj, file_type="obj")
                scene.add_geometry(mesh)
                # print(f"Added mesh from {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return False

        # Export the scene as GLB
        print(f"Exporting scene to {destination}...")
        scene.export(os.fspath(destination))
        print(f"Model successfully exported to {destination}")
        return True
    except Exception as e:
        print(f"An error occurred in convert_to_glb: {e}" + traceback.format_exc())
        return False


def is_digital(name):
    """Check if a string contains only digits."""
    return name.isdigit()


def has_only_digital_subdirs(directory):
    """Check if a directory contains only subdirectories with digital names."""
    if not os.path.isdir(directory):
        return False

    subdirs = [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]

    # Return True if there are subdirs and all of them are digital
    return len(subdirs) > 0 and all(is_digital(subdir) for subdir in subdirs)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert OBJ files to GLB.")
    parser.add_argument(
        "--object_dir",
        type=str,
        help="Directory containing single object (e.g., assets/objects/060_kitchenpot)",
    )
    parser.add_argument(
        "--scan_all",
        action="store_true",
        help="Scan all objects in assets/objects directory",
    )
    args = parser.parse_args()

    total_conversions = 0

    assets_path = (
        Path(args.object_dir).expanduser().resolve()
        if args.object_dir
        else assets_root() / "objects"
    )
    # Process each object directory in assets/objects
    for obj_dir in os.listdir(assets_path):
        obj_path = os.path.join(assets_path, obj_dir)

        # Check if it's a directory and has only digital subdirectories
        if os.path.isdir(obj_path) and has_only_digital_subdirs(obj_path):
            print(obj_path)
            # for final_path in os.listdir(obj_path):
            #     convert_obj_glb(os.path.join(obj_path, final_path))

    print(f"\nTotal completed GLB conversions: {total_conversions}")
