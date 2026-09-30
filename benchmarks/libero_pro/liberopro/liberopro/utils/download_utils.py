"""
Download functionalities adapted from Mandlekar et. al.: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/file_utils.py
"""
import os
import time
from tqdm import tqdm
from termcolor import colored
from pathlib import Path
import zipfile
import io
import urllib.request
import shutil

from liberopro.liberopro import get_libero_path

try:
    from huggingface_hub import snapshot_download
    import shutil
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

import liberopro.liberopro.utils.download_utils as download_utils
from liberopro.liberopro import get_libero_path


class DownloadProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def url_is_alive(url):
    """
    Checks that a given URL is reachable.
    From https://gist.github.com/dehowell/884204.
    Args:
        url (str): url string
    Returns:
        is_alive (bool): True if url is reachable, False otherwise
    """
    request = urllib.request.Request(url)
    # request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_url(url, download_dir, check_overwrite=True, is_zipfile=True):
    """
    First checks that @url is reachable, then downloads the file
    at that url into the directory specified by @download_dir.
    Prints a progress bar during the download using tqdm.
    Modified from https://github.com/tqdm/tqdm#hooks-and-callbacks, and
    https://stackoverflow.com/a/53877507.
    Args:
        url (str): url string
        download_dir (str): path to directory where file should be downloaded
        check_overwrite (bool): if True, will sanity check the download fpath to make sure a file of that name
            doesn't already exist there
    """

    # check if url is reachable. We need the sleep to make sure server doesn't reject subsequent requests
    assert url_is_alive(url), "@download_url got unreachable url: {}".format(url)
    time.sleep(0.5)

    # infer filename from url link
    fname = url.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    # If we're checking overwrite and the path already exists,
    # we ask the user to verify that they want to overwrite the file
    user_response = None
    if check_overwrite and os.path.exists(file_to_write):
        user_response = input(
            f"Warning: file {file_to_write} already exists. Overwrite? y/n\n"
        )
        # assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."

    if user_response is None or user_response.lower() in {"yes", "y"}:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=fname
        ) as t:
            urllib.request.urlretrieve(
                url, filename=file_to_write, reporthook=t.update_to
            )
    if is_zipfile:
        with zipfile.ZipFile(file_to_write, "r") as archive:
            archive.extractall(path=download_dir)
        if os.path.isfile(file_to_write):
            os.remove(file_to_write)


DATASET_LINKS = {
    "libero_object": "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip",
    "libero_goal": "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip",
    "libero_spatial": "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
    "libero_100": "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
}

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def download_from_huggingface(dataset_name, download_dir, check_overwrite=True):
    """
    Download a specific LIBERO dataset from Hugging Face.
    
    Args:
        dataset_name (str): Name of the dataset to download (e.g., 'libero_spatial')
        download_dir (str): Directory where the dataset should be downloaded
        check_overwrite (bool): If True, will check if dataset already exists
    """
    if not HUGGINGFACE_AVAILABLE:
        raise ImportError(
            "Hugging Face Hub is not available. Install it with 'pip install huggingface_hub'"
        )
    
    # Create the destination folder
    os.makedirs(download_dir, exist_ok=True)
    
    # Check if dataset already exists
    dataset_dir = os.path.join(download_dir, dataset_name)
    if check_overwrite and os.path.exists(dataset_dir):
        user_response = input(
            f"Warning: dataset {dataset_name} already exists at {dataset_dir}. Overwrite? y/n\n"
        )
        if user_response.lower() not in {"yes", "y"}:
            print(f"Skipping download of {dataset_name}")
            return
        
        # Remove existing directory
        print(f"Removing existing folder: {dataset_dir}")
        shutil.rmtree(dataset_dir)
    
    # Download the dataset
    print(f"Downloading {dataset_name} from Hugging Face...")
    folder_path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=f"{dataset_name}/*",
        local_dir_use_symlinks=False,  # Prevents using symlinks to cached files
        force_download=True  # Forces re-downloading files
    )
    
    # Verify downloaded files
    file_count = sum([len(files) for _, _, files in os.walk(os.path.join(download_dir, dataset_name))])
    print(f"Downloaded {file_count} files for {dataset_name}")


def libero_dataset_download(datasets="all", download_dir=None, check_overwrite=True, use_huggingface=False):
    """Download libero datasets

    Args:
        datasets (str, optional): Specify which datasets to save. Defaults to "all", downloading all the datasets.
        download_dir (str, optional): Target location for storing datasets. Defaults to None, using the default path.
        check_overwrite (bool, optional): Check if overwriting datasets. Defaults to True.
        use_huggingface (bool, optional): Use Hugging Face instead of the original download links. Defaults to False.
    """
    if download_dir is None:
        download_dir = get_libero_path("datasets")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    assert datasets in [
        "all",
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_100",
        "libero_10",
    ]

    datasets_to_download = [
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_100",
    ] if datasets == "all" else [datasets]

    for dataset_name in datasets_to_download:
        print(f"Downloading {dataset_name}")
        
        if use_huggingface:
            download_from_huggingface(
                dataset_name=dataset_name,
                download_dir=download_dir,
                check_overwrite=check_overwrite
            )
        else:
            print("Using original download links (these may expire soon)")
            download_url(
                DATASET_LINKS[dataset_name],
                download_dir=download_dir,
                check_overwrite=check_overwrite,
            )


def check_libero_dataset(download_dir=None):
    """Check the integrity of the downloaded datasets.

    Args:
        download_dir (str, optional): The path where datasets are stored. Defaults to None, using the default path.

    Returns:
        bool: True if the datasets are successfully downloaded, False otherwise.
    """
    if download_dir is None:
        download_dir = get_libero_path("datasets")
    check_result = True
    for dataset_name in [
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_10",
        "libero_90",
    ]:
        info_str = ""
        dataset_status = False
        dataset_dir = os.path.join(download_dir, dataset_name)
        if os.path.exists(dataset_dir):
            count = 0
            for path in Path(dataset_dir).glob("*.hdf5"):
                count += 1
            if (count == 10 and dataset_name != "libero_90") or (
                count == 90 and dataset_name == "libero_90"
            ):
                dataset_status = True
                info_str = colored(
                    f"[X] Dataset {dataset_name} is complete", "green", attrs=["bold"]
                )
            else:
                colored(
                    f"[?] Dataset {dataset_name} is not downloaded completely",
                    "yellow",
                    attrs=["bold"],
                )
        else:
            info_str = colored(
                f"[ ] Dataset {dataset_name} not found!!!", "red", attrs=["bold"]
            )

        print(info_str)
        check_result = check_result and dataset_status
    return check_result


# --------------------------------------------------------------------------- #
# Assets (mesh / texture / scene files).
#
# The assets tree (~617MB) is too large to ship inside the PyPI wheel, so when
# LIBERO-PRO is installed from PyPI it is downloaded separately from the Hugging
# Face Hub into the package's `liberopro/liberopro/assets` directory (where the
# environment code resolves asset files). Installs from a git checkout already
# contain the assets and do not need this step.
# --------------------------------------------------------------------------- #
HF_ASSETS_REPO_ID = os.environ.get("LIBERO_PRO_ASSETS_REPO", "RLinf/LIBERO-PRO-assets")
HF_ASSETS_REPO_TYPE = os.environ.get("LIBERO_PRO_ASSETS_REPO_TYPE", "dataset")

# If set, `liberopro-download-assets` symlinks the package assets dir to this
# already-downloaded assets tree instead of downloading (same as --link).
LIBERO_ASSET_PATH_ENV = "LIBERO_PRO_ASSET_PATH"


def _default_assets_dir():
    """Where assets must live: the package's own assets directory.

    Note: this is intentionally NOT read from the user config — most of the
    environment code (objects, scenes, bddl domains) resolves asset files
    relative to the installed package, so the package directory is the only
    location that always works. Use ``libero_assets_link`` to point it at an
    assets copy stored elsewhere.
    """
    from liberopro.liberopro import get_default_path_dict

    return os.path.abspath(get_default_path_dict()["assets"])


def assets_are_present(assets_dir=None):
    """Return True if the assets directory looks populated (has scenes)."""
    if assets_dir is None:
        assets_dir = _default_assets_dir()
    return os.path.isdir(os.path.join(assets_dir, "scenes"))


def _link_assets_dir(assets_dir, snapshot_path):
    """Make ``assets_dir`` a symlink pointing at the cached ``snapshot_path``."""
    assets_dir = os.path.abspath(assets_dir)
    os.makedirs(os.path.dirname(assets_dir), exist_ok=True)
    if os.path.islink(assets_dir):
        os.remove(assets_dir)
    elif os.path.isdir(assets_dir):
        if os.listdir(assets_dir):
            print(
                f"[Warning] Replacing existing assets directory {assets_dir} "
                "with a symlink into the Hugging Face cache."
            )
            shutil.rmtree(assets_dir)
        else:
            os.rmdir(assets_dir)
    os.symlink(snapshot_path, assets_dir)


def libero_assets_link(existing_assets_dir, assets_dir=None):
    """Point LIBERO-PRO at an already-downloaded assets directory (no download).

    Creates a symlink at the package's assets location (where the environment
    code looks for meshes/scenes) targeting ``existing_assets_dir``.
    """
    existing = os.path.abspath(os.path.expanduser(existing_assets_dir))
    if not assets_are_present(existing):
        raise FileNotFoundError(
            f"{existing} does not look like a LIBERO-PRO assets tree "
            "(missing a 'scenes' subdirectory)."
        )
    if assets_dir is None:
        assets_dir = _default_assets_dir()
    if os.path.realpath(assets_dir) == os.path.realpath(existing):
        print(f"Assets already resolve to {existing}; nothing to do.")
        return assets_dir
    _link_assets_dir(assets_dir, existing)
    print(f"LIBERO-PRO assets at {assets_dir} now point to {existing}")
    return assets_dir


def libero_assets_download(
    assets_dir=None,
    repo_id=None,
    repo_type=None,
    check_overwrite=True,
    use_cache=True,
):
    """Download the LIBERO-PRO simulation assets from the Hugging Face Hub.

    Args:
        assets_dir (str, optional): Target directory. Defaults to the package's
            ``liberopro/liberopro/assets`` directory.
        repo_id (str, optional): Hub repo id hosting the assets. Defaults to
            ``$LIBERO_PRO_ASSETS_REPO`` or ``RLinf/LIBERO-PRO-assets``.
        repo_type (str, optional): "dataset" (default) or "model".
        check_overwrite (bool, optional): Prompt before re-downloading when the
            assets already appear to be present. Defaults to True.
        use_cache (bool, optional): Download into the shared Hugging Face cache
            (respects ``HF_HOME``; resumable, deduped across environments) and
            expose it at ``assets_dir`` via a symlink. Set False to place a
            plain copy directly in ``assets_dir`` instead. Defaults to True.
    """
    if not HUGGINGFACE_AVAILABLE:
        raise ImportError(
            "Hugging Face Hub is not available. Install it with "
            "'pip install huggingface_hub'"
        )

    if assets_dir is None:
        assets_dir = _default_assets_dir()
    repo_id = repo_id or HF_ASSETS_REPO_ID
    repo_type = repo_type or HF_ASSETS_REPO_TYPE

    if check_overwrite and assets_are_present(assets_dir):
        user_response = input(
            f"Assets already present at {assets_dir}. Re-download? y/n\n"
        )
        if user_response.lower() not in {"yes", "y"}:
            print("Skipping asset download.")
            return assets_dir

    print(f"Downloading LIBERO-PRO assets from '{repo_id}' into {assets_dir} ...")
    try:
        if use_cache:
            snapshot_path = snapshot_download(repo_id=repo_id, repo_type=repo_type)
            _link_assets_dir(assets_dir, snapshot_path)
        else:
            os.makedirs(assets_dir, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=assets_dir,
            )
    except Exception as e:  # noqa: BLE001 - surface a clear, actionable message
        raise RuntimeError(
            f"Failed to download assets from '{repo_id}' (type '{repo_type}').\n"
            f"Original error: {e}\n"
            "Set LIBERO_PRO_ASSETS_REPO to a valid Hugging Face repo that hosts "
            "the LIBERO-PRO assets tree, or install LIBERO-PRO from a git "
            "checkout that already contains liberopro/liberopro/assets."
        ) from e

    print(f"LIBERO-PRO assets ready at {assets_dir}")
    return assets_dir


def _assets_cli():
    """Console entry point: ``liberopro-download-assets``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download LIBERO-PRO simulation assets from the Hugging Face Hub."
    )
    parser.add_argument("--assets-dir", type=str, default=None,
                        help="Target directory (default: the package assets dir).")
    parser.add_argument("--repo-id", type=str, default=None,
                        help=f"HF repo id (default: {HF_ASSETS_REPO_ID}).")
    parser.add_argument("--repo-type", type=str, default=None,
                        choices=["dataset", "model"])
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if assets already exist.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Place a plain copy in the assets dir instead of "
                             "symlinking into the shared Hugging Face cache.")
    parser.add_argument("--link", type=str, default=None, metavar="EXISTING_DIR",
                        help="Skip downloading; symlink the package assets dir "
                             "to an assets tree that already exists at this path. "
                             f"Falls back to ${LIBERO_ASSET_PATH_ENV} if set.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Exit without prompting if assets are already "
                             "present (for non-interactive installs).")
    args = parser.parse_args()
    if args.skip_existing and assets_are_present(args.assets_dir):
        print("LIBERO-PRO assets already present; skipping download.")
        return
    link_target = args.link or os.environ.get(LIBERO_ASSET_PATH_ENV)
    if link_target:
        if args.link is None:
            print(f"Using {LIBERO_ASSET_PATH_ENV}={link_target}")
        libero_assets_link(link_target, assets_dir=args.assets_dir)
        return
    libero_assets_download(
        assets_dir=args.assets_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        check_overwrite=not args.force,
        use_cache=not args.no_cache,
    )
