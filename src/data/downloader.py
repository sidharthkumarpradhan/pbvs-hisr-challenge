"""Download training/validation dataset from Zenodo and test set from provided URL."""

import os
import re
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_ARCHIVE_URL = "https://zenodo.org/api/records/18171738/files-archive"
ZENODO_FILES_URL = "https://zenodo.org/api/records/18171738/files"


def _download_file(url: str, dest: Path, desc: str = "Downloading") -> Path:
    """Stream-download a file to dest with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=desc,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def _get_zenodo_file_list() -> list[dict]:
    """Return list of file metadata dicts from Zenodo API."""
    response = requests.get(ZENODO_FILES_URL, timeout=30)
    response.raise_for_status()
    return response.json()


def download_zenodo_individual(data_dir: Path) -> None:
    """Download Zenodo files individually (preferred - avoids 8GB single download)."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching file list from Zenodo...")
    try:
        files = _get_zenodo_file_list()
    except Exception as e:
        print(f"Could not fetch Zenodo file list: {e}")
        print("Falling back to full archive download...")
        download_zenodo_archive(data_dir)
        return

    print(f"Found {len(files)} files to download.")
    for file_info in files:
        fname = file_info.get("key", file_info.get("filename", "unknown"))
        furl = file_info.get("links", {}).get("self", file_info.get("download", ""))
        dest = raw_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  Skipping {fname} (already exists)")
            continue
        if not furl:
            print(f"  Skipping {fname} (no download URL)")
            continue
        print(f"  Downloading {fname}...")
        try:
            _download_file(furl, dest, desc=fname)
        except Exception as e:
            print(f"  ERROR downloading {fname}: {e}")


def download_zenodo_archive(data_dir: Path) -> None:
    """Download the full Zenodo archive ZIP and extract it."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "zenodo_archive.zip"

    if not archive_path.exists():
        print(f"Downloading full Zenodo archive (~8.6 GB) to {archive_path}...")
        _download_file(ZENODO_ARCHIVE_URL, archive_path, desc="Zenodo Archive")
    else:
        print(f"Archive already exists at {archive_path}")

    print("Extracting archive...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(raw_dir)
    print(f"Extracted to {raw_dir}")


def split_scenes(raw_dir: Path, train_dir: Path, val_dir: Path) -> None:
    """Split downloaded scene files into train/val directories."""
    import shutil

    raw_dir = Path(raw_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    scene_files = sorted(
        [f for f in raw_dir.glob("Scene_*.h5")],
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
    )
    if len(scene_files) == 0:
        print("No Scene_*.h5 files found in raw dir. Check download.")
        return

    print(f"Found {len(scene_files)} scene files.")
    n_train = max(1, len(scene_files) - 10)
    train_files = scene_files[:n_train]
    val_files = scene_files[n_train:]

    for f in train_files:
        dest = train_dir / f.name
        if not dest.exists():
            shutil.copy2(f, dest)

    for f in val_files:
        dest = val_dir / f.name
        if not dest.exists():
            shutil.copy2(f, dest)

    print(f"Split: {len(train_files)} train / {len(val_files)} val scenes")


def check_or_download_dataset(cfg: dict) -> None:
    """Main function to ensure training data is available."""
    data_cfg = cfg["data"]
    data_dir = Path(data_cfg["data_dir"])
    train_dir = Path(data_cfg["train_dir"])
    val_dir = Path(data_cfg["val_dir"])
    raw_dir = data_dir / "raw"

    existing_train = list(train_dir.glob("*.h5"))
    existing_val = list(val_dir.glob("*.h5"))

    if len(existing_train) >= 5 and len(existing_val) >= 1:
        print(f"Dataset already present: {len(existing_train)} train, {len(existing_val)} val scenes.")
        return

    existing_raw = list(raw_dir.glob("Scene_*.h5"))
    if len(existing_raw) < 5:
        print("Dataset not found. Starting download from Zenodo...")
        download_zenodo_individual(data_dir)

    print("Organizing dataset into train/val splits...")
    split_scenes(raw_dir, train_dir, val_dir)
    print("Dataset ready.")


def verify_test_data(test_dir: Path) -> bool:
    """Check if test LR data exists."""
    test_dir = Path(test_dir)
    test_files = list(test_dir.glob("*.h5"))
    if test_files:
        print(f"Test data found: {len(test_files)} files in {test_dir}")
        return True
    print(f"\nNOTE: Test LR data not found in {test_dir}")
    print("Please manually download the test set from:")
    print("  https://filesender.belnet.be/?s=download&token=237989c4-8c1a-4bd8-b652-49e5dbd90778")
    print(f"and place the .h5 files in: {test_dir.absolute()}")
    return False
