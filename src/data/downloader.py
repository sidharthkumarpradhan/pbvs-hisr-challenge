"""Download training/validation dataset from Zenodo and test set from Google Drive."""
import os
import re
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_ARCHIVE_URL = "https://zenodo.org/api/records/18171738/files-archive"
ZENODO_FILES_URL = "https://zenodo.org/api/records/18171738/files"

# Google Drive file ID for test LR data ZIP
TEST_GDRIVE_FILE_ID = "1RAixhOm7HAeLPE8J4ZPZFh7LjRxeBjNk"


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


def _download_gdrive_file(file_id: str, dest: Path, desc: str = "Downloading") -> Path:
    """Download a file from Google Drive with virus scan bypass for large files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()

    response = session.get(url, stream=True, timeout=60)
    # Check if Google Drive returns a virus scan warning page
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
            response = session.get(url, stream=True, timeout=60)
            break

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


def _get_zenodo_file_list() -> list:
    """Return list of file metadata dicts from Zenodo API.

    The Zenodo /files endpoint returns either:
      - A dict with an 'entries' key: {"entries": [{"key": ..., "links": {...}}, ...]}
      - A list directly (older API format): [{"key": ..., "links": {...}}, ...]
    This function normalises both into a plain list of dicts.
    """
    response = requests.get(ZENODO_FILES_URL, timeout=30)
    response.raise_for_status()
    data = response.json()
    # New Zenodo InvenioRDM API wraps results in {"entries": [...]}
    if isinstance(data, dict):
        return data.get("entries", [])
    # Old API returns a bare list
    if isinstance(data, list):
        return data
    return []


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

    if not files:
        print("No files returned from Zenodo API. Falling back to archive download...")
        download_zenodo_archive(data_dir)
        return

    print(f"Found {len(files)} files to download.")
    for file_info in files:
        # file_info must be a dict at this point
        if not isinstance(file_info, dict):
            print(f"  Unexpected file entry format: {file_info!r}, skipping.")
            continue

        # 'key' is the filename in the new API; older API used 'filename'
        fname = file_info.get("key") or file_info.get("filename") or "unknown"

        # Download URL: new API uses links.content, older uses links.self or download
        links = file_info.get("links", {})
        furl = (
            links.get("content")
            or links.get("self")
            or file_info.get("download", "")
        )

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


def download_test_data_gdrive(test_dir: Path, file_id: str = TEST_GDRIVE_FILE_ID) -> bool:
    """Download test LR data from Google Drive.

    Returns True if download succeeded, False otherwise.
    """
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Check if test data already exists
    existing = list(test_dir.glob("*.h5"))
    if existing:
        print(f"Test data already present: {len(existing)} files in {test_dir}")
        return True

    print(f"Downloading test LR data from Google Drive...")
    zip_path = test_dir / "test_lr.zip"
    try:
        _download_gdrive_file(file_id, zip_path, desc="Test LR data")
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(test_dir)
        zip_path.unlink()  # Remove zip after extraction
        test_files = list(test_dir.glob("*.h5"))
        print(f"Test data extracted: {len(test_files)} files")
        return True
    except Exception as e:
        print(f"ERROR downloading test data: {e}")
        print("\nPlease manually download from:")
        print(f"  https://drive.google.com/file/d/{file_id}/view")
        print(f"and extract the .h5 files to: {test_dir.absolute()}")
        return False


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
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 0,
    )
    if len(scene_files) == 0:
        # Also try without the Scene_ prefix in case filenames differ
        scene_files = sorted(
            [f for f in raw_dir.glob("*.h5")],
            key=lambda p: p.stem,
        )

    if len(scene_files) == 0:
        print("No .h5 files found in raw dir. Check download.")
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
        print(
            f"Dataset already present: {len(existing_train)} train, "
            f"{len(existing_val)} val scenes."
        )
        return

    existing_raw = list(raw_dir.glob("*.h5"))
    if len(existing_raw) < 5:
        print("Dataset not found. Starting download from Zenodo...")
        download_zenodo_individual(data_dir)

    print("Organizing dataset into train/val splits...")
    split_scenes(raw_dir, train_dir, val_dir)
    print("Dataset ready.")


def verify_test_data(test_dir: Path) -> bool:
    """Check if test LR data exists, download if not."""
    test_dir = Path(test_dir)
    test_files = list(test_dir.glob("*.h5"))
    if test_files:
        print(f"Test data found: {len(test_files)} files in {test_dir}")
        return True

    print(f"\nTest LR data not found in {test_dir}")
    print("Attempting to download from Google Drive...")
    return download_test_data_gdrive(test_dir)
