"""Package predictions into competition-ready submission ZIP."""

import zipfile
from pathlib import Path


def create_submission_zip(
    predictions_dir: str | Path,
    output_zip: str | Path,
    scale: int = 4,
) -> Path:
    """Create submission ZIP file from prediction directory.

    Expected structure:
        submission.zip/
            x4/
                HR_00.h5
                HR_01.h5
                ...

    Args:
        predictions_dir: Directory containing HR_XX.h5 prediction files.
        output_zip: Path for output ZIP file.
        scale: Scale factor (used for folder name, e.g., 'x4').
    Returns:
        Path to created ZIP file.
    """
    predictions_dir = Path(predictions_dir)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(predictions_dir.glob("HR_*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No HR_*.h5 files found in {predictions_dir}")

    folder_name = f"x{scale}"
    print(f"Creating submission ZIP: {output_zip}")
    print(f"  Including {len(h5_files)} prediction files in '{folder_name}/'")

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for h5_file in h5_files:
            arcname = f"{folder_name}/{h5_file.name}"
            zf.write(h5_file, arcname=arcname)
            print(f"  Added: {arcname}")

    size_mb = output_zip.stat().st_size / 1024 / 1024
    print(f"\nSubmission ZIP created: {output_zip} ({size_mb:.1f} MB)")
    return output_zip


def verify_submission(zip_path: str | Path, scale: int = 4) -> bool:
    """Verify the submission ZIP has the correct structure."""
    import h5py
    import numpy as np

    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"ERROR: ZIP not found: {zip_path}")
        return False

    folder_name = f"x{scale}"
    ok = True
    print(f"\nVerifying submission: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        h5_files = [n for n in names if n.startswith(f"{folder_name}/") and n.endswith(".h5")]

        if not h5_files:
            print(f"  ERROR: No .h5 files found under '{folder_name}/'")
            return False

        print(f"  Found {len(h5_files)} .h5 files under '{folder_name}/'")

        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            for h5_name in h5_files[:3]:  # Check first 3
                extracted = zf.extract(h5_name, tmpdir)
                try:
                    with h5py.File(extracted, "r") as f:
                        keys = list(f.keys())
                        if len(keys) != 1:
                            print(f"  WARNING: {h5_name} has {len(keys)} keys (expected 1)")
                        arr = f[keys[0]][:]
                        if arr.ndim != 3:
                            print(f"  ERROR: {h5_name} array has {arr.ndim} dims (expected 3)")
                            ok = False
                        else:
                            print(f"  OK: {h5_name} shape={arr.shape} (H, W, C)")
                except Exception as e:
                    print(f"  ERROR reading {h5_name}: {e}")
                    ok = False

    if ok:
        print("Submission verification PASSED")
    else:
        print("Submission verification FAILED - please check errors above")
    return ok
