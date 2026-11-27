"""
Download the Lending Club accepted-loans dataset (2007 - 2018Q4) from Kaggle.

Source : https://www.kaggle.com/datasets/wordsforthewise/lending-club
File   : accepted_2007_to_2018Q4.csv.gz  (~390 MB compressed, ~1.6 GB raw)
Rows   : 2,260,668 loans (one row per loan application; member_id is blank)
Columns: 151

Lending Club was a US peer-to-peer lender (a genuine, SEC-registered lending
institution), so the data satisfies the "real lender, >=100k borrowers,
>=50 features" requirement of the project brief. data_prep keeps one loan per
borrower; src/borrower_check.py measures the repeat borrowers that may be left.

Requires a Kaggle API token in ~/.kaggle/kaggle.json
(Kaggle -> Account -> "Create New API Token").

Usage:
    python -m src.download_data
"""
import subprocess
import sys
from pathlib import Path

from src.config import RAW_DIR, RAW_FILE

KAGGLE_DATASET = "wordsforthewise/lending-club"
KAGGLE_FILE = "accepted_2007_to_2018Q4.csv.gz"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        print(f"[download] {RAW_FILE} already exists - skipping download.")
        return

    kaggle_bin = Path(sys.executable).parent / "kaggle"
    cmd = [str(kaggle_bin) if kaggle_bin.exists() else "kaggle",
           "datasets", "download", "-d", KAGGLE_DATASET,
           "-f", KAGGLE_FILE, "-p", str(RAW_DIR)]
    print("[download] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Kaggle sometimes wraps single-file downloads in a .zip
    zipped = RAW_DIR / (KAGGLE_FILE + ".zip")
    if zipped.exists():
        import zipfile
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(RAW_DIR)
        zipped.unlink()

    if not RAW_FILE.exists():
        sys.exit(f"[download] expected {RAW_FILE} but it was not found.")
    print(f"[download] saved {RAW_FILE} ({RAW_FILE.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
