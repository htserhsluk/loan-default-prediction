"""Check borrower uniqueness in the lender archive before enriching the cohort."""
import hashlib
import shutil
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

REVISION = "66a86bb5ddbe6d698fbe4b0180b1475b591a632c"
BASE = f"https://raw.githubusercontent.com/ChicagoBoothML/DATA___LendingClub/{REVISION}/"
FILES = ["LoanStats3b.csv.zip", "LoanStats3c.csv.zip", "LoanStats3d.csv.zip"]
KEYS = ["issue_d", "term", "loan_amnt", "installment", "int_rate", "sub_grade", "purpose",
        "open_acc", "total_acc", "delinq_2yrs", "pub_rec"]

def fetch():
    folder = Path("data")
    folder.mkdir(exist_ok=True)
    for name in FILES + ["LCDataDictionary.xlsx"]:
        path = folder / name
        if not path.exists():
            temporary = path.with_suffix(".part")
            with urlopen(BASE + name, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(path)

def normalize(frame):
    frame = frame.copy()
    for field in ["loan_amnt", "installment", "int_rate", "open_acc", "total_acc", "delinq_2yrs", "pub_rec"]:
        frame[field] = pd.to_numeric(frame[field].str.strip().str.rstrip("%"), errors="coerce").round(2)
    frame["term"] = frame.term.str.strip()
    return frame

def verified_positions(archive, local):
    unique_archive = (archive.member_id.notna() & ~archive.member_id.duplicated(keep=False)
                      & ~archive.duplicated(KEYS, keep=False))
    keys = archive.loc[unique_archive, KEYS].dropna()
    local = local.loc[~local.duplicated(KEYS, keep=False)].dropna(subset=KEYS)
    linked = local.rename_axis("source_row").reset_index().merge(keys, on=KEYS, validate="one_to_one")
    return pd.Index(linked.source_row)

def read_archive():
    parts = []
    digests = {}
    for name in FILES:
        path = Path("data", name)
        with path.open("rb") as stream:
            digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        with ZipFile(path) as archive:
            with archive.open(name.removesuffix(".zip")) as csv:
                parts.append(pd.read_csv(csv, skiprows=1, usecols=KEYS+["member_id"], dtype=str))
    return normalize(pd.concat(parts, ignore_index=True)), digests

if __name__ == "__main__":
    fetch()
