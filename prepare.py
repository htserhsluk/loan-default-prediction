"""Prepare a deterministic LendingClub cohort from the original CSV."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from archive import KEYS, normalize, read_archive, verified_positions

CATEGORY = "home_ownership purpose emp_length application_type".split()
FEATURES = CATEGORY + """
loan_amnt annual_inc dti delinq_2yrs inq_last_6mths mths_since_last_delinq
mths_since_last_record open_acc pub_rec revol_bal revol_util total_acc
collections_12_mths_ex_med mths_since_last_major_derog acc_now_delinq
tot_coll_amt tot_cur_bal total_rev_hi_lim acc_open_past_24mths avg_cur_bal
bc_open_to_buy bc_util chargeoff_within_12_mths delinq_amnt mo_sin_old_il_acct
mo_sin_old_rev_tl_op mo_sin_rcnt_rev_tl_op mo_sin_rcnt_tl mort_acc
mths_since_recent_bc mths_since_recent_bc_dlq mths_since_recent_inq
mths_since_recent_revol_delinq num_accts_ever_120_pd num_actv_bc_tl num_actv_rev_tl
num_bc_sats num_bc_tl num_il_tl num_op_rev_tl num_rev_accts num_rev_tl_bal_gt_0
num_sats num_tl_120dpd_2m num_tl_30dpd num_tl_90g_dpd_24m num_tl_op_past_12m
pct_tl_nvr_dlq percent_bc_gt_75 pub_rec_bankruptcies tax_liens tot_hi_cred_lim
total_bal_ex_mort total_bc_limit total_il_high_credit_limit
""".split()
STATUS = {"Fully Paid": 0, "Charged Off": 1, "Default": 1}
META = ["issue_d", "term", "loan_status", "member_id"]

def eligible_rows(frame):
    year = pd.to_datetime(frame.issue_d, format="%b-%Y", errors="coerce").dt.year
    mask = (year.between(2012, 2015) & frame.term.str.strip().eq("36 months")
            & frame.loan_status.isin(STATUS))
    return pd.DataFrame({"year": year[mask], "target": frame.loan_status[mask].map(STATUS)})

def assign_split(year):
    return np.select([year <= 2013, year == 2014], ["train", "validation"], default="test")

def main(source):
    out = Path("data")
    reports = Path("reports")
    out.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    chunks = []
    matching = []
    members = 0
    rows = 0
    for frame in pd.read_csv(source, usecols=sorted(set(META + KEYS)), dtype=str, chunksize=50000):
        rows += len(frame)
        eligible = eligible_rows(frame)
        members += int(frame.loc[eligible.index, "member_id"].notna().sum())
        chunks.append(eligible)
        vintage = frame.issue_d.str[-4:].isin(["2012", "2013", "2014", "2015"])
        matching.append(normalize(frame.loc[vintage, KEYS]))
    population = pd.concat(chunks)
    pre_match = len(population)
    archive, hashes = read_archive()
    positions = verified_positions(archive, pd.concat(matching))
    population = population.loc[population.index.intersection(positions)]
    sample = population.sample(n=100000, random_state=42).sort_index()
    selected = []
    for frame in pd.read_csv(source, usecols=FEATURES, chunksize=50000):
        keep = frame.index.intersection(sample.index)
        if len(keep):
            selected.append(frame.loc[keep])
    data = pd.concat(selected)[FEATURES].join(sample)
    data["split"] = assign_split(data.year)
    data.index.name = "source_row"
    data.to_csv(out / "sample.csv.gz", compression={"method": "gzip", "mtime": 0})
    summary = data.groupby("split").target.agg(["size", "sum", "mean"])
    summary.to_csv(reports / "cohorts.csv")
    pd.DataFrame({"missing_fraction": data[FEATURES].isna().mean(),
                  "train_distinct": data.loc[data.split == "train", FEATURES].nunique()}).to_csv(reports / "features.csv")
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = dict(rows=rows, source_sha256=digest, source_columns=len(pd.read_csv(source,nrows=0).columns),
                 eligible=len(population), eligible_nonmissing_member_id=members, seed=42,
                 sample=len(data), features=len(FEATURES), eligible_event_rate=population.target.mean(),
                 sample_event_rate=data.target.mean(), unit="one archived lender member per matched loan",
                 resolved_before_match=pre_match, archive_hashes=hashes,
                 archive_distinct_members=archive.member_id.nunique(), sample_distinct_members=len(data),
                 borrower_verification="Unique archive member and one-to-one exact origination signature in both releases")
    (reports / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), summary, sep="\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    main(parser.parse_args().source)
