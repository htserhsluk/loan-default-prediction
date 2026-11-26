"""
Step 1 - Data preparation
=========================
1. Stream the raw 2.26M-row Lending Club file in chunks (memory friendly).
2. Define the target:   default = 1  for Charged Off / Default
                        default = 0  for Fully Paid
   Loans that are still Current / Late / In Grace Period have no final
   outcome yet and are excluded.
3. Matured-vintage filter: keep only loans whose scheduled maturity
   (issue_d + term) is on/before the data snapshot (latest payment date in
   the file, Mar-2019).  Recent loans that are *already* resolved are a biased
   subset (early defaults + early pre-payments) - see the vintage table that
   this script writes to reports/tables/vintage_stats.csv.
4. Remove data leakage (post-origination columns, see config.LEAKAGE_COLS).
   The $ outcome columns are stored in a *separate* file that is used only by
   the cost-sensitive financial analysis, never as model features.
5. Draw a stratified random sample of 300,000 borrowers (stratified on
   issue-year x term x default) and verify that it is representative of the
   population (reports/tables/sample_representativeness.csv).
6. Create a stratified train / validation / test split (70 / 15 / 15).

Usage:
    python -m src.data_prep
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from src.config import (BAD_STATUSES, GOOD_STATUSES, ID_TEXT_COLS, LEAKAGE_COLS,
                        OUTCOME_COLS, OUTCOME_FILE, POPULATION_FILE, RAW_FILE,
                        SAMPLE_FILE, SAMPLE_SIZE, SEED, SNAPSHOT_DATE,
                        SPLIT_FILE, TABLE_DIR, TARGET, TEST_SIZE, VAL_SIZE)

CHUNK = 250_000


def _term_months(s: pd.Series) -> pd.Series:
    return s.astype("string").str.extract(r"(\d+)")[0].astype(float)


def load_resolved_matured() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Stream the raw csv, return (matured+resolved loans, vintage stats)."""
    resolved = GOOD_STATUSES + BAD_STATUSES
    keep, stats = [], []
    n_raw = 0
    snapshot = pd.Timestamp(SNAPSHOT_DATE) if SNAPSHOT_DATE else None
    max_last_pymnt = pd.Timestamp("1900-01-01")

    for i, chunk in enumerate(pd.read_csv(RAW_FILE, chunksize=CHUNK, low_memory=False)):
        n_raw += len(chunk)
        chunk = chunk[chunk["loan_status"].notna()].copy()
        chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        last = pd.to_datetime(chunk["last_pymnt_d"], format="%b-%Y", errors="coerce")
        max_last_pymnt = max(max_last_pymnt, last.max())
        chunk["_term"] = _term_months(chunk["term"])
        chunk["_resolved"] = chunk["loan_status"].isin(resolved)
        chunk["_bad"] = chunk["loan_status"].isin(BAD_STATUSES)
        stats.append(chunk[["issue_d", "_term", "_resolved", "_bad"]])
        keep.append(chunk[chunk["_resolved"]])
        print(f"  chunk {i}: {len(chunk):,} rows read")

    snapshot = snapshot or max_last_pymnt
    print(f"[prep] data snapshot date (latest payment observed): {snapshot:%b-%Y}")

    df = pd.concat(keep, ignore_index=True)
    st = pd.concat(stats, ignore_index=True)

    def matured(frame):
        mat = frame["issue_d"].dt.to_period("M") + frame["_term"].fillna(36).astype(int)
        return mat <= snapshot.to_period("M")

    df = df[matured(df)].copy()
    st["_matured"] = matured(st)

    # ---- vintage table (justifies the matured-vintage rule) -------------
    st["year"] = st["issue_d"].dt.year
    vint = st.groupby("year").agg(
        loans_issued=("_resolved", "size"),
        resolved=("_resolved", "sum"),
        matured=("_matured", "sum"),
    )
    res = st[st["_resolved"]]
    vint["default_rate_all_resolved"] = res.groupby("year")["_bad"].mean()
    mres = st[st["_resolved"] & st["_matured"]]
    vint["matured_and_resolved"] = mres.groupby("year").size()
    vint["default_rate_matured"] = mres.groupby("year")["_bad"].mean()
    vint = vint.fillna(0)
    vint.to_csv(TABLE_DIR / "vintage_stats.csv")
    print(vint)

    df[TARGET] = df["_bad"].astype(int)
    df = df.drop(columns=["_term", "_resolved", "_bad"])
    print(f"[prep] raw rows read: {n_raw:,}")
    return df, vint, n_raw


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Chunked reading can leave numeric columns as strings - fix them."""
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and c not in ("issue_d",):
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().sum() >= 0.98 * df[c].notna().sum() and df[c].notna().any():
                df[c] = conv
    return df


def stratified_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Proportional stratified sample on issue-year x term x target."""
    strata = (df["issue_d"].dt.year.astype(str) + "_" + df["term"].astype(str)
              + "_" + df[TARGET].astype(str))
    frac = n / len(df)
    sample = df.groupby(strata).sample(frac=frac, random_state=SEED)
    # rounding can leave us a handful of rows short/over -> trim / top-up
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=SEED)
    elif len(sample) < n:
        rest = df.drop(sample.index)
        sample = pd.concat([sample, rest.sample(n=n - len(sample), random_state=SEED)])
    return sample


def representativeness(pop: pd.DataFrame, smp: pd.DataFrame) -> pd.DataFrame:
    """Compare population vs sample on key variables (KS test / TVD)."""
    rows = []
    for c in ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_range_low",
              "revol_util", "open_acc", "total_acc", "installment"]:
        a, b = pop[c].dropna(), smp[c].dropna()
        ks = ks_2samp(a.sample(min(len(a), 200_000), random_state=SEED), b)
        rows.append({"variable": c, "type": "numeric",
                     "population_mean": a.mean(), "sample_mean": b.mean(),
                     "KS_statistic": ks.statistic, "KS_pvalue": ks.pvalue,
                     "TVD": np.nan})
    for c in ["grade", "term", "home_ownership", "purpose", "verification_status", TARGET]:
        p = pop[c].value_counts(normalize=True)
        s = smp[c].value_counts(normalize=True)
        tvd = 0.5 * (p.subtract(s, fill_value=0).abs().sum())
        rows.append({"variable": c, "type": "categorical",
                     "population_mean": pop[c].mean() if c == TARGET else np.nan,
                     "sample_mean": smp[c].mean() if c == TARGET else np.nan,
                     "KS_statistic": np.nan, "KS_pvalue": np.nan, "TVD": tvd})
    return pd.DataFrame(rows)


def main() -> None:
    print("[prep] reading raw data in chunks ...")
    df, _, n_raw = load_resolved_matured()
    df = coerce_numeric(df)
    print(f"[prep] matured & resolved loans: {len(df):,}  "
          f"(default rate {df[TARGET].mean():.3%})")

    # ------------------------------------------------------------------
    # Separate the post-loan $ outcomes (cost analysis only) BEFORE removing
    # leakage columns.  Kept in a different file keyed by the same index.
    # ------------------------------------------------------------------
    df = df.reset_index(drop=True)
    df["loan_id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64").astype(str)

    # Population file (leakage-free) - used for the sampling learning curve
    drop_cols = [c for c in LEAKAGE_COLS + ID_TEXT_COLS if c in df.columns]
    # keep a binary "employment title provided" flag before dropping free text
    df["emp_title_missing"] = df["emp_title"].isna().astype(int)
    outcomes = df[["loan_id"] + [c for c in OUTCOME_COLS if c in df.columns]
                  + ["grade", "issue_d", TARGET]].copy()
    clean = df.drop(columns=drop_cols + ["loan_status", "last_pymnt_d"], errors="ignore")
    clean = clean.set_index("loan_id")
    outcomes = outcomes.set_index("loan_id")

    print(f"[prep] removed {len(drop_cols)} leakage / identifier / free-text columns")
    clean.to_parquet(POPULATION_FILE)
    print(f"[prep] population saved -> {POPULATION_FILE}  shape={clean.shape}")

    # ------------------------------------------------------------------
    # Stratified sample
    # ------------------------------------------------------------------
    sample = stratified_sample(clean, SAMPLE_SIZE)
    rep = representativeness(clean, sample)
    rep.to_csv(TABLE_DIR / "sample_representativeness.csv", index=False)
    print("[prep] representativeness check (sample vs population):")
    print(rep.round(4).to_string(index=False))

    sample.to_parquet(SAMPLE_FILE)
    outcomes.loc[sample.index].to_parquet(OUTCOME_FILE)
    print(f"[prep] sample saved -> {SAMPLE_FILE}  shape={sample.shape}")

    # ------------------------------------------------------------------
    # Train / validation / test split (stratified on the target)
    # ------------------------------------------------------------------
    ids = sample.index.to_numpy()
    y = sample[TARGET].to_numpy()
    trval, test = train_test_split(ids, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    y_trval = sample.loc[trval, TARGET].to_numpy()
    train, val = train_test_split(trval, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                  stratify=y_trval, random_state=SEED)
    SPLIT_FILE.write_text(json.dumps({"train": train.tolist(), "val": val.tolist(),
                                      "test": test.tolist()}))
    print(f"[prep] split sizes  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    summary = {
        "raw_rows": int(n_raw),
        "raw_columns": 151,
        "matured_resolved_loans": int(len(clean)),
        "population_default_rate": float(clean[TARGET].mean()),
        "sample_size": int(len(sample)),
        "sample_default_rate": float(sample[TARGET].mean()),
        "columns_after_leakage_removal": int(clean.shape[1] - 1),
        "leakage_columns_removed": [c for c in LEAKAGE_COLS if c in df.columns],
        "id_text_columns_removed": [c for c in ID_TEXT_COLS if c in df.columns],
        "train": len(train), "val": len(val), "test": len(test),
    }
    (TABLE_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
