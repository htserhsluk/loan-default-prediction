"""
Feature engineering + preprocessing shared by every model, the analyses and
the UI.

engineer(df)            row-wise, stateless transformations (safe to apply to
                        a single applicant typed into the UI)
select_columns(train)   decides which engineered columns are used (based on
                        the TRAINING split only) and stores the list on disk
build_preprocessor()    sklearn ColumnTransformer (median imputation for
                        numerics, one-hot encoding for nominal categoricals)
load_splits()           convenience loader returning X/y for train/val/test

Missing-value strategy (missingness is informative, so it is NOT handled
blindly):
  * mths_since_<event> : NaN means "the event never happened".  We add a
    binary flag ever_<event> and fill NaN with a cap of 240 months (20 yrs),
    i.e. "a very long time ago / never".
  * A block of ~30 credit-bureau fields was not collected for loans issued
    before ~2012 -> one flag `bureau_data_missing` + median imputation.
  * emp_length NaN -> `emp_length_missing` flag (often unemployed / not
    stated) + median imputation.
  * Columns that are > 50 % missing in the training split (fields introduced
    by Lending Club only in late 2015, joint-application fields) are dropped.
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (MAX_MISSING_FRAC, PROCESSED_DIR, SAMPLE_FILE, SPLIT_FILE,
                        TARGET)

FEATURE_FILE = PROCESSED_DIR / "feature_columns.json"

MTHS_SINCE = [
    "mths_since_last_delinq", "mths_since_last_record",
    "mths_since_last_major_derog", "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq", "mths_since_recent_inq",
    "mths_since_recent_bc",
]
MTHS_CAP = 240

EMP_MAP = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
           "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
           "10+ years": 10}
GRADES = list("ABCDEFG")
SUB_GRADES = [g + str(i) for g in GRADES for i in range(1, 6)]

NOMINAL = ["home_ownership", "verification_status", "purpose", "addr_state",
           "initial_list_status", "application_type", "disbursement_method"]

ALWAYS_DROP = ["issue_d", "earliest_cr_line", "fico_range_low", "fico_range_high",
               "grade", "sub_grade", "term", "emp_length", "sec_app_earliest_cr_line",
               TARGET]


# --------------------------------------------------------------------------
# Feature groups (used by the ablation study and group-level importance)
# --------------------------------------------------------------------------
FEATURE_GROUPS = {
    "LC risk grade": ["grade_num", "sub_grade_num", "int_rate"],
    "Loan characteristics": ["loan_amnt", "term_months", "installment", "purpose",
                             "initial_list_status", "application_type",
                             "disbursement_method", "loan_to_income",
                             "installment_to_income"],
    "Borrower profile": ["annual_inc", "emp_length_num", "emp_length_missing",
                         "emp_title_missing", "home_ownership",
                         "verification_status", "addr_state", "dti"],
    "Credit score & history": ["fico_score", "credit_history_months",
                               "inq_last_6mths", "mths_since_recent_inq",
                               "ever_recent_inq", "mo_sin_old_il_acct",
                               "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
                               "mo_sin_rcnt_tl", "mths_since_recent_bc",
                               "ever_recent_bc", "num_tl_op_past_12m",
                               "acc_open_past_24mths"],
    "Delinquency & public records": [
        "delinq_2yrs", "pub_rec", "pub_rec_bankruptcies", "tax_liens",
        "collections_12_mths_ex_med", "acc_now_delinq", "chargeoff_within_12_mths",
        "delinq_amnt", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
        "num_accts_ever_120_pd", "pct_tl_nvr_dlq", "tot_coll_amt",
        "mths_since_last_delinq", "ever_last_delinq",
        "mths_since_last_record", "ever_last_record",
        "mths_since_last_major_derog", "ever_last_major_derog",
        "mths_since_recent_bc_dlq", "ever_recent_bc_dlq",
        "mths_since_recent_revol_delinq", "ever_recent_revol_delinq"],
    "Balances & utilisation": [
        "revol_bal", "revol_util", "bc_util", "tot_cur_bal", "total_rev_hi_lim",
        "avg_cur_bal", "bc_open_to_buy", "tot_hi_cred_lim", "total_bal_ex_mort",
        "total_bc_limit", "total_il_high_credit_limit", "percent_bc_gt_75",
        "revol_bal_to_income", "bureau_data_missing"],
    "Account counts": [
        "open_acc", "total_acc", "mort_acc", "num_actv_bc_tl", "num_actv_rev_tl",
        "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
        "num_rev_tl_bal_gt_0", "num_sats"],
}


def group_of(col: str) -> str:
    for g, cols in FEATURE_GROUPS.items():
        if col in cols:
            return g
    return "Other"


# --------------------------------------------------------------------------
# Row-wise feature engineering
# --------------------------------------------------------------------------
def _to_float(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.replace("%", "", regex=False)
                         .str.strip(), errors="coerce")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering. Works for the full dataset and for a
    single applicant row coming from the UI."""
    d = df.copy()

    # --- simple type fixes ---------------------------------------------------
    if "term" in d:
        d["term_months"] = pd.to_numeric(d["term"].astype("string")
                                         .str.extract(r"(\d+)")[0], errors="coerce")
    for c in ("int_rate", "revol_util"):
        if c in d:
            d[c] = _to_float(d[c])

    if "emp_length" in d:
        emp = d["emp_length"].astype("string").str.strip()
        d["emp_length_num"] = emp.map(EMP_MAP).astype(float)
        d["emp_length_missing"] = d["emp_length_num"].isna().astype(int)

    # monthly instalment can be derived from amount, rate and term (annuity formula)
    if "installment" not in d and {"loan_amnt", "int_rate", "term_months"} <= set(d.columns):
        r = d["int_rate"] / 1200
        n = d["term_months"]
        d["installment"] = np.where(r > 0, d["loan_amnt"] * r / (1 - (1 + r) ** (-n)),
                                    d["loan_amnt"] / n)

    if "grade" in d:
        d["grade_num"] = d["grade"].map({g: i + 1 for i, g in enumerate(GRADES)}).astype(float)
    if "sub_grade" in d:
        d["sub_grade_num"] = d["sub_grade"].map(
            {g: i + 1 for i, g in enumerate(SUB_GRADES)}).astype(float)

    # --- credit score & history ------------------------------------------------
    if {"fico_range_low", "fico_range_high"} <= set(d.columns):
        d["fico_score"] = (d["fico_range_low"] + d["fico_range_high"]) / 2
    elif "fico_range_low" in d:                  # LC ranges are 4 points wide
        d["fico_score"] = d["fico_range_low"] + 2
    elif "fico_range_high" in d:
        d["fico_score"] = d["fico_range_high"] - 2
    if {"earliest_cr_line", "issue_d"} <= set(d.columns):
        ecl = pd.to_datetime(d["earliest_cr_line"], format="%b-%Y", errors="coerce")
        iss = pd.to_datetime(d["issue_d"])
        d["credit_history_months"] = ((iss.dt.year - ecl.dt.year) * 12
                                      + (iss.dt.month - ecl.dt.month)).astype(float)

    # --- affordability ratios ---------------------------------------------------
    # (clipped: a handful of applicants state an income of only a few dollars)
    # Each ratio is only built when its inputs exist, so a partial applicant
    # file does not crash; absent features are filled later by the caller.
    if "annual_inc" in d:
        inc = d["annual_inc"].replace(0, np.nan)
        if "loan_amnt" in d:
            d["loan_to_income"] = (d["loan_amnt"] / inc).clip(upper=5)
        if "installment" in d:
            d["installment_to_income"] = (d["installment"] * 12 / inc).clip(upper=2)
        if "revol_bal" in d:
            d["revol_bal_to_income"] = (d["revol_bal"] / inc).clip(upper=10)

    # --- informative missingness ----------------------------------------------
    for c in MTHS_SINCE:
        if c in d:
            flag = "ever_" + c.replace("mths_since_", "")
            d[flag] = d[c].notna().astype(int)
            d[c] = d[c].clip(upper=MTHS_CAP).fillna(MTHS_CAP)
    if "tot_cur_bal" in d:
        d["bureau_data_missing"] = d["tot_cur_bal"].isna().astype(int)

    d = d.drop(columns=[c for c in ALWAYS_DROP if c in d.columns])
    return d


# --------------------------------------------------------------------------
# Applicant files uploaded in the UI (batch scoring)
# --------------------------------------------------------------------------
REQUIRED_APPLICANT_COLS = ["loan_amnt", "annual_inc"]


def prepare_applicants(raw: pd.DataFrame, spec: dict, template: dict):
    """Turn an uploaded file in Lending Club format into model inputs.

    Returns (X, filled, missing_required):
      missing_required  required columns absent from the file (X is None then)
      filled            model inputs absent from the file, set to the typical
                        (median / most frequent) training value
    The instalment is derived from amount, rate and term when it is absent.
    """
    missing = [c for c in REQUIRED_APPLICANT_COLS if c not in raw.columns]
    if missing:
        return None, [], missing
    d = raw.copy()
    if "issue_d" in d and not pd.api.types.is_datetime64_any_dtype(d["issue_d"]):
        d["issue_d"] = pd.to_datetime(d["issue_d"], format="%b-%Y", errors="coerce")
    eng = engineer(d)
    cols = spec["numeric"] + spec["categorical"]
    filled = [c for c in cols if c not in eng.columns]
    for c in filled:
        eng[c] = template[c]
    return eng[cols], filled, []


# --------------------------------------------------------------------------
# Column selection (fitted on the training split only)
# --------------------------------------------------------------------------
def select_columns(train_eng: pd.DataFrame, save: bool = True) -> dict:
    miss = train_eng.isna().mean()
    nunique = train_eng.nunique(dropna=True)
    dropped_missing = miss[miss > MAX_MISSING_FRAC].index.tolist()
    dropped_constant = nunique[nunique <= 1].index.tolist()
    keep = [c for c in train_eng.columns
            if c not in dropped_missing and c not in dropped_constant]
    cat = [c for c in keep if c in NOMINAL]
    num = [c for c in keep if c not in cat]
    spec = {"numeric": num, "categorical": cat,
            "dropped_missing": dropped_missing, "dropped_constant": dropped_constant}
    if save:
        FEATURE_FILE.write_text(json.dumps(spec, indent=2))
    return spec


def load_feature_spec() -> dict:
    return json.loads(FEATURE_FILE.read_text())


def build_preprocessor(num_cols, cat_cols) -> ColumnTransformer:
    num_pipe = SimpleImputer(strategy="median")
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=0.005, sparse_output=False)),
    ])
    transformers = [("num", num_pipe, list(num_cols))]
    if cat_cols:
        transformers.append(("cat", cat_pipe, list(cat_cols)))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def onehot_source(feature_name: str, cat_cols) -> str:
    """Map an encoded column (e.g. 'purpose_small_business') back to its raw
    column ('purpose')."""
    for c in sorted(cat_cols, key=len, reverse=True):
        if feature_name.startswith(c + "_"):
            return c
    return feature_name


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_sample_engineered() -> pd.DataFrame:
    raw = pd.read_parquet(SAMPLE_FILE)
    eng = engineer(raw)
    eng[TARGET] = raw[TARGET].values
    eng["issue_d"] = raw["issue_d"].values      # kept for analysis, never a feature
    eng["grade"] = raw["grade"].values           # kept for analysis / policies
    return eng


def load_splits(refit_columns: bool = False):
    """Return dict with X_train, X_val, X_test (engineered, column-selected),
    y_*, the feature spec and the full engineered frame (with meta cols)."""
    eng = load_sample_engineered()
    split = json.loads(SPLIT_FILE.read_text())
    tr, va, te = (eng.loc[split[k]] for k in ("train", "val", "test"))

    if refit_columns or not FEATURE_FILE.exists():
        spec = select_columns(tr.drop(columns=[TARGET, "issue_d", "grade"]))
    else:
        spec = load_feature_spec()
    cols = spec["numeric"] + spec["categorical"]
    return {
        "X_train": tr[cols], "y_train": tr[TARGET].to_numpy(),
        "X_val": va[cols], "y_val": va[TARGET].to_numpy(),
        "X_test": te[cols], "y_test": te[TARGET].to_numpy(),
        "spec": spec, "frame": eng,
        "meta_test": te[["issue_d", "grade"]],
    }
