"""
Central configuration: paths, random seed, sampling sizes and the
column lists used for leakage removal / feature grouping.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"

RAW_FILE = RAW_DIR / "accepted_2007_to_2018Q4.csv.gz"
POPULATION_FILE = PROCESSED_DIR / "population_resolved.parquet"   # all resolved loans (lean cols)
SAMPLE_FILE = PROCESSED_DIR / "loans_sample.parquet"             # modelling sample (raw, leakage-free)
OUTCOME_FILE = PROCESSED_DIR / "loans_sample_outcomes.parquet"   # post-loan $ outcomes (cost analysis ONLY)
SPLIT_FILE = PROCESSED_DIR / "split_indices.json"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIG_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Experiment settings
# --------------------------------------------------------------------------
SEED = 42
SAMPLE_SIZE = 300_000          # modelling sample (>= 100k required by the brief)
TEST_SIZE = 0.15               # final hold-out test set
VAL_SIZE = 0.15                # validation set (early stopping, threshold, calibration)

# Matured-vintage rule (see data_prep.py): we keep only loans whose scheduled
# maturity date (issue_d + term) is on or before the data snapshot date.
# Recent loans that are already "resolved" are a biased subset (early defaults
# and early pre-payments), so including them would distort the default rate.
# The snapshot date is inferred from the data (latest payment date observed).
SNAPSHOT_DATE = None  # None -> inferred as max(last_pymnt_d)

TARGET = "default"

# --------------------------------------------------------------------------
# Target definition
# --------------------------------------------------------------------------
GOOD_STATUSES = [
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
]
BAD_STATUSES = [
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
]
# Current / In Grace Period / Late (16-30) / Late (31-120) -> outcome unknown -> excluded

# --------------------------------------------------------------------------
# DATA LEAKAGE: columns that are only known AFTER the loan is issued.
# Using any of these would reveal the outcome we are trying to predict.
# --------------------------------------------------------------------------
LEAKAGE_COLS = [
    # repayment history / balances after issuance
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "pymnt_plan",
    # recoveries / collections after charge-off
    "recoveries", "collection_recovery_fee",
    # credit bureau data refreshed after issuance
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    # hardship programme (only offered to struggling borrowers)
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    # debt settlement (only after default)
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage",
    "settlement_term",
    # amount actually funded by investors (decided after listing)
    "funded_amnt", "funded_amnt_inv",
]

# Identifiers, free text and constant columns: no predictive value / not usable
ID_TEXT_COLS = [
    "id", "member_id", "url", "desc", "title", "emp_title", "zip_code",
    "policy_code",
]

# Lending Club publishes no borrower ID. data_prep treats loans that share all of
# these (fixed for a person between applications) as one borrower and keeps only
# that borrower's first loan. The job title is lower-cased and trimmed first.
BORROWER_KEY = ["earliest_cr_line", "zip_code", "emp_title"]

# Post-loan $ outcome columns kept separately ONLY for the cost-sensitive
# financial analysis (never used as model features).
OUTCOME_COLS = ["loan_amnt", "total_pymnt", "recoveries",
                "collection_recovery_fee", "total_rec_prncp", "total_rec_int",
                "int_rate", "term",
                # kept only to DEMONSTRATE leakage in the EDA (never features)
                "last_fico_range_high", "last_pymnt_amnt", "total_rec_late_fee"]

# "Months since <event>" columns: NaN means the event NEVER happened, which is
# informative (a clean credit file) and must not be blindly median-imputed.
MTHS_SINCE_COLS = [
    "mths_since_last_delinq", "mths_since_last_record",
    "mths_since_last_major_derog", "mths_since_recent_bc_dlq",
    "mths_since_recent_inq", "mths_since_recent_revol_delinq",
    "mths_since_recent_bc", "mths_since_rcnt_il",
]

# Columns > this fraction missing are dropped (except MTHS_SINCE_COLS which are
# converted into "ever happened" flags + capped values).
MAX_MISSING_FRAC = 0.50

CATEGORICAL_COLS = [
    "term", "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status", "application_type",
]
ORDINAL_COLS = ["grade", "sub_grade", "emp_length"]
