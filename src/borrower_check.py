"""
Borrower-uniqueness check
=========================
The brief asks for one row per borrower. Lending Club's public release has one
row per loan, and its borrower identifier (`member_id`) is blank in every row.
data_prep therefore identifies a borrower by attributes that stay fixed between
their applications (first-credit month + 3-digit ZIP + job title,
config.BORROWER_KEY) and keeps only each borrower's first loan.

That key cannot catch a repeat borrower who moved or changed jobs, or one with
no job title on file. This script measures how many such repeats may be left:

1. Confirms that `member_id` is empty, that the loan `id` is unique, and that no
   two loans in the modelling population share the borrower key.
2. Looks for possible repeats that are left, with two keys that each drop one
   part of the borrower key and use the stated income instead:
     moved   : first-credit month + job title + stated income   (new ZIP)
     new job : first-credit month + 3-digit ZIP + stated income (new or no title)
   Both also match different people who happen to share these values, so the
   counts are estimates, not exact figures.
3. Counts test loans whose key also occurs in train/validation (a possible
   same-borrower overlap across splits) and recomputes every model's test
   ROC-AUC without them.

Outputs reports/tables/borrower_check.json and borrower_check_auc.csv.
Needs the benchmark predictions, so it runs after src.train_benchmark.

Usage:
    python -m src.borrower_check
"""
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import BORROWER_KEY, POPULATION_FILE, RAW_FILE, SPLIT_FILE, TABLE_DIR
from src.data_prep import borrower_key
from src.models import MODEL_ORDER

KEYS = {
    "moved": ["earliest_cr_line", "emp_title", "annual_inc"],
    "new_job": ["earliest_cr_line", "zip_code", "annual_inc"],
}


def main():
    raw = pd.read_csv(RAW_FILE, low_memory=False,
                      usecols=["id", "member_id", "emp_title", "zip_code",
                               "earliest_cr_line", "annual_inc"])
    raw = raw[pd.to_numeric(raw["id"], errors="coerce").notna()]
    raw.index = pd.to_numeric(raw["id"]).astype("Int64").astype(str)

    summ = json.loads((TABLE_DIR / "data_summary.json").read_text())
    pop = raw.loc[pd.read_parquet(POPULATION_FILE, columns=["default"]).index]
    split = json.loads(SPLIT_FILE.read_text())
    train_val = split["train"] + split["val"]
    pt = pd.read_parquet(TABLE_DIR / "test_predictions.parquet")

    out = {
        "raw_loans": int(len(raw)),
        "member_id_non_missing": int(raw["member_id"].notna().sum()),
        "loan_id_unique": bool(raw.index.is_unique),
        "borrower_key": " + ".join(BORROWER_KEY),
        "matured_resolved_loans": summ["matured_resolved_loans"],
        "repeat_loans_removed": summ["repeat_loans_removed"],
        "repeat_loans_default_rate": summ["repeat_loans_default_rate"],
        "loans_without_borrower_key": summ["loans_without_borrower_key"],
        "population_loans": int(len(pop)),
        "borrower_key_repeats_left": int(borrower_key(pop, BORROWER_KEY).dropna()
                                         .duplicated().sum()),
        "sample_loans": int(len(train_val) + len(split["test"])),
    }
    auc_rows = []
    for name, cols in KEYS.items():
        key = borrower_key(pop, cols)
        counts = key.dropna().value_counts()
        extra = int((counts - 1).sum())                 # loans beyond a key's first
        seen = set(key.reindex(train_val).dropna())
        overlap = key.reindex(pt.index).map(lambda k: isinstance(k, tuple) and k in seen)
        out[name] = {
            "key": " + ".join(cols),
            "loans_with_complete_key": int(key.notna().sum()),
            "possible_repeat_loans": extra,
            "possible_repeat_pct": extra / len(pop),
            "test_loans_sharing_key_with_train_val": int(overlap.sum()),
        }
        clean = pt[~overlap.to_numpy()]
        for m in MODEL_ORDER:
            auc_rows.append({"key": name, "model": m,
                             "test_auc_all": roc_auc_score(pt["y_true"], pt[m]),
                             "test_auc_without_overlap": roc_auc_score(clean["y_true"], clean[m])})

    auc = pd.DataFrame(auc_rows)
    auc["change"] = auc["test_auc_without_overlap"] - auc["test_auc_all"]
    out["max_abs_auc_change"] = float(auc["change"].abs().max())
    auc.to_csv(TABLE_DIR / "borrower_check_auc.csv", index=False)
    (TABLE_DIR / "borrower_check.json").write_text(json.dumps(out, indent=2))

    print(f"[borrowers] member_id filled in {out['member_id_non_missing']:,} of "
          f"{out['raw_loans']:,} loans; loan id unique: {out['loan_id_unique']}")
    print(f"[borrowers] {out['repeat_loans_removed']:,} repeat loans removed in prep; "
          f"loans left sharing the borrower key: {out['borrower_key_repeats_left']}")
    for name in KEYS:
        k = out[name]
        print(f"[borrowers] {name:<7} possible repeats left {k['possible_repeat_loans']:,} "
              f"({k['possible_repeat_pct']:.2%}); test loans sharing a key with train/val: "
              f"{k['test_loans_sharing_key_with_train_val']:,}")
    print(f"[borrowers] largest change in any model's test ROC-AUC without them: "
          f"{out['max_abs_auc_change']:.4f}")


if __name__ == "__main__":
    main()
