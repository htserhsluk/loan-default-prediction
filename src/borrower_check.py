"""
Borrower-uniqueness check
=========================
The brief asks for one row per borrower. Lending Club's public release has one
row per loan, and its borrower identifier (`member_id`) is blank in every row,
so a repeat borrower cannot be identified directly. This script measures how
much that could matter:

1. Confirms that `member_id` is empty and that the loan `id` is unique.
2. Estimates possible repeat borrowers by matching loans on attributes that
   stay fixed for a person between applications:
     strict: first-credit month + 3-digit ZIP + job title + stated income
     loose : first-credit month + 3-digit ZIP + job title
   The strict key misses repeat borrowers whose income changed. The loose key
   over-matches common job titles, but it also misses borrowers who moved or
   changed jobs, and loans with no job title cannot be matched at all. So
   neither key bounds the true rate: each gives the share it can detect.
3. Counts test loans whose key also occurs in train/validation (a possible
   same-borrower overlap across splits) and recomputes every model's test AUC
   without them.

Outputs reports/tables/borrower_check.json and borrower_check_auc.csv.
Needs the benchmark predictions, so it runs after src.train_benchmark.

Usage:
    python -m src.borrower_check
"""
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import POPULATION_FILE, RAW_FILE, SPLIT_FILE, TABLE_DIR
from src.models import MODEL_ORDER

KEYS = {
    "strict": ["earliest_cr_line", "zip_code", "emp_title_norm", "annual_inc"],
    "loose": ["earliest_cr_line", "zip_code", "emp_title_norm"],
}


def main():
    raw = pd.read_csv(RAW_FILE, low_memory=False,
                      usecols=["id", "member_id", "emp_title", "zip_code",
                               "earliest_cr_line", "annual_inc"])
    raw = raw[pd.to_numeric(raw["id"], errors="coerce").notna()]
    raw.index = pd.to_numeric(raw["id"]).astype("Int64").astype(str)
    raw["emp_title_norm"] = raw["emp_title"].str.lower().str.strip()

    pop_ids = pd.read_parquet(POPULATION_FILE, columns=["default"]).index
    split = json.loads(SPLIT_FILE.read_text())
    train_val = split["train"] + split["val"]
    pt = pd.read_parquet(TABLE_DIR / "test_predictions.parquet")

    out = {
        "raw_loans": int(len(raw)),
        "member_id_non_missing": int(raw["member_id"].notna().sum()),
        "loan_id_unique": bool(raw.index.is_unique),
        "cohort_loans": int(len(pop_ids)),
        "sample_loans": int(len(train_val) + len(split["test"])),
    }
    auc_rows = []
    cohort = raw.loc[pop_ids]
    for name, cols in KEYS.items():
        complete = cohort[cols].notna().all(axis=1)
        key = pd.Series(list(zip(*[cohort[c] for c in cols])), index=cohort.index).where(complete)
        counts = key.dropna().value_counts()
        extra = int((counts - 1).sum())                 # loans beyond a key's first
        seen = set(key.reindex(train_val).dropna())
        overlap = key.reindex(pt.index).map(lambda k: isinstance(k, tuple) and k in seen)
        out[name] = {
            "key": " + ".join(cols),
            "loans_with_complete_key": int(complete.sum()),
            "possible_repeat_loans": extra,
            "possible_repeat_pct": extra / len(cohort),
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
    for name in KEYS:
        k = out[name]
        print(f"[borrowers] {name:<6} possible repeat loans {k['possible_repeat_loans']:,} "
              f"({k['possible_repeat_pct']:.2%}); test loans sharing a key with train/val: "
              f"{k['test_loans_sharing_key_with_train_val']:,}")
    print(f"[borrowers] largest change in any model's test AUC without them: "
          f"{out['max_abs_auc_change']:.4f}")


if __name__ == "__main__":
    main()
