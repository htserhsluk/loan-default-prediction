"""
Step 2 - Exploratory Data Analysis
==================================
Produces the figures / tables used in the report and the UI:

  * vintage analysis (why only matured loans are used)
  * class balance
  * missing-value profile + evidence that missingness is informative
  * default rate by key categorical variables
  * distributions of key numeric variables by outcome
  * correlation structure + correlation with the target
  * leakage demonstration: post-loan columns predict the outcome almost
    perfectly, which is exactly why they were removed

Usage:
    python -m src.eda
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import OUTCOME_FILE, SAMPLE_FILE, TABLE_DIR, TARGET
from src.features import engineer
from src.plotting import (CLASS_COLORS, CLASS_LABELS, DIVERGING, INK_2, MUTED,
                          SLOTS, save)

SUB = "eda"


def vintage_plot():
    v = pd.read_csv(TABLE_DIR / "vintage_stats.csv", index_col=0)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    yrs = v.index.astype(int)
    used = v["matured_and_resolved"]
    ax[0].bar(yrs, used / 1e3, color=SLOTS[0], label="Used: matured & resolved", width=0.7)
    ax[0].bar(yrs, (v["loans_issued"] - used) / 1e3, bottom=used / 1e3, color="#c3c2b7",
              label="Excluded: not matured / unresolved", width=0.7)
    ax[0].set(title="Loans issued per year", ylabel="Loans (thousands)")
    ax[0].legend(loc="upper left")
    ax[1].plot(yrs, v["default_rate_all_resolved"] * 100, marker="o", color=SLOTS[1],
               label="All resolved loans")
    m = v["matured_and_resolved"] > 0
    ax[1].plot(yrs[m], v.loc[m, "default_rate_matured"] * 100, marker="o", color=SLOTS[0],
               label="Matured loans only (used)")
    ax[1].set(title="Default rate by issue year", ylabel="Default rate (%)", ylim=(11.5, 27.5))
    ax[1].legend(loc="upper center")
    ax[1].annotate("Recent 'resolved' loans are a biased\nsubset (early defaults / pre-payments)",
                   xy=(2018, v.loc[2018, "default_rate_all_resolved"] * 100),
                   xytext=(2013.2, 12.2), fontsize=8, color=INK_2,
                   arrowprops=dict(arrowstyle="->", color=MUTED))
    save(fig, "vintage_analysis", SUB)


def class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    c = df[TARGET].value_counts().sort_index()
    ax.bar([CLASS_LABELS[i] for i in c.index], c.values,
           color=[CLASS_COLORS[i] for i in c.index], width=0.55)
    for i, v in enumerate(c.values):
        ax.text(i, v, f"{v:,}\n({v / c.sum():.1%})", ha="center", va="bottom", fontsize=9,
                color=INK_2)
    ax.set(title="Target class balance (sample of 300k)", ylabel="Borrowers",
           ylim=(0, c.max() * 1.2))
    ax.grid(axis="x", visible=False)
    save(fig, "class_balance", SUB)


def missing_profile(raw):
    miss = raw.drop(columns=[TARGET]).isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    miss.rename("missing_fraction").to_csv(TABLE_DIR / "missing_values.csv")
    top = miss.head(35)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 9))
    colors = [SLOTS[1] if v > 0.5 else SLOTS[0] for v in top.values]
    ax.barh(top.index, top.values * 100, color=colors, height=0.7)
    ax.axvline(50, color=MUTED, ls=":", lw=0.8)
    ax.text(51, 1, "drop threshold\n(50 %)", fontsize=8, color=INK_2)
    ax.set(title="Missing values - 35 most incomplete columns", xlabel="Missing (%)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "missing_values", SUB)

    # Is missingness informative?  default rate when missing vs present
    checks = {
        "mths_since_last_delinq": "Months since last delinquency",
        "mths_since_last_record": "Months since last public record",
        "mths_since_last_major_derog": "Months since last major derogatory",
        "emp_length": "Employment length",
        "tot_cur_bal": "Credit-bureau block (tot_cur_bal)",
        "mths_since_recent_inq": "Months since recent inquiry",
    }
    rows = []
    for c, lab in checks.items():
        isna = raw[c].isna()
        rows.append({"column": c, "label": lab, "missing_frac": isna.mean(),
                     "default_rate_missing": raw.loc[isna, TARGET].mean(),
                     "default_rate_present": raw.loc[~isna, TARGET].mean()})
    rows.append({"column": "emp_title_missing", "label": "Employment title",
                 "missing_frac": raw["emp_title_missing"].mean(),
                 "default_rate_missing": raw.loc[raw.emp_title_missing == 1, TARGET].mean(),
                 "default_rate_present": raw.loc[raw.emp_title_missing == 0, TARGET].mean()})
    t = pd.DataFrame(rows)
    t.to_csv(TABLE_DIR / "missingness_informative.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t["default_rate_present"] * 100, 0.38, color=SLOTS[0], label="Value present")
    ax.bar(x + 0.2, t["default_rate_missing"] * 100, 0.38, color=SLOTS[1], label="Value missing")
    ax.set_xticks(x, t["label"], rotation=25, ha="right", fontsize=8)
    ax.set(ylabel="Default rate (%)", title="Missingness carries information about default risk")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save(fig, "missingness_informative", SUB)


def default_rate_by_category(raw):
    cats = {"grade": "LC grade", "term": "Term", "home_ownership": "Home ownership",
            "verification_status": "Income verification", "emp_length": "Employment length",
            "purpose": "Loan purpose"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    base = raw[TARGET].mean() * 100
    for ax, (c, lab) in zip(axes.ravel(), cats.items()):
        g = raw.groupby(c, observed=True)[TARGET].agg(["mean", "size"])
        g = g[g["size"] >= 300]
        if c == "emp_length":
            order = ["< 1 year"] + [f"{i} year{'s' if i > 1 else ''}" for i in range(1, 10)] + ["10+ years"]
            g = g.reindex([o for o in order if o in g.index])
        elif c != "grade":
            g = g.sort_values("mean")
        ax.bar(g.index.astype(str), g["mean"] * 100, color=SLOTS[0], width=0.7)
        ax.axhline(base, color=SLOTS[1], ls="--", lw=1, label=f"Overall {base:.1f}%")
        ax.set_title(f"Default rate by {lab}")
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.set_ylabel("Default rate (%)")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "default_rate_by_category", SUB)

    # State map as a sorted bar chart
    g = raw.groupby("addr_state")[TARGET].agg(["mean", "size"])
    g = g[g["size"] >= 500].sort_values("mean")
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.bar(g.index, g["mean"] * 100, color=SLOTS[0], width=0.7)
    ax.axhline(base, color=SLOTS[1], ls="--", lw=1)
    ax.set(title="Default rate by state (states with >= 500 loans)", ylabel="Default rate (%)")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", visible=False)
    save(fig, "default_rate_by_state", SUB)


def numeric_distributions(eng):
    cols = {"int_rate": "Interest rate (%)", "fico_score": "FICO score",
            "dti": "Debt-to-income (%)", "annual_inc": "Annual income ($, log scale)",
            "loan_amnt": "Loan amount ($)", "revol_util": "Revolving utilisation (%)",
            "credit_history_months": "Credit history (months)",
            "installment_to_income": "Instalment / income", "inq_last_6mths": "Inquiries (6 mths)"}
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, (c, lab) in zip(axes.ravel(), cols.items()):
        s = eng[[c, TARGET]].dropna()
        lo, hi = s[c].quantile([0.005, 0.995])
        s = s[(s[c] >= lo) & (s[c] <= hi)]
        log = c == "annual_inc"
        bins = np.logspace(np.log10(max(lo, 1)), np.log10(hi), 50) if log else 50
        if c == "inq_last_6mths":
            bins = np.arange(-0.5, 7.5, 1)
        for k in (0, 1):
            ax.hist(s.loc[s[TARGET] == k, c], bins=bins, density=True, alpha=0.55,
                    color=CLASS_COLORS[k], label=CLASS_LABELS[k])
        if log:
            ax.set_xscale("log")
        ax.set_title(lab)
        ax.set_yticks([])
    axes[0, 0].legend()
    fig.suptitle("Distribution of key variables: repaid vs defaulted borrowers", fontsize=13)
    fig.tight_layout()
    save(fig, "numeric_distributions", SUB)


def correlations(eng):
    num = eng.select_dtypes("number").drop(columns=[TARGET])
    corr_t = num.corrwith(eng[TARGET]).dropna().sort_values()
    corr_t.rename("corr_with_default").to_csv(TABLE_DIR / "correlation_with_target.csv")
    top = pd.concat([corr_t.head(10), corr_t.tail(12)])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(top.index, top.values, color=[SLOTS[0] if v < 0 else SLOTS[1] for v in top.values],
            height=0.7)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set(title="Correlation with default (point-biserial)", xlabel="Pearson r")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "correlation_with_target", SUB)

    keep = corr_t.abs().sort_values(ascending=False).index[:25]
    c = num[keep].corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(c.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(keep)), keep, rotation=90, fontsize=7)
    ax.set_yticks(range(len(keep)), keep, fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Correlation matrix - 25 features most related to default")
    save(fig, "correlation_matrix", SUB)


def leakage_demo(raw):
    """Single-feature AUC of post-loan columns vs genuine application features."""
    out = pd.read_parquet(OUTCOME_FILE)
    y = out[TARGET]
    feats = {
        "total_rec_prncp / loan_amnt  [LEAK]": out["total_rec_prncp"] / out["loan_amnt"],
        "recoveries  [LEAK]": out["recoveries"],
        "last_fico_range_high  [LEAK]": -out["last_fico_range_high"],
        "last_pymnt_amnt  [LEAK]": -out["last_pymnt_amnt"],
        "total_pymnt / loan_amnt  [LEAK]": -(out["total_pymnt"] / out["loan_amnt"]),
        "int_rate  (application)": raw.loc[out.index, "int_rate"],
        "sub_grade  (application)": raw.loc[out.index, "sub_grade"].rank(method="dense"),
        "fico_range_low  (application)": -raw.loc[out.index, "fico_range_low"],
        "dti  (application)": raw.loc[out.index, "dti"],
    }
    rows = []
    for k, v in feats.items():
        m = v.notna()
        auc = roc_auc_score(y[m], v[m])
        rows.append({"feature": k, "single_feature_AUC": max(auc, 1 - auc),
                     "leakage": "LEAK" in k})
    t = pd.DataFrame(rows).sort_values("single_feature_AUC")
    t.to_csv(TABLE_DIR / "leakage_demo.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(t["feature"], t["single_feature_AUC"],
            color=[SLOTS[1] if l else SLOTS[0] for l in t["leakage"]], height=0.65)
    for i, v in enumerate(t["single_feature_AUC"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.axvline(0.5, color=MUTED, ls=":", lw=0.8)
    ax.set(xlim=(0.45, 1.08), xlabel="ROC-AUC using this single column",
           title="Data leakage: post-loan columns (orange) 'predict' default almost perfectly")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "leakage_demo", SUB)


def main():
    raw = pd.read_parquet(SAMPLE_FILE)
    eng = engineer(raw)
    eng[TARGET] = raw[TARGET].values

    desc = eng.describe().T
    desc.to_csv(TABLE_DIR / "descriptive_stats.csv")

    vintage_plot()
    class_balance(raw)
    missing_profile(raw)
    default_rate_by_category(raw)
    numeric_distributions(eng)
    correlations(eng)
    leakage_demo(raw)

    summary = json.loads((TABLE_DIR / "data_summary.json").read_text())
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))
    print("[eda] figures written to reports/figures/eda")


if __name__ == "__main__":
    main()
