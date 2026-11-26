"""
Sampling justification - learning curve on the FULL matured population
=====================================================================
Question: does working with a 300k sample (210k training rows) lose
predictive power compared with using all ~780k matured loans?

Method: hold out 100k random loans from the population, then train the same
LightGBM model on increasingly large random subsets of the remaining ~680k
loans (10k ... all) and measure the hold-out AUC.  Small sizes are repeated
with 3 random draws to show the variance.

If the curve has flattened by ~200k rows, the sample is sufficient, while the
slower prescribed models (sklearn Gradient Boosting / AdaBoost, SHAP, the
ablation grid) stay computationally feasible on a laptop.

Usage:
    python -m src.sampling_check
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import POPULATION_FILE, SEED, TABLE_DIR, TARGET
from src.features import build_preprocessor, engineer, load_feature_spec
from src.plotting import INK_2, MUTED, SLOTS, save

SIZES = [10_000, 25_000, 50_000, 100_000, 210_000, 400_000, None]


def main():
    pop = pd.read_parquet(POPULATION_FILE)
    y = pop[TARGET].to_numpy()
    eng = engineer(pop)
    spec = load_feature_spec()
    X = eng[spec["numeric"] + spec["categorical"]]
    X_pool, X_hold, y_pool, y_hold = train_test_split(
        X, y, test_size=100_000, stratify=y, random_state=SEED)
    print(f"[sampling] population {len(X):,}  pool {len(X_pool):,}  holdout {len(X_hold):,}")

    pre = build_preprocessor(spec["numeric"], spec["categorical"]).fit(X_pool)
    Zp = pre.transform(X_pool).astype(np.float32)
    Zh = pre.transform(X_hold).astype(np.float32)

    rows = []
    rng = np.random.default_rng(SEED)
    for n in SIZES:
        n_eff = len(Zp) if n is None else n
        reps = 3 if n_eff <= 100_000 else 1
        for r in range(reps):
            idx = rng.choice(len(Zp), n_eff, replace=False) if n_eff < len(Zp) else np.arange(len(Zp))
            m = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                               min_child_samples=100, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.7, verbose=-1, random_state=SEED + r)
            t = time.perf_counter()
            m.fit(Zp[idx], y_pool[idx])
            fit_s = time.perf_counter() - t
            auc = roc_auc_score(y_hold, m.predict_proba(Zh)[:, 1])
            rows.append({"train_size": n_eff, "repeat": r, "holdout_AUC": auc,
                         "fit_seconds": fit_s})
            print(f"  n={n_eff:>7,}  rep={r}  AUC={auc:.4f}  fit={fit_s:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "sampling_learning_curve.csv", index=False)
    g = df.groupby("train_size")["holdout_AUC"].agg(["mean", "std"]).fillna(0)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", color=SLOTS[0],
                capsize=3, lw=1.6)
    ax.axvline(210_000, color=SLOTS[1], ls="--", lw=1)
    ax.text(215_000, g["mean"].min(), "  our training set\n  (210k of 300k sample)",
            color=INK_2, fontsize=8, va="bottom")
    full = g["mean"].iloc[-1]
    ax.axhline(full, color=MUTED, ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set(xlabel="Training rows (log scale)", ylabel="Hold-out ROC-AUC",
           title="Learning curve on the full matured population (LightGBM)")
    save(fig, "sampling_learning_curve", "eda")
    at210 = g.loc[210_000, "mean"]
    print(f"[sampling] AUC @210k = {at210:.4f}   AUC @all ({g.index[-1]:,}) = {full:.4f}   "
          f"gap = {full - at210:.4f}")


if __name__ == "__main__":
    main()
