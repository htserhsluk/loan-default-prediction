"""
Analysis A - Feature Analysis & Ablation Studies
=================================================
Run for EVERY implemented model, so the effect of each feature configuration
can be compared across model families.

Experiments
  1. Leave-one-group-out  : drop one feature group, retrain, measure the
                            change in test AUC vs. the full feature set.
  2. Single group only    : train using one feature group at a time.
  3. Feature selection    : keep only the top-k raw features (k = 5, 10, 20,
                            40) ranked by LightGBM gain importance computed on
                            the training set only.
  4. Dimensionality red.  : replace the features with the first k principal
                            components (k = 5, 10, 20, 40).
  5. Engineering value    : remove (a) the engineered ratio features and
                            (b) the informative-missingness flags.

To keep the 9 models x 25 configurations grid tractable, every configuration
is trained on the same stratified 30k-row subsample of the training split and
evaluated on the full 45k test set (relative comparisons are what matter
here; see the sampling learning curve for the effect of training size).

Usage:
    python -m src.analysis_ablation [--train-size 30000] [--jobs 5]
"""
import argparse
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, SEED, TABLE_DIR
from src.features import FEATURE_GROUPS, build_preprocessor, load_splits, onehot_source
from src.models import MODEL_ORDER, fit_estimator, make_estimator, proba
from src.plotting import DIVERGING, INK_2, MODEL_COLORS, MODEL_STYLE, SEQ_BLUE, save

SUB = "analysis_A_ablation"
ENGINEERED = ["loan_to_income", "installment_to_income", "revol_bal_to_income"]
MISSING_FLAGS_PREFIX = ("ever_",)
MISSING_FLAGS = ["emp_length_missing", "emp_title_missing", "bureau_data_missing"]

# fixed, moderate settings for the many re-trainings
ABLATION_PARAMS = {
    "XGBoost": {"n_estimators": 300},
    "LightGBM": {"n_estimators": 400},
    "CatBoost": {"n_estimators": 400},
}


def _limit_threads(name, est, n=2):
    if name == "Random Forest":
        est.set_params(n_jobs=n)
    elif name in ("XGBoost", "LightGBM"):
        est.set_params(n_jobs=n)
    elif name == "CatBoost":
        est.set_params(thread_count=n)
    return est


def _fit_eval(name, Xtr, ytr, Xte, yte, best_params):
    params = dict(best_params.get(name) or {})
    params.update(ABLATION_PARAMS.get(name, {}))
    est = _limit_threads(name, make_estimator(name, params, fast=True))
    t = time.perf_counter()
    fit_estimator(name, est, Xtr, ytr)
    fit_s = time.perf_counter() - t
    p = proba(est, Xte)
    return {"model": name, "ROC_AUC": roc_auc_score(yte, p),
            "PR_AUC": average_precision_score(yte, p), "fit_seconds": fit_s}


def run_config(experiment, config, num, cat, Xtr_df, ytr, Xte_df, yte, best_params,
               jobs, pca_k=None):
    pre = build_preprocessor(num, cat)
    Xtr = pre.fit_transform(Xtr_df[num + cat]).astype(np.float32)
    Xte = pre.transform(Xte_df[num + cat]).astype(np.float32)
    models = list(MODEL_ORDER)
    if pca_k is not None:
        pca = Pipeline([("s", StandardScaler()), ("p", PCA(pca_k, random_state=SEED))])
        Xtr = pca.fit_transform(Xtr).astype(np.float32)
        Xte = pca.transform(Xte).astype(np.float32)
        models.remove("PCA + Logistic Regression")   # identical to LR on PCs here
    out = Parallel(n_jobs=jobs)(delayed(_fit_eval)(m, Xtr, ytr, Xte, yte, best_params)
                                for m in models)
    for r in out:
        r.update({"experiment": experiment, "config": config,
                  "n_raw_features": len(num) + len(cat), "n_model_inputs": Xtr.shape[1]})
    aucs = ", ".join(f"{r['model'].split()[0]}={r['ROC_AUC']:.3f}" for r in out)
    print(f"  [{experiment}] {config:<38} {aucs}")
    return out


def lgbm_raw_ranking(spec):
    """Rank raw features by the gain importance of the benchmark LightGBM
    (trained on the training split only)."""
    lm = joblib.load(MODELS_DIR / "lightgbm.joblib")
    gain = lm.estimator.booster_.feature_importance(importance_type="gain")
    s = pd.Series(gain, index=lm.feature_names)
    s.index = [onehot_source(f, spec["categorical"]) for f in s.index]
    return s.groupby(level=0).sum().sort_values(ascending=False)


# --------------------------------------------------------------------------
def plots(res: pd.DataFrame):
    full = res[res.experiment == "Baseline"].set_index("model")["ROC_AUC"]
    models = [m for m in MODEL_ORDER if m in full.index]

    # 1. leave-one-group-out heat-map (delta AUC)
    lo = res[res.experiment == "Leave-one-group-out"].copy()
    lo["delta"] = lo["ROC_AUC"] - lo["model"].map(full)
    piv = lo.pivot(index="config", columns="model", values="delta")[models]
    piv = piv.loc[piv.mean(axis=1).sort_values().index] * 100
    lim = np.abs(piv.values).max()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(piv.values, cmap=DIVERGING.reversed(), vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(models)), models, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv)), [c.replace("without ", "- ") for c in piv.index], fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(piv.values[i, j]) > lim * 0.6 else INK_2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Change in test AUC x100 vs all features")
    ax.set_title("Leave-one-group-out ablation (AUC points x100; red = performance lost)")
    save(fig, "leave_one_group_out", SUB)

    # 2. single-group heat-map (absolute AUC)
    sg = res[res.experiment == "Single group only"]
    piv = sg.pivot(index="config", columns="model", values="ROC_AUC")[models]
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(piv.values, cmap=SEQ_BLUE, vmin=0.5, vmax=piv.values.max(), aspect="auto")
    ax.set_xticks(range(len(models)), models, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv)), [c.replace("only ", "") for c in piv.index], fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.62 else INK_2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Test AUC")
    ax.set_title("Predictive power of each feature group on its own")
    save(fig, "single_group_only", SUB)

    # 3 + 4. top-k selection and PCA curves
    for exp, fname, xlabel in (("Top-k feature selection", "feature_selection_topk",
                                "Number of raw features kept (top-k by LightGBM gain)"),
                               ("PCA components", "pca_components",
                                "Number of principal components")):
        e = res[res.experiment == exp].copy()
        e["k"] = e["config"].str.extract(r"(\d+)").astype(int)
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in models:
            s = e[e.model == m].sort_values("k")
            if s.empty:
                continue
            ax.plot(s["k"], s["ROC_AUC"], marker="o", ms=4, color=MODEL_COLORS[m],
                    ls=MODEL_STYLE.get(m, "-"), label=m)
            ax.scatter([s["k"].max() * 1.6], [full[m]], marker="*", s=70,
                       color=MODEL_COLORS[m], zorder=3)
        ax.set_xscale("log")
        ticks = sorted(e["k"].unique()) + [int(e["k"].max() * 1.6)]
        ax.set_xticks(ticks, [str(t) for t in ticks[:-1]] + ["all\nfeatures"])
        ax.minorticks_off()
        ax.set(xlabel=xlabel, ylabel="Test ROC-AUC", title=f"{exp}: AUC vs dimensionality")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        save(fig, fname, SUB)

    # 5. engineering value
    en = res[res.experiment == "Feature engineering"].copy()
    en["delta"] = (en["ROC_AUC"] - en["model"].map(full)) * 100
    piv = en.pivot(index="model", columns="config", values="delta").loc[models]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(models))
    for i, (c, col) in enumerate(zip(piv.columns, ["#2a78d6", "#eb6834"])):
        ax.bar(x + (i - 0.5) * 0.38, piv[c], 0.36, color=col, label=c)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, models, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="Change in test AUC (x100)",
           title="Value of feature engineering (negative = removing them hurts)")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save(fig, "feature_engineering_value", SUB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-size", type=int, default=30_000)
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()

    d = load_splits()
    spec = d["spec"]
    num_all, cat_all = spec["numeric"], spec["categorical"]
    Xtr_full, ytr_full = d["X_train"], d["y_train"]
    Xtr, _, ytr, _ = train_test_split(Xtr_full, ytr_full, train_size=args.train_size,
                                      stratify=ytr_full, random_state=SEED)
    Xte, yte = d["X_test"], d["y_test"]
    info = json.loads((TABLE_DIR / "benchmark_info.json").read_text())
    best_params = {m: info[m].get("best_params") for m in info}
    print(f"[ablation] train subsample {len(Xtr):,}  test {len(Xte):,}")

    res = []
    run = lambda exp, cfg, num, cat, pca_k=None: res.extend(  # noqa: E731
        run_config(exp, cfg, num, cat, Xtr, ytr, Xte, yte, best_params, args.jobs, pca_k))

    run("Baseline", "All features", num_all, cat_all)

    for g, cols in FEATURE_GROUPS.items():
        run("Leave-one-group-out", f"without {g}",
            [c for c in num_all if c not in cols], [c for c in cat_all if c not in cols])
    for g, cols in FEATURE_GROUPS.items():
        run("Single group only", f"only {g}",
            [c for c in num_all if c in cols], [c for c in cat_all if c in cols])

    rank = lgbm_raw_ranking(spec)
    rank.rename("lgbm_gain").to_csv(TABLE_DIR / "feature_ranking_lgbm_gain.csv")
    for k in (5, 10, 20, 40):
        top = rank.index[:k].tolist()
        run("Top-k feature selection", f"top {k} features",
            [c for c in num_all if c in top], [c for c in cat_all if c in top])

    for k in (5, 10, 20, 40):
        run("PCA components", f"{k} principal components", num_all, cat_all, pca_k=k)

    flags = [c for c in num_all if c.startswith(MISSING_FLAGS_PREFIX) or c in MISSING_FLAGS]
    run("Feature engineering", "without engineered ratios",
        [c for c in num_all if c not in ENGINEERED], cat_all)
    run("Feature engineering", "without missingness flags",
        [c for c in num_all if c not in flags], cat_all)

    df = pd.DataFrame(res)
    df.to_csv(TABLE_DIR / "ablation_results.csv", index=False)
    plots(df)

    full = df[df.experiment == "Baseline"].set_index("model")["ROC_AUC"]
    lo = df[df.experiment == "Leave-one-group-out"].copy()
    lo["delta"] = lo["ROC_AUC"] - lo["model"].map(full)
    summary = lo.groupby("config")["delta"].mean().sort_values()
    print("\n[ablation] mean AUC change when a group is removed (all models):")
    print((summary * 100).round(3).to_string())


if __name__ == "__main__":
    main()
