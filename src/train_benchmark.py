"""
Step 3 - Model benchmarking
===========================
Trains and compares every prescribed technique on the SAME preprocessed data:

    Logistic Regression (baseline), PCA + Logistic Regression,
    Decision Tree, Random Forest,
    AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost

Protocol
  * train (70 %) -> fit;  validation (15 %) -> hyper-parameter choice,
    early stopping and decision threshold;  test (15 %) -> final report only.
  * No class re-weighting: probabilities stay interpretable (needed for the
    cost-sensitive analysis); the decision threshold is tuned on validation
    (maximum F1) instead of using the naive 0.5.
  * Metrics: ROC-AUC, Gini, PR-AUC, KS, Brier, log-loss + threshold metrics.

Also runs the PCA analysis (explained variance, class separation in PC space,
component loadings).

Outputs: models/*.joblib, reports/tables/benchmark_*.csv,
         reports/tables/test_predictions.parquet, reports/figures/benchmark/*
Usage:
    python -m src.train_benchmark
"""
import json
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.metrics import (ConfusionMatrixDisplay, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, SEED, TABLE_DIR
from src.features import build_preprocessor, load_splits
from src.models import (MODEL_FAMILY, MODEL_ORDER, PARAM_GRIDS, LoanModel,
                        best_f1_threshold, best_iteration, evaluate,
                        fit_estimator, make_estimator, proba, timed)
from src.plotting import (CLASS_COLORS, CLASS_LABELS, INK_2, MODEL_COLORS,
                          MODEL_STYLE, MUTED, SEQ_BLUE, save)

SUB = "benchmark"


def slug(name: str) -> str:
    return name.lower().replace(" + ", "_").replace(" ", "_")


def train_one(name, Xtr, ytr, Xva, yva):
    """Grid-search (if a grid exists) on validation AUC, else fit once with
    early stopping.  Returns (estimator, info, tuning_rows)."""
    rows = []
    grid = PARAM_GRIDS.get(name, [None])
    best = None
    for params in grid:
        est = make_estimator(name, params)
        est, t = timed(fit_estimator, name, est, Xtr, ytr, Xva, yva)
        auc = roc_auc_score(yva, proba(est, Xva))
        rows.append({"model": name, "params": json.dumps(params), "val_AUC": auc,
                     "fit_seconds": t})
        print(f"    {name:<26} {str(params or ''):<40} val AUC={auc:.4f}  ({t:.1f}s)")
        if best is None or auc > best[2]:
            best = (est, params, auc, t)
    est, params, auc, t = best
    info = {"best_params": params, "val_AUC": auc, "train_seconds": t,
            "n_iterations": best_iteration(name, est),
            "tuning_seconds": float(sum(r["fit_seconds"] for r in rows))}
    if name == "PCA + Logistic Regression":
        info["n_components"] = int(est.named_steps["pca"].n_components_)
    return est, info, rows


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_curves(y, preds: dict):
    # ROC
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], ls=MODEL_STYLE.get(name, "-"),
                label=f"{name}  (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    ax.set(xlabel="False positive rate (good borrowers flagged)",
           ylabel="True positive rate (defaulters caught)",
           title="ROC curves - test set")
    ax.legend(loc="lower right")
    save(fig, "roc_curves", SUB)

    # Precision-recall
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for name, p in preds.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=MODEL_COLORS[name], ls=MODEL_STYLE.get(name, "-"), label=name)
    ax.axhline(y.mean(), color=MUTED, lw=0.8, ls=":", label=f"Base rate ({y.mean():.1%})")
    ax.set(xlabel="Recall (share of defaulters caught)", ylabel="Precision",
           title="Precision-recall curves - test set")
    ax.legend(loc="upper right")
    save(fig, "pr_curves", SUB)

    # Calibration / reliability
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":", label="Perfectly calibrated")
    for name, p in preds.items():
        frac, mean_p = calibration_curve(y, p, n_bins=15, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", ms=3, color=MODEL_COLORS[name],
                ls=MODEL_STYLE.get(name, "-"), label=name)
    ax.set(xlabel="Mean predicted probability of default",
           ylabel="Observed default rate", title="Reliability diagram - test set",
           xlim=(0, 0.6), ylim=(0, 0.6))
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "calibration_curves", SUB)


def plot_confusions(y, preds, thresholds):
    names = list(preds)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for ax, name in zip(axes.ravel(), names):
        yhat = (preds[name] >= thresholds[name]).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y, yhat, display_labels=["Paid", "Default"], cmap=SEQ_BLUE,
            colorbar=False, ax=ax, values_format=",")
        ax.set_title(f"{name}\n(threshold {thresholds[name]:.2f})", fontsize=10)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the test set (F1-optimal threshold from validation)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    save(fig, "confusion_matrices", SUB)


def plot_metric_bars(res: pd.DataFrame):
    metrics = ["ROC_AUC", "PR_AUC", "KS", "F1", "Recall", "Precision"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    order = res.sort_values("ROC_AUC")["model"].tolist()
    for ax, m in zip(axes.ravel(), metrics):
        vals = res.set_index("model").loc[order, m]
        ax.barh(order, vals, color=[MODEL_COLORS[n] for n in order], height=0.65)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8, color=INK_2)
        ax.set_title(m.replace("_", "-"))
        ax.set_xlim(0, vals.max() * 1.18)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Test-set performance by model", fontsize=13)
    fig.tight_layout()
    save(fig, "metric_comparison", SUB)


# --------------------------------------------------------------------------
# PCA analysis
# --------------------------------------------------------------------------
def pca_analysis(Xtr, ytr, feature_names):
    Z = StandardScaler().fit_transform(Xtr)
    pca = PCA(random_state=SEED).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    needed = {f"{int(q*100)}%": int(np.searchsorted(cum, q) + 1) for q in (0.8, 0.9, 0.95, 0.99)}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    k = np.arange(1, len(cum) + 1)
    ax[0].bar(k[:40], pca.explained_variance_ratio_[:40], color=MODEL_COLORS["PCA + Logistic Regression"])
    ax[0].set(title="Scree plot (first 40 components)", xlabel="Principal component",
              ylabel="Explained variance ratio")
    ax[1].plot(k, cum, color=MODEL_COLORS["PCA + Logistic Regression"])
    for lab, n in needed.items():
        q = float(lab.strip("%")) / 100
        ax[1].axhline(q, color=MUTED, lw=0.6, ls=":")
        ax[1].annotate(f"{lab}: {n} PCs", (n, q), xytext=(6, -12),
                       textcoords="offset points", fontsize=8, color=INK_2)
    ax[1].set(title="Cumulative explained variance", xlabel="Number of components",
              ylabel="Cumulative variance")
    save(fig, "pca_explained_variance", SUB)

    # Class separation in the first two PCs
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Z), 6000, replace=False)
    P = pca.transform(Z[idx])[:, :2]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for c in (0, 1):
        m = ytr[idx] == c
        ax.scatter(P[m, 0], P[m, 1], s=8, alpha=0.35, color=CLASS_COLORS[c],
                   label=CLASS_LABELS[c], edgecolors="none")
    ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
           title="Borrowers in the first two principal components")
    ax.legend(markerscale=2)
    save(fig, "pca_scatter_pc1_pc2", SUB)

    # Loadings of the first 3 PCs
    load = pd.DataFrame(pca.components_[:3].T, index=feature_names,
                        columns=["PC1", "PC2", "PC3"])
    top = load.abs().max(axis=1).sort_values(ascending=False).index[:20]
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(load.loc[top].values, cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
    ax.set_yticks(range(len(top)), top, fontsize=8)
    ax.set_xticks(range(3), load.columns)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Loading")
    ax.set_title("Top-20 feature loadings on PC1-PC3")
    save(fig, "pca_loadings", SUB)

    pd.DataFrame({"component": k, "explained_variance_ratio": pca.explained_variance_ratio_,
                  "cumulative": cum}).to_csv(TABLE_DIR / "pca_explained_variance.csv", index=False)
    return needed


# --------------------------------------------------------------------------
def main():
    d = load_splits(refit_columns=True)
    spec = d["spec"]
    raw_cols = spec["numeric"] + spec["categorical"]
    pre = build_preprocessor(spec["numeric"], spec["categorical"])
    Xtr = pre.fit_transform(d["X_train"]).astype(np.float32)
    Xva = pre.transform(d["X_val"]).astype(np.float32)
    Xte = pre.transform(d["X_test"]).astype(np.float32)
    ytr, yva, yte = d["y_train"], d["y_val"], d["y_test"]
    feat_names = list(pre.get_feature_names_out())
    print(f"[bench] train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")
    joblib.dump(pre, MODELS_DIR / "preprocessor.joblib")

    results, tuning, infos, test_preds, val_preds, thresholds = [], [], {}, {}, {}, {}
    for name in MODEL_ORDER:
        print(f"[bench] training {name}")
        est, info, rows = train_one(name, Xtr, ytr, Xva, yva)
        tuning += rows

        p_val = proba(est, Xva)
        thr = best_f1_threshold(yva, p_val)
        p_test, t_inf = timed(proba, est, Xte)

        metrics = evaluate(yte, p_test, thr)
        size_mb = len(pickle.dumps(est)) / 1e6
        info.update({"threshold": thr, "test_inference_seconds": t_inf,
                     "model_size_mb": size_mb, "family": MODEL_FAMILY[name]})
        infos[name] = info
        results.append({"model": name, "family": MODEL_FAMILY[name],
                        **metrics, "val_AUC": info["val_AUC"],
                        "train_seconds": info["train_seconds"],
                        "inference_ms_per_1k": 1000 * t_inf / len(yte) * 1000,
                        "model_size_mb": size_mb})
        test_preds[name], val_preds[name], thresholds[name] = p_test, p_val, thr
        print(f"    -> test AUC {metrics['ROC_AUC']:.4f}  F1 {metrics['F1']:.4f}  "
              f"thr {thr:.3f}  ({info['train_seconds']:.1f}s)")

        lm = LoanModel(name, pre, est, threshold=thr, raw_columns=raw_cols, info=info)
        joblib.dump(lm, MODELS_DIR / f"{slug(name)}.joblib", compress=3)

    res = pd.DataFrame(results)
    res.to_csv(TABLE_DIR / "benchmark_results.csv", index=False)
    pd.DataFrame(tuning).to_csv(TABLE_DIR / "tuning_results.csv", index=False)
    (TABLE_DIR / "benchmark_info.json").write_text(json.dumps(infos, indent=2, default=str))

    pt = pd.DataFrame(test_preds, index=d["X_test"].index)
    pt["y_true"] = yte
    pt = pt.join(d["meta_test"])
    pt.to_parquet(TABLE_DIR / "test_predictions.parquet")
    pv = pd.DataFrame(val_preds, index=d["X_val"].index)
    pv["y_true"] = yva
    pv.to_parquet(TABLE_DIR / "val_predictions.parquet")

    print("\n[bench] TEST-SET RESULTS")
    cols = ["model", "ROC_AUC", "PR_AUC", "KS", "Brier", "F1", "Precision",
            "Recall", "train_seconds"]
    print(res[cols].sort_values("ROC_AUC", ascending=False).round(4).to_string(index=False))

    plot_curves(yte, test_preds)
    plot_confusions(yte, test_preds, thresholds)
    plot_metric_bars(res)
    needed = pca_analysis(Xtr, ytr, feat_names)
    (TABLE_DIR / "pca_components_needed.json").write_text(json.dumps(needed, indent=2))
    print("[bench] PCA components needed:", needed)


if __name__ == "__main__":
    main()
