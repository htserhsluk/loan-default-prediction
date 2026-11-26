"""
Analysis F - Model Interpretability & Explainability
====================================================
Which factors drive the predictions, and do the models agree?

  1. Permutation importance (all 9 models) on RAW features: a column is
     shuffled (one-hot columns of a categorical are shuffled together) and the
     drop in test AUC is measured.  Model-agnostic -> comparable across models.
  2. Group permutation importance: shuffle a whole feature group at once.
  3. Agreement between models: Spearman rank correlation of the importances.
  4. SHAP (TreeExplainer) for the gradient-boosted models and the decision
     tree: global summary (beeswarm), dependence plots, and local
     explanations (waterfall) for individual applicants.
  5. Partial dependence (PDP) for key features across all models + ICE curves
     for the best model -> how changing one input moves the predicted risk.
  6. Glass-box views: decision-tree rules and logistic-regression
     coefficients.

Usage:
    python -m src.analysis_explainability [--n-perm 8000]
"""
import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text, plot_tree

from src.config import MODELS_DIR, REPORTS_DIR, SEED, TABLE_DIR
from src.features import FEATURE_GROUPS, load_splits
from src.models import MODEL_ORDER
from src.plotting import (DIVERGING, INK_2, MODEL_COLORS, MODEL_STYLE, MUTED,
                          SEQ_BLUE, SLOTS, save)
from src.train_benchmark import slug

SUB = "analysis_F_explainability"
PDP_FEATURES = {
    "int_rate": "Interest rate (%)", "fico_score": "FICO score",
    "dti": "Debt-to-income (%)", "annual_inc": "Annual income ($)",
    "term_months": "Term (months)", "loan_to_income": "Loan / income",
    "credit_history_months": "Credit history (months)", "revol_util": "Revolving util. (%)",
}
SHAP_MODELS = ["XGBoost", "LightGBM", "CatBoost", "Decision Tree"]


def load_models():
    return {n: joblib.load(MODELS_DIR / f"{slug(n)}.joblib") for n in MODEL_ORDER}


# --------------------------------------------------------------------------
# 1-3 permutation importance
# --------------------------------------------------------------------------
def permutation_importance_raw(model, X, y, cols, n_repeats=3, rng=None):
    base = roc_auc_score(y, model.predict_proba(X))
    out = {}
    for c in cols:
        cs = c if isinstance(c, list) else [c]
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            for cc in cs:
                Xp[cc] = X[cc].to_numpy()[perm]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)))
        out[c if isinstance(c, str) else "|".join(cs)] = float(np.mean(drops))
    return base, out


def importance_section(models, X, y, spec):
    rng = np.random.default_rng(SEED)
    raw = spec["numeric"] + spec["categorical"]
    imp, grp = {}, {}
    for name, m in models.items():
        _, imp[name] = permutation_importance_raw(m, X, y, raw, rng=rng)
        groups = {g: [c for c in cols if c in raw] for g, cols in FEATURE_GROUPS.items()}
        _, g_imp = permutation_importance_raw(m, X, y, list(groups.values()), rng=rng)
        grp[name] = dict(zip(groups.keys(), g_imp.values()))
        top = sorted(imp[name].items(), key=lambda kv: -kv[1])[:3]
        print(f"  {name:<26} top: " + ", ".join(f"{k} ({v*100:.2f})" for k, v in top))
    imp = pd.DataFrame(imp)
    grp = pd.DataFrame(grp)
    imp.to_csv(TABLE_DIR / "permutation_importance_raw.csv")
    grp.to_csv(TABLE_DIR / "permutation_importance_groups.csv")

    # heat-map of the top-20 features (mean rank across models)
    top = imp.rank(ascending=False).mean(axis=1).sort_values().index[:20]
    norm = imp.loc[top].clip(lower=0) / imp.clip(lower=0).max()
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(norm.values, cmap=SEQ_BLUE, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(imp.columns)), imp.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(top)), top, fontsize=8)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = imp.loc[top].values[i, j] * 100
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if norm.values[i, j] > 0.55 else INK_2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Importance (scaled to model max)")
    ax.set_title("Permutation importance - AUC drop x100 when the feature is shuffled\n"
                 "(top-20 features by average rank across models)")
    save(fig, "permutation_importance_heatmap", SUB)

    # group importance
    fig, ax = plt.subplots(figsize=(12, 5))
    gorder = grp.mean(axis=1).sort_values(ascending=False).index
    x = np.arange(len(gorder))
    w = 0.85 / len(grp.columns)
    for i, m in enumerate(grp.columns):
        ax.bar(x + (i - len(grp.columns) / 2) * w + w / 2, grp.loc[gorder, m] * 100, w,
               color=MODEL_COLORS[m], label=m)
    ax.set_xticks(x, gorder, rotation=15, fontsize=9)
    ax.set(ylabel="AUC drop x100 when the group is shuffled",
           title="Feature-group importance across models")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis="x", visible=False)
    save(fig, "group_importance", SUB)

    # agreement between models
    rc = imp.rank(ascending=False).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(rc.values, cmap=DIVERGING.reversed(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(rc)), rc.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(rc)), rc.index, fontsize=8)
    for i in range(len(rc)):
        for j in range(len(rc)):
            ax.text(j, i, f"{rc.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(rc.values[i, j]) > 0.7 else INK_2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rank correlation")
    ax.set_title("Do the models agree on what matters?")
    save(fig, "importance_agreement", SUB)
    return imp, grp


# --------------------------------------------------------------------------
# 4 SHAP
# --------------------------------------------------------------------------
def shap_section(models, X, y, best_name):
    Xs = X.iloc[:2000]
    mean_abs = {}
    for name in SHAP_MODELS:
        m = models[name]
        Z = m.transform(Xs)
        expl = shap.TreeExplainer(m.estimator)
        sv = expl(Z)
        vals = sv.values
        if vals.ndim == 3:                 # (n, features, classes) -> class 1
            vals = vals[..., 1]
            sv = shap.Explanation(vals, base_values=sv.base_values[:, 1], data=Z,
                                  feature_names=m.feature_names)
        else:
            sv.feature_names = m.feature_names
        mean_abs[name] = pd.Series(np.abs(vals).mean(axis=0), index=m.feature_names)

        if name == best_name:
            plt.figure()
            shap.plots.beeswarm(sv, max_display=20, show=False)
            fig = plt.gcf()
            fig.set_size_inches(9, 8)
            plt.title(f"SHAP summary - {name} (log-odds contribution to default)")
            save(fig, "shap_beeswarm_best", SUB)

            top4 = mean_abs[name].sort_values(ascending=False).index[:4]
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
            for ax, f in zip(axes, top4):
                j = m.feature_names.index(f)
                ax.scatter(Z[:, j], vals[:, j], c=y[:len(Z)], cmap=SEQ_BLUE, s=6,
                                alpha=0.6, edgecolors="none", vmin=-0.5, vmax=1.2)
                ax.axhline(0, color=MUTED, lw=0.8)
                ax.set(xlabel=f, ylabel="SHAP value", title=f)
            fig.suptitle(f"SHAP dependence - {name} (dark points = defaulted)", fontsize=12)
            fig.tight_layout()
            save(fig, "shap_dependence_best", SUB)

            # local explanations: highest-risk TP, lowest-risk TN, a missed default
            p = m.predict_proba(Xs)
            yy = y[:len(Xs)]
            cases = {
                "high_risk_defaulter": int(np.argmax(np.where(yy == 1, p, -1))),
                "low_risk_repayer": int(np.argmin(np.where(yy == 0, p, 2))),
                "missed_default": int(np.argmin(np.where(yy == 1, p, 2))),
            }
            for lab, i in cases.items():
                plt.figure()
                shap.plots.waterfall(sv[i], max_display=12, show=False)
                fig = plt.gcf()
                fig.set_size_inches(9, 6)
                plt.title(f"{lab.replace('_', ' ').title()}  -  predicted PD = {p[i]:.1%}, "
                          f"actual = {'default' if yy[i] else 'repaid'}", fontsize=10)
                save(fig, f"shap_waterfall_{lab}", SUB)

    ma = pd.DataFrame(mean_abs)
    ma.to_csv(TABLE_DIR / "shap_mean_abs.csv")
    top = ma.rank(ascending=False).mean(axis=1).sort_values().index[:15]
    fig, axes = plt.subplots(1, len(SHAP_MODELS), figsize=(18, 5.5), sharey=True)
    for ax, name in zip(axes, SHAP_MODELS):
        v = ma.loc[top[::-1], name]
        ax.barh(v.index, v.values, color=MODEL_COLORS[name], height=0.65)
        ax.set_title(name, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Mean |SHAP value| - top-15 features", fontsize=13)
    fig.tight_layout()
    save(fig, "shap_mean_abs_models", SUB)


# --------------------------------------------------------------------------
# 5 PDP / ICE
# --------------------------------------------------------------------------
def pdp_section(models, X, best_name):
    """PDP on CALIBRATED probabilities (isotonic, fitted on validation by
    src.cost_sensitive) so that models with squashed scores (AdaBoost) are on
    the same scale as the others."""
    cal_path = MODELS_DIR / "calibrators.joblib"
    cals = joblib.load(cal_path) if cal_path.exists() else {}

    def pd_of(name, Xv):
        p = models[name].predict_proba(Xv)
        return cals[name].predict(p) if name in cals else p

    Xs = X.iloc[:1500].copy()
    grids, rows = {}, []
    for f in PDP_FEATURES:
        q = X[f].quantile(np.linspace(0.02, 0.98, 18)).unique()
        grids[f] = q if f != "term_months" else np.array([36.0, 60.0])
    for name, m in models.items():
        for f, grid in grids.items():
            for v in grid:
                Xv = Xs.copy()
                Xv[f] = v
                rows.append({"model": name, "feature": f, "value": float(v),
                             "mean_pd": float(pd_of(name, Xv).mean())})
    pdp = pd.DataFrame(rows)
    pdp.to_csv(TABLE_DIR / "partial_dependence.csv", index=False)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, (f, lab) in zip(axes.ravel(), PDP_FEATURES.items()):
        for name in models:
            s = pdp[(pdp.model == name) & (pdp.feature == f)]
            ax.plot(s["value"], s["mean_pd"] * 100, color=MODEL_COLORS[name],
                    ls=MODEL_STYLE.get(name, "-"), marker="o" if f == "term_months" else None,
                    ms=4, label=name)
        ax.set(title=lab, ylabel="Avg calibrated PD (%)")
    h, lab_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab_, loc="lower center", ncol=9, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Partial dependence: how the predicted default probability changes with one "
                 "input (others held at observed values)", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "partial_dependence_all_models", SUB)

    # ICE for the best model
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, f in zip(axes, ["int_rate", "fico_score", "dti"]):
        grid = grids[f]
        sub = Xs.iloc[:60]
        curves = []
        for v in grid:
            Xv = sub.copy()
            Xv[f] = v
            curves.append(pd_of(best_name, Xv))
        curves = np.array(curves).T * 100
        for c in curves:
            ax.plot(grid, c, color=SLOTS[0], alpha=0.18, lw=0.8)
        ax.plot(grid, curves.mean(axis=0), color=SLOTS[1], lw=2.2, label="Average (PDP)")
        ax.set(title=f"ICE - {PDP_FEATURES[f]}", xlabel=PDP_FEATURES[f],
               ylabel="Predicted PD (%)")
        ax.legend()
    fig.suptitle(f"Individual conditional expectation - {best_name} (60 applicants)", fontsize=12)
    fig.tight_layout()
    save(fig, "ice_best_model", SUB)


# --------------------------------------------------------------------------
# 6 glass-box models
# --------------------------------------------------------------------------
def glassbox_section(models, spec):
    dt = models["Decision Tree"]
    rules = export_text(dt.estimator, feature_names=dt.feature_names, max_depth=4,
                        show_weights=False, decimals=2)
    (REPORTS_DIR / "decision_tree_rules.txt").write_text(rules)
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(dt.estimator, max_depth=3, feature_names=dt.feature_names,
              class_names=["Paid", "Default"], filled=True, impurity=False,
              proportion=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Decision tree - first 3 levels (value = class proportions)")
    save(fig, "decision_tree_top_levels", SUB)

    lr = models["Logistic Regression"]
    coef = pd.Series(lr.estimator.named_steps["clf"].coef_[0], index=lr.feature_names)
    coef.to_csv(TABLE_DIR / "logistic_regression_coefficients.csv")
    top = coef.reindex(coef.abs().sort_values(ascending=False).index[:25])[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top.index, top.values, color=[SLOTS[1] if v > 0 else SLOTS[0] for v in top.values],
            height=0.7)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set(xlabel="Standardised coefficient (log-odds per 1 SD)",
           title="Logistic regression - 25 largest coefficients\n(orange raises default risk)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "logistic_coefficients", SUB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=8000)
    args = ap.parse_args()

    d = load_splits()
    spec = d["spec"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(d["X_test"]), args.n_perm, replace=False)
    X = d["X_test"].iloc[idx]
    y = d["y_test"][idx]
    models = load_models()
    info = json.loads((TABLE_DIR / "benchmark_info.json").read_text())
    best = max(info, key=lambda m: info[m]["val_AUC"])
    best_shap = max(SHAP_MODELS[:3], key=lambda m: info[m]["val_AUC"])
    print(f"[explain] best model by validation AUC: {best}  (SHAP focus: {best_shap})")

    print("[explain] permutation importance ...")
    importance_section(models, X, y, spec)
    print("[explain] SHAP ...")
    shap_section(models, X, y, best_shap)
    print("[explain] partial dependence / ICE ...")
    pdp_section(models, X, best_shap)
    glassbox_section(models, spec)
    print("[explain] done -> reports/figures/" + SUB)


if __name__ == "__main__":
    main()
