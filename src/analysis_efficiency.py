"""
Analysis C - Model Efficiency Analysis
======================================
Computational cost of every implemented model and the trade-off with
predictive performance.

Measured for each model
  * training time on the full 210k training set (benchmark run, incl. early
    stopping) + total hyper-parameter tuning time
  * batch inference time (45k test rows) -> microseconds per prediction
  * single-applicant latency (what the UI / a real-time API would see)
  * serialized model size (MB) and structural complexity
    (#tree nodes / #coefficients)
  * training-time scaling with data size (10k -> 210k rows, fixed
    hyper-parameters) -> empirical complexity exponent from a log-log fit

Outputs reports/tables/efficiency_*.csv and reports/figures/analysis_C_efficiency/*
Usage:
    python -m src.analysis_efficiency [--sizes 10000 30000 100000 210000]
"""
import argparse
import json
import pickle
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import MODELS_DIR, SEED, TABLE_DIR
from src.features import load_splits
from src.models import MODEL_ORDER, fit_estimator, fixed_params, make_estimator, proba
from src.plotting import INK_2, MODEL_COLORS, MODEL_STYLE, MUTED, save
from src.train_benchmark import slug

SUB = "analysis_C_efficiency"


def complexity(name, est) -> dict:
    """Structural size of the fitted model."""
    if name in ("Logistic Regression", "PCA + Logistic Regression"):
        clf = est.named_steps["clf"]
        n = clf.coef_.size + clf.intercept_.size
        if "pca" in est.named_steps:
            n += est.named_steps["pca"].components_.size
        return {"n_trees": 0, "n_nodes": 0, "n_parameters": int(n)}
    if name == "Decision Tree":
        return {"n_trees": 1, "n_nodes": int(est.tree_.node_count),
                "n_parameters": int(est.tree_.node_count), "depth": int(est.get_depth())}
    if name in ("Random Forest", "AdaBoost"):
        nodes = sum(t.tree_.node_count for t in est.estimators_)
        return {"n_trees": len(est.estimators_), "n_nodes": int(nodes), "n_parameters": int(nodes)}
    if name == "Gradient Boosting":
        nodes = sum(t[0].tree_.node_count for t in est.estimators_)
        return {"n_trees": len(est.estimators_), "n_nodes": int(nodes), "n_parameters": int(nodes)}
    if name == "XGBoost":
        b = est.get_booster()
        n_trees = est.best_iteration + 1 if getattr(est, "best_iteration", None) is not None else b.num_boosted_rounds()
        nodes = sum(d.count("\n") for d in b.get_dump()[:n_trees])
        return {"n_trees": int(n_trees), "n_nodes": int(nodes), "n_parameters": int(nodes)}
    if name == "LightGBM":
        dm = est.booster_.dump_model()
        nodes = sum(2 * t["num_leaves"] - 1 for t in dm["tree_info"])
        return {"n_trees": len(dm["tree_info"]), "n_nodes": int(nodes), "n_parameters": int(nodes)}
    if name == "CatBoost":
        n_trees = est.tree_count_
        depth = est.get_params().get("depth", 6)
        nodes = n_trees * (2 ** (depth + 1) - 1)        # symmetric (oblivious) trees
        return {"n_trees": int(n_trees), "n_nodes": int(nodes), "n_parameters": int(nodes)}
    return {}


def latency(est, X, single_reps=200) -> tuple[float, float]:
    # batch: best of 3
    batch = min(_t(lambda: proba(est, X)) for _ in range(3))
    # single row: median of many calls
    x1 = X[:1]
    proba(est, x1)  # warm-up
    singles = [_t(lambda: proba(est, x1)) for _ in range(single_reps)]
    return batch / len(X) * 1e6, float(np.median(singles)) * 1e3


def _t(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def pareto(df, x, y="ROC_AUC"):
    """Models not dominated (lower cost AND higher AUC)."""
    s = df.sort_values(x)
    best, front = -np.inf, []
    for _, r in s.iterrows():
        if r[y] > best:
            front.append(r["model"])
            best = r[y]
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 30_000, 100_000, 210_000])
    args = ap.parse_args()

    d = load_splits()
    bench = pd.read_csv(TABLE_DIR / "benchmark_results.csv").set_index("model")
    info = json.loads((TABLE_DIR / "benchmark_info.json").read_text())

    # ---------------- per-model cost of the benchmark models ----------------
    rows = []
    for name in MODEL_ORDER:
        lm = joblib.load(MODELS_DIR / f"{slug(name)}.joblib")
        Xte = lm.transform(d["X_test"])
        us_per_row, single_ms = latency(lm.estimator, Xte)
        t = time.perf_counter()
        lm.transform(d["X_test"].iloc[:1])
        prep_ms = (time.perf_counter() - t) * 1e3
        rows.append({
            "model": name,
            "ROC_AUC": bench.loc[name, "ROC_AUC"],
            "train_seconds": info[name]["train_seconds"],
            "tuning_seconds": info[name]["tuning_seconds"],
            "inference_us_per_row": us_per_row,
            "single_prediction_ms": single_ms,
            "preprocessing_ms_single": prep_ms,
            "model_size_mb": len(pickle.dumps(lm.estimator)) / 1e6,
            **complexity(name, lm.estimator),
        })
        print(f"  {name:<26} train {rows[-1]['train_seconds']:7.1f}s  "
              f"{us_per_row:8.2f} us/row  single {single_ms:6.2f} ms  "
              f"{rows[-1]['model_size_mb']:8.2f} MB")
    eff = pd.DataFrame(rows)
    eff["AUC_per_train_second"] = (eff["ROC_AUC"] - 0.5) / eff["train_seconds"]
    for c in ("train_seconds", "inference_us_per_row", "model_size_mb"):
        eff[f"pareto_{c}"] = eff["model"].isin(pareto(eff, c))
    eff.to_csv(TABLE_DIR / "efficiency_summary.csv", index=False)

    # ---------------- training-time scaling ----------------
    scal = []
    for n in args.sizes:
        if n >= len(d["X_train"]):
            Xs, ys = d["X_train"], d["y_train"]
        else:
            Xs, _, ys, _ = train_test_split(d["X_train"], d["y_train"], train_size=n,
                                            stratify=d["y_train"], random_state=SEED)
        lm0 = joblib.load(MODELS_DIR / "logistic_regression.joblib")
        X = lm0.transform(Xs)                       # same fitted preprocessor for all
        for name in MODEL_ORDER:
            est = make_estimator(name, fixed_params(name, info[name]))
            if name == "Gradient Boosting":       # fixed #trees = the early-stopped count
                est.set_params(n_estimators=info[name]["n_iterations"], n_iter_no_change=None)
            t = time.perf_counter()
            fit_estimator(name, est, X, ys)
            sec = time.perf_counter() - t
            scal.append({"model": name, "train_size": len(ys), "train_seconds": sec})
            print(f"  scaling n={len(ys):>7,}  {name:<26} {sec:7.2f}s")
    sc = pd.DataFrame(scal)
    expo = (sc.groupby("model")
              .apply(lambda g: np.polyfit(np.log(g.train_size), np.log(g.train_seconds), 1)[0],
                     include_groups=False)
              .rename("scaling_exponent"))
    sc.to_csv(TABLE_DIR / "efficiency_scaling.csv", index=False)
    eff = eff.merge(expo, left_on="model", right_index=True, how="left")
    eff.to_csv(TABLE_DIR / "efficiency_summary.csv", index=False)

    plots(eff, sc)
    print("\n", eff[["model", "ROC_AUC", "train_seconds", "inference_us_per_row",
                     "single_prediction_ms", "model_size_mb", "n_nodes",
                     "scaling_exponent"]].round(3).to_string(index=False))


def plots(eff, sc):
    order = eff.sort_values("train_seconds")["model"].tolist()
    e = eff.set_index("model").loc[order]

    # cost bars (log scale), one panel per measure
    measures = [("train_seconds", "Training time (s)"),
                ("inference_us_per_row", "Batch inference (us / row)"),
                ("single_prediction_ms", "Single-applicant latency (ms)"),
                ("model_size_mb", "Serialized size (MB)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for ax, (c, lab) in zip(axes, measures):
        ax.barh(order, e[c], color=[MODEL_COLORS[m] for m in order], height=0.65)
        for i, v in enumerate(e[c]):
            ax.text(v, i, f" {v:.3g}", va="center", fontsize=7, color=INK_2)
        ax.set_xscale("log")
        ax.set_title(lab, fontsize=10)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Computational cost per model (log scale)", fontsize=13)
    fig.tight_layout()
    save(fig, "cost_bars", SUB)

    # trade-off scatter plots with Pareto frontier
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (c, lab) in zip(axes, [("train_seconds", "Training time (s, log)"),
                                   ("inference_us_per_row", "Inference time per row (us, log)"),
                                   ("model_size_mb", "Model size (MB, log)")]):
        front = eff[eff[f"pareto_{c}"]].sort_values(c)
        ax.plot(front[c], front["ROC_AUC"], color=MUTED, lw=1, ls="--", zorder=1,
                label="Pareto frontier")
        for _, r in eff.iterrows():
            ax.scatter(r[c], r["ROC_AUC"], s=70, color=MODEL_COLORS[r["model"]],
                       edgecolors="white", linewidths=1.5, zorder=2)
            off = {"XGBoost": (5, -11), "LightGBM": (-8, 6)}.get(r["model"], (5, 4))
            ax.annotate(r["model"], (r[c], r["ROC_AUC"]), xytext=off,
                        ha="right" if off[0] < 0 else "left",
                        textcoords="offset points", fontsize=7, color=INK_2)
        ax.set_xscale("log")
        ax.set(xlabel=lab, ylabel="Test ROC-AUC")
        ax.legend(loc="lower right")
    fig.suptitle("Performance vs computational cost", fontsize=13)
    fig.tight_layout()
    save(fig, "tradeoff_pareto", SUB)

    # scaling log-log
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for m in eff["model"]:
        s = sc[sc.model == m].sort_values("train_size")
        k = eff.set_index("model").loc[m, "scaling_exponent"]
        ax.plot(s["train_size"], s["train_seconds"], marker="o", ms=4,
                color=MODEL_COLORS[m], ls=MODEL_STYLE.get(m, "-"), label=f"{m} (slope {k:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="Training rows (log)", ylabel="Training time in s (log)",
           title="Training-time scaling with data size")
    ax.legend(fontsize=7)
    save(fig, "training_scaling", SUB)


if __name__ == "__main__":
    main()
