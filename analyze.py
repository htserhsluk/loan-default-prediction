"""Generate held-out comparisons and plots from the benchmark artifacts."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

REPORTS = Path("reports")

def validate_predictions(data):
    if data.duplicated(["model", "source_row"]).any():
        raise ValueError("duplicate model/loan prediction")
    counts = data.groupby("source_row").agg(models=("model", "size"), labels=("target", "nunique"))
    if not counts.models.eq(data.model.nunique()).all() or not counts.labels.eq(1).all():
        raise ValueError("model rows or labels differ")

def save(fig, name):
    fig.tight_layout()
    fig.savefig(REPORTS / name, dpi=135, bbox_inches="tight")
    plt.close(fig)

def main():
    sample = pd.read_csv("data/sample.csv.gz")
    pred = pd.read_parquet("artifacts/predictions.parquet").reset_index().query("split == 'test'").copy()
    validate_predictions(pred)
    table = pd.read_csv(REPORTS / "benchmark.csv").query("split == 'test'").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for target, group in sample.groupby("target"):
        axes[0].hist(group.loan_amnt, bins=np.arange(0, 42500, 2500), density=True, histtype="step",
                     linewidth=2, label=["Repaid", "Default"][target])
    axes[0].set(xlabel="Requested amount ($)", ylabel="Density")
    axes[0].legend()
    sample.groupby("year").target.mean().plot.bar(ax=axes[1], color="#346e83", rot=0)
    axes[1].set(xlabel="Issue year", ylabel="Observed default fraction")
    save(fig, "cohort.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in pred.groupby("model"):
        precision, recall, _ = precision_recall_curve(group.target, group.score)
        ax.plot(recall, precision, label=model, alpha=.8)
    ax.axhline(pred.target.mean(), color="black", linestyle="--", label="Default prevalence")
    ax.legend(fontsize=8, ncol=2)
    ax.set(xlabel="Recall of defaults", ylabel="Precision", ylim=(0, 1))
    save(fig, "precision-recall.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    for ax, column, label in zip(axes, ["fit_seconds", "inference_ms_per_1000", "artifact_bytes"],
                                 ["Fit time (s)", "Inference (ms / 1,000 loans)", "Pipeline size (bytes)"]):
        ax.barh(table.index, table[column], color="#346e83")
        ax.set(xscale="log", xlabel=label)
    save(fig, "efficiency.png")
    pred["amount_band"] = pd.cut(pred.loan_amnt, [-np.inf, 10000, 20000, np.inf],
                                  labels=["Up to $10k", "$10k–20k", "Above $20k"])
    pred["missed"] = pred.target.eq(1) & ~pred.predicted
    pred["false_alarm"] = pred.target.eq(0) & pred.predicted
    errors = pred.groupby(["model", "amount_band"], observed=True).agg(
        loans=("target", "size"), defaults=("target", "sum"), missed=("missed", "sum"),
        false_alarms=("false_alarm", "sum"))
    errors["miss_rate"] = errors.missed / errors.defaults
    errors["false_alarm_rate"] = errors.false_alarms / (errors.loans - errors.defaults)
    errors.to_csv(REPORTS / "errors.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, column, label in zip(axes, ["miss_rate", "false_alarm_rate"], ["Missed defaults", "False alarms"]):
        grid = errors[column].unstack()
        ax.imshow(grid, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set(title=label, xticks=range(len(grid.columns)), xticklabels=list(grid.columns),
               yticks=range(len(grid)), yticklabels=list(grid.index))
        for i, row in enumerate(grid.to_numpy()):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.0%}", ha="center", va="center",
                        color="white" if value > .5 else "black")
    save(fig, "errors.png")

if __name__ == "__main__":
    main()
