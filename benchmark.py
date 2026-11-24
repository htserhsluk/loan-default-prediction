"""Fit and evaluate all models on the same chronological splits."""
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline as chain
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from prepare import CATEGORY, FEATURES

def make_pipeline(model, use_pca):
    numeric = [k for k in FEATURES if k not in CATEGORY]
    numeric_steps = chain(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    category_steps = chain(SimpleImputer(strategy="constant", fill_value="Missing"),
                           OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    preprocess = ColumnTransformer([("numbers", numeric_steps, numeric), ("categories", category_steps, CATEGORY)])
    if use_pca:
        return chain(preprocess, PCA(n_components=.95, svd_solver="full"), model)
    return chain(preprocess, model)

def choose_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])

def models():
    return {
        "Logistic": LogisticRegression(max_iter=500),
        "PCA + Logistic": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=80, min_samples_leaf=10, n_jobs=2, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=60, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=42),
        "Histogram Boosting": HistGradientBoostingClassifier(max_iter=80, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=80, max_depth=3, n_jobs=2, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=80, n_jobs=2, verbosity=-1, random_state=42),
        "CatBoost": CatBoostClassifier(iterations=80, depth=5, thread_count=2, random_seed=42,
                                       verbose=False, allow_writing_files=False),
    }

def main():
    dataset = pd.read_csv("data/sample.csv.gz", index_col="source_row")
    splits = {name: frame for name, frame in dataset.groupby("split")}
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    metrics, predictions = [], []
    train = splits["train"]
    for name, estimator in models().items():
        pipeline = make_pipeline(estimator, name.startswith("PCA"))
        start = perf_counter()
        pipeline.fit(train[FEATURES], train.target)
        fit_seconds = perf_counter() - start
        path = artifacts / (name + ".joblib")
        joblib.dump(pipeline, path)
        for split in ["validation", "test"]:
            frame = splits[split]
            start = perf_counter()
            scores = pipeline.predict_proba(frame[FEATURES])[:, 1]
            elapsed = perf_counter() - start
            if split == "validation":
                threshold = choose_threshold(frame.target, scores)
            predicted = scores >= threshold
            tn, fp, fn, tp = confusion_matrix(frame.target, predicted, labels=[0, 1]).ravel()
            metrics.append(dict(model=name, split=split, threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp,
                                roc_auc=roc_auc_score(frame.target, scores),
                                average_precision=average_precision_score(frame.target, scores),
                                precision=precision_score(frame.target, predicted, zero_division=0),
                                recall=recall_score(frame.target, predicted),
                                fit_seconds=fit_seconds, inference_ms_per_1000=elapsed*1e6/len(frame),
                                artifact_bytes=path.stat().st_size))
            predictions.append(frame[["target", "loan_amnt"]].assign(model=name, split=split,
                                                                    score=scores, predicted=predicted))
        print(name, flush=True)
    table = pd.DataFrame(metrics)
    table.to_csv("reports/benchmark.csv", index=False)
    pd.concat(predictions).to_parquet(artifacts / "predictions.parquet")
    valid = table[table.split.eq("validation")]
    best = valid.loc[valid.average_precision.idxmax()]
    defaults = {k: (str(train[k].mode().iat[0]) if k in CATEGORY else float(train[k].median())) for k in FEATURES}
    categories = {k: sorted(train[k].dropna().unique()) for k in CATEGORY}
    metadata = dict(model=best.model, threshold=best.threshold, defaults=defaults, categories=categories)
    (artifacts / "selected.json").write_text(json.dumps(metadata, allow_nan=False, indent=2))

if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
