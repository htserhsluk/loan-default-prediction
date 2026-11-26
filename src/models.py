"""
Model factory, training helpers, evaluation metrics and the LoanModel wrapper
(preprocessor + estimator) that is saved to disk and used by the UI.

Models benchmarked (as required by the brief):
  * Principal Component Analysis  -> PCA (95 % variance) + Logistic Regression
                                     (plain Logistic Regression is the baseline)
  * Decision Tree
  * Random Forest
  * Boosting: AdaBoost, Gradient Boosting (sklearn), XGBoost, LightGBM, CatBoost
"""
import time

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import SEED

MODEL_ORDER = [
    "Logistic Regression",
    "PCA + Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
]
MODEL_FAMILY = {
    "Logistic Regression": "Linear baseline",
    "PCA + Logistic Regression": "PCA",
    "Decision Tree": "Single tree",
    "Random Forest": "Bagging ensemble",
    "AdaBoost": "Boosting",
    "Gradient Boosting": "Boosting",
    "XGBoost": "Boosting",
    "LightGBM": "Boosting",
    "CatBoost": "Boosting",
}

# Small, validation-based hyper-parameter grids (kept small on purpose -
# compute budget on a laptop).  The best setting is picked on the validation
# AUC; the test set is never used for any choice.
PARAM_GRIDS = {
    "Decision Tree": [{"max_depth": d, "min_samples_leaf": l}
                      for d in (4, 6, 8, 10, 12) for l in (50, 200)],
    "Random Forest": [{"max_depth": None, "min_samples_leaf": 50},
                      {"max_depth": 16, "min_samples_leaf": 20}],
    "Logistic Regression": [{"C": c} for c in (0.01, 1.0)],
    "PCA + Logistic Regression": [{"n_components": k} for k in (0.80, 0.90, 0.95)],
}


# --------------------------------------------------------------------------
# Estimator factory
# --------------------------------------------------------------------------
def make_estimator(name: str, params: dict | None = None, fast: bool = False):
    """Return an *unfitted* estimator working on the preprocessed matrix.
    fast=True gives cheaper settings (used by ablation / scaling studies)."""
    p = dict(params or {})
    if name == "Logistic Regression":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=p.get("C", 1.0), max_iter=2000))])
    if name == "PCA + Logistic Regression":
        return Pipeline([("scale", StandardScaler()),
                         ("pca", PCA(n_components=p.get("n_components", 0.95),
                                     random_state=SEED)),
                         ("clf", LogisticRegression(max_iter=2000))])
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=p.get("max_depth", 8),
                                      min_samples_leaf=p.get("min_samples_leaf", 200),
                                      random_state=SEED)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100 if fast else 300,
                                      max_depth=p.get("max_depth", 16),
                                      min_samples_leaf=p.get("min_samples_leaf", 20),
                                      max_features="sqrt", n_jobs=-1,
                                      random_state=SEED)
    if name == "AdaBoost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, max_features=0.3,
                                             random_state=SEED),
            n_estimators=p.get("n_estimators", 100 if fast else 250),
            learning_rate=0.5, random_state=SEED)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=p.get("n_estimators", 150 if fast else 400),
            learning_rate=0.1, max_depth=3, subsample=0.5, max_features=0.3,
            validation_fraction=0.1, n_iter_no_change=None if fast else 20,
            random_state=SEED)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=p.get("n_estimators", 2000), learning_rate=0.05,
            max_depth=5, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_lambda=1.0, tree_method="hist", eval_metric="auc",
            early_stopping_rounds=None if "n_estimators" in p else 100,
            n_jobs=-1, random_state=SEED)
    if name == "LightGBM":
        return LGBMClassifier(
            n_estimators=p.get("n_estimators", 3000), learning_rate=0.03,
            num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0, n_jobs=-1, random_state=SEED,
            verbose=-1)
    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=p.get("n_estimators", 2000), learning_rate=0.05, depth=6,
            eval_metric="AUC", random_seed=SEED, verbose=0, thread_count=-1,
            allow_writing_files=False)
    raise ValueError(name)


def fit_estimator(name, est, Xtr, ytr, Xva=None, yva=None):
    """Fit with early stopping on the validation set where supported."""
    if name == "XGBoost" and Xva is not None and est.get_params()["early_stopping_rounds"]:
        est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif name == "LightGBM" and Xva is not None and est.get_params()["n_estimators"] >= 1000:
        est.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(yva,), eval_metric="auc",
                callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    elif name == "CatBoost" and Xva is not None and est.get_params()["iterations"] >= 1000:
        est.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=100,
                use_best_model=True)
    else:
        est.fit(Xtr, ytr)
    return est


def best_iteration(name, est):
    """Number of boosting rounds actually used (after early stopping)."""
    if name == "XGBoost":
        return int(getattr(est, "best_iteration", est.n_estimators - 1)) + 1
    if name == "LightGBM":
        return int(est.best_iteration_ or est.n_estimators)
    if name == "CatBoost":
        return int(est.get_best_iteration() or est.tree_count_) + 1
    if name == "Gradient Boosting":
        return int(est.n_estimators_)
    if name == "AdaBoost":
        return len(est.estimators_)
    return None


def fixed_params(name, info: dict) -> dict:
    """Parameters to re-train a model without early stopping (used by the
    ablation / scaling analyses) based on the benchmark result."""
    p = dict(info.get("best_params") or {})
    if info.get("n_iterations") and name in ("XGBoost", "LightGBM", "CatBoost"):
        p["n_estimators"] = int(info["n_iterations"])
    return p


def proba(est, X) -> np.ndarray:
    return est.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])])


def evaluate(y, p, threshold: float) -> dict:
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, p)
    return {
        "ROC_AUC": auc,
        "Gini": 2 * auc - 1,
        "PR_AUC": average_precision_score(y, p),
        "KS": ks_statistic(y, p),
        "Brier": brier_score_loss(y, p),
        "LogLoss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "Threshold": threshold,
        "Accuracy": accuracy_score(y, yhat),
        "Balanced_Acc": balanced_accuracy_score(y, yhat),
        "Precision": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1": f1_score(y, yhat),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


# --------------------------------------------------------------------------
# Deployable model = fitted preprocessor + fitted estimator
# --------------------------------------------------------------------------
class LoanModel:
    """Wraps the fitted ColumnTransformer and estimator so that it can score
    engineered applicant rows directly (used by the analyses and the UI)."""

    def __init__(self, name, preprocessor, estimator, threshold=0.5,
                 raw_columns=None, info=None):
        self.name = name
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.threshold = threshold
        self.raw_columns = list(raw_columns) if raw_columns is not None else None
        self.feature_names = list(preprocessor.get_feature_names_out())
        self.info = info or {}

    def transform(self, X):
        if self.raw_columns is not None:
            X = X[self.raw_columns]
        return self.preprocessor.transform(X).astype(np.float32)

    def predict_proba(self, X) -> np.ndarray:
        return proba(self.estimator, self.transform(X))

    def predict(self, X, threshold=None) -> np.ndarray:
        t = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= t).astype(int)


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0
