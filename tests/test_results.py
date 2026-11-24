"""End-to-end checks for the generated hand-in artifacts."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from benchmark import choose_threshold, models
from prepare import FEATURES

def test_results_match_saved_predictions_and_pipelines():
    sample = pd.read_csv("data/sample.csv.gz").set_index("source_row")
    assert len(sample) == sample.index.nunique() == 100000
    scores = pd.read_parquet("artifacts/predictions.parquet")
    table = pd.read_csv("reports/benchmark.csv").set_index(["model", "split"])
    assert set(scores.model) == set(models())
    assert len(table) == 2 * len(models()) and table.index.is_unique
    for name in models():
        pipeline = joblib.load(Path("artifacts", name + ".joblib"))
        for split in ["validation", "test"]:
            rows = sample[sample.split.eq(split)]
            pred = scores[scores.model.eq(name) & scores.split.eq(split)]
            assert pred.index.equals(rows.index)
            assert np.array_equal(pred.target, rows.target)
            assert np.allclose(pipeline.predict_proba(rows[FEATURES].iloc[:10])[:, 1], pred.score.iloc[:10])
            metric = table.loc[(name, split)]
            if split == "validation":
                threshold = choose_threshold(rows.target, pred.score)
            assert np.isclose(metric.threshold, threshold)
            assert np.array_equal(pred.predicted, pred.score >= threshold)
            counts = confusion_matrix(rows.target, pred.predicted).ravel()
            assert np.array_equal(counts, metric[["tn", "fp", "fn", "tp"]])
            assert np.isclose(average_precision_score(rows.target, pred.score), metric.average_precision)
            assert np.isclose(roc_auc_score(rows.target, pred.score), metric.roc_auc)
    selection = json.loads(Path("artifacts/selected.json").read_text())
    assert selection["model"] == table.xs("validation", level="split").average_precision.idxmax()
