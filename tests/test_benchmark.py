import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from benchmark import make_pipeline, choose_threshold
from prepare import CATEGORY, FEATURES

def test_pipeline_handles_unseen_category_and_missing_number():
    frame = pd.DataFrame({k: ["a", "b", "b", "a"] if k in CATEGORY else [1., 2., 3., 4.] for k in FEATURES})
    pipeline = make_pipeline(DecisionTreeClassifier(random_state=42), False)
    pipeline.fit(frame, [0, 1, 1, 0])
    frame.loc[0, CATEGORY[0]] = "unseen"
    frame.loc[0, "annual_inc"] = np.nan
    assert np.isfinite(pipeline.predict_proba(frame)).all()

def test_threshold_maximizes_validation_f1():
    assert choose_threshold([0, 0, 1, 1], [.1, .2, .3, .4]) == .3
