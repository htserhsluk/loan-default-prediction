import pandas as pd
from prepare import eligible_rows, FEATURES, assign_split

def test_cohort_and_features():
    frame = pd.DataFrame({
        "issue_d": ["Jan-2012", "Jan-2014", "Jan-2015", "Jan-2016", "Jan-2015"],
        "term": [" 36 months"] * 4 + ["60 months"],
        "loan_status": ["Fully Paid", "Current", "Charged Off", "Default", "Default"],
    })
    eligible = eligible_rows(frame)
    assert eligible.index.tolist() == [0, 2]
    assert eligible.target.tolist() == [0, 1]
    assert assign_split(eligible.year).tolist() == ["train", "test"]
    assert len(set(FEATURES)) >= 50
    assert not set(FEATURES) & {"recoveries", "last_pymnt_d", "total_pymnt", "id", "loan_status", "issue_d"}
