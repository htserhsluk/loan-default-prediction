import pandas as pd
import pytest
from analyze import validate_predictions

def test_rejects_duplicate_prediction():
    data = pd.DataFrame({"source_row": [1, 1], "model": ["Tree", "Tree"], "target": [0, 0]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_predictions(data)
