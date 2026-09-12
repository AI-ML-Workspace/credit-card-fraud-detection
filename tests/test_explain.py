import numpy as np
import pandas as pd
import pytest
from src import config
from src.explain import explain_single_prediction, plot_global_summary


def test_explain_single_prediction():
    row = pd.DataFrame([{c: 0.0 for c in config.ALL_FEATURE_COLUMNS}])
    contrib = explain_single_prediction(row)

    assert isinstance(contrib, pd.DataFrame)
    assert "feature" in contrib.columns
    assert "shap_value" in contrib.columns
    assert len(contrib) == len(config.ALL_FEATURE_COLUMNS)
