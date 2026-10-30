import pandas as pd
import pytest

from src.features import MODEL_FEATURES, prepare_features


def test_prepare_features_creates_model_columns():
    raw = pd.DataFrame({"step": [1, 25], "type": ["transfer", "PAYMENT"], "amount": [100.0, 250.0]})
    prepared = prepare_features(raw)
    assert list(prepared.columns) == MODEL_FEATURES
    assert prepared.loc[0, "type"] == "TRANSFER"
    assert prepared.loc[0, "amount_log1p"] > 0


def test_prepare_features_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required"):
        prepare_features(pd.DataFrame({"step": [1], "amount": [10]}))


def test_prepare_features_rejects_negative_amount():
    raw = pd.DataFrame({"step": [1], "type": ["TRANSFER"], "amount": [-1.0]})
    with pytest.raises(ValueError, match="amount cannot be negative"):
        prepare_features(raw)
