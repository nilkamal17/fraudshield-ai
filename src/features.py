"""Feature preparation for transaction-level fraud screening."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_INPUT_COLUMNS = ["step", "type", "amount"]
NUMERIC_FEATURES = ["step", "amount", "amount_log1p", "hour_sin", "hour_cos"]
CATEGORICAL_FEATURES = ["type"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "isFraud"


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Validate raw transaction fields and create model-ready features.

    PaySim's published dataset notes that balance fields should not be used
    for fraud detection because fraudulent transactions are cancelled. This
    project therefore deliberately uses step, transaction type, and amount.
    """
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "Missing required transaction columns: " + ", ".join(missing)
        )

    frame = data[REQUIRED_INPUT_COLUMNS].copy()
    frame["step"] = pd.to_numeric(frame["step"], errors="coerce")
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    frame["type"] = frame["type"].astype("string").str.upper().str.strip()

    if frame[["step", "amount"]].isna().any().any():
        raise ValueError("step and amount must contain valid numeric values.")
    if (frame["amount"] < 0).any():
        raise ValueError("amount cannot be negative.")
    if (frame["step"] < 0).any():
        raise ValueError("step cannot be negative.")
    if frame["type"].isna().any() or (frame["type"] == "").any():
        raise ValueError("type cannot be empty.")

    frame["amount_log1p"] = np.log1p(frame["amount"].astype(float))
    hour_of_day = frame["step"].astype(float) % 24.0
    frame["hour_sin"] = np.sin(2.0 * np.pi * hour_of_day / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hour_of_day / 24.0)

    return frame[MODEL_FEATURES]
