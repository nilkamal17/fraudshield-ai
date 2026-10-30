"""Load the trained model and score new transactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features import prepare_features

DEFAULT_MODEL_PATH = Path("models/fraudshield_model.joblib")


def load_model_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Train the model first with: "
            "python -m src.train"
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "threshold" not in bundle:
        raise ValueError("The saved model bundle is invalid. Please retrain the model.")
    return bundle


def score_transactions(
    transactions: pd.DataFrame,
    bundle: dict[str, Any],
) -> pd.DataFrame:
    """Return transaction rows plus fraud probability and review decision."""
    features = prepare_features(transactions)
    probabilities = bundle["model"].predict_proba(features)[:, 1]
    threshold = float(bundle["threshold"])
    medium_threshold = threshold * 0.5

    results = transactions.copy().reset_index(drop=True)
    results["fraud_probability"] = probabilities
    results["flag_for_review"] = probabilities >= threshold
    results["risk_level"] = [
        "HIGH" if probability >= threshold
        else "MEDIUM" if probability >= medium_threshold
        else "LOW"
        for probability in probabilities
    ]
    return results
