"""Train and evaluate FraudShield AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features import TARGET_COLUMN, prepare_features
from src.model import build_model_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "demo_transactions.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "fraudshield_model.joblib"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "metrics.json"
REQUIRED_COLUMNS = ["step", "type", "amount", TARGET_COLUMN]


def load_dataset(data_path: str | Path, sample_size: int | None = None) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    # Read only the required fields: the full PaySim CSV is large, and ID/balance fields are not used.
    data = pd.read_csv(path, usecols=lambda column: column in REQUIRED_COLUMNS)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
            + ". Expected step, type, amount, and isFraud."
        )

    data = data[REQUIRED_COLUMNS].copy()
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=[TARGET_COLUMN])
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("isFraud must contain only 0 (legitimate) and 1 (fraud).")
    if data[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The dataset must contain both legitimate and fraud examples.")

    # Optional stratified sampling keeps the rare fraud class represented when
    # users load the full multi-million-row PaySim file on a personal computer.
    if sample_size is not None and sample_size < len(data):
        if sample_size < 1000:
            raise ValueError("--sample-size must be at least 1000.")
        if sample_size < data[TARGET_COLUMN].value_counts().min() * 2:
            raise ValueError("Sample size is too small for a stratified sample of this dataset.")
        data, _ = train_test_split(
            data,
            train_size=sample_size,
            stratify=data[TARGET_COLUMN],
            random_state=42,
        )
    return data.reset_index(drop=True)


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Choose an F2-oriented threshold on validation data to favour fraud recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    beta_squared = 4.0
    denominator = beta_squared * precision[:-1] + recall[:-1]
    f2_scores = np.divide(
        (1.0 + beta_squared) * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[int(np.argmax(f2_scores))])


def calculate_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": round(float(threshold), 6),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_true, predictions, zero_division=0)), 6),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "confusion_matrix": matrix.tolist(),
        "test_transactions": int(len(y_true)),
        "test_fraud_count": int(y_true.sum()),
    }


def train(
    data_path: str | Path = DEFAULT_DATA,
    model_path: str | Path = DEFAULT_MODEL,
    report_path: str | Path = DEFAULT_REPORT,
    sample_size: int | None = None,
) -> dict[str, Any]:
    data = load_dataset(data_path, sample_size=sample_size)
    labels = data[TARGET_COLUMN]
    if labels.value_counts().min() < 10:
        raise ValueError("Need at least 10 examples of each class to create reliable splits.")

    features = prepare_features(data)
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=42,
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    model = build_model_pipeline()
    model.fit(x_train, y_train)
    valid_probabilities = model.predict_proba(x_valid)[:, 1]
    threshold = choose_threshold(y_valid, valid_probabilities)
    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_metrics = calculate_metrics(y_test, test_probabilities, threshold)

    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = sorted(
        [
            {"feature": str(name), "importance": round(float(value), 6)}
            for name, value in zip(feature_names, classifier.feature_importances_)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:12]

    bundle = {
        "model": model,
        "threshold": threshold,
        "feature_columns": ["step", "type", "amount"],
        "model_name": "Random Forest (class-weighted)",
        "note": "Screening support only; not proof of fraud or a production payment decision.",
    }
    model_path = Path(model_path)
    report_path = Path(report_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    report = {
        "model_name": bundle["model_name"],
        "dataset_path": str(data_path),
        "dataset_rows_used": int(len(data)),
        "fraud_count_used": int(labels.sum()),
        "fraud_rate_used_percent": round(float(labels.mean() * 100), 4),
        "train_transactions": int(len(y_train)),
        "validation_transactions": int(len(y_valid)),
        **test_metrics,
        "feature_importance": importance,
        "evaluation_note": "Metrics are from a held-out test split of the selected dataset. Synthetic demo results are not real-world bank performance.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFraudShield AI training completed.")
    print(f"Rows used: {len(data):,} | Fraud examples: {int(labels.sum()):,}")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {report_path}")
    print("\nHeld-out test metrics:")
    for metric in ("precision", "recall", "f1_score", "pr_auc", "roc_auc", "threshold"):
        print(f"  {metric}: {report[metric]}")
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(np.array(report["confusion_matrix"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="CSV with step,type,amount,isFraud columns.")
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL), help="Where to save the trained model bundle.")
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT), help="Where to save evaluation metrics JSON.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional stratified row sample for large datasets, e.g. 250000.")
    args = parser.parse_args()
    train(args.data, args.model_output, args.report_output, args.sample_size)


if __name__ == "__main__":
    main()
