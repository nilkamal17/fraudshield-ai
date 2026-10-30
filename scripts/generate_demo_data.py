"""Generate a reproducible synthetic transaction dataset for local practice."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_data(rows: int = 12000, random_state: int = 42) -> pd.DataFrame:
    if rows < 500:
        raise ValueError("Generate at least 500 rows so the train/test split is useful.")

    rng = np.random.default_rng(random_state)
    transaction_types = np.array(["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"])
    type_probabilities = np.array([0.40, 0.25, 0.15, 0.15, 0.05])
    tx_type = rng.choice(transaction_types, size=rows, p=type_probabilities)
    step = rng.integers(1, 745, size=rows)

    fraud_probability = np.select(
        [tx_type == "TRANSFER", tx_type == "CASH_OUT"],
        [0.075, 0.025],
        default=0.0005,
    )
    is_fraud = rng.binomial(1, fraud_probability)

    # Legitimate transactions have a type-specific amount distribution.
    type_median = {
        "PAYMENT": 1800.0,
        "CASH_OUT": 12500.0,
        "CASH_IN": 9500.0,
        "TRANSFER": 16000.0,
        "DEBIT": 900.0,
    }
    medians = np.array([type_median[value] for value in tx_type])
    amount = rng.lognormal(mean=np.log(medians), sigma=0.85)

    # Fraudulent examples are more often higher-value transfers/cash-outs,
    # but overlap is retained so the demo is not perfectly separable.
    fraud_mask = is_fraud == 1
    amount[fraud_mask] = rng.lognormal(
        mean=np.log(medians[fraud_mask] * 2.2), sigma=0.75
    )
    amount = np.clip(amount, 1.0, 2_000_000.0).round(2)

    return pd.DataFrame(
        {
            "step": step.astype(int),
            "type": tx_type,
            "amount": amount,
            "isFraud": is_fraud.astype(int),
        }
    ).sort_values("step", kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=12000, help="Number of demo transactions.")
    parser.add_argument("--output", default="data/demo_transactions.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_demo_data(rows=args.rows, random_state=args.seed)
    data.to_csv(output_path, index=False)
    fraud_count = int(data["isFraud"].sum())
    fraud_rate = 100.0 * fraud_count / len(data)
    print(f"Saved {len(data):,} transactions to {output_path}")
    print(f"Fraud examples: {fraud_count:,} ({fraud_rate:.2f}%)")
    print("Note: this generated dataset is for software/pipeline practice, not real banking evidence.")


if __name__ == "__main__":
    main()
