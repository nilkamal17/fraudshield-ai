"""FraudShield AI — a local transaction fraud-screening dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference import DEFAULT_MODEL_PATH, load_model_bundle, score_transactions

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.set_page_config(
    page_title="FraudShield AI | Transaction Risk Screening",
    page_icon="🛡️",
    layout="wide",
)


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


@st.cache_resource(show_spinner=False)
def get_bundle(model_path: str, modified_time: float):
    # modified_time is part of the cache key, so retraining refreshes the model.
    return load_model_bundle(model_path)


st.title("🛡️ FraudShield AI")
st.subheader("Real-Time Financial Transaction Fraud Screening")
st.caption(
    "A student-built machine-learning prototype that scores transactions and flags suspicious activity for human review."
)

if not DEFAULT_MODEL_PATH.exists():
    st.warning("The model has not been trained yet. Run the training command below first.")
    st.code("python -m src.train", language="powershell")
    st.stop()

try:
    bundle = get_bundle(str(DEFAULT_MODEL_PATH), DEFAULT_MODEL_PATH.stat().st_mtime)
except Exception as exc:  # show a useful message instead of crashing the app
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

metrics = load_metrics()
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Single Transaction Check", "Batch CSV Screening", "Model Performance", "About & Limitations"],
)

if page == "Overview":
    st.markdown("### Project overview")
    st.write(
        "FraudShield AI estimates the fraud risk of a financial transaction using its "
        "time step, transaction type, and amount. Transactions above the tuned decision "
        "threshold are flagged for investigation."
    )
    first, second, third, fourth = st.columns(4)
    first.metric("Model", "Random Forest")
    second.metric("Transactions evaluated", f"{metrics.get('test_transactions', 0):,}")
    third.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}" if metrics else "Train model")
    fourth.metric("Recall", f"{metrics.get('recall', 0):.1%}" if metrics else "Train model")

    st.markdown("### How it works")
    st.write("1. Read transaction fields.  2. Build time and amount features.  3. Estimate fraud probability.  4. Flag high-risk transactions for human review.")
    st.info(
        "This dashboard performs local, on-demand scoring. It is a real-time-style demonstration, "
        "not a live connection to a bank, payment gateway, or streaming transaction system."
    )
    st.markdown("### Quick start")
    st.code("python -m src.train\nstreamlit run app.py", language="powershell")

elif page == "Single Transaction Check":
    st.markdown("### Score one transaction")
    st.write("Enter the fields for a transaction. The model returns a risk score, not a final fraud verdict.")
    with st.form("single_transaction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            step = st.number_input("Step (hour index)", min_value=0, max_value=100000, value=120, step=1)
        with col2:
            transaction_type = st.selectbox("Transaction type", ["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"])
        with col3:
            amount = st.number_input("Transaction amount", min_value=0.0, value=15000.0, step=500.0, format="%.2f")
        submitted = st.form_submit_button("Analyse transaction", type="primary")

    if submitted:
        transaction = pd.DataFrame([{"step": step, "type": transaction_type, "amount": amount}])
        try:
            result = score_transactions(transaction, bundle).iloc[0]
            probability = float(result["fraud_probability"])
            left, right = st.columns(2)
            left.metric("Estimated fraud probability", f"{probability:.1%}")
            right.metric("Risk level", str(result["risk_level"]))
            if bool(result["flag_for_review"]):
                st.error("Flagged for review: this transaction crossed the model's decision threshold.")
            else:
                st.success("Not flagged by the current model threshold.")
            st.caption(f"Decision threshold: {float(bundle['threshold']):.4f}. A human analyst should review any alert.")
        except Exception as exc:
            st.error(f"Could not score this transaction: {exc}")

elif page == "Batch CSV Screening":
    st.markdown("### Screen multiple transactions")
    st.write("Upload a CSV containing `step`, `type`, and `amount`. An optional `isFraud` column is ignored during scoring.")
    sample_path = ROOT / "data" / "demo_transactions.csv"
    if sample_path.exists():
        st.download_button(
            "Download sample transaction CSV",
            data=sample_path.read_bytes(),
            file_name="sample_transactions.csv",
            mime="text/csv",
        )
    uploaded_file = st.file_uploader("Choose a transaction CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            incoming = pd.read_csv(uploaded_file)
            st.write("Preview", incoming.head(10))
            results = score_transactions(incoming, bundle)
            st.success(f"Scored {len(results):,} transactions.")
            high_risk = int(results["flag_for_review"].sum())
            a, b = st.columns(2)
            a.metric("Flagged for review", f"{high_risk:,}")
            b.metric("Flag rate", f"{high_risk / max(len(results), 1):.1%}")
            st.dataframe(results, use_container_width=True)
            st.download_button(
                "Download scored transactions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="fraudshield_scored_transactions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

elif page == "Model Performance":
    st.markdown("### Held-out test performance")
    if not metrics:
        st.info("No evaluation report found. Train the model first: `python -m src.train`.")
    else:
        st.caption("These metrics come from the held-out test split recorded during model training.")
        cols = st.columns(5)
        for col, key, label in zip(
            cols,
            ["precision", "recall", "f1_score", "pr_auc", "roc_auc"],
            ["Precision", "Recall", "F1 score", "PR-AUC", "ROC-AUC"],
        ):
            col.metric(label, f"{metrics.get(key, 0):.3f}")
        st.markdown("#### Confusion matrix")
        matrix = metrics.get("confusion_matrix")
        if matrix and len(matrix) == 2:
            st.dataframe(
                pd.DataFrame(matrix, index=["Actual legitimate", "Actual fraud"], columns=["Predicted legitimate", "Predicted fraud"]),
                use_container_width=True,
            )
        st.markdown("#### Most influential model features")
        importance = metrics.get("feature_importance", [])
        if importance:
            importance_frame = pd.DataFrame(importance).set_index("feature")
            st.bar_chart(importance_frame["importance"])
        st.write(f"Threshold selected on validation data: **{metrics.get('threshold', 'N/A')}**")
        st.caption(metrics.get("evaluation_note", ""))

else:
    st.markdown("### About the project")
    st.write(
        "FraudShield AI demonstrates an end-to-end fraud-screening workflow: data validation, "
        "feature engineering, class-imbalance-aware model training, validation threshold selection, "
        "test evaluation, and a Streamlit interface."
    )
    st.markdown("### Dataset notes")
    st.write(
        "The included CSV is generated synthetic practice data. The project can also read the "
        "PaySim CSV after you download it yourself and place it in the data folder. PaySim is a "
        "simulated mobile-money dataset, not real customer banking records."
    )
    st.markdown("### Responsible use and limitations")
    st.warning(
        "A fraud probability is a prioritisation signal, not proof of criminal activity. "
        "Do not automatically block real customers based only on this student model. "
        "Real deployment requires bank-approved data, privacy controls, fairness testing, "
        "security review, drift monitoring, human oversight, and operational validation."
    )
    st.write(
        "The current app is a local prototype, not a production streaming platform. "
        "It does not connect to a bank or payment gateway, and it does not store uploaded transactions."
    )

st.divider()
st.caption("FraudShield AI • Educational prototype • Local transaction scoring")
