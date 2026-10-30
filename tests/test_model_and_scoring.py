import pandas as pd

from scripts.generate_demo_data import generate_demo_data
from src.features import TARGET_COLUMN, prepare_features
from src.inference import score_transactions
from src.model import build_model_pipeline


def test_pipeline_can_fit_and_score_transactions():
    data = generate_demo_data(rows=2500, random_state=7)
    x = prepare_features(data)
    y = data[TARGET_COLUMN]
    model = build_model_pipeline()
    model.fit(x, y)

    sample = pd.DataFrame(
        {
            "step": [20, 300],
            "type": ["PAYMENT", "TRANSFER"],
            "amount": [300.0, 150000.0],
        }
    )
    bundle = {"model": model, "threshold": 0.5}
    scored = score_transactions(sample, bundle)

    assert len(scored) == 2
    assert scored["fraud_probability"].between(0, 1).all()
    assert set(scored["risk_level"]).issubset({"LOW", "MEDIUM", "HIGH"})
    assert scored["flag_for_review"].dtype == bool
