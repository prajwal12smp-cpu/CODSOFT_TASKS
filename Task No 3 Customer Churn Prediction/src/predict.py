from __future__ import annotations

import joblib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_prediction_model.pkl"


def load_model_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Train the model first.")
    return joblib.load(MODEL_PATH)


def predict_customer(customer_data: dict) -> tuple[str, float]:
    """Return churn prediction label and probability for a single customer."""
    artifact = load_model_artifact()
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    record = {column: customer_data.get(column) for column in feature_columns}
    df = pd.DataFrame([record], columns=feature_columns)

    prediction = pipeline.predict(df)[0]
    probability = float(pipeline.predict_proba(df)[0, 1])
    label = "Yes" if prediction == 1 else "No"
    return label, probability


def prepare_customer_from_inputs(**kwargs):
    return {key: kwargs.get(key) for key in [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ] if key in kwargs}


if __name__ == "__main__":
    sample = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 3,
        "Balance": 15000,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 95000,
    }
    prediction, probability = predict_customer(sample)
    print(f"Prediction: {prediction}")
    print(f"Probability: {probability:.4f}")
