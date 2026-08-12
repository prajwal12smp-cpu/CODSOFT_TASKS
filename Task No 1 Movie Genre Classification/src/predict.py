from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.preprocessing import clean_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "movie_genre_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"


@lru_cache(maxsize=1)
def load_model_artifacts(model_path: str | Path = MODEL_PATH, vectorizer_path: str | Path = VECTORIZER_PATH):
    """Load the trained model and TF-IDF vectorizer from disk using a cache to avoid retraining at startup."""
    model_path = Path(model_path)
    vectorizer_path = Path(vectorizer_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_genre(plot_text: str):
    """Return the predicted genre and probability information when supported."""
    if not isinstance(plot_text, str) or not plot_text.strip():
        raise ValueError("Please enter a non-empty movie plot/description.")

    model, vectorizer = load_model_artifacts()
    cleaned_plot = clean_text(plot_text)
    X = vectorizer.transform([cleaned_plot])

    prediction = model.predict(X)[0]
    result = {
        "predicted_genre": prediction,
        "confidence": None,
        "top_predictions": [],
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        class_labels = model.classes_
        ranked = sorted(zip(class_labels, probabilities), key=lambda item: item[1], reverse=True)
        top_three = ranked[:3]
        result["top_predictions"] = [
            {"genre": genre, "confidence": round(float(confidence) * 100, 2)}
            for genre, confidence in top_three
        ]
        result["confidence"] = round(float(max(probabilities)) * 100, 2)
    else:
        # For models like LinearSVC, convert raw decision scores into a probability-like distribution.
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)[0]
            shifted_scores = scores - np.max(scores)
            exp_scores = np.exp(shifted_scores)
            probabilities = exp_scores / np.sum(exp_scores)
            class_labels = model.classes_
            ranked = sorted(zip(class_labels, probabilities), key=lambda item: item[1], reverse=True)
            top_three = ranked[:3]
            result["top_predictions"] = [
                {"genre": genre, "confidence": round(float(confidence) * 100, 2)}
                for genre, confidence in top_three
            ]
            result["confidence"] = round(float(max(probabilities)) * 100, 2)

    return result
