from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_loader import load_dataset
from src.evaluate import (
    build_classification_report,
    compute_metrics,
    plot_confusion_matrix,
    plot_genre_distribution,
    plot_model_comparison,
    print_model_report,
)
from src.preprocessing import prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def make_pipeline(model_name: str) -> Pipeline:
    """Build a text-classification pipeline using TF-IDF features and a classifier."""
    if model_name == "naive_bayes":
        classifier = MultinomialNB()
    elif model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, random_state=42)
    elif model_name == "linear_svm":
        classifier = LinearSVC(random_state=42)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    return Pipeline([
        ("tfidf", vectorizer),
        ("model", classifier),
    ])


def train_and_evaluate():
    """Train three models and select the best according to weighted F1-score."""
    dataset_path, df, files = load_dataset()
    print("Dataset path:", dataset_path)
    print("Detected files:")
    for file_path in files:
        print(f" - {file_path.name}")

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(3).to_string(index=False))

    cleaned_df = prepare_dataset(df)
    print("Cleaned shape:", cleaned_df.shape)
    print("Missing values:\n", cleaned_df.isna().sum())
    print("Duplicate rows:", cleaned_df.duplicated().sum())
    print("Genre distribution:\n", cleaned_df["genre"].value_counts().head(10))

    labels = sorted(cleaned_df["genre"].unique().tolist())
    X = cleaned_df["plot"].astype(str)
    y = cleaned_df["genre"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_results = {}
    best_model_name = None
    best_metrics = None
    best_model = None
    best_predictions = None

    for model_name in ["naive_bayes", "logistic_regression", "linear_svm"]:
        pipeline = make_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, labels=labels)
        model_results[model_name] = metrics
        print_model_report(model_name, metrics)

        if best_metrics is None or metrics["f1_weighted"] > best_metrics["f1_weighted"]:
            best_model_name = model_name
            best_metrics = metrics
            best_model = pipeline
            best_predictions = y_pred

    print("\nBest model by weighted F1-score:", best_model_name)
    print(best_metrics)

    plot_genre_distribution(cleaned_df["genre"], OUTPUTS_DIR / "genre_distribution.png")
    plot_model_comparison(model_results, OUTPUTS_DIR / "model_comparison.png")
    plot_confusion_matrix(y_test, best_predictions, labels, OUTPUTS_DIR / "confusion_matrix.png")

    report = build_classification_report(y_test, best_predictions, labels)
    print("\nClassification report for best model:\n")
    print(report)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    classifier = best_model.named_steps["model"]
    vectorizer = best_model.named_steps["tfidf"]
    joblib.dump(classifier, MODELS_DIR / "movie_genre_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")

    print(f"\nSaved best model to: {MODELS_DIR / 'movie_genre_model.pkl'}")
    print(f"Saved vectorizer to: {MODELS_DIR / 'tfidf_vectorizer.pkl'}")
    print(f"Saved plots to: {OUTPUTS_DIR}")
    return {
        "dataset_path": str(dataset_path),
        "best_model": best_model_name,
        "results": model_results,
        "metrics": best_metrics,
    }


if __name__ == "__main__":
    train_and_evaluate()
