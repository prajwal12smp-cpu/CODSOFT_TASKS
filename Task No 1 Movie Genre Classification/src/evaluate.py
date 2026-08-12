from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, labels=None):
    """Compute multiclass metrics using a weighted average, suitable for this task."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_weighted": f1,
    }


def plot_genre_distribution(y, output_path: str | Path):
    """Plot the distribution of genre labels."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    genre_counts = pd.Series(y).value_counts()
    sns.countplot(data=pd.DataFrame({"genre": y}), y="genre", order=genre_counts.index, hue="genre", palette="viridis", dodge=False, legend=False)
    plt.title("Genre Distribution")
    plt.xlabel("Count")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_comparison(results: dict, output_path: str | Path):
    """Create a bar chart comparing weighted F1-score across models."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())
    f1_scores = [results[name]["f1_weighted"] for name in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, f1_scores, color=["#4C72B0", "#55A868", "#C44E52"])
    plt.title("Model Comparison by Weighted F1-Score")
    plt.ylabel("Weighted F1-Score")
    plt.ylim(0, 1.05)
    for bar, value in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, output_path: str | Path):
    """Plot the confusion matrix for the best model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Genre")
    ax.set_ylabel("Actual Genre")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def print_model_report(model_name: str, metrics: dict):
    print(f"\n=== {model_name} ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def build_classification_report(y_true, y_pred, labels=None):
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)
