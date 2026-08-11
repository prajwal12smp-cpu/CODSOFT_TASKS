import csv
import os
import joblib
from text_utils import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_PATH = os.path.join("data", "spam.csv")
MODEL_DIR = "models"


def load_data(path: str):
    labels = []
    texts = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label = row["label"].strip().lower()
            if label in {"ham", "spam"}:
                labels.append(0 if label == "ham" else 1)
                texts.append(row["message"])
    return texts, labels


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "classification_report": classification_report(y_test, preds, target_names=["Ham", "Spam"]),
    }


def print_comparison_table(results):
    header = f"{'Model':<15} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1 Score':>10}"
    print(header)
    print("-" * len(header))
    for res in results:
        print(
            f"{res['model']:<15} {res['accuracy']:>9.4f} {res['precision']:>10.4f} {res['recall']:>8.4f} {res['f1_score']:>10.4f}"
        )


def main():
    texts, labels = load_data(DATA_PATH)
    cleaned_texts = [clean_text(text) for text in texts]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vectorized = vectorizer.fit_transform(cleaned_texts)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, labels, test_size=0.2, random_state=42, stratify=labels
    )

    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "LinearSVC": LinearSVC(random_state=42, max_iter=10000),
    }

    results = []
    best_model = None
    best_f1 = 0.0
    best_model_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)
        print(f"{name} performance:")
        print(f"  Accuracy : {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall   : {result['recall']:.4f}")
        print(f"  F1 Score : {result['f1_score']:.4f}\n")

        if result["f1_score"] > best_f1:
            best_f1 = result["f1_score"]
            best_model = model
            best_model_name = name
            best_report = result["classification_report"]
            best_confusion = result["confusion_matrix"]

    print("Model comparison:\n")
    print_comparison_table(results)
    print(f"\nBest model: {best_model_name} with F1 {best_f1:.4f}\n")
    print("Classification Report for best model:")
    print(best_report)
    print("Confusion Matrix for best model:")
    print(best_confusion)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODEL_DIR, "spam_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print("Saved model and vectorizer in models/")


if __name__ == "__main__":
    main()
