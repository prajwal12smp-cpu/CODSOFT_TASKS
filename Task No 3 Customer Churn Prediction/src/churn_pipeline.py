from __future__ import annotations

import glob
from pathlib import Path

import joblib
import kagglehub
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EDA_DIR = PROJECT_ROOT / "reports" / "eda"


def download_dataset() -> Path:
    """Download the Kaggle dataset and return the CSV file path."""
    dataset_path = kagglehub.dataset_download("shantanudhakadd/bank-customer-churn-prediction")
    csv_files = sorted(Path(dataset_path).glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in dataset directory: {dataset_path}")
    return csv_files[0]


def save_raw_dataset(df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "bank_customer_churn.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def identify_target_column(df: pd.DataFrame) -> str:
    """Identify the churn target column automatically if present."""
    lower_map = {col.lower(): col for col in df.columns}
    preferred_names = ["churn", "exited", "target", "is_churn", "label"]
    for name in preferred_names:
        if name in lower_map:
            return lower_map[name]

    for col in df.columns:
        lower = col.lower()
        if "churn" in lower or "exit" in lower:
            return col

    raise ValueError("Could not identify the target column automatically. Please check the dataset columns.")


def prepare_dataset(df: pd.DataFrame):
    """Drop unnecessary identifiers and prepare features and target."""
    df = df.copy()
    target_col = identify_target_column(df)
    drop_cols = [col for col in ["RowNumber", "CustomerId", "Surname"] if col in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    y = df[target_col]
    if not pd.api.types.is_numeric_dtype(y):
        mapping = {
            "yes": 1,
            "no": 0,
            "y": 1,
            "n": 0,
            "true": 1,
            "false": 0,
            "active": 1,
            "inactive": 0,
        }
        y = y.map(mapping)

    y = y.astype(int)
    X = df.drop(columns=target_col)
    return X, y, target_col


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No feature columns found for the preprocessing pipeline.")

    return ColumnTransformer(transformers=transformers)


def get_sample_weights(y_train: pd.Series) -> np.ndarray:
    counts = y_train.value_counts(normalize=False).to_dict()
    n = len(y_train)
    weights = np.ones(n, dtype=float)
    for label in [0, 1]:
        if label in counts:
            weights[y_train.to_numpy() == label] = n / (2 * counts[label])
    return weights


def get_model_definitions():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }


def evaluate_model(model_name: str, fitted_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = fitted_pipeline.predict(X_test)
    y_prob = fitted_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred),
        "FPR": None,
        "TPR": None,
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    metrics["FPR"] = fpr
    metrics["TPR"] = tpr
    return metrics


def make_training_pipeline(model_name: str, preprocessor: ColumnTransformer, estimator):
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def train_and_compare_models() -> tuple[pd.DataFrame, dict, dict, list]:
    raw_df = pd.read_csv(download_dataset())
    X, y, target_col = prepare_dataset(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    preprocessor = build_preprocessor(X_train)
    sample_weights = get_sample_weights(y_train)
    model_results = []
    trained_models = {}

    for model_name, estimator in get_model_definitions().items():
        pipeline = make_training_pipeline(model_name, preprocessor, estimator)
        if model_name == "Gradient Boosting":
            pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
        else:
            pipeline.fit(X_train, y_train)

        metrics = evaluate_model(model_name, pipeline, X_test, y_test)
        model_results.append(metrics)
        trained_models[model_name] = pipeline

    comparison_df = pd.DataFrame(
        [{
            "Model": item["Model"],
            "Accuracy": item["Accuracy"],
            "Precision": item["Precision"],
            "Recall": item["Recall"],
            "F1": item["F1"],
            "ROC_AUC": item["ROC_AUC"],
        } for item in model_results]
    ).sort_values(["F1", "ROC_AUC"], ascending=False)

    return comparison_df, trained_models, {"X": X, "y": y, "target": target_col}, [X_train, X_test, y_train, y_test]


def tune_top_models(trained_models: dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    comparison = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                "ROC_AUC": evaluate_model("Logistic Regression", trained_models["Logistic Regression"], X_test, y_test)["ROC_AUC"],
                "F1": evaluate_model("Logistic Regression", trained_models["Logistic Regression"], X_test, y_test)["F1"],
            },
            {
                "Model": "Random Forest",
                "ROC_AUC": evaluate_model("Random Forest", trained_models["Random Forest"], X_test, y_test)["ROC_AUC"],
                "F1": evaluate_model("Random Forest", trained_models["Random Forest"], X_test, y_test)["F1"],
            },
            {
                "Model": "Gradient Boosting",
                "ROC_AUC": evaluate_model("Gradient Boosting", trained_models["Gradient Boosting"], X_test, y_test)["ROC_AUC"],
                "F1": evaluate_model("Gradient Boosting", trained_models["Gradient Boosting"], X_test, y_test)["F1"],
            },
        ]
    ).sort_values(["F1", "ROC_AUC"], ascending=False)

    top_models = comparison["Model"].head(2).tolist()
    tuned_pipelines = {}

    for model_name in top_models:
        if model_name == "Logistic Regression":
            estimator = LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
            param_grid = {
                "model__C": [0.1, 1, 10],
                "model__class_weight": ["balanced", None],
            }
        elif model_name == "Random Forest":
            estimator = RandomForestClassifier(random_state=42, n_estimators=200)
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": ["balanced", "balanced_subsample"],
            }
        else:
            continue

        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        tuned_pipelines[model_name] = search.best_estimator_

    return tuned_pipelines


def plot_eda(df: pd.DataFrame, target_column: str):
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    churn_counts = df[target_column].value_counts().rename({0: "No", 1: "Yes"})
    plt.figure(figsize=(7, 5))
    sns.barplot(x=churn_counts.index, y=churn_counts.values, hue=churn_counts.index, palette="Set2", dodge=False, legend=False)
    plt.title("Churn Distribution")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "churn_distribution.png")
    plt.close()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_column]
    if numeric_cols:
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 3 * len(numeric_cols)))
        if len(numeric_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, numeric_cols):
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "numerical_distributions.png")
        plt.close(fig)

    corr = df[numeric_cols + [target_column]].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "correlation_heatmap.png")
    plt.close()

    categorical_cols = [col for col in df.columns if col not in numeric_cols and col != target_column]
    if categorical_cols:
        for col in categorical_cols[:3]:
            plt.figure(figsize=(8, 5))
            sns.barplot(data=df, x=col, y=target_column, estimator="mean", palette="viridis")
            plt.title(f"Churn Rate by {col}")
            plt.ylabel("Churn Rate")
            plt.tight_layout()
            plt.savefig(EDA_DIR / f"churn_by_{col}.png")
            plt.close()

    if numeric_cols:
        for col in numeric_cols[:4]:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df, x=target_column, y=col, palette="pastel")
            plt.title(f"{col} by Churn Status")
            plt.tight_layout()
            plt.savefig(EDA_DIR / f"{col}_by_churn.png")
            plt.close()


def plot_roc_curves(model_metrics: list[dict]):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for item in model_metrics:
        plt.plot(item["FPR"], item["TPR"], label=item["Model"])
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(EDA_DIR / "roc_curve_comparison.png")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, model_name: str):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(EDA_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()


def plot_feature_importance(best_pipeline: Pipeline, X: pd.DataFrame):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    model = best_pipeline.named_steps["model"]
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return None

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_df.head(15), x="Importance", y="Feature", palette="viridis")
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "feature_importance.png")
    plt.close()

    importance_df.to_csv(EDA_DIR / "feature_importance.csv", index=False)
    return importance_df


def save_model_artifact(best_pipeline: Pipeline, feature_columns: list[str], target_column: str):
    MODELS_DIR.mkdir(exist_ok=True)
    artifact = {
        "pipeline": best_pipeline,
        "feature_columns": feature_columns,
        "target_column": target_column,
    }
    joblib.dump(artifact, MODELS_DIR / "churn_prediction_model.pkl")
    return MODELS_DIR / "churn_prediction_model.pkl"


def train_model_pipeline():
    raw_df = pd.read_csv(download_dataset())
    X, y, target_col = prepare_dataset(raw_df)
    save_raw_dataset(raw_df)
    plot_eda(raw_df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    preprocessor = build_preprocessor(X_train)
    sample_weights = get_sample_weights(y_train)

    model_defs = get_model_definitions()
    model_results = []
    trained_models = {}

    for name, estimator in model_defs.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        if name == "Gradient Boosting":
            pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
        else:
            pipeline.fit(X_train, y_train)
        trained_models[name] = pipeline
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        model_results.append(metrics)

    comparison_df = pd.DataFrame(
        [{
            "Model": item["Model"],
            "Accuracy": item["Accuracy"],
            "Precision": item["Precision"],
            "Recall": item["Recall"],
            "F1": item["F1"],
            "ROC_AUC": item["ROC_AUC"],
        } for item in model_results]
    ).sort_values(["F1", "ROC_AUC"], ascending=False)

    plot_roc_curves(model_results)
    for item in model_results:
        plot_confusion_matrix(item["ConfusionMatrix"], item["Model"])

    best_candidate_names = comparison_df["Model"].head(2).tolist()
    tuned_models = {}

    for model_name in best_candidate_names:
        if model_name == "Logistic Regression":
            estimator = LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)
            param_grid = {
                "model__C": [0.1, 1, 10],
                "model__class_weight": ["balanced", None],
            }
        elif model_name == "Random Forest":
            estimator = RandomForestClassifier(random_state=42, n_estimators=200)
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 2, 4],
                "model__class_weight": ["balanced", "balanced_subsample"],
            }
        else:
            continue

        tuned_pipeline = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        search = GridSearchCV(
            tuned_pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        tuned_models[model_name] = search.best_estimator_

    if "Random Forest" in tuned_models:
        best_pipeline = tuned_models["Random Forest"]
    elif "Logistic Regression" in tuned_models:
        best_pipeline = tuned_models["Logistic Regression"]
    else:
        best_pipeline = trained_models["Gradient Boosting"]

    plot_feature_importance(best_pipeline, X)
    save_model_artifact(best_pipeline, list(X.columns), target_col)
    return comparison_df, tuned_models, best_pipeline


if __name__ == "__main__":
    comparison, tuned, best = train_model_pipeline()
    print("Model comparison:\n", comparison.to_string(index=False))
    print("\nSaved model artifact at:", MODELS_DIR / "churn_prediction_model.pkl")
