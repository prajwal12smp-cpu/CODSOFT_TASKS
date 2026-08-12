# Movie Genre Classification

A complete machine learning project that predicts a movie's genre from its plot summary or description using TF-IDF text features and classical NLP classifiers.

## Overview

This project uses the Kaggle IMDb genre-classification dataset and trains a multiclass text classifier to predict the most likely genre from a plot description. The pipeline includes dataset loading, preprocessing, text vectorization, model comparison, evaluation, model persistence, and a Streamlit web interface for real-time predictions.

## Features

- Downloads and loads the real IMDb dataset from KaggleHub
- Cleans and standardizes text input
- Removes missing values and duplicate records
- Converts text to TF-IDF features
- Compares multiple classifiers:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Linear SVM
- Selects the best model using weighted F1-score
- Saves the trained model and vectorizer for inference
- Provides a user-friendly Streamlit app for prediction input
- Displays prediction confidence and top 3 genre probabilities

## Dataset

This project uses the Kaggle dataset:

- hijest/genre-classification-dataset-imdb

The project dynamically locates the actual downloaded files instead of assuming a fixed CSV structure. The real data contains the following fields:

- movie_id
- title
- genre
- plot

## Model and Evaluation

The pipeline compares several classifiers and selects the best-performing model based on weighted F1-score.

Verified results from the real dataset:

- Naive Bayes: accuracy = 0.4904, weighted F1 = 0.3837
- Logistic Regression: accuracy = 0.5925, weighted F1 = 0.5436
- Linear SVM: accuracy = 0.5853, weighted F1 = 0.5583

Best model selected: Linear SVM

## Project Workflow

```text
Dataset download
   ↓
Data inspection
   ↓
Missing-value and duplicate handling
   ↓
Text cleaning and preprocessing
   ↓
TF-IDF vectorization
   ↓
Train/test split
   ↓
Model comparison
   ↓
Best model selection
   ↓
Metric generation and plot export
   ↓
Model serialization
   ↓
Streamlit app prediction
```

## Repository Structure

```text
movie-genre-classification/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── models/
│   ├── movie_genre_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── exploration.ipynb
├── outputs/
│   ├── confusion_matrix.png
│   ├── genre_distribution.png
│   └── model_comparison.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── predict.py
│   └── train.py
└── .venv/
```

## Setup

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If your local Windows Python environment has NumPy or DLL import issues, use the uv-managed Python 3.11 interpreter instead:

```powershell
C:\Users\PRAJWAL\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe -m pip install -r requirements.txt
```

## Run the Project

Train the model and generate plots/artifacts:

```bash
python src/train.py
```

If the default environment is restricted on Windows, use:

```powershell
& "C:\Users\PRAJWAL\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe" src/train.py
```

Launch the Streamlit application:

```bash
streamlit run app.py
```

If needed in a restricted Windows environment:

```powershell
& "C:\Users\PRAJWAL\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe" -m streamlit run app.py --server.headless true
```

## Example Prediction

Sample input plot:

> A young detective investigates a mysterious disappearance and discovers a hidden conspiracy involving a powerful criminal organization.

Example output from the trained model:

```json
{
  "predicted_genre": "drama",
  "confidence": 7.41,
  "top_predictions": [
    {"genre": "drama", "confidence": 7.41},
    {"genre": "action", "confidence": 7.05},
    {"genre": "thriller", "confidence": 6.88}
  ]
}
```

## Notes

- The app loads the saved model and vectorizer from disk, so it does not retrain on every run.
- The trained model is intended for multiclass genre classification based on plot text.
- The project follows a modular structure and is designed for extension with additional text classifiers or feature engineering methods.

