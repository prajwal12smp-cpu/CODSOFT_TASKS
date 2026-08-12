# Movie Genre Classification

## Overview

This project builds a machine learning system for predicting the genre of a movie from its plot summary or description. The pipeline uses TF-IDF text vectorization together with classical classifiers such as Multinomial Naive Bayes, Logistic Regression, and Linear SVM. The model is trained on the Kaggle IMDb genre classification dataset and saved for use in a Streamlit web application.

## Dataset

This project uses the Kaggle dataset:

**Kaggle: hijest/genre-classification-dataset-imdb**

The downloaded dataset is inspected dynamically using KaggleHub and the code adapts to the actual file structure. In the real dataset, the training file is a plain text file where each row contains the structure:

- movie_id
- title
- genre
- plot/description

The code reads the files using the actual Kaggle download path instead of assuming a hard-coded pathname or filename.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression / Naive Bayes / SVM
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- KaggleHub

## Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
Train/Test Split
   ↓
TF-IDF Vectorization
   ↓
Model Training
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Streamlit Application
```

## Evaluation

This project compares multiple classifiers and selects the best model based primarily on weighted F1-score.

Verified training results from the real IMDb dataset:

- Naive Bayes: accuracy = 0.4904, weighted F1 = 0.3837
- Logistic Regression: accuracy = 0.5925, weighted F1 = 0.5436
- Linear SVM: accuracy = 0.5853, weighted F1 = 0.5583

Best model selected: Linear SVM

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model and generate evaluation artifacts:

```bash
python src/train.py
```

If the default Windows Python environment has NumPy/DLL import issues, use the local uv-managed Python 3.11 runtime instead:

```bash
C:\Users\PRAJWAL\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe src/train.py
```

Start the Streamlit app:

```bash
streamlit run app.py
```

If the app is launched from a restricted Windows environment, use the same Python interpreter:

```bash
C:\Users\PRAJWAL\AppData\Roaming\uv\python\cpython-3.11.15-windows-x86_64-none\python.exe -m streamlit run app.py
```

## Example Prediction

Example plot:

> A young detective investigates a mysterious disappearance and discovers a hidden conspiracy involving a powerful criminal organization.

Verified output from the trained model:

```json
{
  "predicted_genre": "drama",
  "confidence": 42.95,
  "top_predictions": [
    {"genre": "drama", "confidence": 42.95},
    {"genre": "action", "confidence": 15.89},
    {"genre": "thriller", "confidence": 9.58}
  ]
}
```

The model predicts the genre that best matches the content based on the learned TF-IDF patterns.

## Project Structure

```text
movie-genre-classification/
│
├── data/
│   └── README.md
├── models/
│   ├── movie_genre_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── exploration.ipynb
├── outputs/
│   ├── genre_distribution.png
│   ├── model_comparison.png
│   └── confusion_matrix.png
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
