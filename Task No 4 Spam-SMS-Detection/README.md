# Spam SMS Detection

A complete Spam SMS Detection project built in Python using the SMS Spam Collection dataset. This repository includes exploratory analysis, text preprocessing, model training, evaluation, and a Streamlit web app for real-time spam detection.

## Project Overview

This project classifies SMS messages as Ham or Spam using natural language processing and machine learning. It includes data exploration, preprocessing, TF-IDF feature engineering, and training multiple classification models.

## Features

- Exploratory Data Analysis (EDA)
- Text preprocessing with tokenization, stopword removal, and stemming
- TF-IDF vectorization with n-grams
- Model comparison across Multinomial Naive Bayes, Logistic Regression, and LinearSVC
- Model evaluation with accuracy, precision, recall, F1 score, confusion matrix, and classification report
- Saved trained model and vectorizer
- Streamlit app for interactive SMS spam prediction

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- Joblib
- Streamlit

## Dataset

The `data/spam.csv` file contains SMS messages labeled as `ham` or `spam`.

## Installation

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the environment:
   - Windows: `.venv\\Scripts\\activate`
   - Mac/Linux: `source .venv/bin/activate`
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Train and save the model

Run the training script to train the model and save serialized artifacts:
```bash
python train_model.py
```

### Run the Streamlit app

```bash
streamlit run app.py
```

## Project Structure

```
Spam-SMS-Detection/
├── data/
│   └── spam.csv
├── notebook/
│   └── Spam_SMS_Detection.ipynb
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
├── app.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Screenshots

- `images/streamlit_home.png` (placeholder)
- `images/model_performance.png` (placeholder)

## Future Improvements

- Use a larger, more representative dataset
- Add feature selection and hyperparameter tuning
- Deploy the app with Docker or a cloud service
- Add support for batch SMS classification
