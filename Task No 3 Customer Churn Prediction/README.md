# Customer Churn Prediction

## Project Objective
This project builds a machine learning model to predict whether a bank customer will churn based on customer demographics, account activity, and behavior data. The goal is to support retention strategies by identifying customers at high risk of leaving.

## Problem Statement
Customer churn is a major challenge for banks and financial institutions. Predicting churn in advance allows teams to intervene with targeted offers, service improvements, and retention programs. This model helps estimate the probability that a customer will leave the bank.

## Dataset Description
The project uses the Kaggle Bank Customer Churn Prediction dataset from `shantanudhakadd/bank-customer-churn-prediction`.

The dataset contains features such as:
- Credit score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of products
- Credit card ownership
- Active membership status
- Estimated salary
- Churn label (`Exited`)

The dataset is downloaded automatically via `kagglehub` and the CSV file is identified dynamically. No hardcoded file path is used.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- KaggleHub

## Machine Learning Algorithms
The project trains and compares multiple models:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Hyperparameter tuning is applied to the strongest models using `GridSearchCV`.

## Project Workflow
1. Download the Kaggle dataset automatically.
2. Inspect the dataset structure and target column.
3. Clean and preprocess the features.
4. Perform exploratory data analysis and data visualization.
5. Split the data into train/test sets with stratification.
6. Train multiple classification models.
7. Evaluate metrics using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and ROC curve.
8. Tune the strongest models.
9. Save the final trained pipeline as `models/churn_prediction_model.pkl`.
10. Build a Streamlit interface for interactive prediction.

## Model Evaluation
Model performance is compared based primarily on:
- F1-score
- ROC-AUC

Accuracy is also reported, but it is not used as the primary selection metric because churn datasets are often imbalanced.

## Installation
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the Model
Run:

```bash
python src/train_model.py
```

This loads the dataset, trains the models, evaluates them, tunes the strongest candidates, and saves the final pipeline to `models/churn_prediction_model.pkl`.

## Run the Streamlit App
```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Example Prediction
A sample prediction can be generated with:

```bash
python src/predict.py
```

## Project Structure
```text
Customer Churn Prediction/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── bank_customer_churn.csv
├── models/
│   └── churn_prediction_model.pkl
├── notebooks/
├── reports/
│   └── eda/
├── src/
│   ├── __init__.py
│   ├── churn_pipeline.py
│   ├── predict.py
│   └── train_model.py
└── .venv/
```
## Demo Video



https://github.com/user-attachments/assets/17a5cd27-3a4e-43b1-a219-14cb303142c9





## Future Improvements
- Add more advanced models such as XGBoost or LightGBM.
- Incorporate SHAP or feature attribution explanations.
- Add automated retraining and model monitoring.
- Use a larger or more recent dataset for better business coverage.
- Deploy the app to a cloud platform such as Streamlit Community Cloud or Azure.

## Notes
- The dataset is downloaded automatically from Kaggle using `kagglehub`.
- The saved pipeline is designed to accept customer data in the same schema used during training.
- The project uses a reproducible setup with `random_state=42`.
